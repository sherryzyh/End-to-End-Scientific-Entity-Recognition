from transformers import (
    DataCollatorForTokenClassification, 
    AutoTokenizer, 
    AutoModelForTokenClassification,
    TrainingArguments, 
    Trainer,
    set_seed
)
from argparse import ArgumentParser
import json
import os
import shutil
from datetime import datetime
import pytz

from common import get_dataset, compute_metrics, id2label, label2id

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label2id[label[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

if __name__ == '__main__':
    # parse command line arguments and load config
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/baseline.json',
                        help='Configuration file to use')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Resume from a saved checkpoint')
    args = parser.parse_args()
    config_file = args.config
    checkpoint = args.resume_from_checkpoint
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    general_args = config['general_args']
    data_args = config['data_args']
    train_args = config['train_args']
    
    train_data_directory = data_args['train_data']
    validation_data_directory = data_args['validation_data']
    transformer = general_args['transformer']
    
    set_seed(train_args['seed'])

    tz_EST = pytz.timezone('America/New_York')
    datetime_EST = datetime.now(tz_EST)
    experiment_name = "_".join([general_args["system"], datetime_EST.strftime("%m-%d_%H-%M-%S")])

    # load and preprocess data
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(transformer)
    train_dataset = get_dataset(train_data_directory)
    validation_dataset = get_dataset(validation_data_directory)
    train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
    validation_dataset = validation_dataset.map(tokenize_and_align_labels, batched=True)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    # train and eval
    # reference for "ignore_mismatched_sizes": https://github.com/huggingface/transformers/issues/14218
    model = AutoModelForTokenClassification.from_pretrained(transformer, ignore_mismatched_sizes=True, id2label=id2label, label2id=label2id)
    output_dir = "./results/" + experiment_name
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=train_args['learning_rate'],
        per_device_train_batch_size=train_args['batch_size'],
        per_device_eval_batch_size=train_args['batch_size'],
        num_train_epochs=train_args['num_epochs'],
        weight_decay=train_args['weight_decay']
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    
    trainer.save_model()
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    eval_metrics = trainer.evaluate()
    eval_metrics["eval_samples"] = len(validation_dataset)
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    shutil.copy2(config_file, output_dir)

    # TODO: predict on test dataset (need to add a command line argument for train vs. test)
    
