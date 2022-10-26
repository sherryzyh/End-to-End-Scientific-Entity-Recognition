'''
This file runs each experiment. 
Configs can be specified using command line argument "--config".
'''
from cmath import exp
import torch
from colorama import deinit
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
import statistics

from common import get_dataset, compute_metrics, id2label, label2id, CustomTrainer, get_timestamp

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
        #seq_len_stats.append(len(word_ids))

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

if __name__ == '__main__':
    # parse command line arguments and load config
    parser = ArgumentParser()
    parser.add_argument('--config', '-c',
                        type=str,
                        default='./config/baseline.json',
                        help='Configuration file to use')
    parser.add_argument('--resume_from_checkpoint', '-r',
                        type=str,
                        default=None,
                        help='Resume from a saved checkpoint')
    args = parser.parse_args()
    config_file = args.config
    checkpoint = args.resume_from_checkpoint
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    general_args = config['general_args']
    data_args = config['data_args']
    train_args = config['train_args']
    
    os.environ['TRANSFORMERS_CACHE'] = general_args["cache"]

    train_data_directory = data_args['train_data']
    validation_data_directory = data_args['validation_data']
    transformer_name = general_args['transformer']
    tokenizer_name = general_args['tokenizer']

    # get data loading method and check for the corresponding required argument from config
    method = data_args['data_loading_method']
    kwargs = dict()
    get_data_arg = None
    assert_message = f"for method = {method}, {get_data_arg} must be provided as a parameter"
    if method == "by_num_sentence":
        get_data_arg = "num_sentence"
        assert get_data_arg in data_args, assert_message
        kwargs[get_data_arg] = data_args[get_data_arg]
    elif method == "paragraph_entity_sentence_only":
        get_data_arg = "num_sentence"
        assert get_data_arg in data_args, assert_message
        kwargs[get_data_arg] = data_args[get_data_arg]
    elif method == "paragraph_contains_entity":
        get_data_arg = "num_sentence"
        assert get_data_arg in data_args, assert_message
        kwargs[get_data_arg] = data_args[get_data_arg]
    elif method == "by_seq_len":
        get_data_arg = "seq_len"
        assert get_data_arg in data_args, assert_message
        kwargs[get_data_arg] = data_args[get_data_arg]
    
    set_seed(train_args['seed'])

    # load and preprocess data
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    #seq_len_stats = []
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=general_args["cache"])
    print(tokenizer.vocab_files_names)

    train_dataset = get_dataset(train_data_directory, method, **kwargs)
    validation_dataset = get_dataset(validation_data_directory, method, **kwargs)
    train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
    validation_dataset = validation_dataset.map(tokenize_and_align_labels, batched=True)
    '''
    not_truncated, may_be_truncated = 0, 0
    for i in seq_len_stats:
        if i < 512:
            not_truncated += 1
        else:
            may_be_truncated += 1
    print(f"not truncated: {not_truncated}")
    print(f"may be truncated: {may_be_truncated}")
    print(f"avg of seq len: {statistics.mean(seq_len_stats)}")
    print(f"median of seq len: {statistics.median(seq_len_stats)}")
    print(f"max of seq len: {max(seq_len_stats)}")
    '''
    
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    # train and eval
    # reference for "ignore_mismatched_sizes": https://github.com/huggingface/transformers/issues/14218
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    print(f"deivce: {device}")
    model = AutoModelForTokenClassification.from_pretrained(transformer_name, ignore_mismatched_sizes=True, id2label=id2label, label2id=label2id).to(device)
    # model = AutoModelForTokenClassification.from_pretrained(transformer, cache_dir=general_args["cache"]).to(device)
    print(model)

    experiment_time = get_timestamp()
    experiment_name = "_".join([general_args["system"], experiment_time.strftime("%m-%d_%H-%M-%S")])
    output_dir = "./results/" + experiment_name
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=train_args['learning_rate'],
        per_device_train_batch_size=train_args['batch_size'],
        per_device_eval_batch_size=train_args['batch_size'],
        num_train_epochs=train_args['num_epochs'],
        weight_decay=train_args['weight_decay'],
        load_best_model_at_end=True
    )

    if train_args["loss"] == "unweighted":
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )
    else:
        # CustomTrainer uses weighted loss
        trainer = CustomTrainer(  
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