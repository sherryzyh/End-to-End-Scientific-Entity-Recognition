from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    DataCollatorForLanguageModeling,
    BertForMaskedLM,
    AutoModelForMaskedLM,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler
from common import get_dataset, get_timestamp
from argparse import ArgumentParser
import json
import os
import shutil
from datetime import datetime

def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config_file', '-c',
                        type=str,
                        default='./config/bert_scientific_mlm.json',
                        help='Configuration file to use')
    args = parser.parse_args()
    config_file = args.config_file

    config_file = args.config_file
    with open(config_file, 'r') as f:
        config = json.load(f)

    """
        Config Arguments
    """
    general_args = config['general_args']
    data_args = config['data_args']
    train_args = config['train_args']
    mlm_args = config['mlm_args']
    os.environ['TRANSFORMERS_CACHE'] = general_args["cache"]

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    
    print(f"deivce: {device}")

    model = BertForMaskedLM.from_pretrained(general_args["transformer"], cache_dir=general_args["cache"]).to(device)
    # print("*" * 40)
    # print(model)

    # print("*" * 40)
    # print(model.config)

    experiment_time = get_timestamp()

    """
        Prepare Data
    """
    train_data_directory = data_args['train_data']
    val_data_directory = data_args['val_data']
    data_loading_method = data_args['data_loading_method']
    num_sentence = data_args['num_sentence']
    train_dataset = get_dataset(directory=train_data_directory,
                                method=data_loading_method,
                                **{"num_sentence": num_sentence})
    val_dataset = get_dataset(directory=val_data_directory,
                              method=data_loading_method,
                              **{"num_sentence": num_sentence})
    # print("****** dataset sample ******")
    # print(train_dataset[0])

    """
        Tokenize Dataset
    """
    tokenizer = AutoTokenizer.from_pretrained(general_args["tokenizer"], cache_dir=general_args["cache"])
    # print("*" * 40)
    # print(tokenizer)

    tokenizer.eos_token = "[EOS]"
    # print("*" * 40)
    # print("****** Tokenizer Special Tokens ******")
    # print("eos_token:", tokenizer.eos_token)
    # print("pad_token:", tokenizer.pad_token)
    # print("mask_token:", tokenizer.mask_token)

    # Use Datasets map function to tokenize and align the labels over the entire dataset.
    tokenized_train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=train_dataset.column_names,
    )
    tokenized_val_dataset = val_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=val_dataset.column_names,
    )

    # print("****** tokenized dataset sample ******")
    # print(tokenized_train_dataset[0])

    # Use DataCollatorForTokenClassification to create a batch of examples
    # It will also dynamically pad your text and labels to the length of the longest element in its batch
    # so they are a uniform length.
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, **mlm_args)


    """
        Training
    """
    output_dir = os.path.join(general_args["result_root"], "_".join([general_args["system"], experiment_time.strftime("%m-%d_%H-%M-%S")]))
    logging_dir = os.path.join(output_dir, "log")
    training_args = TrainingArguments(output_dir=output_dir,
                                        **train_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        data_collator=data_collator,
    )

    train_result = trainer.train()
    metrics = train_result.metrics

    eval_result = trainer.evaluate()

    pt_save_directory = os.path.join(output_dir, "save_pretrained")
    model.save_pretrained(pt_save_directory)

    trainer.log_metrics("train", metrics)
    trainer.log_metrics("eval", eval_result)

    shutil.copy2(config_file, pt_save_directory)