from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    DataCollatorForLanguageModeling,
    AutoModelForMaskedLM,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler
from common import get_dataset, compute_metrics, id2label, label2id
from argparse import ArgumentParser
import json
import os
import shutil
from datetime import datetime

def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True)

if __name__ == '__main__':
    config_file = "config/scientifc_mlm.json"
    with open(config_file, 'r') as f:
        config = json.load(f)

    """
        Config Arguments
    """
    general_args = config['general_args']
    data_args = config['data_args']
    train_args = config['train_args']
    mlm_args = config['mlm_args']

    model = AutoModelForMaskedLM.from_pretrained(general_args['transformer']).cuda()

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
    tokenizer = AutoTokenizer.from_pretrained(general_args['transformer'])
    tokenizer.eos_token = "[EOS]"
    print("eos_token:", tokenizer.eos_token)
    print("pad_token:", tokenizer.pad_token)
    print("mask_token:", tokenizer.mask_token)

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
    training_args = TrainingArguments(**train_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        data_collator=data_collator,
    )

    # trainer.train()
    # trainer.evaluate()

    shutil.copy2(config_file, train_args["output_dir"])
    # model.save_pretrained(os.path.join(train_args["output_dir"], "best_saved"))
    # tokenizer.save_pretrained(os.path.join(train_args["output_dir"], "best_saved"))