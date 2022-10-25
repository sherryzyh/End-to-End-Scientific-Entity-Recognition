from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    DataCollatorForLanguageModeling,
    AutoModelForMaskedLM,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    set_seed,
    AutoConfig
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

# def tokenize_and_align_labels(examples):
#     tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
#     labels = []
#     for i, label in enumerate(examples["ner_tags"]):
#         word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
#         previous_word_idx = None
#         label_ids = []
#         for word_idx in word_ids:  # Set the special tokens to -100.
#             if word_idx is None:
#                 label_ids.append(-100)
#             elif word_idx != previous_word_idx:  # Only label the first token of a given word.
#                 label_ids.append(label2id[label[word_idx]])
#             else:
#                 label_ids.append(-100)
#             previous_word_idx = word_idx
#         labels.append(label_ids)
#         seq_len_stats.append(len(word_ids))
#
#     tokenized_inputs["labels"] = labels
#     return tokenized_inputs

# class SERDataset(Dataset):
#     def __init__(self, directory, tokenizer, mlm=True):
#         super(SERDataset, self).__init__(directory, tokenizer, mlm)
#         self.tokenizer = tokenizer
#
#         self.text = None
#         self.label = None
#         if mlm:
#             self.build_mlm_dataset(directory)
#
#     def __len__(self):
#         return len(self.text)
#
#     def __getitem__(self, idx):
#         return {"text": self.text[idx]}
#
#     def build_mlm_dataset(self, directory):
#         paper_list = os.listdir(directory)
#         text = []
#         for paper in paper_list:
#             read_path = os.path.join(directory, paper)
#             try:
#                 with open(read_path, "r", encoding="utf-8") as f:
#                     lines = f.read().splitlines()
#             except:
#                 print(f"An exception occurred when reading << {paper} >>")
#                 continue
#
#             text += lines
#
#         self.text = text

# def get_mlm_dataset(directory, tokenizer, blocksentence):
#     paper_list = os.listdir(directory)
#     text = []
#     block = ""
#     count = 0
#     for paper in paper_list:
#         read_path = os.path.join(directory, paper)
#         try:
#             with open(read_path, "r", encoding="utf-8") as f:
#                 lines = f.read().splitlines()
#         except:
#             print(f"An exception occurred when reading << {paper} >>")
#             continue
#         for line in lines:
#             if count < blocksentence:
#                 block += line + " "
#                 count += 1
#             else:
#                 text.append(block)
#                 count = 0
#                 block = ""
#     if block != "":
#         text.append(block)
#         block = ""
#         count = 0
#
#     print("#sentences:", len(text))
#     print("sample:", text[0])
#     print("*" * 20)
#     print(len(text[0].strip().split(" ")), len(text[0]))
#     inputs = tokenizer(text=text,
#                        return_tensors="pt",
#                        padding="longest",
#                        max_length=512,
#                        is_split_into_words=False)
#     print(inputs.keys())
#     print(f"input['input_ids'] - len: {len(inputs['input_ids'])}, sample: {len(inputs['input_ids'][0])}")
#     print(f"input['token_type_ids'] - len: {len(inputs['token_type_ids'])}, sample: {inputs['token_type_ids'][0]}")
#     print(f"input['attention_mask'] - len: {len(inputs['attention_mask'])}, sample: {inputs['attention_mask'][0]}")
#
#     print("*" * 20)
#
#     return Dataset.from_dict({"text": text, "label": text})

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

    model = AutoModelForMaskedLM.from_pretrained(general_args['transformer'])

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

    trainer.train()