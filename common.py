import os
import itertools
import pandas as pd
import numpy as np
from datasets import Dataset
from datasets import load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import torch

label_list = ['O','B-MISC','I-MISC','B-PER','I-PER','B-ORG','I-ORG','B-LOC','I-LOC']
label_encoding_dict = {'I-PRG': 2,'I-I-MISC': 2, 'I-OR': 6, 'O': 0, 'I-': 0, 'VMISC': 0, 'B-PER': 3, 'I-PER': 4, 'B-ORG': 5, 'I-ORG': 6, 'B-LOC': 7, 'I-LOC': 8, 'B-MISC': 1, 'I-MISC': 2}
    
def get_tokens_and_ner_tags(filename):
    with open(filename, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        tokens, entities = [], []
        current_tokens, current_entities = [], []
        for line in lines:
            data = line.split(" ")
            if len(data) < 2:
                tokens.append(current_tokens)
                entities.append(current_entities)
                current_tokens, current_entities = [], []
            elif len(data) == 2:
                current_tokens.append(data[0])
                current_entities.append(data[1])
        if current_tokens:
            tokens.append(current_tokens)
            entities.append(current_entities)
    return pd.DataFrame({'tokens': tokens, 'ner_tags': entities})
  
def get_dataset(directory):
    df = pd.concat([get_tokens_and_ner_tags(os.path.join(directory, filename)) for filename in os.listdir(directory)]).reset_index().drop('index', axis=1)
    dataset = Dataset.from_pandas(df)
    return dataset

def tokenize_and_align_labels(tokenizer, examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

class DataSequence(torch.utils.data.Dataset):

    def __init__(self, df, tokenizer, max_len, labels_to_ids, label_all_tokens):
        lb = [i.split() for i in df['labels'].values.tolist()]
        txt = df['text'].values.tolist()
        self.texts, self.labels = [], []
        for i, j in zip(txt, lb):
            tokenized_i = tokenizer(i, padding='max_length', max_length = max_len, truncation=True, return_tensors="pt")
            self.texts.append(tokenized_i)
            word_ids = tokenized_i.word_ids()
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(labels_to_ids.get(j[word_idx], -100))
                else:
                    label_ids.append(labels_to_ids.get(j[word_idx], -100) if label_all_tokens else -100)
                previous_word_idx = word_idx
            self.labels.append(label_ids)

    def __len__(self):

        return len(self.labels)

    def get_batch_data(self, idx):

        return self.texts[idx]

    def get_batch_labels(self, idx):

        return torch.LongTensor(self.labels[idx])

    def __getitem__(self, idx):

        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)

        return batch_data, batch_labels