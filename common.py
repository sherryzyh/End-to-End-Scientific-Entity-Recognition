import pandas as pd
import torch 
import numpy as np
from transformers import BertTokenizerFast, BertForTokenClassification
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import SGD

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