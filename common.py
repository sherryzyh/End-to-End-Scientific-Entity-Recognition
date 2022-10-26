'''
This file contains global variables and methods that are used in experiments.
'''

import os
import pandas as pd
import numpy as np
from datasets import Dataset
from datasets import load_metric
import torch
from torch import nn
from transformers import Trainer
from zmq import device
from datetime import datetime
import pytz
from utils import EntitySentence

label_list = [
    'O', 
    'B-MethodName', 'I-MethodName',
    'B-HyperparameterName', 'I-HyperparameterName',
    'B-HyperparameterValue', 'I-HyperparameterValue',
    'B-MetricName', 'I-MetricName',
    'B-MetricValue', 'I-MetricValue',
    'B-TaskName', 'I-TaskName',
    'B-DatasetName', 'I-DatasetName'
    ]
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}

def get_timestamp():
    tz_EST = pytz.timezone('America/New_York')
    datetime_EST = datetime.now(tz_EST)
    return datetime_EST

def get_dataset(directory, method, **kwargs):
    df = pd.concat([get_tokens_and_ner_tags(os.path.join(directory, filename), method, **kwargs) \
        for filename in os.listdir(directory)]).reset_index().drop('index', axis=1)
    dataset = Dataset.from_pandas(df)
    return dataset

def compute_metrics(p):
    metric = load_metric("seqeval")

    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]
    true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in zip(predictions, labels)]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"], "accuracy": results["overall_accuracy"]}

"""
    Data Loading Methods
"""
def get_tokens_and_ner_tags(filename, method, **kwargs):
    # print(f"Loading Method: {method} | filename: {filename}")
    if method == "sentence_contains_entity":
        return get_tokens_and_ner_tags_sentence_contains_entity(filename)
    if method == "by_num_sentence":
        return get_tokens_and_ner_tags_by_num_sentence(filename, **kwargs)
    if method == "paragraph_entity_sentence_only":
        return get_tokens_and_ner_tags_paragraph_entity_sentence_only(filename, **kwargs)
    if method == "paragraph_contains_entity":
        return get_tokens_and_ner_tags_paragraph_contains_entity(filename, **kwargs)
    if method == "by_seq_len":
        return get_tokens_and_ner_tags_by_seq_len(filename, **kwargs)
    if method == "text_by_num_sentence":
        return get_text_by_num_sentence(filename, **kwargs)
    if method == "text_per_sentence":
        return get_text(filename)

def get_text_by_num_sentence(filename, num_sentence):
    count = 0
    currtext = ""
    text = []
    with open(filename, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            currtext = currtext + " " + line
            count += 1
            if count == num_sentence:
                text.append(currtext)
                currtext = ""
                count = 0
    df = pd.DataFrame({'text': text})
    return pd.DataFrame({'text': text})

def get_text(filename):
    text = []
    entitysentence = EntitySentence()
    with open(filename, 'r', encoding="utf-8") as f:
        lines = f.read().splitlines()
        for line in lines:
            entitysentence.readLine(line)
            # print(entitysentence.sentence)
            if entitysentence.isEnd:
                text.append(entitysentence.sentence)
                entitysentence.clear()
    df = pd.DataFrame({'text': text})
    return pd.DataFrame({'text': text})

def get_tokens_and_ner_tags_sentence_contains_entity(filename):
    with open(filename, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        tokens, entities = [], []
        sentence_tokens, sentence_entities = [], []
        entity_flag = False
        for line in lines:
            data = line.split(" ")
            if len(data) < 2:
                if not sentence_tokens:
                    continue
                if entity_flag:
                    tokens.append(sentence_tokens)
                    entities.append(sentence_entities)
                    entity_flag = False
                sentence_tokens, sentence_entities = [], []
            elif len(data) == 2:
                sentence_tokens.append(data[0])
                label = data[1].strip()
                sentence_entities.append(label)
                if label != "O":
                    entity_flag = True
        # if any token remains and a (non-O) entity is contained
        if sentence_tokens and entity_flag:
            tokens.append(sentence_tokens)
            entities.append(sentence_entities)
    df = pd.DataFrame({'tokens': tokens, 'ner_tags': entities})
    return pd.DataFrame({'tokens': tokens, 'ner_tags': entities})

def get_tokens_and_ner_tags_by_num_sentence(filename, num_sentence):
    with open(filename, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        tokens, entities = [], []
        current_tokens, current_entities = [], []
        sentence_count = 0
        for line in lines:
            data = line.split(" ")
            if len(data) < 2:
                if not current_tokens:
                    continue
                sentence_count += 1
                # if enough sentences have been accumulated for a data entry
                if sentence_count == num_sentence:
                    tokens.append(current_tokens)
                    entities.append(current_entities)
                    current_tokens, current_entities = [], []
                    sentence_count = 0
            elif len(data) == 2:
                current_tokens.append(data[0])
                label = data[1].strip()
                current_entities.append(label)
        # if any token remains
        if current_tokens:
            tokens.append(current_tokens)
            entities.append(current_entities)
    df = pd.DataFrame({'tokens': tokens, 'ner_tags': entities})
    return pd.DataFrame({'tokens': tokens, 'ner_tags': entities})

def get_tokens_and_ner_tags_paragraph_entity_sentence_only(filename, num_sentence):
    with open(filename, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        tokens, entities = [], []
        current_tokens, current_entities = [], []
        sentence_tokens, sentence_entities = [], []
        entity_sentence_count = 0
        entity_flag = False
        for line in lines:
            data = line.split(" ")
            if len(data) < 2:
                if not sentence_tokens:
                    continue
                # if the current sentence contains any (non-O) entity
                if entity_flag:
                    current_tokens.extend(sentence_tokens)
                    current_entities.extend(sentence_entities)
                    entity_sentence_count += 1
                    entity_flag = False
                    # if enough entity sentences have been accumulated for a data entry
                    if entity_sentence_count == num_sentence:
                        tokens.append(current_tokens)
                        entities.append(current_entities)
                        current_tokens, current_entities = [], []
                        entity_sentence_count = 0
                sentence_tokens, sentence_entities = [], []
            elif len(data) == 2:
                sentence_tokens.append(data[0])
                label = data[1].strip()
                sentence_entities.append(label)
                if label != "O":
                    entity_flag = True
        # if any token remains and a (non-O) entity is contained
        if current_tokens and entity_flag:
            tokens.append(current_tokens)
            entities.append(current_entities)
    df = pd.DataFrame({'tokens': tokens, 'ner_tags': entities})
    return pd.DataFrame({'tokens': tokens, 'ner_tags': entities})

def get_tokens_and_ner_tags_paragraph_contains_entity(filename, num_sentence):
    with open(filename, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        tokens, entities = [], []
        current_tokens, current_entities = [], []
        sentence_count = 0
        entity_flag = False
        for line in lines:
            data = line.split(" ")
            if len(data) < 2:
                if not current_tokens:
                    continue
                sentence_count += 1
                # if enough sentences have been accumulated for a data entry
                if sentence_count == num_sentence:
                    if entity_flag:
                        tokens.append(current_tokens)
                        entities.append(current_entities)
                        entity_flag = False
                    current_tokens, current_entities = [], []
                    sentence_count = 0
            elif len(data) == 2:
                current_tokens.append(data[0])
                label = data[1].strip()
                current_entities.append(label)
                if label != "O":
                    entity_flag = True
        # if any token remains and a (non-O) entity is contained
        if current_tokens and entity_flag:
            tokens.append(current_tokens)
            entities.append(current_entities)
    df = pd.DataFrame({'tokens': tokens, 'ner_tags': entities})
    return pd.DataFrame({'tokens': tokens, 'ner_tags': entities})

def get_tokens_and_ner_tags_by_seq_len(filename, seq_len):
    with open(filename, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        tokens, entities = [], []
        token_counter = 0
        current_tokens, current_entities = [], []
        for line in lines:
            data = line.split(" ")
            if len(data) != 2:
                continue
            if token_counter < seq_len:
                current_tokens.append(data[0])
                current_entities.append(data[1].strip())
                token_counter += 1
            else:
                tokens.append(current_tokens)
                entities.append(current_entities)
                token_counter = 0
                current_tokens, current_entities = [], []
        if current_tokens:
            tokens.append(current_tokens)
            entities.append(current_entities)
            
    df = pd.DataFrame({'tokens': tokens, 'ner_tags': entities})
    return pd.DataFrame({'tokens': tokens, 'ner_tags': entities})

# reference: https://github.com/huggingface/transformers/issues/9398
class TestDataset:
    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts
    
    def __len__(self):
        return len(self.tokenized_texts["input_ids"])
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tokenized_texts.items()}

"""
    Trainer
"""
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        # print(inputs)
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute weighted cross entropy loss
        weight = torch.tensor([1.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, \
        100.0, 100.0, 100.0, 100.0, 100.0, 100.0]).to(labels.device)
        loss_fct = nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss