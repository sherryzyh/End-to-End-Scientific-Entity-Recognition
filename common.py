import os
from copy import deepcopy
import pandas as pd
import numpy as np
from datasets import Dataset
from datasets import load_metric
from transformers import TrainerCallback

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
    
def get_tokens_and_ner_tags(filename):
    with open(filename, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        tokens, entities = [], []
        current_tokens, current_entities = [], []
        for line in lines:
            data = line.split(" ")
            if len(data) < 2:
                if not current_tokens:
                    continue
                tokens.append(current_tokens)
                entities.append(current_entities)
                current_tokens, current_entities = [], []
            elif len(data) == 2:
                current_tokens.append(data[0])
                current_entities.append(data[1].strip())
        if current_tokens:
            tokens.append(current_tokens)
            entities.append(current_entities)
    return pd.DataFrame({'tokens': tokens, 'ner_tags': entities})
  
def get_dataset(directory):
    df = pd.concat([get_tokens_and_ner_tags(os.path.join(directory, filename)) for filename in os.listdir(directory)]).reset_index().drop('index', axis=1)
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

# https://stackoverflow.com/questions/67457480/how-to-get-the-accuracy-per-epoch-or-step-for-the-huggingface-transformers-train
class CustomCallback(TrainerCallback):
    
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy

'''
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
'''