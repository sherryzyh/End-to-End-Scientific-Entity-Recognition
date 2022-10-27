'''
This file runs inference. 
Model, output file name, and batch size can be specified using command line arguments '--model', '--output' and '--batch_size'.
'''

from transformers import (
    DataCollatorForTokenClassification, 
    AutoTokenizer, 
    AutoModelForTokenClassification,
    TrainingArguments, 
    Trainer,
    set_seed
)
from argparse import ArgumentParser
import os
import torch
import numpy

from common import get_dataset, compute_metrics, id2label, label2id, CustomTrainer, TestDataset

if __name__ == '__main__':
    # parse command line arguments and load config
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, help='Model to load', required=True)
    parser.add_argument('--output', type=str, help='Output file name', required=True)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=8)
    args = parser.parse_args()
    model_path = os.path.join('/data/results', args.model)
    output_path = os.path.join('output_conll', args.output)
    batch_size = args.batch_size
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    test_data_file = 'test_data/sent-input-test.txt'
    #test_data_file = 'test_data/anlp-sciner-test-sentences.txt'
    with open(test_data_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    word_lines = [line.split(" ") for line in lines]
    for i in range(len(word_lines)):
        word_lines[i][-1] =  word_lines[i][-1].strip("\n")
    
    tokens = tokenizer(word_lines, padding=True, is_split_into_words=True)
    test_dataset = TestDataset(tokens)

    model = AutoModelForTokenClassification.from_pretrained(model_path, id2label=id2label, label2id=label2id)
    args = TrainingArguments(output_dir='tmp_trainer_for_inference', per_device_eval_batch_size=batch_size)
    trainer = Trainer(model=model, args=args)
    raw_preds = trainer.predict(test_dataset)
    predictions = torch.tensor(raw_preds.predictions)
    predictions = torch.argmax(predictions.squeeze(), axis=-1)
    num_sent, sent_len = predictions.size()
    preds = [[None] * sent_len for i in range(num_sent)]
    for i in range(num_sent):
        for j in range(sent_len):
            p = predictions[i][j].item()
            preds[i][j] = model.config.id2label[p]
    '''
    with open(output_path, 'w', encoding='utf-8') as f:
        for i in range(num_sent):
            for j in range(sent_len):
                id = tokens.input_ids[i][j]
                token = tokenizer.decode([id])
                label = preds[i][j]
                if token == "[CLS]":
                    continue
                if token == "[SEP]":
                    f.write("\n")
                    break
                f.write(token + " " + label + "\n")
    '''
    with open(output_path, 'w', encoding='utf-8') as f:
        prev_token, prev_label = None, None
        for i in range(num_sent):
            for j in range(sent_len):
                id = tokens.input_ids[i][j]
                token = tokenizer.decode([id])
                label = preds[i][j]
                if token == "[CLS]":
                    continue
                if token == "[SEP]":
                    f.write(prev_token + " " + prev_label + "\n\n")
                    prev_token = None
                    break
                if token[:2] != "##":
                    if prev_token:
                        f.write(prev_token + " " + prev_label + "\n")
                    prev_token = token
                    prev_label = label
                else:
                    prev_token = prev_token + token[2:]