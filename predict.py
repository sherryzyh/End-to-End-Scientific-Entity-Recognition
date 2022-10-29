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
import json
import torch
import numpy as np

from common import get_test_dataset, compute_metrics, id2label, label2id, CustomTrainer

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
    parser = ArgumentParser()
    parser.add_argument('--config_file', '-c',
                        type=str,
                        help='Config file',
                        default='config/predict/bert_mlm_large_10e_trainaug_weighted_ft_10-26_16-07-07.json')
    args = parser.parse_args()

    """
        Arguments
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    with open(args.config_file, 'r') as f:
        config = json.load(f)
    system_config = config["system_config"]

    """
        Model and Tokenizer
    """
    model = os.path.join(system_config["model_root"], system_config["model_name"], system_config["metric"])
    model = AutoModelForTokenClassification.from_pretrained(model,
                                                            ignore_mismatched_sizes=True,
                                                            id2label=id2label,
                                                            label2id=label2id).to(device)


    tokenizer = AutoTokenizer.from_pretrained(system_config["tokenizer"])

    """
        Prepare Test Dataset
    """
    test_file_path = os.path.join("Dataset", "test_data", config["sentence_file"])
    test_dataset = get_test_dataset(test_file_path)
    test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


    """
        Predict
    """
    test_args = TrainingArguments(**config["test_args"])
    trainer = Trainer(model=model,
                        args=test_args,
                        tokenizer=tokenizer,
                        data_collator=data_collator)
    raw_preds = trainer.predict(test_dataset)

    predictions = np.argmax(raw_preds.predictions.squeeze(), axis=-1)
    label_ids = raw_preds.label_ids
    print("predictions shape:", predictions.shape)
    print("label_ids shape:", label_ids.shape)
    print("len(testset):", len(test_dataset))

    n_sentences = predictions.shape[0]
    sen_length = predictions.shape[1]

    """
        Output
    """
    output_config = config["output_config"]
    output_path = os.path.join(output_config["output_root"], f"pred_{output_config['output_level']}_{system_config['model_name']}.conll")
    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(n_sentences):
            # print(test_dataset[i])
            tokens = test_dataset[i]["tokens"]
            labels = []
            for j in range(sen_length):
                if label_ids[i][j] == -100:
                    continue
                pred_label = id2label[predictions[i][j]]
                labels.append(pred_label)

            if len(tokens) != len(labels):
                print(len(tokens), tokens)
                print(len(labels), labels)

            for token, label in zip(tokens, labels):
                f.write(f"{token} {label}\n")
            f.write("\n")