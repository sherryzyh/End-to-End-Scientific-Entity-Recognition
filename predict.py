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
import numpy as np

from common import get_dataset, get_test_dataset, compute_metrics, id2label, label2id, CustomTrainer, TestDataset

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
    parser.add_argument('--model', '-m',
                        type=str,
                        help='Model to load',
                        default='bert_mlm_large_10e_trainaug_weighted_ft_10-26_16-07-07')
    parser.add_argument('--batch_size',
                        type=int,
                        help='Batch size',
                        default=32)
    args = parser.parse_args()

    """
        Arguments
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    output_path = os.path.join('output_conll', f"pred_{args.model}.conll")

    """
        Model and Tokenizer
    """
    model_root = "/work/yinghuan/projects/End-to-End-Scientific-Entity-Recognition/results/confirmed"
    model_path = os.path.join(model_root, args.model, "best_f1_model")
    model = AutoModelForTokenClassification.from_pretrained(model_path,
                                                            ignore_mismatched_sizes=True,
                                                            id2label=id2label,
                                                            label2id=label2id).to(device)


    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    """
        Prepare Test Dataset
    """
    test_data_file = 'test_data/anlp-sciner-test-sentences.txt'
    test_dataset = get_test_dataset(test_data_file)
    test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


    """
        Predict
    """
    train_args = TrainingArguments(output_dir='test_predict',
                                    per_device_eval_batch_size=args.batch_size)
    trainer = Trainer(model=model,
                        args=train_args,
                        tokenizer=tokenizer,
                        data_collator=data_collator)
    raw_preds = trainer.predict(test_dataset)
    
    predictions = np.argmax(raw_preds.predictions.squeeze(), axis=-1)
    label_ids = raw_preds.label_ids
    print(raw_preds.predictions)
    print(raw_preds.label_ids)
    
    # predictions = torch.tensor(raw_preds.predictions)
    # predictions = torch.argmax(predictions.squeeze(), axis=-1)
    # num_sent, sent_len = predictions.size()
    # preds = [[None] * sent_len for i in range(num_sent)]
    # for i in range(num_sent):
    #     for j in range(sent_len):
    #         p = predictions[i][j].item()
    #         preds[i][j] = model.config.id2label[p]
    # '''
    # with open(output_path, 'w', encoding='utf-8') as f:
    #     for i in range(num_sent):
    #         for j in range(sent_len):
    #             id = tokens.input_ids[i][j]
    #             token = tokenizer.decode([id])
    #             label = preds[i][j]
    #             if token == "[CLS]":
    #                 continue
    #             if token == "[SEP]":
    #                 f.write("\n")
    #                 break
    #             f.write(token + " " + label + "\n")
    # '''
    # # with open(output_path, 'w', encoding='utf-8') as f:
    # #     prev_token, prev_label = None, None
    # #     for i in range(num_sent):
    # #         for j in range(sent_len):
    # #             id = tokens.input_ids[i][j]
    # #             token = tokenizer.decode([id])
    # #             label = preds[i][j]
    # #             if token == "[CLS]":
    # #                 continue
    # #             if token == "[SEP]":
    # #                 f.write(prev_token + " " + prev_label + "\n\n")
    # #                 prev_token = None
    # #                 break
    # #             if token[:2] != "##":
    # #                 if prev_token:
    # #                     f.write(prev_token + " " + prev_label + "\n")
    # #                 prev_token = token
    # #                 prev_label = label
    # #             else:
    # #                 prev_token = prev_token + token[2:]