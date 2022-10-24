from transformers import (
    DataCollatorForTokenClassification,
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    set_seed,
    AutoConfig,
    AutoModelForMaskedLM
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

def get_mlm_dataset(directory, tokenizer, blocksentence):
    paper_list = os.listdir(directory)
    text = []
    block = ""
    count = 0
    for paper in paper_list:
        read_path = os.path.join(directory, paper)
        try:
            with open(read_path, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
        except:
            print(f"An exception occurred when reading << {paper} >>")
            continue
        for line in lines:
            if count < blocksentence:
                block += line + " "
                count += 1
            else:
                text.append(block)
                count = 0
                block = ""
    if block != "":
        text.append(block)
        block = ""
        count = 0

    print("#sentences:", len(text))
    print("sample:", text[0])
    print("*" * 20)
    print(len(text[0].strip().split(" ")), len(text[0]))
    inputs = tokenizer(text=text,
                       return_tensors="pt",
                       padding="longest",
                       max_length=512,
                       is_split_into_words=False)
    print(inputs.keys())
    print(f"input['input_ids'] - len: {len(inputs['input_ids'])}, sample: {len(inputs['input_ids'][0])}")
    print(f"input['token_type_ids'] - len: {len(inputs['token_type_ids'])}, sample: {inputs['token_type_ids'][0]}")
    print(f"input['attention_mask'] - len: {len(inputs['attention_mask'])}, sample: {inputs['attention_mask'][0]}")

    print("*" * 20)

    return Dataset.from_dict({"text": text, "label": text})

# def tokenize_function():

if __name__ == '__main__':
    # # parse command line arguments and load config
    # parser = ArgumentParser()
    # parser.add_argument('--config', type=str, default='./config/baseline.json',
    #                     help='Configuration file to use')
    # parser.add_argument('--resume_from_checkpoint', type=str, default=None,
    #                     help='Resume from a saved checkpoint')
    # args = parser.parse_args()
    # config_file = args.config
    # checkpoint = args.resume_from_checkpoint
    # with open(config_file, 'r') as f:
    #     config = json.load(f)
    #
    # general_args = config['general_args']
    # data_args = config['data_args']
    # train_args = config['train_args']
    # num_sentence = data_args['num_sentence_per_seq']
    # transformer = general_args['transformer']
    #
    # set_seed(train_args['seed'])
    # # model = AutoModelForTokenClassification.from_pretrained(transformer, ignore_mismatched_sizes=True,
    # #                                                         id2label=id2label, label2id=label2id)
    # # print(model.config)

    # num_sentence = 20
    # config_file = './config/baseline.json'
    # with open(config_file, 'r') as f:
    #     config = json.load(f)
    # general_args = config['general_args']
    # transformer = general_args['transformer']
    # seq_len_stats = []
    # tokenizer = AutoTokenizer.from_pretrained(transformer)
    # model = AutoModelForMaskedLM.from_pretrained("dslim/bert-base-NER",
    #                                              ignore_mismatched_sizes=True,
    #                                              id2label=id2label,
    #                                              label2id=label2id)
    # print(model.config)

    train_dataset = get_dataset("Raw_Unsupervised_Data/annotation_paper/")
    # val_dataset = get_dataset("Annotated_Data/cleaned_data", num_sentence)

    # Build training dataset

    # train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
    # print(train_dataset)

    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    # dataset = SERDataset(directory="Raw_Unsupervised_Data",
    #                      tokenizer=tokenizer,
    #                      mlm=True)
    dataset = get_mlm_dataset(directory="Raw_Unsupervised_Data/tokenized_paper", tokenizer=tokenizer, blocksentence=10)

    batch_size = 32
    num_workers = 4
    # sampler = BatchSampler(RandomSampler(dataset), batch_size=batch_size, drop_last=False)
    # dataloader = DataLoader(dataset=dataset,
    #                         batch_size=batch_size,
    #                         shuffle=False,
    #                         num_workers=num_workers)

    # for i, data in enumerate(dataset):
    #     print(f"len(dataset) = {len(dataset)}")
    #     print(f"i = {i} | len(data) = {len(data)} | len(data['text']) = {len(data['text'])} | data keys: {data.keys()} | data sample: {data['text']}, {data['label']}")
    #     # print(data)
    #     break

    # inputs = tokenizer(text=text,
    #                    padding="longest",
    #                    truncation=True,
    #                    max_length=512,
    #                    return_tensors="pt")
    # print(inputs.keys())


    # training_args = TrainingArguments(
    #     output_dir="./results",
    #     num_train_epochs=10,
    #     per_gpu_train_batch_size=64,
    #     save_steps=10000,
    #     evaluation_strategy="epoch",
    #     learning_rate=2e-5,
    #     weight_decay=0.01,
    # )
    # print(len(train_dataset), len(val_dataset))
    # tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    # data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=val_dataset,
    #     data_collator=data_collator,
    # )
    #
    # trainer.train()