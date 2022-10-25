'''
This file runs each experiment. 
Configs can be specified using command line argument "--config".
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
import json
import os
import shutil
from datetime import datetime
import pytz
import statistics

from common import get_dataset, compute_metrics, id2label, label2id, CustomTrainer

if __name__ == '__main__':
    # parse command line arguments and load config
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, help='Model to load', required=True)
    parser.add_argument('--test_data', type=str, help='Data to test on', required=True)
    args = parser.parse_args()
    model_path = os.path.join('results', args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    test_data_file = args.test_data
    with open(test_data_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    paragraph = '''Before proceeding further, I should like to inform members that action on draft resolution iv, entitled situation of human rights of Rohingya Muslims and other minorities in Myanmar is postponed to a later date to allow time for the review of its programme budget implications by the fifth committee. The assembly will take action on draft resolution iv as soon as the report of the fifth committee on the programme budget implications is available. I now give the floor to delegations wishing to deliver explanations of vote or position before voting or adoption.'''
    tokens = tokenizer(paragraph)
    torch.tensor(tokens['input_ids']).unsqueeze(0).size()

    model = AutoModelForTokenClassification.from_pretrained(model_path, id2label=id2label, label2id=label2id)
    predictions = model.forward(input_ids=torch.tensor(tokens['input_ids']).unsqueeze(0), attention_mask=torch.tensor(tokens['attention_mask']).unsqueeze(0))
    predictions = torch.argmax(predictions.logits.squeeze(), axis=1)
    predictions = [id2label[i] for i in preds]

    words = tokenizer.batch_decode(tokens['input_ids'])
    pd.DataFrame({'ner': predictions, 'words': words}).to_csv('un_ner.csv')
