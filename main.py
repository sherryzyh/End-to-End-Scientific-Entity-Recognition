import pandas as pd
import torch 
import numpy as np
from transformers import BertTokenizerFast, BertForTokenClassification
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import SGD
from argparse import ArgumentParser
import json
import random

from common import DataSequence



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/baseline.json',
                        help='Configuration file to use')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed for training')
    args = parser.parse_args()
    config_file = args.config
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    general_args = config['general_args']
    data_args = config['data']
    tokenization_args = config['tokenization']
    
    train_data = data_args['train_data']
    test_data = data_args['test_data']
    unique_labels = data_args['unique_labels'].split(',')
    transformer = general_args['transformer']
    label_all_tokens = general_args['label_all_tokens']
    
    SEED = args.seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    
    df_train = pd.read_csv(train_data)
    df_test = pd.read_csv(test_data)
    labels_to_ids = {k: v for v, k in enumerate(unique_labels)}
    ids_to_labels = {v: k for v, k in enumerate(unique_labels)}
    tokenizer = BertTokenizerFast.from_pretrained(transformer)
