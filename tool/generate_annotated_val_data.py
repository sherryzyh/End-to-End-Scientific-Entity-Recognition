from argparse import ArgumentParser
import os
import json
import torch
import numpy as np

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_source', '-s',
                        type=str,
                        default='Dataset/validation_data')
    parser.add_argument('--data_gt', '-g',
                        type=str,
                        default='predictions/validation_data_ground_truth.conll')
    parser.add_argument('--data_tokens', '-t',
                        type=str,
                        default='predictions/validation_data_tokens.txt')
    args = parser.parse_args()

    val_data_with_ground_truth = []
    val_data_tokens = []
    for file in os.listdir(args.data_source):
        with open(os.path.join(args.data_source, file), "r", encoding="utf-8") as f:
            token_pairs = f.read().splitlines()
        non_null_token_pairs = []
        tokens = []
        for pair in token_pairs:
            pair_split = pair.strip().split(" ")
            if len(pair_split) == 1 and len(pair_split[0]) > 0:
                continue
            elif len(pair_split) == 2:
                non_null_token_pairs.append(pair)
                token, entity = pair_split
                tokens.append(token)
            elif len(pair_split) == 1:
                non_null_token_pairs.append(pair)
                tokens.append("\n")
            else:
                print("pair_split:", pair_split)

        val_data_tokens.append((" ".join(tokens)).strip())
        # val_data_tokens += tokens
        # print(non_null_token_pairs)
        val_data_with_ground_truth += non_null_token_pairs
    
    with open(args.data_gt, "w", encoding="utf-8") as f:
        f.writelines("\n".join(val_data_with_ground_truth))
    
    data_token_output = os.path.join("Dataset", "test_data", "validation_data_for_predict.txt")
    with open(data_token_output, "w", encoding="utf-8") as f:
        f.writelines("\n".join(val_data_tokens))
