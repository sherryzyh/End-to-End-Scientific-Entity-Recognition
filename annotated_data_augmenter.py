import os
import json
import openai
import utils
import spacy
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
from dotenv import load_dotenv
import time

load_dotenv()

class AnnotatedDataAug:
    def __init__(self, working_dir = "./Annotated_Data"):
        os.chdir(working_dir)
        # if change the dir name of aug papers in Annotated Data we also need to change the parameters here
        self.aug_dir = "./clean_data_aug/"
        # if change the dir name of raw papers in Annotated Data we also need to change the parameters here
        self.raw_dir = "./cleaned_data/"
        self.openai_client = utils.OpenAIClient()
        self.tokenizer = utils.MyTokenizer()
    
    def recover_and_label(self, annotated_tokenized_paper: str):
        """
        we need to recover the tokenized paper to the original sentences,
        and obtain the label for each token of the given sentence. (the token->label mapping is specified within a given sentence)

        Args:
            annotated_tokenized_paper: the path of the paper that is consisted of (token, label) pair for each line
        Returns:
            A list of sentences in the original paper;  
            A dict which map the sentenceIdx to its corresponding token->label map
        """
        # global result
        sentences = []
        sentenceIdx_to_tokenLabelDict = {} # map the index of the sentence to the corresponding token->label dict
        with open(annotated_tokenized_paper) as f:
            lines = f.readlines()
            # cur maintainer
            cur_dict = {}
            label_set = set()
            cur_sentence = []
            for idx, token_label in enumerate(lines):
                if token_label == "\n":
                    # if split point
                    # add the period "." manually (always make sure there is a period at the end)
                    # if do not want only to paraphase labeled data, remove the condition
                    if len(label_set) > 0:
                        if cur_sentence and cur_sentence[-1] != ".":
                            cur_sentence.append(".")
                        sentences.append(" ".join(cur_sentence))
                        sentenceIdx_to_tokenLabelDict[len(sentences)-1] = cur_dict
                    cur_dict = {}
                    cur_sentence = []
                    label_set = set()
                    continue
                else:
                    # o.w.
                    token = " "
                    if token_label[0] != " ":
                        # if len(token_label.split(" ")) > 2:
                        #     print(token_label.split(" ")[0])
                        token_label = token_label.strip()
                        token, label = token_label.split(" ")
                        cur_dict[token] = label[:-1]
                        if label[0] == "B":
                            label_set.add(label)
                    cur_sentence.append(token)
            if cur_sentence:
                sentences.append(" ".join(cur_sentence))
        return sentences, sentenceIdx_to_tokenLabelDict
    
    def parapharase_and_relabel(self, sentences, sentenceIdx_to_tokenLabelDict):
        paraphrased_result_lines = []
        for idx, sentence in enumerate(sentences):
            parapharased_sentence = self.openai_client.getParaphrasedSentence(sentence)
            parapharased_tokens = self.tokenizer.get_tokens(parapharased_sentence)
            original_tokenLabelDict = sentenceIdx_to_tokenLabelDict[idx]
            para_token_label_lst = []
            for x in parapharased_tokens:
                lower_x = x.text.lower().strip()
                for key, label in original_tokenLabelDict.items():
                    if key.lower().strip() == lower_x and label != "O":
                        para_token_label_lst.append(f"{key} {label}\n")
                        break 
                else:
                    para_token_label_lst.append(x.text + " O\n")
            para_token_label_str = "".join(para_token_label_lst)
            paraphrased_result_lines.append(para_token_label_str)
        return paraphrased_result_lines
    
    def data_aug(self, raw_dir = None, aug_dir=None):
        raw_dir = raw_dir if raw_dir else self.raw_dir
        aug_dir = aug_dir if aug_dir else self.aug_dir
        auged_files = set(os.listdir(aug_dir))
        cnt = 0
        print(f"Already augemented files: {auged_files}" )
        for file in os.listdir(raw_dir):
            if file and file[0] == ".":
                continue
            print(f"Total Augmented File Cnt: {cnt}")
            aug_file = f"aug_{file}"
            print(f"Augmenting {aug_file}...")
            if aug_file in auged_files:
                print(f"Already Parsed, Skip the annotated paper {file}")
                continue
            
            print(f"Paraphrase and relabelling from OpenAI...")
            sentences, sentenceIdx_to_tokenLabelDict = self.recover_and_label(raw_dir + file)
            paraphrased_result_lines = self.parapharase_and_relabel(sentences, sentenceIdx_to_tokenLabelDict)
            
            print(f"Write the result to file...")
            aug_file_path = os.path.join(aug_dir, aug_file)
            
            tokenf = open(aug_file_path, "w")
            tokenf.write("\n".join(paraphrased_result_lines))
            tokenf.close()
            # update the seen set to avoid duplicate aug
            auged_files.add(aug_file)
            

        # os.chdir(aug_dir if aug_dir else self.aug_dir)

# working_dir = "./Annotated_Data"
# data_augmenter = AnnotatedDataAug(working_dir)
# data_augmenter.data_aug()



