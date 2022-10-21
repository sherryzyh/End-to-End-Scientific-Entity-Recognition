import os
from PyPDF2 import PdfReader
from utils import MyTokenizer, ACLScraper
import spacy
import argparse


class RawDataCollector:
    def __init__(self,
                 raw_data_root="Raw_Data",
                 pdf_root="pdf_paper",
                 parsed_root="parsed_paper",
                 tokenized_root="tokenized_paper",
                 annotate_data_root="annotation_paper"):
        self.pdf_root = os.path.join(raw_data_root, pdf_root)
        self.parsed_root = os.path.join(raw_data_root, parsed_root)
        self.tokenized_root = os.path.join(raw_data_root, tokenized_root)
        self.annotate_data_root = os.path.join(raw_data_root, annotate_data_root)

        if not os.path.exists(self.pdf_root):
            os.mkdir(self.pdf_root)
        if not os.path.exists(self.parsed_root):
            os.mkdir(self.parsed_root)
        if not os.path.exists(self.tokenized_root):
            os.mkdir(self.tokenized_root)
        if not os.path.exists(self.annotate_data_root):
            os.mkdir(self.annotate_data_root)

        self.scraper = ACLScraper(self.pdf_root)
        self.tokenizer = MyTokenizer()

    def collect_pdf_papers(self):
        for year in range(2014, 2015):
            if year < 2015:
                self.scraper.getEachConferenceForYear(year, 1)
            elif 2015 <= year <= 2020:
                self.scraper.getEachConferenceForYear(year, 5)
                # self.scraper.getACLsForYear(year, 5)
                # self.scraper.getNAACLsForYear(year, 5)
                # self.scraper.getEMNLPsForYear(year, 5)
            elif year > 2020:
                self.scraper.getEachConferenceForYear(year, 10)
                # self.scraper.getACLsForYear(year, 10)
                # self.scraper.getNAACLsForYear(year, 10)
                # self.scraper.getEMNLPsForYear(year, 10)

    def parse_papers(self, source, destination):
        nlp_parser = spacy.load('en_core_web_sm')
        for folder in os.listdir(source):
            if not os.path.isdir(os.path.join(source, folder)):
                continue

            for pdf in os.listdir(os.path.join(source, folder)):
                reader = PdfReader(os.path.join(source, folder, pdf))
                article_info = pdf.split(".")
                year, conference, article, _  = article_info
                txt = year + "_" + conference + "_" + article + ".txt"
                save_as = os.path.join(destination, txt)
                with open(save_as, "w", encoding="utf-8") as txt_file:
                    for page in reader.pages:
                        try:
                            raw_text = page.extract_text()
                            text_list = raw_text.split("\n")
                            text = " ".join(text_list)
                            tokens = nlp_parser(text)
                            for sent in tokens.sents:
                                txt_file.write(sent.text.strip() + "\n")
                        except Exception as e:
                            print(f"An exception was raised while parsing {txt}:")
                            print(e)
                            break

    def prep_one_paper(self, read_path, tokenized_path, anno_raw_path):
        with open(read_path, "r", encoding='utf-8') as f:
            lines = f.read().splitlines()

        annof = open(anno_raw_path, "w", encoding='utf-8')
        tokenized_lines = []
        for l in lines:
            # tokenize the line in the paper
            token_space_str, token_label_str = self.tokenizer.get_tokenized_line(l)
            tokens_list = token_space_str
            tokenized_lines.append(tokens_list)
            annof.write(token_label_str)
            annof.write("\n")
        annof.close()

        tokenf = open(tokenized_path, "w", encoding='utf-8')
        tokenf.write("\n".join(tokenized_lines))
        tokenf.close()

    def tokenize_papers(self):
        paper_list = sorted(os.listdir(self.parsed_root))
        for paper in paper_list:
            read_path = os.path.join(self.parsed_root, paper)
            tokenized_path = os.path.join(self.tokenized_root, f"tokenized_{paper}")
            anno_raw_path = os.path.join(self.annotate_data_root, f"anno_{paper}")
            self.prep_one_paper(read_path, tokenized_path, anno_raw_path)

    def prep_raw_data(self, tokenize, parse, collect=False):
        print("*" * 40)
        print(f"collect :{collect}")
        print(f"parse   :{parse}")
        print(f"tokenize:{tokenize}")
        print("*" * 40)

        if collect:
            self.collect_pdf_papers()
        if parse:
            self.parse_papers(self.pdf_root, self.parsed_root)
        if tokenize:
            self.tokenize_papers()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='ARGUMENTS: ALL DEFAULT SET TO FALSE')

    parser.add_argument('--collect', '-c', action='store_true', help="collecting papers from the ACL Anthology")
    parser.add_argument('--parse', '-p', action='store_true', help="reading the pdf papers and parsing it into txt")
    parser.add_argument('--tokenize', '-t', action='store_true', help="tokenizing the parsed paper")

    args = parser.parse_args()

    project_root = os.getcwd()
    DataCollector = RawDataCollector(raw_data_root=os.path.join(project_root, "Raw_Data"))
    DataCollector.prep_raw_data(tokenize=args.tokenize,
                                parse=args.parse,
                                collect=args.tokenize)