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
                 annotate_data_root="annotation_paper",
                 collection_mode="supervised"):
        self.pdf_root = os.path.join(raw_data_root, pdf_root)
        self.parsed_root = os.path.join(raw_data_root, parsed_root)
        self.tokenized_root = os.path.join(raw_data_root, tokenized_root)
        self.annotate_data_root = os.path.join(raw_data_root, annotate_data_root)
        self.collection_mode = collection_mode

        if not os.path.exists(raw_data_root):
            os.mkdir(raw_data_root)
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
        self.nlp_parser = spacy.load('en_core_web_sm')

    def collect_pdf_papers(self, year_start = 2015, year_end = 2023):
        if self.collection_mode == "supervised":
            for year in range(year_start, year_end):
                if year < 2015:
                    num_limit = 1
                elif 2015 <= year <= 2020:
                    num_limit = 5
                elif year > 2020:
                    num_limit = 10
                self.scraper.getEachConferenceForYear(year, num_limit)
        elif self.collection_mode == "unsupervised":
            for year in range(year_start, year_end):
                self.scraper.getEachConferenceForYear(year)

    def parse_papers(self, source, destination):
        print("***** parse *****")
        for folder in os.listdir(source):
            if not os.path.isdir(os.path.join(source, folder)):
                continue

            for pdf in os.listdir(os.path.join(source, folder)):
                if not pdf.endswith(".pdf"):
                    continue
                reader = PdfReader(os.path.join(source, folder, pdf))
                article_info = pdf.split(".")
                year = article_info[0]
                conference = article_info[1]
                publish_info = f"{year}_{conference}_"
                txt = publish_info + pdf[len(publish_info):-4] + ".txt"
                save_as = os.path.join(destination, txt)
                with open(save_as, "w", encoding="utf-8") as txt_file:
                    for page in reader.pages:
                        try:
                            raw_text = page.extract_text()
                            text_list = raw_text.split("\n")
                            text = " ".join(text_list)
                            tokens = self.nlp_parser(text)
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
        print("***** tokenize *****")
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
    parser.add_argument('--unsupervised', '-un', action='store_true', help="unsupervised dataset collection")
    args = parser.parse_args()

    project_root = os.getcwd()
    if args.unsupervised:
        print("Unsupervised Dataset Collection")
        DataCollector = RawDataCollector(raw_data_root=os.path.join(project_root, "Unsupervised_Data"),
                                         collection_mode="unsupervised")
        DataCollector.prep_raw_data(tokenize=True,
                                    parse=True,
                                    collect=False)
    else:
        DataCollector = RawDataCollector(raw_data_root=os.path.join(project_root, "Raw_Data"))
        DataCollector.prep_raw_data(tokenize=args.tokenize,
                                    parse=args.parse,
                                    collect=args.collect)