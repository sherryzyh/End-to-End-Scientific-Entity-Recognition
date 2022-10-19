import os
from PyPDF2 import PdfReader
import json
from bs4 import BeautifulSoup
import json
import pwd
import numpy as np
import requests
from tqdm import tqdm
from utils import MyTokenizer
import spacy

class ACLScraper:
    def __init__(self, workingdir = '/content/drive/MyDrive/NLP'):
        os.chdir(workingdir)
        self.cnt = 0

    def prepareACLInfoForYear(self, year):
        if year > 2022 or year < 2000:
            print(f"{year} is a Wrong Year for ACL, Please Try Another Year")
            return
        self.page_url = f"https://aclanthology.org/events/acl-{year}"
        self.conf_name = f'acl_{year}_main'
        if year == 2020:
            self.conf_id = f'{year}-acl-main'
        elif year > 2020:
            self.conf_id = f'{year}-acl-long'
        else:
            year = str(year)
            self.conf_id = f'p{year[2:]}-1'
        return True

    def prepareEMNLPInfoForYear(self, year):
        if year >= 2022 or year < 2010:
            print(f"{year} is a Wrong Year for EMNLP, Please Try Another Year")
            return False
        self.page_url = f"https://aclanthology.org/events/emnlp-{year}/"
        self.conf_name = f"emnlp_{year}_main"
        if year >= 2020:
            self.conf_id = f"{year}-emnlp-main"
        else:
            year = str(year)
            self.conf_id = f"d{year[2:]}-1"
        return True

    def prepareNAACLInfoForYear(self, year):
        invalidYears = {2011, 2014, 2017, 2020}
        if year > 2022 or year < 2010 or year in invalidYears:
            print(f"{year} is a Wrong Year for NAACL, Please Try Another Year")
            return False
        self.page_url = f"https://aclanthology.org/events/naacl-{year}/"
        self.conf_name = f"naacl_{year}_main"
        if year >= 2020:
            self.conf_id = f"{year}-naacl-main"
        else:
            year = str(year)
            self.conf_id = f"n{year[2:]}-1"
        return True

    def getACLsForYear(self, year, num_limit = None):
        if self.prepareACLInfoForYear(year):
            self.scrape(year, num_limit)

    def getNAACLsForYear(self, year, num_limit = None):
        if self.prepareNAACLInfoForYear(year):
            self.scrape(year, num_limit)

    def getEMNLPsForYear(self, year, num_limit = None):
        if self.prepareEMNLPInfoForYear(year):
            self.scrape(year, num_limit)

    def scrape(self, year, num_limit):
        # %%
        html_doc = requests.get(self.page_url).text
        soup = BeautifulSoup(html_doc, 'html.parser')
        # %%
        main_papers = soup.find('div', id = self.conf_id).find_all('p', class_ = "d-sm-flex")
        paper_list = []
        for paper_p in main_papers:
            pdf_url = paper_p.contents[0].contents[0]['href']
            paper_span = paper_p.contents[-1]
            assert paper_span.name == 'span'
            paper_a = paper_span.strong.a
            title = paper_a.get_text()
            url = "https://aclanthology.org" + paper_a['href']
            paper_list.append([title, url, pdf_url])
            if num_limit is not None:
                if len(paper_list) >= num_limit:
                    break
        with open(self.conf_name + '.json', 'w', encoding='utf8') as f:
            json.dump(paper_list, f, indent = 2, ensure_ascii= False)

        print(f'Scraped total {len(paper_list)} papers for {self.conf_name}')

        if not os.path.exists(self.conf_name):
            os.mkdir(self.conf_name)

        illegal_chr = r'\/:*?<>|'
        table = ''.maketrans('', '', illegal_chr)
        for i, paper in tqdm(list(enumerate(paper_list))):
            r = requests.get(paper[2])
            n = '{}.{}.pdf'.format( i +1, paper[0].translate(table))
            with open('./{}/{}'.format(self.conf_name, n), 'wb') as f:
                f.write(r.content)
            self.cnt += 1


class RawDataCollector:
    def __init__(self,
                 raw_data_root="Raw_Data",
                 pdf_root="pdf_paper",
                 parsed_root="parsed_paper",
                 tokenized_root="tokenized_paper",
                 annotate_data_root="annotation_paper"):
        self.pdf_root = os.path.join(raw_data_root, pdf_root)
        # print(f"pdf_root: {self.pdf_root}")
        self.parsed_root = os.path.join(raw_data_root, parsed_root)
        # print(f"parsed_root: {self.parsed_root}")
        self.tokenized_root = os.path.join(raw_data_root, tokenized_root)
        # print(f"tokenized_root: {self.tokenized_root}")
        self.annotate_data_root = os.path.join(raw_data_root, annotate_data_root)
        if not os.path.exists(self.parsed_root):
            os.mkdir(self.parsed_root)
        if not os.path.exists(self.tokenized_root):
            os.mkdir(self.tokenized_root)
        if not os.path.exists(self.annotate_data_root):
            os.mkdir(self.annotate_data_root)

        self.scraper = ACLScraper(self.pdf_root)
        self.tokenizer = MyTokenizer()

    def collect_pdf_papers(self):
        for year in range(2015, 2023):
            if 2015 <= year <= 2020:
                self.scraper.getACLsForYear(year, 5)
                self.scraper.getNAACLsForYear(year, 5)
                self.scraper.getEMNLPsForYear(year, 5)
            else:
                self.scraper.getACLsForYear(year, 10)
                self.scraper.getNAACLsForYear(year, 10)
                self.scraper.getEMNLPsForYear(year, 10)

    def parse_papers(self, source, destination):
        nlp_parser = spacy.load('en_core_web_sm')
        for folder in os.listdir(source):
            if not os.path.isdir(os.path.join(source, folder)):
                continue
            folder_info = folder.split("_")
            conference, year = folder_info[0], folder_info[1]
            for pdf in os.listdir(os.path.join(source, folder)):
                reader = PdfReader(os.path.join(source, folder, pdf))
                article_info = pdf.split(".")
                order, article = article_info[0], article_info[1]
                if order == "1":
                    continue
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
        with open(read_path, "r") as f:
            lines = f.read().splitlines()

        annof = open(anno_raw_path, "w")
        tokenized_lines = []
        for l in lines:
            # tokenize the line in the paper
            token_space_str, token_label_str = self.tokenizer.get_tokenized_line(l)
            tokens_list = token_space_str
            tokenized_lines.append(tokens_list)
            annof.write(token_label_str)
            annof.write("\n")
        annof.close()

        tokenf = open(tokenized_path, "w")
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
        if collect:
            self.collect_pdf_papers()
        if parse:
            self.parse_papers(self.pdf_root, self.parsed_root)
        if tokenize:
            self.tokenize_papers()

if __name__=="__main__":
    project_root = os.getcwd()
    DataCollector = RawDataCollector(raw_data_root=os.path.join(project_root, "Raw_Data"))
    DataCollector.prep_raw_data(tokenize=True,
                                parse=False,
                                collect=False)