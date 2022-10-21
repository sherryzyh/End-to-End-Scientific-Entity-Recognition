'''
Common Utilities in the project 
for maintainability, wheel reusability and code consistency :>
'''

import spacy
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
import os
from bs4 import BeautifulSoup
import json
import requests
from tqdm import tqdm


# import openai
# from openapi_schema_to_json_schema import to_json_schema
# from dotenv import load_dotenv
# from ratelimit import limits, RateLimitException, sleep_and_retry

"""
    CONSTS for rate limiting feature
"""
MAX_CALLS_PER_MINUTE = 20
ONE_MINUTE = 60


class ACLScraper:
    def __init__(self, workingdir = '/content/drive/MyDrive/NLP'):
        os.chdir(workingdir)
        self.cnt = 0
        self.summary = os.path.join(workingdir, "summary.txt")
        with open(self.summary, "w", encoding="utf-8") as f:
            f.write("Scrapper Summary\n\n")

    def prepareACLInfoForYear(self, year):
        if year > 2022 or year < 2000:
            self.printsummary(year, "acl")
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
            self.printsummary(year, "emnlp")
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
            self.printsummary(year, "naacl")
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
            self.scrape(year, "acl", num_limit)

    def getNAACLsForYear(self, year, num_limit = None):
        if self.prepareNAACLInfoForYear(year):
            self.scrape(year, "naacl", num_limit)

    def getEMNLPsForYear(self, year, num_limit = None):
        if self.prepareEMNLPInfoForYear(year):
            self.scrape(year, "emnlp", num_limit)

    def getEachConferenceForYear(self, year, num_limit = None):
        self.getACLsForYear(year, num_limit)
        self.getNAACLsForYear(year, num_limit)
        self.getEMNLPsForYear(year, num_limit)

    def printsummary(self, year, conference, num=None):
        with open(self.summary, "a", encoding="utf-8") as f:
            if num is None:
                f.write(f'{conference:6}\t{year:4}\t{"N/A":6} paper\n')
            else:
                f.write(f'{conference:6}\t{year:4}\t{num:6} paper\n')

    def scrape(self, year, conference, num_limit):
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
            if "Proceedings" in title:
                continue
            paper_list.append([title, url, pdf_url])
            if num_limit is not None:
                if len(paper_list) >= num_limit:
                    break
        with open(self.conf_name + '.json', 'w', encoding='utf8') as f:
            json.dump(paper_list, f, indent = 2, ensure_ascii= False)

        self.printsummary(year, conference, len(paper_list))

        if not os.path.exists(self.conf_name):
            os.mkdir(self.conf_name)

        illegal_chr = r'\/:*?<>|'
        table = ''.maketrans('', '', illegal_chr)
        for i, paper in list(enumerate(paper_list)):
            r = requests.get(paper[2])
            n = '{}.{}.{}.pdf'.format(year, conference, paper[0].translate(table))
            with open('./{}/{}'.format(self.conf_name, n), 'wb') as f:
                f.write(r.content)
            self.cnt += 1


class MyTokenizer:
    def __init__(self):
        nlp = English()
        self.tokenizer = nlp.tokenizer

    def get_tokens(self, line):
        return self.tokenizer(line)

    def get_tokenized_line(self, line):
        '''
        Args:
            1. line: the line is from the raw paper
        Return:
            1. tokne_space_str: tokens from the line seperated by space, i.e. tokens_list of the given line
            2. token_label_str: "{token} O" pair for the tokens in the given line
        '''
        tokens = self.get_tokens(line)
        token_space_str = "".join([x.text + " " for x in tokens])
        token_label_str = "".join([x.text + " O\n" for x in tokens])
        return token_space_str, token_label_str



#
# class OpenAIClient:
#     '''
#     Leverage the trained GPT-3 Model posted by OpenAI for multiple usages (mainly for data aug in our project)
#     '''
#     def __init__(self):
#         '''
#         Notice you need to define the OPENAI_API_KEY in your .env file before using the OpenAIClient
#         Prepare the necessary environment for open ai gpt 3 model
#         '''
#         load_dotenv()
#         openai.api_key = os.getenv("OPENAI_API_KEY")
#         self.parapharase_prefix = "Paraphrase the following sentence: "
#         self.parapharase_model = "text-davinci-002"
#
#     @sleep_and_retry
#     @limits(calls=MAX_CALLS_PER_MINUTE, period=ONE_MINUTE)
#     def getParaphrasedSentence(self, sentence):
#         '''
#         The API calls is configured with the rate limiter (60 requests / minute)
#         Args:
#             sentence: original sentence which we try to paraphrase
#         Return:
#             paraphrased_sentence
#         '''
#         response = openai.Completion.create(
#         model=self.parapharase_model,
#         prompt=f"{self.parapharase_prefix}{sentence}",
#         temperature=0.7,
#         max_tokens=256,
#         top_p=1,
#         frequency_penalty=0,
#         presence_penalty=0
#         )
#         return self.getSentenceFromOpenAIResponse(response)
#
#
#     def getSentenceFromOpenAIResponse(self, response):
#         response_json = to_json_schema(response)
#         return response_json.get("choices")[0].get("text").strip()