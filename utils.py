'''
Common Utilities in the project 
for maintainability, wheel reusability and code consistency :>
'''

import spacy
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
import os
# import openai
# import json
# from openapi_schema_to_json_schema import to_json_schema
# from dotenv import load_dotenv
# from ratelimit import limits, RateLimitException, sleep_and_retry
# consts for rate limiting feature
MAX_CALLS_PER_MINUTE = 20
ONE_MINUTE = 60



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