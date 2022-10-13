import os
import spacy
from PyPDF2 import PdfReader

source = "NLP"
destination = "parsed_pdf_txt_files"
nlp = spacy.load('en_core_web_sm')

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
                    tokens = nlp(text)
                    for sent in tokens.sents:
                        txt_file.write(sent.text.strip() + "\n")
                except Exception as e:
                    print(f"An exception was raised while parsing {txt}:")
                    print(e)
                    break