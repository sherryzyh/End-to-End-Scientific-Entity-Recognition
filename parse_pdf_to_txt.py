import os
from PyPDF2 import PdfReader

source = "NLP"
destination = "parsed_pdf_txt_files"
for folder in os.listdir(source):
    if not os.path.isdir(os.path.join(source, folder)):
        continue
    folder_info = folder.split("_")
    conference, year = folder_info[0], folder_info[1]
    for pdf in os.listdir(os.path.join(source, folder)):
        reader = PdfReader(os.path.join(source, folder, pdf))
        article = pdf.split(".")[1]
        txt = year + "_" + conference + "_" + article + ".txt"
        save_as = os.path.join(destination, txt)
        with open(save_as, "w") as txt_file:
            for page in reader.pages:
                try:
                    txt_file.write(page.extract_text())
                except Exception as e:
                    print(f"An exception was raised while parsing {txt}:")
                    print(e)
                    break