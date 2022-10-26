import os
import shutil
from collections import defaultdict

if __name__ == '__main__':
    dataroot = "/work/yinghuan/data/SER_Unsupervised_Data"

    destination = "/work/yinghuan/data/SER_Unsupervised_Data_Large"

    venue = defaultdict(lambda: 0)
    for file in os.listdir(os.path.join(dataroot, "tokenized_paper")):
        article_info = file.split("_")
        year = article_info[1]
        conference = article_info[2]
        if venue[year+conference] > 30:
            continue
        tokenized_file = os.path.join(dataroot, "tokenized_paper", file)
        annotation_file = os.path.join(dataroot, "annotation_paper", "anno"+file[9:])
        shutil.copy2(tokenized_file, os.path.join(destination, "tokenized_paper", file))
        shutil.copy2(annotation_file, os.path.join(destination, "annotation_paper", file))
        venue[year+conference] += 1

    print(sum(venue.values()), " papers in total")
        