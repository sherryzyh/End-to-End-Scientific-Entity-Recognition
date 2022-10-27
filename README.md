# End-to-End-Scientific-Entity-Recognition

## Environment setup
### Python(Conda) Environment
Environment setup:
1. Create and activate conda environment, with python=3.8
   `conda create -n ser python=3.8 -y`
   
2. Activate conda environment
   `conda activate ser`
   
3. Run setup script to install all required libraries
   `bash setup.sh`

4. Install OpenAI
   `pip install openai`

### OPENAI Environment Variables Setup
1. Create the `.env` file in the project root.   
2. Configure the necessary environment variables in it, e.g.
```
OPENAI_API_KEY={YOUR_API_KEY}
```   
## Pipeline
### Dataset Preparation
- Use [raw data collector](https://github.com/sherryzyh/End-to-End-Scientific-Entity-Recognition/edit/main/README.md#raw-data-collector) and [data augmenter](https://github.com/sherryzyh/End-to-End-Scientific-Entity-Recognition/edit/main/README.md#data-augementer) to build scientific research paper dataset
- Manually annotate supervised dataset for Scientific Name Entity Recognition, and use [data cleaner](https://github.com/sherryzyh/End-to-End-Scientific-Entity-Recognition/edit/main/README.md#annotated-data-cleaner) to clean the dataset

### (Optional) Pre-training on Unsupervised Dataset
```
python mlm.py -c [config_file]
```

### Fine-tuning
```
python main.py -c [config_file]
```

### Predict
- For each experiment, trainer automatically save checkpoints for each epoch, use [find best model](https://github.com/sherryzyh/End-to-End-Scientific-Entity-Recognition/edit/main/README.md#find-best-model) script to find best models
- Predict data should be **ONE** text file with **ONE** sentence per line.
- Prepare your predict config file
   - Modify "sentence_file" path
   - Modify "model_name" from your saved best model
```
python predict.py -c [config_file]
```

## Modules and Tools
### Raw Data Collector
- Collect Supervised Data
   - including collecting papers from ACL anthology
      - 10 papers per conference for year 2021 - 2022
      - 5 papers per conference for year 2015 - 2020
      - 1 paper per conference for year before 2015
   - parsing pdf papers
      - sentence per line
   - tokenizing papers
      - into conll style
```
python tool/raw_data_collector.py -c -p -t
[-c][--collect]  scrape paper
[-p][--parse]    parse pdf paper
[-t][--tokenize] tokenize paper
```
- Collect Unsupervised Data: similar to supervised data except that we collect all papers to do pre-training
- Ues raw data collecter to collect all papers from year range
- Use build unsupervised dataset tool to split a year-conference-balanced smaller dataset, depending on your computational resource
```
python tool/raw_data_collector.py -un
python tool/build_unsupervised_dataset.py
```

### Data Augementer
- Module: `annotated_data_augmenter.py`     
- Driver: `data_aug_driver.py`     
- Testing File/Playground: `data_aug_test.ipynb`      
**MUST READ BEFORE USING**: Before using the module, we must configure the API Key for OpenAI in `.env` file.     
```
OPENAI_API_KEY={YOUR_API_KEY}
```
Since the data augmenter is relying on the paid service, it also relies on the rate limiting feature lying in `utils.py`.  

### Annotated Data Cleaner
- Detect possible errors introduced by manual annotations
   - Len(line) $\neq$ 2
   - Label mis-spelling
```
python tool/clean_annotated_data.py
```
### Data Analyzer
- Analyze the statistics of supervised dataset
```
python tool/data_analyzer.py
```
### Find Best Model
- Find best models
   - with maximum f1 score
   - with minimum evaluation loss

## Reference

https://huggingface.co/docs/transformers/tasks/token_classification

https://github.com/marcellusruben/medium-resources/blob/main/NER_BERT/NER_with_BERT.ipynb

https://medium.com/@andrewmarmon/fine-tuned-named-entity-recognition-with-hugging-face-bert-d51d4cb3d7b5
