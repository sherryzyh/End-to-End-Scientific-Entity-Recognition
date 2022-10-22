# End-to-End-Scientific-Entity-Recognition
## Environment setup:
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

### Environment Variables Setup
1. Create the `.env` file in the project root.   
2. Configure the necessary environment variables in it, e.g.
```
OPENAI_API_KEY={YOUR_API_KEY}
```   

## Modules & Dependency Configurations   
### Data Augementer
- Module: `annotated_data_augmenter.py`     
- Driver: `data_aug_driver.py`     
- Testing File/Playground: `data_aug_test.ipynb`      
MUST READ BEFORE USING: Before using the module, we must configure the API Key for OpenAI in `.env` file.     
```
OPENAI_API_KEY={YOUR_API_KEY}
```
Since the data augmenter is relying on the paid service, it also relies on the rate limiting feature lying in `utils.py`.  

### Common Utilities
- Module: `utils.py`  
- Functionalities
  - OpenAI Client   
  - Tokenizer   

The module defines the usages of some functionalities that are commonly used in preprocessing procedure (like `raw_data_collector` and `data_augmenter`).     

References: 
Baseline: 
https://huggingface.co/docs/transformers/tasks/token_classification
https://github.com/marcellusruben/medium-resources/blob/main/NER_BERT/NER_with_BERT.ipynb
https://medium.com/@andrewmarmon/fine-tuned-named-entity-recognition-with-hugging-face-bert-d51d4cb3d7b5
