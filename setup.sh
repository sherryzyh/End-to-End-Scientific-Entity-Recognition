conda install -c conda-forge pypdf2 -y
conda install typing_extensions -y
conda install -c huggingface transformers
conda install -c huggingface -c conda-forge datasets
conda install pytorch torchvision torchaudio -c pytorch # specify cuda version on Linux
python3 -m pip install -r requirement.txt
python3 -m spacy download en_core_web_sm