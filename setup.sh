device=${1:-"cpu"}
cudaversion=${2:-"11.6"}

conda install -c conda-forge pypdf2 -y
conda install typing_extensions -y
conda install -c huggingface transformers -y
conda install -c huggingface -c conda-forge datasets -y
# if [${device} = "cpu"]; then
# #        echo ${device}
#         conda install pytorch torchvision torchaudio -c pytorch -y
# else
# #        echo ${device} ${cudaversion}
#         conda install pytorch torchvision torchaudio cudatoolkit=${cudaversion} -c pytorch -c conda-forge -y
# fi
conda install pytorch torchvision torchaudio cudatoolkit=${cudaversion} -c pytorch -c conda-forge -y
pip install -r requirement.txt
python3 -m spacy download en_core_web_sm
