Install opencompass as descibed by the github page:

sudo apt-get install zip unzip
conda create --name opencompass python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
conda activate opencompass
git clone https://github.com/open-compass/opencompass opencompass
cd opencompass
pip install -e .
pip install oss2
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip
pip install modelscope
export DATASET_SOURCE=ModelScope

Once that has been installed, some files must be moved/changed in order to test the custom models 
Move:
base_model_config.py into opencompass/configs/
edullmodel.py into opencompass/opencompass/  ???
huggingface_custom.py into opencompass/opencompass/models/
dev-v2.0.json into opencompass/opencompass/data/SQUAD2.0/

Edit: 
opencompass/opencompass/models/__init__.py add 'from .huggingface_custom import EduLLMwithChatTemplate' to the top
opencompass/opencompass/datasets/gsm8k.py  change line 23 to 'dataset = MsDataset.load(dataset_name=path)#, trust_remote_code=True)