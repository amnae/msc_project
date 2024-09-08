---

## Installation Instructions

To get started with this project, follow the steps below to install OpenCompass and set up the necessary files for testing custom models.

### 1. Install OpenCompass

First, install OpenCompass by following the official instructions on their GitHub page (https://github.com/open-compass/opencompass). Below is a guide to help you:

```bash
# Install zip and unzip utilities
sudo apt-get install zip unzip

# Create and activate a conda environment with Python 3.10 and PyTorch
conda create --name opencompass python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
conda activate opencompass

# Clone the OpenCompass repository
git clone https://github.com/open-compass/opencompass opencompass
cd opencompass

# Install OpenCompass in editable mode
pip install -e .

# Install additional dependencies
pip install oss2

# Download and unzip the OpenCompass dataset
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip

# Install ModelScope
pip install modelscope

# Set the dataset source environment variable
export DATASET_SOURCE=ModelScope
```

### 2. File Movements and Modifications

Once OpenCompass is installed, a few files need to be moved and certain configurations adjusted to enable testing of custom models.

#### Move Files

Move the following files into the appropriate directories within the OpenCompass project:

- `base_model_config.py` → `opencompass/configs/`
- `edullmodel.py` → `opencompass/opencompass/`  (Ensure the file path is correct; this step might need clarification)
- `huggingface_custom.py` → `opencompass/opencompass/models/`
- `dev-v2.0.json` → `opencompass/opencompass/data/SQUAD2.0/`

#### Edit Files

1. **Edit `opencompass/opencompass/models/__init__.py`**  
   Add the following import at the top of the file to include the custom HuggingFace model:

   ```python
   from .huggingface_custom import EduLLMwithChatTemplate
   ```

2. **Edit `opencompass/opencompass/datasets/gsm8k.py`**  
   Modify line 23 to load the dataset properly:

   ```python
   dataset = MsDataset.load(dataset_name=path)  # Uncomment trust_remote_code if needed
   ```

---

This setup should configure OpenCompass with the necessary modifications to test the models. Make sure to verify paths and configurations based on your specific environment.