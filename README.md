# MSC Project

This repository contains the code for my MSc Data Science and Machine Learning project.

## Table of Contents

- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Setup](#setup)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Structure

```
msc_project-main/
│
├── analysis/
│   ├── static_analysis.py       # Script for performing static analysis of the models
│   ├── dynamic_analysis.py      # Script for performing dynamic analysis of the models
│   └── wikitext103_test.csv     # Text used in dynamic analysis
├── eval/
│   ├── base_model_config.py     # Base configuration for model evaluation
│   ├── dev-v2.0.json            # Dataset for evaluation
│   ├── huggingface_custom.py    # Custom HuggingFace utilities for opencompass evaluation
│   └── README.md                # Instructions related to evaluation
│
├── training/
│   ├── create_sparse_moe.py     # Script for creating a sparse Mixture of Experts model
│   ├── create_train_data.py     # Script for creating the training dataset
│   ├── training.py              # Script for model training
│   └── __init__.py              # Init file for the training module
│
├── modelling_edullm.py           # Main script for model creation and handling
├── __init__.py                   # Project-level init file
└── README.md                     # Project-level README file
```

## Dependencies

To run the scripts in this project, you need the following libraries:

- Python 3.x
- Transformers (HuggingFace)
- PyTorch
- NumPy
- JSON

You can install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/username/msc_project-main.git
   cd msc_project-main
   ```

2. Install the necessary dependencies as mentioned above.

3. Prepare the dataset for training or evaluation by following the instructions in the respective subdirectories (`training` and `eval`).

## Usage

### Training

The training scripts are located in the `training/` directory. You can create the training dataset and initiate training as follows:

```bash
# Create the training data
python training/create_train_data.py
```

```bash
# Set up and create a Sparse Mixture of Experts (MoE) model and upload it to HuggingFace.
python training/create_sparse_moe.py
```

#### Arguments:
- `--cache_dir` (`-c`): Cache directory for model weights.
- `--device` (`-d`): Device for training (e.g., `cpu`, `cuda`).

```bash
# Train the model
python training/training.py
```

#### Arguments:
- `--model_num` or `-m`: Model to train (`0`: Mixtral, `1`: Damex, `2`: XMOE, `all`: train all). (default: 0)
- `--cache_dir` or `-c`: Cache directory for models. (default: None)
- `--device` or `-d`: Device for training (`cpu`, `cuda`, `auto`). (default:'auto')
- `--num_epochs` or `-e`: Number of training epochs. (default: 1)
- `--batch_size` or `-b`: Training batch size. (default: 1)

This script loads datasets, trains models, and saves them to the HuggingFace Hub.

#### Usage:
```bash
python training/training.py --model_num 0 --device cuda --cache_dir .cache --num_epochs 1 --batch_size 1
```

### Evaluation

The evaluation scripts can be found in the `eval/` directory. You can run model evaluations by installinng opencompass, following the README.md in the `eval/` directory, and running the following command. Change the 'model_path' variable  in line 12 to the model you would like to test.

```bash
# Run evaluation
python run.py configs/base_model_config.py

```

### Static Analysis

You can run static analysis on the model using the script in the `analysis/` directory:

```bash
#Analyse model parameters like expert weight matrices and gate embeddings.
python analysis/static_analysis.py
```

#### Arguments:
- `--model_num` (`-m`): Model to analyze (`0`: Mixtral, `1`: Damex, `2`: XMOE, `'all'`: all models).
- `--cache_dir` (`-c`): Cache directory.
- `--device` (`-d`): Device for execution (`cpu`, `cuda`).

#### Usage:
```bash
python analysis/static_analysis.py --model_num 0 --device cuda --cache_dir .cache
```

### Dynamic Analysis

You can analyse dynamic behaviors of models, including expert outputs, norms, intermediate states, and chosen experts.

```bash
#Analyse dynamic model behaviors.
python analysis/dynamic_analysis.py
```

#### Arguments:
- `--cache_dir` (`-c`): Directory to cache models.
- `--device` (`-d`): Device for model execution (`cpu`, `cuda`).
- `--model_num` (`-m`): Model to analyze (`0`: Mixtral, `1`: Damex, `2`: XMOE, `all`: all models).

#### Usage:
```bash
python analysis/dynamic_analysis.py --model_num 0 --device cuda --cache_dir .cache
```

The script evaluates expert outputs and generates visualizations of dynamic model behaviors.