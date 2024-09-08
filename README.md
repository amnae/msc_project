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
│   └── static_analysis.py       # Script for performing static analysis of the models│
├── eval/
│   ├── base_model_config.py     # Base configuration for model evaluation
│   ├── dev-v2.0.json            # Dataset for evaluation
│   ├── huggingface_custom.py    # Custom HuggingFace utilities for evaluation
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

*(Note: A `requirements.txt` file should be added with the necessary packages)*

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

# Train the model
python training/training.py
```

### Evaluation

The evaluation scripts can be found in the `eval/` directory. You can run model evaluations by configuring the settings in `base_model_config.py` and using the custom HuggingFace utilities:

```bash
# Run evaluation
python eval/huggingface_custom.py
```

### Static Analysis

You can run static analysis on the model using the script in the `analysis/` directory:

```bash
python analysis/static_analysis.py
```

## Contributing

If you'd like to contribute, please fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

Feel free to customize the sections as per your project's specific details!