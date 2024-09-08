from opencompass.models import EduLLMwithChatTemplate
from mmengine.config import read_base
import torch
#from huggingface_hub import login
import faulthandler; faulthandler.enable()

import os
from dotenv import load_dotenv

load_dotenv()

model_path = 'amnae/base_edu_llm_mixtral_trained'

model_kwargs = {
    "torch_dtype": torch.float16,
    "device_map": "cpu"
}


models = [
    dict(
        type=EduLLMwithChatTemplate,
        # Parameters for `HuggingFaceCausalLM` initialization.
        path=model_path,
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left'),
        max_seq_len=512,
        batch_padding=False,
        # Common parameters shared by various models, not specific to `HuggingFaceCausalLM` initialization.
        abbr='base-model',            # Model abbreviation used for result display.
        max_out_len=215,            # Maximum number of generated tokens.
        batch_size=1,              # The size of a batch during inference.
        run_cfg=dict(num_gpus=1),   # Run configuration to specify resource requirements.
        model_kwargs = model_kwargs
    )
]

with read_base():
    from .datasets.ARC_e.ARC_e_gen import ARC_e_datasets
    from .datasets.squad20.squad20_gen import squad20_datasets
    from .datasets.mbpp.mbpp_gen import mbpp_datasets
    from .datasets.gsm8k.gsm8k_gen import gsm8k_datasets 

datasets = [ARC_e_datasets, squad20_datasets, mbpp_datasets, gsm8k_datasets]