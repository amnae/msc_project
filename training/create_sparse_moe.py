import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
    print(project_root)
    
from modelling_edullm import EduLLMForCausalLM, MixtralConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import re
import logging
import sys
import argparse
import torch
import numpy as np

from huggingface_hub import login
login("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")

def main(device = 'cpu', cache_dir=None):
    print('Starting.')
    print(device)
    print(cache_dir)
    print('Loading dense model...')
    dense_model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(dense_model_name, 
                                              cache_dir = cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    dense_model = AutoModelForCausalLM.from_pretrained(dense_model_name,
                                                cache_dir = cache_dir,
                                                torch_dtype = torch.bfloat16,
                                                device_map  = device) 
    print('Loaded dense model.')
    dense_weights = dense_model.state_dict()

    print('Loading sparse base model...')
    num_experts = 5
    repo_name = "amnae/edu_llm_base"
    preconfig = MixtralConfig(num_local_experts=num_experts, device_map  = device, model_type = 'mixtral')
    sparse_model = EduLLMForCausalLM._from_config(preconfig).to(device).half()
    print('Loaded sparse base model.')

    sparse_weights = sparse_model.state_dict()
    new_sparse_weights = {}
    pattern = re.compile(r"block_sparse_moe\.experts\.\d", re.IGNORECASE)

    noise_std_dev = 0.0001
    for key in sparse_weights:
        print(key)
        if key in dense_weights:
           #print(torch.std_mean(dense_weights[key]))
           new_sparse_weights[key] = dense_weights[key] + torch.normal(0, noise_std_dev, dense_weights[key].shape)
           if sparse_weights[key].shape != new_sparse_weights[key].shape:
                print(f'{key} done. Shape match: {sparse_weights[key].shape == new_sparse_weights[key].shape}. New Shape: {new_sparse_weights[key].shape}. Old Shape:{sparse_weights[key].shape}.')
        elif 'gate' in key:
            # Transfer randomised numbers
            new_sparse_weights[key] = sparse_weights[key] + torch.normal(0, noise_std_dev, sparse_weights[key].shape)
            if sparse_weights[key].shape != new_sparse_weights[key].shape:
                print(f'{key} done. Shape match: {sparse_weights[key].shape == new_sparse_weights[key].shape}. New Shape: {new_sparse_weights[key].shape}. Old Shape:{sparse_weights[key].shape}.')
        elif 'block_sparse_moe' in key:
            # Transfer with with modified key
            for i in range(num_experts):
                new_key = re.sub(pattern,'mlp',key)
                if 'w1' in new_key:
                    new_key = new_key.replace('w1', 'gate_proj')
                elif 'w2' in new_key:
                    new_key = new_key.replace('w2', 'down_proj')
                elif 'w3' in new_key:
                    new_key = new_key.replace('w3', 'up_proj')
                else:
                    print(f'Strange key: {new_key}')                
                if new_key in dense_weights:
                    #print(torch.std_mean(dense_weights[new_key]))
                    new_sparse_weights[key] = dense_weights[new_key] + torch.normal(0, noise_std_dev, dense_weights[new_key].shape)
                    if sparse_weights[key].shape != new_sparse_weights[key].shape:
                        print(f'{key} done. Shape match: {sparse_weights[key].shape == new_sparse_weights[key].shape}. New Shape: {new_sparse_weights[key].shape}. Old Shape:{sparse_weights[key].shape}.')
                else:
                    print(f'Key not handled: {key}')
        else:
            print(f'Key not handled: {key}')
    print("Loading to model...")
    print(new_sparse_weights)
    sparse_model.load_state_dict(new_sparse_weights, strict=True, )


    print("Saving model...")
    for i in range(0,10):
        try:
            sparse_model.push_to_hub(repo_name, cache_dir=cache_dir)
            tokenizer.push_to_hub(repo_name)
        except Exception as error:
            print(error)
            print(f'Trying again: {i+1}/10')
            continue
        else:
            if i == 9:
                print("10th fail. Give up.")
                sys.exit(0)
            break
    print("Model saved.")

    messages = [
        {"role": "user", "content": "What do these two things have in common? a) bleaching clothes and b) an apple turning brown?\nLet's think step by step\nAnswer:\n."}
    ]
    model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
    print("Tokenized.")
    print("Generating response.")
    generated_ids = sparse_model.generate(model_inputs, max_new_tokens=200, do_sample=True)
    generated_ids = generated_ids.to('cuda') 
    tokens = tokenizer.batch_decode(generated_ids)[0]
    print(tokens)

    messages = [
        {"role": "user", "content": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\nLet's think step by step\nAnswer:\n."}
    ]
    model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
    print("Tokenized.")
    print("Generating response.")
    generated_ids = sparse_model.generate(model_inputs, max_new_tokens=200, do_sample=True)
    generated_ids = generated_ids.to('cuda') 
    tokens = tokenizer.batch_decode(generated_ids)[0]
    print(tokens)


if __name__ == '__main__':
    #python training/create_sparse_moe.py -c '/cs/student/projects1/dsml/2023/elbadawi/project/.cache'
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', 
                        '--cache_dir',
                        default=None,
                        help="Cache dir.")

    parser.add_argument('-d', 
                        '--device',
                        default='cpu',
                        help="Choose device for model.")    

    args = parser.parse_args()

    main(device = args.device, cache_dir = args.cache_dir)