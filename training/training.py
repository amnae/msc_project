from transformers import AutoTokenizer
import torch
from datasets import load_from_disk
from datetime import datetime
from peft import LoraConfig, get_peft_model 
import argparse

from trl import SFTConfig, SFTTrainer
import os
import sys
from create_train_data import main as create_train_data

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
    print(project_root)
from modelling_edullm import EduLLMForCausalLM, MixtralConfig

def main(model_number = 0, device = 'auto', cache_dir='.cache', num_epochs = 1, batch_size = 1):
    create_train_data()
    model_types = ['mixtral', 'damex', 'xmoe']
    model_type = model_types[model_number]
    repo_name = "amnae/base_edu_llm_" + model_type
    dataset_path = "data/combined_dataset"

    print('Loading sparse base model...')
    print(cache_dir)
    model = EduLLMForCausalLM.from_pretrained("amnae/edu_llm_base",
                                            torch_dtype=torch.bfloat16,
                                            device_map = device,
                                            cache_dir = cache_dir)

    if model_number == 0:
        model.config.router_aux_loss_coef = 0.001
        model.config.model_type = 'mixtral'
        model.config.output_router_logits = True
    elif model_number == 1:
        model.config.router_aux_loss_coef = 0.02
        model.config.output_router_logits = True
        model.config.model_type = 'damex'
    elif model_number == 2:
        model.config.router_aux_loss_coef = 0.02
        model.config.model_type = 'xmoe'
        model.config.output_router_logits = True
    else: 
        print("Model number not valid.")
        sys.exit(0)
    
    print('Loaded model.')

    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

    tokenizer = AutoTokenizer.from_pretrained(
        "amnae/edu_llm_base",
        padding_side="right",
        add_eos_token=True,
        add_bos_token=True,
        cache_dir = cache_dir
    )

    tokenizer.pad_token = tokenizer.eos_token
    train_dataset = load_from_disk(dataset_path)

    cols_to_remove = train_dataset.column_names
    cols_to_remove.remove("text")
    cols_to_remove.remove("expert_number")
    train_dataset = train_dataset.remove_columns(cols_to_remove)

    encoded_dataset = train_dataset.map(lambda examples: tokenizer(examples['text']), batched=True)
    print("Dataset loaded and encoded")

    peft_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0,
        r=128,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules = "all-linear"
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    print(model.device)
    print("Model peft'd")
    print(f"Batch size: {batch_size}")
    args = SFTConfig(
        output_dir=repo_name+"_sft",
        dataset_text_field="text",    
        max_seq_length=512,
        per_device_train_batch_size = batch_size,
        logging_steps = 5,
        num_train_epochs=num_epochs,
        include_tokens_per_second = True
        )
    
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=encoded_dataset,
        tokenizer=tokenizer,
    )

    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

    model.config.gradient_checkpointing = True
    model.config.use_cache = False
    print("Start training.")
    trainer.train()

    model = model.merge_and_unload()

    print("Saving model...")
    for i in range(0,10):
        try:
            model.push_to_hub(repo_name + "_trained", cache_dir=cache_dir)
            tokenizer.push_to_hub(repo_name + "_trained")
        except Exception as error:
            print(error)
            print(f"Trying again: {i+1} / 10")
            continue
        else:
            if i == 9:
                print("10th fail. Give up.")
                sys.exit(0)
            break
    print("Model saved.")

if __name__ == '__main__':
    model_types = ['mixtral', 'damex', 'xmoe']
    dataset_path = "data/combined_dataset"
    batch_size = 1

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', 
                        '--model_num',
                        default=0,
                        choices=['all', '0', '1', '2'],
                        help="Pick which model to train: 'all' for all models, 0 for mixtral, 1 for damex, 2 for xmoe.")
    
    parser.add_argument('-c', 
                        '--cache_dir',
                        default=None,
                        help="Cache dir.")

    parser.add_argument('-d', 
                        '--device',
                        default='auto',
                        help="Choose device for model.")    
    
    parser.add_argument('-e', 
                        '--num_epochs',
                        default='1',
                        help="Choose number of epochs for training.")  

    parser.add_argument('-b', 
                        '--batch_size',
                        default='1',
                        help="Choose batch size for training.")  

    args = parser.parse_args()

    if args.model_num:
        if args.model_num == 'all':
            main(0, device = args.device, cache_dir = args.cache_dir, batch_size = int(args.batch_size), num_epochs = int(args.num_epochs))
            main(1, device = args.device, cache_dir = args.cache_dir, batch_size = int(args.batch_size), num_epochs = int(args.num_epochs))
            main(2, device = args.device, cache_dir = args.cache_dir, batch_size = int(args.batch_size), num_epochs = int(args.num_epochs))
        else:
            main(int(args.model_num), device = args.device, cache_dir = args.cache_dir, batch_size = int(args.batch_size), num_epochs = int(args.num_epochs))
    else:
        main(0, device = args.device, cache_dir = args.cache_dir, batch_size = int(args.batch_size), num_epochs = int(args.num_epochs))