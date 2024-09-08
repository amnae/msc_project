from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, concatenate_datasets
import pandas as pd
import random
#
seed = 42

def apply_chat_template(data, tokenizer, dataset):
    if dataset == 'openai/gsm8k':
        answer =  data['answer'].split("####")[0]
        messages = [{"role": "user", "content": data['question']}, 
                     {"role": "assistant", "content": answer}]
        data['messages'] = messages
        data["text"] = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt=False)
        data["expert_number"] = 0
    
    elif dataset == 'google-research-datasets/mbpp':
        messages = [{"role": "user", "content": data['text']}, 
                 {"role": "assistant", "content": data['code']}]
        
        data["text"] = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt=False)
        data['messages'] = messages
        data["expert_number"] = 1

    elif dataset == 'allenai/ai2_arc':
        choices = data['choices']
        if data['answerKey'] in ['1','A']: answer = choices['text'][0]
        elif data['answerKey'] in ['2','B']: answer = choices['text'][1]
        elif data['answerKey'] in ['3','C']: answer = choices['text'][2]
        elif data['answerKey'] in ['4','D']: answer = choices['text'][3]
        elif data['answerKey'] in ['5','E']: answer = choices['text'][4]

        question = data['question'] + "\nOptions:"+"".join([f"\n{i+1}. {choices['text'][i]}" for i in range(len(choices['text']))])+"\n"
        messages = [{"role": "user", "content": question}, 
            {"role": "assistant", "content": answer}]
        
        data['messages'] = messages
        data["text"] = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt=False)
        data["expert_number"] = 2

    elif dataset == 'rajpurkar/squad_v2':
        context = data['context']
        question = data['question']
        answer = data['answers']['text'][0] if not (data['answers']['text'] == []) else "No answer available."
        
        messages = [{"role": "user", "content": f"Context: {context}\nQuestion: {question}"},
            {"role": "assistant", "content": answer}]
        
        data["text"] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        data['messages'] = messages
        data["expert_number"] = 3

    elif dataset == 'jhu-clsp/jfleg':
        sentence = data['sentence']
        correction = data['correction']
        
        messages = [{"role": "user", "content": f"Original sentence: {sentence}"},
            {"role": "assistant", "content": f"Correction: {correction}"}]
        
        data["text"] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        data['messages'] = messages
        data["expert_number"] = 4
    data["labels"] = 1
    return data

def sample_random_datapoints(dataset, num_samples):
    random.seed(42)
    indices = random.sample(range(len(dataset)), num_samples)
    return dataset.select(indices)

def save_transformed_data(transformed_data, output_path):
    df = pd.DataFrame(transformed_data)
    df.to_json(output_path, orient='records', lines=True)

def transform_jfleg(data):
    transformed_data = []
    for example in data:
        sentence = example['sentence']
        for correction in example['corrections']:
            transformed_data.append({
                "sentence": sentence,
                "correction": correction.strip()
            })
    return Dataset.from_list(transformed_data)

def get_split_info(dataset_name):
    split='train'
    if dataset_name == 'allenai/ai2_arc':
        datasplit = 'ARC-Easy'
    elif dataset_name == 'jhu-clsp/jfleg': 
        datasplit = 'default'
        split = 'validation'
    elif dataset_name == 'openai/gsm8k': 
        datasplit = 'main'
    elif dataset_name == 'rajpurkar/squad_v2': 
        datasplit = 'squad_v2'
    elif dataset_name == 'google-research-datasets/mbpp':
        datasplit = 'full'
    return split,datasplit

def create_sampled_datasets(cache_dir = None):    
    base_model_file = "mistralai/Mistral-7B-Instruct-v0.3"
    #base_model_file = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    datasets = ['allenai/ai2_arc', 'jhu-clsp/jfleg', 'openai/gsm8k', 'rajpurkar/squad_v2', 'google-research-datasets/mbpp']
    
    num_samples_per_dataset = 200
    #model = AutoModelForCausalLM.from_pretrained(base_model_file, 
    #                                        quantization_config=quantization_config,
    #                                        torch_dtype=torch.bfloat16,
    #                                        cache_dir = cache_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_file,
        cache_dir = cache_dir,
        padding_side="right",
        add_eos_token=True,
        add_bos_token=True,
    )
    
    sampled_datasets=[]
    for dataset_name in datasets:
        #print(dataset_name)
        split, datasplit = get_split_info(dataset_name)
        #print(split, datasplit)
        train_dataset = load_dataset(dataset_name, datasplit, split=split, cache_dir = cache_dir)
    
        if dataset_name == 'jhu-clsp/jfleg': 
            train_dataset = transform_jfleg(train_dataset)
    
        train_dataset = train_dataset.map(apply_chat_template,
                                        fn_kwargs={"tokenizer": tokenizer, "dataset":dataset_name},
                                        desc="Applying chat template",)
        sample_set = sample_random_datapoints(train_dataset, num_samples_per_dataset)
        
        sampled_datasets.append(sample_set)
    return sampled_datasets

def main(cache_dir = None):
  print("Creating dataset")
  sampled_datasets= create_sampled_datasets(cache_dir)
  combined_dataset = concatenate_datasets(sampled_datasets)
  combined_dataset = combined_dataset.shuffle(seed=42)
  combined_dataset.save_to_disk(f"data/combined_dataset")
  print("Created dataset")

if __name__ == '__main__':
    main(cache_dir = None)