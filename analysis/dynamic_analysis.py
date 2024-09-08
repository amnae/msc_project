# Analysis of Dynamic Behaviours
#This notebook includes experiments listed below:
#- Outputs of experts
#- Norms of expert outputs and gate scores
#- Intermediate states of experts
#- Chosen experts

#The models have their own code blocks for each experiment. The overall logic of the code belonging to different models is alike, and the minor differences stem from the unique settings of the corresponding model.

#Usually, the figures are plotted in two ways: 'auto_colorbar' and 'full_colorbar'. The former allows the matplotlib methods to automatically dicide the range of the color bar for each layer. For the latter, we manually set it to be the global minimum/maximum for all the layers.
import csv
import math
import ml_dtypes
import os
import pickle
import argparse

import functools
import matplotlib as mlp
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.functional import normalize
from transformers import AutoTokenizer
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
    print(project_root)
    
from modelling_edullm import EduLLMForCausalLM, MixtralDecoderLayer, MixtralBlockSparseTop2MLP

# The root directory for saving the output figures and data.
WORK_DIR = './outputs'

def main(model_number = 0 , device_map = 'auto', cache_dir = None):
    models = ['mixtral', 'damex', 'xmoe']
    model_name = models[model_number]
    model_name = "amnae/base_edu_llm_" + model_name + "_trained"
    #Run one or more cells below to load the models you need.
    model = EduLLMForCausalLM.from_pretrained(
        model_name, 
        low_cpu_mem_usage=True, 
        device_map="auto", 
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir
    )
    tokeniser = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model.eval()
    ## Outputs of Experts
    #We use both the short and long sequence in this experiment. We plot the similarity heat map of each token in the short sequence, while only the averaged heat map is plotted for the long sequence. To employ the long sequence as the input, set `use_short_input=False`.
    # Input.
    use_short_input = True # Set False to use the long sequence.
    sentence_lst = []
    if use_short_input:
        raw_input = "As an open source alternative to"
        sentence_lst.append(raw_input)
    else:
        with open('./wikitext103_test.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\n')
            for row in csv_reader:
                sentences = row[0].split('\n')
                for sent in sentences:
                    sent = sent.strip()
                    if sent.startswith('=') or sent == '':
                        continue
                    sentence_lst.append(sent)

    cos = torch.nn.CosineSimilarity(dim=0)
    matrices = [('w3', 'up_proj'), ('w1', 'gate_proj'), ('w2', 'down_proj')]
    num_layers = model.config.num_hidden_layers
    num_experts = model.config.num_local_experts

    tick_labels = [str(i) for i in range(num_experts)]
    tick_labels.append('F')
    save_dir = os.path.join(WORK_DIR, f'edullm/{model_name}_experts_outsim')
    if not use_short_input:
        save_dir += '_average'
    plot_dir = os.path.join(save_dir, 'figure')
    output_dir = os.path.join(save_dir, 'data')
    os.makedirs(os.path.join(plot_dir, 'auto_colorbar'), exist_ok=True)
    os.makedirs(os.path.join(plot_dir, 'full_colorbar'), exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)


    def plot_one_layer_short_seq(arr_lst, all_gate_indices, layer_idx, num_tokens, range_type, global_vmin=None, global_vmax=None):
        imlst = []
        fig, axs = plt.subplots(ncols=num_tokens, layout='constrained', figsize=(8.0, 2.5))
        for i, sim_arr in enumerate(arr_lst):
            if range_type == 'auto_colorbar':
                print(i)
                print(sim_arr.shape)
                im = axs[i].imshow(sim_arr)
                imlst.append(im)
            elif range_type == 'full_colorbar':
                im = axs[i].imshow(sim_arr, vmin=global_vmin, vmax=global_vmax)
            exp1, exp2 = all_gate_indices[layer_idx][0][i, 0], all_gate_indices[layer_idx][0][i, 1]
            axs[i].set_title(f'exp {exp1},{exp2}', fontsize=15)
            axs[i].set_xticks(np.arange(num_experts+1), labels=tick_labels, fontsize=15)
            axs[i].set_yticks(np.arange(num_experts+1), labels=tick_labels, fontsize=15)
            if i == 0:
                axs[i].set_ylabel(f'Layer {layer_idx}', labelpad=12., fontsize=20)
        if range_type == 'auto_colorbar':
            local_vmin = min(img.get_array().min() for img in imlst)
            local_vmax = max(img.get_array().max() for img in imlst)
            norm = colors.Normalize(vmin=local_vmin, vmax=local_vmax)
            for img in imlst:
                img.set_norm(norm)
        cbar = fig.colorbar(im, ax=axs, shrink=0.88)
        cbar.ax.tick_params(labelsize=15)
        plt.savefig(os.path.join(plot_dir, range_type, f'layer_{layer_idx}.png'))
        plt.close()


    def plot_one_layer_long_seq(avg_arr, layer_idx, range_type, global_vmin=None, global_vmax=None):
        imlst = []
        fig, ax = plt.subplots(ncols=1, layout='constrained', figsize=(3.5, 2.0))
        if range_type == 'auto_colorbar':
            im = ax.imshow(avg_arr)
            imlst.append(im)
        elif range_type == 'full_colorbar':
            im = ax.imshow(avg_arr, vmin=global_vmin, vmax=global_vmax)
        ax.set_xticks(np.arange(num_experts+1), labels=tick_labels, fontsize=14.5)
        ax.set_yticks(np.arange(num_experts+1), labels=tick_labels, fontsize=14.5)
        ax.set_ylabel(f'Layer {layer_idx}', labelpad=12., fontsize=20)
        cbar = fig.colorbar(im, ax=ax, shrink=1.)
        cbar.ax.tick_params(labelsize=14.5)
        plt.savefig(os.path.join(plot_dir, range_type, f'layer_{layer_idx}.png'))
        plt.close()

    # Forward pass.

    def get_angular_similarity(v1, v2):
        batch_cos = torch.nn.CosineSimilarity(dim=2)
        return 1 - (torch.acos(batch_cos(v1, v2)) / math.pi)


    def record_layer_output(module, input, output, layer_idx):
        all_layer_output[layer_idx].append(output[0])


    def record_gate_output(module, input, output, layer_idx):  
        scores = output
        _, expert_indices = torch.topk(scores, 2, dim=-1)
        all_gate_indices[layer_idx].append(expert_indices.cpu().detach().numpy())


    def record_expert_output(module, input, output, layer_idx, expert_idx):
        # output shape = [num_tokens, hidden_dim]
        print(f"Recording: {layer_idx} {expert_idx}")
        all_expert_output[layer_idx][expert_idx] = output 
        print(output.shape)


    total_token_count = 0
    #all_sim_arr = [np.zeros((num_experts+1, num_experts+1)) for _ in range(num_layers)]
    all_sim_arr = [[] for _ in range(num_layers)]
    for s, sent in enumerate(sentence_lst):
        if s == 10:
            # For the long sequence, use the first ten sentences only.
            break
        mix_enc_input = tokeniser.encode(sent, return_tensors='pt') # mix_enc_input is actually the same as mis_enc_input.
        mis_enc_input = tokeniser.encode(sent, return_tensors='pt')
        assert mix_enc_input.shape[1] == mis_enc_input.shape[1]
        num_tokens = mix_enc_input.shape[1]
        total_token_count += num_tokens
        print(s, num_tokens, total_token_count)
        all_layer_output = [[] for _ in range(num_layers)]
        all_expert_output = [{} for _ in range(num_layers)]
        all_gate_indices = [[] for _ in range(num_layers)]
        handles = []

        # Obtain the original output feature vectors of experts 
        # and gate choices when topk=2. 
        for name, module in model.named_modules():
            if isinstance(module, MixtralDecoderLayer):
                layer_idx = int(name.split('.')[2])
                handles.append(module.register_forward_hook(
                    functools.partial(record_layer_output, layer_idx=layer_idx)
                ))
            elif isinstance(module, torch.nn.Linear) and 'gate' in name:
                layer_idx = int(name.split('.')[2])
                handles.append(module.register_forward_hook(
                    functools.partial(record_gate_output, layer_idx=layer_idx)
                ))

        mix_output = model(mix_enc_input.to(device_map))
        for h in handles:
            h.remove()
        handles = []

        # Modify the number of chosen experts to ALL.
        for i in range(num_layers):
            model.model.layers[i].block_sparse_moe.top_k = num_experts
        # Iterate over the layers and register a hook once a time.
        for i in range(num_layers):
            for name, module in model.named_modules():
                if isinstance(module, MixtralBlockSparseTop2MLP):
                    layer_idx = int(name.split('.')[2])
                    if layer_idx == i:
                        expert_idx = int(name.split('.')[-1])
                        handles.append(module.register_forward_hook(
                            functools.partial(record_expert_output, layer_idx=layer_idx, expert_idx=expert_idx)
                        ))
                    elif layer_idx > i:
                        break
            if i == 0:
                with torch.no_grad():
                    mix_output = model(mix_enc_input.to(device_map), decoder_layer_idx=i, use_cache=False) # Set use_cache=False to prevent error.
            else: 
                with torch.no_grad():
                    # Feed the topk=2 output of previous layer as input.
                    mix_output = model(inputs_embeds=all_layer_output[i-1][0], decoder_layer_idx=i, use_cache=False) 
            for h in handles:
                h.remove()
            handles = []
        # Revert to the original value.
        for i in range(num_layers):
            model.model.layers[i].block_sparse_moe.num_experts_per_tok = model.config.num_experts_per_tok

        # Compute similarity between outputs of current sentence.
        if use_short_input:
            for i in range(num_layers):
                print(f"Layer {i}")
                for j in range(num_tokens):
                    #print(f"Token {j}")
                    sim_arr = np.ones((num_experts, num_experts))
                    for k in range(num_experts):
                        #print(f"Expert {k}")
                        for l in range(k, num_experts):
                            #print(f"vs Expert {l}")
                            # Mixtral and Mistral layers can be loaded on differnet GPUs, so put them on the same device manually. 
                            #print(all_expert_output[i][k].shape)
                            #print(all_expert_output[i][l].shape)
                            #print(np.array(all_expert_output).shape)
                            sim = cos(all_expert_output[i][k][j].to('cuda:0'), all_expert_output[i][l][j].to('cuda:0')).float().cpu().detach().numpy().astype(ml_dtypes.bfloat16) 
                            sim_arr[k][l] = sim
                            sim_arr[l][k] = sim
                    print(all_sim_arr[i])
                    all_sim_arr[i].append(sim_arr)
                    #all_sim_arr[i] = np.concatenate(all_sim_arr[i], sim_arr)
                    print(all_sim_arr[i])
        else:
            output_dim = all_expert_output[0][0][0].shape[0]
            for i in range(num_layers):
                for j in range(num_tokens):
                    # Reorganize recorded data to compute similarity in parallel.
                    expert_output_self = torch.empty(num_experts+1, 1, output_dim).cuda()
                    expert_output_other = torch.empty(1, num_experts+1, output_dim).cuda()
                    for k in range(num_experts+1):
                        k = -1 if k == num_experts else k
                        expert_output_self[k, 0] = all_expert_output[i][k][j]
                        expert_output_other[0, k] = all_expert_output[i][k][j]
                    sim = get_angular_similarity(expert_output_self, expert_output_other).fill_diagonal_(1.) # Replace nan values due to numerical instability
                    sim = sim.float().cpu().detach().numpy().astype(ml_dtypes.bfloat16) 
                    all_sim_arr[i] += sim
    # Save and plot.
    if use_short_input:
        # Record the maximum and minimum values for plotting.
        global_vmax, global_vmin = -1 * math.inf, math.inf
        for i in range(num_layers):
            for j in range(num_tokens):
                sim_arr = all_sim_arr[i][j]
                curr_vmax = np.max(sim_arr)
                curr_vmin = np.min(sim_arr)
                if curr_vmin < global_vmin:
                    global_vmin = curr_vmin
                if curr_vmax > global_vmax:
                    global_vmax = curr_vmax
        
        output_dict = {'global_vmax':global_vmax, 'global_vmin':global_vmin}
        with open(os.path.join(output_dir, 'all_sim_arr'), 'wb') as f:
            pickle.dump(all_sim_arr, f)
        with open(os.path.join(output_dir, 'all_gate_indices'), 'wb') as f:
            pickle.dump(all_gate_indices, f)
        with open(os.path.join(output_dir, 'output_dict'), 'wb') as f:
            pickle.dump(output_dict, f)
        for i in range(num_layers):
            plot_one_layer_short_seq(all_sim_arr[i], all_gate_indices, i, num_tokens, 'auto_colorbar')
            plot_one_layer_short_seq(all_sim_arr[i], all_gate_indices, i, num_tokens, 'full_colorbar', global_vmin, global_vmax)

    else:
        global_vmax, global_vmin = -1 * math.inf, math.inf
        all_avg_arr = []
        for i in range(num_layers):
            avg_arr = all_sim_arr[i] / total_token_count
            all_avg_arr.append(avg_arr)
            curr_vmax = np.max(avg_arr)
            curr_vmin = np.min(avg_arr)
            if curr_vmin < global_vmin:
                global_vmin = curr_vmin
            if curr_vmax > global_vmax:
                global_vmax = curr_vmax
        output_dict = {'global_vmax':global_vmax, 'global_vmin':global_vmin}
        with open(os.path.join(output_dir, 'all_avg_arr'), 'wb') as f:
            pickle.dump(all_avg_arr, f)
        with open(os.path.join(output_dir, 'output_dict'), 'wb') as f:
            pickle.dump(output_dict, f)
        
        for i in range(num_layers):
            avg_arr = all_sim_arr[i] / total_token_count
            plot_one_layer_long_seq(all_avg_arr[i], i, 'auto_colorbar')
            plot_one_layer_long_seq(all_avg_arr[i], i, 'full_colorbar', global_vmin, global_vmax)

    ## Norms of Expert Outputs and Gate Scores
    #We use both the short and long sequence in this experiment. We plot the norm and gate score of every expert for each token in the short sequence, while only the rank counting is plotted for the long sequence. To employ the long sequence as the input, set `use_short_input=False`.

    # Input.
    use_short_input = True # Set False to use the long sequence.
    sentence_lst = []
    if use_short_input:
        raw_input = "As an open source alternative to"
        sentence_lst.append(raw_input)
    else:
        with open('./wikitext103_test.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter='\n')
            for row in csv_reader:
                sentences = row[0].split('\n')
                for sent in sentences:
                    sent = sent.strip()
                    if sent.startswith('=') or sent == '':
                        continue
                    sentence_lst.append(sent)

    num_layers = model.config.num_hidden_layers
    num_experts = model.config.num_local_experts

    tick_labels = [str(i) for i in range(num_experts)]
    save_dir = os.path.join(WORK_DIR, f'edullm/{model_name}_expert_norm')
    if not use_short_input:
        save_dir += '_count'
    plot_dir = os.path.join(save_dir, 'figure')
    output_dir = os.path.join(save_dir, 'data')
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)


    def plot_one_layer_short_seq(all_gate_scores, all_gate_indices, all_expert_output, layer_idx, num_tokens):
        fig, axs = plt.subplots(ncols=num_tokens, layout='constrained', figsize=(22.0, 2.8))
        for i in range(num_tokens):
            # Plot the norm of feature vectors output by experts.
            norm_lst = []
            for j in range(num_experts):
                norm_lst.append(all_expert_output[layer_idx][j][i])
            im1 = axs[i].bar(np.arange(num_experts)*2-0.35, norm_lst, label='Norm', width=0.6)
            # Plot the gate scores.
            twin_ax = axs[i].twinx()
            im2 = twin_ax.bar(np.arange(num_experts)*2, all_gate_scores[layer_idx][0][i, :], tick_label=tick_labels, 
                            color='darkorange', align='edge', label='Score', width=0.5)
            axs[i].set_xticks(np.arange(num_experts)*2, labels=tick_labels, fontsize=18)
            if i == 0:
                axs[i].set_ylabel(f'Layer {layer_idx}', labelpad=14., fontsize=22)
            exp1, exp2 = all_gate_indices[layer_idx][0][i, 0], all_gate_indices[layer_idx][0][i, 1]
            axs[i].set_title(f'exp {exp1},{exp2}', fontsize=18)
            axs[i].legend(loc='upper left', fontsize=12)
            twin_ax.legend(loc='upper right', fontsize=12)
        plt.savefig(os.path.join(plot_dir, f'layer_{layer_idx}.png'))
        plt.close()


    def plot_one_layer_long_seq(rankings_counts, layer_idx):
        fig, ax = plt.subplots(layout='constrained', figsize=(6.5, 4.0))
        bar_width = 0.1
        x = np.arange(num_experts)
        for i in range(num_experts):
            offset = bar_width * i
            im = ax.bar(x+offset, rankings_counts[i, :], bar_width, label=f'score rank {i+1}')
        ax.set_xticks(x+3.5*bar_width, [str(i+1) for i in range(num_experts)], fontsize=13)
        ax.tick_params(axis='y', labelsize=11)
        ax.legend(loc='best', fontsize=11)
        ax.set_xlabel('Expert output norm ranking', fontsize=15)
        ax.set_ylabel('Count of gate score ranking', fontsize=15)
        plt.savefig(os.path.join(plot_dir, f'layer{layer_idx}.png'))
        plt.close()
    # Forward pass.

    def record_layer_output(module, input, output, layer_idx):
        all_layer_output[layer_idx].append(output[0])


    def record_gate_output(module, input, output, layer_idx):  
        scores = output
        _, expert_indices = torch.topk(scores, 2, dim=-1, sorted=True)
        all_gate_indices[layer_idx].append(expert_indices.cpu().detach().numpy())
        all_gate_scores[layer_idx].append(scores.softmax(dim=-1).float().cpu().detach().numpy().astype(ml_dtypes.bfloat16))


    def record_expert_output(module, input, output, layer_idx, expert_idx):
        # output size = [num_tokens, hidden_dim]
        all_expert_output[layer_idx][expert_idx] = torch.norm(output, dim=1).float().cpu().detach().numpy().astype(ml_dtypes.bfloat16)


    token_count = 0
    rankings_counts = [np.zeros((num_experts, num_experts)) for _ in range(num_layers)]
    for s, sent in enumerate(sentence_lst):
        if s == 10:
            break
        enc_input = tokeniser.encode(sent, return_tensors="pt").cuda()
        num_tokens = enc_input.shape[1]
        token_count += num_tokens
        print(s, num_tokens, token_count)
        all_gate_scores = [[] for _ in range(num_layers)]
        all_gate_indices = [[] for _ in range(num_layers)]
        all_layer_output = [[] for _ in range(num_layers)]
        all_expert_output = [{} for _ in range(num_layers)]
        handles = []

        # Obtain the original output feature vectors of experts 
        # and gate choices when topk=2. 
        for name, module in model.named_modules():
            if isinstance(module, MixtralDecoderLayer):
                layer_idx = int(name.split('.')[2])
                handles.append(module.register_forward_hook(
                    functools.partial(record_layer_output, layer_idx=layer_idx)
                ))
            elif isinstance(module, torch.nn.Linear) and 'gate' in name:
                layer_idx = int(name.split('.')[2])
                handles.append(module.register_forward_hook(
                    functools.partial(record_gate_output, layer_idx=layer_idx)
                ))

        with torch.no_grad():
            mix_output = model(enc_input.to(device_map))
        for h in handles:
            h.remove()
        handles = []

        # Modify the number of chosen experts to ALL.
        for i in range(num_layers):
            model.model.layers[i].block_sparse_moe.num_experts_per_tok = num_experts
        # Iterate over the layers and register a hook once a time.
        for i in range(num_layers):
            for name, module in model.named_modules():
                if isinstance(module, MixtralBlockSparseTop2MLP):
                    layer_idx = int(name.split('.')[2])
                    if layer_idx == i:
                        expert_idx = int(name.split('.')[-1])
                        print(layer_idx)
                        print(expert_idx)
                        handles.append(module.register_forward_hook(
                            functools.partial(record_expert_output, layer_idx=layer_idx, expert_idx=expert_idx)
                        ))
                    elif layer_idx > i:
                        break
            if i == 0:
                with torch.no_grad():
                    mix_output = model(enc_input.to(device_map), decoder_layer_idx=i, use_cache=False) # Set use_cache=False to prevent error.
            else: 
                with torch.no_grad():
                # Feed the topk=2 output of previous layer as input.
                    mix_output = model(inputs_embeds=all_layer_output[i-1][0], decoder_layer_idx=i, use_cache=False) 
            for h in handles:
                h.remove()
            handles = []
        # Revert to the original value.
        for i in range(num_layers):
            model.model.layers[i].block_sparse_moe.num_experts_per_tok = model.config.num_experts_per_tok

        if not use_short_input:
            # Count the norm-score ranking pairs.
            for i in range(num_layers):
                for j in range(num_tokens):
                    curr_token_output = np.array([])
                    for k in range(num_experts):
                        curr_token_output = np.append(curr_token_output, all_expert_output[i][k][j])
                    curr_gate_score = all_gate_scores[i][0][j, :]
                    norm_rank = np.argsort(curr_token_output)
                    score_rank = np.argsort(curr_gate_score)
                    # Replace the values with the corresponding rankings.
                    for rank, idx in enumerate(norm_rank):
                        curr_token_output[idx] = rank
                    for rank, idx in enumerate(score_rank):
                        curr_gate_score[idx] = rank
                    for row, col in zip(curr_gate_score.tolist(), curr_token_output.tolist()):
                        rankings_counts[i][int(row), int(col)] += 1
    # Save and plot.
    if use_short_input:
        with open(os.path.join(output_dir, 'all_gate_scores'), 'wb') as f:
            pickle.dump(all_gate_scores, f)
        with open(os.path.join(output_dir, 'all_gate_indices'), 'wb') as f:
            pickle.dump(all_gate_indices, f)
        with open(os.path.join(output_dir, 'all_expert_output'), 'wb') as f:
            pickle.dump(all_expert_output, f)

        for i in range(num_layers):
            plot_one_layer_short_seq(all_gate_scores, all_gate_indices, all_expert_output, i, num_tokens)

    else:
        with open(os.path.join(output_dir, 'rankings_counts'), 'wb') as f:
            pickle.dump(rankings_counts, f)
        # Plot layer one by one.
        for l in range(num_layers):
            plot_one_layer_long_seq(rankings_counts[l], l)
        # Plot all layers.
        total_rankings_counts = rankings_counts[0]
        for l in range(1, num_layers):
            total_rankings_counts += rankings_counts[l]
        plot_one_layer_long_seq(total_rankings_counts, 'ALL')
    ## Intermediate States of Experts
    #Only the short sequence is used in this section.
    # Input.
    raw_input = "As an open source alternative to"
    mix_enc_input = tokeniser.encode(raw_input, return_tensors='pt') # mix_enc_input is actually the same as mis_enc_input.
    mis_enc_input = tokeniser.encode(raw_input, return_tensors='pt')

    num_layers = model.config.num_hidden_layers
    num_experts = model.config.num_local_experts
    intermediate_size = model.config.intermediate_size
    num_tokens = mix_enc_input.shape[1]
    all_layer_output = [[] for _ in range(num_layers)]
    all_expert_act = [{} for _ in range(num_layers)]
    all_gate_indices = [[] for _ in range(num_layers)]
    handles = []


    def record_layer_output(module, input, output, layer_idx):
        all_layer_output[layer_idx].append(output[0])


    def record_gate_output(module, input, output, layer_idx):  
        scores = output
        _, expert_indices = torch.topk(scores, 2, dim=-1, sorted=True)
        all_gate_indices[layer_idx].append(expert_indices.float().cpu().detach().numpy().astype(ml_dtypes.bfloat16))


    def record_expert_act(module, input, output, layer_idx, expert_idx):
        # act_neurons size = [num_tokens, intermediate_size]
        act_neurons = F.silu(module.w1(input[0]))
        all_expert_act[layer_idx][expert_idx] = act_neurons.float().cpu().detach().numpy().astype(ml_dtypes.bfloat16)

    # Obtain the original output feature vectors of experts 
    # and gate choices when topk=2. 
    for name, module in model.named_modules():
        if isinstance(module, MixtralDecoderLayer):
            layer_idx = int(name.split('.')[2])
            handles.append(module.register_forward_hook(
                functools.partial(record_layer_output, layer_idx=layer_idx)
            ))
        elif isinstance(module, torch.nn.Linear) and 'gate' in name:
            layer_idx = int(name.split('.')[2])
            handles.append(module.register_forward_hook(
                functools.partial(record_gate_output, layer_idx=layer_idx)
            ))

    with torch.no_grad():
        mix_output = model(mix_enc_input.to(device_map))
    for h in handles:
        h.remove()
    handles = []

    # Modify the number of chosen experts to ALL.
    for i in range(num_layers):
        model.model.layers[i].block_sparse_moe.num_experts_per_tok = num_experts
    # Iterate over the layers and register a hook once a time.
    for i in range(num_layers):
        for name, module in model.named_modules():
            if isinstance(module, MixtralBlockSparseTop2MLP):
                layer_idx = int(name.split('.')[2])
                if layer_idx == i:
                    expert_idx = int(name.split('.')[-1])
                    handles.append(module.register_forward_hook(
                        functools.partial(record_expert_act, layer_idx=layer_idx, expert_idx=expert_idx)
                    ))
                elif layer_idx > i:
                    break
        if i == 0:
            with torch.no_grad():
                mix_output = model(mix_enc_input.to(device_map), decoder_layer_idx=i, use_cache=False) # Set use_cache=False to prevent error.
        else: 
            with torch.no_grad():
                # Feed the topk=2 output of previous layer as input.
                mix_output =  model(inputs_embeds=all_layer_output[i-1][0], decoder_layer_idx=i, use_cache=False) 
        for h in handles:
            h.remove()
        handles = []

    # Revert to the original value.
    for i in range(num_layers):
        model.model.layers[i].block_sparse_moe.num_experts_per_tok = model.config.num_experts_per_tok
        
    global_vmin = math.inf
    for i in range(num_layers):
        for act in all_expert_act[i].values():
            curr_vmin = np.min(act)
            if curr_vmin < global_vmin:
                global_vmin = curr_vmin
    # Save and plot.
    xtick_labels = [str(i) for i in range(0, intermediate_size, 4000)]
    ytick_labels = [str(i) for i in range(num_experts)]
    ytick_labels.append('F')
    save_dir = os.path.join(WORK_DIR, f'edullm/{model_name}_experts_inter') 
    plot_dir = os.path.join(save_dir, 'figure')
    output_dir = os.path.join(save_dir, 'data')
    os.makedirs(os.path.join(plot_dir, 'auto_colorbar'), exist_ok=True)
    os.makedirs(os.path.join(plot_dir, 'full_colorbar'), exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    output_dict = {'global_vmin':global_vmin}
    with open(os.path.join(output_dir, 'all_expert_act'), 'wb') as f:
        pickle.dump(all_expert_act, f)
    with open(os.path.join(output_dir, 'all_gate_indices'), 'wb') as f:
        pickle.dump(all_gate_indices, f)
    with open(os.path.join(output_dir, 'output_dict'), 'wb') as f:
        pickle.dump(output_dict, f)


    def plot_one_layer(all_expert_act, all_gate_indices, layer_idx, range_type, global_vmin=None):
        fig, axs = plt.subplots(nrows=num_tokens, layout='constrained', figsize=(16.0, 7.0))
        imlst = []
        for i in range(num_tokens):
            curr_map = np.empty((num_experts+1, intermediate_size))
            for j in range(num_experts):
                curr_map[j] = all_expert_act[layer_idx][j][i, :]
            print(len(all_expert_act[layer_idx]))
            print(all_expert_act[layer_idx].keys())
            print(curr_map[-1])
            #curr_map[-1] = all_expert_act[layer_idx][num_experts-1][0, i, :]
            if range_type == 'auto_colorbar':
                im = axs[i].imshow(curr_map, aspect='auto')
                imlst.append(im)
            elif range_type == 'full_colorbar':
                im = axs[i].imshow(curr_map, aspect='auto', vmin=global_vmin, vmax=1.0)
            axs[i].set_xticks(np.arange(0, intermediate_size, 4000), labels=xtick_labels, fontsize=13)
            axs[i].set_yticks(np.arange(num_experts+1), labels=ytick_labels, fontsize=13)
            axs[i].set_yticks(np.arange(-.5, num_experts+1, 1), minor=True)
            axs[i].tick_params(axis='y', which='minor', length=0)
            axs[i].grid(axis='y', which='minor', color='k', linestyle='-', linewidth=.2)
            exp1, exp2 = all_gate_indices[layer_idx][0][i, 0], all_gate_indices[layer_idx][0][i, 1]
            axs[i].set_title(f'expert {exp1},{exp2}', fontsize=16)
        if range_type == 'auto_colorbar':
            local_vmin = min(img.get_array().min() for img in imlst)
            local_vmax = max(img.get_array().max() for img in imlst)
            norm = colors.Normalize(vmin=local_vmin, vmax=local_vmax)
            for img in imlst:
                img.set_norm(norm)
        fig.suptitle(f'Layer {layer_idx}', fontsize=22)
        cbar = fig.colorbar(im, ax=axs, shrink=1.)
        cbar.ax.tick_params(labelsize=15)
        plt.savefig(os.path.join(plot_dir, range_type, f'layer_{layer_idx}.png'))
        plt.close()


    for i in range(num_layers):
        print(i)
        #plot_one_layer(all_expert_act, all_gate_indices, i, 'auto_colorbar')
        #plot_one_layer(all_expert_act, all_gate_indices, i, 'full_colorbar', global_vmin)

    ## Chosen Experts
    #In this experiment, we utilize another input containing about 64 tokens. In addition to the base model of Mixtral (Mixtral-Base), we include its instruct version (Mixtral-Instruct).
    ### Mixtral-Instruct
    raw_input = "As an open source alternative to Chat GPT, I do not have personal opinions. However, I can provide objective information about Chat GPT's capabilities and limitations based on its architecture and training data. Chat GPT is a powerful language model based on the GPT (Generative Pre-trained Transformer"
    enc_input = tokeniser.encode(raw_input, return_tensors="pt")

    num_layers = model.config.num_hidden_layers
    num_experts = model.config.num_local_experts
    num_tokens = enc_input.shape[1]
    xtick_labels = [tokeniser.decode(t) for t in enc_input[0]]
    ytick_labels = [str(i) for i in range(num_experts)]
    gate_outputs = [[] for _ in range(num_layers)]
    handles = []
    save_dir = os.path.join(WORK_DIR, f'edullm/{model_name}_instuct_gate_choice')
    plot_dir = os.path.join(save_dir, 'figure')
    output_dir = os.path.join(save_dir, 'data')
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)


    def plot_one_layer(all_expert_weights, layer_idx):
        fig, ax = plt.subplots(layout='constrained', figsize=(14.0, 3.))
        im = ax.imshow(all_expert_weights, cmap=mlp.colormaps['Blues'])
        ax.set_xticks(np.arange(num_tokens), labels=xtick_labels, rotation='vertical', fontsize=13)
        ax.set_yticks(np.arange(num_experts), labels=ytick_labels, fontsize=13)
        ax.set_ylabel(f'Layer {layer_idx}', labelpad=14., fontsize=20)
        l, b, w, h = ax.get_position().bounds
        ax.set_position([l, b, w, h+0.3])
        plt.savefig(os.path.join(plot_dir, f'layer_{layer_idx}.png'))
        plt.close()


    def record_output(module, input, output, layer_idx):  
        scores = output
        expert_weights, expert_indices = torch.topk(scores, 2, dim=-1)
        expert_weights = expert_weights.softmax(dim=-1)
        gate_outputs[layer_idx].append((expert_weights.float().cpu().detach().numpy().astype(ml_dtypes.bfloat16), 
                                        expert_indices.cpu().detach().numpy()))

        
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and 'gate' in name:
            layer_idx = int(name.split(".")[2])
            handles.append(module.register_forward_hook(
                functools.partial(record_output, layer_idx=layer_idx)
            ))

    with torch.no_grad():
        output = model(enc_input.to(device_map))
    for h in handles:
        h.remove()

    with open(os.path.join(output_dir, 'gate_outputs'), 'wb') as f:
        pickle.dump(gate_outputs, f)

    for i, gate_output in enumerate(gate_outputs):
        expert_weights, expert_indices = gate_output[0]
        all_expert_weights = np.zeros((num_tokens, num_experts))
        all_expert_weights[np.arange(0, num_tokens), expert_indices.T] = expert_weights.T
        plot_one_layer(all_expert_weights.T, i)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', 
                        '--cache_dir',
                        default=None,
                        help="Cache dir.")

    parser.add_argument('-d', 
                        '--device',
                        default='cuda',
                        help="Choose device for model.")    
    
    parser.add_argument('-m', 
                        '--model_num',
                        default=0,
                        choices=['all', '0', '1', '2'],
                        help="Pick which model to train: 'all' for all models, 0 for mixtral, 1 for damex, 2 for xmoe.")
    args = parser.parse_args()
    main(model_number = int(args.model_num), device_map = args.device, cache_dir = args.cache_dir)