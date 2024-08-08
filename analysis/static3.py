# %% [markdown]
# # Analysis of Static Parameters

# %% [markdown]
# This notebook includes experiments listed below:
# - Weight matrices of experts
#     - Matrix-level
#     - Neuron-level (averaging and reordering)
# - Gate embedding 
#     - Qualitative
#     - Quantitative (linear regression)
# - Projection of expert matrices in low-dimensional space
#     - Matrix-level
#     - Neuron-level
# 
# The models have their own code blocks for each experiment. The overall logic of the code belonging to different models is alike, and the minor differences stem from the unique settings of the corresponding model.
# 
# Usually, the figures are plotted in two ways: 'auto_colorbar' and 'full_colorbar'. The former allows the matplotlib methods to automatically dicide the range of the color bar for each layer. For the latter, we manually set it to be the global minimum/maximum for all the layers.

# %%
import math
import ml_dtypes
import os
import pickle

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

#from mixtral_base.modeling_moe_mistral import MixtralForCausalLM
from modelling_edullm import EduLLMForCausalLM
#from deepseekmoe.modeling_deepseek import DeepseekForCausalLM
#from grok.modeling_grok1 import Grok1ModelForCausalLM

# The root directory for saving the output figures and data.
WORK_DIR = './outputs'

# %% [markdown]
# Run one or more cells below to load the models you need.

# %%
#mixtral_model = MixtralForCausalLM.from_pretrained(
#    "./ckpt/mixtral", 
#    low_cpu_mem_usage=True, device_map="auto", torch_dtype=torch.bfloat16
#)
#mixtral_tok = AutoTokenizer.from_pretrained("./ckpt/mixtral")
#mixtral_model.eval()

# %%
cache_dir = '/cs/student/projects1/dsml/2023/elbadawi/project/.cache'

model_type = "mixtral"
model = EduLLMForCausalLM.from_pretrained(
    f'amnae/base_edu_llm_{model_type}_trained',
    low_cpu_mem_usage=True, device_map="cpu", torch_dtype=torch.bfloat16,
    cache_dir=cache_dir
)
tokeniser = AutoTokenizer.from_pretrained(f'amnae/base_edu_llm_{model_type}_trained')

#model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1",
#											cache_dir = cache_dir,
#											device_map  = "cpu") 
#tokeniser = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")

model.eval()

# %% [markdown]
# ## Weight Matrices of Experts

# %% [markdown]
# This section contains the code of both **matrix-level** and **neuron-level** analyses. We propose two methods for the neuron-level analyses: averaging and reordering. The code of matrix-level part and averaging is identical, except that you have to set `average=True` for the latter. Note that the reordering approach takes considerable time to run.

# %% [markdown]
# ### Mixtral and Mistral

# %% [markdown]
# #### Matrix-level / Neuron-level (averaging)
"""
# %%
cos = torch.nn.CosineSimilarity(dim=0)
matrices = [('w3', 'up_proj'), ('w1', 'gate_proj'), ('w2', 'down_proj')]
num_layers = model.config.num_hidden_layers
num_experts = model.config.num_local_experts
num_neurons = model.config.intermediate_size
average = False # True for neuron-level.

all_sim_arr = [[] for _ in range(num_layers)]
global_vmax, global_vmin = -1 * math.inf, math.inf
for i in range(num_layers):
    print(f"Layer {i}")
    for (mix_mat, mis_mat) in matrices:
        if average:
            mean_dim = 1 if mix_mat == 'w2' else 0
            all_matrix = [torch.mean(getattr(model.model.layers[i].block_sparse_moe.experts[idx], mix_mat).weight, dim=mean_dim)
                           for idx in range(num_experts)]
        else:
            all_matrix = [getattr(model.model.layers[i].block_sparse_moe.experts[idx], mix_mat).weight.flatten() 
                           for idx in range(num_experts)]
        sim_arr = np.empty((num_experts, num_experts))
        for j in range(num_experts):
            for k in range(j, num_experts):
                # Mixtral and Mistral layers can be loaded on differnet GPUs, so put them on the same device manually. 
                sim = cos(all_matrix[j].to('cpu'), all_matrix[k].to('cpu')).float().cpu().detach().numpy().astype(ml_dtypes.bfloat16)
                #print(sim)
                #print(torch.norm(all_matrix[j].to('cpu') - all_matrix[k].to('cpu')))
                #print("----")
                sim_arr[j][k] = sim
                sim_arr[k][j] = sim
        all_sim_arr[i].append(sim_arr)
        # Record the maximum and minimum values for plotting.
        curr_vmax = np.max(sim_arr)
        curr_vmin = np.min(sim_arr)
        if curr_vmin < global_vmin:
            global_vmin = curr_vmin
        if curr_vmax > global_vmax:
            global_vmax = curr_vmax

# %%
# Save and plot.
tick_labels = [str(i) for i in range(num_experts)]
tick_labels.append('F')
save_dir = os.path.join(WORK_DIR, f'edullm/{model_type}_experts_sim')
if average:
    save_dir += '_average'
plot_dir = os.path.join(save_dir, 'figure')
output_dir = os.path.join(save_dir, 'data')
os.makedirs(os.path.join(plot_dir, 'auto_colorbar'), exist_ok=True)
os.makedirs(os.path.join(plot_dir, 'full_colorbar'), exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

output_dict = {'global_vmax': global_vmax, 'global_vmin':global_vmin}
with open(os.path.join(output_dir, 'all_sim_arr'), 'wb') as f:
    pickle.dump(all_sim_arr, f)
with open(os.path.join(output_dir, 'output_dict'), 'wb') as f:
    pickle.dump(output_dict, f)


def plot_one_layer(arr_lst, layer_idx, range_type, global_vmin=None, global_vmax=None):
    fig, axs = plt.subplots(ncols=3, layout='constrained', figsize=(6., 1.85))
    imlst = []
    for l, sim_arr in enumerate(arr_lst):
        if range_type == 'auto_colorbar':
            im = axs[l].imshow(sim_arr)
            imlst.append(im)
        elif range_type == 'full_colorbar':
            im = axs[l].imshow(sim_arr, vmin=global_vmin, vmax=global_vmax)
        axs[l].set_xticks(np.arange(num_experts+1), labels=tick_labels, fontsize=13)
        axs[l].set_yticks(np.arange(num_experts+1), labels=tick_labels, fontsize=13)
        if l == 0:
            axs[l].set_ylabel(f'Layer {layer_idx}', labelpad=10., fontsize=16)
    if range_type == 'auto_colorbar':
        local_vmin = min(img.get_array().min() for img in imlst)
        local_vmax = max(img.get_array().max() for img in imlst)
        norm = colors.Normalize(vmin=local_vmin, vmax=local_vmax)
        for img in imlst:
            img.set_norm(norm)
    cbar = fig.colorbar(im, ax=axs, shrink=.85)
    cbar.ax.tick_params(labelsize=13)
    plt.savefig(os.path.join(plot_dir, range_type, f'layer_{layer_idx}.png'))
    plt.close()


for i in range(num_layers):
    plot_one_layer(all_sim_arr[i], i, 'auto_colorbar')
    plot_one_layer(all_sim_arr[i], i, 'full_colorbar', global_vmin, global_vmax)

# %% [markdown]
# #### Neuron-level (reordering)

# %%
from scipy.optimize import linear_sum_assignment
from scipy.stats import kendalltau

cos_dim0 = torch.nn.CosineSimilarity(dim=0)
cos_dim1 = torch.nn.CosineSimilarity(dim=1)
matrices = [('w3', 'up_proj'), ('w1', 'gate_proj'), ('w2', 'down_proj')]
num_layers = model.config.num_hidden_layers
num_experts = model.config.num_local_experts
num_neurons = model.config.intermediate_size
save_dir = os.path.join(WORK_DIR, f'mixtral/mixtral_experts_sim_reorder')
output_dir = os.path.join(save_dir, 'data')
os.makedirs(output_dir, exist_ok=True)


def get_permute_idx(mat1, mat2, mat_type):
    # Find the permutation of mat2 which maximizes the 
    # cosine similarity between mat1 and mat2.
    all_neuron_sim = np.empty((num_neurons, num_neurons))
    for i in range(num_neurons):
        neuron1 = mat1[:, i] if mat_type == 'w2' else mat1[i, :]
        if mat_type == 'w2':
            all_neuron_sim[i] = cos_dim0(neuron1.unsqueeze(1), mat2).float().cpu().detach().numpy().astype(ml_dtypes.bfloat16)
        else:
            all_neuron_sim[i] = cos_dim0(neuron1.unsqueeze(1), mat2.T).float().cpu().detach().numpy().astype(ml_dtypes.bfloat16)
    _, col_idx = linear_sum_assignment(all_neuron_sim, maximize=True)
    return col_idx
"""

"""
all_sim_arr = [[] for _ in range(num_layers)]
all_kerror_arr = [[] for _ in range(num_layers)]
layers = [0, 5, 10, 15, 20, 25, 30, 31]
for i in layers:
    print(f'Layer {i}')
    for m, (mix_mat, mis_mat) in enumerate(matrices):
        print(f'{mix_mat}/{mis_mat}')
        all_matrix = [getattr(model.model.layers[i].block_sparse_moe.experts[idx], mix_mat).weight
                           for idx in range(num_experts)]
        #mistral_matrix = getattr(mis_model.layers[i].block_sparse_moe, mis_mat).weight
        #all_matrix.append(mistral_matrix)
        sim_arr = np.empty((num_experts, num_experts))
        kerror_arr = np.empty((num_experts, num_experts))
        for j in range(num_experts):
            kerror_lst = [[] for _ in range(len(matrices))]
            for k in range(j, num_experts):
                if j == k:
                    max_sim = 1.
                    kerror = 1.
                else:
                    permute_idx = get_permute_idx(all_matrix[j], all_matrix[k], mis_mat)
                    kerror = kendalltau(permute_idx, np.arange(num_neurons)).statistic
                    permute_mat = all_matrix[k][:, permute_idx] if mis_mat == 'down_proj' else all_matrix[k][permute_idx]
                    cos = cos_dim0 if mis_mat == 'down_proj' else cos_dim1
                    max_sim = cos(all_matrix[j].flatten(), permute_mat.flatten()).float().cpu().detach().numpy().astype(ml_dtypes.bfloat16)
                # ori_sim = cos_dim0(all_matrix[j].flatten(), all_matrix[k].flatten()).float().cpu().detach().numpy().astype(ml_dtypes.bfloat16)
                # print(max_sim - ori_sim)
                sim_arr[j][k] = max_sim
                sim_arr[k][j] = max_sim
                kerror_arr[j][k] = kerror
                kerror_arr[k][j] = kerror
        all_sim_arr[i].append(sim_arr)
        all_kerror_arr[i].append(kerror_arr)

        with open(os.path.join(output_dir, f'layer{i}_all_sim_arr'), 'wb') as f:
            pickle.dump(all_sim_arr[i], f)
        with open(os.path.join(output_dir, f'layer{i}_all_kerror_arr'), 'wb') as f:
            pickle.dump(all_kerror_arr[i], f)
"""

# %% [markdown]
# ## Gate Embedding

# %% [markdown]
# We have conducted both qualitative and quantitative analysis for the gate embedding. To run this experiment, you have to first execute the 1st code block to compute the similarities, then you can plot the heat map (qualitative) and/or perform the linear regression (quantitative).

# %% [markdown]
# ### Mixtral and Mistral

# %%
cos = torch.nn.CosineSimilarity(dim=0)
matrices = ['gate','w3', 'w1','w2']

num_layers = model.config.num_hidden_layers
num_experts = model.config.num_local_experts
average = True

all_sim_arr = [[] for _ in range(num_layers)]
global_vmax, global_vmin = -1 * math.inf, math.inf
for i in range(num_layers):
    # Calcualte similarity between neurons in gate embedding.
    print(f"Layer {i}")
    gate = model.model.layers[i].block_sparse_moe.gate.weight
    sim_arr = np.empty((num_experts, num_experts))
    for j in range(num_experts):
        for k in range(j, num_experts):
            print(f"Layer {j} vs {k}")
            sim = cos(gate[j, :], gate[k, :]).float().cpu().detach().numpy().astype(ml_dtypes.bfloat16)  
            sim_arr[j][k] = sim
            sim_arr[k][j] = sim
    all_sim_arr[i].append(sim_arr)
    curr_vmax = np.max(sim_arr)
    curr_vmin = np.min(sim_arr)
    if curr_vmin < global_vmin:
        global_vmin = curr_vmin
    if curr_vmax > global_vmax:
        global_vmax = curr_vmax
    # Calculate similarity between Mixtral experts.
    for mix_mat in matrices[1:]:
        if average:
            mean_dim = 1 if mix_mat == 'w2' else 0
            all_experts = [torch.mean(getattr(model.model.layers[i].block_sparse_moe.experts[idx], mix_mat).weight, dim=mean_dim)
                           for idx in range(num_experts)]
        else:
            all_experts = [getattr(model.model.layers[i].block_sparse_moe.experts[idx], mix_mat).weight.flatten() 
                           for idx in range(num_experts)]
        sim_arr = np.empty((num_experts, num_experts))
        for j in range(num_experts):
            for k in range(j, num_experts):
                sim = cos(all_experts[j], all_experts[k]).float().cpu().detach().numpy().astype(ml_dtypes.bfloat16)  
                sim_arr[j][k] = sim
                sim_arr[k][j] = sim
        all_sim_arr[i].append(sim_arr)
        curr_vmax = np.max(sim_arr)
        curr_vmin = np.min(sim_arr)
        if curr_vmin < global_vmin:
            global_vmin = curr_vmin
        if curr_vmax > global_vmax:
            global_vmax = curr_vmax

# %% [markdown]
# Qualitative analysis (plotted along with the neuron-level heat maps of expert weight matrices)

# %%
# Save and plot.
save_dir = os.path.join(WORK_DIR, f'edullm/{model_type}_gate_sim')
plot_dir = os.path.join(save_dir, 'figure')
output_dir = os.path.join(save_dir, 'data')
os.makedirs(os.path.join(plot_dir, 'auto_colorbar'), exist_ok=True)
os.makedirs(os.path.join(plot_dir, 'full_colorbar'), exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
tick_labels = [str(i) for i in range(num_experts)]

output_dict = {'global_vmax':global_vmax, 'global_vmin':global_vmin}
with open(os.path.join(output_dir, 'all_sim_arr'), 'wb') as f:
    pickle.dump(all_sim_arr, f)
with open(os.path.join(output_dir, 'output_dict'), 'wb') as f:
    pickle.dump(output_dict, f)
    

def plot_one_layer(arr_lst, layer_idx, range_type, global_vmin=None, global_vmax=None):
    imlst = []
    fig, axs = plt.subplots(ncols=4, layout='constrained', figsize=(8.5, 2.0))
    for i, sim_arr in enumerate(arr_lst):
        if range_type == 'auto_colorbar':
            im = axs[i].imshow(sim_arr)
            imlst.append(im)
        elif range_type == 'full_colorbar':
            im = axs[i].imshow(sim_arr, vmin=global_vmin, vmax=global_vmax)
        axs[i].set_xticks(np.arange(num_experts), labels=tick_labels, fontsize=15)
        axs[i].set_yticks(np.arange(num_experts), labels=tick_labels, fontsize=15)
        if i == 0:
            axs[i].set_ylabel(f'Layer {layer_idx}', labelpad=14., fontsize=20)
    if range_type == 'auto_colorbar':
        local_vmin = min(img.get_array().min() for img in imlst)
        local_vmax = max(img.get_array().max() for img in imlst)
        norm = colors.Normalize(vmin=local_vmin, vmax=local_vmax)
        for img in imlst:
            img.set_norm(norm)
    cbar = fig.colorbar(im, ax=axs, shrink=1.)
    cbar.ax.tick_params(labelsize=15)
    plt.savefig(os.path.join(plot_dir, range_type, f'layer_{layer_idx}.png'))
    plt.close()

for i in range(num_layers):
    plot_one_layer(all_sim_arr[i], i, 'auto_colorbar')
    plot_one_layer(all_sim_arr[i], i, 'full_colorbar', global_vmin, global_vmax)

# %% [markdown]
# Quatitative analysis (linear regression)

# %%
from scipy.stats import linregress

save_dir = os.path.join(WORK_DIR, f'edullm/{model_type}_gate_sim_reg')
output_dir = os.path.join(save_dir, 'data')
os.makedirs(output_dir, exist_ok=True)

# Reorganize the similarity matrices to be one-to-one value pairs.
all_data = [[np.array([]) for _ in range(len(matrices))] for _ in range(num_layers)]
for i in range(num_layers):
    for j, sim_arr in enumerate(all_sim_arr[i]):
        # Iterate over the similarity array to flatten the 
        # low triangle area (excluding the diagonal).
        for row in range(num_experts):
            for col in range(row):
                all_data[i][j] = np.append(all_data[i][j], sim_arr[row][col])

# Perform linear regression.
sum_r2 = [0. for _ in range(len(matrices))]
all_r_lst = [[] for _ in range(num_layers)]
for i in range(num_layers):
    for j in range(1, len(matrices)):
        X, Y = all_data[i][j], all_data[i][0]
        slope, intercept, r, p, stderr = linregress(X, Y)
        r2 = round(r**2, 2)
        sum_r2[j] += r2
        all_r_lst[i].append(r)

print('Average regression score\nup_proj: {:.2f}\ngate_proj: {:.2f}\ndown_proj: {:.2f}'.format(
    sum_r2[1]/num_layers, sum_r2[2]/num_layers, sum_r2[3]/num_layers))

with open(os.path.join(output_dir, 'all_data'), 'wb') as f:
    pickle.dump(all_data, f)
with open(os.path.join(output_dir, 'all_r_lst'), 'wb') as f:
    pickle.dump(all_r_lst, f)

# %% [markdown]
# ## Projection of Expert Matrices in Low-dimensional Space

# %% [markdown]
# This section includes the PCA projection code in both the matrix level and neuron level. For the neuron level, you can set `n_dim=2` or `n_dim=3` to change the dimension.

# %% [markdown]
# ### Mixtral and Mistral

# %% [markdown]
# #### Matrix-level

# %%
from sklearn.decomposition import PCA

matrices = [('w3', 'up_proj'), ('w1', 'gate_proj'), ('w2', 'down_proj')]
num_layers = model.config.num_hidden_layers
num_experts = model.config.num_local_experts
num_neurons = model.config.intermediate_size
hidden_size = model.config.hidden_size
use_normalize = True
        
all_projected_matrix = [[] for _ in range(num_layers)]
for i in range(num_layers):
    print(i)
    for mix_mat, mis_mat in matrices:
        all_matrix = torch.empty(num_experts, num_neurons*hidden_size)
        for idx in range(num_experts):
            if mix_mat == 'w2':
                all_matrix[idx] = getattr(model.model.layers[i].block_sparse_moe.experts[idx], mix_mat).weight.T.flatten()
            else:
                all_matrix[idx] = getattr(model.model.layers[i].block_sparse_moe.experts[idx], mix_mat).weight.flatten()
        if use_normalize:
            mean, std = torch.mean(all_matrix, dim=0), torch.std(all_matrix, dim=0)
            all_matrix = (all_matrix - mean) / std
        pca = PCA(n_components=2, svd_solver='full')
        projected_matrix = pca.fit_transform(all_matrix.float().cpu().detach().numpy().astype(ml_dtypes.bfloat16))
        all_projected_matrix[i].append(projected_matrix)

# %%
# Save and plot.
save_dir = os.path.join(WORK_DIR, f'edullm/{model_type}_experts_pca')
plot_dir = os.path.join(save_dir, 'figure')
output_dir = os.path.join(save_dir, 'data')
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, 'all_projected_matrix'), 'wb') as f:
    pickle.dump(all_projected_matrix, f)


def plot_one_layer(projected_matrix_lst, layer_idx):
    fig, axs = plt.subplots(ncols=3, layout='constrained', figsize=(9., 2.5))
    for l, projected_matrix in enumerate(projected_matrix_lst):
        mix_X = projected_matrix[:-1, 0]
        mix_Y = projected_matrix[:-1, 1]
        mis_X = projected_matrix[-1, 0]
        mis_Y = projected_matrix[-1, 1]
        axs[l].scatter(mix_X, mix_Y, marker='o', label='expert')
        axs[l].scatter(mis_X, mis_Y, marker='^', label='FFN')
        axs[l].legend(loc='best', fontsize=11)
        if l == 0:
            axs[l].set_ylabel(f'Layer {layer_idx}', labelpad=10., fontsize=16)
        for i in range(num_experts):
            axs[l].annotate(i, (mix_X[i], mix_Y[i]), fontsize=12)
    plt.savefig(os.path.join(plot_dir, f'layer_{layer_idx}.png'))
    plt.close()
    

for i in range(num_layers):
    plot_one_layer(all_projected_matrix[i], i)

# %% [markdown]
# #### Neuron-level

# %%
from sklearn.decomposition import PCA

matrices = [('w3', 'up_proj'), ('w1', 'gate_proj'), ('w2', 'down_proj')]
num_layers = model.config.num_hidden_layers
num_experts = model.config.num_local_experts
num_neurons = model.config.intermediate_size
use_normalize = True
n_dim = 2 # 2D or 3D space
assert n_dim in [2, 3]

save_dir = os.path.join(WORK_DIR, f'edullm/{model_type}_experts_pca_neuron/{n_dim}d')
plot_dir = os.path.join(save_dir, 'figure')
output_dir = os.path.join(save_dir, 'data')
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)


def plot_one_2d_layer(projected_neuron_lst, layer_idx):
    fig, axs = plt.subplots(ncols=3, layout='constrained', figsize=(9., 2.7))
    for l, projected_neuron in enumerate(projected_neuron_lst):
        mix_X = projected_neuron[:-1*num_neurons, 0]
        mix_Y = projected_neuron[:-1*num_neurons, 1]
        mis_X = projected_neuron[-1*num_neurons:, 0]
        mis_Y = projected_neuron[-1*num_neurons:, 1]
        color = []
        for i in range(num_experts+1):
            color.extend([i]*num_neurons)
        if l == 0:
            axs[l].set_ylabel(f'Layer {layer_idx}', labelpad=10., fontsize=16)
        axs[l].scatter(mix_X, mix_Y, marker='o', c=color[:-1*num_neurons], cmap=plt.cm.Spectral)
        axs[l].scatter(mis_X, mis_Y, marker='^', c=color[-1*num_neurons:], cmap=plt.cm.Spectral)
    plt.savefig(os.path.join(plot_dir, f'layer_{layer_idx}.png'))
    plt.close()


def plot_one_3d_layer(projected_neuron_lst, layer_idx):
    fig, axs = plt.subplots(ncols=3, layout='constrained', figsize=(9., 2.7), subplot_kw=dict(projection='3d'))
    for l, projected_neuron in enumerate(projected_neuron_lst):
        mix_X = projected_neuron[:-1*num_neurons, 0]
        mix_Y = projected_neuron[:-1*num_neurons, 1]
        mix_Z = projected_neuron[:-1*num_neurons, 2]
        mis_X = projected_neuron[-1*num_neurons:, 0]
        mis_Y = projected_neuron[-1*num_neurons:, 1]
        mis_Z = projected_neuron[-1*num_neurons:, 2]
        color = []
        for i in range(num_experts+1):
            color.extend([i]*num_neurons)
        axs[l].scatter(mix_X, mix_Y, mix_Z, marker='o', c=color[:-1*num_neurons], cmap=plt.cm.Spectral)
        axs[l].scatter(mis_X, mis_Y, mis_Z, marker='^', c=color[-1*num_neurons:], cmap=plt.cm.Spectral)
    plt.savefig(os.path.join(plot_dir, f'layer_{layer_idx}.png'))
    plt.close()


for i in range(num_layers):
    print(i)
    projected_neuron_lst = []
    for mix_mat, mis_mat in matrices:
        all_neuron_lst = []
        for idx in range(num_experts):
            if mis_mat == 'down_proj':
                all_neuron_lst.append(getattr(model.model.layers[i].block_sparse_moe.experts[idx], mix_mat).weight.T.to('cpu'))
            else:
                all_neuron_lst.append(getattr(model.model.layers[i].block_sparse_moe.experts[idx], mix_mat).weight.to('cpu'))
        all_neuron = torch.cat(all_neuron_lst, dim=0)
        if use_normalize:
            mean, std = torch.mean(all_neuron, dim=0), torch.std(all_neuron, dim=0)
            all_neuron = (all_neuron - mean) / std
        pca = PCA(n_components=n_dim, svd_solver='full')
        projected_neuron = pca.fit_transform(all_neuron.float().cpu().detach().numpy().astype(ml_dtypes.bfloat16))
        projected_neuron_lst.append(projected_neuron)
    
    with open(os.path.join(output_dir, 'layer{i}_projected_neuron_lst'), 'wb') as f:
        pickle.dump(projected_neuron_lst, f)

    if n_dim == 2:
        plot_one_2d_layer(projected_neuron_lst, i)
    elif n_dim == 3:
        plot_one_3d_layer(projected_neuron_lst, i)



