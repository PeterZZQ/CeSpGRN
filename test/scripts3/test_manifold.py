# In[0]
import sys
import os
sys.path.append('../../src/')

import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.decomposition import PCA

import g_admm
import genie3
import bmk_beeline as bmk

import pandas as pd
import numpy as np
import torch
import torch.nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import time
import kernel 

def preprocess(counts): 
    """\
    Input:
    counts = (ntimes, ngenes)
    
    Description:
    ------------
    Preprocess the dataset
    """
    # normalize according to the library size
    
    libsize = np.median(np.sum(counts, axis = 1))
    counts = counts / np.sum(counts, axis = 1)[:,None] * libsize
        
    counts = np.log1p(counts)
    return counts

# In[1] read in data and preprocessing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ntimes = 3000
interval = 100
ngenes = 20
ntfs = 5
bandwidth = 1

# GGM data don't have trajectory, according to visualization
# path = "../../data/GGM/"
# # the data smapled from GGM is zero-mean
# X = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/expr.npy")[:500, :]
# gt_adj = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/Gs.npy")[:500, :, :]


# path = "../../data/boolODE_Sep13/"
# # the data smapled from GGM is zero-mean
# X = np.load(path + "continue_sorted_exp_1to2.npy").T[:500, :]
# gt_adj = np.load(path + "continue_gt_adj_1to2.npy").T[:500, :, :]
# X = preprocess(X)

# continuousODE
path = "../../data/continuousODE/node_20_dyn_2_step_200_re/"
# the data smapled from GGM is zero-mean
X = np.load(path + "gene_exp_20_42_2_200.npy").T[:500, :]
gt_adj = np.load(path + "gt_graph_20_42_2_200.npy").T[:500, :, :]
X = preprocess(X)


# sort the genes
print("Raw TimePoints: {}, no.Genes: {}".format(X.shape[0], X.shape[1]))
# X = StandardScaler().fit_transform(X)

ntimes, ngenes = X.shape
# In[] better
import importlib
importlib.reload(kernel)
# optional, conduct dimensionality reduction on X first, then put it into kernel calculation function.
for bandwidth in [0.01, 0.1, 0.5, 1, 10, 100]:
    start_time = time.time()
    # should have the smallest k that make the graph connected
    K, K_trun = kernel.calc_kernel(X, k = 3, bandwidth = bandwidth, truncate = True)
    print("time cost: {:.3f} sec".format(time.time() - start_time))
    t = 200
    Kt = K[t, :]
    # plot kernel function
    fig = plt.figure(figsize = (10, 5))
    ax = fig.add_subplot()
    ax.plot(Kt)
    ax.set_xlabel("time", fontsize = 20)
    ax.set_ylabel("weight", fontsize = 20)
    ax.set_title("shortest path, bandwidth = {:.2f}".format(bandwidth), fontsize = 20)

# In[]
import importlib
importlib.reload(kernel)
start_time = time.time()
for bandwidth in [0.01, 0.1, 1, 10, 100]:
    for diff_t in [1, 5, 10]: 
        M, K_diff, K_diff_trun = kernel.calc_diffu_kernel(X, k = 5, bandwidth = bandwidth, t = diff_t, n_eign = None, truncate = True)
        # K_diff, K_diff_trun = kernel.calc_diffu_kernel2(X, k = 10, bandwidth = bandwidth, truncate = True)
        print("time cost: {:.3f} sec".format(time.time() - start_time))
        t = 1
        Kt = K_diff[t, :]
        fig = plt.figure(figsize = (10, 5))
        ax = fig.add_subplot()
        ax.plot(Kt)
        ax.set_xlabel("time", fontsize = 20)
        ax.set_ylabel("weight", fontsize = 20)
        ax.set_title("diffusion, bandwidth = {:.2f}, t = {:d}".format(bandwidth, diff_t), fontsize = 20)


# In[2]
# the ground truth
fig = plt.figure(figsize = (10, 5))
ax = fig.add_subplot()
pca_op = PCA(n_components = 2)
umap_op = UMAP(n_components = 2, n_neighbors = 15, min_dist = 0.1)
x_umap = umap_op.fit_transform(X)

ax.scatter(x_umap[:,0], x_umap[:,1], c = np.arange(x_umap.shape[0]))

# calculated kernel function
fig = plt.figure(figsize = (10, 5))
ax = fig.add_subplot()
pca_op = PCA(n_components = 2)
umap_op = UMAP(n_components = 2, n_neighbors = 15, min_dist = 0.1)
x_umap = umap_op.fit_transform(X)

ax.scatter(x_umap[:,0], x_umap[:,1], c = Kt)




# %%
