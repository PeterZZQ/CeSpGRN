# In[]
import sys, os
sys.path.append('./src/')
from os.path import exists

import numpy as np
import pandas as pd
import time
import gc
import torch

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from umap import UMAP

import bmk_beeline as bmk
import g_admm as CeSpGRN
import kernel
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rcParams["font.size"] = 20

# In[]
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

# In[] Read in data
path = "./data/COUNTS-THP-1/"

counts = pd.read_csv(path + "counts.csv", index_col = 0).values
annotation = pd.read_csv(path + "anno.csv", index_col = 0)

ncells, ngenes = counts.shape
assert ncells == 8 * 120
assert ngenes == 45
print("Raw TimePoints: {}, no.Genes: {}".format(counts.shape[0],counts.shape[1]))

libsize = np.median(np.sum(counts, axis = 1))
counts = counts / np.sum(counts, axis = 1)[:,None] * libsize
# the distribution of the original count is log-normal distribution, conduct log transform
counts = np.log1p(counts)

pca_op = PCA(n_components = 20)
umap_op = UMAP(n_components = 2, min_dist = 0.8, random_state = 0)

X_pca = pca_op.fit_transform(counts)

# In[] Estimate cell-specific GRNs
# hyper-parameters
bandwidth = 0.1
truncate_param = 30
lamb = 0.1
max_iters = 1000

# calculate the kernel function
start_time = time.time()
empir_cov = torch.zeros(ncells, ngenes, ngenes)
K, K_trun = kernel.calc_kernel_neigh(X_pca, k = 5, bandwidth = bandwidth, truncate = True, truncate_param = truncate_param)
print("number of neighbor being considered: " + str(np.sum(K_trun[int(ncells/2), :] > 0)))

# estimate covariance matrix, output is empir_cov of the shape (ncells, ngenes, ngenes)
empir_cov = CeSpGRN.est_cov(X = counts, K_trun = K_trun, weighted_kt = True)

# estimate cell-specific GRNs
gadmm_batch = CeSpGRN.G_admm_minibatch(X=counts[:, None, :], K=K, pre_cov=empir_cov, batchsize = 120)
thetas = gadmm_batch.train(max_iters=max_iters, n_intervals=100, lamb=lamb)
np.save(file = "./thetas_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + ".npy", arr = thetas) 
print("time calculating thetas: {:.2f} sec".format(time.time() - start_time))


# In[] Plots
thetas = thetas.reshape(thetas.shape[0], -1)
thetas_pca = pca_op.fit_transform(thetas)
thetas_umap = umap_op.fit_transform(thetas)

fig = plt.figure(figsize  = (10,7))
ax = fig.add_subplot()
for i in np.sort(np.unique(annotation.values.squeeze())):
    idx = np.where(annotation.values.squeeze() == i)
    ax.scatter(thetas_pca[idx, 0], thetas_pca[idx, 1], label = i, s = 10)
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False, markerscale = 3)
ax.set_title("bandwidth: " + str(bandwidth) + ", truncate_param: " + str(truncate_param) + ", lamb: " + str(lamb))
fig.savefig("plot_thetas_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_pca.png", bbox_inches = "tight")

fig = plt.figure(figsize  = (10,7))
ax = fig.add_subplot()
for i in np.sort(np.unique(annotation.values.squeeze())):
    idx = np.where(annotation.values.squeeze() == i)
    ax.scatter(thetas_umap[idx, 0], thetas_umap[idx, 1], label = i, s = 10)
ax.set_xlabel("UMAP1")
ax.set_ylabel("UMAP2")
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False, markerscale = 3)
ax.set_title("bandwidth: " + str(bandwidth) + ", truncate_param: " + str(truncate_param) + ", lamb: " + str(lamb))
fig.savefig("plot_thetas_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_umap.png", bbox_inches = "tight")

            

# %%
