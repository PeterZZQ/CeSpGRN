# In[0]
import sys, os
sys.path.append('../../src/')

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import time
import gc
import torch
import torch.nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from umap import UMAP

import bmk_beeline as bmk
import genie3, g_admm, kernel
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rcParams["font.size"] = 20

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

# In[1]
path = "../../data/COUNTS-THP-1/"
result_dir = "../results_THP-1/"

counts = pd.read_csv(path + "counts.csv", index_col = 0).values
annotation = pd.read_csv(path + "anno.csv", index_col = 0)
dpt = pd.read_csv(path + "dpt_time.txt", index_col = 0, sep = "\t", header = None).values.squeeze()
# counts = counts[np.argsort(dpt), :]
ncells, ngenes = counts.shape
assert ncells == 8 * 120
assert ngenes == 45
print("Raw TimePoints: {}, no.Genes: {}".format(counts.shape[0],counts.shape[1]))

# check distribution
fig = plt.figure(figsize = (10,7))
ax = fig.add_subplot()
_ = ax.hist(counts.reshape(-1), bins = 20)
fig = plt.figure(figsize = (10,7))
ax = fig.add_subplot()
_ = ax.hist(np.log1p(counts.reshape(-1)), bins = 20)
# the distribution of the original count is log-normal distribution, conduct log transform
counts = np.log1p(counts)

pca_op = PCA(n_components = 20)
umap_op = UMAP(n_components = 2, min_dist = 0.8)
mds_op = MDS(n_components = 2)
X_pca = pca_op.fit_transform(counts)
# X_pca = umap_op.fit_transform(counts)
# X_pca = mds_op.fit_transform(counts)

fig = plt.figure(figsize  = (10,7))
ax = fig.add_subplot()
for i in np.sort(np.unique(annotation.values.squeeze())):
    idx = np.where(annotation.values.squeeze() == i)
    ax.scatter(X_pca[idx, 0], X_pca[idx, 1], label = i)

ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False)
fig.savefig(result_dir + "X_pca.png", bbox_inches = "tight")

X = torch.FloatTensor(counts).to(device)


# In[2] ADMM
# hyper-parameter
for bandwidth in [0.01, 0.1, 1]:
    for truncate_param in [5, 15, 30]:
        for lamb in [0.001, 0.01, 0.05, 0.1]:
            alpha = 2
            rho = 1.7
            max_iters = 1000

            # calculate empirical covariance matrix
            start_time = time.time()
            empir_cov = torch.zeros(ncells, ngenes, ngenes)
            # calculate the kernel function
            # K, K_trun = kernel.calc_kernel(counts, k = 5, bandwidth = bandwidth, truncate = True, truncate_param = truncate_param)
            K, K_trun = kernel.calc_kernel_neigh(X_pca, k = 5, bandwidth = bandwidth, truncate = True, truncate_param = truncate_param)

            # plot kernel function
            fig = plt.figure(figsize = (20, 7))
            axs = fig.subplots(1, 2)
            axs[0].plot(K[int(ncells/2), :])
            axs[1].plot(K_trun[int(ncells/2), :])
            fig.suptitle("kernel_" + str(bandwidth) + "_" + str(truncate_param))
            fig.savefig(result_dir + "kernel_" + str(bandwidth) + "_" + str(truncate_param) + ".png", bbox_inches = "tight")
            print("number of neighbor being considered: " + str(np.sum(K_trun[int(ncells/2), :] > 0)))
            # building weighted covariance matrix, output is empir_cov of the shape (ncells, ngenes, ngenes)
            for t in range(ncells):
                weight = torch.FloatTensor(K_trun[t, :]).to(device)
                # assert torch.sum(weight) == 1

                bin_weight = torch.FloatTensor((K_trun[t, :] > 0).astype(np.int))
                sample_mean = torch.sum(X * weight[:, None], dim = 0)
                # sample_mean = torch.sum(X * (bin_weight/torch.sum(bin_weight))[:, None], dim = 0)

                norm_sample = X - sample_mean[None, :]
                empir_cov[t] = torch.sum(torch.bmm(norm_sample[:,:,None], norm_sample[:,None,:]) * weight[:,None, None], dim=0)
            print("time calculating the kernel function: {:.2f} sec".format(time.time() - start_time))

            start_time = time.time() 
            # test model without TF
            thetas = np.zeros((ncells, ngenes, ngenes))

            # gadmm_batch = g_admm.G_admm_batch(X=X[:, None, :], K=K, pre_cov=empir_cov)
            gadmm_batch = g_admm.G_admm_minibatch(X=counts[:, None, :], K=K, pre_cov=empir_cov, batchsize = 120)
            thetas = gadmm_batch.train(max_iters=max_iters, n_intervals=100, alpha=alpha, lamb=lamb, rho=rho, theta_init_offset=0.1)
            np.save(file = result_dir + "thetas_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_0.npy", arr = thetas) 
            print("time calculating thetas: {:.2f} sec".format(time.time() - start_time))
            del thetas
            gadmm_batch = None
            gc.collect()

# In[3] GENIE3
# genie_theta = genie3.GENIE3(counts, gene_names=["gene_" + str(x) for x in range(ngenes)], regulators='all',tree_method='RF',K='sqrt',ntrees=1000,nthreads=1)
# genie_theta = np.repeat(genie_theta[None, :, :], ncells, axis=0)
# np.save(file = result_dir + "theta_genie.npy", arr = genie_theta)


