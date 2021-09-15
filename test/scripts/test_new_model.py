# In[0]
import pandas as pd
import numpy as np
import torch
import torch.nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sys, os
sys.path.append('../../src/')

import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

import bmk_beeline as bmk
import g_glad, genie3, g_admm

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

# weighted Kernel for weighted covariance matrix and weighting the losses for different time points
def kernel_band(bandwidth, ntimes, truncate = False):
    # bandwidth decide the shape (width), no matter the length ntimes
    t = (np.arange(ntimes)/ntimes).reshape(ntimes,1)
    tdis = np.square(pdist(t))
    mdis = 0.5 * bandwidth * np.median(tdis)

    K = squareform(np.exp(-tdis/mdis))+np.identity(ntimes)

    if truncate == True:
        cutoff = mdis * 1.5
        mask = (squareform(tdis) < cutoff).astype(np.int)
        K = K * mask

    return K/np.sum(K,axis=1)[:,None]


# In[1] read in data
path = "../../data/boolODE_Sep13/"
result_dir = "./continuous/"
sorted_exp = np.load(path + "continue_sorted_exp_1to2.npy")[:, :1500]
gt_adj = np.load(path + "continue_gt_adj_1to2.npy")[:1500, :, :]

# preprocessing
X = preprocess(sorted_exp.T)
# sort the genes
print("Raw TimePoints: {}, no.Genes: {}".format(X.shape[0],X.shape[1]))
X = StandardScaler().fit_transform(X)

ntimes, ngenes = X.shape

# transcription factor
tf = list(set(np.nonzero(gt_adj[0])[1]))

# In[2] Model
sample = torch.FloatTensor(X)
max_iters = 1000

for bandwidth in [0.01, 0.1, 1, 10]:

    empir_cov = torch.zeros(ntimes, ngenes, ngenes)
    K = kernel_band(bandwidth, ntimes)
    K_trun = kernel_band(bandwidth, ntimes, truncate = True)

    # building weighted covariance matrix, output is empir_cov of the shape (ntimes, ngenes, ngenes)
    for time in range(ntimes):
        weight = torch.FloatTensor(K_trun[time, :])
        # assert torch.sum(weight) == 1

        bin_weight = torch.FloatTensor((K_trun[time, :] > 0).astype(np.int))
        sample_mean = torch.sum(sample * weight[:, None], dim = 0)
        # sample_mean = torch.sum(sample * (bin_weight/torch.sum(bin_weight))[:, None], dim = 0)

        norm_sample = sample - sample_mean[None, :]
        empir_cov[time] = torch.sum(torch.bmm(norm_sample[:,:,None], norm_sample[:,None,:]) * weight[:,None, None], dim=0)
        
    # run the model
    for lamb in [0.01, 0.1, 1]:
        for rho in [1, 10]:
            # test model without TF
            thetas_weighted = np.zeros((ntimes,ngenes,ngenes))

            for time in range(ntimes):
                #X > X[:,None,:]
                lamb = 0.1
                rho = 0.1
                # TODO: the dual residual keep exploding, maybe check the def of dual residual and tune the parameter (in this case, reducing rh), in addition, include the mode of varying dual residual.
                gadmm_single = g_admm.G_admm(X = X[:,None,:], K = K, pre_cov = empir_cov)
                thetas_weighted[time] = gadmm_single.train(t = time, max_iters = max_iters, n_intervals = 10, lamb = lamb , rho = None, theta_init_offset = 0.1)


            # gadmm_batch = g_admm.G_admm_batch(X = X[:,None,:], K = K, pre_cov = empir_cov)
            # glad_thetas_weighted = gadmm_batch.train(max_iters = max_iters, n_intervals = 1, lamb = lamb , rho = 1, theta_init_offset = 0.1)
            # np.save(file = result_dir + "glad_thetas_weighted_" + str(bandwidth) + "_" + str(lamb) + "_" + str(rho) + ".npy", arr = glad_thetas_weighted)  



# %%
