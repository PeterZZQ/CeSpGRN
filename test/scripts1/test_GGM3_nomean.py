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
import genie3, g_admm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

# In[1] read in data and preprocessing
ntimes = 3000
interval = 200
ngenes = 20
path = "../../data/GGM/"
result_dir = "../results/GGM_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "/"
# the data smapled from GGM is zero-mean
X = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/expr.npy")
gt_adj = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/Gs.npy")

# sort the genes
print("Raw TimePoints: {}, no.Genes: {}".format(X.shape[0],X.shape[1]))
# X = StandardScaler().fit_transform(X)

ntimes, ngenes = X.shape
ntfs = 5


# In[2] Model
sample = torch.FloatTensor(X).to(device)
max_iters = 2000
###############################################
#
# test without TF information
#
###############################################

for bandwidth in [0.01, 0.1, 1, 10]:

    empir_cov = torch.zeros(ntimes, ngenes, ngenes).to(device)
    K = kernel_band(bandwidth, ntimes)
    K_trun = kernel_band(bandwidth, ntimes, truncate = True)

    # building weighted covariance matrix, output is empir_cov of the shape (ntimes, ngenes, ngenes)
    for time in range(ntimes):
        weight = torch.FloatTensor(K_trun[time, :]).to(device)
        # assert torch.sum(weight) == 1

        bin_weight = torch.FloatTensor((K_trun[time, :] > 0).astype(np.int))
        sample_mean = torch.sum(sample * weight[:, None], dim = 0)
        # sample_mean = torch.sum(sample * (bin_weight/torch.sum(bin_weight))[:, None], dim = 0)

        norm_sample = sample #- sample_mean[None, :]
        empir_cov[time] = torch.sum(torch.bmm(norm_sample[:,:,None], norm_sample[:,None,:]) * weight[:,None, None], dim=0)
        
    # run the model
    for lamb in [0.01, 0.1, 1]:
        # test model without TF
        thetas = np.zeros((ntimes,ngenes,ngenes))

        # adaptive rho
        alpha = 1
        gadmm_batch = g_admm.G_admm_batch(X = X[:,None,:], K = K, pre_cov = empir_cov)
        thetas = gadmm_batch.train(max_iters = max_iters, n_intervals = 100, alpha = alpha, lamb = lamb , rho = None, theta_init_offset = 0.1)
        np.save(file = result_dir + "thetas_" + str(bandwidth) + "_" + str(alpha) + "_" + str(lamb) + "_adaptive_nomean.npy", arr = thetas) 

###############################################
#
# test with TF information
#
###############################################

for bandwidth in [0.01, 0.1, 1, 10]:

    empir_cov = torch.zeros(ntimes, ngenes, ngenes).to(device)
    K = kernel_band(bandwidth, ntimes)
    K_trun = kernel_band(bandwidth, ntimes, truncate = True)

    # building weighted covariance matrix, output is empir_cov of the shape (ntimes, ngenes, ngenes)
    for time in range(ntimes):
        weight = torch.FloatTensor(K_trun[time, :]).to(device)
        # assert torch.sum(weight) == 1

        bin_weight = torch.FloatTensor((K_trun[time, :] > 0).astype(np.int))
        sample_mean = torch.sum(sample * weight[:, None], dim = 0)
        # sample_mean = torch.sum(sample * (bin_weight/torch.sum(bin_weight))[:, None], dim = 0)

        norm_sample = sample #- sample_mean[None, :]
        empir_cov[time] = torch.sum(torch.bmm(norm_sample[:,:,None], norm_sample[:,None,:]) * weight[:,None, None], dim=0)
        
    # run the model
    for lamb in [0.01, 0.1, 1]:
        # test model without TF
        thetas = np.zeros((ntimes,ngenes,ngenes))

        # adaptive rho
        alpha = 1
        gadmm_batch = g_admm.G_admm_batch(X = X[:,None,:], K = K, pre_cov = empir_cov, TF = np.arange(ntfs))
        thetas = gadmm_batch.train(max_iters = max_iters, n_intervals = 100, alpha = alpha, lamb = lamb , rho = None, beta = 100, theta_init_offset = 0.1)
        np.save(file = result_dir + "thetas_" + str(bandwidth) + "_" + str(alpha) + "_" + str(lamb) + "_adaptive_tfs_nomean.npy", arr = thetas) 






# %%
