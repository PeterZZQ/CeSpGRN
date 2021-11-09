# In[0]
import pandas as pd
import numpy as np
import torch
import torch.nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sys, os
sys.path.append('../../src/')

from scipy.spatial.distance import pdist, squareform

import bmk_beeline as bmk
import genie3, g_admm
import kernel
import time
import gc
from sklearn.decomposition import PCA

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

pca_op = PCA(n_components = 10)

def cal_cov(X, K_trun):
    start_time = time.time()
    X = torch.FloatTensor(X)
    ntimes = X.shape[0]
    ngenes = X.shape[1]

    for t in range(ntimes * nsamples):
        weight = torch.FloatTensor(K_trun[t, :])
        # assert torch.sum(weight) == 1

        bin_weight = torch.FloatTensor((K_trun[t, :] > 0).astype(np.int))
        sample_mean = torch.sum(X * weight[:, None], dim = 0)

        norm_sample = X - sample_mean[None, :]
        empir_cov[t] = torch.sum(torch.bmm(norm_sample[:,:,None], norm_sample[:,None,:]) * weight[:,None, None], dim=0)
    print("time calculating the kernel function: {:.2f} sec".format(time.time() - start_time))
    return empir_cov



# In[1] test with the first set of hyper-parameters
kts = [True, False]
intervals = [5, 25]
seeds = [0,1,2]

ntimes = 1000
nsamples = 1
path = "../../data/GGM_bifurcate/"
max_iters = 500

for interval in intervals:
    for (ngenes, ntfs) in [(50, 20), (200, 20)]:
        for seed in seeds:
            result_dir = "../results_GGM/bifur_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "_" + str(seed) + "_sergio/"
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            # the data smapled from GGM is zero-mean
            X = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "_" + str(seed)+ "_sergio/expr.npy")

            ###############################################
            #
            # GENIE3
            #
            ###############################################

            X_genie = X.reshape(ntimes * nsamples, ngenes)
            # genie_theta of the shape (ntimes, ngenes, ngenes)
            genie_theta = genie3.GENIE3(X_genie, gene_names=["gene_" + str(x) for x in range(ngenes)], regulators='all',tree_method='RF',K='sqrt',ntrees=1000,nthreads=1)
            genie_theta = np.repeat(genie_theta[None, :, :],ntimes,axis=0)
            np.save(file = result_dir + "theta_genie.npy", arr = genie_theta)      


            ###############################################
            #
            # GENIE-dyn
            #
            ###############################################

            interval_size = 100
            genie_thetas = []
            for i in range(np.int(ntimes/interval_size)):
                if i != np.int(ntimes/interval_size) - 1:
                    X_genie = X[i*interval_size:(i+1)*interval_size, :]
                else:
                    X_genie = X[i*interval_size:, :]
                # genie_theta of the shape (ntimes, ngenes, ngenes)
                genie_theta = genie3.GENIE3(X_genie, gene_names=["gene_" + str(x) for x in range(ngenes)], regulators='all',tree_method='RF',K='sqrt',ntrees=1000,nthreads=1)
                genie_theta = np.repeat(genie_theta[None, :, :], X_genie.shape[0], axis=0)
                genie_thetas.append(genie_theta.copy())

            np.save(file = result_dir + "theta_genie_dyn.npy", arr = np.concatenate(genie_thetas, axis = 0))      
