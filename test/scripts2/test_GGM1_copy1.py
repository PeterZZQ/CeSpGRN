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
import kernel
import time

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


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
path = "../../data/GGM/"
for interval in [100]:
    # for (ngenes, ntfs) in [(20, 5), (30, 10), (50, 20), (100, 50)]:
    for (ngenes, ntfs) in [(50, 20), (100, 50)]:
        result_dir = "../results/GGM_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "/"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        # the data smapled from GGM is zero-mean
        X = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/expr.npy")
        # gt_adj = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/Gs.npy")

        # sort the genes
        print("Raw TimePoints: {}, no.Genes: {}".format(X.shape[0],X.shape[1]))
        # X = StandardScaler().fit_transform(X)

        # make sure the dimensions are correct
        assert X.shape[0] == ntimes
        assert X.shape[1] == ngenes

        sample = torch.FloatTensor(X).to(device)
        max_iters = 2000
        ###############################################
        #
        # test without TF information
        #
        ###############################################

        for bandwidth in [0.1]:
            start_time = time.time()
            empir_cov = torch.zeros(ntimes, ngenes, ngenes).to(device)
            # calculate the kernel function
            K, K_trun = kernel.kernel_band(bandwidth, ntimes, truncate = True)

            # building weighted covariance matrix, output is empir_cov of the shape (ntimes, ngenes, ngenes)
            for t in range(ntimes):
                weight = torch.FloatTensor(K_trun[t, :]).to(device)
                # assert torch.sum(weight) == 1

                bin_weight = torch.FloatTensor((K_trun[t, :] > 0).astype(np.int))
                sample_mean = torch.sum(sample * weight[:, None], dim = 0)
                # sample_mean = torch.sum(sample * (bin_weight/torch.sum(bin_weight))[:, None], dim = 0)

                norm_sample = sample - sample_mean[None, :]
                empir_cov[t] = torch.sum(torch.bmm(norm_sample[:,:,None], norm_sample[:,None,:]) * weight[:,None, None], dim=0)
            print("time calculating the kernel function: {:.2f} sec".format(time.time() - start_time))
            
            start_time = time.time()                    
            # run the model
            for lamb in [0.01]:
                # test model without TF
                thetas = np.zeros((ntimes,ngenes,ngenes))

                # setting from the paper over-relaxation model
                alpha = 2
                rho = 1.7
                gadmm_batch = g_admm.G_admm_batch(X = X[:,None,:], K = K, pre_cov = empir_cov)
                thetas = gadmm_batch.train(max_iters = max_iters, n_intervals = 100, alpha = alpha, lamb = lamb , rho = rho, theta_init_offset = 0.1)
                np.save(file = result_dir + "thetas_" + str(bandwidth) + "_" + str(alpha) + "_" + str(lamb) + "_" + str(rho) + ".npy", arr = thetas) 
                print("time calculating thetas: {:.2f} sec".format(time.time() - start_time))
            del thetas
        ###############################################
        #
        # test with TF information
        #
        ###############################################
        print("test with TF information")
        for bandwidth in [0.1]:
            start_time = time.time()
            empir_cov = torch.zeros(ntimes, ngenes, ngenes).to(device)
            K, K_trun = kernel.kernel_band(bandwidth, ntimes, truncate = True)

            # building weighted covariance matrix, output is empir_cov of the shape (ntimes, ngenes, ngenes)
            for t in range(ntimes):
                weight = torch.FloatTensor(K_trun[t, :]).to(device)
                # assert torch.sum(weight) == 1

                bin_weight = torch.FloatTensor((K_trun[t, :] > 0).astype(np.int))
                sample_mean = torch.sum(sample * weight[:, None], dim = 0)
                # sample_mean = torch.sum(sample * (bin_weight/torch.sum(bin_weight))[:, None], dim = 0)

                norm_sample = sample - sample_mean[None, :]
                empir_cov[t] = torch.sum(torch.bmm(norm_sample[:,:,None], norm_sample[:,None,:]) * weight[:,None, None], dim=0)
            print("time calculating the kernel function: {:.2f} sec".format(time.time() - start_time))
            
            start_time = time.time()                                  
            # run the model
            for lamb in [0.01]:
                # test model without TF
                thetas = np.zeros((ntimes,ngenes,ngenes))

                # setting from the paper over-relaxation model
                alpha = 2
                rho = 1.7
                gadmm_batch = g_admm.G_admm_batch(X = X[:,None,:], K = K, pre_cov = empir_cov, TF = np.arange(ntfs))
                thetas = gadmm_batch.train(max_iters = max_iters, n_intervals = 100, alpha = alpha, lamb = lamb , rho = rho, beta = 100, theta_init_offset = 0.1)
                np.save(file = result_dir + "thetas_" + str(bandwidth) + "_" + str(alpha) + "_" + str(lamb) + "_" + str(rho) + "_tfs.npy", arr = thetas) 
                print("time calculating thetas: {:.2f} sec".format(time.time() - start_time))
            del thetas

# %%
