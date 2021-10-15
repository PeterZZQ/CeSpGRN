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
plt.rcParams["font.size"] = 16


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

# In[1] run model
ntimes = 1000
path = "../../data/continuousODE/sergio_dense/"
pca_op = PCA(n_components = 5)
# for ngenes, ntfs in [(20, 5), (30, 10), (50, 20), (100, 50)]:
for ngenes, ntfs in [(20, 5)]:
    for interval in [200]:
        stepsize = 0.0001
        result_dir = "../results_softODE_sergio/results_ngenes_" + str(ngenes) + "_interval_" + str(interval) + "_stepsize_" + str(stepsize) + "/"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        X = np.load(path + "ngenes_" + str(ngenes) + "_interval_" + str(interval) + "_stepsize_" + str(stepsize) + "/true_count.npy")
        # X = np.load(path + "ngenes_" + str(ngenes) + "_interval_" + str(interval) + "_stepsize_" + str(stepsize) + "/obs_count.npy")
        pt = np.load(path + "ngenes_" + str(ngenes) + "_interval_" + str(interval) + "_stepsize_" + str(stepsize) + "/pseudotime.npy")
        # sort the genes
        print("Raw TimePoints: {}, no.Genes: {}".format(X.shape[0],X.shape[1]))
        # X = preprocess(X)
        X_pca = pca_op.fit_transform(X)

        fig = plt.figure(figsize = (10, 7))
        ax = fig.add_subplot()
        ax.scatter(X_pca[:, 0], X_pca[:, 1], s = 5, c = pt)

        # make sure the dimensions are correct
        assert X.shape[0] == ntimes
        assert X.shape[1] == ngenes

        sample = torch.FloatTensor(X).to(device)

        # hyper-parameter
        max_iters = 500
        for alpha, rho in [(2, 1.7)]:
            for bandwidth in [0.01, 0.1, 1, 10]:
                for lamb in [0.001, 0.01, 0.1, 0.5]:
                    start_time = time.time()
                    empir_cov = torch.zeros(ntimes, ngenes, ngenes)
                    # calculate the kernel function
                    K, K_trun = kernel.calc_kernel(X_pca, k = 5, bandwidth = bandwidth, truncate = True, truncate_param = 1)

                    # plot kernel function
                    fig = plt.figure(figsize = (20, 7))
                    axs = fig.subplots(1, 2)
                    axs[0].plot(K[int(ntimes/2), :])
                    axs[1].plot(K_trun[int(ntimes/2), :])
                    fig.savefig(result_dir + "kernel_" + str(bandwidth) + ".png", bbox_inches = "tight")

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


                    # run the model without TF
                    thetas = np.zeros((ntimes,ngenes,ngenes))

                    gadmm_batch = g_admm.G_admm_minibatch(X=X[:, None, :], K=K, pre_cov=empir_cov, batchsize = 100)
                    thetas = gadmm_batch.train(max_iters=max_iters, n_intervals=100, alpha=alpha, lamb=lamb, rho=rho, theta_init_offset=0.1)
                    np.save(file = result_dir + "thetas_" + str(bandwidth) + "_" + str(alpha) + "_" + str(lamb) + "_" + str(rho) + ".npy", arr = thetas) 
                    print("time calculating thetas: {:.2f} sec".format(time.time() - start_time))
                    del thetas
                    gadmm_batch = None
                    gc.collect()
        
# %%
