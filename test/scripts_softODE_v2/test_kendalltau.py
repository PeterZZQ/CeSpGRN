# In[]
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


def kendall_tau(xs, ys, ws = None):
    """\
    Description:
    -----------
       Calculate weighted kendall tau score, mxi, mxj, myi, myj are use to account for the missing values
    Parameters:
    -----------
        xs: the first array
        ys: the second array
        ws: the weight
    """
    n = len(xs)
    if ws is None:
        ws = np.ones(n)
    assert len(ys) == n
    assert len(ws) == n
    kt = 0
    norm = 0
    # mask array
    mx = (xs != 0)
    my = (ys != 0)
    for i in range(n):
        for j in range(n):
            if i != j:
                kt += np.sign(xs[i] - xs[j]) * np.sign(ys[i] - ys[j])
                
    return kt/n/(n-1)

# In[]
import importlib
importlib.reload(g_admm)
import scipy.stats as stats
# x1 = [12, 2, 1, 12, 2]
# # x2 = [1, 4, 7, 1, 1]
# x2 = [12, 2, 1, 12, 2]
x1 = [13, 3, 1, 12, 2]
# x2 = [1, 4, 7, 1, 1]
x2 = [13, 3, 1, 12, 2]
tau, p_value = stats.kendalltau(x1, x2)
print(tau)
kt = g_admm.weighted_kendall_tau(torch.tensor(x1), torch.tensor(x2))
print(kt)
kt = kendall_tau(np.array(x1), np.array(x2))
print(kt)

# In[]
ntimes = 1000
nsamples = 1
path = "../../data/GGM_bifurcate/"
max_iters = 500
seed = 0
use_init = "sergio"
for interval in [5]:
    for (ngenes, ntfs) in [(50, 20)]:
        result_dir = "../results_GGM/bifur_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "_" + str(seed) + "_" + use_init + "/"
        # if not os.path.exists(result_dir):
        #     os.makedirs(result_dir)
        # the data smapled from GGM is zero-mean
        X = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "_" + str(seed)  + "_" + use_init + "/expr.npy")

        # sort the genes
        print("Raw TimePoints: {}, no.Genes: {}".format(X.shape[0],X.shape[1]))
        X_pca = pca_op.fit_transform(StandardScaler().fit_transform(X))

        ###############################################
        #
        # test with the first set of hyper-parameters, without TF information
        #
        ###############################################
        print("test without TF information")
        for bandwidth in [1]:
            for truncate_param in [15]:
                # calculate the kernel function, didn't use dimension reduction
                K, K_trun = kernel.calc_kernel_neigh(X_pca, k = 5, bandwidth = bandwidth, truncate = True, truncate_param = truncate_param)

                # plot kernel function
                fig = plt.figure(figsize = (20, 7))
                axs = fig.subplots(1, 2)
                axs[0].plot(K[int(ntimes * nsamples/2), :])
                axs[1].plot(K_trun[int(ntimes * nsamples/2), :])
                fig.suptitle("kernel_" + str(bandwidth) + "_" + str(truncate_param))
                print("number of neighbor being considered: " + str(np.sum(K_trun[int(ntimes * nsamples/2), :] > 0)))
                
                # building weighted covariance matrix, output is empir_cov of the shape (ntimes * nsamples, ngenes, ngenes)
                empir_cov = g_admm.est_cov(X = X, K_trun = K_trun, weighted_kt = True)
                # not using weight, direct kendall tau, find that the empirical covariance matrix is very close to not using weight
                # empir_cov = g_admm.est_cov(X = X, K_trun = K_trun, weighted_kt = False)
                            
                # run the model
                for lamb in [0.1]:
                    start_time = time.time() 
                    # test model without TF
                    thetas = np.zeros((ntimes * nsamples, ngenes, ngenes))

                    # setting from the paper over-relaxation model
                    alpha = 2
                    rho = 1.7
                    # gadmm_batch = g_admm.G_admm_batch(X=X[:, None, :], K=K, pre_cov=empir_cov)
                    gadmm_batch = g_admm.G_admm_minibatch(X=X[:, None, :], K=K, pre_cov=empir_cov, batchsize = 100)
                    thetas = gadmm_batch.train(max_iters=max_iters, n_intervals=100, alpha=alpha, lamb=lamb, rho=rho, theta_init_offset=0.1)
                    print("time calculating thetas: {:.2f} sec".format(time.time() - start_time))

# %%
