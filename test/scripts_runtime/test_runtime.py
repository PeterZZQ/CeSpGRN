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
kt = True
interval = 5
seeds = [0,1,2]
gene_tfs = [(30, 20), (50, 20), (100, 20), (200, 20), (300,20)]

bandwidth = 1
truncate_param = 30
lamb = 0.05
alpha = 2
rho = 1.7

ntimes = 1000
nsamples = 1
path = "../../data/test_runtime/"
max_iters = 500

runtime = pd.DataFrame(data = np.zeros((1,2)), columns = ["empirical covariance", "theta estimate"],)

for (ngenes, ntfs) in [gene_tfs[eval(sys.argv[1])]]:
    for seed in [seeds[eval(sys.argv[2])]]:
        result_dir = "../results_runtime/" + str(ngenes) + "_" + str(seed) + "/"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        # the data smapled from GGM is zero-mean
        X = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "_" + str(seed)+ "_sergio/expr.npy")
        print("ngenes: " + str(ngenes) + ", seed: " + str(seed))

        # sort the genes
        X_pca = pca_op.fit_transform(StandardScaler().fit_transform(X))

        ###############################################
        #
        # CeSpGRN
        #
        ###############################################

        empir_cov = torch.zeros(ntimes * nsamples, ngenes, ngenes)
        # calculate the kernel function, didn't use dimension reduction
        K, K_trun = kernel.calc_kernel_neigh(X_pca, k = 5, bandwidth = bandwidth, truncate = True, truncate_param = truncate_param)

        # plot kernel function
        fig = plt.figure(figsize = (20, 7))
        axs = fig.subplots(1, 2)
        axs[0].plot(K[int(ntimes * nsamples/2), :])
        axs[1].plot(K_trun[int(ntimes * nsamples/2), :])
        fig.suptitle("kernel_" + str(bandwidth) + "_" + str(truncate_param))
        fig.savefig(result_dir + "kernel_" + str(bandwidth) + "_" + str(truncate_param) + ".png", bbox_inches = "tight")
        print("number of neighbor being considered: " + str(np.sum(K_trun[int(ntimes * nsamples/2), :] > 0)))

        # building weighted covariance matrix, output is empir_cov of the shape (ntimes * nsamples, ngenes, ngenes)
        start_time = time.time()
        if kt:
            empir_cov = g_admm.est_cov(X = X, K_trun = K_trun, weighted_kt = True)
        else:
            empir_cov = cal_cov(X = X, K_trun = K_trun)
        end_time = time.time()
        print("time cost (estimating empirical covariance matrix): {:.2f} sec".format(end_time - start_time))
        runtime.iloc[0,0] = end_time - start_time
        
                    
        # run the model
        start_time = time.time() 
        # test model without TF
        thetas = np.zeros((ntimes * nsamples,ngenes,ngenes))
        gadmm_batch = g_admm.G_admm_minibatch(X=X[:, None, :], K=K, pre_cov=empir_cov, batchsize = 100)
        thetas = gadmm_batch.train(max_iters=max_iters, n_intervals=500, alpha=alpha, lamb=lamb, rho=rho, theta_init_offset=0.1)
        end_time = time.time()
        print("time cost (estimating theta): {:.2f} sec".format(end_time - start_time))
        runtime.iloc[0,1] = end_time - start_time
        
        if kt:
            np.save(file = result_dir + "thetas_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_kt.npy", arr = thetas) 
        else:
            np.save(file = result_dir + "thetas_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + ".npy", arr = thetas) 
        
        del thetas
        gadmm_batch = None
        gc.collect()

        runtime.to_csv(result_dir + "runtime.csv")
        

