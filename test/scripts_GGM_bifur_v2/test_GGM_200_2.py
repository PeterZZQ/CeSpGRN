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

# In[1] test with the first set of hyper-parameters
if __name__ == "__main__":
    assert len(sys.argv) == 5
    use_init = sys.argv[1]
    assert use_init in ["sergio", "random"]
    seed = eval(sys.argv[2])
    bandwidth = eval(sys.argv[3])
    truncate_param = eval(sys.argv[4])


    ntimes = 1000
    nsamples = 1
    path = "../../data/GGM_bifurcate/"
    max_iters = 500
    for interval in [25]:
        for (ngenes, ntfs) in [(200, 20)]:
            result_dir = "../results_GGM/bifur_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "_" + str(seed) + "_" + use_init + "/"
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            # the data smapled from GGM is zero-mean
            X = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "_" + str(seed)  + "_" + use_init + "/expr.npy")

            # sort the genes
            print("Raw TimePoints: {}, no.Genes: {}".format(X.shape[0],X.shape[1]))
            X_pca = pca_op.fit_transform(StandardScaler().fit_transform(X))

            sample = torch.FloatTensor(X).to(device)
            ###############################################
            #
            # test with the first set of hyper-parameters, without TF information
            #
            ###############################################
            print("test without TF information")
            for bandwidth in [bandwidth]:
                for truncate_param in [truncate_param]:
                    start_time = time.time()
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
                    for t in range(ntimes * nsamples):
                        weight = torch.FloatTensor(K_trun[t, :]).to(device)
                        # assert torch.sum(weight) == 1

                        bin_weight = torch.FloatTensor((K_trun[t, :] > 0).astype(np.int))
                        sample_mean = torch.sum(sample * weight[:, None], dim = 0)
                        # sample_mean = torch.sum(sample * (bin_weight/torch.sum(bin_weight))[:, None], dim = 0)

                        norm_sample = sample - sample_mean[None, :]
                        empir_cov[t] = torch.sum(torch.bmm(norm_sample[:,:,None], norm_sample[:,None,:]) * weight[:,None, None], dim=0)
                    print("time calculating the kernel function: {:.2f} sec".format(time.time() - start_time))
                    
                                
                    # run the model
                    for lamb in [0.001, 0.01, 0.1]:
                        start_time = time.time() 
                        # test model without TF
                        thetas = np.zeros((ntimes * nsamples,ngenes,ngenes))

                        # setting from the paper over-relaxation model
                        alpha = 2
                        rho = 1.7
                        # gadmm_batch = g_admm.G_admm_batch(X=X[:, None, :], K=K, pre_cov=empir_cov)
                        gadmm_batch = g_admm.G_admm_minibatch(X=X[:, None, :], K=K, pre_cov=empir_cov, batchsize = 100)
                        thetas = gadmm_batch.train(max_iters=max_iters, n_intervals=100, alpha=alpha, lamb=lamb, rho=rho, theta_init_offset=0.1)
                        np.save(file = result_dir + "thetas_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_0.npy", arr = thetas) 
                        print("time calculating thetas: {:.2f} sec".format(time.time() - start_time))
                        del thetas
                        gadmm_batch = None
                        gc.collect()
            

# %%
