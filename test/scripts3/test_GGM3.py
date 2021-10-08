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

# In[1] test with the first set of hyper-parameters
ntimes = 1000
path = "../../data/GGM_changing_mean/"
max_iters = 2000
truncate_param = 7
for interval in [200]:
    # for (ngenes, ntfs) in [(20, 5), (30, 10), (50, 20), (100, 50)]:
    for (ngenes, ntfs) in [(50, 20), (100, 50)]:
        result_dir = "../results/GGM_changing_mean_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "/"
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
        ###############################################
        #
        # test with the first set of hyper-parameters, without TF information
        #
        ###############################################
        if ngenes != 50:
            print("test without TF information")
            for bandwidth in [0.1]:
                start_time = time.time()
                empir_cov = torch.zeros(ntimes, ngenes, ngenes)
                # calculate the kernel function
                K, K_trun = kernel.calc_kernel(X, k = 5, bandwidth = bandwidth, truncate = True, truncate_param = truncate_param)

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
                
                start_time = time.time()                    
                # run the model
                for lamb in [0.01]:
                    # test model without TF
                    thetas = np.zeros((ntimes,ngenes,ngenes))

                    # setting from the paper over-relaxation model
                    alpha = 2
                    rho = 1.7
                    # gadmm_batch = g_admm.G_admm_batch(X=X[:, None, :], K=K, pre_cov=empir_cov)
                    gadmm_batch = g_admm.G_admm_minibatch(X=X[:, None, :], K=K, pre_cov=empir_cov, batchsize = 100)
                    thetas = gadmm_batch.train(max_iters=max_iters, n_intervals=100, alpha=alpha, lamb=lamb, rho=rho, theta_init_offset=0.1)
                    np.save(file = result_dir + "thetas_" + str(bandwidth) + "_" + str(alpha) + "_" + str(lamb) + "_" + str(rho) + ".npy", arr = thetas) 
                    print("time calculating thetas: {:.2f} sec".format(time.time() - start_time))
                del thetas
                gadmm_batch = None
                gc.collect()

            ###############################################
            #
            # test with TF information
            #
            ###############################################
            print("test with TF information")
            for bandwidth in [0.1]:
                start_time = time.time()
                empir_cov = torch.zeros(ntimes, ngenes, ngenes)
                K, K_trun = kernel.calc_kernel(X, k = 5, bandwidth = bandwidth, truncate = True, truncate_param = truncate_param)

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
                    # gadmm_batch = g_admm.G_admm_batch(X=X[:, None, :], K=K, pre_cov=empir_cov, TF=np.arange(ntfs))
                    gadmm_batch = g_admm.G_admm_minibatch(X=X[:, None, :], K=K, pre_cov=empir_cov, batchsize = 100, TF=np.arange(ntfs))
                    thetas = gadmm_batch.train(max_iters=max_iters, n_intervals=100, alpha=alpha, lamb=lamb, rho=rho, beta=100, theta_init_offset=0.1)
                    np.save(file = result_dir + "thetas_" + str(bandwidth) + "_" + str(alpha) + "_" + str(lamb) + "_" + str(rho) + "_tfs.npy", arr = thetas) 
                    print("time calculating thetas: {:.2f} sec".format(time.time() - start_time))
                del thetas
                gadmm_batch = None
                gc.collect()


        ###############################################
        #
        # test with the second set of hyper-parameters, without TF information
        #
        ###############################################
        print("test without TF information")
        for bandwidth in [0.1]:
            start_time = time.time()
            empir_cov = torch.zeros(ntimes, ngenes, ngenes)
            # calculate the kernel function
            K, K_trun = kernel.calc_kernel(X, k = 5, bandwidth = bandwidth, truncate = True, truncate_param = truncate_param)

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
                alpha = 1
                rho = None
                # gadmm_batch = g_admm.G_admm_batch(X=X[:, None, :], K=K, pre_cov=empir_cov)
                gadmm_batch = g_admm.G_admm_minibatch(X=X[:, None, :], K=K, pre_cov=empir_cov, batchsize = 100)
                thetas = gadmm_batch.train(max_iters=max_iters, n_intervals=100, alpha=alpha, lamb=lamb, rho=rho, theta_init_offset=0.1)
                np.save(file = result_dir + "thetas_" + str(bandwidth) + "_" + str(alpha) + "_" + str(lamb) + "_" + str(rho) + ".npy", arr = thetas) 
                print("time calculating thetas: {:.2f} sec".format(time.time() - start_time))
            del thetas
            gadmm_batch = None
            gc.collect()

        ###############################################
        #
        # test with TF information
        #
        ###############################################
        print("test with TF information")
        for bandwidth in [0.1]:
            start_time = time.time()
            empir_cov = torch.zeros(ntimes, ngenes, ngenes)
            K, K_trun = kernel.calc_kernel(X, k = 5, bandwidth = bandwidth, truncate = True, truncate_param = truncate_param)

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
                alpha = 1
                rho = None
                # gadmm_batch = g_admm.G_admm_batch(X=X[:, None, :], K=K, pre_cov=empir_cov, TF=np.arange(ntfs))
                gadmm_batch = g_admm.G_admm_minibatch(X=X[:, None, :], K=K, pre_cov=empir_cov, batchsize = 100, TF=np.arange(ntfs))
                thetas = gadmm_batch.train(max_iters=max_iters, n_intervals=100, alpha=alpha, lamb=lamb, rho=rho, beta=100, theta_init_offset=0.1)
                np.save(file = result_dir + "thetas_" + str(bandwidth) + "_" + str(alpha) + "_" + str(lamb) + "_" + str(rho) + "_tfs.npy", arr = thetas) 
                print("time calculating thetas: {:.2f} sec".format(time.time() - start_time))
            del thetas
            gadmm_batch = None
            gc.collect()
# %%
