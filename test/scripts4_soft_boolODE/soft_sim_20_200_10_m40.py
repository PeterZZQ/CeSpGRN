# In[0]
import pandas as pd
import numpy as np
import torch
import torch.nn
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import sys, os
sys.path.append('../src/')

import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

import bmk_beeline as bmk
import g_glad, genie3, g_admm
import simulator_soft_ODE as soft_sim


# In[1] define functions of preprocessing, kernel bandwidth
def preprocess(counts):
    # normalize according to the library size
    libsize = np.median(np.sum(counts, axis = 1))
    counts = counts / np.sum(counts, axis = 1)[:,None] * libsize
    counts = np.log1p(counts)
    return counts

# weighted Kernel for weighted covariance matrix and weighting the losses for different time points
def kernel_band(bandwidth, ntimes, truncate = False, trun_size = 1.5):
    # bandwidth decide the shape (width), no matter the length ntimes
    t = (np.arange(ntimes)/ntimes).reshape(ntimes,1)
    tdis = np.square(pdist(t))
    mdis = 0.5 * bandwidth * np.median(tdis)
    K = squareform(np.exp(-tdis/mdis))+np.identity(ntimes)

    if truncate == True:
        cutoff = mdis * trun_size
        mask = (squareform(tdis) < cutoff).astype(int)
        K = K * mask

    return K/np.sum(K,axis=1)[:,None]


# In[2] # Simulation Setting
result_dir = "./result/"
result_path = Path(result_dir)

if not os.path.exists(result_path):
    os.makedirs(result_path)

# hypter parameters: (variables)
ngenes = 20             # number of genes
kinetic_m = 40          # mRNA transcription factor: larger m, increase regulation effect
change_stepsize = 200   # step size of graph change: smaller stepsize, increase graph changing speed
nchanges= 10            # the number of dynamic (generated/deleted) edges

# hypter parameters: (fixed)
ncells = 100            # number of independent experiments
ntfs = 10               # number of transcription factors
density = 0.1           # decide the number of edges: number of edges = (ngenes**2)*density, 1-density = sparsity
tmax = 10               # decide the number of edges: number of edges = (ngenes**2)*density

# gene expression data, sampled from multiple cells
# TODO: linear cell path > bifurcating cell path
sorted_exp, gt_adj = soft_sim.run_simulator(ncells=ncells, ngenes=ngenes, density=density, ntfs=ntfs,tmax=tmax, mode = "TF-TF&target", \
    kinetic_m=kinetic_m, nchanges=nchanges, change_stepsize=change_stepsize, integration_step_size = 0.01)


# In[2] save simulation data (gene expression, ground-truth adjacent matrices)
assert len(np.nonzero(gt_adj[0])[0]) == (ngenes**2)*density + nchanges
nedges = len(np.nonzero(gt_adj[0])[0])

print("Number of Edges:{}, Number of Nodes:{}, Density:{}".format(nedges, ngenes, nedges/(ngenes**2)))

np.save(file = result_dir + "gt_graph_m" + str(kinetic_m) + "_" + str(ngenes) + "_" + str(nedges) + "_" + str(nchanges) + "_" + str(change_stepsize) + ".npy", arr = gt_adj)
np.save(file = result_dir + "gene_exp_m" + str(kinetic_m) + "_" + str(ngenes) + "_" + str(nedges) + "_" + str(nchanges) + "_" + str(change_stepsize) + ".npy", arr = sorted_exp)


# In[3] data preprocessing
X = preprocess(sorted_exp.T)
print("Raw TimePoints: {}, no.Genes: {}".format(X.shape[0],X.shape[1]))
X = StandardScaler().fit_transform(X)

ntimes, ngenes = X.shape

# chekc the size of transcription factor
tf = list(set(np.nonzero(gt_adj[0])[0]))
print("Number of TFs:",len(tf))


# In[4] run weighted covariance ADMM
import importlib
importlib.reload(g_admm)

sample = torch.FloatTensor(X)
max_iters = 1000

for bandwidth in [0.1, 1, 10]:
    for trun_size in [1.5]:
    
        empir_cov = torch.zeros(ntimes, ngenes, ngenes)
        K = kernel_band(bandwidth, ntimes)
        K_trun = kernel_band(bandwidth, ntimes, truncate = True, trun_size = trun_size)

        # building weighted covariance matrix, output is empir_cov of the shape (ntimes, ngenes, ngenes)
        for time in range(ntimes):
            weight = torch.FloatTensor(K_trun[time, :])
            # assert torch.sum(weight) == 1

            bin_weight = torch.FloatTensor((K_trun[time, :] > 0).astype(int))
            sample_mean = torch.sum(sample * weight[:, None], dim = 0) # -- weighted sample mean
            # sample_mean = torch.sum(sample * (bin_weight/torch.sum(bin_weight))[:, None], dim = 0) # -- conventional sample mean

            norm_sample = sample - sample_mean[None, :]
            empir_cov[time] = torch.sum(torch.bmm(norm_sample[:,:,None], norm_sample[:,None,:]) * weight[:,None, None], dim=0)

        for lamb in [0.05, 0.1, 0.2, 0.25]:
            for rho in [1.7]:
                # test model without TF
                thetas_weighted = np.zeros((ntimes,ngenes,ngenes))

                for time in range(ntimes):
                    print("Est. timepoint:", time)
                    # TODO: the dual residual keep exploding, maybe check the def of dual residual and tune the parameter (in this case, reducing rh), in addition, include the mode of varying dual residual.
                    admm_single = g_admm.G_admm(X = X[:,None,:], K = K, pre_cov = empir_cov)
                    # setting from the paper over-relaxation model
                    thetas_weighted[time] = admm_single.train(t = time, max_iters = max_iters, n_intervals = 100, lamb = lamb , alpha = 2, rho = rho, theta_init_offset = 0.1)
                    
                np.save(file = result_dir + "admm_soft_m" + str(kinetic_m) + "_" + str(bandwidth) + "_" + str(lamb) + "_graph_" + str(ngenes) + "_" + str(nedges) + "_" + str(nchanges) + "_" + str(change_stepsize) + ".npy", arr = thetas_weighted)  

# In[5]