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

# In[1]
def preprocess(counts):
    # normalize according to the library size
    libsize = np.median(np.sum(counts, axis = 1))
    counts = counts / np.sum(counts, axis = 1)[:,None] * libsize
    counts = np.log1p(counts)
    return counts


# In[2] load simulation data (gene expression, ground-truth adjacent matrices)
result_dir = "./result/"

# hypter parameters: (variables)
ngenes = 20             # number of genes
kinetic_m = 40          # mRNA transcription factor: larger m, increase regulation effect
change_stepsize = 200   # step size of graph change: smaller stepsize, increase graph changing speed
nchanges= 2             # the number of dynamic (generated/deleted) edges

# hypter parameters: (fixed)
ncells = 100            # number of independent experiments
ntfs = 10               # number of transcription factors
density = 0.1           # decide the number of edges: number of edges = (ngenes**2)*density, 1-density = sparsity
tmax = 10               # decide the number of edges: number of edges = (ngenes**2)*density

nsamples = 100          # set sample size for genie3
nedges = int((ngenes**2)*density + nchanges)

gt_path = result_dir + "gt_graph_m" + str(kinetic_m) + "_" + str(ngenes) + "_" + str(nedges) + "_" + str(nchanges) + "_" + str(change_stepsize) + ".npy"
ex_path = result_dir + "gene_exp_m" + str(kinetic_m) + "_" + str(ngenes) + "_" + str(nedges) + "_" + str(nchanges) + "_" + str(change_stepsize) + ".npy"

sorted_exp = np.load(ex_path)
gt_adj = np.load(gt_path)

# preprocessing
X = preprocess(sorted_exp.T)
print("Raw TimePoints: {}, no.Genes: {}".format(X.shape[0],X.shape[1]))
X = StandardScaler().fit_transform(X)
ntimes, ngenes = X.shape


# In[3]
import importlib
importlib.reload(genie3)

nseg = ntimes//nsamples
X_genies = X.reshape(nseg,nsamples,ngenes)
genie_thetas = np.zeros((nseg,ngenes,ngenes))

# Run Genie3 (weighted = False, Single Estimation)
for i in range(nseg):
    genie_thetas[i,:,:] = genie3.GENIE3(X_genies[i], gene_names=None, regulators='all',tree_method='RF',K='sqrt',ntrees=1000,nthreads=1)

genie_res = np.repeat(genie_thetas, nsamples, axis = 0)
np.save(file = result_dir + "genie_m" + str(kinetic_m) + "_" + str(nsamples) + "_" + str(nseg) + "_graph_" + str(ngenes) + "_" + str(nedges) + "_" + str(nchanges) + "_" + str(change_stepsize) + ".npy", arr = genie_res)

# In[6]