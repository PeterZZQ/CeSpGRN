import numpy as np
import torch
import torch.nn
import matplotlib.pyplot as plt
import sys, os
sys.path.append('../../src/')

import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

import bmk_beeline as bmk
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns
import genie3


ntimes = 1000
nsamples = 1
path = "../../data/GGM_bifurcate/"
for interval in [50, 100, 200]:
    for (ngenes, ntfs) in [(20, 5), (30, 10), (50, 20),(100, 50)]:
        result_dir = "../results_GGM/bifur_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "/"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        # the data smapled from GGM is zero-mean
        X = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/expr.npy")
        X_genie = X.reshape(ntimes * nsamples, ngenes)
        # genie_theta of the shape (ntimes, ngenes, ngenes)
        genie_theta = genie3.GENIE3(X_genie, gene_names=["gene_" + str(x) for x in range(ngenes)], regulators='all',tree_method='RF',K='sqrt',ntrees=1000,nthreads=1)
        genie_theta = np.repeat(genie_theta[None, :, :],ntimes,axis=0)
        np.save(file = result_dir + "theta_genie.npy", arr = genie_theta)

        # the data smapled from GGM is zero-mean
        X = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/expr.npy")
        X_genie = X.reshape(ntimes * nsamples, ngenes)
        # genie_theta of the shape (ntimes, ngenes, ngenes)
        genie_theta = genie3.GENIE3(X_genie, gene_names=["gene_" + str(x) for x in range(ngenes)], regulators=["gene_" + str(x) for x in range(ntfs)],tree_method='RF',K='sqrt',ntrees=1000,nthreads=1)
        genie_theta = np.repeat(genie_theta[None, :, :],ntimes,axis=0)
        np.save(file = result_dir + "theta_genie_tf.npy", arr = genie_theta)


ntimes = 250
nsamples = 10
path = "../../data/GGM_bifurcate/"
for interval in [10, 25, 50]:
    for (ngenes, ntfs) in [(20, 5), (30, 10), (50, 20),(100, 50)]:
        result_dir = "../results_GGM/bifur_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "/"
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        # the data smapled from GGM is zero-mean
        X = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/expr.npy")
        X_genie = X.reshape(ntimes * nsamples, ngenes)
        # genie_theta of the shape (ntimes, ngenes, ngenes)
        genie_theta = genie3.GENIE3(X_genie, gene_names=["gene_" + str(x) for x in range(ngenes)], regulators='all',tree_method='RF',K='sqrt',ntrees=1000,nthreads=1)
        genie_theta = np.repeat(genie_theta[None, :, :],ntimes,axis=0)
        np.save(file = result_dir + "theta_genie.npy", arr = genie_theta)

        # the data smapled from GGM is zero-mean
        X = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/expr.npy")
        X_genie = X.reshape(ntimes * nsamples, ngenes)
        # genie_theta of the shape (ntimes, ngenes, ngenes)
        genie_theta = genie3.GENIE3(X_genie, gene_names=["gene_" + str(x) for x in range(ngenes)], regulators=["gene_" + str(x) for x in range(ntfs)],tree_method='RF',K='sqrt',ntrees=1000,nthreads=1)
        genie_theta = np.repeat(genie_theta[None, :, :],ntimes,axis=0)
        np.save(file = result_dir + "theta_genie_tf.npy", arr = genie_theta)