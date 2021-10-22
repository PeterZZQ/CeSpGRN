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



if __name__ == "__main__":
    assert len(sys.argv) == 2
    seed = eval(sys.argv[1])

    ntimes = 1000
    nsamples = 1
    path = "../../data/GGM_bifurcate/"

    use_init = "sergio"

    for interval in [5, 25, 100]:
        for (ngenes, ntfs) in [(50, 20), (200, 20)]:
            result_dir = "../results_GGM/bifur_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "_" + str(seed) + "_" + use_init + "/"
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            # the data smapled from GGM is zero-mean
            X = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "_" + str(seed) + "_" + use_init + "/expr.npy")
            X_genie = X.reshape(ntimes * nsamples, ngenes)
            # genie_theta of the shape (ntimes, ngenes, ngenes)
            genie_theta = genie3.GENIE3(X_genie, gene_names=["gene_" + str(x) for x in range(ngenes)], regulators='all',tree_method='RF',K='sqrt',ntrees=1000,nthreads=1)
            genie_theta = np.repeat(genie_theta[None, :, :],ntimes,axis=0)
            np.save(file = result_dir + "theta_genie.npy", arr = genie_theta)

            # the data smapled from GGM is zero-mean
            X = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "_" + str(seed) + "_" + use_init + "/expr.npy")
            X_genie = X.reshape(ntimes * nsamples, ngenes)
            # genie_theta of the shape (ntimes, ngenes, ngenes)
            genie_theta = genie3.GENIE3(X_genie, gene_names=["gene_" + str(x) for x in range(ngenes)], regulators=["gene_" + str(x) for x in range(ntfs)],tree_method='RF',K='sqrt',ntrees=1000,nthreads=1)
            genie_theta = np.repeat(genie_theta[None, :, :],ntimes,axis=0)
            np.save(file = result_dir + "theta_genie_tf.npy", arr = genie_theta)

    for interval in [5, 25, 100]:
        for (ngenes, ntfs) in [(50, 20), (200, 20)]:
            result_dir = "../results_GGM/bifur_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "_" + str(seed) + "_" + use_init + "/"
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            # the data smapled from GGM is zero-mean
            X = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "_" + str(seed) + "_" + use_init + "/expr.npy")

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

            # the data smapled from GGM is zero-mean
            X = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "_" + str(seed) + "_" + use_init + "/expr.npy")
            
            interval_size = 100

            genie_thetas = []
            for i in range(np.int(ntimes/interval_size)):
                if i != np.int(ntimes/interval_size) - 1:
                    X_genie = X[i*interval_size:(i+1)*interval_size, :]
                else:
                    X_genie = X[i*interval_size:, :]
                # genie_theta of the shape (ntimes, ngenes, ngenes)
                genie_theta = genie3.GENIE3(X_genie, gene_names=["gene_" + str(x) for x in range(ngenes)], regulators=["gene_" + str(x) for x in range(ntfs)],tree_method='RF',K='sqrt',ntrees=1000,nthreads=1)
                genie_theta = np.repeat(genie_theta[None, :, :], X_genie.shape[0], axis=0)
                genie_thetas.append(genie_theta.copy())

            np.save(file = result_dir + "theta_genie_dyn_tf.npy", arr = np.concatenate(genie_thetas, axis = 0))


    use_init = "random"

    for interval in [5, 25, 100]:
        for (ngenes, ntfs) in [(50, 20), (200, 20)]:
            result_dir = "../results_GGM/bifur_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "_" + str(seed) + "_" + use_init + "/"
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            # the data smapled from GGM is zero-mean
            X = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "_" + str(seed) + "_" + use_init + "/expr.npy")
            X_genie = X.reshape(ntimes * nsamples, ngenes)
            # genie_theta of the shape (ntimes, ngenes, ngenes)
            genie_theta = genie3.GENIE3(X_genie, gene_names=["gene_" + str(x) for x in range(ngenes)], regulators='all',tree_method='RF',K='sqrt',ntrees=1000,nthreads=1)
            genie_theta = np.repeat(genie_theta[None, :, :],ntimes,axis=0)
            np.save(file = result_dir + "theta_genie.npy", arr = genie_theta)

            # the data smapled from GGM is zero-mean
            X = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "_" + str(seed) + "_" + use_init + "/expr.npy")
            X_genie = X.reshape(ntimes * nsamples, ngenes)
            # genie_theta of the shape (ntimes, ngenes, ngenes)
            genie_theta = genie3.GENIE3(X_genie, gene_names=["gene_" + str(x) for x in range(ngenes)], regulators=["gene_" + str(x) for x in range(ntfs)],tree_method='RF',K='sqrt',ntrees=1000,nthreads=1)
            genie_theta = np.repeat(genie_theta[None, :, :],ntimes,axis=0)
            np.save(file = result_dir + "theta_genie_tf.npy", arr = genie_theta)

    for interval in [5, 25, 100]:
        for (ngenes, ntfs) in [(50, 20), (200, 20)]:
            result_dir = "../results_GGM/bifur_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "_" + str(seed) + "_" + use_init + "/"
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            # the data smapled from GGM is zero-mean
            X = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "_" + str(seed) + "_" + use_init + "/expr.npy")

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

            # the data smapled from GGM is zero-mean
            X = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "_" + str(seed) + "_" + use_init + "/expr.npy")
            
            interval_size = 100

            genie_thetas = []
            for i in range(np.int(ntimes/interval_size)):
                if i != np.int(ntimes/interval_size) - 1:
                    X_genie = X[i*interval_size:(i+1)*interval_size, :]
                else:
                    X_genie = X[i*interval_size:, :]
                # genie_theta of the shape (ntimes, ngenes, ngenes)
                genie_theta = genie3.GENIE3(X_genie, gene_names=["gene_" + str(x) for x in range(ngenes)], regulators=["gene_" + str(x) for x in range(ntfs)],tree_method='RF',K='sqrt',ntrees=1000,nthreads=1)
                genie_theta = np.repeat(genie_theta[None, :, :], X_genie.shape[0], axis=0)
                genie_thetas.append(genie_theta.copy())

            np.save(file = result_dir + "theta_genie_dyn_tf.npy", arr = np.concatenate(genie_thetas, axis = 0))
