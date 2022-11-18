# In[0]
from math import trunc
import sys, os
sys.path.append('../../src/')

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import time
import gc
import torch
import torch.nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from umap import UMAP

import bmk_beeline as bmk
import genie3, g_admm, kernel
import warnings
warnings.filterwarnings("ignore")
from scipy.sparse import csr_matrix, save_npz

from os.path import exists

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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

# In[1]
path = "../../data/mESC/"
result_dir = "../results_mESC_invprec/"

# counts_96 is the raw count with 96 genes, counts_96_norm is the normalized count with 96 genes.
counts = pd.read_csv(path + "counts_44.csv", index_col = 0).values
annotation = pd.read_csv(path + "anno.csv", index_col = 0)
genes = pd.read_csv(path + "counts_44.csv", index_col = 0).columns.values
tfs = ['Pou5f1', 'Nr5a2', 'Sox2', 'Sall4', 'Otx2', 'Esrrb', 'Stat3','Tcf7', 'Nanog', 'Etv5']
tf_ids = np.array([np.where(genes == x)[0][0] for x in tfs])


ncells, ngenes = counts.shape
# assert ncells == 8 * 120
assert ngenes == 44
print("Raw TimePoints: {}, no.Genes: {}".format(counts.shape[0],counts.shape[1]))

# normalization step
libsize = np.median(np.sum(counts, axis = 1))
counts_norm = counts /(np.sum(counts, axis = 1, keepdims = True) + 1e-6) * libsize
# the distribution of the original count is log-normal distribution, conduct log transform
counts_norm = np.log1p(counts_norm)

# check distribution
fig = plt.figure(figsize = (10,7))
ax = fig.add_subplot()
_ = ax.hist(counts.reshape(-1), bins = 20)
ax.set_yscale('log')
fig = plt.figure(figsize = (10,7))
ax = fig.add_subplot()
_ = ax.hist(counts_norm.reshape(-1), bins = 20)

# check the time evolution of the un-preprocessed gene expression data 
expr =  counts.T[:20,:]
fig = plt.figure(figsize = (15,10))
ax = fig.add_subplot()
for gene_expr in expr:
    ax.scatter(np.arange(gene_expr.shape[0]), gene_expr)
# ax.set_yscale("log")
ax.set_title("Gene expression")

expr =  counts_norm.T[:20,:]
fig = plt.figure(figsize = (15,10))
ax = fig.add_subplot()
for gene_expr in expr:
    ax.scatter(np.arange(gene_expr.shape[0]), gene_expr)
ax.set_title("Gene expression normalized")


pca_op = PCA(n_components = 20)
umap_op = UMAP(n_components = 2, min_dist = 0.8, random_state = 0)

X_pca = pca_op.fit_transform(counts_norm)
X_umap = umap_op.fit_transform(counts_norm)

fig = plt.figure(figsize  = (10,7))
ax = fig.add_subplot()
for i in np.sort(np.unique(annotation.values.squeeze())):
    idx = np.where(annotation.values.squeeze() == i)
    ax.scatter(X_pca[idx, 0], X_pca[idx, 1], label = i)

ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False)
fig.savefig(result_dir + "X_pca.png", bbox_inches = "tight")

fig = plt.figure(figsize  = (10,7))
ax = fig.add_subplot()
for i in np.sort(np.unique(annotation.values.squeeze())):
    idx = np.where(annotation.values.squeeze() == i)
    ax.scatter(X_umap[idx, 0], X_umap[idx, 1], label = i)

ax.set_xlabel("UMAP1")
ax.set_ylabel("UMAP2")
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False)
fig.savefig(result_dir + "X_umap.png", bbox_inches = "tight")

# X = torch.FloatTensor(counts).to(device)

# In[2] ADMM
import importlib 
importlib.reload(g_admm)
# hyper-parameter
bandwidths = [0.1, 1, 10]
truncate_params = [400, 1000, 2000]
# lambs = [0, 0.001, 0.01, 0.05, 0.1]
lambs = [0.01]
truncate_params = [100]
# assert len(sys.argv) == 3
# for bandwidth in [bandwidths[eval(sys.argv[1])]]:
#     for truncate_param in [truncate_params[eval(sys.argv[2])]]:

for bandwidth in [bandwidths[1]]:
    for truncate_param in [truncate_params[0]]:
        # calculate empirical covariance matrix
        start_time = time.time()
        empir_cov = torch.zeros(ncells, ngenes, ngenes)
        # calculate the kernel function
        K, K_trun = kernel.calc_kernel_neigh(X_pca, k = 5, bandwidth = bandwidth, truncate = True, truncate_param = truncate_param)

        # # plot kernel function
        # fig = plt.figure(figsize = (20, 7))
        # axs = fig.subplots(1, 2)
        # axs[0].plot(K[int(ncells/2), :])
        # axs[1].plot(K_trun[int(ncells/2), :])
        # fig.suptitle("kernel_" + str(bandwidth) + "_" + str(truncate_param))
        # fig.savefig(result_dir + "plots/kernel_" + str(bandwidth) + "_" + str(truncate_param) + ".png", bbox_inches = "tight")

        # fig = plt.figure(figsize  = (10,7))
        # ax = fig.add_subplot()
        # ax.scatter(X_pca[:, 0], X_pca[:, 1], c = K[int(ncells/2), :])

        # ax.set_xlabel("PCA1")
        # ax.set_ylabel("PCA2")
        # ax.set_title("kernel_" + str(bandwidth))
        # fig.savefig(result_dir + "plots/kernel_" + str(bandwidth) + "_pca.png", bbox_inches = "tight")

        # fig = plt.figure(figsize  = (10,7))
        # ax = fig.add_subplot()
        # ax.scatter(X_umap[:, 0], X_umap[:, 1], c = K[int(ncells/2), :])

        # ax.set_xlabel("UMAP1")
        # ax.set_ylabel("UMAP1")
        # ax.set_title("kernel_" + str(bandwidth))
        # fig.savefig(result_dir + "plots/kernel_" + str(bandwidth) + "_umap.png", bbox_inches = "tight")


        print("number of neighbor being considered: " + str(np.sum(K_trun[int(ncells/2), :] > 0)))

        # building weighted covariance matrix, output is empir_cov of the shape (ncells, ngenes, ngenes)
        empir_cov = g_admm.est_cov(X = counts_norm, K_trun = K_trun, weighted_kt = True)
        # empir_cov = g_admm.est_cov_para(X = counts_norm, K_trun = K_trun, weighted_kt = True, ncpus = 2)
        np.save(result_dir + "cov_44/cov_" + str(bandwidth) + "_" + str(truncate_param) + "_norm.npy", empir_cov)

        for lamb in lambs:
            for beta in [0]:
                alpha = 2
                rho = 1.7
                max_iters = 1000

                # if exists(result_dir + "thetas_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_kt.npy"):
                #     continue 
                # else:

                start_time = time.time() 
                # test model without TF
                gadmm_batch = g_admm.G_admm_minibatch(X=counts[:, None, :], K=K, pre_cov=empir_cov, batchsize = 120, TF = tf_ids, device = device)
                w_empir_cov = gadmm_batch.w_empir_cov.detach().cpu().numpy()
                thetas = gadmm_batch.train(max_iters=max_iters, n_intervals=100, alpha=alpha, lamb=lamb, rho=rho, theta_init_offset=0.1, beta = beta)
                # NOTE: transform the precision into the conditional independence matrix
                Gs = g_admm.construct_weighted_G(thetas, ncpus = 2)

                np.save(file = result_dir + "thetas_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_"+ str(beta) +"_kt.npy", arr = thetas) 
                np.save(file = result_dir + "thetas_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_"+ str(beta) +"_kt_cond.npy", arr = Gs) 
                print("time calculating thetas: {:.2f} sec".format(time.time() - start_time))


                loss_lkl = gadmm_batch.neg_lkl_loss(torch.FloatTensor(thetas).cuda(), torch.FloatTensor(w_empir_cov).cuda()).sum().item()
                print("negative likelihood loss for the inferred GRN: {:.4e}".format(loss_lkl))
                # the loss should be similar to gadmm without lamb, the distribution should also be similar
                precision = np.concatenate([np.linalg.pinv(x)[None,:,:] for x in w_empir_cov], axis = 0)
                loss_lkl = gadmm_batch.neg_lkl_loss(torch.FloatTensor(precision).cuda(), torch.FloatTensor(w_empir_cov).cuda()).sum().item()
                print("negative likelihood loss for the inverse covariance: {:.4e}".format(loss_lkl))
                
                # baseline, just diagonal matrix have a smaller negative likelihood
                torch.manual_seed(0)
                theta_rand = torch.zeros(ngenes, ngenes)
                theta_rand = torch.mm(theta_rand, theta_rand.t())
                theta_rand.add_(torch.eye(ngenes))
                thetas_rand = torch.cat([theta_rand[None, :, :]] * thetas.shape[0], dim = 0)
                loss_lkl = gadmm_batch.neg_lkl_loss(thetas_rand.cuda(), torch.FloatTensor(w_empir_cov).cuda()).sum().item()
                print("negative likelihood loss for the diagonal matrix: {:.4e}".format(loss_lkl))

                ###################################################################################################
                #
                # Check graph weight distribution
                #
                ###################################################################################################

                # remove diagonal values
                for t in range(w_empir_cov.shape[0]):
                    np.fill_diagonal(w_empir_cov[t, :, :], 0)
                for t in range(precision.shape[0]):
                    np.fill_diagonal(precision[t, :, :], 0)
                for t in range(thetas.shape[0]):
                    np.fill_diagonal(thetas[t, :, :], 0)

                fig = plt.figure(figsize = (14,10))
                ax = fig.subplots(nrows = 2, ncols = 2)
                _ = ax[0, 0].hist(w_empir_cov.reshape(-1), bins = 30)
                ax[0, 0].set_title("weighted covariance matrix")

                _ = ax[0,1].hist(precision.reshape(-1), bins = 30)
                ax[0,1].set_title("precision matrix")  

                _ = ax[1, 0].hist(thetas.reshape(-1), bins = 30)
                ax[1, 0].set_title("inferred GRN")

                # Gs are removed of diagonal values
                _ = ax[1, 1].hist(Gs.reshape(-1), bins = 30)
                ax[1, 1].set_title("conditional covariance matrix")
                fig.savefig(result_dir + "plots/dens_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_" + str(beta) + ".png", bbox_inches = "tight")

                thetas = thetas.reshape(thetas.shape[0], -1)
                thetas_pca = pca_op.fit_transform(thetas)
                thetas_umap = umap_op.fit_transform(thetas)

                fig = plt.figure(figsize  = (10,7))
                ax = fig.add_subplot()
                for i in np.sort(np.unique(annotation.values.squeeze())):
                    idx = np.where(annotation.values.squeeze() == i)
                    ax.scatter(thetas_pca[idx, 0], thetas_pca[idx, 1], label = i, s = 10)
                ax.set_xlabel("PCA1")
                ax.set_ylabel("PCA2")
                ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False, markerscale = 3)
                ax.set_title("bandwidth: " + str(bandwidth) + ", truncate_param: " + str(truncate_param) + ", lamb: " + str(lamb))
                fig.savefig(result_dir + "plots/thetas_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_" + str(beta) + "_pca.png", bbox_inches = "tight")

                fig = plt.figure(figsize  = (10,7))
                ax = fig.add_subplot()
                for i in np.sort(np.unique(annotation.values.squeeze())):
                    idx = np.where(annotation.values.squeeze() == i)
                    ax.scatter(thetas_umap[idx, 0], thetas_umap[idx, 1], label = i, s = 10)
                ax.set_xlabel("UMAP1")
                ax.set_ylabel("UMAP2")
                ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False, markerscale = 3)
                ax.set_title("bandwidth: " + str(bandwidth) + ", truncate_param: " + str(truncate_param) + ", lamb: " + str(lamb))
                fig.savefig(result_dir + "plots/thetas_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_" + str(beta) + "_umap.png", bbox_inches = "tight")

                del thetas
                gadmm_batch = None
                gc.collect()


# In[]
# GENIE3
# genie_theta = genie3.GENIE3(counts, gene_names=[x for x in genes], regulators='all',tree_method='RF',K='sqrt',ntrees=1000,nthreads=1)
# genie_theta = np.repeat(genie_theta[None, :, :], ncells, axis=0)
# np.save(file = result_dir + "theta_genie.npy", arr = genie_theta)

# genie_theta = genie3.GENIE3(counts, gene_names=[x for x in genes], regulators = tfs, tree_method='RF',K='sqrt',ntrees=1000,nthreads=1)
# genie_theta = np.repeat(genie_theta[None, :, :], ncells, axis=0)
# np.save(file = result_dir + "theta_genie_tfs.npy", arr = genie_theta)


# In[]
# import importlib
# importlib.reload(g_admm)
# # Single GGM
# lamb = 0.005
# alpha = 2
# rho = 1.7
# max_iters = 1000

# gadmm_batch = g_admm.G_admm_batch(X=counts[None, :, :], K=torch.FloatTensor([1])[:, None], pre_cov=None)
# thetas = gadmm_batch.train(max_iters=max_iters, n_intervals=100, alpha=alpha, lamb=lamb, rho=rho, theta_init_offset=0.1)
# np.save(file = result_dir + "theta_static.npy", arr = thetas)

# %%
