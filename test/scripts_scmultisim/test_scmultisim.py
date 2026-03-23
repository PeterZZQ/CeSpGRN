# In[]
import sys, os
sys.path.append('../../src/')
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from umap import UMAP
import g_admm, kernel, genie3, bmk, cov
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import seaborn as sns
plt.rcParams.update(plt.rcParamsDefault)

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)
plt.rcParams["font.size"] = 20

PROJECT_DIR = "/localscratch/ziqi/CeSpGRN/"

# In[] 
# -------------------------------------
#
# hyper-parameter
#
# -------------------------------------

truncate_param = 100
lamb_list = [0.01, 0.05, 0.1, 0.5]
bandwidths = [0.01, 0.1, 1, 10]

# 1. ATAC
beta = 1
use_tf = False

# 2. TF
# beta = 1
# use_tf = True

# # 3. Original
# beta = 0
# use_tf = False

# In[]
dataset = "simulated_8000_20_10_100_0.1_2_0_4"
path = PROJECT_DIR + f"data/scMultiSim/{dataset}/"

result_dir = PROJECT_DIR + f"test/results_scmultisim/{dataset}/"

# original
# if use_tf & (beta > 0):
#     cespgrn_dir = result_dir + f"raw_beta_{beta}_tf/"
# elif beta > 0:
#     cespgrn_dir = result_dir + f"raw_beta_{beta}_atac/"
# else:
#     cespgrn_dir = result_dir + f"raw_beta_{beta}/"

# with sequence normalization
if use_tf & (beta > 0):
    cespgrn_dir = result_dir + "seqnorm/" + f"raw_beta_{beta}_tf/"
elif beta > 0:
    cespgrn_dir = result_dir + "seqnorm/" + f"raw_beta_{beta}_atac/"
else:
    cespgrn_dir = result_dir + "seqnorm/" + f"raw_beta_{beta}/"


# without the sequence normalization
# if use_tf & (beta > 0):
#     cespgrn_dir = result_dir + "noseqnorm/" + f"raw_beta_{beta}_tf/"
# elif beta > 0:
#     cespgrn_dir = result_dir + "noseqnorm/" + f"raw_beta_{beta}_atac/"
# else:
#     cespgrn_dir = result_dir + "noseqnorm/" + f"raw_beta_{beta}/"


if not os.path.exists(cespgrn_dir):
    os.makedirs(cespgrn_dir)

# In[]
# -------------------------------------
#
# read data
#
# -------------------------------------
# count data
counts_rna = pd.read_csv(path + "counts_rna_true.txt", sep = "\t", index_col = 0).T.values
meta_cells = pd.read_csv(path + "meta_cells.txt", delimiter = "\t")

ncells, ngenes = counts_rna.shape
# Choice 1: normalization step
libsize = np.median(np.sum(counts_rna, axis = 1))
counts_rna_norm = counts_rna /(np.sum(counts_rna, axis = 1, keepdims = True) + 1e-6) * libsize
counts_rna_norm = np.log1p(counts_rna_norm)

# np.savetxt(path + "counts_norm.txt", counts_rna_norm.T, delimiter = "\t")

# Choice 2: no sequence depth normalization
# counts_rna_norm = counts_rna

# In[]
# -------------------------------------
# 
# Required calculation (only calculate once)
# 
# -------------------------------------
'''
# ground truth GRNs, symmetric  
Gs_gt = []
for i in range(1, ncells + 1):
    grn_gt = pd.read_csv(path + f"grn_gt_{i}.txt", sep = "\t",index_col = 0)
    G_gt = np.zeros((ngenes + 1, ngenes + 1))
    G_gt[np.ix_(grn_gt.index.values.squeeze().astype(int), grn_gt.columns.values.squeeze().astype(int))] = grn_gt.values
    G_gt += G_gt.T
    Gs_gt.append(G_gt[None,1:,1:])

Gs_gt = np.concatenate(Gs_gt, axis = 0)
np.save(file = path + "graph_gt.npy", arr = Gs_gt)

# mask matrix, from TF information
tfs = []
for i in range(1, ncells + 1):
    grn_gt = pd.read_csv(path + f"grn_gt_{i}.txt", sep = "\t",index_col = 0)
    tf = [x-1 for x in grn_gt.columns.values.squeeze().astype(int)]
    tfs.extend(tf)
tfs = np.unique(np.array(tfs))
masks = np.zeros((ngenes, ngenes))
masks[tfs,:] = 1
masks[:,tfs] = 1
masks = 1 - masks
masks = np.repeat(masks[None,:,:], ncells, 0)
print("sparsity of mask:" + str(np.sum(masks == 0)/(masks.shape[0] * masks.shape[1] * masks.shape[2])))
np.save(result_dir + "masks_tf.npy", masks)

# mask matrix, from scATAC-seq information
masks = []
for i in range(1, ncells + 1):
    grn_prior = pd.read_csv(path + f"grn_prior_{i}.txt", sep = "\t",index_col = 0)
    mask = np.zeros((ngenes + 1, ngenes + 1))
    mask[np.ix_(grn_prior.index.values.squeeze().astype(int), grn_prior.columns.values.squeeze().astype(int))] = grn_prior.values
    mask += mask.T
    mask = (mask == 0).astype(np.float)
    masks.append(mask[None,1:,1:])
masks = np.concatenate(masks, axis = 0)
print("sparsity of mask:" + str(np.sum(masks == 0)/(masks.shape[0] * masks.shape[1] * masks.shape[2])))
np.save(result_dir + "masks_atac.npy", masks)

x_pca = PCA(n_components = 30).fit_transform(counts_rna_norm)
x_umap = UMAP(min_dist = 0.4, random_state = 0).fit_transform(x_pca)
fig = plt.figure(figsize  = (10,7))
ax = fig.add_subplot()
ax.scatter(x_umap[:, 0], x_umap[:, 1], c = meta_cells["depth"].values.squeeze(), s = 10)
ax.set_xlabel("UMAP1")
ax.set_ylabel("UMAP2")
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False, markerscale = 3)
fig.savefig(result_dir + "x_umap.png", bbox_inches = "tight")
'''

# In[]
if (beta > 0) and use_tf:
    masks = np.load(result_dir + "masks_tf.npy")
elif beta > 0:
    masks = np.load(result_dir + "masks_atac.npy")
else:
    masks = None

# In[]
if False:
    for bandwidth in bandwidths:
        X_pca = PCA(n_components = 20).fit_transform(counts_rna_norm)
        K, K_trun = kernel.calc_kernel_neigh(X_pca, bandwidth = bandwidth, truncate = True, truncate_param = truncate_param)
        
        if not os.path.exists(cespgrn_dir + "cov_" + str(bandwidth) + "_" + str(truncate_param) + ".npy"):
            empir_cov = cov.est_cov_para(X = counts_rna_norm, K_trun = K_trun, njobs = 8)
            np.save(cespgrn_dir + "cov_" + str(bandwidth) + "_" + str(truncate_param) + ".npy", empir_cov)

        empir_cov = np.load(cespgrn_dir + "cov_" + str(bandwidth) + "_" + str(truncate_param) + ".npy")

        for lamb in lamb_list:
            print("scanning...")
            print("bandwidth: " + str(bandwidth))
            print("lambda: " + str(lamb))
            print("number of threads: " + str(torch.get_num_threads()))

            alpha = 2
            rho = 1.7
            max_iters = 100
            
            # test model without TF
            gadmm_batch = g_admm.G_admm_mask(X = counts_rna_norm[:, None, :], K = K, mask = masks, pre_cov = empir_cov, batchsize = 512, device = device)
            w_empir_cov = gadmm_batch.w_empir_cov.detach().cpu().numpy()
            Gs = gadmm_batch.train(max_iters = max_iters, n_intervals = 100, alpha = alpha, lamb = lamb, rho = rho, theta_init_offset = 0.1, beta = beta)
            
            sparsity = np.sum(Gs != 0)/(Gs.shape[0] * Gs.shape[1] * Gs.shape[2])
            print("sparsity: " + str(sparsity))
            
            np.save(file = cespgrn_dir + "precision_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_"+ str(beta) + ".npy", arr = gadmm_batch.thetas) 


# In[] Ensemble
thetas_ensemble = 0
Gs_ensemble = 0
Gs_ensemble_soft = 0

# bandwidths = [0.01, 0.1]
for bandwidth in bandwidths:
    for lamb in lamb_list:
        thetas = np.load(file = cespgrn_dir + "precision_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_"+ str(beta) +".npy") 
        Gs_soft = g_admm.construct_weighted_G(thetas, njobs = 1)

        # check sparsity
        # sparsity = np.sum(Gs != 0)/(Gs.shape[0] * Gs.shape[1] * Gs.shape[2])
        # if (sparsity >= 0.05) & (sparsity <= 0.95):
        thetas_ensemble += thetas
        Gs_ensemble_soft += Gs_soft

if masks is not None:
    Gs_ensemble = Gs_ensemble_soft * (1 - masks)
else:
    Gs_ensemble = Gs_ensemble_soft
# average ensemble
thetas_ensemble /= (len(bandwidths) * len(lamb_list))
Gs_ensemble /= (len(bandwidths) * len(lamb_list))
Gs_ensemble_soft /= (len(bandwidths) * len(lamb_list))

np.save(file = cespgrn_dir + f"cespgrn_ensemble_{truncate_param}_{beta}.npy", arr = Gs_ensemble)


# In[]
'''
# ---------------------------------------------------------------------------
#
# Running GRN inference using GENIE 3
#
# ---------------------------------------------------------------------------

# benchmark the methods that infer GRN with TF information
if use_tf & (beta > 0):
    genie_theta = genie3.GENIE3(counts_rna_norm, gene_names=["gene_" + str(x+1) for x in range(ngenes)], regulators='all',tree_method='RF',K='sqrt',ntrees=1000,nthreads=1)
    genie_theta = np.repeat(genie_theta[None, :, :], ncells, axis=0)
    np.save(file = result_dir + "Gs_genie.npy", arr = genie_theta)

    genie_theta_tf = genie3.GENIE3(counts_rna_norm, gene_names=["gene_" + str(x+1) for x in range(ngenes)], regulators=["gene_2", "gene_6", "gene_10", "gene_19", "gene_80", "gene_91"], tree_method='RF',K='sqrt',ntrees=1000,nthreads=1)
    genie_theta_tf = np.repeat(genie_theta_tf[None, :, :], ncells, axis=0)
    np.save(file = result_dir + "Gs_genie_tf.npy", arr = genie_theta_tf)
'''

# %%
