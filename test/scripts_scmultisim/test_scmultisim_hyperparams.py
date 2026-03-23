# In[]
import sys, os
PROJECT_DIR = "/localscratch/ziqi/CeSpGRN/"

sys.path.append(PROJECT_DIR + 'src/')
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

dataset_list = ["simulated_8000_20_10_100_0.01_0_0_4", "simulated_8000_20_10_100_0.01_1_0_4", "simulated_8000_20_10_100_0.01_2_0_4",
                "simulated_8000_20_10_100_0.1_0_0_4", "simulated_8000_20_10_100_0.1_1_0_4", "simulated_8000_20_10_100_0.1_2_0_4"]

dataset = dataset_list[0]
path = f"../../data/scMultiSim/{dataset}/"
result_dir = f"../results_scmultisim/{dataset}/"

# In[]
# --------------------------------------
#
# Hyper-parameters                                    
# 
# --------------------------------------
bandwidths = [0.01, 0.1, 1, 10]
lamb_list = [0.01, 0.05, 0.1, 0.5]
use_tf = False

# # 1. beta
# beta_list = [1e-2, 1e-1, 10, 100]
# beta = beta_list[0]
# truncate_param = 100
# cespgrn_dir = result_dir + f"beta_{beta}_atac/"
# if not os.path.exists(cespgrn_dir):
#     os.makedirs(cespgrn_dir)

# 2. truncate params
truncate_param_list = [10, 100, 200, 500]
truncate_param = truncate_param_list[3]
beta = 1
cespgrn_dir = result_dir + f"truncate_{truncate_param}_atac/"
if not os.path.exists(cespgrn_dir):
    os.makedirs(cespgrn_dir)

# In[]
# --------------------------------------
#
# Ablation study                                    
# 
# --------------------------------------
'''
truncate_param = 100
bandwidths = [0.01, 0.1, 1, 10]
use_tf = False

# # 1. beta ablation
# beta = 0
# lamb_list = [0.01, 0.05, 0.1, 0.5]
# cespgrn_dir = result_dir + f"ablation_beta_{beta}/"

# 2. lambda ablation
# no TF
# beta = 1
# lamb_list = [0.0]
# use_tf = False
# cespgrn_dir = result_dir + f"ablation_lamb_{lamb_list}/"

# use TF
beta = 1
lamb_list = [0.0]
use_tf = True
cespgrn_dir = result_dir + f"ablation_lamb_{lamb_list}_tf/"

if not os.path.exists(cespgrn_dir):
    os.makedirs(cespgrn_dir)
'''

# In[]
# -------------------------------------
#
# read data
#
# -------------------------------------
# count data
counts_rna = pd.read_csv(path + "counts_rna_true.txt", sep = "\t", index_col = 0).T.values
ncells, ngenes = counts_rna.shape
# normalization step
libsize = np.median(np.sum(counts_rna, axis = 1))
counts_rna_norm = counts_rna /(np.sum(counts_rna, axis = 1, keepdims = True) + 1e-6) * libsize
counts_rna_norm = np.log1p(counts_rna_norm)
meta_cells = pd.read_csv(path + "meta_cells.txt", delimiter = "\t")

# In[]
if (beta > 0) and use_tf:
    masks = np.load(result_dir + "masks_tf.npy")
elif beta > 0:
    masks = np.load(result_dir + "masks_atac.npy")
else:
    masks = None

# In[]
for bandwidth in bandwidths:
    X_pca = PCA(n_components = 20).fit_transform(counts_rna_norm)
    K, K_trun = kernel.calc_kernel_neigh(X_pca, k = 5, bandwidth = bandwidth, truncate = True, truncate_param = truncate_param)

    if os.path.isfile(cespgrn_dir + "cov_" + str(bandwidth) + "_" + str(truncate_param) + ".npy"):
        empir_cov = np.load(cespgrn_dir + "cov_" + str(bandwidth) + "_" + str(truncate_param) + ".npy")
    else:
        empir_cov = cov.est_cov_ggm_para(X = counts_rna_norm, K_trun = K_trun, njobs = 8)
        np.save(cespgrn_dir + "cov_" + str(bandwidth) + "_" + str(truncate_param) + ".npy", empir_cov)

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



