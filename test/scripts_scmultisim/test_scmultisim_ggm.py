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


# In[] 
# -------------------------------------
#
# hyper-parameter
#
# -------------------------------------

truncate_param = 100
lamb_list = [0.01, 0.05, 0.1, 0.5]
bandwidths = [0.01, 0.1, 1, 10]
ggm_choice = "gcgm"

# # 1. ATAC
beta = 1
use_tf = False

# 2. TF
# beta = 1
# use_tf = True

# # 3. Original
# beta = 0
# use_tf = False

# In[]
dataset = "simulated_8000_20_10_100_0.1_1_0_4"
path = PROJECT_DIR + f"data/scMultiSim/{dataset}/"
result_dir = PROJECT_DIR + f"test/results_scmultisim/{dataset}/"
if use_tf & (beta > 0):
    cespgrn_dir = result_dir + f"{ggm_choice}_beta_{beta}_tf/"
elif beta > 0:
    cespgrn_dir = result_dir + f"{ggm_choice}_beta_{beta}_atac/"
else:
    cespgrn_dir = result_dir + f"{ggm_choice}_beta_{beta}/"

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
ncells, ngenes = counts_rna.shape
# normalization step
libsize = np.median(np.sum(counts_rna, axis = 1))
counts_rna_norm = counts_rna /(np.sum(counts_rna, axis = 1, keepdims = True) + 1e-6) * libsize
counts_rna_norm = np.log1p(counts_rna_norm)
meta_cells = pd.read_csv(path + "meta_cells.txt", delimiter = "\t")

# np.savetxt(path + "counts_norm.txt", counts_rna_norm.T, delimiter = "\t")


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
    K, K_trun = kernel.calc_kernel_neigh(X_pca, bandwidth = bandwidth, truncate = True, truncate_param = truncate_param)

    if ggm_choice == "gcgm_w":
        empir_cov = cov.est_cov_para(X = counts_rna_norm, K_trun = K_trun, weighted_kt = True, njobs = 8)
    elif ggm_choice == "gcgm":
        empir_cov = cov.est_cov_para(X = counts_rna_norm, K_trun = K_trun, weighted_kt = False, njobs = 8)
    elif ggm_choice == "ggm":
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

        try:
            Gs = np.load(cespgrn_dir + "precision_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_"+ str(beta) + ".npy")
            print("Inferred.")
        except:

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
assert False
# In[]
# -----------------------------------------------------
# 
# NOTE: Ablation test
#
# -----------------------------------------------------

datasets = [
    "simulated_8000_20_10_100_0.01_0_0_4",
    "simulated_8000_20_10_100_0.01_1_0_4",
    "simulated_8000_20_10_100_0.01_2_0_4",
    "simulated_8000_20_10_100_0.1_0_0_4",
    "simulated_8000_20_10_100_0.1_1_0_4",
    "simulated_8000_20_10_100_0.1_2_0_4"
]

use_tf = False
beta = 1
truncate_param = 100

scores = []

for dataset in datasets:
    print(dataset)
    score_filename = PROJECT_DIR + f"test/results_scmultisim/ablation_ggm/scores_{dataset}.csv"
    # if os.path.exists(score_filename):
    #     print("score already calculated...")
    #     continue
    path = PROJECT_DIR + f"data/scMultiSim/{dataset}/"
    result_dir = PROJECT_DIR + f"test/results_scmultisim/{dataset}/"
    Gs_gt = np.load(file = path + "graph_gt.npy")

    print("Calculating score (CeSpGRN-ATAC)...")
    Gs_cespgrn_atac = np.load(file = result_dir + f"beta_1_atac/cespgrn_ensemble_{truncate_param}_1.npy")
    scores_cespgrn_atac = bmk.calc_scores_para(thetas_inf = Gs_cespgrn_atac, thetas_gt = Gs_gt, interval = None, model = "CeSpGRN (GCGM)", bandwidth = None, truncate_param = truncate_param, lamb = None, beta = 1, njobs = 16)
    scores_cespgrn_atac["dataset"] = dataset

    print("Calculating score (CeSpGRN-ATAC-GGM)...")
    Gs_cespgrn_atac_ggm = np.load(file = result_dir + f"ggm_beta_1_atac/cespgrn_ensemble_{truncate_param}_1.npy")
    scores_cespgrn_atac_ggm = bmk.calc_scores_para(thetas_inf = Gs_cespgrn_atac_ggm, thetas_gt = Gs_gt, interval = None, model = "CeSpGRN (GGM)", bandwidth = None, truncate_param = truncate_param, lamb = None, beta = 1, njobs = 16)
    scores_cespgrn_atac_ggm["dataset"] = dataset

    print("Calculating score (CeSpGRN-ATAC-GCGM)")
    Gs_cespgrn_atac_gcgm = np.load(file = result_dir + f"gcgm_beta_1_atac/cespgrn_ensemble_{truncate_param}_1.npy")
    scores_cespgrn_atac_gcgm = bmk.calc_scores_para(thetas_inf = Gs_cespgrn_atac_gcgm, thetas_gt = Gs_gt, interval = None, model = "CeSpGRN (GCGM-ori)", bandwidth = None, truncate_param = truncate_param, lamb = None, beta = 1, njobs = 16)
    scores_cespgrn_atac_gcgm["dataset"] = dataset

    score = pd.concat([scores_cespgrn_atac, scores_cespgrn_atac_ggm, scores_cespgrn_atac_gcgm], axis = 0, ignore_index = True)
    score.to_csv(score_filename)

    scores.append(score)

scores = pd.concat(scores, axis = 0, ignore_index = True)
scores.to_csv(PROJECT_DIR + "test/results_scmultisim/ablation_ggm/scores.csv")

# In[]
# NOTE: Plot barplot
use_tf = False
sns.set_theme(font_scale = 1.7)
scores = pd.DataFrame() 
plt.rcParams["font.size"] = 17

for nchanging_edges in [20]:
    for fp in [0.01]:
        for seed in [0, 1, 2]:
            result_dir = PROJECT_DIR + f"test/results_scmultisim/ablation_ggm/"
            score = pd.read_csv(result_dir + f"scores_simulated_8000_{nchanging_edges}_10_100_{fp}_{seed}_0_4.csv", index_col = 0)
            score["fp"] = fp
            scores = pd.concat([scores, score], axis = 0)

fig = plt.figure(figsize = (15,7))
ax = fig.subplots(nrows =1, ncols = 2)
# error_bar = ("pi", 100)
error_bar = "sd"
bar1 = sns.barplot(scores, x = "model", y = "AUPRC (abs)", hue = "fp", ax = ax[0], width=0.5, palette = "Set2", estimator=np.mean, errorbar=error_bar, capsize = 0.1, err_kws = {"color": "blue", "linewidth": 2, "linestyle": "--"})
for i in bar1.containers:
    bar1.bar_label(i, fmt = "%.3f", fontsize = 20)
bar2 = sns.barplot(scores, x = "model", y = "Early Precision (abs)", hue = "fp", ax = ax[1], width=0.5, palette = "Set2", estimator=np.mean, errorbar=error_bar, capsize = 0.1, err_kws = {"color": "blue", "linewidth": 2, "linestyle": "--"})
for i in bar2.containers:
    bar2.bar_label(i, fmt = "%.3f", fontsize = 20)

_ = ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation = 45, fontsize = 20)
_ = ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation = 45, fontsize = 20)
ax[0].get_legend().remove()
ax[0].set_xlabel(None)
ax[0].set_ylabel("AUPRC", fontsize = 20)
ax[1].set_xlabel(None)
ax[1].set_ylabel("Eprec", fontsize = 20)
# leg = ax[1].legend(loc='upper left', prop={'size': 20}, frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1), markerscale = 6, title = "Noise level", title_fontsize = 20)
ax[1].get_legend().remove()
plt.tight_layout()

fig.savefig(PROJECT_DIR + "test/results_scmultisim/ablation_ggm/auprc.png", bbox_inches = "tight")


# %%
