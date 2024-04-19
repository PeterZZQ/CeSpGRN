# In[0]
import sys, os
sys.path.append('../../src/')
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
import torch
import torch.nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from umap import UMAP
import g_admm, kernel
import warnings
warnings.filterwarnings("ignore")

import anndata
import scanpy as sc
import pandas as pd
import seaborn as sns
plt.rcParams.update(plt.rcParamsDefault)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)
plt.rcParams["font.size"] = 20

# In[]
path = "../../data/nmp/"
result_dir = "../results_nmp/wt_2000/beta_1/"
if not os.path.exists(result_dir + "plots/"):
    os.makedirs(result_dir)
    os.makedirs(result_dir + "plots/")

# In[]
adata_rna_wt = anndata.read_h5ad(path + "adata_rna_wt_filtered_2000.h5ad")
adata_rna_ko = anndata.read_h5ad(path + "adata_rna_ko_filtered_2000.h5ad")
masks_wt = np.load(path + "cell_specific_mask_wt_2000.npy")
masks_ko = np.load(path + "cell_specific_mask_ko_2000.npy")

np.random.seed(0)
idx_wt = np.random.choice(adata_rna_wt.shape[0], size = 300)
adata_rna_wt = adata_rna_wt[idx_wt,:]
masks_wt = masks_wt[idx_wt,:,:]

idx_ko = np.random.choice(adata_rna_ko.shape[0], size = 300)
adata_rna_ko = adata_rna_ko[idx_ko,:]
masks_ko = masks_ko[idx_ko,:,:]

assert masks_wt.shape[0] == adata_rna_wt.shape[0]
assert masks_ko.shape[0] == adata_rna_ko.shape[0]
assert masks_wt.shape[1] == adata_rna_wt.shape[1]
assert masks_ko.shape[1] == adata_rna_ko.shape[1]
assert masks_wt.shape[2] == adata_rna_wt.shape[1]
assert masks_ko.shape[2] == adata_rna_ko.shape[1]


counts_wt = adata_rna_wt.X.toarray()
meta_wt = adata_rna_wt.obs
counts_ko = adata_rna_ko.X.toarray()
meta_ko = adata_rna_ko.obs
genes = adata_rna_wt.var.index.values.squeeze()

# normalization step
libsize = np.median(np.sum(counts_wt, axis = 1))
counts_wt_norm = counts_wt /(np.sum(counts_wt, axis = 1, keepdims = True) + 1e-6) * libsize
counts_wt_norm = np.log1p(counts_wt_norm)


libsize = np.median(np.sum(counts_ko, axis = 1))
counts_ko_norm = counts_ko /(np.sum(counts_ko, axis = 1, keepdims = True) + 1e-6) * libsize
counts_ko_norm = np.log1p(counts_ko_norm)

# In[]
x_pca = PCA(n_components = 30).fit_transform(counts_wt_norm)
x_umap = UMAP(min_dist = 0.4, random_state = 0).fit_transform(x_pca)
fig = plt.figure(figsize  = (10,7))
ax = fig.add_subplot()
for i in np.sort(np.unique(adata_rna_wt.obs["celltype"].values)):
    idx = np.where(adata_rna_wt.obs["celltype"].values == i)
    ax.scatter(x_umap[idx, 0], x_umap[idx, 1], label = i, s = 10)
ax.set_xlabel("UMAP1")
ax.set_ylabel("UMAP2")
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False, markerscale = 3)
fig.savefig(result_dir + "plots/x_umap.png", bbox_inches = "tight")



# In[2] ADMM
# -----------------------------------------------------------------------------------------------------
#
# Infer cell-specific GRN using ADMM
#
# -----------------------------------------------------------------------------------------------------
import importlib 
importlib.reload(g_admm)
import cov
# hyper-parameter
truncate_param = 100
beta = 1

lamb_list = [0.01, 0.05, 0.1, 0.5]
bandwidths = [0.01, 0.1, 1, 10]

counts = counts_wt_norm

for bandwidth in bandwidths:
    X_pca = PCA(n_components = 20).fit_transform(counts)
    K, K_trun = kernel.calc_kernel_neigh(X_pca, bandwidth = bandwidth, truncate = True, truncate_param = truncate_param)
    empir_cov = cov.est_cov_para(X = counts, K_trun = K_trun, njobs = 8)

    np.save(result_dir + "cov_" + str(bandwidth) + "_" + str(truncate_param) + ".npy", empir_cov)
    empir_cov = np.load(result_dir + "cov_" + str(bandwidth) + "_" + str(truncate_param) + ".npy")

    for lamb in lamb_list:
        print("scanning...")
        print("bandwidth: " + str(bandwidth))
        print("lambda: " + str(lamb))
        print("number of threads: " + str(torch.get_num_threads()))
        alpha = 2
        rho = 1.7
        max_iters = 100
        
        # test model without TF
        gadmm_batch = g_admm.G_admm_mask(X = counts[:, None, :], K = K, mask = masks_wt, pre_cov = empir_cov, batchsize = 8, device = device)
        w_empir_cov = gadmm_batch.w_empir_cov.detach().cpu().numpy()
        Gs = gadmm_batch.train(max_iters = max_iters, n_intervals = 100, alpha = alpha, lamb = lamb, rho = rho, theta_init_offset = 0.1, beta = beta)
        
        sparsity = np.sum(Gs != 0)/(Gs.shape[0] * Gs.shape[1] * Gs.shape[2])
        print("sparsity: " + str(sparsity))
        np.save(file = result_dir + "precision_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_"+ str(beta) + ".npy", arr = gadmm_batch.thetas) 
        np.save(file = result_dir + "graph_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_"+ str(beta) + ".npy", arr = Gs) 

# In[] Ensemble
thetas_ensemble = 0
Gs_ensemble = 0
Gs_ensemble_soft = 0

for bandwidth in bandwidths:
    for lamb in lamb_list:
        thetas = np.load(file = result_dir + "precision_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_"+ str(beta) +".npy") 
        Gs = np.load(file = result_dir + "graph_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_"+ str(beta) +".npy") 
        Gs_soft = g_admm.construct_weighted_G(thetas, njobs = 1)

        # check sparsity
        sparsity = np.sum(Gs != 0)/(Gs.shape[0] * Gs.shape[1] * Gs.shape[2])
        if (sparsity >= 0.05) & (sparsity <= 0.95):
            thetas_ensemble += thetas
            Gs_ensemble += Gs 
            Gs_ensemble_soft += Gs_soft

# average ensemble
thetas_ensemble /= (len(bandwidths) * len(lamb_list))
Gs_ensemble /= (len(bandwidths) * len(lamb_list))
Gs_ensemble_soft /= (len(bandwidths) * len(lamb_list))

np.save(file = result_dir + f"precision_ensemble_{truncate_param}_{beta}.npy", arr = thetas_ensemble)
np.save(file = result_dir + f"graph_ensemble_{truncate_param}_{beta}.npy", arr = Gs_ensemble)
np.save(file = result_dir + f"graph_ensemble_soft_{truncate_param}_{beta}.npy", arr = Gs_ensemble_soft)


# In[]
validate = True
thetas_ensemble = np.load(file = result_dir + f"precision_ensemble_{truncate_param}_{beta}.npy")
# force the mask
Gs_ensemble = np.load(file = result_dir + f"graph_ensemble_soft_{truncate_param}_{beta}.npy")
# soft-weighted
meta_rna_wt = adata_rna_wt.obs
meta_rna_ko = adata_rna_ko.obs

Gs_ensemble = Gs_ensemble.reshape(Gs_ensemble.shape[0], -1)
var = np.mean((Gs_ensemble - np.mean(Gs_ensemble, axis = 0, keepdims = True)) ** 2, axis = 0) 
print("variance (Gs): " + str(np.sum(var)))

plt.rcParams["font.size"] = 20
Gs_pca = PCA(n_components = 100).fit_transform(Gs_ensemble)
Gs_umap = UMAP(n_components = 2, min_dist = 0.8, random_state = 0, n_neighbors = 50).fit_transform(Gs_pca)

fig = plt.figure(figsize  = (10,7))
ax = fig.add_subplot()
for i in np.sort(np.unique(meta_rna_wt["celltype"].values)):
    idx = np.where(meta_rna_wt["celltype"].values == i)
    ax.scatter(Gs_umap[idx, 0], Gs_umap[idx, 1], label = i, s = 10)
ax.set_xlabel("UMAP1")
ax.set_ylabel("UMAP2")
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False, markerscale = 3)
# ax.set_title("bandwidth: " + str(bandwidth) + ", truncate_param: " + str(truncate_param) + ", lamb: " + str(lamb))
fig.savefig(result_dir + "plots/graphs_ensemble_umap.png", bbox_inches = "tight")

# In[]
# ----------------------------------------------------------------------
#
# Analysis use wt dataset, as it has two trajectories
#
# ----------------------------------------------------------------------
# trajectory 1: NMP -> Caudal/Somitic mesoderm; 2. NMP -> Spinal cord
# check the variation of hub-gene scores in different trajectory
thetas_ensemble = np.load(file = result_dir + f"precision_ensemble_{truncate_param}_{beta}.npy")
Gs_ensemble = np.load(file = result_dir + f"graph_ensemble_soft_{truncate_param}_{beta}.npy")
genes = adata_rna_wt.var.index.values
if validate:
    hub_scores = np.sum(np.abs(Gs_ensemble), axis = 2)

    hub_scores1 = hub_scores[meta_rna_wt["celltype"].isin(["NMP", "Caudal_mesoderm", "Somitic_mesoderm"]),:]
    variance1 = np.mean((hub_scores1 - np.mean(hub_scores1, axis = 0, keepdims = True))**2, axis = 0)
    # Find top genes including Ripply2, Mesp2, the regulation of Mesp2-Ripply2-Tbx6 is important in PSM.
    np.savetxt(result_dir + "hub_gene_mesoderm.txt", genes[np.argsort(variance1)[:-100:-1]], fmt = "%s")

    hub_scores2 = hub_scores[meta_rna_wt["celltype"].isin(["NMP", "Spinal_cord"]),:]
    variance2 = np.mean((hub_scores2 - np.mean(hub_scores2, axis = 0, keepdims = True))**2, axis = 0)
    np.savetxt(result_dir + "hub_gene_spinal_cord.txt", genes[np.argsort(variance2)[:-100:-1]], fmt = "%s")



# In[]
# Check the variance of edges during the differentiation process
if validate:
    tf_target = []
    for gene1 in genes:
        tf_target.extend([gene1 + "-" + gene2 for gene2 in genes])
    tf_target = np.array(tf_target)
    idx = np.array([idx for idx, x in enumerate(tf_target) if x.split("-")[0] != x.split("-")[1]])

    Gs_ensemble_reshaped = Gs_ensemble.reshape(Gs_ensemble.shape[0], -1)
    Gs_ensemble_mesoderm = Gs_ensemble_reshaped[meta_rna_wt["celltype"].isin(["NMP", "Caudal_mesoderm", "Somitic_mesoderm"]),:]
    Gs_ensemble_spinal_cord = Gs_ensemble_reshaped[meta_rna_wt["celltype"].isin(["NMP", "Spinal_cord"]),:]
    variance_mesoderm = np.sum((Gs_ensemble_mesoderm - np.mean(Gs_ensemble_mesoderm, axis = 0, keepdims = True)) ** 2, axis = 0)/Gs_ensemble_mesoderm.shape[0]
    variance_spinal_cord = np.sum((Gs_ensemble_spinal_cord - np.mean(Gs_ensemble_spinal_cord, axis = 0, keepdims = True)) ** 2, axis = 0)/Gs_ensemble_spinal_cord.shape[0]

    tf_target = tf_target[idx]
    variance_mesoderm = variance_mesoderm[idx]
    variance_spinal_cord = variance_spinal_cord[idx]

    tf_target_mesoderm_sorted = tf_target[np.argsort(variance_mesoderm)[::-1]]
    tf_target_spinal_cord_sorted = tf_target[np.argsort(variance_spinal_cord)[::-1]]
    variance_mesoderm_sorted = variance_mesoderm[np.argsort(variance_mesoderm)[::-1]]
    variance_spinal_cord_sorted = variance_spinal_cord[np.argsort(variance_spinal_cord)[::-1]]

    variance_mesoderm_sorted = pd.DataFrame(data = variance_mesoderm_sorted[:,None], columns = ["variance"], index = tf_target_mesoderm_sorted)
    variance_mesoderm_sorted.to_csv(result_dir + "edge_variance_mesoderm.csv")
    variance_spinal_cord_sorted = pd.DataFrame(data = variance_spinal_cord_sorted[:,None], columns = ["variance"], index = tf_target_spinal_cord_sorted)
    variance_spinal_cord_sorted.to_csv(result_dir + "edge_variance_spinal_cord.csv")

# In[]
# Check the trend
plt.rcParams["font.size"] = 20
sns.set_style("darkgrid")

if validate:
    # infer pseudotime
    sc.pp.normalize_per_cell(adata_rna_wt)
    sc.pp.log1p(adata_rna_wt)
    sc.pp.neighbors(adata_rna_wt)
    sc.tl.diffmap(adata_rna_wt)
    adata_rna_wt.uns['iroot'] = np.flatnonzero(adata_rna_wt.obs["celltype"]  == "NMP")[0]
    sc.tl.dpt(adata_rna_wt)
    pt = adata_rna_wt.obs["dpt_pseudotime"].values
    meta_rna_wt = adata_rna_wt.obs

    tf_target = []
    for gene1 in genes:
        tf_target.extend([gene1 + "-" + gene2 for gene2 in genes])
    tf_target = np.array(tf_target)

    Gs_ensemble_reshaped = Gs_ensemble.reshape(Gs_ensemble.shape[0], -1)

    weights = Gs_ensemble_reshaped[:,np.where(tf_target == "Ripply2-Mesp2")[0][0]]
    weights = weights[np.argsort(pt)]
    meta_rna_wt = meta_rna_wt.iloc[np.argsort(pt),:]
    # mesoderm branch
    weights1 = weights[meta_rna_wt["celltype"].isin(["NMP", "Caudal_mesoderm", "Somitic_mesoderm"])]
    weights2 = weights[~meta_rna_wt["celltype"].isin(["NMP", "Caudal_mesoderm", "Somitic_mesoderm"])]

    fig = plt.figure(figsize = (12,5))
    ax = fig.subplots(nrows = 1, ncols = 2)
    ax[0].plot(weights1)
    ax[1].plot(weights2)
    ax[0].set_ylim([0, 0.07])
    ax[1].set_ylim([0, 0.07])
    ax[0].set_ylabel("Weight")
    ax[1].set_ylabel("Weight")
    ax[0].set_xlabel("pseudo-time")
    ax[1].set_xlabel("pseudo-time")
    ax[0].set_title("Mesoderm branch")
    ax[1].set_title("Spinal cord branch")
    plt.tight_layout()
    fig.savefig(result_dir + "plots/Mesp2_Ripply2.png", bbox_inches = "tight")

# In[]
fig = plt.figure(figsize  = (10,7))
ax = fig.add_subplot()
pic = ax.scatter(Gs_umap[:, 0], Gs_umap[:, 1], c = pt, s = 10)
ax.set_xlabel("UMAP1")
ax.set_ylabel("UMAP2")
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False, markerscale = 3)
cbar = fig.colorbar(pic, fraction=0.046, pad=0.04, ax = ax)
cbar.ax.tick_params(labelsize = 20)
fig.savefig(result_dir + "plots/pseudotime.png", bbox_inches = "tight")

# %%
