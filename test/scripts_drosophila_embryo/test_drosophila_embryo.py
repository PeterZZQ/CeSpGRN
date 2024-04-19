# In[]
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

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)
plt.rcParams["font.size"] = 20

result_dir = "../results_drosophila_embryo/"
if not os.path.exists(result_dir + "plots/"):
    os.makedirs(result_dir)
    os.makedirs(result_dir + "plots/")

# # In[]
# # spaOTsc processed data
# data_dir = "../../data/drosophila_embryo/processed/"
# spaLoc = pd.read_csv(data_dir + "dm_geometry.txt", sep = " ")
# expr = pd.read_csv(data_dir + "dm_is.txt", sep = "\t", index_col = 0)

# In[]
# raw data
data_dir = "../../data/drosophila_embryo/"
spaLoc = pd.read_csv(data_dir + "geometry.txt", sep = " ")
expr = pd.read_csv(data_dir + "bdtnp.txt", sep = "\t")

genes = expr.columns.values
# normalize
expr_norm = expr.values/(np.sum(expr.values, axis = 1, keepdims = True) + 1e-6)
expr_norm = np.log1p(expr_norm)

# In[]
fig = plt.figure(figsize = (10, 7))
ax = fig.subplots(nrows = 1, ncols = 1)
# visualize x and z coordinates
ax.scatter(spaLoc.values[:, 0], spaLoc.values[:, 2])


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

counts = expr_norm

for bandwidth in bandwidths:
    X_pca = PCA(n_components = 20).fit_transform(counts)
    # using the spatial location to calculate the kernal function
    K, K_trun = kernel.calc_kernel_neigh(spaLoc.values, bandwidth = bandwidth, truncate = True, truncate_param = truncate_param)
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
        gadmm_batch = g_admm.G_admm_mask(X = counts[:, None, :], K = K, mask = None, pre_cov = empir_cov, batchsize = 500, device = device)
        w_empir_cov = gadmm_batch.w_empir_cov.detach().cpu().numpy()
        Gs = gadmm_batch.train(max_iters = max_iters, n_intervals = 100, alpha = alpha, lamb = lamb, rho = rho, theta_init_offset = 0.1, beta = beta)
        
        sparsity = np.sum(Gs != 0)/(Gs.shape[0] * Gs.shape[1] * Gs.shape[2])
        print("sparsity: " + str(sparsity))
        np.save(file = result_dir + "precision_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_"+ str(beta) + ".npy", arr = gadmm_batch.thetas) 
        np.save(file = result_dir + "graph_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_"+ str(beta) + ".npy", arr = Gs) 

# In[] Ensemble
thetas_ensemble = 0
Gs_ensemble = 0

for bandwidth in bandwidths:
    for lamb in lamb_list:
        thetas = np.load(file = result_dir + "precision_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_"+ str(beta) +".npy") 
        Gs = np.load(file = result_dir + "graph_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_"+ str(beta) +".npy") 

        # check sparsity
        sparsity = np.sum(Gs != 0)/(Gs.shape[0] * Gs.shape[1] * Gs.shape[2])
        if (sparsity >= 0.05) & (sparsity <= 0.95):
            thetas_ensemble += thetas
            Gs_ensemble += Gs 

# average ensemble
thetas_ensemble /= (len(bandwidths) * len(lamb_list))
Gs_ensemble /= (len(bandwidths) * len(lamb_list))

np.save(file = result_dir + f"precision_ensemble_{truncate_param}_{beta}.npy", arr = thetas_ensemble)
np.save(file = result_dir + f"graph_ensemble_{truncate_param}_{beta}.npy", arr = Gs_ensemble)

# In[]
Gs_ensemble = np.load(file = result_dir + f"graph_ensemble_{truncate_param}_{beta}.npy")

Gs_ensemble_reshape = Gs_ensemble.reshape(Gs_ensemble.shape[0], -1)
var = np.mean((Gs_ensemble_reshape - np.mean(Gs_ensemble_reshape, axis = 0, keepdims = True)) ** 2, axis = 0) 
print("variance (Gs): " + str(np.sum(var)))

plt.rcParams["font.size"] = 20
Gs_pca = PCA(n_components = 100).fit_transform(Gs_ensemble_reshape)
Gs_umap = UMAP(n_components = 2, min_dist = 0.8, random_state = 0, n_neighbors = 50).fit_transform(Gs_pca)

fig = plt.figure(figsize  = (10,7))
ax = fig.add_subplot()
ax.scatter(Gs_umap[:, 0], Gs_umap[:, 1], s = 10)
ax.set_xlabel("UMAP1")
ax.set_ylabel("UMAP2")
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False, markerscale = 3)

# In[]
# check the variation of hub-gene scores in different trajectory
Gs_ensemble = np.load(file = result_dir + f"graph_ensemble_{truncate_param}_{beta}.npy")
# hub_scores = np.sum(np.abs(Gs_ensemble), axis = 2)
# variance = np.mean((hub_scores - np.mean(hub_scores, axis = 0, keepdims = True))**2, axis = 0)
# variance does not mean the importance of genes
# np.savetxt(result_dir + "hub_gene_variance.txt", genes[np.argsort(variance)[:-100:-1]], fmt = "%s")

# sum of weight corresponding to each gene, the larger the weight, the more important the gene is, conduct GO analysis and find important GO term.
hub_scores = pd.DataFrame(columns = ["weight"])
hub_scores["weight"] = np.mean(np.sum(np.abs(Gs_ensemble), axis = 2), axis = 0)
hub_scores.index = genes
hub_scores = hub_scores.iloc[np.argsort(hub_scores["weight"].values)[::-1],:]
hub_scores.to_csv(result_dir + "hub_gene.csv")

# In[]
plt.rcParams["font.size"] = 25
from adjustText import adjust_text
fig = plt.figure(figsize = (10,7))
ax = fig.add_subplot()
sns.violinplot(data = hub_scores, y = "weight", color="0.8", ax = ax)
g = sns.stripplot(data = hub_scores, y = "weight", jitter=True, size = 7)
# texts = []
# for i in range(hub_scores.shape[0]):
#     if i <= 15:
#         texts.append(g.text(y=hub_scores["weight"].values[i], x = 0.001, s=hub_scores.index.values[i], horizontalalignment='right', color='black', fontsize = 15))
# adjust_text(texts, only_move={'points':'x', 'texts':'x'})
fig.savefig(result_dir + "hub_gene.png", bbox_inches = "tight")


# In[] Check the variance of edges during the differentiation process
tf_targets = []
for gene1 in genes:
    tf_targets.extend([gene1 + "--" + gene2 for gene2 in genes])
tf_targets = np.array(tf_targets)

tf_target_traverse = []
idx = []
for i, tf_target in enumerate(tf_targets):
    gene1, gene2 = tf_target.split("--")
    if (gene1 + "--" + gene2 not in tf_target_traverse) and (gene2 + "--" + gene1 not in tf_target_traverse) and (gene1 != gene2):
        tf_target_traverse.append(tf_target)
        idx.append(i)
idx = np.array(idx)

Gs_ensemble_reshape = Gs_ensemble.reshape(Gs_ensemble.shape[0], -1)
variances = np.sum((Gs_ensemble_reshape - np.mean(Gs_ensemble_reshape, axis = 0, keepdims = True)) ** 2, axis = 0)/Gs_ensemble_reshape.shape[0]

tf_targets = tf_targets[idx]
variances = variances[idx]
Gs_ensemble_reshape = Gs_ensemble_reshape[:, idx]

tf_targets_sorted = tf_targets[np.argsort(variances)[::-1]]
variances_sorted = variances[np.argsort(variances)[::-1]]

variances_sorted = pd.DataFrame(data = variances_sorted[:,None], columns = ["variance"], index = tf_targets_sorted)

variances_sorted.to_csv(result_dir + "edge_variance.csv")

# In[]
plt.rcParams["font.size"] = 25
from adjustText import adjust_text
fig = plt.figure(figsize = (10,7))
ax = fig.add_subplot()

sns.violinplot(data = variances_sorted, y = "variance", color="0.8", ax = ax)
variances_shown = variances_sorted.loc[np.array(["eve--ftz", "eve--zen", "ftz--odd", "fkh--ftz", "eve--odd", "odd--prd"]),:]
g = sns.stripplot(data = variances_shown, y = "variance", jitter=True, size = 7)
# texts = []
# for i in range(variances_shown.shape[0]):
#     marker = variances_shown.index.values[i]
#     texts.append(g.text(y=variances_shown["variance"].values[i], x = 0.001, s=marker.split("--")[0] + "-" + marker.split("--")[1], horizontalalignment='right', color='black', fontsize = 17))
# adjust_text(texts, only_move={'points':'x', 'texts':'x'})
fig.savefig(result_dir + "edge_variance.png", bbox_inches = "tight")

# In[]
# change faster, capture detailed pattern
Gs_ensemble_reshape = np.load(file = result_dir + f"graph_0.01_0.1_100_1.npy").reshape(Gs_ensemble.shape[0], -1)
Gs_ensemble_reshape = Gs_ensemble_reshape[:, idx]
fig = plt.figure(figsize = (10, 7))
ax = fig.subplots(nrows = 1, ncols = 1)
# visualize x and z coordinates
pic = ax.scatter(spaLoc.values[:, 0], spaLoc.values[:, 2], c = expr_norm[:, genes  == "ftz"].squeeze())
cbar = fig.colorbar(pic, fraction=0.046, pad=0.04, ax = ax)
cbar.ax.tick_params(labelsize = 20)
ax.set_title("ftz expression")

fig = plt.figure(figsize = (10, 7))
ax = fig.subplots(nrows = 1, ncols = 1)
# visualize x and z coordinates
pic = ax.scatter(spaLoc.values[:, 0], spaLoc.values[:, 2], c = expr_norm[:, genes  == "eve"].squeeze())
cbar = fig.colorbar(pic, fraction=0.046, pad=0.04, ax = ax)
cbar.ax.tick_params(labelsize = 20)
ax.set_title("eve expression")

fig = plt.figure(figsize = (10, 7))
ax = fig.subplots(nrows = 1, ncols = 1)
# visualize x and z coordinates
pic = ax.scatter(spaLoc.values[:, 0], spaLoc.values[:, 2], c = expr_norm[:, genes  == "zen"].squeeze())
cbar = fig.colorbar(pic, fraction=0.046, pad=0.04, ax = ax)
cbar.ax.tick_params(labelsize = 20)
ax.set_title("zen expression")

# Check the weight of edges
fig = plt.figure(figsize = (10, 7))
ax = fig.subplots(nrows = 1, ncols = 1)
# visualize x and z coordinates
pic = ax.scatter(spaLoc.values[:, 0], spaLoc.values[:, 2], c = np.abs(Gs_ensemble_reshape[:, tf_targets == "eve--ftz"].squeeze()))
cbar = fig.colorbar(pic, fraction=0.046, pad=0.04, ax = ax)
cbar.ax.tick_params(labelsize = 20)
ax.set_title("ftz-eve")
fig.savefig(result_dir + "ftz_eve.png", bbox_inches = "tight")


fig = plt.figure(figsize = (10, 7))
ax = fig.subplots(nrows = 1, ncols = 1)
# visualize x and z coordinates
pic = ax.scatter(spaLoc.values[:, 0], spaLoc.values[:, 2], c = np.abs(Gs_ensemble_reshape[:, tf_targets == "ftz--odd"].squeeze()))
cbar = fig.colorbar(pic, fraction=0.046, pad=0.04, ax = ax)
cbar.ax.tick_params(labelsize = 20)
ax.set_title("ftz-odd")
fig.savefig(result_dir + "ftz_odd.png", bbox_inches = "tight")

fig = plt.figure(figsize = (10, 7))
ax = fig.subplots(nrows = 1, ncols = 1)
# visualize x and z coordinates
pic = ax.scatter(spaLoc.values[:, 0], spaLoc.values[:, 2], c = np.abs(Gs_ensemble_reshape[:, tf_targets == "odd--prd"].squeeze()))
cbar = fig.colorbar(pic, fraction=0.046, pad=0.04, ax = ax)
cbar.ax.tick_params(labelsize = 20)
ax.set_title("prd-odd")
fig.savefig(result_dir + "prd_odd.png", bbox_inches = "tight")

# zen only expressed on the back, should have strong pattern from back to belly
fig = plt.figure(figsize = (10, 7))
ax = fig.subplots(nrows = 1, ncols = 1)
# visualize x and z coordinates
pic = ax.scatter(spaLoc.values[:, 0], spaLoc.values[:, 2], c = np.abs(Gs_ensemble_reshape[:, tf_targets == "eve--zen"].squeeze()))
cbar = fig.colorbar(pic, fraction=0.046, pad=0.04, ax = ax)
cbar.ax.tick_params(labelsize = 20)
ax.set_title("eve-zen")
fig.savefig(result_dir + "eve_zen.png", bbox_inches = "tight")

# In[]



# %%
