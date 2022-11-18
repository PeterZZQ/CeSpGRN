# In[0]
from math import trunc
from operator import index
import pandas as pd
import numpy as np
import torch
import torch.nn
import sys, os
sys.path.append('../../src/')

import bmk_beeline as bmk
import de_analysis as de
import genie3

import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns

from umap import UMAP
from sklearn.decomposition import PCA

from multiprocessing import Pool, cpu_count
import time

import matplotlib.patheffects as path_effects

plt.rcParams["font.size"] = 15


def check_symmetric(a, rtol=1e-04, atol=1e-04):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def calc_scores_static(setting):
    theta_inf = setting["theta_inf"]
    theta_gt = setting["theta_gt"]
    model = setting["model"]
    bandwidth = setting["bandwidth"]
    truncate_param = setting["truncate_param"]
    lamb = setting["lamb"]

    score = pd.DataFrame(columns = ["model", "bandwidth", "truncate_param", "lambda","density ratio", "kendall-tau", "pearson", \
        "spearman", "cosine similarity", "AUPRC (pos)", "AUPRC (neg)", "AUPRC (abs)", "Early Precision (pos)", "Early Precision (neg)", "Early Precision (abs)", "AUPRC random (pos)",\
            "AUPRC random (neg)", "AUPRC random (abs)", "Early Precision random (pos)", "Early Precision random (neg)","Early Precision random (abs)","AUPRC Ratio (pos)", \
                "AUPRC Ratio (neg)", "AUPRC Ratio (abs)", "Early Precision Ratio (pos)", "Early Precision Ratio (neg)", "Early Precision Ratio (abs)"])

    ngenes = theta_inf.shape[0]
    np.random.seed(0)
    thetas_rand = np.random.randn(ngenes, ngenes)
    # make symmetric
    thetas_rand = (thetas_rand + thetas_rand.T)/2    
    assert check_symmetric(theta_gt)
    if not check_symmetric(theta_inf):
        theta_inf = (theta_inf + theta_inf.T)/2

    # CeSpGRN should infer symmetric matrix
    if (model == "CeSpGRN")|(model == "CeSpGRN-kt")|(model == "CeSpGRN-kt-TF"):
        assert check_symmetric(theta_inf) 

    pearson_val, _ = bmk.pearson(G_inf = theta_inf, G_true = theta_gt)
    kt, _ = bmk.kendalltau(G_inf = theta_inf, G_true = theta_gt)
    spearman_val, _ = bmk.spearman(G_inf = theta_inf, G_true = theta_gt)
    cosine_sim = bmk.cossim(G_inf = theta_inf, G_true = theta_gt)

    AUPRC_pos, AUPRC_neg = bmk.compute_auc_signed(G_inf = theta_inf, G_true = theta_gt)     
    AUPRC_pos_rand, AUPRC_neg_rand = bmk.compute_auc_signed(G_inf = thetas_rand, G_true = theta_gt)     
    AUPRC = bmk.compute_auc_abs(G_inf = theta_inf, G_true = theta_gt)
    AUPRC_rand = bmk.compute_auc_abs(G_inf = thetas_rand, G_true = theta_gt)

    Eprec_pos, Eprec_neg = bmk.compute_eprec_signed(G_inf = theta_inf, G_true = theta_gt)
    Eprec_pos_rand, Eprec_neg_rand = bmk.compute_eprec_signed(G_inf = thetas_rand, G_true = theta_gt)
    Eprec = bmk.compute_eprec_abs(G_inf = theta_inf, G_true = theta_gt)
    Eprec_rand = bmk.compute_eprec_abs(G_inf = thetas_rand, G_true = theta_gt)
    
    score = score.append({"pearson": pearson_val, 
                        "kendall-tau": kt,
                        "spearman": spearman_val,
                        "cosine similarity": cosine_sim, 
                        "AUPRC (pos)": AUPRC_pos,
                        "AUPRC (neg)": AUPRC_neg,
                        "AUPRC (abs)": AUPRC,
                        "Early Precision (pos)": Eprec_pos,
                        "Early Precision (neg)": Eprec_neg,
                        "Early Precision (abs)":Eprec,
                        "AUPRC random (pos)": AUPRC_pos_rand,
                        "AUPRC random (neg)": AUPRC_neg_rand,
                        "AUPRC random (abs)": AUPRC_rand,
                        "Early Precision random (pos)": Eprec_pos_rand,
                        "Early Precision random (neg)": Eprec_neg_rand,
                        "Early Precision random (abs)":Eprec_rand,
                        "AUPRC Ratio (pos)": AUPRC_pos/AUPRC_pos_rand,
                        "AUPRC Ratio (neg)": AUPRC_neg/AUPRC_neg_rand,
                        "AUPRC Ratio (abs)": AUPRC/AUPRC_rand,
                        "Early Precision Ratio (pos)": Eprec_pos/(Eprec_pos_rand + 1e-12),
                        "Early Precision Ratio (neg)": Eprec_neg/(Eprec_neg_rand + 1e-12),
                        "Early Precision Ratio (abs)":Eprec/(Eprec_rand + 1e-12),
                        "density ratio": np.sum(theta_inf!=0)/np.sum(theta_gt!=0),
                        "model": model,
                        "bandwidth": bandwidth,
                        "truncate_param":truncate_param,
                        "lambda":lamb}, ignore_index=True)  

    return score


ntimes = 1000
nsample = 1
path = "../../data/COUNTS-THP-1/"
result_path = "../results_THP-1_kt/"

pca_op = PCA(n_components = 5)
umap_op = UMAP(n_components = 2, min_dist = 0.8, n_neighbors = 30, random_state = 0)

counts = pd.read_csv(path + "counts.csv", index_col = 0).values
annotation = pd.read_csv(path + "anno.csv", index_col = 0)
genes = pd.read_csv(path + "genes.csv", header = None)

# GENIE3
# libsize = np.median(np.sum(counts, axis = 1))
# counts = counts / np.sum(counts, axis = 1)[:,None] * libsize
# # the distribution of the original count is log-normal distribution, conduct log transform
# counts = np.log1p(counts)

# genie_theta = genie3.GENIE3(counts, gene_names=np.squeeze(genes.values), regulators='all',tree_method='RF',K='sqrt',ntrees=1000,nthreads=1)
# np.save(file = result_path + "theta_genie.npy", arr = genie_theta)


# In[] generated perturbation matrix
# perturbation = pd.read_excel(path + "perturbation.xlsx")

# perturb_genes = []
# for row in range(perturbation.shape[0]):
#     row_val = perturbation.iloc[row, :]
#     perturb_genes.append(row_val["Input gene"])
#     perturb_genes.extend([x for x in row_val.iloc[2:] if isinstance(x, str)])

# perturb_genes = sorted(set(perturb_genes))
# ptb_x = pd.DataFrame(data = 0, index = perturb_genes, columns = perturb_genes)

# for row in range(perturbation.shape[0]):
#     row_val = perturbation.iloc[row, :]
#     regulator = row_val["Input gene"]
#     targets = [x for x in row_val.iloc[2:] if isinstance(x, str)]
#     if row_val["direction"] == "activate":
#         ptb_x.loc[regulator, targets] = 1
#         # make symmetric
#         ptb_x.loc[targets, regulator] = 1
#     else:
#         ptb_x.loc[regulator, targets] = -1
#         # make symmetric
#         ptb_x.loc[targets, regulator] = -1

# ptb_x = ptb_x.loc[np.squeeze(genes.values), np.squeeze(genes.values)]
# ptb_x.to_csv(path + "GRN_asymm.csv")


# In[] Test accuracy
bandwidths = [0.01, 0.1, 0.2, 0.5, 1, 10]
truncate_params = [5, 15, 30]
lambs = [0.001, 0.002, 0.005, 0.01, 0.05, 0.1]
scores = pd.DataFrame(columns = ["model", "bandwidth", "truncate_param", "lambda","density ratio", "kendall-tau", "pearson", \
    "spearman", "cosine similarity", "AUPRC (pos)", "AUPRC (neg)", "AUPRC (abs)", "Early Precision (pos)", "Early Precision (neg)", "Early Precision (abs)", "AUPRC random (pos)",\
        "AUPRC random (neg)", "AUPRC random (abs)", "Early Precision random (pos)", "Early Precision random (neg)","Early Precision random (abs)","AUPRC Ratio (pos)", \
            "AUPRC Ratio (neg)", "AUPRC Ratio (abs)", "Early Precision Ratio (pos)", "Early Precision Ratio (neg)", "Early Precision Ratio (abs)"])

thetas_gt = pd.read_csv(path + "GRN.csv", index_col = 0).values

for bandwidth in bandwidths:
    for truncate_param in truncate_params:
        for lamb in lambs:
            print(str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param))
            thetas = np.load(file = result_path + "thetas_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_kt.npy") 
            thetas = np.sum(thetas, axis = 0)
            assert thetas.shape[0] == 45
            assert thetas.shape[1] == 45
            setting = {
                "theta_inf": thetas,
                "theta_gt": thetas_gt,
                "model": "CeSpGRN",
                "bandwidth": bandwidth,
                "truncate_param": truncate_param,
                "lamb": lamb
            }
            scores = scores.append(calc_scores_static(setting), ignore_index = True)

theta_genie = np.load(result_path + "theta_genie.npy")
setting = {
    "theta_inf": theta_genie,
    "theta_gt": thetas_gt,
    "model": "GENIE3",
    "bandwidth": 0,
    "truncate_param": 0,
    "lamb": 0
}
scores = scores.append(calc_scores_static(setting), ignore_index = True)

# add CSN and SCODE
thetas_csn = np.load("../results_THP-1/theta_CSN.npy")
thetas_csn = np.sum(thetas_csn, axis = 0)
setting = {
    "theta_inf": thetas_csn,
    "theta_gt": thetas_gt,
    "model": "CSN",
    "bandwidth": 0,
    "truncate_param": 0,
    "lamb": 0
}
scores = scores.append(calc_scores_static(setting), ignore_index = True)

thetas_scode = np.load("../results_THP-1/theta_scode.npy")
setting = {
    "theta_inf": thetas_scode[0],
    "theta_gt": thetas_gt,
    "model": "SCODE",
    "bandwidth": 0,
    "truncate_param": 0,
    "lamb": 0
}
scores = scores.append(calc_scores_static(setting), ignore_index = True)

scores["AUPRC Ratio (signed)"] = (scores["AUPRC Ratio (pos)"].values + scores["AUPRC Ratio (neg)"].values)/2
scores["Early Precision Ratio (signed)"] = (scores["Early Precision Ratio (pos)"].values + scores["Early Precision Ratio (neg)"].values)/2
scores["AUPRC (signed)"] = (scores["AUPRC (pos)"].values + scores["AUPRC (neg)"].values)/2
scores["Early Precision (signed)"] = (scores["Early Precision (pos)"].values + scores["Early Precision (neg)"].values)/2

# scores.to_csv(result_path + "scores.csv")  


# In[] check the influence of lambda, lambda = 0.001, 0.002 and 0.005, 0.01 have similar performance. 
# The other are not good.

scores = pd.read_csv(result_path + "scores.csv", index_col = 0)

fig = plt.figure(figsize = (15, 7))
ax = fig.subplots(nrows = 1, ncols = 3)

sns.boxplot(data = scores, x = "lambda", y = "AUPRC (signed)", ax = ax[0])
sns.boxplot(data = scores, x = "lambda", y = "Early Precision (signed)", ax = ax[1])
sns.boxplot(data = scores, x = "lambda", y = "density ratio", ax = ax[2])
plt.tight_layout()


# check the influence of bandwidth, bandwidth = 0.1/0.2 has the highest accuracy, 0.5 comes the next
scores = pd.read_csv(result_path + "scores.csv", index_col = 0)
# scores = scores[(scores["lambda"] != 0.05)&(scores["lambda"] != 0.1)]
fig = plt.figure(figsize = (15, 7))
ax = fig.subplots(nrows = 1, ncols = 3)

sns.boxplot(data = scores, x = "bandwidth", y = "AUPRC (signed)", ax = ax[0])
sns.boxplot(data = scores, x = "bandwidth", y = "Early Precision (signed)", ax = ax[1])
sns.boxplot(data = scores, x = "bandwidth", y = "density ratio", ax = ax[2])
plt.tight_layout()

# check neighborhood size, 30 is the best with AUPRC
scores = pd.read_csv(result_path + "scores.csv", index_col = 0)
# scores = scores[(scores["lambda"] != 0.05)&(scores["lambda"] != 0.1)]
fig = plt.figure(figsize = (15, 7))
ax = fig.subplots(nrows = 1, ncols = 3)

sns.boxplot(data = scores, x = "bandwidth", y = "AUPRC (signed)", hue = "truncate_param", ax = ax[0])
sns.boxplot(data = scores, x = "bandwidth", y = "Early Precision (signed)", hue = "truncate_param", ax = ax[1])
sns.boxplot(data = scores, x = "bandwidth", y = "density ratio", hue = "truncate_param", ax = ax[2])
plt.tight_layout()


# In[] The best hyper-parameter bandwidth = 0.1, truncate 30, lamb = 0.1/0.01
bandwidth = 0.1
truncate_param = 30
lamb = 0.1

pt = pd.read_csv(result_path + "de_edges/pt_slingshot_0.1_0.1_30.csv", header = None)
score_cespgrn = scores.loc[(scores["bandwidth"] == bandwidth)&(scores["truncate_param"] == truncate_param)&(scores["lambda"] == lamb), ["density ratio", "AUPRC Ratio (pos)", "AUPRC Ratio (neg)", "AUPRC Ratio (signed)", "Early Precision Ratio (pos)", "Early Precision Ratio (neg)", "Early Precision Ratio (signed)", "AUPRC (pos)", "AUPRC (neg)", "AUPRC (signed)", "Early Precision (pos)", "Early Precision (neg)", "Early Precision (signed)"]]
score_genie = scores.loc[(scores["model"] == "GENIE3"), ["density ratio","AUPRC Ratio (pos)", "AUPRC Ratio (neg)", "AUPRC Ratio (signed)", "Early Precision Ratio (pos)", "Early Precision Ratio (neg)", "Early Precision Ratio (signed)", "AUPRC (pos)", "AUPRC (neg)", "AUPRC (signed)", "Early Precision (pos)", "Early Precision (neg)", "Early Precision (signed)"]]

display(score_cespgrn)
display(score_genie)

thetas = np.load(file = result_path + "thetas_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_kt.npy") 
thetas_pca = pca_op.fit_transform(thetas.reshape(thetas.shape[0], -1))
thetas_df = pd.DataFrame(data = thetas.reshape(thetas.shape[0], -1))
thetas_df.to_csv(result_path + "de_edges/thetas.csv", index = False, header = False)
thetas_pca_df = pd.DataFrame(data = thetas_pca.reshape(thetas_pca.shape[0], -1))
thetas_pca_df.to_csv(result_path + "de_edges/thetas_pca.csv", index = False, header = False)

fig = plt.figure(figsize  = (10,7))
ax = fig.add_subplot()
for i in np.sort(np.unique(annotation.values.squeeze())):
    idx = np.where(annotation.values.squeeze() == i)
    ax.scatter(thetas_pca[idx, 0], thetas_pca[idx, 1], label = i, s = 10)
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False, markerscale = 3)
ax.set_title("bandwidth: " + str(bandwidth) + ", truncate_param: " + str(truncate_param) + ", lamb: " + str(lamb))
ax.savefig()

thetas_pca = pd.read_csv(result_path + "de_edges/thetas_pca_0.1_0.1_30.csv", header = None).values

fig = plt.figure(figsize  = (10,7))
ax = fig.add_subplot()
t = pt.values.squeeze()
pic = ax.scatter(thetas_pca[:, 0], thetas_pca[:, 1], c = t/np.max(t), s = 10)
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False, markerscale = 3)
ax.set_title("bandwidth: " + str(bandwidth) + ", truncate_param: " + str(truncate_param) + ", lamb: " + str(lamb))
plt.colorbar(pic)
fig.savefig(result_path + "de_edges/thetas_pca_pt_0.1_0.1_30.png", bbox_inches = "tight")
# In[]
pt = pd.read_csv(result_path + "de_edges/pt_slingshot_0.1_0.1_30.csv", header = None)
thetas_df = pd.read_csv(result_path + "de_edges/thetas_0.1_0.1_30.csv", header = None)
pt.columns = ["Traj0"]
de_edges = de.de_analy(thetas_df, pt, distri = "normal", fdr_correct = True)
print(len(de_edges["Traj0"]))
figs = de.de_plot(thetas_df, pt, de_edges, figsize = (20,80), n_genes = 40)


cut_off = 100
# already sorted
pvals = [x["p_val"] for x in de_edges["Traj0"]]
de_edges_selected = de_edges["Traj0"][:cut_off]
gene_pair = np.zeros((genes.shape[0],genes.shape[0])).astype(np.object)
for i, gene_i in enumerate(np.squeeze(genes.values)):
    for j, gene_j in enumerate(np.squeeze(genes.values)):
        gene_pair[i,j] = gene_i + "-" + gene_j
gene_pair = gene_pair.reshape(-1)
changing_idx = [x["gene"] for x in de_edges_selected]
changing_pair = gene_pair[changing_idx]

# In[]
thetas_gt = pd.read_csv(path + "GRN_asymm.csv", index_col = 0)
pos_edges = []
neg_edges = []
# ground truth gene pairs
for i, gene_i in enumerate(np.squeeze(genes.values)):
    for j, gene_j in enumerate(np.squeeze(genes.values)):
        if thetas_gt.loc[gene_i, gene_j] > 0:
            pos_edges.append(gene_i + "-" + gene_j)
        elif thetas_gt.loc[gene_i, gene_j] < 0:
            neg_edges.append(gene_i + "-" + gene_j)


# find overlapping pairs (detected changing pairs & ground truth gene pairs)
pos_overlap = []
neg_overlap = []
for edge in changing_pair:
    if edge in pos_edges:
        pos_overlap.append(edge)
    elif edge in neg_edges:
        neg_overlap.append(edge)

overlap = pos_overlap + neg_overlap

pos_overlap = pd.DataFrame(data = np.array(pos_overlap)[:,None])
neg_overlap = pd.DataFrame(data = np.array(neg_overlap)[:,None])
overlap = pd.DataFrame(data = np.array(overlap)[:,None])

pos_overlap.to_csv(result_path + "de_edges/pos_overlap_0.1_0.1_30.csv")
neg_overlap.to_csv(result_path + "de_edges/neg_overlap_0.1_0.1_30.csv")
overlap.to_csv(result_path + "de_edges/overlap_0.1_0.1_30.csv")
# In[]
nrows = 20
ncols = 2
fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize = (14, 75))

for i, edge in enumerate(overlap.values.squeeze()):
    idx = np.where(gene_pair == edge)[0][0]
    de_edge = de_edges_selected[np.where(changing_idx == idx)[0][0]]
    edge_dynamics = np.squeeze(thetas_df.iloc[:,idx].values)
    pseudotime = np.squeeze(pt.values)

    axs[i%nrows, i//nrows].scatter(np.arange(edge_dynamics.shape[0]), edge_dynamics[np.argsort(pseudotime)], alpha = 0.7)
    axs[i%nrows, i//nrows].plot(np.arange(edge_dynamics.shape[0]), de_edge['null'], color = "red", alpha = 0.7)
    axs[i%nrows, i//nrows].plot(np.arange(edge_dynamics.shape[0]), de_edge['regression'], color = "black", alpha = 0.7)
    axs[i%nrows, i//nrows].set_title(edge)
plt.tight_layout()
fig.savefig(result_path + "de_edges/trend_0.1_0.1_30.png", bbox_inches = "tight")

# In[]
activate = pd.read_csv(result_path + "de_edges/activating_genes.txt", header = None).values
activate = [x for x in activate.reshape(-1)]

repress = pd.read_csv(result_path + "de_edges/repressing_genes.txt", header = None).values
repress = [x for x in repress.reshape(-1)]

# %%
