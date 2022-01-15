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
    beta = setting["beta"]

    score = pd.DataFrame(columns = ["model", "bandwidth", "truncate_param", "lambda", "beta", "density ratio", "kendall-tau", "pearson", \
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
                        "lambda":lamb, 
                        "beta": beta}, ignore_index=True)  

    return score


# path = "../../data/mESC/"
# result_path = "../results_mESC_small/"
# pca_op = PCA(n_components = 5)
# umap_op = UMAP(n_components = 2, min_dist = 0.8, n_neighbors = 30, random_state = 0)
# counts = pd.read_csv(path + "counts_small.csv", index_col = 0).values
# annotation = pd.read_csv(path + "anno.csv", index_col = 0)
# genes = pd.read_csv(path + "counts_small.csv", index_col = 0).columns.values


path = "../../data/mESC/"
result_path = "../results_mESC/"
pca_op = PCA(n_components = 5)
umap_op = UMAP(n_components = 2, min_dist = 0.8, n_neighbors = 30, random_state = 0)
counts = pd.read_csv(path + "counts.csv", index_col = 0).values
annotation = pd.read_csv(path + "anno.csv", index_col = 0)
genes = pd.read_csv(path + "counts.csv", index_col = 0).columns.values

# GENIE3
# libsize = np.median(np.sum(counts, axis = 1))
# counts = counts / np.sum(counts, axis = 1)[:,None] * libsize
# # the distribution of the original count is log-normal distribution, conduct log transform
# counts = np.log1p(counts)

# genie_theta = genie3.GENIE3(counts, gene_names=np.squeeze(genes.values), regulators='all',tree_method='RF',K='sqrt',ntrees=1000,nthreads=1)
# np.save(file = result_path + "theta_genie.npy", arr = genie_theta)

# In[] Test accuracy
# bandwidths = [0.1, 0.5, 1, 10]
bandwidths = [0.1, 1, 10]
truncate_params = [5, 15, 30]
lambs = [0.001, 0.01, 0.05, 0.1]
betas = [0.01, 0.1, 1, 10, 100]
scores = pd.DataFrame(columns = ["model", "bandwidth", "truncate_param", "lambda", "beta", "density ratio", "kendall-tau", "pearson", \
    "spearman", "cosine similarity", "AUPRC (pos)", "AUPRC (neg)", "AUPRC (abs)", "Early Precision (pos)", "Early Precision (neg)", "Early Precision (abs)", "AUPRC random (pos)",\
        "AUPRC random (neg)", "AUPRC random (abs)", "Early Precision random (pos)", "Early Precision random (neg)","Early Precision random (abs)","AUPRC Ratio (pos)", \
            "AUPRC Ratio (neg)", "AUPRC Ratio (abs)", "Early Precision Ratio (pos)", "Early Precision Ratio (neg)", "Early Precision Ratio (abs)"])

thetas_gt = pd.read_csv(path + "GRN_small.csv", index_col = 0).values


for bandwidth in bandwidths:
    for truncate_param in truncate_params:
        for lamb in lambs:
            print(str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param))
            thetas = np.load(file = result_path + "thetas_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_kt.npy") 
            # thetas = np.abs(thetas)
            thetas = np.sum(thetas, axis = 0)
            assert thetas.shape[0] == 44
            assert thetas.shape[1] == 44
            setting = {
                "theta_inf": thetas,
                "theta_gt": thetas_gt,
                "model": "CeSpGRN",
                "bandwidth": bandwidth,
                "truncate_param": truncate_param,
                "lamb": lamb,
                "beta": 0
            }
            scores = scores.append(calc_scores_static(setting), ignore_index = True)


for bandwidth in bandwidths:
    for truncate_param in truncate_params:
        for lamb in lambs:
            for beta in betas:
                print(str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_" + str(beta))
                thetas = np.load(file = result_path + "thetas_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_" + str(beta) + "_kt.npy") 
                # in case positive and negative cancel out each other? not very useful
                # thetas = np.abs(thetas)
                thetas = np.sum(thetas, axis = 0)
                assert thetas.shape[0] == 44
                assert thetas.shape[1] == 44
                setting = {
                    "theta_inf": thetas,
                    "theta_gt": thetas_gt,
                    "model": "CeSpGRN-TF",
                    "bandwidth": bandwidth,
                    "truncate_param": truncate_param,
                    "lamb": lamb,
                    "beta": beta
                }
                scores = scores.append(calc_scores_static(setting), ignore_index = True)


theta_genie = np.load(result_path + "theta_genie.npy")[0]
setting = {
    "theta_inf": theta_genie,
    "theta_gt": thetas_gt,
    "model": "GENIE3",
    "bandwidth": 0,
    "truncate_param": 0,
    "lamb": 0,
    "beta": 0
}
scores = scores.append(calc_scores_static(setting), ignore_index = True)

theta_genie = np.load(result_path + "theta_genie_tfs.npy")[0]
setting = {
    "theta_inf": theta_genie,
    "theta_gt": thetas_gt,
    "model": "GENIE3-TF",
    "bandwidth": 0,
    "truncate_param": 0,
    "lamb": 0,
    "beta": 0
}
scores = scores.append(calc_scores_static(setting), ignore_index = True)

# add CSN and SCODE
thetas_csn = np.load(result_path + "theta_CSN.npy")
assert thetas_csn.shape[0] == 2717
assert thetas_csn.shape[1] == 44
assert thetas_csn.shape[2] == 44


thetas_csn = np.sum(thetas_csn, axis = 0)
setting = {
    "theta_inf": thetas_csn,
    "theta_gt": thetas_gt,
    "model": "CSN",
    "bandwidth": 0,
    "truncate_param": 0,
    "lamb": 0,
    "beta": 0
}
scores = scores.append(calc_scores_static(setting), ignore_index = True)

# thetas_scode = np.load("../results_THP-1/theta_scode.npy")
# setting = {
#     "theta_inf": thetas_scode[0],
#     "theta_gt": thetas_gt,
#     "model": "SCODE",
#     "bandwidth": 0,
#     "truncate_param": 0,
#     "lamb": 0,
#     "beta": 0
# }
scores = scores.append(calc_scores_static(setting), ignore_index = True)

scores["AUPRC Ratio (signed)"] = (scores["AUPRC Ratio (pos)"].values + scores["AUPRC Ratio (neg)"].values)/2
scores["Early Precision Ratio (signed)"] = (scores["Early Precision Ratio (pos)"].values + scores["Early Precision Ratio (neg)"].values)/2
scores["AUPRC (signed)"] = (scores["AUPRC (pos)"].values + scores["AUPRC (neg)"].values)/2
scores["Early Precision (signed)"] = (scores["Early Precision (pos)"].values + scores["Early Precision (neg)"].values)/2

scores.to_csv(result_path + "scores.csv")  


# In[]
# According to above the best hyper-parameters for static network (from AUPRC ratio signed, ER ratio signed) is bandwidth = 1, lambda = 0.001 truncate_param = 5/15/30
# Check the gene with the most degree. Then see the overlap between the genes and the TFs
# bandwidth = 1
# truncate_param = 30
# lamb = 0.1
# beta = 100

bandwidth = 1
truncate_param = 5
lamb = 0.001
beta = 100
# thetas = np.load(file = result_path + "thetas_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_kt.npy")
# thetas = np.sum(thetas, axis = 0)
# # make the diagonal value of thetas 0
# np.fill_diagonal(thetas, 0)
# degree = np.sum(np.abs(thetas), axis = 1)
# rank = np.argsort(degree)[::-1]

# tfs = ['Pou5f1', 'Nr5a2', 'Sox2', 'Sall4', 'Otx2', 'Esrrb', 'Stat3','Tcf7', 'Nanog', 'Etv5']
# tf_ids = np.array([np.where(genes == x)[0][0] for x in tfs])

# print("degree")
# print(rank[tf_ids])
# the rank is not good except  Sall4
# [44 21 93  5 69 88 53 14 62 61]

thetas = np.load(file = result_path + "thetas_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_" + str(beta) + "_kt.npy")
thetas = np.sum(thetas, axis = 0)
# make the diagonal value of thetas 0
np.fill_diagonal(thetas, 0)
degree = np.sum(np.abs(thetas), axis = 1)
rank = np.argsort(degree)[::-1]

tfs = ['Pou5f1', 'Nr5a2', 'Sox2', 'Sall4', 'Otx2', 'Esrrb', 'Stat3','Tcf7', 'Nanog', 'Etv5']
tf_ids = np.array([np.where(genes == x)[0][0] for x in tfs])
print("degree")
print(rank[tf_ids])


# Check the variance of the edges
thetas = np.load(file = result_path + "thetas_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_kt.npy")

variance = np.sum((thetas - np.mean(thetas, axis = 0, keepdims=True)) ** 2, axis = 0)/thetas.shape[0]
# make the diagonal value of thetas 0
np.fill_diagonal(variance, 0)
sum_var = np.sum(np.abs(variance), axis = 1)
rank = np.argsort(sum_var)[::-1]

tfs = ['Pou5f1', 'Nr5a2', 'Sox2', 'Sall4', 'Otx2', 'Esrrb', 'Stat3','Tcf7', 'Nanog', 'Etv5']
tf_ids = np.array([np.where(genes == x)[0][0] for x in tfs])


thetas = np.load(file = result_path + "thetas_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_" + str(beta) + "_kt.npy")

variance = np.sum((thetas - np.mean(thetas, axis = 0, keepdims=True)) ** 2, axis = 0)/thetas.shape[0]
# make the diagonal value of thetas 0
np.fill_diagonal(variance, 0)
sum_var = np.sum(np.abs(variance), axis = 1)
rank = np.argsort(sum_var)[::-1]

tfs = ['Pou5f1', 'Nr5a2', 'Sox2', 'Sall4', 'Otx2', 'Esrrb', 'Stat3','Tcf7', 'Nanog', 'Etv5']
tf_ids = np.array([np.where(genes == x)[0][0] for x in tfs])
print("variance")
print(rank[tf_ids])

# find the edges that connect to Pou5f1, Sox2, Nanog. The variance of the edges, how the edges change over time.
# Pou5f1
print("Pou5f1")
var_pou5f1 = variance[tf_ids[0], :]
rank_pou5f1 = np.argsort(var_pou5f1)[::-1]
var_pou5f1 = var_pou5f1[rank_pou5f1]
rank_pou5f1 = genes[rank_pou5f1]
rank_pou5f1 = pd.DataFrame(index = rank_pou5f1, data = var_pou5f1[:, None], columns = ["variance"])
display(rank_pou5f1)

# Sox2
print("Sox2")
var_sox2 = variance[tf_ids[2], :]
rank_sox2 = np.argsort(var_sox2)[::-1]
var_sox2 = var_sox2[rank_sox2]
rank_sox2 = genes[rank_sox2]
rank_sox2 = pd.DataFrame(index = rank_sox2, data = var_sox2[:, None], columns = ["variance"])
display(rank_sox2)

# Nanog
print("Nanog")
var_nanog = variance[tf_ids[8], :]
rank_nanog = np.argsort(var_nanog)[::-1]
var_nanog = var_nanog[rank_nanog]
rank_nanog = genes[rank_nanog]
rank_nanog = pd.DataFrame(index = rank_nanog, data = var_nanog[:, None], columns = ["variance"])
display(rank_nanog)

# In[] Umap visualization of thetas
thetas_pca = umap_op.fit_transform(thetas.reshape(thetas.shape[0], -1))
thetas_df = pd.DataFrame(data = thetas.reshape(thetas.shape[0], -1))
thetas_pca_df = pd.DataFrame(data = thetas_pca.reshape(thetas_pca.shape[0], -1))

fig = plt.figure(figsize  = (10,7))
ax = fig.add_subplot()
for i in np.sort(np.unique(annotation.values.squeeze())):
    idx = np.where(annotation.values.squeeze() == i)
    ax.scatter(thetas_pca[idx, 0], thetas_pca[idx, 1], label = i, s = 10)
ax.set_xlabel("UMAP1")
ax.set_ylabel("UMAP2")
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False, markerscale = 3)
# ax.set_title("bandwidth: " + str(bandwidth) + ", truncate_param: " + str(truncate_param) + ", lamb: " + str(lamb))
fig.savefig("thetas_pca.png", bbox_inches = "tight")

# In[]
gene_pair = np.zeros((genes.shape[0],genes.shape[0])).astype(np.object)
for i, gene_i in enumerate(np.squeeze(genes)):
    for j, gene_j in enumerate(np.squeeze(genes)):
        gene_pair[i,j] = gene_i + "-" + gene_j
gene_pair = gene_pair.reshape(-1)

thetas_df = pd.DataFrame(data = thetas.reshape(thetas.shape[0], -1), columns = gene_pair)
pt = pd.read_csv(result_path + "de_edges/pt_slingshot.csv", header = None)
# thetas_df = pd.read_csv(result_path + "de_edges/thetas.csv", header = None)
pt.columns = ["Traj0"]
de_edges = de.de_analy(thetas_df, pt, distri = "normal", fdr_correct = True)
print(len(de_edges["Traj0"]))

# ordering of genes
sorted_pt = pt["Traj0"].dropna(axis = 0).sort_values()
# ordering = [int(x.split("_")[1]) for x in sorted_pt.index]
ordering = sorted_pt.index.values.squeeze()
X_traj = thetas_df.iloc[ordering, :]

# make plot
nrows = 2
ncols = 2
figsize = (20,10)
fig, axs = plt.subplots(nrows = nrows, ncols = ncols, figsize = figsize)
colormap = plt.cm.get_cmap('tab20b', 40)
idx = 0
for _, gene in enumerate(de_edges["Traj0"]):
    if gene["gene"] in ["Pou5f1-Nanog", "Sox2-Tcf7", "Esrrb-Nanog", "Sox2-Otx2"]:
        # plot log transformed version
        gene_dynamic = np.squeeze(X_traj.loc[:,gene["gene"]].values)
        pse_t = np.arange(gene_dynamic.shape[0])[:,np.newaxis]

        gene_null = gene['null']
        gene_pred = gene['regression']

        axs[idx%nrows, idx//nrows].scatter(np.arange(gene_dynamic.shape[0]), gene_dynamic, color = colormap(idx), alpha = 0.7)
        axs[idx%nrows, idx//nrows].plot(pse_t, gene_pred, color = "black", alpha = 1)
        axs[idx%nrows, idx//nrows].plot(pse_t, gene_null, color = "red", alpha = 1)
        axs[idx%nrows, idx//nrows].set_title(gene['gene'])


# In[]
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


# In[] The best hyper-parameter bandwidth = 0.1, truncate 30, lamb = 0.1/0.01
'''
scores = pd.read_csv(result_path + "scores.csv", index_col = 0)
bandwidth = 1
truncate_param = 30
lamb = 0.1
beta = 100

score_cespgrn = scores.loc[(scores["bandwidth"] == bandwidth)&(scores["truncate_param"] == truncate_param)&(scores["lambda"] == lamb)&(scores["beta"] == beta), ["density ratio", "AUPRC Ratio (abs)", "Early Precision Ratio (abs)"]]
score_genie = scores.loc[(scores["model"] == "GENIE3")|(scores["model"] == "GENIE3-TF"), ["density ratio", "AUPRC Ratio (abs)", "Early Precision Ratio (abs)"]]

display(score_cespgrn)
display(score_genie)

thetas = np.load(file = result_path + "thetas_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_" + str(beta) + "_kt.npy") 
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

# thetas_pca = pd.read_csv(result_path + "de_edges/thetas_pca.csv", header = None).values

# fig = plt.figure(figsize  = (10,7))
# ax = fig.add_subplot()
# t = pt.values.squeeze()
# pic = ax.scatter(thetas_pca[:, 0], thetas_pca[:, 1], c = t/np.max(t), s = 10)
# ax.set_xlabel("PCA1")
# ax.set_ylabel("PCA2")
# ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False, markerscale = 3)
# ax.set_title("bandwidth: " + str(bandwidth) + ", truncate_param: " + str(truncate_param) + ", lamb: " + str(lamb))
# plt.colorbar(pic)
# fig.savefig(result_path + "de_edges/thetas_pca_pt.png", bbox_inches = "tight")
# In[]
pt = pd.read_csv(result_path + "de_edges/pt_slingshot.csv", header = None)
# thetas_df = pd.read_csv(result_path + "de_edges/thetas.csv", header = None)
pt.columns = ["Traj0"]
de_edges = de.de_analy(thetas_df, pt, distri = "normal", fdr_correct = True)
print(len(de_edges["Traj0"]))
figs = de.de_plot(thetas_df, pt, de_edges, figsize = (20,80), n_genes = 40)


cut_off = 100
# already sorted
pvals = [x["p_val"] for x in de_edges["Traj0"]]
de_edges_selected = de_edges["Traj0"][:cut_off]
pvals_selected = pvals[:cut_off]
gene_pair = np.zeros((genes.shape[0],genes.shape[0])).astype(np.object)
for i, gene_i in enumerate(np.squeeze(genes)):
    for j, gene_j in enumerate(np.squeeze(genes)):
        gene_pair[i,j] = gene_i + "-" + gene_j
gene_pair = gene_pair.reshape(-1)
changing_idx = [x["gene"] for x in de_edges_selected]
changing_pair = gene_pair[changing_idx]
changing_pair_df = pd.DataFrame(data = changing_pair[:, None], columns = ["edges"])
changing_pair_df["p-vals"] = pvals_selected
changing_pair_df.to_csv(result_path + "de_edges/de_edges.csv", header = True, index = False)

# In[]
# binary matrix
thetas_gt = pd.read_csv(path + "GRN.csv", index_col = 0)
edges = []
# ground truth gene pairs
for i, gene_i in enumerate(np.squeeze(genes)):
    for j, gene_j in enumerate(np.squeeze(genes)):
        if thetas_gt.loc[gene_i, gene_j] != 0:
            edges.append(gene_i + "-" + gene_j)

# find overlapping pairs (detected changing pairs & ground truth gene pairs)
overlap = []
for edge in changing_pair:
    if edge in edges:
        overlap.append(edge)

overlap = pd.DataFrame(data = np.array(overlap)[:,None])
overlap.to_csv(result_path + "de_edges/overlap_edges.csv")
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
'''