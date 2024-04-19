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

data_dir = "../../data/mESC/"
result_dir = "../results_mESC_96genes/"
pca_op = PCA(n_components = 5)
umap_op = UMAP(n_components = 2, min_dist = 0.8, n_neighbors = 30, random_state = 0)
counts = pd.read_csv(data_dir + "counts_96.csv", index_col = 0).values
annotation = pd.read_csv(data_dir + "anno.csv", index_col = 0)
genes = pd.read_csv(data_dir + "counts_96.csv", index_col = 0).columns.values

# GENIE3
# libsize = np.median(np.sum(counts, axis = 1))
# counts = counts / np.sum(counts, axis = 1)[:,None] * libsize
# # the distribution of the original count is log-normal distribution, conduct log transform
# counts = np.log1p(counts)

# genie_theta = genie3.GENIE3(counts, gene_names=np.squeeze(genes.values), regulators='all',tree_method='RF',K='sqrt',ntrees=1000,nthreads=1)
# np.save(file = result_path + "theta_genie.npy", arr = genie_theta)

# In[] Test accuracy
truncate_param = 100
beta = 0
scores = pd.DataFrame(columns = ["model", "bandwidth", "truncate_param", "lambda", "beta", "density ratio", "kendall-tau", "pearson", \
    "spearman", "cosine similarity", "AUPRC (pos)", "AUPRC (neg)", "AUPRC (abs)", "Early Precision (pos)", "Early Precision (neg)", "Early Precision (abs)", "AUPRC random (pos)",\
        "AUPRC random (neg)", "AUPRC random (abs)", "Early Precision random (pos)", "Early Precision random (neg)","Early Precision random (abs)","AUPRC Ratio (pos)", \
            "AUPRC Ratio (neg)", "AUPRC Ratio (abs)", "Early Precision Ratio (pos)", "Early Precision Ratio (neg)", "Early Precision Ratio (abs)"])

thetas_gt = pd.read_csv(data_dir + "GRN_96.csv", index_col = 0).values

thetas = np.load(file = result_dir + f"beta_0/graph_ensemble_{truncate_param}_{beta}.npy") 
# thetas = np.abs(thetas)
thetas = np.sum(thetas, axis = 0)
assert thetas.shape[0] == 96
assert thetas.shape[1] == 96
setting = {
    "theta_inf": thetas,
    "theta_gt": thetas_gt,
    "model": "CeSpGRN",
    "bandwidth": None,
    "truncate_param": truncate_param,
    "lamb": None,
    "beta": 0
}
scores = scores.append(calc_scores_static(setting), ignore_index = True)

truncate_param = 100
beta = 1
thetas = np.load(file = result_dir + f"beta_1/graph_ensemble_{truncate_param}_{beta}.npy") 
# thetas = np.abs(thetas)
thetas = np.sum(thetas, axis = 0)
assert thetas.shape[0] == 96
assert thetas.shape[1] == 96
setting = {
    "theta_inf": thetas,
    "theta_gt": thetas_gt,
    "model": "CeSpGRN (mask)",
    "bandwidth": None,
    "truncate_param": truncate_param,
    "lamb": None,
    "beta": 1
}
scores = scores.append(calc_scores_static(setting), ignore_index = True)

theta_genie = np.load(result_dir + "theta_genie.npy")[0]
setting = {
    "theta_inf": theta_genie,
    "theta_gt": thetas_gt,
    "model": "GENIE3",
    "bandwidth": None,
    "truncate_param": 0,
    "lamb": None,
    "beta": 0
}
scores = scores.append(calc_scores_static(setting), ignore_index = True)

theta_genie = np.load(result_dir + "theta_genie_tfs.npy")[0]
setting = {
    "theta_inf": theta_genie,
    "theta_gt": thetas_gt,
    "model": "GENIE3-TF",
    "bandwidth": None,
    "truncate_param": 0,
    "lamb": None,
    "beta": 0
}
scores = scores.append(calc_scores_static(setting), ignore_index = True)

# add CSN and SCODE
thetas_csn = np.load(result_dir + "theta_CSN.npy")
assert thetas_csn.shape[0] == 2717
assert thetas_csn.shape[1] == 96
assert thetas_csn.shape[2] == 96


thetas_csn = np.sum(thetas_csn, axis = 0)
setting = {
    "theta_inf": thetas_csn,
    "theta_gt": thetas_gt,
    "model": "CSN",
    "bandwidth": None,
    "truncate_param": 0,
    "lamb": None,
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
# scores = scores.append(calc_scores_static(setting), ignore_index = True)

scores["AUPRC Ratio (signed)"] = (scores["AUPRC Ratio (pos)"].values + scores["AUPRC Ratio (neg)"].values)/2
scores["Early Precision Ratio (signed)"] = (scores["Early Precision Ratio (pos)"].values + scores["Early Precision Ratio (neg)"].values)/2
scores["AUPRC (signed)"] = (scores["AUPRC (pos)"].values + scores["AUPRC (neg)"].values)/2
scores["Early Precision (signed)"] = (scores["Early Precision (pos)"].values + scores["Early Precision (neg)"].values)/2

scores.to_csv(result_dir + "scores.csv")  


# %%
