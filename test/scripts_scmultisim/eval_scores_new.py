# In[]
import numpy as np 
import sys, os
PROJECT_DIR = "/localscratch/ziqi/CeSpGRN/"
sys.path.append(PROJECT_DIR + 'src/')
import bmk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib 

import warnings
warnings.filterwarnings("ignore")

# In[]
# ---------------------------------------
#
# Load the dataset
#
# ---------------------------------------
datasets = [
    "simulated_8000_20_10_100_0.01_0_0_4",
    "simulated_8000_20_10_100_0.01_1_0_4",
    "simulated_8000_20_10_100_0.01_2_0_4",
    "simulated_8000_20_10_100_0.1_0_0_4",
    "simulated_8000_20_10_100_0.1_1_0_4",
    "simulated_8000_20_10_100_0.1_2_0_4"
]

dataset = datasets[5]
ncells = 8000
ngenes = 110

# In[]
# ---------------------------------------
#
# SCENIC+
#
# ---------------------------------------
# NOTE: only one time
'''
Gs_gt = []
for i in range(1, ncells + 1):
    grn_gt = pd.read_csv(path + f"grn_gts/grn_gt_{i}.txt", sep = "\t",index_col = 0)
    G_gt = np.zeros((ngenes + 1, ngenes + 1))
    G_gt[np.ix_(grn_gt.index.values.squeeze().astype(int), grn_gt.columns.values.squeeze().astype(int))] = grn_gt.values
    G_gt += G_gt.T
    Gs_gt.append(G_gt[None,1:,1:])

Gs_gt = np.concatenate(Gs_gt, axis = 0)
np.save(file = path + "graph_gt.npy", arr = Gs_gt)
'''

print("Calculating score (SCENIC+)...")
path = PROJECT_DIR + f"data/scMultiSim/{dataset}/scenicplus/"
result_dir = PROJECT_DIR + f"test/results_scmultisim/{dataset}/scenicplus/"

Gs_gt = np.load(file = path + "graph_gt.npy")
Gs_scenicplus = np.load(file = result_dir + "Gs_scenicplus.npy")
scores_scenicplus = bmk.calc_scores_para(thetas_inf = Gs_scenicplus, thetas_gt = Gs_gt, interval = None, model = "SCENIC+", bandwidth = None, truncate_param = 100, lamb = None, beta = 1, njobs = 16)
scores_scenicplus.to_csv(result_dir + "scores_scenicplus.csv")

# In[]
# ---------------------------------------
#
# scMultiomeGRN
#
# ---------------------------------------
print("Calculating score (scMultiomeGRN)...")
path = PROJECT_DIR + f"data/scMultiSim/{dataset}/scenicplus/"
result_dir = PROJECT_DIR + f"test/results_scmultisim/{dataset}/scmultiomeGRN/"

Gs_gt = np.load(file = path + "graph_gt.npy")
Gs_scmultiomegrn = np.load(file = result_dir + "Gs_scmultiomegrn.npy")
scores_scmultiomegrn = bmk.calc_scores_para(thetas_inf = Gs_scmultiomegrn, thetas_gt = Gs_gt, interval = None, model = "scmultiomeGRN", bandwidth = None, truncate_param = 100, lamb = None, beta = 1, njobs = 16)
scores_scmultiomegrn.to_csv(result_dir + "scores_scmultiomegrn.csv")


# In[]
# ---------------------------------------
#
# LocCSN
#
# ---------------------------------------
path = PROJECT_DIR + f"data/scMultiSim/{dataset}/"
result_dir = PROJECT_DIR + f"test/results_scmultisim/{dataset}/loccsn/"

Gs_gt = np.load(file = path + "graph_gt.npy")
print("Calculate score (LocCSN)...")
Gs_loccsn = np.load(file = result_dir + "Gs_loccsn.npy")
scores_loccsn = bmk.calc_scores_para(thetas_inf = Gs_loccsn, thetas_gt = Gs_gt, interval = None, model = "LocCSN", bandwidth = None, truncate_param = 100, lamb = None, beta = 1, njobs = 16)
scores_loccsn.to_csv(result_dir + "scores_loccsn.csv")


# In[]
# ---------------------------------------
#
# scores with scATAC-seq data
#
# ---------------------------------------
sns.set_theme(font_scale = 1.7)
scores = pd.DataFrame() 
plt.rcParams["font.size"] = 17

for nchanging_edges in [20]:
    for fp in [0.01, 0.1]:
        for seed in [0, 1, 2]:
            result_dir = PROJECT_DIR + f"test/results_scmultisim/simulated_8000_{nchanging_edges}_10_100_{fp}_{seed}_0_4/"
            score = pd.read_csv(result_dir + "scores.csv", index_col = 0)
            score_scenicplus = pd.read_csv(result_dir + "scenicplus/scores_scenicplus.csv", index_col = 0)
            scores_scmultiomegrn = pd.read_csv(result_dir + "scmultiomeGRN/scores_scmultiomegrn.csv", index_col = 0)
            
            score = pd.concat([score, score_scenicplus, scores_scmultiomegrn], axis = 0, ignore_index = True)
            score["fp"] = fp
            scores = pd.concat([scores, score], axis = 0, ignore_index = True)


fig = plt.figure(figsize = (23,7))
ax = fig.subplots(nrows =1, ncols = 2)
scores = scores[(scores["model"] == "CellOracle") | (scores["model"] == "scMTNI (ATAC)") | (scores["model"] == "CeSpGRN (ATAC)") | (scores["model"] == "SCENIC+") | (scores["model"] == "scmultiomeGRN")]
# error_bar = ("pi", 100)
error_bar = "sd"
bar1 = sns.barplot(scores, x = "model", y = "AUPRC (abs)", hue = "fp", ax = ax[0], order = ["CeSpGRN (ATAC)", "SCENIC+", "scMTNI (ATAC)", "CellOracle", "scmultiomeGRN"], palette = "Set2", estimator=np.mean, errorbar=error_bar, capsize = 0.1, err_kws = {"color": "blue", "linewidth": 2, "linestyle": "--"})
bar2 = sns.barplot(scores, x = "model", y = "Early Precision (abs)", hue = "fp", ax = ax[1], order = ["CeSpGRN (ATAC)", "SCENIC+", "scMTNI (ATAC)", "CellOracle", "scmultiomeGRN"], palette = "Set2", estimator=np.mean, errorbar=error_bar, capsize = 0.1, err_kws = {"color": "blue", "linewidth": 2, "linestyle": "--"})

for i in bar1.containers:
    bar1.bar_label(i, fmt='%.3f')  
for i in bar2.containers:
    bar2.bar_label(i, fmt='%.3f')

_ = ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation = 45, fontsize = 20)
_ = ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation = 45, fontsize = 20)
ax[0].get_legend().remove()
ax[0].set_xlabel(None)
ax[0].set_ylabel("AUPRC", fontsize = 20)
ax[1].set_xlabel(None)
ax[1].set_ylabel("Eprec", fontsize = 20)
leg = ax[1].legend(loc='upper left', prop={'size': 20}, frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1), markerscale = 6, title = "Noise level", title_fontsize = 20)
plt.tight_layout()

fig.savefig("../results_scmultisim/acc_atac_ext.png", bbox_inches = "tight")

# In[]
# ---------------------------------------
#
# scores with TF information
#
# ---------------------------------------
sns.set_theme(font_scale = 1.7)
scores = pd.DataFrame() 
plt.rcParams["font.size"] = 17

for nchanging_edges in [20]:
    # fp is only valid for scATAC-seq data
    for fp in [0.01]:
        for seed in [0, 1, 2]:
            result_dir = f"../results_scmultisim/simulated_8000_{nchanging_edges}_10_100_{fp}_{seed}_0_4/"
            score = pd.read_csv(result_dir + "scores_wcsn.csv", index_col = 0)
            score_loccsn = pd.read_csv(result_dir + "loccsn/scores_loccsn.csv", index_col = 0)
            
            score["fp"] = fp
            score_loccsn["fp"] = fp

            scores = pd.concat([scores, score, score_loccsn], axis = 0, ignore_index = True)

scores = scores[(scores["model"] == "GENIE3 (TF)") | (scores["model"] == "scMTNI (TF)") | (scores["model"] == "CeSpGRN (TF)") | (scores["model"] == "CSN") | (scores["model"] == "SCODE") | (scores["model"] == "LocCSN")]
scores.loc[scores["model"] == "GENIE3 (TF)", "model"] = "GENIE3"
fig = plt.figure(figsize = (15,7))
ax = fig.subplots(nrows =1, ncols = 2)
# error_bar = ("pi", 100)
error_bar = "sd"
bar1 = sns.barplot(scores[scores["model"] != "CSN"], x = "model", y = "AUPRC (abs)", hue = "fp", ax = ax[0], order = ["CeSpGRN (TF)", "scMTNI (TF)", "GENIE3", "LocCSN", "SCODE"], width=0.5, palette = "Set2", estimator=np.mean, errorbar=error_bar, capsize = 0.1, err_kws = {"color": "blue", "linewidth": 2, "linestyle": "--"})
for i in bar1.containers:
    bar1.bar_label(i, fmt = "%.3f")
bar2 = sns.barplot(scores, x = "model", y = "Early Precision (abs)", hue = "fp", ax = ax[1], order = ["CeSpGRN (TF)", "scMTNI (TF)", "GENIE3", "CSN", "LocCSN", "SCODE"], width=0.5, palette = "Set2", estimator=np.mean, errorbar=error_bar, capsize = 0.1, err_kws = {"color": "blue", "linewidth": 2, "linestyle": "--"})
for i in bar2.containers:
    bar2.bar_label(i, fmt = "%.3f")

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
fig.savefig("../results_scmultisim/acc_tf_ext.png", bbox_inches = "tight")



# In[]
# ---------------------------------------
#
# NOTE: plot the ablation test result
#
# ---------------------------------------
# 1. ablation test on beta
datasets = [
    "simulated_8000_20_10_100_0.01_0_0_4",
    "simulated_8000_20_10_100_0.01_1_0_4",
    "simulated_8000_20_10_100_0.01_2_0_4",
    "simulated_8000_20_10_100_0.1_0_0_4",
    "simulated_8000_20_10_100_0.1_1_0_4",
    "simulated_8000_20_10_100_0.1_2_0_4"
]

for dataset in datasets:
    if os.path.exists(PROJECT_DIR + f"test/results_scmultisim/{dataset}/scores_ablation_beta.csv"):
        continue
    Gs_gt = np.load(file = PROJECT_DIR + f"data/scMultiSim/{dataset}/graph_gt.npy")
    # beta is 0
    Gs_cespgrn_beta0 = np.load(file = PROJECT_DIR + f"test/results_scmultisim/{dataset}/ablation_beta_0_atac/cespgrn_ensemble_100_0.npy")
    scores_cespgrn_beta0 = bmk.calc_scores_para(thetas_inf = Gs_cespgrn_beta0, thetas_gt = Gs_gt, interval = None, model = "CeSpGRN (w\o beta)", bandwidth = None, truncate_param = 100, lamb = None, beta = 0, njobs = 16)

    # beta is 1
    Gs_cespgrn = np.load(file = PROJECT_DIR + f"test/results_scmultisim/{dataset}/beta_1_atac/cespgrn_ensemble_100_1.npy")
    scores_cespgrn = bmk.calc_scores_para(thetas_inf = Gs_cespgrn, thetas_gt = Gs_gt, interval = None, model = "CeSpGRN", bandwidth = None, truncate_param = 100, lamb = None, beta = 1, njobs = 16)

    scores = pd.concat([scores_cespgrn_beta0, scores_cespgrn], axis = 0, ignore_index = True)
    scores.to_csv(PROJECT_DIR + f"test/results_scmultisim/{dataset}/scores_ablation_beta.csv")

# In[]
sns.set_theme(font_scale = 1.2)
scores_all = pd.DataFrame()
for nchanging_edges in [20]:
    for fp in [0.01, 0.1]:
        for seed in [0, 1, 2]:
            score = pd.read_csv(PROJECT_DIR + f"test/results_scmultisim/simulated_8000_{nchanging_edges}_10_100_{fp}_{seed}_0_4/scores_ablation_beta.csv", index_col = 0)            
            score["fp"] = fp

            scores_all = pd.concat([scores_all, score], axis = 0, ignore_index = True)

# NOTE: BARPLOT
# fig = plt.figure(figsize = (15,7))
# ax = fig.subplots(nrows =1, ncols = 2)
# bar1 = sns.barplot(scores_all, x = "model", y = "AUPRC (abs)", hue = "fp", ax = ax[0], width=0.5, palette = "Set2", estimator=np.mean, errorbar=("pi", 100), capsize = 0.1, err_kws = {"color": "blue", "linewidth": 2, "linestyle": "--"})
# for i in bar1.containers:
#     bar1.bar_label(i, fmt = "%.3f")
# bar2 = sns.barplot(scores_all, x = "model", y = "Early Precision (abs)", hue = "fp", ax = ax[1], width=0.5, palette = "Set2", estimator=np.mean, errorbar=("pi", 100), capsize = 0.1, err_kws = {"color": "blue", "linewidth": 2, "linestyle": "--"})
# for i in bar2.containers:
#     bar2.bar_label(i, fmt = "%.3f")

# _ = ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation = 45, fontsize = 20)
# _ = ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation = 45, fontsize = 20)
# ax[0].get_legend().remove()
# ax[0].set_xlabel(None)
# ax[0].set_ylabel("AUPRC", fontsize = 20)
# ax[1].set_xlabel(None)
# ax[1].set_ylabel("Eprec", fontsize = 20)
# leg = ax[1].legend(loc='upper left', prop={'size': 20}, frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1), markerscale = 6, title = "Noise level", title_fontsize = 20)
# # ax[1].get_legend().remove()
# plt.tight_layout()
# fig.savefig(PROJECT_DIR + f"test/results_scmultisim/ablation_beta.png", bbox_inches = "tight", dpi = 150)


# NOTE: BOXPLOT
scores_all.loc[scores_all["model"] == "CeSpGRN (w\o beta)", "model"] = "CeSpGRN (w/o beta)"
fig = plt.figure(figsize = (15,7))
ax = fig.subplots(nrows =1, ncols = 2)
bar1 = sns.boxplot(scores_all, x = "model", y = "AUPRC (abs)", hue = "fp", ax = ax[0], width=0.5, palette = "Set2")
# sns.pointplot(x="model", y="AUPRC (abs)", data=scores_all, estimator="mean", color="red", markers="D", linestyles="", ax = ax[0])
# Compute medians per group
medians = scores_all.groupby("model")["AUPRC (abs)"].median()[::-1]
# print(medians)
# Annotate
# for i, median in enumerate(medians):
    # ax[0].text(i, median, f"{median:.2f}", ha='center', va='bottom', color='blue', fontsize=15, fontweight='bold')

bar2 = sns.boxplot(scores_all, x = "model", y = "Early Precision (abs)", hue = "fp", ax = ax[1], width=0.5, palette = "Set2")
# sns.pointplot(x="model", y="Early Precision (abs)", data=scores_all, estimator="mean", color="red", markers="D", linestyles="", ax = ax[1])
# Compute medians per group
# medians = scores_all.groupby("model")["Early Precision (abs)"].median()[::-1]
# Annotate
# for i, median in enumerate(medians):
#     ax[1].text(i, median, f"{median:.2f}", ha='center', va='bottom', color='blue', fontsize=15, fontweight='bold')
    
_ = ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation = 45, fontsize = 20)
_ = ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation = 45, fontsize = 20)
ax[0].get_legend().remove()
ax[0].set_xlabel(None)
ax[0].set_ylabel("AUPRC", fontsize = 20)
ax[1].set_xlabel(None)
ax[1].set_ylabel("Eprec", fontsize = 20)
leg = ax[1].legend(loc='upper left', prop={'size': 20}, frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1), markerscale = 6, title = "Noise level", title_fontsize = 20)
# ax[1].get_legend().remove()
plt.tight_layout()
fig.savefig(PROJECT_DIR + f"test/results_scmultisim/ablation_beta.png", bbox_inches = "tight", dpi = 150)


# In[]
# 2. ablation test on lambda
datasets = [
    "simulated_8000_20_10_100_0.01_0_0_4",
    "simulated_8000_20_10_100_0.01_1_0_4",
    "simulated_8000_20_10_100_0.01_2_0_4",
    "simulated_8000_20_10_100_0.1_0_0_4",
    "simulated_8000_20_10_100_0.1_1_0_4",
    "simulated_8000_20_10_100_0.1_2_0_4"
]

for dataset in datasets:
    if os.path.exists(PROJECT_DIR + f"test/results_scmultisim/{dataset}/scores_ablation_lamb.csv"):
        continue

    Gs_gt = np.load(file = PROJECT_DIR + f"data/scMultiSim/{dataset}/graph_gt.npy")
    # lamb is 0
    Gs_cespgrn_lamb0 = np.load(file = PROJECT_DIR + f"test/results_scmultisim/{dataset}/ablation_lamb_[0.0]_tf/cespgrn_ensemble_100_1.npy")
    scores_cespgrn_lamb0 = bmk.calc_scores_para(thetas_inf = Gs_cespgrn_lamb0, thetas_gt = Gs_gt, interval = None, model = "CeSpGRN (w\o lambda)", bandwidth = None, truncate_param = 100, lamb = None, beta = 1, njobs = 16)

    # lamb is 1
    Gs_cespgrn = np.load(file = PROJECT_DIR + f"test/results_scmultisim/{dataset}/beta_1_tf/cespgrn_ensemble_100_1.npy")
    scores_cespgrn = bmk.calc_scores_para(thetas_inf = Gs_cespgrn, thetas_gt = Gs_gt, interval = None, model = "CeSpGRN", bandwidth = None, truncate_param = 100, lamb = None, beta = 1, njobs = 16)

    scores = pd.concat([scores_cespgrn_lamb0, scores_cespgrn], axis = 0, ignore_index = True)
    scores.to_csv(PROJECT_DIR + f"test/results_scmultisim/{dataset}/scores_ablation_lamb.csv")


# In[]
sns.set_theme(font_scale = 1.2)
scores_all = pd.DataFrame()
for nchanging_edges in [20]:
    for fp in [0.01]:
        for seed in [0, 1, 2]:
            score = pd.read_csv(PROJECT_DIR + f"test/results_scmultisim/simulated_8000_{nchanging_edges}_10_100_{fp}_{seed}_0_4/scores_ablation_lamb.csv", index_col = 0)            
            score["fp"] = fp

            scores_all = pd.concat([scores_all, score], axis = 0, ignore_index = True)

# NOTE: BARPLOT
# fig = plt.figure(figsize = (15,7))
# ax = fig.subplots(nrows =1, ncols = 2)
# bar1 = sns.barplot(scores_all, x = "model", y = "AUPRC (abs)", hue = "fp", ax = ax[0], width=0.5, palette = "Set2", estimator=np.mean, errorbar=("pi", 100), capsize = 0.1, err_kws = {"color": "blue", "linewidth": 2, "linestyle": "--"})
# for i in bar1.containers:
#     bar1.bar_label(i, fmt = "%.3f")
# bar2 = sns.barplot(scores_all, x = "model", y = "Early Precision (abs)", hue = "fp", ax = ax[1], width=0.5, palette = "Set2", estimator=np.mean, errorbar=("pi", 100), capsize = 0.1, err_kws = {"color": "blue", "linewidth": 2, "linestyle": "--"})
# for i in bar2.containers:
#     bar2.bar_label(i, fmt = "%.3f")

# _ = ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation = 45, fontsize = 20)
# _ = ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation = 45, fontsize = 20)
# ax[0].get_legend().remove()
# ax[0].set_xlabel(None)
# ax[0].set_ylabel("AUPRC", fontsize = 20)
# ax[1].set_xlabel(None)
# ax[1].set_ylabel("Eprec", fontsize = 20)
# leg = ax[1].legend(loc='upper left', prop={'size': 20}, frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1), markerscale = 6, title = "Noise level", title_fontsize = 20)
# # ax[1].get_legend().remove()
# plt.tight_layout()
# fig.savefig(PROJECT_DIR + f"test/results_scmultisim/ablation_lamb.png", bbox_inches = "tight", dpi = 150)


# NOTE: BOXPLOT
scores_all.loc[scores_all["model"] == "CeSpGRN (w\o lambda)", "model"] = "CeSpGRN (w/o lambda)"
fig = plt.figure(figsize = (15,7))
ax = fig.subplots(nrows =1, ncols = 2)
bar1 = sns.boxplot(scores_all, x = "model", y = "AUPRC (abs)", hue = "fp", ax = ax[0], width=0.5, palette = "Set2")
# sns.pointplot(x="model", y="AUPRC (abs)", data=scores_all, estimator="mean", color="red", markers="D", linestyles="", ax = ax[0])
# Compute medians per group
medians = scores_all.groupby("model")["AUPRC (abs)"].median()[::-1]
print(medians)
# Annotate
for i, median in enumerate(medians):
    ax[0].text(i, median, f"{median:.2f}", ha='center', va='bottom', color='blue', fontsize=15, fontweight='bold')

bar2 = sns.boxplot(scores_all, x = "model", y = "Early Precision (abs)", hue = "fp", ax = ax[1], width=0.5, palette = "Set2")
# sns.pointplot(x="model", y="Early Precision (abs)", data=scores_all, estimator="mean", color="red", markers="D", linestyles="", ax = ax[1])
# Compute medians per group
medians = scores_all.groupby("model")["Early Precision (abs)"].median()[::-1]
# Annotate
for i, median in enumerate(medians):
    ax[1].text(i, median, f"{median:.2f}", ha='center', va='bottom', color='blue', fontsize=15, fontweight='bold')
    
_ = ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation = 45, fontsize = 20)
_ = ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation = 45, fontsize = 20)
ax[0].get_legend().remove()
ax[0].set_xlabel(None)
ax[0].set_ylabel("AUPRC", fontsize = 20)
ax[1].set_xlabel(None)
ax[1].set_ylabel("Eprec", fontsize = 20)
leg = ax[1].legend(loc='upper left', prop={'size': 20}, frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1), markerscale = 6, title = "Noise level", title_fontsize = 20)
ax[1].get_legend().remove()
plt.tight_layout()
fig.savefig(PROJECT_DIR + f"test/results_scmultisim/ablation_lamb.png", bbox_inches = "tight", dpi = 150)


# In[]
# 2. ablation test on sequencing depth normalization
datasets = [
    # "simulated_8000_20_10_100_0.01_0_0_4",
    # "simulated_8000_20_10_100_0.01_1_0_4",
    # "simulated_8000_20_10_100_0.01_2_0_4",
    "simulated_8000_20_10_100_0.1_0_0_4",
    "simulated_8000_20_10_100_0.1_1_0_4",
    "simulated_8000_20_10_100_0.1_2_0_4"
]

use_atac = False
for dataset in datasets:
    # if os.path.exists(PROJECT_DIR + f"test/results_scmultisim/{dataset}/noseqnorm/scores_noseqnorm_atac.csv"):
    #     continue

    Gs_gt = np.load(file = PROJECT_DIR + f"data/scMultiSim/{dataset}/graph_gt.npy")
    # with sequencing normalization
    if use_atac:
        Gs_cespgrn = np.load(file = PROJECT_DIR + f"test/results_scmultisim/{dataset}/seqnorm/raw_beta_1_atac/cespgrn_ensemble_100_1.npy")
    else:
        Gs_cespgrn = np.load(file = PROJECT_DIR + f"test/results_scmultisim/{dataset}/beta_1_tf/cespgrn_ensemble_100_1.npy")
    scores_cespgrn = bmk.calc_scores_para(thetas_inf = Gs_cespgrn, thetas_gt = Gs_gt, interval = None, model = "CeSpGRN", bandwidth = None, truncate_param = 100, lamb = None, beta = 1, njobs = 16)

    # without sequencing normalization
    if use_atac:
        Gs_cespgrn_noseqnorm = np.load(file = PROJECT_DIR + f"test/results_scmultisim/{dataset}/noseqnorm/raw_beta_1_atac/cespgrn_ensemble_100_1.npy")
    else:
        Gs_cespgrn_noseqnorm = np.load(file = PROJECT_DIR + f"test/results_scmultisim/{dataset}/noseqnorm/raw_beta_1_tf/cespgrn_ensemble_100_1.npy")
    scores_cespgrn_noseqnorm = bmk.calc_scores_para(thetas_inf = Gs_cespgrn_noseqnorm, thetas_gt = Gs_gt, interval = None, model = "CeSpGRN (w/o seq-norm)", bandwidth = None, truncate_param = 100, lamb = None, beta = 1, njobs = 16)

    scores = pd.concat([scores_cespgrn, scores_cespgrn_noseqnorm], axis = 0, ignore_index = True)
    if use_atac:
        scores.to_csv(PROJECT_DIR + f"test/results_scmultisim/{dataset}/scores_seqnorm_atac.csv")
    else:
        scores.to_csv(PROJECT_DIR + f"test/results_scmultisim/{dataset}/scores_seqnorm_tf.csv")


# In[]
use_atac = False
sns.set_theme(font_scale = 1.2)
scores_all = pd.DataFrame()
for nchanging_edges in [20]:
    for fp in [0.1]:
        for seed in [0, 1, 2]:
            if use_atac:
                score = pd.read_csv(PROJECT_DIR + f"test/results_scmultisim/simulated_8000_{nchanging_edges}_10_100_{fp}_{seed}_0_4/scores_seqnorm_atac.csv", index_col = 0)        
            else:
                score = pd.read_csv(PROJECT_DIR + f"test/results_scmultisim/simulated_8000_{nchanging_edges}_10_100_{fp}_{seed}_0_4/scores_seqnorm_tf.csv", index_col = 0)        

            score["fp"] = fp
            scores_all = pd.concat([scores_all, score], axis = 0, ignore_index = True)



# NOTE: BOXPLOT
fig = plt.figure(figsize = (15,7))
ax = fig.subplots(nrows =1, ncols = 2)
error_bar = "sd"
# bar1 = sns.boxplot(scores_all, x = "model", y = "AUPRC (abs)", hue = "fp", ax = ax[0], width=0.5, palette = "Set2")
bar1 = sns.barplot(scores_all, x = "model", y = "AUPRC (abs)", hue = "fp", ax = ax[0], width=0.5, palette = "Set2", estimator=np.mean, errorbar=error_bar, capsize = 0.1, err_kws = {"color": "blue", "linewidth": 2, "linestyle": "--"})
# sns.pointplot(x="model", y="AUPRC (abs)", data=scores_all, estimator="mean", color="red", markers="D", linestyles="", ax = ax[0])
# Compute medians per group
# medians = scores_all.groupby("model")["AUPRC (abs)"].median()[::-1]
# print(medians)
# # Annotate
# for i, median in enumerate(medians):
#     ax[0].text(i, median, f"{median:.2f}", ha='center', va='bottom', color='blue', fontsize=15, fontweight='bold')

# bar2 = sns.boxplot(scores_all, x = "model", y = "Early Precision (abs)", hue = "fp", ax = ax[1], width=0.5, palette = "Set2")
bar2 = sns.barplot(scores_all, x = "model", y = "Early Precision (abs)", hue = "fp", ax = ax[1], width=0.5, palette = "Set2", estimator=np.mean, errorbar=error_bar, capsize = 0.1, err_kws = {"color": "blue", "linewidth": 2, "linestyle": "--"})
# sns.pointplot(x="model", y="Early Precision (abs)", data=scores_all, estimator="mean", color="red", markers="D", linestyles="", ax = ax[1])
# # Compute medians per group
# medians = scores_all.groupby("model")["Early Precision (abs)"].median()[::-1]
# # Annotate
# for i, median in enumerate(medians):
#     ax[1].text(i, median, f"{median:.2f}", ha='center', va='bottom', color='blue', fontsize=15, fontweight='bold')
    
for i in bar1.containers:
    bar1.bar_label(i, fmt='%.3f')  
for i in bar2.containers:
    bar2.bar_label(i, fmt='%.3f')

_ = ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation = 45, fontsize = 20)
_ = ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation = 45, fontsize = 20)
ax[0].get_legend().remove()
ax[0].set_xlabel(None)
ax[0].set_ylabel("AUPRC", fontsize = 20)
ax[1].set_xlabel(None)
ax[1].set_ylabel("Eprec", fontsize = 20)
leg = ax[1].legend(loc='upper left', prop={'size': 20}, frameon = False, ncol = 1, bbox_to_anchor=(1.04, 1), markerscale = 6, title = "Noise level", title_fontsize = 20)
ax[1].get_legend().remove()
plt.tight_layout()
# fig.savefig(PROJECT_DIR + f"test/results_scmultisim/ablation_seqnorm.png", bbox_inches = "tight", dpi = 150)





# In[]
# ---------------------------------------
#
# NOTE: plot the hyper-parameter test result
#
# ---------------------------------------
truncate_param_list = [10, 100, 200, 500]
beta_list = [1e-2, 1e-1, 10, 100]

dataset_list = ["simulated_8000_20_10_100_0.01_0_0_4", "simulated_8000_20_10_100_0.01_1_0_4", "simulated_8000_20_10_100_0.01_2_0_4",
                "simulated_8000_20_10_100_0.1_0_0_4", "simulated_8000_20_10_100_0.1_1_0_4", "simulated_8000_20_10_100_0.1_2_0_4"]
dataset = dataset_list[0]

# In[]
# Hyper-parameter test on beta
scores_cespgrn = []
for beta in beta_list:
    truncate_param = 100
    Gs_gt = np.load(file = PROJECT_DIR + f"data/scMultiSim/{dataset}/graph_gt.npy")
    Gs = np.load(PROJECT_DIR + f"test/results_scmultisim/{dataset}/beta_{beta}_atac/" + f"cespgrn_ensemble_{truncate_param}_{beta}.npy")

    scores = bmk.calc_scores_para(thetas_inf = Gs, thetas_gt = Gs_gt, interval = None, model = "CeSpGRN", bandwidth = None, truncate_param = truncate_param, lamb = None, beta = beta, njobs = 16)
    scores_cespgrn.append(scores)
scores_cespgrn = pd.concat(scores_cespgrn, axis = 0, ignore_index = True)
scores_cespgrn.to_csv(PROJECT_DIR + f"test/results_scmultisim/hyperparams_beta.csv")

# BOXPLOT
sns.set_theme(font_scale = 2)
fig = plt.figure(figsize = (20, 7))
ax = fig.subplots(nrows = 1, ncols = 2)
bar1 = sns.boxplot(scores_cespgrn, x = "beta", y = "AUPRC (abs)", ax = ax[0], width=0.5, palette = "Set2")
bar2 = sns.boxplot(scores_cespgrn, x = "beta", y = "Early Precision (abs)", ax = ax[1], width=0.5, palette = "Set2")
fig.savefig(PROJECT_DIR + f"test/results_scmultisim/hyperparams_beta.png", dpi = 300, bbox_inches = "tight")

# In[]
# Hyper-parameter test on truncate params
beta = 1
scores_cespgrn = []
for truncate_param in truncate_param_list:
    Gs_gt = np.load(file = PROJECT_DIR + f"data/scMultiSim/{dataset}/graph_gt.npy")
    if truncate_param == 100:
        Gs = np.load(PROJECT_DIR + f"test/results_scmultisim/{dataset}/beta_1_atac/cespgrn_ensemble_{truncate_param}_1.npy")

    else:
        Gs = np.load(PROJECT_DIR + f"test/results_scmultisim/{dataset}/truncate_{truncate_param}_atac/cespgrn_ensemble_{truncate_param}_1.npy")

    scores = bmk.calc_scores_para(thetas_inf = Gs, thetas_gt = Gs_gt, interval = None, model = "CeSpGRN", bandwidth = None, truncate_param = truncate_param, lamb = None, beta = 1, njobs = 16)
    scores_cespgrn.append(scores)

scores_cespgrn = pd.concat(scores_cespgrn, axis = 0, ignore_index = True)
scores_cespgrn.to_csv(PROJECT_DIR + f"test/results_scmultisim/hyperparams_truncate_params.csv")

# BOXPLOT
sns.set_theme(font_scale = 2)
fig = plt.figure(figsize = (20, 7))
ax = fig.subplots(nrows = 1, ncols = 2)
bar1 = sns.boxplot(scores_cespgrn, x = "truncate_param", y = "AUPRC (abs)", ax = ax[0], width=0.5, palette = "Set2")
bar2 = sns.boxplot(scores_cespgrn, x = "truncate_param", y = "Early Precision (abs)", ax = ax[1], width=0.5, palette = "Set2")
fig.savefig(PROJECT_DIR + f"test/results_scmultisim/hyperparams_truncateparams.png", dpi = 300, bbox_inches = "tight")

# %%
