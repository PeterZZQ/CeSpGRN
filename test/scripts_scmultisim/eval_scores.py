# In[]
import numpy as np 
import sys, os
PROJECT_DIR = "/localscratch/ziqi/CeSpGRN/"
sys.path.append(PROJECT_DIR + 'src/')
import bmk
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

datasets = [
    "simulated_8000_20_10_100_0.01_0_0_4",
    "simulated_8000_20_10_100_0.01_1_0_4",
    "simulated_8000_20_10_100_0.01_2_0_4",
    "simulated_8000_20_10_100_0.1_0_0_4",
    "simulated_8000_20_10_100_0.1_1_0_4",
    "simulated_8000_20_10_100_0.1_2_0_4"
]
dataset = datasets[0]

path = f"../../data/scMultiSim/{dataset}/"
result_dir = f"../results_scmultisim/{dataset}/"

use_tf = True
beta = 1
truncate_param = 100

# In[]
Gs_gt = np.load(file = path + "graph_gt.npy")
if use_tf & (beta > 0):
    print("Calculating score (GENIE3-TF)...")
    Gs_genie_tf = np.load(file = result_dir + "Gs_genie_tf.npy")
    scores_genie3_tf = bmk.calc_scores_para(thetas_inf = Gs_genie_tf, thetas_gt = Gs_gt, interval = None, model = "GENIE3 (TF)", bandwidth = None, truncate_param = truncate_param, lamb = None, beta = beta, njobs = 16)

    print("Calculating score (scMTNI-TF)...")
    Gs_scmtni_tf = np.load(file = result_dir + "Gs_scmtni_tf.npy")
    scores_scmtni_tf = bmk.calc_scores_para(thetas_inf = Gs_scmtni_tf, thetas_gt = Gs_gt, interval = None, model = "scMTNI (TF)", bandwidth = None, truncate_param = truncate_param, lamb = None, beta = beta, njobs = 16)

    print("Calculating score (prior tf)...")
    prior_tf = 1 - np.load(result_dir + "masks_tf.npy")
    scores_prior_tf = bmk.calc_scores_para(thetas_inf = prior_tf, thetas_gt = Gs_gt, interval = None, model = "Prior (TF)", bandwidth = None, truncate_param = truncate_param, lamb = None, beta = beta, njobs = 16)

    print("Calculating score (CeSpGRN-TF)...")
    Gs_cespgrn_tf = np.load(file = result_dir + f"beta_1_tf/cespgrn_ensemble_{truncate_param}_1.npy")
    scores_cespgrn_tf = bmk.calc_scores_para(thetas_inf = Gs_cespgrn_tf, thetas_gt = Gs_gt, interval = None, model = "CeSpGRN (TF)", bandwidth = None, truncate_param = truncate_param, lamb = None, beta = 1, njobs = 16)

    print("Calculate score (CSN)...")
    Gs_csn = np.load(file = result_dir + "theta_CSN.npy")
    scores_csn = bmk.calc_scores_para(thetas_inf = Gs_csn, thetas_gt = Gs_gt, interval = None, model = "CSN", bandwidth = None, truncate_param = truncate_param, lamb = None, beta = 1, njobs = 16)

    print("Calculate score (SCODE)...")
    Gs_scode = pd.read_csv(result_dir + "/SCODE/meanA.txt", sep = "\t", header = None).T.values
    Gs_scode = np.repeat(Gs_scode[None, :, :], repeats = Gs_gt.shape[0], axis = 0)
    scores_scode = bmk.calc_scores_para(thetas_inf = Gs_scode, thetas_gt = Gs_gt, interval = None, model = "SCODE", bandwidth = None, truncate_param = truncate_param, lamb = None, beta = 1, njobs = 16)

    scores = pd.concat([scores_genie3_tf, scores_scmtni_tf, scores_prior_tf, scores_cespgrn_tf, scores_csn, scores_scode], axis = 0, ignore_index = True)

elif beta > 0:
    # print("Calculating score (INDEP-ATAC)...")
    # # one cluster INDEP
    # Gs_indep_atac = np.load(file = result_dir + "Gs_scmtni_atac_1.npy")
    # scores_indep_atac = bmk.calc_scores_para(thetas_inf = Gs_indep_atac, thetas_gt = Gs_gt, interval = None, model = "INDEP (ATAC)", bandwidth = None, truncate_param = truncate_param, lamb = None, beta = beta, njobs = 16)

    print("Calculating score (scMTNI-ATAC)...")
    # one cluster scMTNI
    Gs_scmtni_atac = np.load(file = result_dir + "Gs_scmtni_atac_3_5.npy")
    scores_scmtni_atac = bmk.calc_scores_para(thetas_inf = Gs_scmtni_atac, thetas_gt = Gs_gt, interval = None, model = "scMTNI (ATAC)", bandwidth = None, truncate_param = truncate_param, lamb = None, beta = beta, njobs = 16)

    print("Calculating score (CellOracle)...")
    Gs_celloracle = np.load(file = result_dir + "Gs_coef_mean_celloracle_3.npy")
    scores_celloracle = bmk.calc_scores_para(thetas_inf = Gs_celloracle, thetas_gt = Gs_gt, interval = None, model = "CellOracle", bandwidth = None, truncate_param = truncate_param, lamb = None, beta = beta, njobs = 16)

    print("Calculating score (SCENIC+)...")
    Gs_scenicplus = np.load(file= result_dir + "scenicplus/Gs_scenicplus.npy")
    scores_scenicplus = bmk.calc_scores_para(thetas_inf = Gs_scenicplus, thetas_gt = Gs_gt, interval = None, model = "SCENIC+", bandwidth = None, truncate_param = truncate_param, lamb = None, beta = beta, njobs = 16)
    
    print("Calculating score (prior atac)...")
    prior_atac = 1 - np.load(result_dir + "masks_atac.npy")
    scores_prior_atac = bmk.calc_scores_para(thetas_inf = prior_atac, thetas_gt = Gs_gt, interval = None, model = "Prior (ATAC)", bandwidth = None, truncate_param = truncate_param, lamb = None, beta = beta, njobs = 16)


    print("Calculating score (prior atac cluster)...")
    prior_atac_clusters = []
    for i in range(10):
        interval = int(prior_atac.shape[0]/10)
        prior_atac_cluster = np.mean(prior_atac[(i * interval):((i+1) * interval), :, :], axis = 0, keepdims = True)
        print("sparsity: {:.3f}".format(np.sum(prior_atac_cluster!= 0)/prior_atac_cluster.shape[1]/prior_atac_cluster.shape[2]))
        prior_atac_cluster = prior_atac_cluster/np.max(prior_atac_cluster)
        # remove the edges that show up in less than 10% of cells
        prior_atac_cluster = prior_atac_cluster * (prior_atac_cluster > 0.1)
        print("sparsity: {:.3f}".format(np.sum(prior_atac_cluster!= 0)/prior_atac_cluster.shape[1]/prior_atac_cluster.shape[2]))
        prior_atac_clusters.append(np.repeat(prior_atac_cluster, repeats = interval, axis = 0))
    prior_atac_clusters = np.concatenate(prior_atac_clusters, axis = 0)
    scores_prior_atac_cluster = bmk.calc_scores_para(thetas_inf = prior_atac_clusters, thetas_gt = Gs_gt, interval = None, model = "Prior (ATAC cluster)", bandwidth = None, truncate_param = truncate_param, lamb = None, beta = beta, njobs = 16)

    print("Calculating score (CeSpGRN-ATAC)...")
    Gs_cespgrn_atac = np.load(file = result_dir + f"beta_1_atac/cespgrn_ensemble_{truncate_param}_1.npy")
    scores_cespgrn_atac = bmk.calc_scores_para(thetas_inf = Gs_cespgrn_atac, thetas_gt = Gs_gt, interval = None, model = "CeSpGRN (ATAC)", bandwidth = None, truncate_param = truncate_param, lamb = None, beta = 1, njobs = 16)

    scores = pd.concat([scores_prior_atac, scores_prior_atac_cluster, scores_cespgrn_atac, scores_scmtni_atac, scores_celloracle], axis = 0, ignore_index = True)

# print("Calculating score (GENIE3)...")
# Gs_genie = np.load(file = result_dir + "Gs_genie.npy")
# scores_genie3 = bmk.calc_scores_para(thetas_inf = Gs_genie, thetas_gt = Gs_gt, interval = None, model = "GENIE3", bandwidth = None, truncate_param = truncate_param, lamb = None, beta = beta, njobs = 16)

# Gs_cespgrn = np.load(file = result_dir + f"beta_0/cespgrn_ensemble_{truncate_param}_0.npy")
# scores_cespgrn = bmk.calc_scores_para(thetas_inf = Gs_cespgrn, thetas_gt = Gs_gt, interval = None, model = "CeSpGRN", bandwidth = None, truncate_param = truncate_param, lamb = None, beta = 0, njobs = 16)

scores.to_csv(result_dir + "scores.csv")

# In[]
# TEMP: include new scores
for nchanging_edges in [20]:
    for fp in [0.01]:
        for seed in [0, 1, 2]:
            result_dir = f"../results_scmultisim/simulated_8000_{nchanging_edges}_10_100_{fp}_{seed}_0_4/"
            data_dir = f"../../data/scMultiSim/simulated_8000_{nchanging_edges}_10_100_{fp}_{seed}_0_4/"
            score = pd.read_csv(result_dir + "scores.csv", index_col = 0)
            
            Gs_gt = np.load(data_dir + "graph_gt.npy")
            Gs_csn = np.load(file = result_dir + "theta_CSN.npy")
            score_csn = bmk.calc_scores_para(thetas_inf = Gs_csn, thetas_gt = Gs_gt, interval = None, model = "CSN", bandwidth = None, truncate_param = None, lamb = None, beta = 1, njobs = 16)

            Gs_scode = pd.read_csv(result_dir + "/SCODE/meanA.txt", sep = "\t", header = None).T.values
            Gs_scode = np.repeat(Gs_scode[None, :, :], repeats = Gs_gt.shape[0], axis = 0)

            score_scode = bmk.calc_scores_para(thetas_inf = Gs_scode, thetas_gt = Gs_gt, interval = None, model = "SCODE", bandwidth = None, truncate_param = None, lamb = None, beta = 1, njobs = 16)

            score = pd.concat([score, score_csn, score_scode])
            score.to_csv(result_dir + "scores_wcsn.csv")

# In[]
use_tf = True
sns.set_theme(font_scale = 1.7)
scores = pd.DataFrame() 
plt.rcParams["font.size"] = 17

if use_tf & (beta > 0):
    for nchanging_edges in [20]:
        for fp in [0.01]:
            for seed in [0, 1, 2]:
                result_dir = f"../results_scmultisim/simulated_8000_{nchanging_edges}_10_100_{fp}_{seed}_0_4/"
                score = pd.read_csv(result_dir + "scores_wcsn.csv", index_col = 0)
                score["fp"] = fp
                scores = pd.concat([scores, score], axis = 0)

    scores = scores[(scores["model"] == "GENIE3 (TF)") | (scores["model"] == "scMTNI (TF)") | (scores["model"] == "CeSpGRN (TF)") | (scores["model"] == "CSN") | (scores["model"] == "SCODE")]
    fig = plt.figure(figsize = (15,7))
    ax = fig.subplots(nrows =1, ncols = 2)
    bar1 = sns.barplot(scores[scores["model"] != "CSN"], x = "model", y = "AUPRC (abs)", hue = "fp", ax = ax[0], order = ["CeSpGRN (TF)", "scMTNI (TF)", "GENIE3 (TF)", "SCODE"], width=0.5, palette = "Set2", estimator=np.mean, errorbar=("pi", 100), capsize = 0.1, err_kws = {"color": "blue", "linewidth": 2, "linestyle": "--"})
    for i in bar1.containers:
        bar1.bar_label(i, fmt = "%.3f")
    bar2 = sns.barplot(scores, x = "model", y = "Early Precision (abs)", hue = "fp", ax = ax[1], order = ["CeSpGRN (TF)", "scMTNI (TF)", "GENIE3 (TF)", "CSN", "SCODE"], width=0.5, palette = "Set2", estimator=np.mean, errorbar=("pi", 100), capsize = 0.1, err_kws = {"color": "blue", "linewidth": 2, "linestyle": "--"})
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
    fig.savefig("../results_scmultisim/acc_tf.png", bbox_inches = "tight")

elif beta > 0:
    for nchanging_edges in [20]:
        for fp in [0.01, 0.1]:
            for seed in [0, 1, 2]:
                result_dir = f"../results_scmultisim/simulated_8000_{nchanging_edges}_10_100_{fp}_{seed}_0_4/"
                score = pd.read_csv(result_dir + "scores.csv", index_col = 0)
                score["fp"] = fp
                scores = pd.concat([scores, score], axis = 0)

    scores = scores[(scores["model"] == "CellOracle") | (scores["model"] == "scMTNI (ATAC)") | (scores["model"] == "CeSpGRN (ATAC)")]

    fig = plt.figure(figsize = (15,7))
    ax = fig.subplots(nrows =1, ncols = 2)
    bar1 = sns.barplot(scores, x = "model", y = "AUPRC (abs)", hue = "fp", ax = ax[0], order = ["CeSpGRN (ATAC)", "scMTNI (ATAC)", "CellOracle"], palette = "Set2", estimator=np.mean, errorbar=("pi", 100), capsize = 0.1, err_kws = {"color": "blue", "linewidth": 2, "linestyle": "--"})
    for i in bar1.containers:
        bar1.bar_label(i, fmt='%.3f')  
    bar2 = sns.barplot(scores, x = "model", y = "Early Precision (abs)", hue = "fp", ax = ax[1], order = ["CeSpGRN (ATAC)", "scMTNI (ATAC)", "CellOracle"], palette = "Set2", estimator=np.mean, errorbar=("pi", 100), capsize = 0.1, err_kws = {"color": "blue", "linewidth": 2, "linestyle": "--"})
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
    fig.savefig("../results_scmultisim/acc_atac.png", bbox_inches = "tight")
 # In[]
