# In[]
from operator import index
import pandas as pd
import numpy as np
import torch
import torch.nn
import sys, os
sys.path.append('../../src/')

from scipy.spatial.distance import pdist, squareform

import bmk_beeline as bmk
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns

from umap import UMAP

plt.rcParams["font.size"] = 16

from multiprocessing import Pool, cpu_count


def calc_scores(thetas_inf, thetas_gt, interval, model, bandwidth = 0, truncate_param = 0, lamb = 0):
    np.random.seed(0)

    scores = pd.DataFrame(columns = ["interval", "ngenes", "time", "model", "bandwidth", "truncate_param", "lambda", "nmse","kendall-tau", "pearson", "spearman", "cosine similarity", "AUPRC Ratio (pos)", "AUPRC Ratio (neg)", "AUPRC Ratio (abs)", "Early Precision Ratio (pos)", "Early Precision Ratio (neg)", "Early Precision Ratio (abs)"])

    assert thetas_inf.shape[0] == thetas_gt.shape[0]

    ntimes = thetas_inf.shape[0]
    ngenes = thetas_inf.shape[1]
    
    for time in range(0, ntimes):
        thetas_rand = np.random.randn(ngenes, ngenes)
        nmse = bmk.NMSE(G_inf = thetas_inf[time], G_true = thetas_gt[time])
        pearson_val, _ = bmk.pearson(G_inf = thetas_inf[time], G_true = thetas_gt[time])
        kt, _ = bmk.kendalltau(G_inf = thetas_inf[time], G_true = thetas_gt[time])
        spearman_val, _ = bmk.spearman(G_inf = thetas_inf[time], G_true = thetas_gt[time])
        cosine_sim = bmk.cossim(G_inf = thetas_inf[time], G_true = thetas_gt[time])

        AUPRC_pos, AUPRC_neg = bmk.compute_auc_signed(G_inf = thetas_inf[time], G_true = thetas_gt[time])     
        AUPRC_pos_rand, AUPRC_neg_rand = bmk.compute_auc_signed(G_inf = thetas_rand, G_true = thetas_gt[time])     
        AUPRC = bmk.compute_auc_abs(G_inf = thetas_inf[time], G_true = thetas_gt[time])
        AUPRC_rand = bmk.compute_auc_abs(G_inf = thetas_rand, G_true = thetas_gt[time])

        Eprec_pos, Eprec_neg = bmk.compute_eprec_signed(G_inf = thetas_inf[time], G_true = thetas_gt[time])
        Eprec_pos_rand, Eprec_neg_rand = bmk.compute_eprec_signed(G_inf = thetas_rand, G_true = thetas_gt[time])
        Eprec = bmk.compute_eprec_abs(G_inf = thetas_inf[time], G_true = thetas_gt[time])
        Eprec_rand = bmk.compute_eprec_abs(G_inf = thetas_rand, G_true = thetas_gt[time])
        
        scores = scores.append({"interval": interval,
                            "ngenes": ngenes,
                            "nmse": nmse, 
                            "pearson": pearson_val, 
                            "kendall-tau": kt,
                            "spearman": spearman_val,
                            "cosine similarity": cosine_sim, 
                            "AUPRC Ratio (pos)": AUPRC_pos/AUPRC_pos_rand,
                            "AUPRC Ratio (neg)": AUPRC_neg/AUPRC_neg_rand,
                            "AUPRC Ratio (abs)": AUPRC/AUPRC_rand,
                            "Early Precision Ratio (pos)": Eprec_pos/(Eprec_pos_rand + 1e-12),
                            "Early Precision Ratio (neg)": Eprec_neg/(Eprec_neg_rand + 1e-12),
                            "Early Precision Ratio (abs)":Eprec/(Eprec_rand + 1e-12),
                            "time":time,
                            "model": model,
                            "bandwidth": bandwidth,
                            "truncate_param":truncate_param,
                            "lambda":lamb}, ignore_index=True)  


    return scores      
    


# In[] benchmark accuracy

print("------------------------------------------------------------------")
print("benchmark accuracy")
print("------------------------------------------------------------------")
ntimes = 1000
nsample = 1
path = "../../data/continuousODE/"
umap_op = UMAP(n_components = 2, min_dist = 0.8, n_neighbors = 30, random_state = 0)
traj = "bifurc"


def summarize_scores(setting):
    ntimes = setting["ntimes"]
    interval = setting["interval"]
    ngenes = setting["ngenes"]
    ntfs = setting["ntfs"]
    seed = setting["seed"]

    print("ntimes: " + str(ntimes) + ", interval: " + str(interval) + ", ngenes: " + str(ngenes) + ", seed: " + str(seed) + ", initial graph: sergio\n")
    scores = pd.DataFrame(columns = ["interval", "ngenes", "time", "model", "bandwidth", "truncate_param", "lambda", "nmse","kendall-tau", "pearson", "spearman", "cosine similarity", "AUPRC Ratio (pos)", "AUPRC Ratio (neg)", "AUPRC Ratio (abs)", "Early Precision Ratio (pos)", "Early Precision Ratio (neg)", "Early Precision Ratio (abs)"])
    result_dir = "../results_softODE_v3/" + traj + "_ngenes_" + str(ngenes) + "_ncell_" + str(ntimes) + "_seed_" + str(seed) + "/"
    data_dir = path + traj + "_ngenes_" + str(ngenes) + "_ncell_" + str(ntimes) + "_seed_" + str(seed) + "/"
    
    # X = np.load(data_dir + "expr.npy")
    gt_adj = np.load(data_dir + "GRNs.npy")
    sim_time = np.load(data_dir + "pseudotime.npy")
    print("data loaded.")

    # admm, hyper-parameter
    for bandwidth in [0.1, 1, 10]:
        for truncate_param in [15, 30, 100]:
            for lamb in [0.0001, 0.001, 0.01, 0.1]:
                thetas = np.load(file = result_dir + "thetas_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_kt.npy")
                print(result_dir + "thetas_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param))
                score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "CeSpGRN-kt", bandwidth = bandwidth, truncate_param = truncate_param, lamb = lamb)
                scores = pd.concat([scores, score], axis = 0, ignore_index = True)

                fig = plt.figure(figsize = (10,7))
                X_umap = umap_op.fit_transform(thetas.reshape(ntimes * nsample, -1))
                ax = fig.add_subplot()
                ax.scatter(X_umap[:,0], X_umap[:,1], c = sim_time, s = 5)
                fig.savefig(result_dir + "infer_G_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_kt.png", bbox_inches = "tight")

                thetas = np.load(file = result_dir + "thetas_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + ".npy")
                print(result_dir + "thetas_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param))
                score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "CeSpGRN", bandwidth = bandwidth, truncate_param = truncate_param, lamb = lamb)
                scores = pd.concat([scores, score], axis = 0, ignore_index = True)

                fig = plt.figure(figsize = (10,7))
                X_umap = umap_op.fit_transform(thetas.reshape(ntimes * nsample, -1))
                ax = fig.add_subplot()
                ax.scatter(X_umap[:,0], X_umap[:,1], c = sim_time, s = 5)
                fig.savefig(result_dir + "infer_G_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + ".png", bbox_inches = "tight")

    scores.to_csv(result_dir + "score.csv")


settings = []
for interval in [20]:
    for (ngenes, ntfs) in [(20, 5)]:
        for seed in [4]:
            settings.append({
                "ntimes": ntimes,
                "interval": interval,
                "ngenes": ngenes,
                "ntfs": ntfs,
                "seed": seed
            })


pool = Pool(5) 
pool.map(summarize_scores, [x for x in settings])
pool.close()
pool.join()

# In[] 
print("------------------------------------------------------------------")
print("benchmark differences")
print("------------------------------------------------------------------")
ntimes = 1000
nsample = 1
path = "../../data/continuousODE/"
umap_op = UMAP(n_components = 2, min_dist = 0.8, n_neighbors = 30, random_state = 0)

def summarize_scores_diff(setting):
    ntimes = setting["ntimes"]
    interval = setting["interval"]
    ngenes = setting["ngenes"]
    ntfs = setting["ntfs"]
    seed = setting["seed"]

    print("ntimes: " + str(ntimes) + ", interval: " + str(interval) + ", ngenes: " + str(ngenes) + ", seed: " + str(seed) + ", initial graph: sergio\n")
    scores = pd.DataFrame(columns = ["interval", "ngenes", "time", "model", "bandwidth", "truncate_param", "lambda", "nmse","kendall-tau", "pearson", "spearman", "cosine similarity", "AUPRC Ratio (pos)", "AUPRC Ratio (neg)", "AUPRC Ratio (abs)", "Early Precision Ratio (pos)", "Early Precision Ratio (neg)", "Early Precision Ratio (abs)"])
    result_dir = "../results_softODE_v3/" + traj + "_ngenes_" + str(ngenes) + "_ncell_" + str(ntimes) + "_seed_" + str(seed) + "/"
    data_dir = path + traj + "_ngenes_" + str(ngenes) + "_ncell_" + str(ntimes) + "_seed_" + str(seed) + "/"
    
    # X = np.load(data_dir + "expr.npy")
    gt_adj = np.load(data_dir + "GRNs.npy")
    sim_time = np.load(data_dir + "pseudotime.npy")
    print("data loaded.")

    step = 400
    gt_adj = gt_adj[step::,:,:] - gt_adj[:-step:,:,:]  

    # admm, hyper-parameter
    for bandwidth in [0.1, 1, 10]:
        for truncate_param in [15, 30, 100]:
            for lamb in [0.0001, 0.001, 0.01, 0.1]:
                thetas = np.load(file = result_dir + "thetas_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_kt.npy")
                thetas = thetas[step::,:,:] - thetas[:-step:,:,:]  
                score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "CeSpGRN-kt", bandwidth = bandwidth, truncate_param = truncate_param, lamb = lamb)
                scores = pd.concat([scores, score], axis = 0, ignore_index = True)

                thetas = np.load(file = result_dir + "thetas_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + ".npy")
                thetas = thetas[step::,:,:] - thetas[:-step:,:,:] 
                score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "CeSpGRN", bandwidth = bandwidth, truncate_param = truncate_param, lamb = lamb)
                scores = pd.concat([scores, score], axis = 0, ignore_index = True)

    scores.to_csv(result_dir + "score_diff.csv")

settings = []
for interval in [20]:
    for (ngenes, ntfs) in [(20, 5)]:
        for seed in [4]:
            settings.append({
                "ntimes": ntimes,
                "interval": interval,
                "ngenes": ngenes,
                "ntfs": ntfs,
                "seed": seed
            })


pool = Pool(5) 
pool.map(summarize_scores_diff, [x for x in settings])
pool.close()
pool.join()

# In[] summarize the mean result in csv file

ntimes = 1000
nsample = 1
for traj in ["linear", "bifurc"]:
    for interval in [20]:
        for (ngenes, ntfs) in [(20, 5)]:
            for seed in [0,1,2,3,4]:
                print("ntimes: " + str(ntimes) + ", interval: " + str(interval) + ", ngenes: " + str(ngenes) + ", seed: " + str(seed) + ", initial graph: sergio")
                result_dir = "../results_softODE_v3/" + traj + "_ngenes_" + str(ngenes) + "_ncell_" + str(ntimes) + "_seed_" + str(seed) + "/"
                score = pd.read_csv(result_dir + "score.csv", index_col = 0)
                mean_score = score.groupby(by = ["model", "bandwidth", "truncate_param", "lambda"]).mean()
                mean_score = mean_score.drop(["time"], axis = 1)
                mean_score.to_csv(result_dir + "mean_score.csv")
                # display(mean_score)

    print("\ndifferences\n")

    for interval in [20]:
        for (ngenes, ntfs) in [(20, 5)]:
            for seed in [0,1,2,3,4]:
                print("ntimes: " + str(ntimes) + ", interval: " + str(interval) + ", ngenes: " + str(ngenes) + ", seed: " + str(seed) + ", initial graph: sergio")
                result_dir = "../results_softODE_v3/" + traj + "_ngenes_" + str(ngenes) + "_ncell_" + str(ntimes) + "_seed_" + str(seed) + "/"
                score = pd.read_csv(result_dir + "score_diff.csv", index_col = 0)
                mean_score = score.groupby(by = ["model", "bandwidth", "truncate_param", "lambda"]).mean()
                mean_score = mean_score.drop(["time"], axis = 1)
                mean_score.to_csv(result_dir + "mean_score_diff.csv")
                # display(mean_score)


# In[]
import matplotlib.patheffects as path_effects

def add_median_labels(ax, precision='.2f'):
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))
    for median in lines[4:len(lines):lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        text = ax.text(x, y, f'{value:{precision}}', ha='center', va='center',
                       fontweight='bold', color='white', fontsize = 10)
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground=median.get_color()),
            path_effects.Normal(),
        ])

score_all = pd.DataFrame(columns = ["interval", "ngenes", "time", "model", "bandwidth", "truncate_param", "lambda", "nmse","kendall-tau", "pearson", "spearman", "cosine similarity", "AUPRC Ratio (pos)", "AUPRC Ratio (neg)", "AUPRC Ratio (abs)", "Early Precision Ratio (pos)", "Early Precision Ratio (neg)", "Early Precision Ratio (abs)"])
score_all_diff = pd.DataFrame(columns = ["interval", "ngenes", "time", "model", "bandwidth", "truncate_param", "lambda", "nmse","kendall-tau", "pearson", "spearman", "cosine similarity", "AUPRC Ratio (pos)", "AUPRC Ratio (neg)", "AUPRC Ratio (abs)", "Early Precision Ratio (pos)", "Early Precision Ratio (neg)", "Early Precision Ratio (abs)"])
ntimes = 1000
for interval in [20]:
    for (ngenes, ntfs) in [(20, 5)]:
        for seed in [0,1,2,3,4]:
            # bifurcating
            result_dir = "../results_softODE_v3/bifurc_ngenes_" + str(ngenes) + "_ncell_" + str(ntimes) + "_seed_" + str(seed) + "/"
    
            score = pd.read_csv(result_dir + "score.csv", index_col = 0)
            score_diff = pd.read_csv(result_dir + "score_diff.csv", index_col = 0)


            score = score[score["model"] == "CeSpGRN"]
            score_diff = score_diff[score_diff["model"] == "CeSpGRN"]

            score_all = pd.concat([score_all, score], axis = 0)
            score_all_diff = pd.concat([score_all_diff, score_diff], axis = 0)


            # linear
            result_dir = "../results_softODE_v3/linear_ngenes_" + str(ngenes) + "_ncell_" + str(ntimes) + "_seed_" + str(seed) + "/"
    
            score = pd.read_csv(result_dir + "score.csv", index_col = 0)
            score_diff = pd.read_csv(result_dir + "score_diff.csv", index_col = 0)

            score = score[score["model"] == "CeSpGRN"]
            score_diff = score_diff[score_diff["model"] == "CeSpGRN"]

            score_all = pd.concat([score_all, score], axis = 0)
            score_all_diff = pd.concat([score_all_diff, score_diff], axis = 0)


fig, ax = plt.subplots( figsize=(12.0, 7.0) , nrows=2, ncols=2) 
boxplot1 = sns.boxplot(data = score_all, x = "bandwidth", y = "pearson", hue = "lambda", ax = ax[0,0])
boxplot2 = sns.boxplot(data = score_all, x = "bandwidth", y = "spearman", hue = "lambda", ax = ax[0,1])
boxplot3 = sns.boxplot(data = score_all, x = "bandwidth", y = "AUPRC Ratio (pos)", hue = "lambda", ax = ax[1,0])
boxplot4 = sns.boxplot(data = score_all, x = "bandwidth", y = "AUPRC Ratio (neg)", hue = "lambda", ax = ax[1,1])
add_median_labels(boxplot1.axes)
add_median_labels(boxplot2.axes)
add_median_labels(boxplot3.axes)
add_median_labels(boxplot4.axes)

ax[0,0].set_xticklabels(boxplot1.get_xticklabels(), rotation = 45)
ax[0,1].set_xticklabels(boxplot2.get_xticklabels(), rotation = 45)
ax[1,0].set_xticklabels(boxplot1.get_xticklabels(), rotation = 45)
ax[1,1].set_xticklabels(boxplot2.get_xticklabels(), rotation = 45)
ax[0,0].get_legend().remove()
ax[1,0].get_legend().remove()
ax[0,1].legend(loc="upper left", prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.01, 1), title = "Neighborhood size")
ax[1,1].legend(loc="upper left", prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.01, 1), title = "Neighborhood size")


fig.set_facecolor('w')
fig.suptitle("score of edge detection")
plt.tight_layout()
fig.savefig("../results_softODE_v3/hyper-parameter.png", bbox_inches = "tight")



fig, ax = plt.subplots( figsize=(12.0, 7.0) , nrows=2, ncols=2) 
boxplot1 = sns.boxplot(data = score_all_diff, x = "bandwidth", y = "pearson", hue = "lambda", ax = ax[0,0])
boxplot2 = sns.boxplot(data = score_all_diff, x = "bandwidth", y = "spearman", hue = "lambda", ax = ax[0,1])
boxplot3 = sns.boxplot(data = score_all_diff, x = "bandwidth", y = "AUPRC Ratio (pos)", hue = "lambda", ax = ax[1,0])
boxplot4 = sns.boxplot(data = score_all_diff, x = "bandwidth", y = "AUPRC Ratio (neg)", hue = "lambda", ax = ax[1,1])
add_median_labels(boxplot1.axes)
add_median_labels(boxplot2.axes)
add_median_labels(boxplot3.axes)
add_median_labels(boxplot4.axes)

ax[0,0].set_xticklabels(boxplot1.get_xticklabels(), rotation = 45)
ax[0,1].set_xticklabels(boxplot2.get_xticklabels(), rotation = 45)
ax[1,0].set_xticklabels(boxplot1.get_xticklabels(), rotation = 45)
ax[1,1].set_xticklabels(boxplot2.get_xticklabels(), rotation = 45)
ax[0,0].get_legend().remove()
ax[1,0].get_legend().remove()
ax[0,1].legend(loc="upper left", prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.01, 1), title = "Neighborhood size")
ax[1,1].legend(loc="upper left", prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.01, 1), title = "Neighborhood size")
    

fig.set_facecolor('w')
fig.suptitle("score of changing edge detection")
plt.tight_layout()
fig.savefig("../results_softODE_v3/hyper-parameter-diff.png", bbox_inches = "tight")

# In[]
score_all = pd.DataFrame(columns = ["interval", "ngenes", "time", "model", "bandwidth", "truncate_param", "lambda", "nmse","kendall-tau", "pearson", "spearman", "cosine similarity", "AUPRC Ratio (pos)", "AUPRC Ratio (neg)", "AUPRC Ratio (abs)", "Early Precision Ratio (pos)", "Early Precision Ratio (neg)", "Early Precision Ratio (abs)"])
score_all_diff = pd.DataFrame(columns = ["interval", "ngenes", "time", "model", "bandwidth", "truncate_param", "lambda", "nmse","kendall-tau", "pearson", "spearman", "cosine similarity", "AUPRC Ratio (pos)", "AUPRC Ratio (neg)", "AUPRC Ratio (abs)", "Early Precision Ratio (pos)", "Early Precision Ratio (neg)", "Early Precision Ratio (abs)"])
ntimes = 1000
for interval in [20]:
    for (ngenes, ntfs) in [(20, 5)]:
        for seed in [0,1,2,3,4]:
            # bifurcating
            result_dir = "../results_softODE_v3/bifurc_ngenes_" + str(ngenes) + "_ncell_" + str(ntimes) + "_seed_" + str(seed) + "/"
    
            score = pd.read_csv(result_dir + "score.csv", index_col = 0)
            score_diff = pd.read_csv(result_dir + "score_diff.csv", index_col = 0)

            score = score[score["model"] == "CeSpGRN-kt"]
            score_diff = score_diff[score_diff["model"] == "CeSpGRN-kt"]

            score_all = pd.concat([score_all, score], axis = 0)
            score_all_diff = pd.concat([score_all_diff, score_diff], axis = 0)


            # linear
            result_dir = "../results_softODE_v3/linear_ngenes_" + str(ngenes) + "_ncell_" + str(ntimes) + "_seed_" + str(seed) + "/"
    
            score = pd.read_csv(result_dir + "score.csv", index_col = 0)
            score_diff = pd.read_csv(result_dir + "score_diff.csv", index_col = 0)

            score = score[score["model"] == "CeSpGRN-kt"]
            score_diff = score_diff[score_diff["model"] == "CeSpGRN-kt"]

            score_all = pd.concat([score_all, score], axis = 0)
            score_all_diff = pd.concat([score_all_diff, score_diff], axis = 0)


fig, ax = plt.subplots( figsize=(12.0, 7.0) , nrows=2, ncols=2) 
boxplot1 = sns.boxplot(data = score_all, x = "bandwidth", y = "pearson", hue = "lambda", ax = ax[0,0])
boxplot2 = sns.boxplot(data = score_all, x = "bandwidth", y = "spearman", hue = "lambda", ax = ax[0,1])
boxplot3 = sns.boxplot(data = score_all, x = "bandwidth", y = "AUPRC Ratio (pos)", hue = "lambda", ax = ax[1,0])
boxplot4 = sns.boxplot(data = score_all, x = "bandwidth", y = "AUPRC Ratio (neg)", hue = "lambda", ax = ax[1,1])
add_median_labels(boxplot1.axes)
add_median_labels(boxplot2.axes)
add_median_labels(boxplot3.axes)
add_median_labels(boxplot4.axes)

ax[0,0].set_xticklabels(boxplot1.get_xticklabels(), rotation = 45)
ax[0,1].set_xticklabels(boxplot2.get_xticklabels(), rotation = 45)
ax[1,0].set_xticklabels(boxplot1.get_xticklabels(), rotation = 45)
ax[1,1].set_xticklabels(boxplot2.get_xticklabels(), rotation = 45)
ax[0,0].get_legend().remove()
ax[1,0].get_legend().remove()
ax[0,1].legend(loc="upper left", prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.01, 1), title = "Neighborhood size")
ax[1,1].legend(loc="upper left", prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.01, 1), title = "Neighborhood size")


fig.set_facecolor('w')
fig.suptitle("score of edge detection")
plt.tight_layout()
fig.savefig("../results_softODE_v3/hyper-parameter-kt.png", bbox_inches = "tight")



fig, ax = plt.subplots( figsize=(12.0, 7.0) , nrows=2, ncols=2) 
boxplot1 = sns.boxplot(data = score_all_diff, x = "bandwidth", y = "pearson", hue = "lambda", ax = ax[0,0])
boxplot2 = sns.boxplot(data = score_all_diff, x = "bandwidth", y = "spearman", hue = "lambda", ax = ax[0,1])
boxplot3 = sns.boxplot(data = score_all_diff, x = "bandwidth", y = "AUPRC Ratio (pos)", hue = "lambda", ax = ax[1,0])
boxplot4 = sns.boxplot(data = score_all_diff, x = "bandwidth", y = "AUPRC Ratio (neg)", hue = "lambda", ax = ax[1,1])
add_median_labels(boxplot1.axes)
add_median_labels(boxplot2.axes)
add_median_labels(boxplot3.axes)
add_median_labels(boxplot4.axes)

ax[0,0].set_xticklabels(boxplot1.get_xticklabels(), rotation = 45)
ax[0,1].set_xticklabels(boxplot2.get_xticklabels(), rotation = 45)
ax[1,0].set_xticklabels(boxplot1.get_xticklabels(), rotation = 45)
ax[1,1].set_xticklabels(boxplot2.get_xticklabels(), rotation = 45)
ax[0,0].get_legend().remove()
ax[1,0].get_legend().remove()
ax[0,1].legend(loc="upper left", prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.01, 1), title = "Neighborhood size")
ax[1,1].legend(loc="upper left", prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.01, 1), title = "Neighborhood size")
    

fig.set_facecolor('w')
fig.suptitle("score of changing edge detection")
plt.tight_layout()
fig.savefig("../results_softODE_v3/hyper-parameter-diff-kt.png", bbox_inches = "tight")


# In[] find setting with the highest mean score
score_all = pd.DataFrame(columns = ["interval", "ngenes", "time", "model", "bandwidth", "truncate_param", "lambda", "nmse","kendall-tau", "pearson", "spearman", "cosine similarity", "AUPRC Ratio (pos)", "AUPRC Ratio (neg)", "AUPRC Ratio (abs)", "Early Precision Ratio (pos)", "Early Precision Ratio (neg)", "Early Precision Ratio (abs)"])
score_all_diff = pd.DataFrame(columns = ["interval", "ngenes", "time", "model", "bandwidth", "truncate_param", "lambda", "nmse","kendall-tau", "pearson", "spearman", "cosine similarity", "AUPRC Ratio (pos)", "AUPRC Ratio (neg)", "AUPRC Ratio (abs)", "Early Precision Ratio (pos)", "Early Precision Ratio (neg)", "Early Precision Ratio (abs)"])
ntimes = 1000
for interval in [20]:
    for (ngenes, ntfs) in [(20, 5)]:
        for seed in [0,1,2,3,4]:
            # bifurcating
            result_dir = "../results_softODE_v3/bifurc_ngenes_" + str(ngenes) + "_ncell_" + str(ntimes) + "_seed_" + str(seed) + "/"
    
            score = pd.read_csv(result_dir + "score.csv", index_col = 0)
            score_diff = pd.read_csv(result_dir + "score_diff.csv", index_col = 0)


            score = score[(score["model"] == "CeSpGRN-kt")|(score["model"] == "CeSpGRN")]
            score_diff = score_diff[(score_diff["model"] == "CeSpGRN-kt")|(score_diff["model"] == "CeSpGRN")]

            score_all = pd.concat([score_all, score], axis = 0)
            score_all_diff = pd.concat([score_all_diff, score_diff], axis = 0)


            # linear
            result_dir = "../results_softODE_v3/linear_ngenes_" + str(ngenes) + "_ncell_" + str(ntimes) + "_seed_" + str(seed) + "/"
    
            score = pd.read_csv(result_dir + "score.csv", index_col = 0)
            score_diff = pd.read_csv(result_dir + "score_diff.csv", index_col = 0)

            score = score[(score["model"] == "CeSpGRN-kt")|(score["model"] == "CeSpGRN")]
            score_diff = score_diff[(score_diff["model"] == "CeSpGRN-kt")|(score_diff["model"] == "CeSpGRN")]

            score_all = pd.concat([score_all, score], axis = 0)
            score_all_diff = pd.concat([score_all_diff, score_diff], axis = 0)

mean_score = score_all.groupby(by = ["model", "bandwidth", "truncate_param", "lambda"]).mean()
mean_score = mean_score.drop(["time"], axis = 1)
mean_score.to_csv("../results_softODE_v3/mean_score.csv")

mean_score_diff = score_all_diff.groupby(by = ["model", "bandwidth", "truncate_param", "lambda"]).mean()
mean_score_diff = mean_score_diff.drop(["time"], axis = 1)
mean_score_diff.to_csv("../results_softODE_v3/mean_score_diff.csv")

# %%
