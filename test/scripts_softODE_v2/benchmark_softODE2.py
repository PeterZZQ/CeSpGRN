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


def summarize_scores(setting):
    ntimes = setting["ntimes"]
    interval = setting["interval"]
    ngenes = setting["ngenes"]
    ntfs = setting["ntfs"]
    seed = setting["seed"]

    print("ntimes: " + str(ntimes) + ", interval: " + str(interval) + ", ngenes: " + str(ngenes) + ", seed: " + str(seed) + ", initial graph: sergio\n")
    scores = pd.DataFrame(columns = ["interval", "ngenes", "time", "model", "bandwidth", "truncate_param", "lambda", "nmse","kendall-tau", "pearson", "spearman", "cosine similarity", "AUPRC Ratio (pos)", "AUPRC Ratio (neg)", "AUPRC Ratio (abs)", "Early Precision Ratio (pos)", "Early Precision Ratio (neg)", "Early Precision Ratio (abs)"])
    result_dir = "../results_softODE_v2/linear_ngenes_" + str(ngenes) + "_ncell_" + str(ntimes) + "_seed_" + str(seed) + "/"
    data_dir = path + "linear_ngenes_" + str(ngenes) + "_ncell_" + str(ntimes) + "_seed_" + str(seed) + "/"
    
    # the data smapled from GGM is zero-mean
    # X = np.load(data_dir + "expr.npy")
    gt_adj = np.load(data_dir + "GRNs.npy")
    sim_time = np.load(data_dir + "pseudotime.npy")
    print("data loaded.")
    
    # genie3            
    thetas = np.load(file = result_dir + "theta_genie.npy")
    score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "GENIE3")
    scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    print("genie3.")

    # genie3-tf
    thetas = np.load(file = result_dir + "theta_genie_tf.npy")
    score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "GENIE3-TF")
    scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    print("genie3-tf.")

    # genie3-dyn
    thetas = np.load(file = result_dir + "theta_genie_dyn.npy")
    score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "GENIE3-Dyn")
    scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    print("genie3-dyn.")
    
    
    # genie3-dyn-tf 
    thetas = np.load(file = result_dir + "theta_genie_dyn_tf.npy")
    score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "GENIE3-Dyn-TF")
    scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    print("genie3-dyn-tf.")


    # SCODE (normalize)
    thetas = np.load(file = result_dir + "theta_scode_norm.npy")
    score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "SCODE")
    scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    print("SCODE (True T).")

    
    # SCODE-DYN (nomralize)
    thetas = np.load(file = result_dir + "theta_scode_dyn_norm.npy")
    score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "SCODE-Dyn")
    scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    print("SCODE-DYN (True T).")

    # CSN        
    thetas = np.load(file = result_dir + "theta_CSN.npy")
    score = calc_scores(thetas_inf = thetas, thetas_gt = np.abs(gt_adj), interval = interval, model = "CSN")
    scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    print("CSN.") 

    # admm, hyper-parameter
    for bandwidth in [0.1, 1, 10]:
        for truncate_param in [30, 100]:
            for lamb in [0.0001]:
                thetas = np.load(file = result_dir + "neigh_thetas_ngenes20_band_" + str(bandwidth) + "_lamb_" + str(lamb) + "_trun_" + str(truncate_param) + ".npy")
                score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "CeSpGRN", bandwidth = bandwidth, truncate_param = truncate_param, lamb = lamb)
                scores = pd.concat([scores, score], axis = 0, ignore_index = True)

                fig = plt.figure(figsize = (10,7))
                X_umap = umap_op.fit_transform(thetas.reshape(ntimes * nsample, -1))
                ax = fig.add_subplot()
                ax.scatter(X_umap[:,0], X_umap[:,1], c = sim_time, s = 5)
                fig.savefig(result_dir + "infer_G_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_0.png", bbox_inches = "tight")

                thetas = np.load(file = result_dir + "TF_neigh_thetas_ngenes20_band_" + str(bandwidth) + "_beta_100_trun_" + str(truncate_param) + ".npy")
                score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "CeSpGRN-TF", bandwidth = bandwidth, truncate_param = truncate_param, lamb = lamb)
                scores = pd.concat([scores, score], axis = 0, ignore_index = True)

    scores.to_csv(result_dir + "score.csv")


settings = []
for interval in [20]:
    for (ngenes, ntfs) in [(20, 5)]:
        for seed in [0,1,2,3,4]:
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
    result_dir = "../results_softODE_v2/linear_ngenes_" + str(ngenes) + "_ncell_" + str(ntimes) + "_seed_" + str(seed) + "/"
    data_dir = path + "linear_ngenes_" + str(ngenes) + "_ncell_" + str(ntimes) + "_seed_" + str(seed) + "/"
    
    # the data smapled from GGM is zero-mean
    # X = np.load(data_dir + "expr.npy")
    gt_adj = np.load(data_dir + "GRNs.npy")
    sim_time = np.load(data_dir + "pseudotime.npy")
    print("data loaded.")

    step = 400
    gt_adj = gt_adj[step::,:,:] - gt_adj[:-step:,:,:]   

    # genie3            
    thetas = np.load(file = result_dir + "theta_genie.npy")
    thetas = thetas[step::,:,:] - thetas[:-step:,:,:]
    score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "GENIE3")
    scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    print("genie3.")

    # genie3-tf
    thetas = np.load(file = result_dir + "theta_genie_tf.npy")
    thetas = thetas[step::,:,:] - thetas[:-step:,:,:]
    score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "GENIE3-TF")
    scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    print("genie3-tf.")

    # genie3-dyn
    thetas = np.load(file = result_dir + "theta_genie_dyn.npy")
    thetas = thetas[step::,:,:] - thetas[:-step:,:,:]
    score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "GENIE3-Dyn")
    scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    print("genie3-dyn.")
    
    
    # genie3-dyn-tf 
    thetas = np.load(file = result_dir + "theta_genie_dyn_tf.npy")
    thetas = thetas[step::,:,:] - thetas[:-step:,:,:]
    score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "GENIE3-Dyn-TF")
    scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    print("genie3-dyn-tf.")


    # SCODE (normalize)
    thetas = np.load(file = result_dir + "theta_scode_norm.npy")
    thetas = thetas[step::,:,:] - thetas[:-step:,:,:]
    score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "SCODE")
    scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    print("SCODE (True T).")

    
    # SCODE-DYN (nomralize)
    thetas = np.load(file = result_dir + "theta_scode_dyn_norm.npy")
    thetas = thetas[step::,:,:] - thetas[:-step:,:,:]
    score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "SCODE-Dyn")
    scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    print("SCODE-DYN (True T).")

    # CSN       
    thetas = np.load(file = result_dir + "theta_CSN.npy")
    thetas = thetas[step::,:,:] - thetas[:-step:,:,:]  
    score = calc_scores(thetas_inf = thetas, thetas_gt = np.abs(gt_adj), interval = interval, model = "CSN")
    scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    print("CSN.") 

    for bandwidth in [0.1, 1, 10]:
        for truncate_param in [30, 100]:
            for lamb in [0.0001]:
                thetas = np.load(file = result_dir + "neigh_thetas_ngenes20_band_" + str(bandwidth) + "_lamb_" + str(lamb) + "_trun_" + str(truncate_param) + ".npy")
                thetas = thetas[step::,:,:] - thetas[:-step:,:,:]  
                score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "CeSpGRN", bandwidth = bandwidth, truncate_param = truncate_param, lamb = lamb)
                scores = pd.concat([scores, score], axis = 0, ignore_index = True)

                fig = plt.figure(figsize = (10,7))
                X_umap = umap_op.fit_transform(thetas.reshape(ntimes * nsample, -1))
                ax = fig.add_subplot()
                ax.scatter(X_umap[:,0], X_umap[:,1], c = sim_time, s = 5)
                fig.savefig(result_dir + "infer_G_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_0.png", bbox_inches = "tight")

                thetas = np.load(file = result_dir + "TF_neigh_thetas_ngenes20_band_" + str(bandwidth) + "_beta_100_trun_" + str(truncate_param) + ".npy")
                thetas = thetas[step::,:,:] - thetas[:-step:,:,:]  
                score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "CeSpGRN-TF", bandwidth = bandwidth, truncate_param = truncate_param, lamb = lamb)
                scores = pd.concat([scores, score], axis = 0, ignore_index = True)

    scores.to_csv(result_dir + "score_diff.csv")

settings = []
for interval in [20]:
    for (ngenes, ntfs) in [(20, 5)]:
        for seed in [0,1,2,3,4]:
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
for interval in [20]:
    for (ngenes, ntfs) in [(20, 5)]:
        for seed in [0,1,2,3,4]:
            print("ntimes: " + str(ntimes) + ", interval: " + str(interval) + ", ngenes: " + str(ngenes) + ", seed: " + str(seed) + ", initial graph: sergio")
            result_dir = "../results_softODE_v2/linear_ngenes_" + str(ngenes) + "_ncell_" + str(ntimes) + "_seed_" + str(seed) + "/"
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
            result_dir = "../results_softODE_v2/linear_ngenes_" + str(ngenes) + "_ncell_" + str(ntimes) + "_seed_" + str(seed) + "/"
            score = pd.read_csv(result_dir + "score_diff.csv", index_col = 0)
            mean_score = score.groupby(by = ["model", "bandwidth", "truncate_param", "lambda"]).mean()
            mean_score = mean_score.drop(["time"], axis = 1)
            mean_score.to_csv(result_dir + "mean_score_diff.csv")
            # display(mean_score)

# In[]
for interval in [20]:
    for (ngenes, ntfs) in [(20, 5)]:
        for seed in [0,1,2,3,4]:
            # bifurcating
            result_dir = "../results_softODE_v2/bifurc_ngenes_" + str(ngenes) + "_ncell_" + str(ntimes) + "_seed_" + str(seed) + "/"
    
            score = pd.read_csv(result_dir + "score.csv", index_col = 0)
            score_diff = pd.read_csv(result_dir + "score_diff.csv", index_col = 0)

            score = score[score["model"] == "CeSpGRN"]
            score_diff = score_diff[score_diff["model"] == "CeSpGRN"]

            score_all = pd.concat([score_all, score], axis = 0)
            score_all_diff = pd.concat([score_all_diff, score_diff], axis = 0)


            # linear
            result_dir = "../results_softODE_v2/linear_ngenes_" + str(ngenes) + "_ncell_" + str(ntimes) + "_seed_" + str(seed) + "/"
    
            score = pd.read_csv(result_dir + "score.csv", index_col = 0)
            score_diff = pd.read_csv(result_dir + "score_diff.csv", index_col = 0)

            score = score[score["model"] == "CeSpGRN"]
            score_diff = score_diff[score_diff["model"] == "CeSpGRN"]

            score_all = pd.concat([score_all, score], axis = 0)
            score_all_diff = pd.concat([score_all_diff, score_diff], axis = 0)


fig, ax = plt.subplots( figsize=(10.0, 10.0) , nrows=2, ncols=2) 
boxplot1 = sns.boxplot(data = score_all, x = "bandwidth", y = "pearson", hue = "truncate_param", ax = ax[0,0])
boxplot2 = sns.boxplot(data = score_all, x = "bandwidth", y = "spearman", hue = "truncate_param", ax = ax[0,1])
boxplot3 = sns.boxplot(data = score_all, x = "bandwidth", y = "AUPRC (pos)", hue = "truncate_param", ax = ax[1,0])
boxplot4 = sns.boxplot(data = score_all, x = "bandwidth", y = "AUPRC (neg)", hue = "truncate_param", ax = ax[1,1])


ax[0,0].set_xticklabels(boxplot1.get_xticklabels(), rotation = 45)
ax[0,1].set_xticklabels(boxplot2.get_xticklabels(), rotation = 45)
ax[1,0].set_xticklabels(boxplot1.get_xticklabels(), rotation = 45)
ax[1,1].set_xticklabels(boxplot2.get_xticklabels(), rotation = 45)

fig.set_facecolor('w')
fig.suptitle("score of edge detection")
plt.tight_layout()
fig.savefig(result_dir + "hyper-parameter.png", bbox_inches = "tight")



fig, ax = plt.subplots( figsize=(10.0, 10.0) , nrows=2, ncols=2) 
boxplot1 = sns.boxplot(data = score_all_diff, x = "bandwidth", y = "pearson", hue = "truncate_param", ax = ax[0,0])
boxplot2 = sns.boxplot(data = score_all_diff, x = "bandwidth", y = "spearman", hue = "truncate_param", ax = ax[0,1])
boxplot3 = sns.boxplot(data = score_all_diff, x = "bandwidth", y = "AUPRC (pos)", hue = "truncate_param", ax = ax[1,0])
boxplot4 = sns.boxplot(data = score_all_diff, x = "bandwidth", y = "AUPRC (neg)", hue = "truncate_param", ax = ax[1,1])


ax[0,0].set_xticklabels(boxplot1.get_xticklabels(), rotation = 45)
ax[0,1].set_xticklabels(boxplot2.get_xticklabels(), rotation = 45)
ax[1,0].set_xticklabels(boxplot1.get_xticklabels(), rotation = 45)
ax[1,1].set_xticklabels(boxplot2.get_xticklabels(), rotation = 45)

fig.set_facecolor('w')
fig.suptitle("score of changing edge detection")
plt.tight_layout()
fig.savefig(result_dir + "hyper-parameter-diff.png", bbox_inches = "tight")





# In[]
for interval in [20]:
    for (ngenes, ntfs) in [(20, 5)]:
        for seed in [0,1,2,3,4]:
            # bifurcating
            result_dir = "../results_softODE_v2/bifurc_ngenes_" + str(ngenes) + "_ncell_" + str(ntimes) + "_seed_" + str(seed) + "/"
    
            score = pd.read_csv(result_dir + "score.csv", index_col = 0)
            score_diff = pd.read_csv(result_dir + "score_diff.csv", index_col = 0)

            score = score[(score["model"] == "GENIE3")|(score["model"] == "GENIE3-Dyn")|(score["model"] == "SCODE")|(score["model"] == "SCODE-Dyn")|(score["model"] == "CeSpGRN")|(score["model"] == "CSN")]
            score_diff = score_diff[(score_diff["model"] == "GENIE3")|(score_diff["model"] == "GENIE3-Dyn")|(score_diff["model"] == "SCODE")|(score_diff["model"] == "SCODE-Dyn")|(score_diff["model"] == "CeSpGRN")|(score_diff["model"] == "CSN")]

            score_all = pd.concat([score_all, score], axis = 0)
            score_all_diff = pd.concat([score_all_diff, score_diff], axis = 0)

            fig, ax = plt.subplots( figsize=(12.0, 7.0) , nrows=1, ncols=2) 
            boxplot1 = sns.boxplot(data = score, x = "model", y = "pearson", ax = ax[0])
            boxplot2 = sns.boxplot(data = score, x = "model", y = "spearman", ax = ax[1])

            ax[0].set_xticklabels(boxplot1.get_xticklabels(), rotation = 45)
            ax[1].set_xticklabels(boxplot2.get_xticklabels(), rotation = 45)

            
            fig.set_facecolor('w')
            fig.suptitle("score of edge detection, change interval: " + str(interval))
            plt.tight_layout()
            fig.savefig(result_dir + "compare_models.png", bbox_inches = "tight")

            # linear
            result_dir = "../results_softODE_v2/linear_ngenes_" + str(ngenes) + "_ncell_" + str(ntimes) + "_seed_" + str(seed) + "/"
    
            score = pd.read_csv(result_dir + "score.csv", index_col = 0)
            score_diff = pd.read_csv(result_dir + "score_diff.csv", index_col = 0)

            score = score[(score["model"] == "GENIE3")|(score["model"] == "GENIE3-Dyn")|(score["model"] == "SCODE")|(score["model"] == "SCODE-Dyn")|(score["model"] == "CeSpGRN")|(score["model"] == "CSN")]
            score_diff = score_diff[(score_diff["model"] == "GENIE3")|(score_diff["model"] == "GENIE3-Dyn")|(score_diff["model"] == "SCODE")|(score_diff["model"] == "SCODE-Dyn")|(score_diff["model"] == "CeSpGRN")|(score_diff["model"] == "CSN")]

            score_all = pd.concat([score_all, score], axis = 0)
            score_all_diff = pd.concat([score_all_diff, score_diff], axis = 0)

            fig, ax = plt.subplots( figsize=(12.0, 7.0) , nrows=1, ncols=2) 
            boxplot1 = sns.boxplot(data = score, x = "model", y = "pearson", ax = ax[0])
            boxplot2 = sns.boxplot(data = score, x = "model", y = "spearman", ax = ax[1])

            ax[0].set_xticklabels(boxplot1.get_xticklabels(), rotation = 45)
            ax[1].set_xticklabels(boxplot2.get_xticklabels(), rotation = 45)

            
            fig.set_facecolor('w')
            fig.suptitle("score of edge detection, change interval: " + str(interval))
            plt.tight_layout()
            fig.savefig(result_dir + "compare_models.png", bbox_inches = "tight")


fig, ax = plt.subplots( figsize=(12.0, 10.0) , nrows=1, ncols=2) 
boxplot1 = sns.boxplot(data = score_all, x = "bandwidth", y = "pearson", hue = "truncate_param", ax = ax[0,0])
boxplot2 = sns.boxplot(data = score_all, x = "bandwidth", y = "spearman", hue = "truncate_param", ax = ax[0,1])
boxplot3 = sns.boxplot(data = score_all, x = "bandwidth", y = "AUPRC (pos)", hue = "truncate_param", ax = ax[1,0])
boxplot4 = sns.boxplot(data = score_all, x = "bandwidth", y = "AUPRC (neg)", hue = "truncate_param", ax = ax[1,1])


ax[0,0].set_xticklabels(boxplot1.get_xticklabels(), rotation = 45)
ax[0,1].set_xticklabels(boxplot2.get_xticklabels(), rotation = 45)
ax[1,0].set_xticklabels(boxplot1.get_xticklabels(), rotation = 45)
ax[1,1].set_xticklabels(boxplot2.get_xticklabels(), rotation = 45)


fig.set_facecolor('w')
fig.suptitle("score of edge detection")
plt.tight_layout()
fig.savefig("../results_GGM/compare_models.png", bbox_inches = "tight")        


fig, ax = plt.subplots( figsize=(12.0, 10.0) , nrows=1, ncols=2) 
boxplot1 = sns.boxplot(data = score_all_diff, x = "bandwidth", y = "pearson", hue = "truncate_param", ax = ax[0,0])
boxplot2 = sns.boxplot(data = score_all_diff, x = "bandwidth", y = "spearman", hue = "truncate_param", ax = ax[0,1])
boxplot3 = sns.boxplot(data = score_all_diff, x = "bandwidth", y = "AUPRC (pos)", hue = "truncate_param", ax = ax[1,0])
boxplot4 = sns.boxplot(data = score_all_diff, x = "bandwidth", y = "AUPRC (neg)", hue = "truncate_param", ax = ax[1,1])


ax[0,0].set_xticklabels(boxplot1.get_xticklabels(), rotation = 45)
ax[0,1].set_xticklabels(boxplot2.get_xticklabels(), rotation = 45)
ax[1,0].set_xticklabels(boxplot1.get_xticklabels(), rotation = 45)
ax[1,1].set_xticklabels(boxplot2.get_xticklabels(), rotation = 45)


fig.set_facecolor('w')
fig.suptitle("score of changing edge detection")
plt.tight_layout()
fig.savefig("../results_GGM/compare_models_diff.png", bbox_inches = "tight")    