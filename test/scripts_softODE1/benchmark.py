# In[0]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import sys, os
sys.path.append('../../src/')
import bmk_beeline as bmk
import genie3

plt.rcParams["font.size"] = 16


def preprocess(counts): 
    """\
    Input:
    counts = (ntimes, ngenes)
    
    Description:
    ------------
    Preprocess the dataset
    """
    # normalize according to the library size
    
    libsize = np.median(np.sum(counts, axis = 1))
    counts = counts / np.sum(counts, axis = 1)[:,None] * libsize
        
    counts = np.log1p(counts)
    return counts

# In[1] test hyper-parameter, use 20 genes as example
print("test results:")
ntimes = 1000
path = "../../data/continuousODE/sergio_dense/"
# setting
ngenes = 20
ntfs = 5
stepsize = 0.0001
for interval in [50, 100, 200]:
    # read in the data
    gt_adj = np.load(path + "ngenes_" + str(ngenes) + "_interval_" + str(interval) + "_stepsize_" + str(stepsize) + "/GRNs.npy")
    score = pd.DataFrame(columns =["model", "kendall-tau","probability of success", "pearson", "cosine similarity", "time", "bandwidth", "lamb"])
    result_dir = "../results_softODE_sergio/results_ngenes_" + str(ngenes) + "_interval_" + str(interval) + "_stepsize_" + str(stepsize) + "/"           
    print("ngenes_" + str(ngenes) + "_interval_" + str(interval) + "_stepsize_" + str(stepsize))

    # admm
    alpha = 2
    rho = 1.7
    for bandwidth in [0.01, 0.1, 1, 10]:
        for lamb in [0.001, 0.01, 0.1, 0.5]:
            thetas = np.load(result_dir + "thetas_" + str(bandwidth) + "_" + str(alpha) + "_" + str(lamb) + "_" + str(rho) + "_true_log.npy") 
            for time in range(0, ntimes):
                # benchmark genie
                kt,_ = bmk.kendalltau(G_inf = thetas[time], G_true = gt_adj[time])
                ps = bmk.PS(G_inf = thetas[time], G_true = gt_adj[time])
                pearson_val, pval = bmk.pearson(G_inf = thetas[time], G_true = gt_adj[time])
                cosine_sim = bmk.cossim(G_inf = thetas[time], G_true = gt_adj[time])            
                score = score.append({"kendall-tau": kt, "probability of success":ps, "pearson": pearson_val, "cosine similarity": cosine_sim, 
                                    "model":"admm_2_1.7", "time":time, "bandwidth":bandwidth, "lamb":lamb}, ignore_index=True)
    
    # alpha = 1
    # rho = None
    # for bandwidth in [0.01, 0.1, 1, 10]:
    #     for lamb in [0.001, 0.01, 0.1, 0.5]:
    #         thetas = np.load(result_dir + "thetas_" + str(bandwidth) + "_" + str(alpha) + "_" + str(lamb) + "_" + str(rho) + "_obs.npy") 
    #         for time in range(0, ntimes):
    #             kt,_ = bmk.kendalltau(G_inf = thetas[time], G_true = gt_adj[time])
    #             ps = bmk.PS(G_inf = thetas[time], G_true = gt_adj[time])
    #             pearson_val, pval = bmk.pearson(G_inf = thetas[time], G_true = gt_adj[time])
    #             cosine_sim = bmk.cossim(G_inf = thetas[time], G_true = gt_adj[time])            
    #             score = score.append({"kendall-tau": kt, "probability of success":ps, "pearson": pearson_val, "cosine similarity": cosine_sim, 
    #                                 "model":"admm_1_None", "time":time, "bandwidth":bandwidth, "lamb":lamb}, ignore_index=True)
    
    score.to_csv(result_dir + "score_admm_true_log.csv")

# In[2] check the performance of different alpha/rho choices, result is they are similar
# for interval in [50, 100, 200]:
#     result_dir = "../results_softODE_sergio/results_ngenes_" + str(ngenes) + "_interval_" + str(interval) + "_stepsize_" + str(stepsize) + "/"
#     score = pd.read_csv(result_dir + "score_admm_hyperparams_obs.csv", index_col = 0)
#     fig = plt.figure(figsize = (30,7))
#     axs = fig.subplots(1,3)
#     sns.boxplot(data = score, x = "model", y = "kendall-tau", ax = axs[0])
#     axs[0].set_title("kendall-tau: ngenes: {:d}, interval: {:d}, stepsize: {:f}".format(int(ngenes), int(interval), stepsize))
#     sns.boxplot(data = score, x = "model", y = "pearson", ax = axs[1])
#     axs[1].set_title("Pearson: ngenes: {:d}, interval: {:d}, stepsize: {:f}".format(int(ngenes), int(interval), stepsize))            
#     sns.boxplot(data = score, x = "model", y = "cosine similarity", ax = axs[2])
#     axs[2].set_title("Cosine: ngenes: {:d}, interval: {:d}, stepsize: {:f}".format(int(ngenes), int(interval), stepsize))
    # fig.savefig(result_dir + "boxplot_mode_obs.png", bbox_inches = "tight")

# In[3] check the performance of different hyper-parameters
for interval in [50, 100, 200]:
    result_dir = "../results_softODE_sergio/results_ngenes_" + str(ngenes) + "_interval_" + str(interval) + "_stepsize_" + str(stepsize) + "/"
    score = pd.read_csv(result_dir + "score_admm_hyperparams_true_log.csv", index_col = 0)
    # use the one with 2 and 1.7
    score = score[score["model"] == "admm_2_1.7"]
    fig = plt.figure(figsize = (30,7))
    axs = fig.subplots(1,3)
    sns.boxplot(data = score, x = "bandwidth", y = "kendall-tau", hue = "lamb", ax = axs[0])
    axs[0].set_title("kendall-tau: ngenes: {:d}, interval: {:d}".format(int(ngenes), int(interval)))
    sns.boxplot(data = score, x = "bandwidth", y = "pearson", hue = "lamb", ax = axs[1])
    axs[1].set_title("Pearson: ngenes: {:d}, interval: {:d}".format(int(ngenes), int(interval)))            
    sns.boxplot(data = score, x = "bandwidth", y = "cosine similarity", hue = "lamb", ax = axs[2])
    axs[2].set_title("Cosine: ngenes: {:d}, interval: {:d}".format(int(ngenes), int(interval)))
    fig.savefig(result_dir + "boxplot_hyperparams_true_log.png", bbox_inches = "tight")
            

# In[] include genie3
# using observed counts give better performance for genie3
ntimes = 1000
# setting
ngenes = 20
ntfs = 5
stepsize = 0.0001
for interval in [50, 100, 200]:
    result_dir = "../results_softODE_sergio/results_ngenes_" + str(ngenes) + "_interval_" + str(interval) + "_stepsize_" + str(stepsize) + "/"
    # admm
    score_admm = pd.read_csv(result_dir + "score_admm_hyperparams.csv", index_col = 0)
    # use the one with 2 and 1.7
    score_admm = score_admm[score_admm["model"] == "admm_2_1.7"]
    score_admm = score_admm[score_admm["lamb"] == 0.1]
    # genie3 score
    score_genie = pd.read_csv(result_dir + "score.csv", index_col = 0)
    score_genie = score_genie[(score_genie["model"] == "genie")|(score_genie["model"] == "genie_obs")]

    score = pd.concat([score_admm, score_genie], axis = 0)

    fig = plt.figure(figsize = (25,7))
    ax = fig.subplots(1,3)
    sns.boxplot(data = score, x = "model", y = "kendall-tau", ax = ax[0])
    ax[0].set_title("Kendall-tau: ngenes: {:d}, interval: {:d}, \nstepsize: {:f}".format(int(ngenes), int(interval), stepsize))
    
    sns.boxplot(data = score, x = "model", y = "pearson", ax = ax[1])
    ax[1].set_title("Pearson: ngenes: {:d}, interval: {:d}, \nstepsize: {:f}".format(int(ngenes), int(interval), stepsize))            

    sns.boxplot(data = score, x = "model", y = "cosine similarity", ax = ax[2])
    ax[2].set_title("Cosine: ngenes: {:d}, interval: {:d}, \nstepsize: {:f}".format(int(ngenes), int(interval), stepsize))
    fig.savefig(result_dir + "compare_genie_admm.png", bbox_inches = "tight")

            

# %%
