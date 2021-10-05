# In[0]
from operator import index
import pandas as pd
import numpy as np
import torch
import torch.nn
import matplotlib.pyplot as plt
import sys, os
sys.path.append('../../src/')

import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

import bmk_beeline as bmk
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns
import genie3

# In[] Test Genie3
ntimes = 3000
path = "../../data/GGM/"
for interval in [50, 100, 200]:
    for (ngenes, ntfs) in [(20, 5), (30, 10), (50, 20)]:
        result_dir = "../results/GGM_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "/"
        # the data smapled from GGM is zero-mean
        X = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/expr.npy")
        gt_adj = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/Gs.npy")

        X_genie = X.reshape(ntimes, ngenes)
        # genie_theta of the shape (ntimes, ngenes, ngenes)
        genie_theta = genie3.GENIE3(X_genie, gene_names=None, regulators='all',tree_method='RF',K='sqrt',ntrees=1000,nthreads=1)
        genie_theta = np.repeat(genie_theta[None, :, :],ntimes,axis=0)
        np.save(file = result_dir + "genie_theta.npy", arr = genie_theta)

# In[]
ntimes = 3000
path = "../../data/GGM/"
score1 = pd.DataFrame(columns =["interval", "ngenes", "nmse", "probability of success", "pearson", "cosine similarity", "time"])
score2 = pd.DataFrame(columns =["interval", "ngenes", "nmse", "probability of success", "pearson", "cosine similarity", "time"])
score3 = pd.DataFrame(columns =["interval", "ngenes", "nmse", "probability of success", "pearson", "cosine similarity", "time"])
score_rand = pd.DataFrame(columns =["interval", "ngenes", "nmse","probability of success", "pearson", "cosine similarity", "time"])

for interval in [50, 100, 200]:
    for (ngenes, ntfs) in [(20, 5), (30, 10), (50, 20)]:
        result_dir = "../results/GGM_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "/"
        # the data smapled from GGM is zero-mean
        X = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/expr.npy")
        gt_adj = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/Gs.npy")

        # hyper-parameter
        alpha = 1
        rho = "None"
        bandwidth = 0.1
        lamb = 0.01
        data = str(bandwidth) + "_" + str(alpha) + "_" + str(lamb) + "_" + rho
        thetas = np.load(file = result_dir + "thetas_" + data + ".npy")
        
        for time in range(0, ntimes):
            np.random.seed(0)
            # benchmark random baseline
            thetas_rand = np.random.randn(ngenes,ngenes)
            # TODO: include testing method, check glad paper
            nmse_rand = bmk.NMSE(G_inf = thetas_rand, G_true = gt_adj[time])
            ps_rand = bmk.PS(G_inf = thetas_rand, G_true = gt_adj[time])

            pearson_rand_val = bmk.pearson(G_inf = thetas_rand, G_true = gt_adj[time])
            cosine_sim_rand = bmk.cossim(G_inf = thetas_rand, G_true = gt_adj[time])        
            score_rand = score_rand.append({"interval": interval,
                                            "ngenes": ngenes,
                                            "nmse": nmse_rand, 
                                            "probability of success":ps_rand, 
                                            "pearson": pearson_rand_val, 
                                            "cosine similarity": cosine_sim_rand, 
                                            "time":time}, ignore_index=True)
            # benchmark admm
            nmse = bmk.NMSE(G_inf = thetas[time], G_true = gt_adj[time])
            ps = bmk.PS(G_inf = thetas[time], G_true = gt_adj[time])

            pearson_val, pval = bmk.pearson(G_inf = thetas[time], G_true = gt_adj[time])
            cosine_sim = bmk.cossim(G_inf = thetas[time], G_true = gt_adj[time])            
            score1 = score1.append({"interval": interval,
                                    "ngenes": ngenes,
                                    "nmse": nmse, 
                                    "probability of success":ps, 
                                    "pearson": pearson_val, 
                                    "cosine similarity": cosine_sim, 
                                    "time":time}, ignore_index=True)

        # hyper-parameter
        alpha = 2
        rho = "1.7"
        bandwidth = 0.1
        lamb = 0.01
        data = str(bandwidth) + "_" + str(alpha) + "_" + str(lamb) + "_" + rho
        thetas = np.load(file = result_dir + "thetas_" + data + ".npy")
        for time in range(0, ntimes):
            # benchmark admm
            nmse = bmk.NMSE(G_inf = thetas[time], G_true = gt_adj[time])
            ps = bmk.PS(G_inf = thetas[time], G_true = gt_adj[time])

            pearson_val, pval = bmk.pearson(G_inf = thetas[time], G_true = gt_adj[time])
            cosine_sim = bmk.cossim(G_inf = thetas[time], G_true = gt_adj[time])            
            score2 = score2.append({"interval": interval,
                                    "ngenes": ngenes,
                                    "nmse": nmse, 
                                    "probability of success":ps, 
                                    "pearson": pearson_val, 
                                    "cosine similarity": cosine_sim, 
                                    "time":time}, ignore_index=True)

        # hyper-parameter
        alpha = 1
        rho = "1.7"
        bandwidth = 0.1
        lamb = 0.01
        data = str(bandwidth) + "_" + str(alpha) + "_" + str(lamb) + "_" + rho
        thetas = np.load(file = result_dir + "thetas_" + data + ".npy")
        for time in range(0, ntimes):
            # benchmark admm
            nmse = bmk.NMSE(G_inf = thetas[time], G_true = gt_adj[time])
            ps = bmk.PS(G_inf = thetas[time], G_true = gt_adj[time])

            pearson_val, pval = bmk.pearson(G_inf = thetas[time], G_true = gt_adj[time])
            cosine_sim = bmk.cossim(G_inf = thetas[time], G_true = gt_adj[time])            
            score3 = score3.append({"interval": interval,
                                    "ngenes": ngenes,
                                    "nmse": nmse, 
                                    "probability of success":ps, 
                                    "pearson": pearson_val, 
                                    "cosine similarity": cosine_sim, 
                                    "time":time}, ignore_index=True)

# save results
score1.to_csv("../results/script2_scores/score_1_None.csv")
score2.to_csv("../results/script2_scores/score_2_1.7.csv")
score3.to_csv("../results/script2_scores/score_1_1.7.csv")
score_rand.to_csv("../results/script2_scores/score_rand.csv")



# In[] Differences
ntimes = 3000
path = "../../data/GGM/"
score1 = pd.DataFrame(columns =["interval", "ngenes", "nmse", "probability of success", "pearson", "cosine similarity", "time"])
score2 = pd.DataFrame(columns =["interval", "ngenes", "nmse", "probability of success", "pearson", "cosine similarity", "time"])
score3 = pd.DataFrame(columns =["interval", "ngenes", "nmse", "probability of success", "pearson", "cosine similarity", "time"])
score_rand = pd.DataFrame(columns =["interval", "ngenes", "nmse","probability of success", "pearson", "cosine similarity", "time"])

for interval in [50, 100, 200]:
    for (ngenes, ntfs) in [(20, 5), (30, 10), (50, 20)]:
        result_dir = "../results/GGM_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "/"
        # the data smapled from GGM is zero-mean
        X = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/expr.npy")
        gt_adj = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/Gs.npy")
        
        # calculate the diffs
        # compare every 100 steps
        steps = 100
        gt_adj = gt_adj[steps::,:,:] - gt_adj[:-steps,:,:]
        # hyper-parameter
        alpha = 1
        rho = "None"
        bandwidth = 0.1
        lamb = 0.01
        data = str(bandwidth) + "_" + str(alpha) + "_" + str(lamb) + "_" + rho
        thetas = np.load(file = result_dir + "thetas_" + data + ".npy")
        # calculate the diffs
        thetas = thetas[steps::,:,:] - thetas[:-steps,:,:]
        ntimes_diff = thetas.shape[0]
        
        for time in range(0, ntimes_diff):
            np.random.seed(0)
            # benchmark random baseline
            thetas_rand = np.random.randn(ngenes,ngenes)
            # TODO: include testing method, check glad paper
            nmse_rand = bmk.NMSE(G_inf = thetas_rand, G_true = gt_adj[time])
            ps_rand = bmk.PS(G_inf = thetas_rand, G_true = gt_adj[time])

            pearson_rand_val = bmk.pearson(G_inf = thetas_rand, G_true = gt_adj[time])
            cosine_sim_rand = bmk.cossim(G_inf = thetas_rand, G_true = gt_adj[time])        
            score_rand = score_rand.append({"interval": interval,
                                            "ngenes": ngenes,
                                            "nmse": nmse_rand, 
                                            "probability of success":ps_rand, 
                                            "pearson": pearson_rand_val, 
                                            "cosine similarity": cosine_sim_rand, 
                                            "time":time}, ignore_index=True)
            # benchmark admm
            nmse = bmk.NMSE(G_inf = thetas[time], G_true = gt_adj[time])
            ps = bmk.PS(G_inf = thetas[time], G_true = gt_adj[time])

            pearson_val, pval = bmk.pearson(G_inf = thetas[time], G_true = gt_adj[time])
            cosine_sim = bmk.cossim(G_inf = thetas[time], G_true = gt_adj[time])            
            score1 = score1.append({"interval": interval,
                                    "ngenes": ngenes,
                                    "nmse": nmse, 
                                    "probability of success":ps, 
                                    "pearson": pearson_val, 
                                    "cosine similarity": cosine_sim, 
                                    "time":time}, ignore_index=True)

        # hyper-parameter
        alpha = 2
        rho = "1.7"
        bandwidth = 0.1
        lamb = 0.01
        data = str(bandwidth) + "_" + str(alpha) + "_" + str(lamb) + "_" + rho
        thetas = np.load(file = result_dir + "thetas_" + data + ".npy")
        # calculate the diffs
        thetas = thetas[steps::,:,:] - thetas[:-steps,:,:]
        for time in range(0, ntimes_diff):
            # benchmark admm
            nmse = bmk.NMSE(G_inf = thetas[time], G_true = gt_adj[time])
            ps = bmk.PS(G_inf = thetas[time], G_true = gt_adj[time])

            pearson_val, pval = bmk.pearson(G_inf = thetas[time], G_true = gt_adj[time])
            cosine_sim = bmk.cossim(G_inf = thetas[time], G_true = gt_adj[time])            
            score2 = score2.append({"interval": interval,
                                    "ngenes": ngenes,
                                    "nmse": nmse, 
                                    "probability of success":ps, 
                                    "pearson": pearson_val, 
                                    "cosine similarity": cosine_sim, 
                                    "time":time}, ignore_index=True)

        # hyper-parameter
        alpha = 1
        rho = "1.7"
        bandwidth = 0.1
        lamb = 0.01
        data = str(bandwidth) + "_" + str(alpha) + "_" + str(lamb) + "_" + rho
        thetas = np.load(file = result_dir + "thetas_" + data + ".npy")
        # calculate the diffs
        thetas = thetas[steps::,:,:] - thetas[:-steps,:,:]
        for time in range(0, ntimes_diff):
            # benchmark admm
            nmse = bmk.NMSE(G_inf = thetas[time], G_true = gt_adj[time])
            ps = bmk.PS(G_inf = thetas[time], G_true = gt_adj[time])

            pearson_val, pval = bmk.pearson(G_inf = thetas[time], G_true = gt_adj[time])
            cosine_sim = bmk.cossim(G_inf = thetas[time], G_true = gt_adj[time])            
            score3 = score3.append({"interval": interval,
                                    "ngenes": ngenes,
                                    "nmse": nmse, 
                                    "probability of success":ps, 
                                    "pearson": pearson_val, 
                                    "cosine similarity": cosine_sim, 
                                    "time":time}, ignore_index=True)

# save results
score1.to_csv("../results/script2_scores/score_diff_1_None.csv")
score2.to_csv("../results/script2_scores/score_diff_2_1.7.csv")
score3.to_csv("../results/script2_scores/score_diff_1_1.7.csv")
score_rand.to_csv("../results/script2_scores/score_diff_rand.csv")


#In[] Plot boxplots
# load scores
plt.rcParams["font.size"] = 20
score1 = pd.read_csv("../results/script2_scores/score_1_None.csv", index_col = 0)
score2 = pd.read_csv("../results/script2_scores/score_2_1.7.csv", index_col = 0)
score3 = pd.read_csv("../results/script2_scores/score_1_1.7.csv", index_col = 0)
score_rand = pd.read_csv("../results/script2_scores/score_rand.csv", index_col = 0)

score1_diff = pd.read_csv("../results/script2_scores/score_diff_1_None.csv", index_col = 0)
score2_diff = pd.read_csv("../results/script2_scores/score_diff_2_1.7.csv", index_col = 0)
score3_diff = pd.read_csv("../results/script2_scores/score_diff_1_1.7.csv", index_col = 0)
score_rand_diff = pd.read_csv("../results/script2_scores/score_diff_rand.csv", index_col = 0)

fig = plt.figure(figsize = (20, 7))
ax = fig.subplots(nrows = 1, ncols = 2)
sns.boxplot(data = score1, x = "ngenes", y = "nmse", hue = "interval", ax = ax[0])
sns.boxplot(data = score1_diff, x = "ngenes", y = "nmse", hue = "interval", ax = ax[1])
ax[0].set_title("detect true edges")
ax[1].set_title("detect changes")
ax[0].set_ylim([0, 0.15])
ax[1].set_ylim([0, 15])
fig.savefig("../results/script2_scores/nmse_1_None.png", bbox_inches = "tight")

fig = plt.figure(figsize = (20, 7))
ax = fig.subplots(nrows = 1, ncols = 2)
sns.boxplot(data = score2, x = "ngenes", y = "nmse", hue = "interval", ax = ax[0])
sns.boxplot(data = score2_diff, x = "ngenes", y = "nmse", hue = "interval", ax = ax[1])
ax[0].set_title("detect true edges")
ax[1].set_title("detect changes")
ax[0].set_ylim([0, 0.15])
ax[1].set_ylim([0, 15])
fig.savefig("../results/script2_scores/nmse_2_1.7.png", bbox_inches = "tight")

fig = plt.figure(figsize = (20, 7))
ax = fig.subplots(nrows = 1, ncols = 2)
sns.boxplot(data = score3, x = "ngenes", y = "nmse", hue = "interval", ax = ax[0])
sns.boxplot(data = score3_diff, x = "ngenes", y = "nmse", hue = "interval", ax = ax[1])
ax[0].set_title("detect true edges")
ax[1].set_title("detect changes")
ax[0].set_ylim([0, 0.15])
ax[1].set_ylim([0, 15])
fig.savefig("../results/script2_scores/nmse_1_1.7.png", bbox_inches = "tight")


fig = plt.figure(figsize = (20, 7))
ax = fig.subplots(nrows = 1, ncols = 2)
sns.boxplot(data = score1, x = "ngenes", y = "pearson", hue = "interval", ax = ax[0])
sns.boxplot(data = score1_diff, x = "ngenes", y = "pearson", hue = "interval", ax = ax[1])
ax[0].set_title("detect true edges")
ax[1].set_title("detect changes")
ax[0].set_ylim([0.9, 1])
ax[1].set_ylim([0.0, 1])
fig.savefig("../results/script2_scores/pearson_1_None.png", bbox_inches = "tight")

fig = plt.figure(figsize = (20, 7))
ax = fig.subplots(nrows = 1, ncols = 2)
sns.boxplot(data = score2, x = "ngenes", y = "pearson", hue = "interval", ax = ax[0])
sns.boxplot(data = score2_diff, x = "ngenes", y = "pearson", hue = "interval", ax = ax[1])
ax[0].set_title("detect true edges")
ax[1].set_title("detect changes")
ax[0].set_ylim([0.9, 1])
ax[1].set_ylim([0.0, 1])
fig.savefig("../results/script2_scores/pearson_2_1.7.png", bbox_inches = "tight")

fig = plt.figure(figsize = (20, 7))
ax = fig.subplots(nrows = 1, ncols = 2)
sns.boxplot(data = score3, x = "ngenes", y = "pearson", hue = "interval", ax = ax[0])
sns.boxplot(data = score3_diff, x = "ngenes", y = "pearson", hue = "interval", ax = ax[1])
ax[0].set_title("detect true edges")
ax[1].set_title("detect changes")
ax[0].set_ylim([0.9, 1])
ax[1].set_ylim([0.0, 1])
fig.savefig("../results/script2_scores/pearson_1_1.7.png", bbox_inches = "tight")

# %%
