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
'''
ntimes = 1000
path = "../../data/GGM_changing_mean/"
for interval in [50, 100, 200]:
    for (ngenes, ntfs) in [(20, 5), (30, 10), (50, 20), (100, 50)]:
        result_dir = "../results/GGM_changing_mean_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "/"
        # the data smapled from GGM is zero-mean
        X = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/expr.npy")
        X_genie = X.reshape(ntimes, ngenes)
        # genie_theta of the shape (ntimes, ngenes, ngenes)
        genie_theta = genie3.GENIE3(X_genie, gene_names=["gene_" + str(x) for x in range(ngenes)], regulators='all',tree_method='RF',K='sqrt',ntrees=1000,nthreads=1)
        genie_theta = np.repeat(genie_theta[None, :, :],ntimes,axis=0)
        np.save(file = result_dir + "genie_theta.npy", arr = genie_theta)

        # the data smapled from GGM is zero-mean
        X = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/expr.npy")
        X_genie = X.reshape(ntimes, ngenes)
        # genie_theta of the shape (ntimes, ngenes, ngenes)
        genie_theta = genie3.GENIE3(X_genie, gene_names=["gene_" + str(x) for x in range(ngenes)], regulators=["gene_" + str(x) for x in range(ntfs)],tree_method='RF',K='sqrt',ntrees=1000,nthreads=1)
        genie_theta = np.repeat(genie_theta[None, :, :],ntimes,axis=0)
        np.save(file = result_dir + "genie_theta_tfs.npy", arr = genie_theta)
'''

# In[] benchmark accuracy
# Note that the current bandwith is tested on the changing interval 200, ngenes 20, density 0.1
# when the interval changed, don't know how the performance will change.
ntimes = 1000
path = "../../data/GGM_changing_mean/"
score = pd.DataFrame(columns = ["interval", "ngenes", "nmse","probability of success", "pearson", "cosine similarity", "time", "model"])

for interval in [50, 100, 200]:
    for (ngenes, ntfs) in [(20, 5), (30, 10), (50, 20), (100, 50)]:
        result_dir = "../results/GGM_changing_mean_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "/"
        # the data smapled from GGM is zero-mean
        X = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/expr.npy")
        gt_adj = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/Gs.npy")
        
        for time in range(0, ntimes):
            np.random.seed(0)
            # benchmark random baseline
            thetas_rand = np.random.randn(ngenes,ngenes)
            nmse_rand = bmk.NMSE(G_inf = thetas_rand, G_true = gt_adj[time])
            ps_rand = bmk.PS(G_inf = thetas_rand, G_true = gt_adj[time])

            pearson_rand_val, pval = bmk.pearson(G_inf = thetas_rand, G_true = gt_adj[time])
            cosine_sim_rand = bmk.cossim(G_inf = thetas_rand, G_true = gt_adj[time])        
            score = score.append({"interval": interval,
                                "ngenes": ngenes,
                                "nmse": nmse_rand, 
                                "probability of success":ps_rand, 
                                "pearson": pearson_rand_val, 
                                "cosine similarity": cosine_sim_rand, 
                                "time":time,
                                "model": "random"}, ignore_index=True)

        # genie3 
        thetas = np.load(file = result_dir + "genie_theta.npy")
        for time in range(0, ntimes):
            nmse = bmk.NMSE(G_inf = thetas[time], G_true = gt_adj[time])
            ps = bmk.PS(G_inf = thetas[time], G_true = gt_adj[time])

            pearson_val, pval = bmk.pearson(G_inf = thetas[time], G_true = gt_adj[time])
            cosine_sim = bmk.cossim(G_inf = thetas[time], G_true = gt_adj[time])            
            score = score.append({"interval": interval,
                                    "ngenes": ngenes,
                                    "nmse": nmse, 
                                    "probability of success":ps, 
                                    "pearson": pearson_val, 
                                    "cosine similarity": cosine_sim, 
                                    "time":time, 
                                    "model": "genie"}, ignore_index=True)        

        thetas = np.load(file = result_dir + "genie_theta_tfs.npy")
        for time in range(0, ntimes):
            nmse = bmk.NMSE(G_inf = thetas[time], G_true = gt_adj[time])
            ps = bmk.PS(G_inf = thetas[time], G_true = gt_adj[time])

            pearson_val, pval = bmk.pearson(G_inf = thetas[time], G_true = gt_adj[time])
            cosine_sim = bmk.cossim(G_inf = thetas[time], G_true = gt_adj[time])            
            score = score.append({"interval": interval,
                                    "ngenes": ngenes,
                                    "nmse": nmse, 
                                    "probability of success":ps, 
                                    "pearson": pearson_val, 
                                    "cosine similarity": cosine_sim, 
                                    "time":time, 
                                    "model": "genie_tfs"}, ignore_index=True)        


        # benchmark admm, hyper-parameter
        alpha = 2
        rho = "1.7"
        bandwidth = 0.1
        lamb = 0.01
        data = str(bandwidth) + "_" + str(alpha) + "_" + str(lamb) + "_" + rho
        thetas = np.load(file = result_dir + "thetas_" + data + ".npy")
        
        for time in range(0, ntimes):
            nmse = bmk.NMSE(G_inf = thetas[time], G_true = gt_adj[time])
            ps = bmk.PS(G_inf = thetas[time], G_true = gt_adj[time])

            pearson_val, pval = bmk.pearson(G_inf = thetas[time], G_true = gt_adj[time])
            cosine_sim = bmk.cossim(G_inf = thetas[time], G_true = gt_adj[time])            
            score = score.append({"interval": interval,
                                    "ngenes": ngenes,
                                    "nmse": nmse, 
                                    "probability of success":ps, 
                                    "pearson": pearson_val, 
                                    "cosine similarity": cosine_sim, 
                                    "time":time, 
                                    "model": "alpha_" + str(alpha) + "_rho_" + rho}, ignore_index=True)

        # benchmark admm, hyper-parameter
        alpha = 2
        rho = "1.7"
        bandwidth = 0.1
        lamb = 0.01
        data = str(bandwidth) + "_" + str(alpha) + "_" + str(lamb) + "_" + rho + "_tfs"
        thetas = np.load(file = result_dir + "thetas_" + data + ".npy")
        
        for time in range(0, ntimes):
            nmse = bmk.NMSE(G_inf = thetas[time], G_true = gt_adj[time])
            ps = bmk.PS(G_inf = thetas[time], G_true = gt_adj[time])

            pearson_val, pval = bmk.pearson(G_inf = thetas[time], G_true = gt_adj[time])
            cosine_sim = bmk.cossim(G_inf = thetas[time], G_true = gt_adj[time])            
            score = score.append({"interval": interval,
                                    "ngenes": ngenes,
                                    "nmse": nmse, 
                                    "probability of success":ps, 
                                    "pearson": pearson_val, 
                                    "cosine similarity": cosine_sim, 
                                    "time":time, 
                                    "model": "alpha_" + str(alpha) + "_rho_" + rho + "_tf"}, ignore_index=True)

        # benchmark admm, hyper-parameter
        alpha = 1
        rho = "None"
        bandwidth = 0.1
        lamb = 0.01
        data = str(bandwidth) + "_" + str(alpha) + "_" + str(lamb) + "_" + rho
        thetas = np.load(file = result_dir + "thetas_" + data + ".npy")
        
        for time in range(0, ntimes):
            nmse = bmk.NMSE(G_inf = thetas[time], G_true = gt_adj[time])
            ps = bmk.PS(G_inf = thetas[time], G_true = gt_adj[time])

            pearson_val, pval = bmk.pearson(G_inf = thetas[time], G_true = gt_adj[time])
            cosine_sim = bmk.cossim(G_inf = thetas[time], G_true = gt_adj[time])            
            score = score.append({"interval": interval,
                                    "ngenes": ngenes,
                                    "nmse": nmse, 
                                    "probability of success":ps, 
                                    "pearson": pearson_val, 
                                    "cosine similarity": cosine_sim, 
                                    "time":time, 
                                    "model": "alpha_" + str(alpha) + "_rho_" + rho}, ignore_index=True)

        # benchmark admm, hyper-parameter
        alpha = 1
        rho = "None"
        bandwidth = 0.1
        lamb = 0.01
        data = str(bandwidth) + "_" + str(alpha) + "_" + str(lamb) + "_" + rho + "_tfs"
        thetas = np.load(file = result_dir + "thetas_" + data + ".npy")
        
        for time in range(0, ntimes):
            nmse = bmk.NMSE(G_inf = thetas[time], G_true = gt_adj[time])
            ps = bmk.PS(G_inf = thetas[time], G_true = gt_adj[time])

            pearson_val, pval = bmk.pearson(G_inf = thetas[time], G_true = gt_adj[time])
            cosine_sim = bmk.cossim(G_inf = thetas[time], G_true = gt_adj[time])            
            score = score.append({"interval": interval,
                                    "ngenes": ngenes,
                                    "nmse": nmse, 
                                    "probability of success":ps, 
                                    "pearson": pearson_val, 
                                    "cosine similarity": cosine_sim, 
                                    "time":time, 
                                    "model": "alpha_" + str(alpha) + "_rho_" + rho + "_tf"}, ignore_index=True)

# save results
score.to_csv("../results/script3_scores/score.csv")


# In[] benchmark changing accuracy
ntimes = 1000
path = "../../data/GGM_changing_mean/"
score_diff = pd.DataFrame(columns = ["interval", "ngenes", "nmse","probability of success", "pearson", "cosine similarity", "time", "model"])

for interval in [50, 100, 200]:
    for (ngenes, ntfs) in [(20, 5), (30, 10), (50, 20), (100, 50)]:
        result_dir = "../results/GGM_changing_mean_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "/"
        # the data smapled from GGM is zero-mean
        X = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/expr.npy")
        gt_adj = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/Gs.npy")

        # calculate the diffs
        # compare every 100 steps
        steps = 100
        gt_adj = gt_adj[steps::,:,:] - gt_adj[:-steps,:,:]
        ntimes_diff = gt_adj.shape[0]

        for time in range(0, ntimes_diff):
            np.random.seed(0)
            # benchmark random baseline
            thetas_rand = np.random.randn(ngenes,ngenes)
            nmse_rand = bmk.NMSE(G_inf = thetas_rand, G_true = gt_adj[time])
            ps_rand = bmk.PS(G_inf = thetas_rand, G_true = gt_adj[time])

            pearson_rand_val, pval = bmk.pearson(G_inf = thetas_rand, G_true = gt_adj[time])
            cosine_sim_rand = bmk.cossim(G_inf = thetas_rand, G_true = gt_adj[time])        
            score_diff = score_diff.append({"interval": interval,
                                "ngenes": ngenes,
                                "nmse": nmse_rand, 
                                "probability of success":ps_rand, 
                                "pearson": pearson_rand_val, 
                                "cosine similarity": cosine_sim_rand, 
                                "time":time,
                                "model": "random"}, ignore_index=True)
        
        # benchmark admm, hyper-parameter
        alpha = 2
        rho = "1.7"
        bandwidth = 0.1
        lamb = 0.01
        data = str(bandwidth) + "_" + str(alpha) + "_" + str(lamb) + "_" + rho
        thetas = np.load(file = result_dir + "thetas_" + data + ".npy")
        thetas = thetas[steps::,:,:] - thetas[:-steps,:,:]
        ntimes_diff = thetas.shape[0]
        assert gt_adj.shape[0] == thetas.shape[0]

        for time in range(0, ntimes_diff):
            nmse = bmk.NMSE(G_inf = thetas[time], G_true = gt_adj[time])
            ps = bmk.PS(G_inf = thetas[time], G_true = gt_adj[time])

            pearson_val, pval = bmk.pearson(G_inf = thetas[time], G_true = gt_adj[time])
            cosine_sim = bmk.cossim(G_inf = thetas[time], G_true = gt_adj[time])            
            score_diff = score_diff.append({"interval": interval,
                                    "ngenes": ngenes,
                                    "nmse": nmse, 
                                    "probability of success":ps, 
                                    "pearson": pearson_val, 
                                    "cosine similarity": cosine_sim, 
                                    "time":time, 
                                    "model": "alpha_" + str(alpha) + "_rho_" + rho}, ignore_index=True)

        # benchmark admm, hyper-parameter
        alpha = 2
        rho = "1.7"
        bandwidth = 0.1
        lamb = 0.01
        data = str(bandwidth) + "_" + str(alpha) + "_" + str(lamb) + "_" + rho + "_tfs"
        thetas = np.load(file = result_dir + "thetas_" + data + ".npy")
        thetas = thetas[steps::,:,:] - thetas[:-steps,:,:]
        ntimes_diff = thetas.shape[0]
        assert gt_adj.shape[0] == thetas.shape[0]

        for time in range(0, ntimes_diff):
            nmse = bmk.NMSE(G_inf = thetas[time], G_true = gt_adj[time])
            ps = bmk.PS(G_inf = thetas[time], G_true = gt_adj[time])

            pearson_val, pval = bmk.pearson(G_inf = thetas[time], G_true = gt_adj[time])
            cosine_sim = bmk.cossim(G_inf = thetas[time], G_true = gt_adj[time])            
            score_diff = score_diff.append({"interval": interval,
                                    "ngenes": ngenes,
                                    "nmse": nmse, 
                                    "probability of success":ps, 
                                    "pearson": pearson_val, 
                                    "cosine similarity": cosine_sim, 
                                    "time":time, 
                                    "model": "alpha_" + str(alpha) + "_rho_" + rho + "_tf"}, ignore_index=True)

        # benchmark admm, hyper-parameter
        alpha = 1
        rho = "None"
        bandwidth = 0.1
        lamb = 0.01
        data = str(bandwidth) + "_" + str(alpha) + "_" + str(lamb) + "_" + rho
        thetas = np.load(file = result_dir + "thetas_" + data + ".npy")
        thetas = thetas[steps::,:,:] - thetas[:-steps,:,:]
        ntimes_diff = thetas.shape[0]
        assert gt_adj.shape[0] == thetas.shape[0]
        
        for time in range(0, ntimes_diff):
            nmse = bmk.NMSE(G_inf = thetas[time], G_true = gt_adj[time])
            ps = bmk.PS(G_inf = thetas[time], G_true = gt_adj[time])

            pearson_val, pval = bmk.pearson(G_inf = thetas[time], G_true = gt_adj[time])
            cosine_sim = bmk.cossim(G_inf = thetas[time], G_true = gt_adj[time])            
            score_diff = score_diff.append({"interval": interval,
                                    "ngenes": ngenes,
                                    "nmse": nmse, 
                                    "probability of success":ps, 
                                    "pearson": pearson_val, 
                                    "cosine similarity": cosine_sim, 
                                    "time":time, 
                                    "model": "alpha_" + str(alpha) + "_rho_" + rho}, ignore_index=True)

        # benchmark admm, hyper-parameter
        alpha = 1
        rho = "None"
        bandwidth = 0.1
        lamb = 0.01
        data = str(bandwidth) + "_" + str(alpha) + "_" + str(lamb) + "_" + rho + "_tfs"
        thetas = np.load(file = result_dir + "thetas_" + data + ".npy")
        thetas = thetas[steps::,:,:] - thetas[:-steps,:,:]
        ntimes_diff = thetas.shape[0]
        assert gt_adj.shape[0] == thetas.shape[0]
        
        for time in range(0, ntimes_diff):
            nmse = bmk.NMSE(G_inf = thetas[time], G_true = gt_adj[time])
            ps = bmk.PS(G_inf = thetas[time], G_true = gt_adj[time])

            pearson_val, pval = bmk.pearson(G_inf = thetas[time], G_true = gt_adj[time])
            cosine_sim = bmk.cossim(G_inf = thetas[time], G_true = gt_adj[time])            
            score_diff = score_diff.append({"interval": interval,
                                    "ngenes": ngenes,
                                    "nmse": nmse, 
                                    "probability of success":ps, 
                                    "pearson": pearson_val, 
                                    "cosine similarity": cosine_sim, 
                                    "time":time, 
                                    "model": "alpha_" + str(alpha) + "_rho_" + rho + "_tf"}, ignore_index=True)

# save results
score_diff.to_csv("../results/script3_scores/score_diff.csv")


#In[] Plot boxplots
# load scores
plt.rcParams["font.size"] = 20
score = pd.read_csv("../results/script3_scores/score.csv", index_col = 0)
score_diff = pd.read_csv("../results/script3_scores/score_diff.csv", index_col = 0)

# test different models, if the model don't have huge difference, then we consider using one model.
fig = plt.figure(figsize = (30, 7))
ax = fig.subplots(nrows = 1, ncols = 2)
sns.boxplot(data = score, x = "ngenes", y = "nmse", hue = "model", ax = ax[0])
ax[0].legend(loc="upper left", prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.01, 1))
sns.boxplot(data = score_diff, x = "ngenes", y = "nmse", hue = "model", ax = ax[1])
ax[1].legend(loc="upper left", prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.01, 1))

ax[0].set_title("detect true edges")
ax[1].set_title("detect changes")
fig.tight_layout()
fig.savefig("../results/script3_scores/nmse.png", bbox_inches = "tight")


fig = plt.figure(figsize = (30, 7))
ax = fig.subplots(nrows = 1, ncols = 2)
sns.boxplot(data = score, x = "ngenes", y = "pearson", hue = "model", ax = ax[0])
ax[0].legend(loc="upper left", prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.01, 1))
sns.boxplot(data = score_diff, x = "ngenes", y = "pearson", hue = "model", ax = ax[1])
ax[1].legend(loc="upper left", prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.01, 1))

ax[0].set_title("detect true edges")
ax[1].set_title("detect changes")
fig.tight_layout()
fig.savefig("../results/script3_scores/pearson.png", bbox_inches = "tight")


fig = plt.figure(figsize = (30, 7))
ax = fig.subplots(nrows = 1, ncols = 2)
sns.boxplot(data = score, x = "ngenes", y = "cosine similarity", hue = "model", ax = ax[0])
ax[0].legend(loc="upper left", prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.01, 1))
sns.boxplot(data = score_diff, x = "ngenes", y = "cosine similarity", hue = "model", ax = ax[1])
ax[1].legend(loc="upper left", prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.01, 1))

ax[0].set_title("detect true edges")
ax[1].set_title("detect changes")
fig.tight_layout()
fig.savefig("../results/script3_scores/cosine.png", bbox_inches = "tight")


# %%
