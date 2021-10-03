# In[0]
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

# weighted Kernel for weighted covariance matrix and weighting the losses for different time points
def kernel_band(bandwidth, ntimes, truncate = False):
    # bandwidth decide the shape (width), no matter the length ntimes
    t = (np.arange(ntimes)/ntimes).reshape(ntimes,1)
    tdis = np.square(pdist(t))
    mdis = 0.5 * bandwidth * np.median(tdis)

    K = squareform(np.exp(-tdis/mdis))+np.identity(ntimes)

    if truncate == True:
        cutoff = mdis * 1.5
        mask = (squareform(tdis) < cutoff).astype(np.int)
        K = K * mask

    return K/np.sum(K,axis=1)[:,None]


ntimes = 3000
interval = 200
ngenes = 20
data_dir = "../../data/GGM/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/"
result_dir = "../results/GGM_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "/"

gt_adj = np.load(data_dir + "Gs.npy")
ntimes = gt_adj.shape[0]
ngenes = gt_adj.shape[1]

# transcription factor, select all the genes that controls the other genes
tf = np.array([0, 1, 2, 3, 4])

# In[] time comparison, using mean
# TODO: update score
alpha = 2
rho = str(1.7)

for bandwidth in [0.01, 0.1, 1, 10]:
    for lamb in [0.01, 0.1, 1]:
        score = pd.DataFrame(columns =["model", "nmse","probability of success", "pearson", "cosine similarity", "time"])
        score_rand = pd.DataFrame(columns =["model", "nmse","probability of success", "pearson", "cosine similarity", "time"])
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
            score_rand = score_rand.append({"nmse": nmse_rand, "probability of success":ps_rand, "pearson": pearson_rand_val, "cosine similarity": cosine_sim_rand, 
                                            "model":"random", "time":time}, ignore_index=True)
            # benchmark admm
            nmse = bmk.NMSE(G_inf = thetas[time], G_true = gt_adj[time])
            ps = bmk.PS(G_inf = thetas[time], G_true = gt_adj[time])
            pearson_val = bmk.pearson(G_inf = thetas[time], G_true = gt_adj[time])
            cosine_sim = bmk.cossim(G_inf = thetas[time], G_true = gt_adj[time])            
            score = score.append({"nmse": nmse, "probability of success":ps, "pearson": pearson_val, "cosine similarity": cosine_sim, 
                                  "model":"GADMM_"+ data, "time":time}, ignore_index=True)

        score.to_csv(result_dir + "score_" + data + ".csv")
        # save result
        score_rand.to_csv(result_dir + "score_rand.csv")



# In[] Differences
interval = 100
gt_adj = gt_adj[interval::,:,:] - gt_adj[:-interval,:,:]

ntimes = gt_adj.shape[0]
# TODO: update score_diff
alpha = 2
rho = str(1.7)

for bandwidth in [0.01, 0.1, 1, 10]:
    for lamb in [0.01, 0.1, 1]:
        score = pd.DataFrame(columns =["model", "nmse","probability of success", "pearson", "cosine similarity", "time"])
        score_rand = pd.DataFrame(columns =["model", "nmse","probability of success", "pearson", "cosine similarity", "time"])
        data = str(bandwidth) + "_" + str(alpha) + "_" + str(lamb) + "_" + rho
        print(data)
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
            score_rand = score_rand.append({"nmse": nmse_rand, "probability of success":ps_rand, "pearson": pearson_rand_val, "cosine similarity": cosine_sim_rand, 
                                            "model":"random", "time":time}, ignore_index=True)
            # benchmark admm
            nmse = bmk.NMSE(G_inf = thetas[time], G_true = gt_adj[time])
            ps = bmk.PS(G_inf = thetas[time], G_true = gt_adj[time])
            pearson_val = bmk.pearson(G_inf = thetas[time], G_true = gt_adj[time])
            cosine_sim = bmk.cossim(G_inf = thetas[time], G_true = gt_adj[time])            
            score = score.append({"nmse": nmse, "probability of success":ps, "pearson": pearson_val, "cosine similarity": cosine_sim, 
                                  "model":"GADMM_"+ data, "time":time}, ignore_index=True)

        score.to_csv(result_dir + "score_" + data + "_diff.csv")
        # save result
        score_rand.to_csv(result_dir + "score_rand_diff.csv")


# In[] time comparison, don't use mean
# TODO: update score
alpha = 2
rho = str(1.7)

for bandwidth in [0.01, 0.1, 1, 10]:
    for lamb in [0.01, 0.1, 1]:
        score = pd.DataFrame(columns =["model", "nmse","probability of success", "pearson", "cosine similarity", "time"])
        score_rand = pd.DataFrame(columns =["model", "nmse","probability of success", "pearson", "cosine similarity", "time"])
        data = str(bandwidth) + "_" + str(alpha) + "_" + str(lamb) + "_" + rho
        print(data)
        thetas = np.load(file = result_dir + "thetas_" + data + "_nomean.npy")

        for time in range(0, ntimes):
            np.random.seed(0)

            # benchmark random baseline
            thetas_rand = np.random.randn(ngenes,ngenes)
            # TODO: include testing method, check glad paper
            nmse_rand = bmk.NMSE(G_inf = thetas_rand, G_true = gt_adj[time])
            ps_rand = bmk.PS(G_inf = thetas_rand, G_true = gt_adj[time])
            pearson_rand_val = bmk.pearson(G_inf = thetas_rand, G_true = gt_adj[time])
            cosine_sim_rand = bmk.cossim(G_inf = thetas_rand, G_true = gt_adj[time])        
            score_rand = score_rand.append({"nmse": nmse_rand, "probability of success":ps_rand, "pearson": pearson_rand_val, "cosine similarity": cosine_sim_rand, 
                                            "model":"random", "time":time}, ignore_index=True)
            # benchmark admm
            nmse = bmk.NMSE(G_inf = thetas[time], G_true = gt_adj[time])
            ps = bmk.PS(G_inf = thetas[time], G_true = gt_adj[time])
            pearson_val = bmk.pearson(G_inf = thetas[time], G_true = gt_adj[time])
            cosine_sim = bmk.cossim(G_inf = thetas[time], G_true = gt_adj[time])            
            score = score.append({"nmse": nmse, "probability of success":ps, "pearson": pearson_val, "cosine similarity": cosine_sim, 
                                  "model":"GADMM_"+ data, "time":time}, ignore_index=True)

        score.to_csv(result_dir + "score_" + data + "_nomean.csv")
        # save result
        score_rand.to_csv(result_dir + "score_rand.csv")



# In[] Differences
interval = 100
gt_adj = gt_adj[interval::,:,:] - gt_adj[:-interval,:,:]

ntimes = gt_adj.shape[0]
# TODO: update score_diff
alpha = 2
rho = str(1.7)

for bandwidth in [0.01, 0.1, 1, 10]:
    for lamb in [0.01, 0.1, 1]:
        score = pd.DataFrame(columns =["model", "nmse","probability of success", "pearson", "cosine similarity", "time"])
        score_rand = pd.DataFrame(columns =["model", "nmse","probability of success", "pearson", "cosine similarity", "time"])
        data = str(bandwidth) + "_" + str(alpha) + "_" + str(lamb) + "_" + rho
        print(data)
        thetas = np.load(file = result_dir + "thetas_" + data + "_nomean.npy")

        for time in range(0, ntimes):
            np.random.seed(0)

            # benchmark random baseline
            thetas_rand = np.random.randn(ngenes,ngenes)
            # TODO: include testing method, check glad paper
            nmse_rand = bmk.NMSE(G_inf = thetas_rand, G_true = gt_adj[time])
            ps_rand = bmk.PS(G_inf = thetas_rand, G_true = gt_adj[time])
            pearson_rand_val = bmk.pearson(G_inf = thetas_rand, G_true = gt_adj[time])
            cosine_sim_rand = bmk.cossim(G_inf = thetas_rand, G_true = gt_adj[time])        
            score_rand = score_rand.append({"nmse": nmse_rand, "probability of success":ps_rand, "pearson": pearson_rand_val, "cosine similarity": cosine_sim_rand, 
                                            "model":"random", "time":time}, ignore_index=True)
            # benchmark admm
            nmse = bmk.NMSE(G_inf = thetas[time], G_true = gt_adj[time])
            ps = bmk.PS(G_inf = thetas[time], G_true = gt_adj[time])
            pearson_val = bmk.pearson(G_inf = thetas[time], G_true = gt_adj[time])
            cosine_sim = bmk.cossim(G_inf = thetas[time], G_true = gt_adj[time])            
            score = score.append({"nmse": nmse, "probability of success":ps, "pearson": pearson_val, "cosine similarity": cosine_sim, 
                                  "model":"GADMM_"+ data, "time":time}, ignore_index=True)

        score.to_csv(result_dir + "score_" + data + "_nomean_diff.csv")
        # save result
        score_rand.to_csv(result_dir + "score_rand_diff.csv")



alpha = 2
rho = str(1.7)

for bandwidth in [0.01, 0.1, 1, 10]:
    for lamb in [0.01, 0.1, 1]:
        score = pd.DataFrame(columns =["model", "nmse","probability of success", "pearson", "cosine similarity", "time"])
        score_rand = pd.DataFrame(columns =["model", "nmse","probability of success", "pearson", "cosine similarity", "time"])
        data = str(bandwidth) + "_" + str(alpha) + "_" + str(lamb) + "_" + rho
        thetas = np.load(file = result_dir + "thetas_" + data + "_tfs.npy")

        for time in range(0, ntimes):
            np.random.seed(0)

            # benchmark random baseline
            thetas_rand = np.random.randn(ngenes,ngenes)
            # TODO: include testing method, check glad paper
            nmse_rand = bmk.NMSE(G_inf = thetas_rand, G_true = gt_adj[time])
            ps_rand = bmk.PS(G_inf = thetas_rand, G_true = gt_adj[time])
            pearson_rand_val = bmk.pearson(G_inf = thetas_rand, G_true = gt_adj[time])
            cosine_sim_rand = bmk.cossim(G_inf = thetas_rand, G_true = gt_adj[time])        
            score_rand = score_rand.append({"nmse": nmse_rand, "probability of success":ps_rand, "pearson": pearson_rand_val, "cosine similarity": cosine_sim_rand, 
                                            "model":"random", "time":time}, ignore_index=True)
            # benchmark admm
            nmse = bmk.NMSE(G_inf = thetas[time], G_true = gt_adj[time])
            ps = bmk.PS(G_inf = thetas[time], G_true = gt_adj[time])
            pearson_val = bmk.pearson(G_inf = thetas[time], G_true = gt_adj[time])
            cosine_sim = bmk.cossim(G_inf = thetas[time], G_true = gt_adj[time])            
            score = score.append({"nmse": nmse, "probability of success":ps, "pearson": pearson_val, "cosine similarity": cosine_sim, 
                                  "model":"GADMM_"+ data, "time":time}, ignore_index=True)

        score.to_csv(result_dir + "score_" + data + "_tfs.csv")
        # save result
        score_rand.to_csv(result_dir + "score_rand.csv")



# In[] Differences
interval = 100
gt_adj = gt_adj[interval::,:,:] - gt_adj[:-interval,:,:]

ntimes = gt_adj.shape[0]
# TODO: update score_diff
alpha = 2
rho = str(1.7)

for bandwidth in [0.01, 0.1, 1, 10]:
    for lamb in [0.01, 0.1, 1]:
        score = pd.DataFrame(columns =["model", "nmse","probability of success", "pearson", "cosine similarity", "time"])
        score_rand = pd.DataFrame(columns =["model", "nmse","probability of success", "pearson", "cosine similarity", "time"])
        data = str(bandwidth) + "_" + str(alpha) + "_" + str(lamb) + "_" + rho
        print(data)
        thetas = np.load(file = result_dir + "thetas_" + data + "_tfs.npy")

        for time in range(0, ntimes):
            np.random.seed(0)

            # benchmark random baseline
            thetas_rand = np.random.randn(ngenes,ngenes)
            # TODO: include testing method, check glad paper
            nmse_rand = bmk.NMSE(G_inf = thetas_rand, G_true = gt_adj[time])
            ps_rand = bmk.PS(G_inf = thetas_rand, G_true = gt_adj[time])
            pearson_rand_val = bmk.pearson(G_inf = thetas_rand, G_true = gt_adj[time])
            cosine_sim_rand = bmk.cossim(G_inf = thetas_rand, G_true = gt_adj[time])        
            score_rand = score_rand.append({"nmse": nmse_rand, "probability of success":ps_rand, "pearson": pearson_rand_val, "cosine similarity": cosine_sim_rand, 
                                            "model":"random", "time":time}, ignore_index=True)
            # benchmark admm
            nmse = bmk.NMSE(G_inf = thetas[time], G_true = gt_adj[time])
            ps = bmk.PS(G_inf = thetas[time], G_true = gt_adj[time])
            pearson_val = bmk.pearson(G_inf = thetas[time], G_true = gt_adj[time])
            cosine_sim = bmk.cossim(G_inf = thetas[time], G_true = gt_adj[time])            
            score = score.append({"nmse": nmse, "probability of success":ps, "pearson": pearson_val, "cosine similarity": cosine_sim, 
                                  "model":"GADMM_"+ data, "time":time}, ignore_index=True)

        score.to_csv(result_dir + "score_" + data + "_tfs_diff.csv")
        # save result
        score_rand.to_csv(result_dir + "score_rand_diff.csv")


# In[] time comparison, don't use mean
# TODO: update score
alpha = 2
rho = str(1.7)

for bandwidth in [0.01, 0.1, 1, 10]:
    for lamb in [0.01, 0.1, 1]:
        score = pd.DataFrame(columns =["model", "nmse","probability of success", "pearson", "cosine similarity", "time"])
        score_rand = pd.DataFrame(columns =["model", "nmse","probability of success", "pearson", "cosine similarity", "time"])
        data = str(bandwidth) + "_" + str(alpha) + "_" + str(lamb) + "_" + rho
        print(data)
        thetas = np.load(file = result_dir + "thetas_" + data + "_tfs_nomean.npy")

        for time in range(0, ntimes):
            np.random.seed(0)

            # benchmark random baseline
            thetas_rand = np.random.randn(ngenes,ngenes)
            # TODO: include testing method, check glad paper
            nmse_rand = bmk.NMSE(G_inf = thetas_rand, G_true = gt_adj[time])
            ps_rand = bmk.PS(G_inf = thetas_rand, G_true = gt_adj[time])
            pearson_rand_val = bmk.pearson(G_inf = thetas_rand, G_true = gt_adj[time])
            cosine_sim_rand = bmk.cossim(G_inf = thetas_rand, G_true = gt_adj[time])        
            score_rand = score_rand.append({"nmse": nmse_rand, "probability of success":ps_rand, "pearson": pearson_rand_val, "cosine similarity": cosine_sim_rand, 
                                            "model":"random", "time":time}, ignore_index=True)
            # benchmark admm
            nmse = bmk.NMSE(G_inf = thetas[time], G_true = gt_adj[time])
            ps = bmk.PS(G_inf = thetas[time], G_true = gt_adj[time])
            pearson_val = bmk.pearson(G_inf = thetas[time], G_true = gt_adj[time])
            cosine_sim = bmk.cossim(G_inf = thetas[time], G_true = gt_adj[time])            
            score = score.append({"nmse": nmse, "probability of success":ps, "pearson": pearson_val, "cosine similarity": cosine_sim, 
                                  "model":"GADMM_"+ data, "time":time}, ignore_index=True)

        score.to_csv(result_dir + "score_" + data + "_tfs_nomean.csv")
        # save result
        score_rand.to_csv(result_dir + "score_rand.csv")



# In[] Differences
interval = 100
gt_adj = gt_adj[interval::,:,:] - gt_adj[:-interval,:,:]

ntimes = gt_adj.shape[0]
# TODO: update score_diff
alpha = 2
rho = str(1.7)

for bandwidth in [0.01, 0.1, 1, 10]:
    for lamb in [0.01, 0.1, 1]:
        score = pd.DataFrame(columns =["model", "nmse","probability of success", "pearson", "cosine similarity", "time"])
        score_rand = pd.DataFrame(columns =["model", "nmse","probability of success", "pearson", "cosine similarity", "time"])

        data = str(bandwidth) + "_" + str(alpha) + "_" + str(lamb) + "_" + rho
        print(data)
        thetas = np.load(file = result_dir + "thetas_" + data + "_tfs_nomean.npy")

        for time in range(0, ntimes):
            np.random.seed(0)

            # benchmark random baseline
            thetas_rand = np.random.randn(ngenes,ngenes)
            # TODO: include testing method, check glad paper
            nmse_rand = bmk.NMSE(G_inf = thetas_rand, G_true = gt_adj[time])
            ps_rand = bmk.PS(G_inf = thetas_rand, G_true = gt_adj[time])
            pearson_rand_val = bmk.pearson(G_inf = thetas_rand, G_true = gt_adj[time])
            cosine_sim_rand = bmk.cossim(G_inf = thetas_rand, G_true = gt_adj[time])        
            score_rand = score_rand.append({"nmse": nmse_rand, "probability of success":ps_rand, "pearson": pearson_rand_val, "cosine similarity": cosine_sim_rand, 
                                            "model":"random", "time":time}, ignore_index=True)
            # benchmark admm
            nmse = bmk.NMSE(G_inf = thetas[time], G_true = gt_adj[time])
            ps = bmk.PS(G_inf = thetas[time], G_true = gt_adj[time])
            pearson_val = bmk.pearson(G_inf = thetas[time], G_true = gt_adj[time])
            cosine_sim = bmk.cossim(G_inf = thetas[time], G_true = gt_adj[time])            
            score = score.append({"nmse": nmse, "probability of success":ps, "pearson": pearson_val, "cosine similarity": cosine_sim, 
                                  "model":"GADMM_"+ data, "time":time}, ignore_index=True)

        score.to_csv(result_dir + "score_" + data + "_tfs_nomean_diff.csv")
        # save result
        score_rand.to_csv(result_dir + "score_rand_diff.csv")
# %%
