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

from umap import UMAP

plt.rcParams["font.size"] = 16

# In[] benchmark accuracy
print("------------------------------------------------------------------")
print("benchmark accuracy")
print("------------------------------------------------------------------")
'''
ntimes = 1000
nsample = 1
path = "../../data/GGM_bifurcate/"

score_all = pd.DataFrame(columns = ["interval", "ngenes", "nmse","kendall-tau", "pearson", "spearman", "cosine similarity", "time", "model", "bandwidth", "truncate_param", "lambda"])

umap_op = UMAP(n_components = 2, min_dist = 0.8, n_neighbors = 30, random_state = 0)
for interval in [50, 100, 200]:
    for (ngenes, ntfs) in [(20, 5), (30, 10), (50, 20), (100, 50)]:
        score = pd.DataFrame(columns = ["interval", "ngenes", "nmse","kendall-tau", "pearson", "spearman", "cosine similarity", "time", "model", "bandwidth", "truncate_param", "lambda"])
        result_dir = "../results_GGM/bifur_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "/"
        
        # the data smapled from GGM is zero-mean
        X = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/expr.npy")
        gt_adj = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/Gs.npy")
        sim_time = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/sim_time.npy")
        
        for time in range(0, ntimes * nsample):
            np.random.seed(0)
            # benchmark random baseline
            thetas_rand = np.random.randn(ngenes,ngenes)
            nmse = bmk.NMSE(G_inf = thetas_rand, G_true = gt_adj[time])
            pearson_val, _ = bmk.pearson(G_inf = thetas_rand, G_true = gt_adj[time])
            kt, _ = bmk.kendalltau(G_inf = thetas_rand, G_true = gt_adj[time])
            spearman_val, _ = bmk.spearman(G_inf = thetas_rand, G_true = gt_adj[time])
            cosine_sim = bmk.cossim(G_inf = thetas_rand, G_true = gt_adj[time])        
            score = score.append({"interval": interval,
                                "ngenes": ngenes,
                                "nmse": nmse, 
                                "pearson": pearson_val, 
                                "kendall-tau": kt,
                                "spearman": spearman_val,
                                "cosine similarity": cosine_sim, 
                                "time":time,
                                "model": "RANDOM",
                                "bandwidth": 0,
                                "truncate_param":0,
                                "lambda":0
                                }, ignore_index=True)

        # genie3 
        thetas = np.load(file = result_dir + "theta_genie.npy")
        for time in range(0, ntimes * nsample):
            nmse = bmk.NMSE(G_inf = thetas[time], G_true = gt_adj[time])
            pearson_val, pval = bmk.pearson(G_inf = thetas[time], G_true = gt_adj[time])
            kt, _ = bmk.kendalltau(G_inf = thetas[time], G_true = gt_adj[time])
            spearman_val, _ = bmk.spearman(G_inf = thetas[time], G_true = gt_adj[time])
            cosine_sim = bmk.cossim(G_inf = thetas[time], G_true = gt_adj[time])   
            score = score.append({"interval": interval,
                                "ngenes": ngenes,
                                "nmse": nmse, 
                                "pearson": pearson_val, 
                                "kendall-tau": kt,
                                "spearman": spearman_val,
                                "cosine similarity": cosine_sim, 
                                "time":time,
                                "model": "GENIE3",
                                "bandwidth": 0,
                                "truncate_param":0,
                                "lambda":0}, ignore_index=True)    

        # genie3 with tf 
        thetas = np.load(file = result_dir + "theta_genie_tf.npy")
        for time in range(0, ntimes * nsample):
            nmse = bmk.NMSE(G_inf = thetas[time], G_true = gt_adj[time])
            pearson_val, pval = bmk.pearson(G_inf = thetas[time], G_true = gt_adj[time])
            kt, _ = bmk.kendalltau(G_inf = thetas[time], G_true = gt_adj[time])
            spearman_val, _ = bmk.spearman(G_inf = thetas[time], G_true = gt_adj[time])
            cosine_sim = bmk.cossim(G_inf = thetas[time], G_true = gt_adj[time])   
            score = score.append({"interval": interval,
                                "ngenes": ngenes,
                                "nmse": nmse, 
                                "pearson": pearson_val, 
                                "kendall-tau": kt,
                                "spearman": spearman_val,
                                "cosine similarity": cosine_sim, 
                                "time":time,
                                "model": "GENIE3-TF",                                
                                "bandwidth": 0,
                                "truncate_param":0,
                                "lambda":0}, ignore_index=True)    

        # scode
        try:
            thetas = np.load(file = result_dir + "theta_scode_dpt.npy")
            for time in range(0, ntimes * nsample):
                nmse = bmk.NMSE(G_inf = thetas[time], G_true = gt_adj[time])
                pearson_val, pval = bmk.pearson(G_inf = thetas[time], G_true = gt_adj[time])
                kt, _ = bmk.kendalltau(G_inf = thetas[time], G_true = gt_adj[time])
                spearman_val, _ = bmk.spearman(G_inf = thetas[time], G_true = gt_adj[time])
                cosine_sim = bmk.cossim(G_inf = thetas[time], G_true = gt_adj[time])   
                score = score.append({"interval": interval,
                                    "ngenes": ngenes,
                                    "nmse": nmse, 
                                    "pearson": pearson_val, 
                                    "kendall-tau": kt,
                                    "spearman": spearman_val,
                                    "cosine similarity": cosine_sim, 
                                    "time":time,
                                    "model": "SCODE (dpt)",
                                    "bandwidth": 0,
                                    "truncate_param":0,
                                    "lambda":0}, ignore_index=True)
        except:
            pass    

        
        thetas = np.load(file = result_dir + "theta_scode_truet.npy")
        for time in range(0, ntimes * nsample):
            nmse = bmk.NMSE(G_inf = thetas[time], G_true = gt_adj[time])
            pearson_val, pval = bmk.pearson(G_inf = thetas[time], G_true = gt_adj[time])
            kt, _ = bmk.kendalltau(G_inf = thetas[time], G_true = gt_adj[time])
            spearman_val, _ = bmk.spearman(G_inf = thetas[time], G_true = gt_adj[time])
            cosine_sim = bmk.cossim(G_inf = thetas[time], G_true = gt_adj[time])   
            score = score.append({"interval": interval,
                                "ngenes": ngenes,
                                "nmse": nmse, 
                                "pearson": pearson_val, 
                                "kendall-tau": kt,
                                "spearman": spearman_val,
                                "cosine similarity": cosine_sim, 
                                "time":time,
                                "model": "SCODE (true time)",
                                "bandwidth": 0,
                                "truncate_param":0,
                                "lambda":0}, ignore_index=True)  

        print("Not using TF information")
        # admm, hyper-parameter
        for bandwidth in [0.01, 0.1, 1]:
            for truncate_param in [0.1, 1, 5]:
                for lamb in [0.01, 0.1]:
                    data = str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_0"
                    thetas = np.load(file = result_dir + "thetas_" + data + ".npy")

                    mean_nmse = 0
                    mean_pearson = 0
                    mean_kt = 0
                    mean_spearman = 0
                    mean_cosine = 0

                    for time in range(0, ntimes * nsample):
                        nmse = bmk.NMSE(G_inf = thetas[time], G_true = gt_adj[time])
                        pearson_val, pval = bmk.pearson(G_inf = thetas[time], G_true = gt_adj[time])
                        kt, _ = bmk.kendalltau(G_inf = thetas[time], G_true = gt_adj[time])
                        spearman_val, _ = bmk.spearman(G_inf = thetas[time], G_true = gt_adj[time])
                        cosine_sim = bmk.cossim(G_inf = thetas[time], G_true = gt_adj[time])   
                        score = score.append({"interval": interval,
                                            "ngenes": ngenes,
                                            "nmse": nmse, 
                                            "pearson": pearson_val, 
                                            "kendall-tau": kt,
                                            "spearman": spearman_val,
                                            "cosine similarity": cosine_sim, 
                                            "time":time,
                                            "model": "Dyn-GRN",                                
                                            "bandwidth": bandwidth,
                                            "truncate_param":truncate_param,
                                            "lambda":lamb}, ignore_index=True)    

                        mean_nmse += nmse
                        mean_pearson += pearson_val
                        mean_kt += kt
                        mean_spearman += spearman_val
                        mean_cosine += cosine_sim
                    mean_nmse = mean_nmse/(ntimes * nsample)
                    mean_pearson = mean_pearson/(ntimes * nsample)
                    mean_kt = mean_kt/(ntimes * nsample)
                    mean_spearman = mean_spearman/(ntimes * nsample)
                    mean_cosine = mean_cosine/(ntimes * nsample)
                    print("\tHyper-parameter: bandwidth = {:.2f}, truncate_param = {:.2f}, lambda = {:.2f}".format(bandwidth, truncate_param, lamb)) 
                    print("\tmean nmse: {:.4f}, mean pearson: {:.4f}, mean kt: {:.4f}, mean spearman: {:.4f}, mean cosine: {:.4f}".format(mean_nmse, mean_pearson, mean_kt, mean_spearman, mean_cosine))
                    print()

                    fig = plt.figure(figsize = (10,7))
                    X_umap = umap_op.fit_transform(thetas.reshape(ntimes * nsample, -1))
                    ax = fig.add_subplot()
                    ax.scatter(X_umap[:,0], X_umap[:,1], c = sim_time, s = 5)
                    fig.savefig(result_dir + "infer_G_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_0.png", bbox_inches = "tight")

        print()
        print("Using TF information")
        # admm with tf, hyper-parameter
        for bandwidth in [0.01, 0.1, 1]:
            for truncate_param in [0.1, 1, 5]:
                for lamb in [0.01, 0.1]:
                    data = str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_1"
                    thetas = np.load(file = result_dir + "thetas_" + data + ".npy")

                    mean_nmse = 0
                    mean_pearson = 0
                    mean_kt = 0
                    mean_spearman = 0
                    mean_cosine = 0

                    for time in range(0, ntimes * nsample):
                        nmse = bmk.NMSE(G_inf = thetas[time], G_true = gt_adj[time])
                        pearson_val, pval = bmk.pearson(G_inf = thetas[time], G_true = gt_adj[time])
                        kt, _ = bmk.kendalltau(G_inf = thetas[time], G_true = gt_adj[time])
                        spearman_val, _ = bmk.spearman(G_inf = thetas[time], G_true = gt_adj[time])
                        cosine_sim = bmk.cossim(G_inf = thetas[time], G_true = gt_adj[time])   
                        score = score.append({"interval": interval,
                                            "ngenes": ngenes,
                                            "nmse": nmse, 
                                            "pearson": pearson_val, 
                                            "kendall-tau": kt,
                                            "spearman": spearman_val,
                                            "cosine similarity": cosine_sim, 
                                            "time":time,
                                            "model": "Dyn-GRN-TF",                                
                                            "bandwidth": bandwidth,
                                            "truncate_param":truncate_param,
                                            "lambda":lamb}, ignore_index=True)

                        mean_nmse += nmse
                        mean_pearson += pearson_val
                        mean_kt += kt
                        mean_spearman += spearman_val
                        mean_cosine += cosine_sim
                    mean_nmse = mean_nmse/(ntimes * nsample)
                    mean_pearson = mean_pearson/(ntimes * nsample)
                    mean_kt = mean_kt/(ntimes * nsample)
                    mean_spearman = mean_spearman/(ntimes * nsample)
                    mean_cosine = mean_cosine/(ntimes * nsample)
                    print("\tHyper-parameter: bandwidth = {:.2f}, truncate_param = {:.2f}, lambda = {:.2f}".format(bandwidth, truncate_param, lamb)) 
                    print("\tmean nmse: {:.4f}, mean pearson: {:.4f}, mean kt: {:.4f}, mean spearman: {:.4f}, mean cosine: {:.4f}".format(mean_nmse, mean_pearson, mean_kt, mean_spearman, mean_cosine))
                    print()   

                    fig = plt.figure(figsize = (10,7))
                    X_umap = umap_op.fit_transform(thetas.reshape(ntimes * nsample, -1))
                    ax = fig.add_subplot()
                    ax.scatter(X_umap[:,0], X_umap[:,1], c = sim_time, s = 5)
                    fig.savefig(result_dir + "infer_G_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_1.png", bbox_inches = "tight")

        # save results
        score.to_csv(result_dir + "score.csv")
        score_all = pd.concat([score_all, score], axis = 0)

score_all.to_csv("../results_GGM/score_bifur_all.csv")
'''
# In[] 
print("------------------------------------------------------------------")
print("benchmark differences")
print("------------------------------------------------------------------")
'''
ntimes = 1000
nsample = 1

score_all = pd.DataFrame(columns = ["interval", "ngenes", "nmse","kendall-tau", "pearson", "spearman", "cosine similarity", "time", "model", "bandwidth", "truncate_param", "lambda"])

for interval in [50, 100, 200]:
    for (ngenes, ntfs) in [(20, 5), (30, 10), (50, 20), (100, 50)]:
        print("ntimes: " + str(ntimes) + ", interval: " + str(interval) + ", ngenes: " + str(ngenes))

        score = pd.DataFrame(columns = ["interval", "ngenes", "nmse","kendall-tau", "pearson", "spearman", "cosine similarity", "time", "model", "bandwidth", "truncate_param", "lambda"])
        result_dir = "../results_GGM/bifur_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "/"
        
        # the data smapled from GGM is zero-mean
        X = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/expr.npy")
        gt_adj = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/Gs.npy")
        gt_adj = gt_adj[interval::,:,:] - gt_adj[:-interval,:,:]
        ntimes_diff = gt_adj.shape[0]


        for time in range(0, ntimes_diff):
            np.random.seed(0)
            # random method infer exactly the same graph
            thetas_rand = np.zeros((ngenes,ngenes))
            nmse = bmk.NMSE(G_inf = thetas_rand, G_true = gt_adj[time])
            pearson_val, _ = bmk.pearson(G_inf = thetas_rand, G_true = gt_adj[time])
            kt, _ = bmk.kendalltau(G_inf = thetas_rand, G_true = gt_adj[time])
            spearman_val, _ = bmk.spearman(G_inf = thetas_rand, G_true = gt_adj[time])
            cosine_sim = bmk.cossim(G_inf = thetas_rand, G_true = gt_adj[time])        
            score = score.append({"interval": interval,
                                "ngenes": ngenes,
                                "nmse": nmse, 
                                "pearson": pearson_val, 
                                "kendall-tau": kt,
                                "spearman": spearman_val,
                                "cosine similarity": cosine_sim, 
                                "time":time,
                                "model": "RANDOM",
                                "bandwidth": 0,
                                "truncate_param":0,
                                "lambda":0
                                }, ignore_index=True)
            


        # genie3 
        thetas = np.load(file = result_dir + "theta_genie.npy")
        thetas = thetas[interval::,:,:] - thetas[:-interval,:,:]
        for time in range(0, ntimes_diff):
            nmse = bmk.NMSE(G_inf = thetas[time], G_true = gt_adj[time])
            pearson_val, pval = bmk.pearson(G_inf = thetas[time], G_true = gt_adj[time])
            kt, _ = bmk.kendalltau(G_inf = thetas[time], G_true = gt_adj[time])
            spearman_val, _ = bmk.spearman(G_inf = thetas[time], G_true = gt_adj[time])
            cosine_sim = bmk.cossim(G_inf = thetas[time], G_true = gt_adj[time])   
            score = score.append({"interval": interval,
                                "ngenes": ngenes,
                                "nmse": nmse, 
                                "pearson": pearson_val, 
                                "kendall-tau": kt,
                                "spearman": spearman_val,
                                "cosine similarity": cosine_sim, 
                                "time":time,
                                "model": "GENIE3",
                                "bandwidth": 0,
                                "truncate_param":0,
                                "lambda":0}, ignore_index=True)    

        # genie3 with tf 
        thetas = np.load(file = result_dir + "theta_genie_tf.npy")
        thetas = thetas[interval::,:,:] - thetas[:-interval,:,:]
        for time in range(0, ntimes_diff):
            nmse = bmk.NMSE(G_inf = thetas[time], G_true = gt_adj[time])
            pearson_val, pval = bmk.pearson(G_inf = thetas[time], G_true = gt_adj[time])
            kt, _ = bmk.kendalltau(G_inf = thetas[time], G_true = gt_adj[time])
            spearman_val, _ = bmk.spearman(G_inf = thetas[time], G_true = gt_adj[time])
            cosine_sim = bmk.cossim(G_inf = thetas[time], G_true = gt_adj[time])   
            score = score.append({"interval": interval,
                                "ngenes": ngenes,
                                "nmse": nmse, 
                                "pearson": pearson_val, 
                                "kendall-tau": kt,
                                "spearman": spearman_val,
                                "cosine similarity": cosine_sim, 
                                "time":time,
                                "model": "GENIE3-TF",                                
                                "bandwidth": 0,
                                "truncate_param":0,
                                "lambda":0}, ignore_index=True)    

        # scode diffusion pseudotime
        try:
            thetas = np.load(file = result_dir + "theta_scode_dpt.npy")
            thetas = thetas[interval::,:,:] - thetas[:-interval,:,:]
            for time in range(0, ntimes_diff):
                nmse = bmk.NMSE(G_inf = thetas[time], G_true = gt_adj[time])
                pearson_val, pval = bmk.pearson(G_inf = thetas[time], G_true = gt_adj[time])
                kt, _ = bmk.kendalltau(G_inf = thetas[time], G_true = gt_adj[time])
                spearman_val, _ = bmk.spearman(G_inf = thetas[time], G_true = gt_adj[time])
                cosine_sim = bmk.cossim(G_inf = thetas[time], G_true = gt_adj[time])   
                score = score.append({"interval": interval,
                                    "ngenes": ngenes,
                                    "nmse": nmse, 
                                    "pearson": pearson_val, 
                                    "kendall-tau": kt,
                                    "spearman": spearman_val,
                                    "cosine similarity": cosine_sim, 
                                    "time":time,
                                    "model": "SCODE (dpt)",
                                    "bandwidth": 0,
                                    "truncate_param":0,
                                    "lambda":0}, ignore_index=True)
        except:
            pass    

        # scode true time
        thetas = np.load(file = result_dir + "theta_scode_truet.npy")
        thetas = thetas[interval::,:,:] - thetas[:-interval,:,:]
        for time in range(0, ntimes_diff):
            nmse = bmk.NMSE(G_inf = thetas[time], G_true = gt_adj[time])
            pearson_val, pval = bmk.pearson(G_inf = thetas[time], G_true = gt_adj[time])
            kt, _ = bmk.kendalltau(G_inf = thetas[time], G_true = gt_adj[time])
            spearman_val, _ = bmk.spearman(G_inf = thetas[time], G_true = gt_adj[time])
            cosine_sim = bmk.cossim(G_inf = thetas[time], G_true = gt_adj[time])   
            score = score.append({"interval": interval,
                                "ngenes": ngenes,
                                "nmse": nmse, 
                                "pearson": pearson_val, 
                                "kendall-tau": kt,
                                "spearman": spearman_val,
                                "cosine similarity": cosine_sim, 
                                "time":time,
                                "model": "SCODE (true time)",
                                "bandwidth": 0,
                                "truncate_param":0,
                                "lambda":0}, ignore_index=True)    


        print("Not using TF information")
        # admm, hyper-parameter
        for bandwidth in [0.01, 0.1, 1]:
            for truncate_param in [0.1, 1, 5]:
                for lamb in [0.01, 0.1]:
                    data = str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_0"
                    thetas = np.load(file = result_dir + "thetas_" + data + ".npy")
                    thetas = thetas[interval::,:,:] - thetas[:-interval,:,:]

                    mean_nmse = 0
                    mean_pearson = 0
                    mean_kt = 0
                    mean_spearman = 0
                    mean_cosine = 0
                    
                    for time in range(0, ntimes_diff):
                        nmse = bmk.NMSE(G_inf = thetas[time], G_true = gt_adj[time])
                        pearson_val, pval = bmk.pearson(G_inf = thetas[time], G_true = gt_adj[time])
                        kt, _ = bmk.kendalltau(G_inf = thetas[time], G_true = gt_adj[time])
                        spearman_val, _ = bmk.spearman(G_inf = thetas[time], G_true = gt_adj[time])
                        cosine_sim = bmk.cossim(G_inf = thetas[time], G_true = gt_adj[time])   
                        score = score.append({"interval": interval,
                                            "ngenes": ngenes,
                                            "nmse": nmse, 
                                            "pearson": pearson_val, 
                                            "kendall-tau": kt,
                                            "spearman": spearman_val,
                                            "cosine similarity": cosine_sim, 
                                            "time":time,
                                            "model": "Dyn-GRN",                                
                                            "bandwidth": bandwidth,
                                            "truncate_param":truncate_param,
                                            "lambda":lamb}, ignore_index=True)    

                        mean_nmse += nmse
                        mean_pearson += pearson_val
                        mean_kt += kt
                        mean_spearman += spearman_val
                        mean_cosine += cosine_sim
                    mean_nmse = mean_nmse/ntimes
                    mean_pearson = mean_pearson/ntimes
                    mean_kt = mean_kt/ntimes
                    mean_spearman = mean_spearman/ntimes
                    mean_cosine = mean_cosine/ntimes
                    print("\tHyper-parameter: bandwidth = {:.2f}, truncate_param = {:.2f}, lambda = {:.2f}".format(bandwidth, truncate_param, lamb)) 
                    print("\tmean nmse: {:.4f}, mean pearson: {:.4f}, mean kt: {:.4f}, mean spearman: {:.4f}, mean cosine: {:.4f}".format(mean_nmse, mean_pearson, mean_kt, mean_spearman, mean_cosine))
                    print()   


        print()
        print("Using TF information")
        # admm with tf, hyper-parameter
        for bandwidth in [0.01, 0.1, 1]:
            for truncate_param in [0.1, 1, 5]:
                for lamb in [0.01, 0.1]:
                    data = str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_1"
                    thetas = np.load(file = result_dir + "thetas_" + data + ".npy")
                    thetas = thetas[interval::,:,:] - thetas[:-interval,:,:]

                    mean_nmse = 0
                    mean_pearson = 0
                    mean_kt = 0
                    mean_spearman = 0
                    mean_cosine = 0

                    for time in range(0, ntimes_diff):
                        nmse = bmk.NMSE(G_inf = thetas[time], G_true = gt_adj[time])
                        pearson_val, pval = bmk.pearson(G_inf = thetas[time], G_true = gt_adj[time])
                        kt, _ = bmk.kendalltau(G_inf = thetas[time], G_true = gt_adj[time])
                        spearman_val, _ = bmk.spearman(G_inf = thetas[time], G_true = gt_adj[time])
                        cosine_sim = bmk.cossim(G_inf = thetas[time], G_true = gt_adj[time])   
                        score = score.append({"interval": interval,
                                            "ngenes": ngenes,
                                            "nmse": nmse, 
                                            "pearson": pearson_val, 
                                            "kendall-tau": kt,
                                            "spearman": spearman_val,
                                            "cosine similarity": cosine_sim, 
                                            "time":time,
                                            "model": "Dyn-GRN-TF",                                
                                            "bandwidth": bandwidth,
                                            "truncate_param":truncate_param,
                                            "lambda":lamb}, ignore_index=True)    

                        mean_nmse += nmse
                        mean_pearson += pearson_val
                        mean_kt += kt
                        mean_spearman += spearman_val
                        mean_cosine += cosine_sim
                    mean_nmse = mean_nmse/ntimes
                    mean_pearson = mean_pearson/ntimes
                    mean_kt = mean_kt/ntimes
                    mean_spearman = mean_spearman/ntimes
                    mean_cosine = mean_cosine/ntimes
                    print("\tHyper-parameter: bandwidth = {:.2f}, truncate_param = {:.2f}, lambda = {:.2f}".format(bandwidth, truncate_param, lamb)) 
                    print("\tmean nmse: {:.4f}, mean pearson: {:.4f}, mean kt: {:.4f}, mean spearman: {:.4f}, mean cosine: {:.4f}".format(mean_nmse, mean_pearson, mean_kt, mean_spearman, mean_cosine))
                    print()   

        # save results
        score.to_csv(result_dir + "score_diff.csv")
        score_all = pd.concat([score_all, score], axis = 0)

score_all.to_csv("../results_GGM/score_bifur_diff_all.csv")
'''
# In[] summarize the mean result in csv file
ntimes = 1000
nsample = 1
for interval in [50, 100, 200]:
    for (ngenes, ntfs) in [(20, 5), (30, 10), (50, 20), (100, 50)]:
        result_dir = "../results_GGM/bifur_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "/"
        score = pd.read_csv(result_dir + "score.csv", index_col = 0)
        mean_score = score.groupby(by = ["model", "bandwidth", "truncate_param", "lambda"]).mean()
        mean_score = mean_score.drop(["time"], axis = 1)
        mean_score.to_csv(result_dir + "mean_score.csv")
        display(mean_score)

for interval in [50, 100, 200]:
    for (ngenes, ntfs) in [(20, 5), (30, 10), (50, 20), (100, 50)]:
        result_dir = "../results_GGM/bifur_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "/"
        score = pd.read_csv(result_dir + "score_diff.csv", index_col = 0)
        mean_score = score.groupby(by = ["model", "bandwidth", "truncate_param", "lambda"]).mean()
        mean_score = mean_score.drop(["time"], axis = 1)
        mean_score.to_csv(result_dir + "mean_score_diff.csv")
        display(mean_score)

#In[] Plot boxplots, rom above, we are able to find the setting with the best performance for each dataset already

ntimes = 1000

# ----------------------------------- without TF information ---------------------------------#
# How the bandwidth and truncate parameter is affected by the interval
for interval in [50, 100, 200]:
    score_interval = pd.DataFrame(columns = ["interval", "ngenes", "nmse","kendall-tau", "pearson", "spearman", "cosine similarity", "time", "model", "bandwidth", "truncate_param", "lambda"])
    score_interval_diff = pd.DataFrame(columns = ["interval", "ngenes", "nmse","kendall-tau", "pearson", "spearman", "cosine similarity", "time", "model", "bandwidth", "truncate_param", "lambda"])


    for (ngenes, ntfs) in [(20, 5), (30, 10), (50, 20), (100, 50)]:
        result_dir = "../results_GGM/bifur_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "/"

        score = pd.read_csv(result_dir + "score.csv", index_col = 0)
        score_diff = pd.read_csv(result_dir + "score_diff.csv", index_col = 0)

        # score for admm
        score = score[score["model"] == "Dyn-GRN"]
        score_diff = score_diff[score_diff["model"] == "Dyn-GRN"]
        score_interval = pd.concat([score_interval, score], axis = 0)
        score_interval_diff = pd.concat([score_interval_diff, score_diff], axis = 0)

    # Plot including lambda
    fig, big_axes = plt.subplots( figsize=(20.0, 10.0) , nrows=2, ncols=1, sharey=True) 
    lambdas = [0.01, 0.1]
    for row, big_ax in enumerate(big_axes, start=1):
        big_ax.set_title("Lambda {:.2f}".format(lambdas[row - 1]), fontsize=20)

        # Turn off axis lines and ticks of the big subplot 
        # obs alpha is 0 in RGBA string!
        big_ax.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
        # removes the white frame
        big_ax._frameon = False

    for i in range(1,7):
        ax = fig.add_subplot(2,3,i)
        if i%3 == 1:
            sns.boxplot(data = score_interval[score_interval["lambda"] == lambdas[(i-1)//3]], x = "bandwidth", y = "pearson", hue = "truncate_param", ax = ax)
        elif i%3 == 2:
            sns.boxplot(data = score_interval[score_interval["lambda"] == lambdas[(i-1)//3]], x = "bandwidth", y = "spearman", hue = "truncate_param", ax = ax)
        else:
            sns.boxplot(data = score_interval[score_interval["lambda"] == lambdas[(i-1)//3]], x = "bandwidth", y = "nmse", hue = "truncate_param", ax = ax)
        ax.legend(loc="upper left", prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.01, 1))
    
    fig.set_facecolor('w')
    fig.suptitle("score of edge detection, change interval: " + str(interval) + ", method: Dyn-GRN")
    plt.tight_layout()
    fig.savefig("../results_GGM/DynGRN_bifur_score_full_interval_" + str(interval) + ".png", bbox_inches = "tight")

    fig, big_axes = plt.subplots( figsize=(20.0, 10.0) , nrows=2, ncols=1, sharey=True) 
    lambdas = [0.01, 0.1]
    for row, big_ax in enumerate(big_axes, start=1):
        big_ax.set_title("Lambda {:.2f}".format(lambdas[row - 1]), fontsize=20)

        # Turn off axis lines and ticks of the big subplot 
        # obs alpha is 0 in RGBA string!
        big_ax.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
        # removes the white frame
        big_ax._frameon = False

    for i in range(1,7):
        ax = fig.add_subplot(2,3,i)
        if i%3 == 1:
            sns.boxplot(data = score_interval_diff[score_interval_diff["lambda"] == lambdas[(i-1)//3]], x = "bandwidth", y = "pearson", hue = "truncate_param", ax = ax)
        elif i%3 == 2:
            sns.boxplot(data = score_interval_diff[score_interval_diff["lambda"] == lambdas[(i-1)//3]], x = "bandwidth", y = "spearman", hue = "truncate_param", ax = ax)
        else:
            sns.boxplot(data = score_interval_diff[score_interval_diff["lambda"] == lambdas[(i-1)//3]], x = "bandwidth", y = "nmse", hue = "truncate_param", ax = ax)
        ax.legend(loc="upper left", prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.01, 1))

    fig.set_facecolor('w')
    fig.suptitle("score of changing edges detection, change interval: " + str(interval) + ", method: Dyn-GRN")
    plt.tight_layout()
    fig.savefig("../results_GGM/DynGRN_bifur_score_full_diff_interval_" + str(interval) + ".png", bbox_inches = "tight")    


# ----------------------------------- with TF information ---------------------------------#
for interval in [50, 100, 200]:
    score_interval = pd.DataFrame(columns = ["interval", "ngenes", "nmse","kendall-tau", "pearson", "spearman", "cosine similarity", "time", "model", "bandwidth", "truncate_param", "lambda"])
    score_interval_diff = pd.DataFrame(columns = ["interval", "ngenes", "nmse","kendall-tau", "pearson", "spearman", "cosine similarity", "time", "model", "bandwidth", "truncate_param", "lambda"])


    for (ngenes, ntfs) in [(20, 5), (30, 10), (50, 20), (100, 50)]:
        result_dir = "../results_GGM/bifur_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "/"

        score = pd.read_csv(result_dir + "score.csv", index_col = 0)
        score_diff = pd.read_csv(result_dir + "score_diff.csv", index_col = 0)

        # score for admm
        score = score[score["model"] == "Dyn-GRN-TF"]
        score_diff = score_diff[score_diff["model"] == "Dyn-GRN-TF"]
        score_interval = pd.concat([score_interval, score], axis = 0)
        score_interval_diff = pd.concat([score_interval_diff, score_diff], axis = 0)

    # Plot including lambda
    fig, big_axes = plt.subplots( figsize=(20.0, 10.0) , nrows=2, ncols=1, sharey=True) 
    lambdas = [0.01, 0.1]
    for row, big_ax in enumerate(big_axes, start=1):
        big_ax.set_title("Lambda {:.2f}".format(lambdas[row - 1]), fontsize=20)

        # Turn off axis lines and ticks of the big subplot 
        # obs alpha is 0 in RGBA string!
        big_ax.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
        # removes the white frame
        big_ax._frameon = False

    for i in range(1,7):
        ax = fig.add_subplot(2,3,i)
        if i%3 == 1:
            sns.boxplot(data = score_interval[score_interval["lambda"] == lambdas[(i-1)//3]], x = "bandwidth", y = "pearson", hue = "truncate_param", ax = ax)
        elif i%3 == 2:
            sns.boxplot(data = score_interval[score_interval["lambda"] == lambdas[(i-1)//3]], x = "bandwidth", y = "spearman", hue = "truncate_param", ax = ax)
        else:
            sns.boxplot(data = score_interval[score_interval["lambda"] == lambdas[(i-1)//3]], x = "bandwidth", y = "nmse", hue = "truncate_param", ax = ax)
        ax.legend(loc="upper left", prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.01, 1))

    fig.set_facecolor('w')
    fig.suptitle("score of edge detection, change interval: " + str(interval) + ", method: Dyn-GRN-TF")
    plt.tight_layout()
    fig.savefig("../results_GGM/DynGRN_bifur_score_full_interval_" + str(interval) + "_tf.png", bbox_inches = "tight")

    fig, big_axes = plt.subplots( figsize=(20.0, 10.0) , nrows=2, ncols=1, sharey=True) 
    lambdas = [0.01, 0.1]
    for row, big_ax in enumerate(big_axes, start=1):
        big_ax.set_title("Lambda {:.2f}".format(lambdas[row - 1]), fontsize=20)

        # Turn off axis lines and ticks of the big subplot 
        # obs alpha is 0 in RGBA string!
        big_ax.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
        # removes the white frame
        big_ax._frameon = False

    for i in range(1,7):
        ax = fig.add_subplot(2,3,i)
        if i%3 == 1:
            sns.boxplot(data = score_interval_diff[score_interval_diff["lambda"] == lambdas[(i-1)//3]], x = "bandwidth", y = "pearson", hue = "truncate_param", ax = ax)
        elif i%3 == 2:
            sns.boxplot(data = score_interval_diff[score_interval_diff["lambda"] == lambdas[(i-1)//3]], x = "bandwidth", y = "spearman", hue = "truncate_param", ax = ax)
        else:
            sns.boxplot(data = score_interval_diff[score_interval_diff["lambda"] == lambdas[(i-1)//3]], x = "bandwidth", y = "nmse", hue = "truncate_param", ax = ax)
        ax.legend(loc="upper left", prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.01, 1))

    fig.set_facecolor('w')
    fig.suptitle("score of changing edges detection, change interval: " + str(interval) + ", method: Dyn-GRN-TF")
    plt.tight_layout()
    fig.savefig("../results_GGM/DynGRN_bifur_score_full_diff_interval_" + str(interval) + "_tf.png", bbox_inches = "tight")

    # Select the best bandwidth, truncate_param, and lambda setting for each interval size. 

# In[]
ntimes = 1000

# ----------------------------------- without TF information ---------------------------------#
# Bandwidth: 0.1, truncate parameter: 0.1, lambda: 0.1
bandwith = 0.1
truncate_param = 0.1
lamb = 0.1
score_all = pd.DataFrame(columns = ["interval", "ngenes", "nmse","kendall-tau", "pearson", "spearman", "cosine similarity", "time", "model", "bandwidth", "truncate_param", "lambda"])
score_all_diff = pd.DataFrame(columns = ["interval", "ngenes", "nmse","kendall-tau", "pearson", "spearman", "cosine similarity", "time", "model", "bandwidth", "truncate_param", "lambda"])
    
for interval in [50, 100, 200]:
    for (ngenes, ntfs) in [(20, 5), (30, 10), (50, 20), (100, 50)]:
        result_dir = "../results_GGM/bifur_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "/"

        score = pd.read_csv(result_dir + "score.csv", index_col = 0)
        score_diff = pd.read_csv(result_dir + "score_diff.csv", index_col = 0)

        # score for admm
        score_dyngrn = score[(score["model"] == "Dyn-GRN")&(score["lambda"] == 0.1)&(score["bandwidth"] == 0.1)&(score["truncate_param"] == 0.1)]
        score_dyngrn_diff = score_diff[(score_diff["model"] == "Dyn-GRN")&(score_diff["lambda"] == 0.1)&(score_diff["bandwidth"] == 0.1)&(score_diff["truncate_param"] == 0.1)]
        score_dyngrn_tf = score[(score["model"] == "Dyn-GRN-TF")&(score["lambda"] == 0.1)&(score["bandwidth"] == 0.1)&(score["truncate_param"] == 0.1)]
        score_dyngrn_tf_diff = score_diff[(score_diff["model"] == "Dyn-GRN-TF")&(score_diff["lambda"] == 0.1)&(score_diff["bandwidth"] == 0.1)&(score_diff["truncate_param"] == 0.1)]

        score_other = score[(score["model"] != "Dyn-GRN")|(score["model"] != "Dyn-GRN-TF")]
        score_other_diff = score_diff[(score_diff["model"] != "Dyn-GRN")|(score_diff["model"] != "Dyn-GRN-TF")]
        score = pd.concat([score_dyngrn, score_dyngrn_tf, score_other], axis = 0)
        score_diff = pd.concat([score_dyngrn_diff, score_dyngrn_tf_diff, score_other_diff], axis = 0)
        score_all = pd.concat([score_all, score], axis = 0)
        score_all_diff = pd.concat([score_all_diff, score_diff], axis = 0)

        fig, ax = plt.subplots( figsize=(40.0, 10.0) , nrows=2, ncols=3) 
        sns.boxplot(data = score, x = "model", y = "pearson", ax = ax[0,0])
        sns.boxplot(data = score, x = "model", y = "kendall-tau", ax = ax[0,1])
        sns.boxplot(data = score, x = "model", y = "spearman", ax = ax[0,2])
        sns.boxplot(data = score, x = "model", y = "cosine similarity", ax = ax[1,0])
        sns.boxplot(data = score, x = "model", y = "nmse", ax = ax[1,1])
        ax[0,0].set_ylim([-1 , 1])
        ax[0,1].set_ylim([-1 , 1])
        ax[0,2].set_ylim([-1 , 1])
        ax[1,0].set_ylim([-1 , 1])
        
        fig.set_facecolor('w')
        fig.suptitle("score of edge detection, change interval: " + str(interval))
        plt.tight_layout()
        fig.savefig(result_dir + "compare_models.png", bbox_inches = "tight")

fig, ax = plt.subplots( figsize=(40.0, 10.0) , nrows=2, ncols=3) 
sns.boxplot(data = score_all, x = "model", y = "pearson", ax = ax[0,0])
sns.boxplot(data = score_all, x = "model", y = "kendall-tau", ax = ax[0,1])
sns.boxplot(data = score_all, x = "model", y = "spearman", ax = ax[0,2])
sns.boxplot(data = score_all, x = "model", y = "cosine similarity", ax = ax[1,0])
sns.boxplot(data = score_all, x = "model", y = "nmse", ax = ax[1,1])
ax[0,0].set_ylim([-1 , 1])
ax[0,1].set_ylim([-1 , 1])
ax[0,2].set_ylim([-1 , 1])
ax[1,0].set_ylim([-1 , 1])


fig.set_facecolor('w')
fig.suptitle("score of edge detection")
plt.tight_layout()
fig.savefig("../results_GGM/compare_models_bifur.png", bbox_inches = "tight")        


# %%
