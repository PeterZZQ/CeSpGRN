# In[]
from operator import index
from re import S
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
from sklearn.decomposition import PCA

plt.rcParams["font.size"] = 16

from multiprocessing import Pool, cpu_count

def check_symmetric(a, rtol=1e-04, atol=1e-04):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def calc_scores(thetas_inf, thetas_gt, interval, model, bandwidth = 0, truncate_param = 0, lamb = 0):

    scores = pd.DataFrame(columns = ["interval", "ngenes", "time", "model", "bandwidth", "truncate_param", "lambda", "nmse","kendall-tau", "pearson", \
        "spearman", "cosine similarity", "AUPRC (pos)", "AUPRC (neg)", "AUPRC (abs)", "Early Precision (pos)", "Early Precision (neg)", "Early Precision (abs)", "AUPRC random (pos)",\
            "AUPRC random (neg)", "AUPRC random (abs)", "Early Precision random (pos)", "Early Precision random (neg)","Early Precision random (abs)","AUPRC Ratio (pos)", \
                "AUPRC Ratio (neg)", "AUPRC Ratio (abs)", "Early Precision Ratio (pos)", "Early Precision Ratio (neg)", "Early Precision Ratio (abs)"])

    assert thetas_inf.shape[0] == thetas_gt.shape[0]

    ntimes = thetas_inf.shape[0]
    ngenes = thetas_inf.shape[1]
    
    for time in range(0, ntimes):
        np.random.seed(time)
        thetas_rand = np.random.randn(ngenes, ngenes)
        # make symmetric
        thetas_rand = (thetas_rand + thetas_rand.T)/2

        # ground truth should be symmetric too
        assert check_symmetric(thetas_gt[time])
        
        if not check_symmetric(thetas_inf[time]):
            thetas_inf[time] = (thetas_inf[time] + thetas_inf[time].T)/2

        # CeSpGRN should infer symmetric matrix
        if (model == "CeSpGRN")|(model == "CeSpGRN-kt")|(model == "CeSpGRN-kt-TF"):
            assert check_symmetric(thetas_inf[time])
        
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
                            "time":time,
                            "model": model,
                            "bandwidth": bandwidth,
                            "truncate_param":truncate_param,
                            "lambda":lamb}, ignore_index=True)  


    return scores    

def calc_scores_static(setting):
    theta_inf = setting["theta_inf"]
    theta_gt = setting["theta_gt"]
    interval = setting["interval"]
    model = setting["model"]
    time = setting["time"] 
    bandwidth = setting["bandwidth"]
    truncate_param = setting["truncate_param"]
    lamb = setting["lamb"]

    score = pd.DataFrame(columns = ["interval", "ngenes", "time", "model", "bandwidth", "truncate_param", "lambda", "nmse","kendall-tau", "pearson", \
        "spearman", "cosine similarity", "AUPRC (pos)", "AUPRC (neg)", "AUPRC (abs)", "Early Precision (pos)", "Early Precision (neg)", "Early Precision (abs)", "AUPRC random (pos)",\
            "AUPRC random (neg)", "AUPRC random (abs)", "Early Precision random (pos)", "Early Precision random (neg)","Early Precision random (abs)","AUPRC Ratio (pos)", \
                "AUPRC Ratio (neg)", "AUPRC Ratio (abs)", "Early Precision Ratio (pos)", "Early Precision Ratio (neg)", "Early Precision Ratio (abs)"])

    ngenes = theta_inf.shape[0]
    np.random.seed(time)
    thetas_rand = np.random.randn(ngenes, ngenes)
    # make symmetric
    thetas_rand = (thetas_rand + thetas_rand.T)/2    
    assert check_symmetric(theta_gt)
    if not check_symmetric(theta_inf):
        theta_inf = (theta_inf + theta_inf.T)/2

    # CeSpGRN should infer symmetric matrix
    if (model == "CeSpGRN")|(model == "CeSpGRN-kt")|(model == "CeSpGRN-kt-TF"):
        assert check_symmetric(theta_inf) 

    nmse = bmk.NMSE(G_inf = theta_inf, G_true = theta_gt)
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
    
    score = score.append({"interval": interval,
                        "ngenes": ngenes,
                        "nmse": nmse, 
                        "pearson": pearson_val, 
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
                        "time":time,
                        "model": model,
                        "bandwidth": bandwidth,
                        "truncate_param":truncate_param,
                        "lambda":lamb}, ignore_index=True)  

    return score

def calc_scores_para(thetas_inf, thetas_gt, interval, model, bandwidth = 0, truncate_param = 0, lamb = 0):
    assert thetas_inf.shape[0] == thetas_gt.shape[0]

    ntimes = thetas_inf.shape[0]
    settings = []
    for time in range(0, ntimes):
        settings.append({
            "theta_inf": thetas_inf[time],
            "theta_gt": thetas_gt[time],
            "interval": interval,
            "model": model,
            "bandwidth": bandwidth,
            "truncate_param": truncate_param,
            "lamb": lamb,
            "time": time
        })


    pool = Pool(8) 
    scores = pool.map(calc_scores_static, [x for x in settings])
    pool.close()
    pool.join()
    scores.sort(key = lambda x: x["time"].values[0])
    scores = pd.concat(scores, axis = 0, ignore_index = True)
    return scores


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
    traj = setting["traj"]

    print("ntimes: " + str(ntimes) + ", interval: " + str(interval) + ", ngenes: " + str(ngenes) + ", seed: " + str(seed) + ", initial graph: sergio\n")
    result_dir = "../results_softODE_hyperparameter2/" + traj + "_ngenes_" + str(ngenes) + "_ncell_" + str(ntimes) + "_seed_" + str(seed) + "/"
    data_dir = path + traj + "_ngenes_" + str(ngenes) + "_ncell_" + str(ntimes) + "_seed_" + str(seed) + "/"
    
    # X = np.load(data_dir + "expr.npy")
    gt_adj = np.load(data_dir + "GRNs.npy")
    sim_time = np.load(data_dir + "pseudotime.npy")
    print("data loaded.")
    
    # genie3            
    thetas = np.load(file = result_dir + "theta_genie.npy")
    scores = calc_scores_para(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "GENIE3")
    print("genie3.")

    # # genie3-tf
    # thetas = np.load(file = result_dir + "theta_genie_tf.npy")
    # score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "GENIE3-TF")
    # scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    # print("genie3-tf.")

    # genie3-dyn
    thetas = np.load(file = result_dir + "theta_genie_dyn.npy")
    score = calc_scores_para(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "GENIE3-Dyn")
    scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    print("genie3-dyn.")
    
    
    # # genie3-dyn-tf 
    # thetas = np.load(file = result_dir + "theta_genie_dyn_tf.npy")
    # score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "GENIE3-Dyn-TF")
    # scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    # print("genie3-dyn-tf.")


    # SCODE (normalize)
    thetas = np.load(file = result_dir + "theta_scode_norm.npy")
    score = calc_scores_para(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "SCODE")
    scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    print("SCODE (True T).")

    
    # SCODE-DYN (nomralize)
    thetas = np.load(file = result_dir + "theta_scode_dyn_norm.npy")
    score = calc_scores_para(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "SCODE-Dyn")
    scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    print("SCODE-DYN (True T).")

    # CSN        
    thetas = np.load(file = result_dir + "theta_CSN.npy")
    score = calc_scores_para(thetas_inf = thetas, thetas_gt = np.abs(gt_adj), interval = interval, model = "CSN")
    scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    print("CSN.") 

    # admm, hyper-parameter
    # for bandwidth in [0.1, 1, 10]:
    #     for truncate_param in [30, 100]:
    #         for lamb in [0.0001]:
    #             thetas = np.load(file = result_dir + "neigh_thetas_ngenes20_band_" + str(bandwidth) + "_lamb_" + str(lamb) + "_trun_" + str(truncate_param) + ".npy")
    #             score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "CeSpGRN", bandwidth = bandwidth, truncate_param = truncate_param, lamb = lamb)
    #             scores = pd.concat([scores, score], axis = 0, ignore_index = True)

    #             fig = plt.figure(figsize = (10,7))
    #             X_umap = umap_op.fit_transform(thetas.reshape(ntimes * nsample, -1))
    #             ax = fig.add_subplot()
    #             ax.scatter(X_umap[:,0], X_umap[:,1], c = sim_time, s = 5)
    #             fig.savefig(result_dir + "infer_G_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_0.png", bbox_inches = "tight")

    #             thetas = np.load(file = result_dir + "TF_neigh_thetas_ngenes20_band_" + str(bandwidth) + "_beta_100_trun_" + str(truncate_param) + ".npy")
    #             score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "CeSpGRN-TF", bandwidth = bandwidth, truncate_param = truncate_param, lamb = lamb)
    #             scores = pd.concat([scores, score], axis = 0, ignore_index = True)

    bandwidth = 10
    lamb = 0.001
    truncate_param = 30
    beta = 100
    # CeSpGRN
    thetas = np.load(file = result_dir + "thetas_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + ".npy")
    score = calc_scores_para(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "CeSpGRN", bandwidth = bandwidth, truncate_param = truncate_param, lamb = lamb)
    scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    print("CeSpGRN.") 

    # CeSpGRN-kt
    thetas = np.load(file = result_dir + "thetas_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_kt.npy")
    score = calc_scores_para(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "CeSpGRN-kt", bandwidth = bandwidth, truncate_param = truncate_param, lamb = lamb)
    scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    print("CeSpGRN-kt.") 

    # CeSpGRN-kt-TF
    thetas = np.load(file = result_dir + "thetas_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) +"_" + str(beta) + "_kt.npy")
    score = calc_scores_para(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "CeSpGRN-kt-TF", bandwidth = bandwidth, truncate_param = truncate_param, lamb = lamb)
    scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    print("CeSpGRN-kt-TF.") 

    scores.to_csv(result_dir + "score.csv")



seeds = [0,1,2,3,4]
# settings = []
# for traj in ["linear", "bifurc"]:
#     for (ngenes, ntfs) in [(20, 5)]:
#         for seed in seeds:
#             settings.append({
#                 "ntimes": ntimes,
#                 "interval": 20,
#                 "ngenes": ngenes,
#                 "ntfs": ntfs,
#                 "seed": seed,
#                 "traj": traj
#             })


# pool = Pool(10) 
# pool.map(summarize_scores, [x for x in settings])
# pool.close()
# pool.join()

setting = {
    "ntimes": ntimes,
    "interval": 20,
    "ngenes": 20,
    "ntfs": 5,
    "seed": seeds[eval(sys.argv[1])],
    "traj": "linear"
}
summarize_scores(setting)

setting = {
    "ntimes": ntimes,
    "interval": 20,
    "ngenes": 20,
    "ntfs": 5,
    "seed": seeds[eval(sys.argv[1])],
    "traj": "bifurc"
}
summarize_scores(setting)

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
    traj = setting["traj"]

    print("ntimes: " + str(ntimes) + ", interval: " + str(interval) + ", ngenes: " + str(ngenes) + ", seed: " + str(seed) + ", initial graph: sergio\n")
    result_dir = "../results_softODE/" + traj + "_ngenes_" + str(ngenes) + "_ncell_" + str(ntimes) + "_seed_" + str(seed) + "/"
    data_dir = path + traj + "_ngenes_" + str(ngenes) + "_ncell_" + str(ntimes) + "_seed_" + str(seed) + "/"
    
    # X = np.load(data_dir + "expr.npy")
    gt_adj = np.load(data_dir + "GRNs.npy")
    sim_time = np.load(data_dir + "pseudotime.npy")
    print("data loaded.")

    step = 400
    gt_adj = gt_adj[step::,:,:] - gt_adj[:-step:,:,:]   

    # # genie3            
    # thetas = np.load(file = result_dir + "theta_genie.npy")
    # thetas = thetas[step::,:,:] - thetas[:-step:,:,:]
    # score = calc_scores_para(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "GENIE3")
    # scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    # print("genie3.")

    # # genie3-tf
    # thetas = np.load(file = result_dir + "theta_genie_tf.npy")
    # thetas = thetas[step::,:,:] - thetas[:-step:,:,:]
    # score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "GENIE3-TF")
    # scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    # print("genie3-tf.")

    # genie3-dyn
    thetas = np.load(file = result_dir + "theta_genie_dyn.npy")
    thetas = thetas[step::,:,:] - thetas[:-step:,:,:]
    scores = calc_scores_para(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "GENIE3-Dyn")
    # scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    print("genie3-dyn.")
    
    
    # # genie3-dyn-tf 
    # thetas = np.load(file = result_dir + "theta_genie_dyn_tf.npy")
    # thetas = thetas[step::,:,:] - thetas[:-step:,:,:]
    # score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "GENIE3-Dyn-TF")
    # scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    # print("genie3-dyn-tf.")


    # # SCODE (normalize)
    # thetas = np.load(file = result_dir + "theta_scode_norm.npy")
    # thetas = thetas[step::,:,:] - thetas[:-step:,:,:]
    # score = calc_scores_para(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "SCODE")
    # scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    # print("SCODE (True T).")

    
    # SCODE-DYN (nomralize)
    thetas = np.load(file = result_dir + "theta_scode_dyn_norm.npy")
    thetas = thetas[step::,:,:] - thetas[:-step:,:,:]
    score = calc_scores_para(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "SCODE-Dyn")
    scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    print("SCODE-DYN (True T).")

    # CSN       
    thetas = np.load(file = result_dir + "theta_CSN.npy")
    thetas = thetas[step::,:,:] - thetas[:-step:,:,:]  
    score = calc_scores_para(thetas_inf = thetas, thetas_gt = np.abs(gt_adj), interval = interval, model = "CSN")
    scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    print("CSN.") 

    # for bandwidth in [0.1, 1, 10]:
    #     for truncate_param in [30, 100]:
    #         for lamb in [0.0001]:
    #             thetas = np.load(file = result_dir + "neigh_thetas_ngenes20_band_" + str(bandwidth) + "_lamb_" + str(lamb) + "_trun_" + str(truncate_param) + ".npy")
    #             thetas = thetas[step::,:,:] - thetas[:-step:,:,:]  
    #             score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "CeSpGRN", bandwidth = bandwidth, truncate_param = truncate_param, lamb = lamb)
    #             scores = pd.concat([scores, score], axis = 0, ignore_index = True)

    #             thetas = np.load(file = result_dir + "TF_neigh_thetas_ngenes20_band_" + str(bandwidth) + "_beta_100_trun_" + str(truncate_param) + ".npy")
    #             thetas = thetas[step::,:,:] - thetas[:-step:,:,:]  
    #             score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "CeSpGRN-TF", bandwidth = bandwidth, truncate_param = truncate_param, lamb = lamb)
    #             scores = pd.concat([scores, score], axis = 0, ignore_index = True)

    bandwidth = 10
    lamb = 0.001
    truncate_param = 30
    beta = 100
    # CeSpGRN
    thetas = np.load(file = result_dir + "thetas_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + ".npy")
    thetas = thetas[step::,:,:] - thetas[:-step:,:,:] 
    score = calc_scores_para(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "CeSpGRN", bandwidth = bandwidth, truncate_param = truncate_param, lamb = lamb)
    scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    print("CeSpGRN.") 

    # CeSpGRN-kt
    thetas = np.load(file = result_dir + "thetas_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_kt.npy")
    thetas = thetas[step::,:,:] - thetas[:-step:,:,:] 
    score = calc_scores_para(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "CeSpGRN-kt", bandwidth = bandwidth, truncate_param = truncate_param, lamb = lamb)
    scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    print("CeSpGRN-kt.") 

    # CeSpGRN-kt-TF
    thetas = np.load(file = result_dir + "thetas_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) +"_" + str(beta) + "_kt.npy")
    thetas = thetas[step::,:,:] - thetas[:-step:,:,:] 
    score = calc_scores_para(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "CeSpGRN-kt-TF", bandwidth = bandwidth, truncate_param = truncate_param, lamb = lamb)
    scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    print("CeSpGRN-kt-TF.") 

    scores.to_csv(result_dir + "score_diff.csv")

seeds = [0,1,2,3,4]
# settings = []
# for traj in ["linear", "bifurc"]:
#     for (ngenes, ntfs) in [(20, 5)]:
#         for seed in [0,1,2,3,4]:
#             settings.append({
#                 "ntimes": ntimes,
#                 "interval": 20,
#                 "ngenes": ngenes,
#                 "ntfs": ntfs,
#                 "seed": seed,
#                 "traj": traj
#             })


# pool = Pool(10) 
# pool.map(summarize_scores_diff, [x for x in settings])
# pool.close()
# pool.join()

setting = {
    "ntimes": ntimes,
    "interval": 20,
    "ngenes": 20,
    "ntfs": 5,
    "seed": seeds[eval(sys.argv[1])],
    "traj": "linear"
}
summarize_scores_diff(setting)

setting = {
    "ntimes": ntimes,
    "interval": 20,
    "ngenes": 20,
    "ntfs": 5,
    "seed": seeds[eval(sys.argv[1])],
    "traj": "bifurc"
}
summarize_scores_diff(setting)

# In[] summarize the mean result in csv file

ntimes = 1000
nsample = 1
interval = 20
for traj in ["linear", "bifurc"]:
    for (ngenes, ntfs) in [(20, 5)]:
        for seed in [0,1,2,3,4]:
            print("ntimes: " + str(ntimes) + ", interval: " + str(interval) + ", ngenes: " + str(ngenes) + ", seed: " + str(seed) + ", initial graph: sergio")
            result_dir = "../results_softODE/" + traj + "_ngenes_" + str(ngenes) + "_ncell_" + str(ntimes) + "_seed_" + str(seed) + "/"
            score = pd.read_csv(result_dir + "score.csv", index_col = 0)
            mean_score = score.groupby(by = ["model", "bandwidth", "truncate_param", "lambda"]).mean()
            mean_score = mean_score.drop(["time"], axis = 1)
            mean_score.to_csv(result_dir + "mean_score.csv")
            # display(mean_score)

print("\ndifferences\n")

for traj in ["linear", "bifurc"]:
    for (ngenes, ntfs) in [(20, 5)]:
        for seed in [0,1,2,3,4]:
            print("ntimes: " + str(ntimes) + ", interval: " + str(interval) + ", ngenes: " + str(ngenes) + ", seed: " + str(seed) + ", initial graph: sergio")
            result_dir = "../results_softODE/" + traj + "_ngenes_" + str(ngenes) + "_ncell_" + str(ntimes) + "_seed_" + str(seed) + "/"
            score = pd.read_csv(result_dir + "score_diff.csv", index_col = 0)
            mean_score = score.groupby(by = ["model", "bandwidth", "truncate_param", "lambda"]).mean()
            mean_score = mean_score.drop(["time"], axis = 1)
            mean_score.to_csv(result_dir + "mean_score_diff.csv")
            # display(mean_score)




# In[]
ntimes = 1000
score_all = pd.DataFrame(columns = ["interval", "ngenes", "time", "model", "bandwidth", "truncate_param", "lambda", "nmse","kendall-tau", "pearson", "spearman", "cosine similarity", "AUPRC Ratio (pos)", "AUPRC Ratio (neg)", "AUPRC Ratio (abs)", "Early Precision Ratio (pos)", "Early Precision Ratio (neg)", "Early Precision Ratio (abs)"])
score_all_diff = pd.DataFrame(columns = ["interval", "ngenes", "time", "model", "bandwidth", "truncate_param", "lambda", "nmse","kendall-tau", "pearson", "spearman", "cosine similarity", "AUPRC Ratio (pos)", "AUPRC Ratio (neg)", "AUPRC Ratio (abs)", "Early Precision Ratio (pos)", "Early Precision Ratio (neg)", "Early Precision Ratio (abs)"])
interval = 20

for traj in ["bifurc", "linear"]:
    for (ngenes, ntfs) in [(20, 5)]:
        for seed in [0,1,2,3,4]:
            result_dir = "../results_softODE/" + traj + "_ngenes_" + str(ngenes) + "_ncell_" + str(ntimes) + "_seed_" + str(seed) + "/"
    
            score = pd.read_csv(result_dir + "score.csv", index_col = 0)
            score_diff = pd.read_csv(result_dir + "score_diff.csv", index_col = 0)
            # score_CeSpGRN = score[(score["model"] == "CeSpGRN-kt")&(score["bandwidth"] == 10)&(score["truncate_param"] == 30)]
            # score_CeSpGRN_TF = score[(score["model"] == "CeSpGRN-kt-TF")&(score["bandwidth"] == 10)&(score["truncate_param"] == 30)]
            # score_CeSpGRN_diff = score_diff[(score_diff["model"] == "CeSpGRN-kt")&(score_diff["bandwidth"] == 10)&(score_diff["truncate_param"] == 30)]
            # score_CeSpGRN_TF_diff = score_diff[(score_diff["model"] == "CeSpGRN-kt-TF")&(score_diff["bandwidth"] == 10)&(score_diff["truncate_param"] == 30)]
            # score_other = score[(score["model"] == "GENIE3")|(score["model"] == "GENIE3-Dyn")|(score["model"] == "SCODE")|(score["model"] == "SCODE-Dyn")|(score["model"] == "CSN")]
            # score_other_diff = score_diff[(score_diff["model"] == "GENIE3")|(score_diff["model"] == "GENIE3-Dyn")|(score_diff["model"] == "SCODE")|(score_diff["model"] == "SCODE-Dyn")|(score_diff["model"] == "CSN")]
            # score = pd.concat([score_other, score_CeSpGRN, score_CeSpGRN_TF], axis = 0)
            # score_diff = pd.concat([score_other_diff, score_CeSpGRN_diff, score_CeSpGRN_TF_diff], axis = 0)

            score = score[(score["model"] == "GENIE3")|(score["model"] == "GENIE3-Dyn")|(score["model"] == "SCODE")|(score["model"] == "SCODE-Dyn")|(score["model"] == "CSN")|(score["model"] == "CeSpGRN-kt")|(score["model"] == "CeSpGRN-kt-TF")|(score["model"] == "CeSpGRN")]
            score_diff = score_diff[(score_diff["model"] == "GENIE3")|(score_diff["model"] == "GENIE3-Dyn")|(score_diff["model"] == "SCODE")|(score_diff["model"] == "SCODE-Dyn")|(score_diff["model"] == "CSN")|(score_diff["model"] == "CeSpGRN-kt")|(score_diff["model"] == "CeSpGRN-kt-TF")|(score_diff["model"] == "CeSpGRN")]            
            
            score_all = pd.concat([score_all, score], axis = 0)
            score_all_diff = pd.concat([score_all_diff, score_diff], axis = 0)

            # fig, ax = plt.subplots( figsize=(12.0, 7.0) , nrows=1, ncols=2) 
            # boxplot1 = sns.boxplot(data = score, x = "model", y = "pearson", ax = ax[0])
            # boxplot2 = sns.boxplot(data = score, x = "model", y = "spearman", ax = ax[1])

            # ax[0].set_xticklabels(boxplot1.get_xticklabels(), rotation = 45)
            # ax[1].set_xticklabels(boxplot2.get_xticklabels(), rotation = 45)

            
            # fig.set_facecolor('w')
            # fig.suptitle("score of edge detection, change interval: " + str(interval))
            # plt.tight_layout()
            # fig.savefig(result_dir + "compare_models.png", bbox_inches = "tight")

score_all_ori = score_all.copy()
score_all_diff_ori = score_all_diff.copy()   

# In[]
score_all = score_all_ori.copy()
score_all_diff = score_all_diff_ori.copy()   

fig, ax = plt.subplots( figsize=(25.0, 7.0) , nrows=1, ncols=4) 
score_all.loc[score_all["model"]!="CSN", "AUPRC Ratio (signed)"] = score_all.loc[score_all["model"]!="CSN", "AUPRC Ratio (pos)"].values
score_all.loc[score_all["model"]=="CSN", "AUPRC Ratio (signed)"] = score_all.loc[score_all["model"]=="CSN", "AUPRC Ratio (abs)"].values
score_all.loc[score_all["model"]!="CSN", "Early Precision Ratio (signed)"] = score_all.loc[score_all["model"]!="CSN", "Early Precision Ratio (pos)"].values
score_all.loc[score_all["model"]=="CSN", "Early Precision Ratio (signed)"] = score_all.loc[score_all["model"]=="CSN", "Early Precision Ratio (abs)"].values
score_all.loc[score_all["model"] != "CSN", "AUPRC (signed)"] = (score_all.loc[score_all["model"] != "CSN", "AUPRC (pos)"].values + score_all.loc[score_all["model"] != "CSN", "AUPRC (neg)"].values)/2
score_all.loc[score_all["model"] == "CSN", "AUPRC (signed)"] = score_all.loc[score_all["model"] == "CSN", "AUPRC (abs)"].values
score_all.loc[score_all["model"] != "CSN", "Early Precision (signed)"] = (score_all.loc[score_all["model"] != "CSN", "Early Precision (pos)"].values + score_all.loc[score_all["model"] != "CSN", "Early Precision (neg)"].values)/2
score_all.loc[score_all["model"] == "CSN", "Early Precision (signed)"] = score_all.loc[score_all["model"] == "CSN", "Early Precision (abs)"].values

score_all_diff.loc[score_all_diff["model"]!="CSN","AUPRC Ratio (signed)"] = (score_all_diff.loc[score_all_diff["model"]!="CSN","AUPRC Ratio (pos)"].values + score_all_diff.loc[score_all_diff["model"]!="CSN","AUPRC Ratio (neg)"].values)/2
score_all_diff.loc[score_all_diff["model"]=="CSN","AUPRC Ratio (signed)"] = score_all_diff.loc[score_all_diff["model"]=="CSN","AUPRC Ratio (abs)"].values
score_all_diff.loc[score_all_diff["model"]!="CSN", "Early Precision Ratio (signed)"] = (score_all_diff.loc[score_all_diff["model"]!="CSN", "Early Precision Ratio (pos)"].values + score_all_diff.loc[score_all_diff["model"]!="CSN", "Early Precision Ratio (neg)"].values)/2
score_all_diff.loc[score_all_diff["model"]=="CSN", "Early Precision Ratio (signed)"] = score_all_diff.loc[score_all_diff["model"]=="CSN", "Early Precision Ratio (abs)"].values
score_all_diff.loc[score_all_diff["model"] != "CSN", "AUPRC (signed)"] = (score_all_diff.loc[score_all_diff["model"] != "CSN", "AUPRC (pos)"].values + score_all_diff.loc[score_all_diff["model"] != "CSN", "AUPRC (neg)"].values)/2
score_all_diff.loc[score_all_diff["model"] == "CSN", "AUPRC (signed)"] = score_all_diff.loc[score_all_diff["model"] == "CSN", "AUPRC (abs)"].values
score_all_diff.loc[score_all_diff["model"] != "CSN", "Early Precision (signed)"] = (score_all_diff.loc[score_all_diff["model"] != "CSN", "Early Precision (pos)"].values + score_all_diff.loc[score_all_diff["model"] != "CSN", "Early Precision (neg)"].values)/2
score_all_diff.loc[score_all_diff["model"] == "CSN", "Early Precision (signed)"] = score_all_diff.loc[score_all_diff["model"] == "CSN", "Early Precision (abs)"].values


score_all = score_all[score_all["model"] != "CeSpGRN-kt-TF"]
score_all_diff = score_all_diff[score_all_diff["model"] != "CeSpGRN-kt-TF"]

boxplot1 = sns.boxplot(data = score_all, x = "model", y = "pearson", ax = ax[0])
boxplot2 = sns.boxplot(data = score_all, x = "model", y = "spearman", ax = ax[1])
boxplot3 = sns.boxplot(data = score_all[score_all["model"]!="CSN"], x = "model", y = "AUPRC Ratio (signed)", ax = ax[2])
boxplot4 = sns.boxplot(data = score_all, x = "model", y = "Early Precision Ratio (signed)", ax = ax[3])

ax[0].set_xticklabels(boxplot1.get_xticklabels(), rotation = 45)
ax[1].set_xticklabels(boxplot2.get_xticklabels(), rotation = 45)
ax[2].set_xticklabels(boxplot3.get_xticklabels(), rotation = 45)
ax[3].set_xticklabels(boxplot4.get_xticklabels(), rotation = 45)

add_median_labels(boxplot1.axes)
add_median_labels(boxplot2.axes)
add_median_labels(boxplot3.axes)
add_median_labels(boxplot4.axes)

fig.set_facecolor('w')
fig.suptitle("score of edge detection")
plt.tight_layout()
fig.savefig("../results_softODE/compare_models_all.png", bbox_inches = "tight")        


fig, ax = plt.subplots( figsize=(25.0, 7.0) , nrows=1, ncols=4) 
boxplot1 = sns.boxplot(data = score_all_diff, x = "model", y = "pearson", ax = ax[0])
boxplot2 = sns.boxplot(data = score_all_diff, x = "model", y = "spearman", ax = ax[1])
boxplot3 = sns.boxplot(data = score_all_diff[score_all_diff["model"]!="CSN"], x = "model", y = "AUPRC Ratio (signed)", ax = ax[2])
boxplot4 = sns.boxplot(data = score_all_diff, x = "model", y = "Early Precision Ratio (signed)", ax = ax[3])

ax[0].set_xticklabels(boxplot1.get_xticklabels(), rotation = 45)
ax[1].set_xticklabels(boxplot2.get_xticklabels(), rotation = 45)
ax[2].set_xticklabels(boxplot3.get_xticklabels(), rotation = 45)
ax[3].set_xticklabels(boxplot4.get_xticklabels(), rotation = 45)

add_median_labels(boxplot1.axes)
add_median_labels(boxplot2.axes)
add_median_labels(boxplot3.axes)
add_median_labels(boxplot4.axes)

fig.set_facecolor('w')
fig.suptitle("score of changing edge detection")
plt.tight_layout()
fig.savefig("../results_softODE/compare_models_all_diff.png", bbox_inches = "tight")  

# score_all = score_all_ori.copy()
# score_all_diff = score_all_diff_ori.copy()  

# fig, ax = plt.subplots( figsize=(12.0, 5.0) , nrows=1, ncols=4) 
# score_all.loc[score_all["model"]!="CSN", "AUPRC Ratio (signed)"] = score_all.loc[score_all["model"]!="CSN", "AUPRC Ratio (pos)"].values
# score_all.loc[score_all["model"]=="CSN", "AUPRC Ratio (signed)"] = score_all.loc[score_all["model"]=="CSN", "AUPRC Ratio (abs)"].values
# score_all_diff.loc[score_all_diff["model"]!="CSN","AUPRC Ratio (signed)"] = (score_all_diff.loc[score_all_diff["model"]!="CSN","AUPRC Ratio (pos)"].values + score_all_diff.loc[score_all_diff["model"]!="CSN","AUPRC Ratio (neg)"].values)/2
# score_all_diff.loc[score_all_diff["model"]=="CSN","AUPRC Ratio (signed)"] = score_all_diff.loc[score_all_diff["model"]=="CSN","AUPRC Ratio (abs)"].values
# score_all.loc[score_all["model"]!="CSN", "Early Precision Ratio (signed)"] = score_all.loc[score_all["model"]!="CSN", "Early Precision Ratio (pos)"].values
# score_all.loc[score_all["model"]=="CSN", "Early Precision Ratio (signed)"] = score_all.loc[score_all["model"]=="CSN", "Early Precision Ratio (abs)"].values
# score_all_diff.loc[score_all_diff["model"]!="CSN", "Early Precision Ratio (signed)"] = (score_all_diff.loc[score_all_diff["model"]!="CSN", "Early Precision Ratio (pos)"].values + score_all_diff.loc[score_all_diff["model"]!="CSN", "Early Precision Ratio (neg)"].values)/2
# score_all_diff.loc[score_all_diff["model"]=="CSN", "Early Precision Ratio (signed)"] = score_all_diff.loc[score_all_diff["model"]=="CSN", "Early Precision Ratio (abs)"].values

# score_all = score_all[(score_all["model"] == "CeSpGRN-kt-TF")|(score_all["model"] == "CeSpGRN-kt")]
# score_all_diff = score_all_diff[(score_all_diff["model"] == "CeSpGRN-kt-TF")|(score_all_diff["model"] == "CeSpGRN-kt")]


# boxplot1 = sns.boxplot(data = score_all, x = "model", y = "pearson", ax = ax[0])
# boxplot2 = sns.boxplot(data = score_all, x = "model", y = "spearman", ax = ax[1])
# boxplot3 = sns.boxplot(data = score_all, x = "model", y = "AUPRC Ratio (signed)", ax = ax[2])
# boxplot4 = sns.boxplot(data = score_all, x = "model", y = "Early Precision Ratio (signed)", ax = ax[3])

# add_median_labels(boxplot1.axes)
# add_median_labels(boxplot2.axes)
# add_median_labels(boxplot3.axes)
# add_median_labels(boxplot4.axes)

# ax[0].set_xticklabels(boxplot1.get_xticklabels(), rotation = 45)
# ax[1].set_xticklabels(boxplot2.get_xticklabels(), rotation = 45)
# ax[2].set_xticklabels(boxplot3.get_xticklabels(), rotation = 45)
# ax[3].set_xticklabels(boxplot4.get_xticklabels(), rotation = 45)

# fig.set_facecolor('w')
# fig.suptitle("score of edge detection")
# plt.tight_layout()
# fig.savefig("../results_softODE/compare_models_tf.png", bbox_inches = "tight")        


# fig, ax = plt.subplots( figsize=(12.0, 5.0) , nrows=1, ncols=4) 
# boxplot1 = sns.boxplot(data = score_all_diff, x = "model", y = "pearson", ax = ax[0])
# boxplot2 = sns.boxplot(data = score_all_diff, x = "model", y = "spearman", ax = ax[1])
# boxplot3 = sns.boxplot(data = score_all_diff, x = "model", y = "AUPRC Ratio (signed)", ax = ax[2])
# boxplot4 = sns.boxplot(data = score_all_diff, x = "model", y = "Early Precision Ratio (signed)", ax = ax[3])

# add_median_labels(boxplot1.axes)
# add_median_labels(boxplot2.axes)
# add_median_labels(boxplot3.axes)
# add_median_labels(boxplot4.axes)

# ax[0].set_xticklabels(boxplot1.get_xticklabels(), rotation = 45)
# ax[1].set_xticklabels(boxplot2.get_xticklabels(), rotation = 45)
# ax[2].set_xticklabels(boxplot3.get_xticklabels(), rotation = 45)
# ax[3].set_xticklabels(boxplot4.get_xticklabels(), rotation = 45)


# fig.set_facecolor('w')
# fig.suptitle("score of changing edge detection")
# plt.tight_layout()
# fig.savefig("../results_softODE/compare_models_diff_tf.png", bbox_inches = "tight")  

# In[] Final plots

score_all = score_all_ori.copy()
score_all_diff = score_all_diff_ori.copy()   

score_all.loc[score_all["model"]!="CSN", "AUPRC Ratio (signed)"] = score_all.loc[score_all["model"]!="CSN", "AUPRC Ratio (pos)"].values
score_all.loc[score_all["model"]=="CSN", "AUPRC Ratio (signed)"] = score_all.loc[score_all["model"]=="CSN", "AUPRC Ratio (abs)"].values
score_all.loc[score_all["model"]!="CSN", "Early Precision Ratio (signed)"] = score_all.loc[score_all["model"]!="CSN", "Early Precision Ratio (pos)"].values
score_all.loc[score_all["model"]=="CSN", "Early Precision Ratio (signed)"] = score_all.loc[score_all["model"]=="CSN", "Early Precision Ratio (abs)"].values
score_all.loc[score_all["model"] != "CSN", "AUPRC (signed)"] = (score_all.loc[score_all["model"] != "CSN", "AUPRC (pos)"].values + score_all.loc[score_all["model"] != "CSN", "AUPRC (neg)"].values)/2
score_all.loc[score_all["model"] == "CSN", "AUPRC (signed)"] = score_all.loc[score_all["model"] == "CSN", "AUPRC (abs)"].values
score_all.loc[score_all["model"] != "CSN", "Early Precision (signed)"] = (score_all.loc[score_all["model"] != "CSN", "Early Precision (pos)"].values + score_all.loc[score_all["model"] != "CSN", "Early Precision (neg)"].values)/2
score_all.loc[score_all["model"] == "CSN", "Early Precision (signed)"] = score_all.loc[score_all["model"] == "CSN", "Early Precision (abs)"].values

score_all = score_all.loc[score_all["model"] != "CeSpGRN",:]
score_all.loc[score_all["model"] == "CeSpGRN-kt", "model"] = "CeSpGRN"
score_all.loc[score_all["model"] == "CeSpGRN-kt-TF", "model"] = "CeSpGRN-TF"
score_all = score_all[score_all["model"] != "CeSpGRN-TF"]
  
fig, ax = plt.subplots( figsize=(15.0, 5.0) , nrows=1, ncols=4) 
boxplot1 = sns.boxplot(data = score_all, x = "model", y = "Early Precision (signed)", ax = ax[0])
boxplot2 = sns.boxplot(data = score_all[score_all["model"] != "CSN"], x = "model", y = "AUPRC (signed)", ax = ax[1])
boxplot3 = sns.boxplot(data = score_all, x = "model", y = "pearson", ax = ax[2])
boxplot4 = sns.boxplot(data = score_all, x = "model", y = "spearman", ax = ax[3])
# add_median_labels(boxplot1.axes)
# add_median_labels(boxplot2.axes)
# add_median_labels(boxplot3.axes)
# add_median_labels(boxplot4.axes)
# ax[0].set_ylim([-1 , 1])
# ax[1].set_ylim([-1 , 1])
ax[0].set_xticklabels(boxplot1.get_xticklabels(), rotation = 90, fontsize = 15)
ax[1].set_xticklabels(boxplot2.get_xticklabels(), rotation = 90, fontsize = 15)
ax[2].set_xticklabels(boxplot3.get_xticklabels(), rotation = 90, fontsize = 15)
ax[3].set_xticklabels(boxplot4.get_xticklabels(), rotation = 90, fontsize = 15)

ax[2].set_ylim([-0.4,0.4])

ax[0].set(xlabel=None)
ax[1].set(xlabel=None)
ax[2].set(xlabel=None)
ax[3].set(xlabel=None)
ax[2].set_ylabel("Pearson")
ax[3].set_ylabel("Spearman")
# ax[0,0].set_yscale('log')
# ax[0,1].set_yscale('log')
fig.set_facecolor('w')
plt.tight_layout()
fig.savefig("../results_softODE/compare_models.png", bbox_inches = "tight") 

fig, ax = plt.subplots( figsize=(15.0, 5.0) , nrows=1, ncols=4) 
boxplot1 = sns.boxplot(data = score_all, x = "model", y = "Early Precision Ratio (signed)", ax = ax[0])
boxplot2 = sns.boxplot(data = score_all[score_all["model"] != "CSN"], x = "model", y = "AUPRC Ratio (signed)", ax = ax[1])
boxplot3 = sns.boxplot(data = score_all, x = "model", y = "pearson", ax = ax[2])
boxplot4 = sns.boxplot(data = score_all, x = "model", y = "spearman", ax = ax[3])
# add_median_labels(boxplot1.axes)
# add_median_labels(boxplot2.axes)
# add_median_labels(boxplot3.axes)
# add_median_labels(boxplot4.axes)
# ax[0].set_ylim([-1 , 1])
# ax[1].set_ylim([-1 , 1])
ax[0].set_xticklabels(boxplot1.get_xticklabels(), rotation = 90, fontsize = 15)
ax[1].set_xticklabels(boxplot2.get_xticklabels(), rotation = 90, fontsize = 15)
ax[2].set_xticklabels(boxplot3.get_xticklabels(), rotation = 90, fontsize = 15)
ax[3].set_xticklabels(boxplot4.get_xticklabels(), rotation = 90, fontsize = 15)

ax[2].set_ylim([-0.4,0.4])

ax[0].set(xlabel=None)
ax[1].set(xlabel=None)
ax[2].set(xlabel=None)
ax[3].set(xlabel=None)
ax[2].set_ylabel("Pearson")
ax[3].set_ylabel("Spearman")
# ax[0,0].set_yscale('log')
# ax[0,1].set_yscale('log')
fig.set_facecolor('w')
plt.tight_layout()
fig.savefig("../results_softODE/compare_models_ratio.png", bbox_inches = "tight") 

# In[]
score_all = score_all_ori.copy()
score_all_diff = score_all_diff_ori.copy()   

score_all.loc[score_all["model"]!="CSN", "AUPRC Ratio (signed)"] = score_all.loc[score_all["model"]!="CSN", "AUPRC Ratio (pos)"].values
score_all.loc[score_all["model"]=="CSN", "AUPRC Ratio (signed)"] = score_all.loc[score_all["model"]=="CSN", "AUPRC Ratio (abs)"].values
score_all.loc[score_all["model"]!="CSN", "Early Precision Ratio (signed)"] = score_all.loc[score_all["model"]!="CSN", "Early Precision Ratio (pos)"].values
score_all.loc[score_all["model"]=="CSN", "Early Precision Ratio (signed)"] = score_all.loc[score_all["model"]=="CSN", "Early Precision Ratio (abs)"].values
score_all.loc[score_all["model"] != "CSN", "AUPRC (signed)"] = (score_all.loc[score_all["model"] != "CSN", "AUPRC (pos)"].values + score_all.loc[score_all["model"] != "CSN", "AUPRC (neg)"].values)/2
score_all.loc[score_all["model"] == "CSN", "AUPRC (signed)"] = score_all.loc[score_all["model"] == "CSN", "AUPRC (abs)"].values
score_all.loc[score_all["model"] != "CSN", "Early Precision (signed)"] = (score_all.loc[score_all["model"] != "CSN", "Early Precision (pos)"].values + score_all.loc[score_all["model"] != "CSN", "Early Precision (neg)"].values)/2
score_all.loc[score_all["model"] == "CSN", "Early Precision (signed)"] = score_all.loc[score_all["model"] == "CSN", "Early Precision (abs)"].values

score_all = score_all.loc[score_all["model"] != "CeSpGRN",:]
score_all.loc[score_all["model"] == "CeSpGRN-kt", "model"] = "CeSpGRN"
score_all.loc[score_all["model"] == "CeSpGRN-kt-TF", "model"] = "CeSpGRN-TF"
score_all = score_all[(score_all["model"] == "CeSpGRN-TF")|(score_all["model"] == "CeSpGRN")]

fig, ax = plt.subplots( figsize=(15.0, 5.0) , nrows=1, ncols=4) 
boxplot1 = sns.boxplot(data = score_all, x = "model", y = "Early Precision (signed)", ax = ax[0])
boxplot2 = sns.boxplot(data = score_all[score_all["model"] != "CSN"], x = "model", y = "AUPRC (signed)", ax = ax[1])
boxplot3 = sns.boxplot(data = score_all, x = "model", y = "pearson", ax = ax[2])
boxplot4 = sns.boxplot(data = score_all, x = "model", y = "spearman", ax = ax[3])
# add_median_labels(boxplot1.axes)
# add_median_labels(boxplot2.axes)
# add_median_labels(boxplot3.axes)
# add_median_labels(boxplot4.axes)
# ax[0].set_ylim([-1 , 1])
# ax[1].set_ylim([-1 , 1])
ax[0].set_xticklabels(boxplot1.get_xticklabels(), rotation = 90, fontsize = 15)
ax[1].set_xticklabels(boxplot2.get_xticklabels(), rotation = 90, fontsize = 15)
ax[2].set_xticklabels(boxplot3.get_xticklabels(), rotation = 90, fontsize = 15)
ax[3].set_xticklabels(boxplot4.get_xticklabels(), rotation = 90, fontsize = 15)

ax[2].set_ylim([-0.4,0.4])

ax[0].set(xlabel=None)
ax[1].set(xlabel=None)
ax[2].set(xlabel=None)
ax[3].set(xlabel=None)
ax[2].set_ylabel("Pearson")
ax[3].set_ylabel("Spearman")
# ax[0,0].set_yscale('log')
# ax[0,1].set_yscale('log')
fig.set_facecolor('w')
plt.tight_layout()
fig.savefig("../results_softODE/compare_models_tf.png", bbox_inches = "tight") 

fig, ax = plt.subplots( figsize=(15.0, 5.0) , nrows=1, ncols=4) 
boxplot1 = sns.boxplot(data = score_all, x = "model", y = "Early Precision Ratio (signed)", ax = ax[0])
boxplot2 = sns.boxplot(data = score_all[score_all["model"] != "CSN"], x = "model", y = "AUPRC Ratio (signed)", ax = ax[1])
boxplot3 = sns.boxplot(data = score_all, x = "model", y = "pearson", ax = ax[2])
boxplot4 = sns.boxplot(data = score_all, x = "model", y = "spearman", ax = ax[3])
# add_median_labels(boxplot1.axes)
# add_median_labels(boxplot2.axes)
# add_median_labels(boxplot3.axes)
# add_median_labels(boxplot4.axes)
# ax[0].set_ylim([-1 , 1])
# ax[1].set_ylim([-1 , 1])
ax[0].set_xticklabels(boxplot1.get_xticklabels(), rotation = 90, fontsize = 15)
ax[1].set_xticklabels(boxplot2.get_xticklabels(), rotation = 90, fontsize = 15)
ax[2].set_xticklabels(boxplot3.get_xticklabels(), rotation = 90, fontsize = 15)
ax[3].set_xticklabels(boxplot4.get_xticklabels(), rotation = 90, fontsize = 15)

ax[2].set_ylim([-0.4,0.4])

ax[0].set(xlabel=None)
ax[1].set(xlabel=None)
ax[2].set(xlabel=None)
ax[3].set(xlabel=None)
ax[2].set_ylabel("Pearson")
ax[3].set_ylabel("Spearman")
# ax[0,0].set_yscale('log')
# ax[0,1].set_yscale('log')
fig.set_facecolor('w')
plt.tight_layout()
fig.savefig("../results_softODE/compare_models_tf_ratio.png", bbox_inches = "tight")       


# In[]
plt.rcParams["font.size"] = 20
ntimes = 1000
ngenes = 20
umap_op = UMAP(n_components = 2, min_dist = 0.8, n_neighbors = 30, random_state = 0)
pca_op = PCA(n_components = 2)
for traj in ["linear", "bifurc"]:
    for seed in [0,1,2,3,4]:
        result_dir = "../results_softODE/" + traj + "_ngenes_" + str(ngenes) + "_ncell_" + str(ntimes) + "_seed_" + str(seed) + "/"
        data_dir = path + traj + "_ngenes_" + str(ngenes) + "_ncell_" + str(ntimes) + "_seed_" + str(seed) + "/"
        
        X = np.load(data_dir + "true_count.npy")
        gt_adj = np.load(data_dir + "GRNs.npy")
        sim_time = np.load(data_dir + "pseudotime.npy")
        sim_time = sim_time/np.max(sim_time)
        print("data loaded.")
    

        fig = plt.figure(figsize = (10,7))
        X_umap = umap_op.fit_transform(X)
        ax = fig.add_subplot()
        pic = ax.scatter(X_umap[:,0], X_umap[:,1], c = sim_time, s = 5)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        cbar = plt.colorbar(pic)

        fig.savefig(result_dir + "X_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_umap.png", bbox_inches = "tight")
        
        fig = plt.figure(figsize = (10,7))
        X_pca = pca_op.fit_transform(X)
        ax = fig.add_subplot()
        pic = ax.scatter(X_pca[:,0], X_pca[:,1], c = sim_time, s = 5)
        ax.set_xlabel("PCA1")
        ax.set_ylabel("PCA2")
        cbar = plt.colorbar(pic)
        fig.savefig(result_dir + "X_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_pca.png", bbox_inches = "tight")

        bandwidth = 10
        lamb = 0.001
        truncate_param = 30
        beta = 100

        # # CeSpGRN-kt
        # thetas = np.load(file = result_dir + "thetas_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_kt.npy")

        # fig = plt.figure(figsize = (10,7))
        # X_umap = umap_op.fit_transform(thetas.reshape(ntimes * nsample, -1))
        # ax = fig.add_subplot()
        # pic = ax.scatter(X_umap[:,0], X_umap[:,1], c = sim_time, s = 5)
        # ax.set_xlabel("UMAP1")
        # ax.set_ylabel("UMAP2")
        # cbar = plt.colorbar(pic)

        # fig.savefig(result_dir + "infer_G_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_umap.png", bbox_inches = "tight")
        
        # fig = plt.figure(figsize = (10,7))
        # X_pca = pca_op.fit_transform(thetas.reshape(ntimes * nsample, -1))
        # ax = fig.add_subplot()
        # pic = ax.scatter(X_pca[:,0], X_pca[:,1], c = sim_time, s = 5)
        # ax.set_xlabel("PCA1")
        # ax.set_ylabel("PCA2")
        # cbar = plt.colorbar(pic)
        # fig.savefig(result_dir + "infer_G_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_pca.png", bbox_inches = "tight")



# In[]

score_all = pd.DataFrame(columns = ["interval", "ngenes", "time", "model", "bandwidth", "truncate_param", "lambda", "nmse","kendall-tau", "pearson", "spearman", "cosine similarity", "AUPRC Ratio (pos)", "AUPRC Ratio (neg)", "AUPRC Ratio (abs)", "Early Precision Ratio (pos)", "Early Precision Ratio (neg)", "Early Precision Ratio (abs)"])
score_all_diff = pd.DataFrame(columns = ["interval", "ngenes", "time", "model", "bandwidth", "truncate_param", "lambda", "nmse","kendall-tau", "pearson", "spearman", "cosine similarity", "AUPRC Ratio (pos)", "AUPRC Ratio (neg)", "AUPRC Ratio (abs)", "Early Precision Ratio (pos)", "Early Precision Ratio (neg)", "Early Precision Ratio (abs)"])
ntimes = 1000
for interval in [20]:
    for (ngenes, ntfs) in [(20, 5)]:
        for seed in [0,1,2,3,4]:
            # bifurcating
            result_dir = "../results_softODE/bifurc_ngenes_" + str(ngenes) + "_ncell_" + str(ntimes) + "_seed_" + str(seed) + "/"
    
            score = pd.read_csv(result_dir + "score.csv", index_col = 0)
            score_diff = pd.read_csv(result_dir + "score_diff.csv", index_col = 0)

            score = score[score["model"] == "CeSpGRN"]
            score_diff = score_diff[score_diff["model"] == "CeSpGRN"]

            score_all = pd.concat([score_all, score], axis = 0)
            score_all_diff = pd.concat([score_all_diff, score_diff], axis = 0)


            # linear
            result_dir = "../results_softODE/linear_ngenes_" + str(ngenes) + "_ncell_" + str(ntimes) + "_seed_" + str(seed) + "/"
    
            score = pd.read_csv(result_dir + "score.csv", index_col = 0)
            score_diff = pd.read_csv(result_dir + "score_diff.csv", index_col = 0)

            score = score[score["model"] == "CeSpGRN"]
            score_diff = score_diff[score_diff["model"] == "CeSpGRN"]

            score_all = pd.concat([score_all, score], axis = 0)
            score_all_diff = pd.concat([score_all_diff, score_diff], axis = 0)


fig, ax = plt.subplots( figsize=(12.0, 7.0) , nrows=2, ncols=2) 
boxplot1 = sns.boxplot(data = score_all, x = "bandwidth", y = "pearson", hue = "truncate_param", ax = ax[0,0])
boxplot2 = sns.boxplot(data = score_all, x = "bandwidth", y = "spearman", hue = "truncate_param", ax = ax[0,1])
boxplot3 = sns.boxplot(data = score_all, x = "bandwidth", y = "AUPRC Ratio (pos)", hue = "truncate_param", ax = ax[1,0])
boxplot4 = sns.boxplot(data = score_all, x = "bandwidth", y = "AUPRC Ratio (neg)", hue = "truncate_param", ax = ax[1,1])
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
fig.savefig("../results_softODE/hyper-parameter.png", bbox_inches = "tight")



fig, ax = plt.subplots( figsize=(12.0, 7.0) , nrows=2, ncols=2) 
boxplot1 = sns.boxplot(data = score_all_diff, x = "bandwidth", y = "pearson", hue = "truncate_param", ax = ax[0,0])
boxplot2 = sns.boxplot(data = score_all_diff, x = "bandwidth", y = "spearman", hue = "truncate_param", ax = ax[0,1])
boxplot3 = sns.boxplot(data = score_all_diff, x = "bandwidth", y = "AUPRC Ratio (pos)", hue = "truncate_param", ax = ax[1,0])
boxplot4 = sns.boxplot(data = score_all_diff, x = "bandwidth", y = "AUPRC Ratio (neg)", hue = "truncate_param", ax = ax[1,1])
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
fig.savefig("../results_softODE/hyper-parameter-diff.png", bbox_inches = "tight")



score_all = pd.DataFrame(columns = ["interval", "ngenes", "time", "model", "bandwidth", "truncate_param", "lambda", "nmse","kendall-tau", "pearson", "spearman", "cosine similarity", "AUPRC Ratio (pos)", "AUPRC Ratio (neg)", "AUPRC Ratio (abs)", "Early Precision Ratio (pos)", "Early Precision Ratio (neg)", "Early Precision Ratio (abs)"])
score_all_diff = pd.DataFrame(columns = ["interval", "ngenes", "time", "model", "bandwidth", "truncate_param", "lambda", "nmse","kendall-tau", "pearson", "spearman", "cosine similarity", "AUPRC Ratio (pos)", "AUPRC Ratio (neg)", "AUPRC Ratio (abs)", "Early Precision Ratio (pos)", "Early Precision Ratio (neg)", "Early Precision Ratio (abs)"])
ntimes = 1000
for interval in [20]:
    for (ngenes, ntfs) in [(20, 5)]:
        for seed in [0,1,2,3,4]:
            # bifurcating
            result_dir = "../results_softODE/bifurc_ngenes_" + str(ngenes) + "_ncell_" + str(ntimes) + "_seed_" + str(seed) + "/"
    
            score = pd.read_csv(result_dir + "score.csv", index_col = 0)
            score_diff = pd.read_csv(result_dir + "score_diff.csv", index_col = 0)

            score = score[score["model"] == "CeSpGRN-TF"]
            score_diff = score_diff[score_diff["model"] == "CeSpGRN-TF"]

            score_all = pd.concat([score_all, score], axis = 0)
            score_all_diff = pd.concat([score_all_diff, score_diff], axis = 0)


            # linear
            result_dir = "../results_softODE/linear_ngenes_" + str(ngenes) + "_ncell_" + str(ntimes) + "_seed_" + str(seed) + "/"
    
            score = pd.read_csv(result_dir + "score.csv", index_col = 0)
            score_diff = pd.read_csv(result_dir + "score_diff.csv", index_col = 0)

            score = score[score["model"] == "CeSpGRN-TF"]
            score_diff = score_diff[score_diff["model"] == "CeSpGRN-TF"]

            score_all = pd.concat([score_all, score], axis = 0)
            score_all_diff = pd.concat([score_all_diff, score_diff], axis = 0)


fig, ax = plt.subplots( figsize=(12.0, 7.0) , nrows=2, ncols=2) 
boxplot1 = sns.boxplot(data = score_all, x = "bandwidth", y = "pearson", hue = "truncate_param", ax = ax[0,0])
boxplot2 = sns.boxplot(data = score_all, x = "bandwidth", y = "spearman", hue = "truncate_param", ax = ax[0,1])
boxplot3 = sns.boxplot(data = score_all, x = "bandwidth", y = "AUPRC Ratio (pos)", hue = "truncate_param", ax = ax[1,0])
boxplot4 = sns.boxplot(data = score_all, x = "bandwidth", y = "AUPRC Ratio (neg)", hue = "truncate_param", ax = ax[1,1])
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
fig.savefig("../results_softODE/hyper-parameter-tf.png", bbox_inches = "tight")



fig, ax = plt.subplots( figsize=(12.0, 7.0) , nrows=2, ncols=2) 
boxplot1 = sns.boxplot(data = score_all_diff, x = "bandwidth", y = "pearson", hue = "truncate_param", ax = ax[0,0])
boxplot2 = sns.boxplot(data = score_all_diff, x = "bandwidth", y = "spearman", hue = "truncate_param", ax = ax[0,1])
boxplot3 = sns.boxplot(data = score_all_diff, x = "bandwidth", y = "AUPRC Ratio (pos)", hue = "truncate_param", ax = ax[1,0])
boxplot4 = sns.boxplot(data = score_all_diff, x = "bandwidth", y = "AUPRC Ratio (neg)", hue = "truncate_param", ax = ax[1,1])
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
fig.savefig("../results_softODE/hyper-parameter-diff-tf.png", bbox_inches = "tight")


# %%
