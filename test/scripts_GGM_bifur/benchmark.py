# In[0]
from math import trunc
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

plt.rcParams["font.size"] = 15

from multiprocessing import Pool, cpu_count
import time

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
    diff = setting["diff"]

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

    AUPRC_ratio_pos = AUPRC_pos/AUPRC_pos_rand
    AUPRC_ratio_neg = AUPRC_neg/AUPRC_neg_rand
    AUPRC_ratio_abs = AUPRC/AUPRC_rand
    Eprec_ratio_pos = Eprec_pos/(Eprec_pos_rand + 1e-12)
    Eprec_ratio_neg = Eprec_neg/(Eprec_neg_rand + 1e-12)
    Eprec_ratio_abs = Eprec/(Eprec_rand + 1e-12)

    if ((model == "CSN")|(model == "GENIE3")|(model == "GENIE3-Dyn")) & (diff == False):
        AUPRC_signed = AUPRC
        Eprec_signed = Eprec
        AUPRC_ratio_signed = AUPRC_ratio_abs
        Eprec_ratio_signed = Eprec_ratio_abs
    else:
        AUPRC_signed = (AUPRC_pos + AUPRC_neg)/2
        Eprec_signed = (Eprec_pos + Eprec_neg)/2
        AUPRC_ratio_signed = (AUPRC_ratio_pos + AUPRC_ratio_neg)/2
        Eprec_ratio_signed = (Eprec_ratio_pos + Eprec_ratio_neg)/2

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
                        "AUPRC (signed)": AUPRC_signed,
                        "Early Precision (pos)": Eprec_pos,
                        "Early Precision (neg)": Eprec_neg,
                        "Early Precision (abs)":Eprec,
                        "Early Precision (signed)": Eprec_signed,
                        "AUPRC random (pos)": AUPRC_pos_rand,
                        "AUPRC random (neg)": AUPRC_neg_rand,
                        "AUPRC random (abs)": AUPRC_rand,
                        "Early Precision random (pos)": Eprec_pos_rand,
                        "Early Precision random (neg)": Eprec_neg_rand,
                        "Early Precision random (abs)":Eprec_rand,
                        "AUPRC Ratio (pos)": AUPRC_ratio_pos,
                        "AUPRC Ratio (neg)": AUPRC_ratio_neg,
                        "AUPRC Ratio (abs)": AUPRC_ratio_abs,
                        "AUPRC Ratio (signed)": AUPRC_ratio_signed,
                        "Early Precision Ratio (pos)": Eprec_ratio_pos,
                        "Early Precision Ratio (neg)": Eprec_ratio_neg,
                        "Early Precision Ratio (abs)":Eprec_ratio_abs,
                        "Early Precision Ratio (signed)":Eprec_ratio_signed,
                        "time":time,
                        "model": model,
                        "bandwidth": bandwidth,
                        "truncate_param":truncate_param,
                        "lambda":lamb}, ignore_index=True)  

    return score

def calc_scores_para(thetas_inf, thetas_gt, interval, model, bandwidth = 0, truncate_param = 0, lamb = 0, diff = False):
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
            "time": time,
            "diff": diff
        })


    pool = Pool(24) 
    scores = pool.map(calc_scores_static, [x for x in settings])
    pool.close()
    pool.join()
    scores.sort(key = lambda x: x["time"].values[0])
    scores = pd.concat(scores, axis = 0, ignore_index = True)
    return scores

ntimes = 1000
nsample = 1
path = "../../data/GGM_bifurcate/"
result_path = "../results_GGM/"

umap_op = UMAP(n_components = 2, min_dist = 0.8, n_neighbors = 30, random_state = 0)

# In[] benchmark accuracy
print("------------------------------------------------------------------")
print("benchmark accuracy")
print("------------------------------------------------------------------")

def summarize_scores(setting):
    ntimes = setting["ntimes"]
    interval = setting["interval"]
    ngenes = setting["ngenes"]
    ntfs = setting["ntfs"]
    seed = setting["seed"]

    print("ntimes: " + str(ntimes) + ", interval: " + str(interval) + ", ngenes: " + str(ngenes) + ", seed: " + str(seed) + ", initial graph: sergio\n")
    # scores = pd.DataFrame(columns = ["interval", "ngenes", "time", "model", "bandwidth", "truncate_param", "lambda", "nmse","kendall-tau", "pearson", \
    #     "spearman", "cosine similarity", "AUPRC (pos)", "AUPRC (neg)", "AUPRC (abs)", "Early Precision (pos)", "Early Precision (neg)", "Early Precision (abs)", "AUPRC random (pos)",\
    #         "AUPRC random (neg)", "AUPRC random (abs)", "Early Precision random (pos)", "Early Precision random (neg)","Early Precision random (abs)","AUPRC Ratio (pos)", \
    #             "AUPRC Ratio (neg)", "AUPRC Ratio (abs)", "Early Precision Ratio (pos)", "Early Precision Ratio (neg)", "Early Precision Ratio (abs)"])
    result_dir = "../results_GGM/bifur_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "_" + str(seed) + "_sergio/"
    
    # the data smapled from GGM is zero-mean
    X = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "_" + str(seed) + "_sergio/expr.npy")
    gt_adj = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "_" + str(seed) + "_sergio/Gs.npy")
    sim_time = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "_" + str(seed) + "_sergio/sim_time.npy")
    print("data loaded.")
    
    # genie3            
    thetas = np.load(file = result_dir + "theta_genie.npy")
    # scores = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "GENIE3")
    scores = calc_scores_para(thetas_inf = thetas, thetas_gt = np.abs(gt_adj), interval = interval, model = "GENIE3")
    # first one don't need to concat
    # scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    print("GENIE3")
    
    # # genie3-tf
    # thetas = np.load(file = result_dir + "theta_genie_tf.npy")
    # score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "GENIE3-TF")
    # scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    # print("genie3-tf.")

    # genie3-dyn
    thetas = np.load(file = result_dir + "theta_genie_dyn.npy")
    # score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "GENIE3-Dyn")
    score = calc_scores_para(thetas_inf = thetas, thetas_gt = np.abs(gt_adj), interval = interval, model = "GENIE3-Dyn")
    scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    print("GENIE3-Dyn")
    
    
    # # genie3-dyn-tf 
    # thetas = np.load(file = result_dir + "theta_genie_dyn_tf.npy")
    # score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "GENIE3-Dyn-TF")
    # scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    # print("genie3-dyn-tf.")


    # SCODE (True T)
    thetas = np.load(file = result_dir + "theta_scode.npy")
    # score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "SCODE")
    score = calc_scores_para(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "SCODE")
    scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    print("SCODE")

    
    # SCODE-DYN (True T)
    thetas = np.load(file = result_dir + "theta_scode_dyn.npy")
    # score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "SCODE-Dyn")
    score = calc_scores_para(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "SCODE-Dyn")
    scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    print("SCODE-Dyn")

    # CSN          
    thetas = np.load(file = result_dir + "theta_CSN.npy")
    # score = calc_scores(thetas_inf = thetas, thetas_gt = np.abs(gt_adj), interval = interval, model = "CSN")
    score = calc_scores_para(thetas_inf = thetas, thetas_gt = np.abs(gt_adj), interval = interval, model = "CSN")
    scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    print("CSN") 

    # admm, hyper-parameter
    for bandwidth in [0.1, 1, 10]:
        for truncate_param in [15, 30, 100]:
            for lamb in [0.001, 0.01, 0.1]:
                data = str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param)
                print(data)
                thetas = np.load(file = result_dir + "thetas_" + data + "_kt.npy")
                # score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "CeSpGRN", bandwidth = bandwidth, truncate_param = truncate_param, lamb = lamb)
                score = calc_scores_para(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "CeSpGRN", bandwidth = bandwidth, truncate_param = truncate_param, lamb = lamb)
                scores = pd.concat([scores, score], axis = 0, ignore_index = True)

                # fig = plt.figure(figsize = (10,7))
                # X_umap = umap_op.fit_transform(thetas.reshape(ntimes * nsample, -1))
                # ax = fig.add_subplot()
                # ax.scatter(X_umap[:,0], X_umap[:,1], c = sim_time, s = 5)
                # fig.savefig(result_dir + "infer_G_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_0.png", bbox_inches = "tight")
    print("")
    scores.to_csv(result_dir + "score.csv")
    
if __name__ == "__main__":
    # assert len(sys.argv) == 2
    # seeds = [eval(sys.argv[1])]
    # seeds = [0,1,2]
    # settings = []
    # for interval in [5, 25]:
    #     for (ngenes, ntfs) in [(50, 20), (200,20)]:
    #         for seed in seeds:
    #             settings.append({
    #                 "ntimes": ntimes,
    #                 "interval": interval,
    #                 "ngenes": ngenes,
    #                 "ntfs": ntfs,
    #                 "seed": seed
    #             })


    # pool = Pool(6) 
    # pool.map(summarize_scores, [x for x in settings])
    # pool.close()
    # pool.join()

    intervals = [5, 25]
    ngenes = [50, 200]
    seeds = [0, 1, 2]

    setting = {
        "ntimes": ntimes,
        "interval": intervals[eval(sys.argv[1])],
        "ngenes": ngenes[eval(sys.argv[2])],
        "ntfs": 20,
        "seed": seeds[eval(sys.argv[3])]
    }
    summarize_scores(setting)


# In[] 
print("------------------------------------------------------------------")
print("benchmark differences")
print("------------------------------------------------------------------")
ntimes = 1000
nsample = 1
def summarize_scores_diff(setting):
    ntimes = setting["ntimes"]
    interval = setting["interval"]
    ngenes = setting["ngenes"]
    ntfs = setting["ntfs"]
    seed = setting["seed"]

    print("ntimes: " + str(ntimes) + ", interval: " + str(interval) + ", ngenes: " + str(ngenes) + ", seed: " + str(seed) + ", initial graph: sergio")
    # scores = pd.DataFrame(columns = ["interval", "ngenes", "time", "model", "bandwidth", "truncate_param", "lambda", "nmse","kendall-tau", "pearson", \
    #     "spearman", "cosine similarity", "AUPRC (pos)", "AUPRC (neg)", "AUPRC (abs)", "Early Precision (pos)", "Early Precision (neg)", "Early Precision (abs)", "AUPRC random (pos)",\
    #         "AUPRC random (neg)", "AUPRC random (abs)", "Early Precision random (pos)", "Early Precision random (neg)","Early Precision random (abs)","AUPRC Ratio (pos)", \
    #             "AUPRC Ratio (neg)", "AUPRC Ratio (abs)", "Early Precision Ratio (pos)", "Early Precision Ratio (neg)", "Early Precision Ratio (abs)"])
    result_dir = result_path + "bifur_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "_" + str(seed) + "_sergio/"
    
    # the data smapled from GGM is zero-mean
    X = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "_" + str(seed) + "_sergio/expr.npy")
    gt_adj = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "_" + str(seed) + "_sergio/Gs.npy")
    gt_adj_abs = np.abs(gt_adj)
    gt_adj_diff = np.concatenate((gt_adj[400:600,:,:] - gt_adj[:200,:,:], gt_adj[800:,:,:] - gt_adj[:200,:,:]), axis = 0)
    gt_adj_diff_abs = np.concatenate((gt_adj_abs[400:600,:,:] - gt_adj_abs[:200,:,:], gt_adj_abs[800:,:,:] - gt_adj_abs[:200,:,:]), axis = 0)
    sim_time = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "_" + str(seed) + "_sergio/sim_time.npy")

    # # genie3            
    # thetas = np.load(file = result_dir + "theta_genie.npy")
    # score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "GENIE3")
    # scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    # print("genie3.")

    # # genie3-tf
    # thetas = np.load(file = result_dir + "theta_genie_tf.npy")
    # score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "GENIE3-TF")
    # scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    # print("genie3-tf.")

    # genie3-dyn, only infer positive
    thetas = np.load(file = result_dir + "theta_genie_dyn.npy")
    thetas_diff = np.concatenate((thetas[400:600,:,:] - thetas[:200,:,:], thetas[800:,:,:] - thetas[:200,:,:]), axis = 0)
    # scores = calc_scores(thetas_inf = thetas_diff, thetas_gt = gt_adj_diff, interval = interval, model = "GENIE3-Dyn")
    scores = calc_scores_para(thetas_inf = thetas_diff, thetas_gt = gt_adj_diff_abs, interval = interval, model = "GENIE3-Dyn", diff = True)
    # scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    print("GENIE3-Dyn")
    
    
    # # genie3-dyn-tf 
    # thetas = np.load(file = result_dir + "theta_genie_dyn_tf.npy")
    # score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "GENIE3-Dyn-TF")
    # scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    # print("genie3-dyn-tf.")


    # # SCODE (True T)
    # thetas = np.load(file = result_dir + "theta_scode.npy")
    # score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "SCODE")
    # scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    # print("SCODE (True T)")

    
    # SCODE-DYN (True T)
    thetas = np.load(file = result_dir + "theta_scode_dyn.npy")
    thetas_diff = np.concatenate((thetas[400:600,:,:] - thetas[:200,:,:], thetas[800:,:,:] - thetas[:200,:,:]), axis = 0)
    # score = calc_scores(thetas_inf = thetas_diff, thetas_gt = gt_adj_diff, interval = interval, model = "SCODE-Dyn")
    score = calc_scores_para(thetas_inf = thetas_diff, thetas_gt = gt_adj_diff, interval = interval, model = "SCODE-Dyn", diff = True)
    scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    print("SCODE-Dyn")

    # CSN
    thetas = np.load(file = result_dir + "theta_CSN.npy")
    thetas_diff = np.concatenate((thetas[400:600,:,:] - thetas[:200,:,:], thetas[800:,:,:] - thetas[:200,:,:]), axis = 0)
    # score = calc_scores(thetas_inf = thetas_diff, thetas_gt = np.abs(gt_adj_diff), interval = interval, model = "CSN")
    score = calc_scores_para(thetas_inf = thetas_diff, thetas_gt = gt_adj_diff_abs, interval = interval, model = "CSN", diff = True)
    scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    print("CSN")

    # admm, hyper-parameter
    for bandwidth in [0.1, 1, 10]:
        for truncate_param in [15, 30, 100]:
            for lamb in [0.001, 0.01, 0.1]:
                data = str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param)
                print(data)
                thetas = np.load(file = result_dir + "thetas_" + data + "_kt.npy")
                thetas_diff = np.concatenate((thetas[400:600,:,:] - thetas[:200,:,:], thetas[800:,:,:] - thetas[:200,:,:]), axis = 0)
                # score = calc_scores(thetas_inf = thetas_diff, thetas_gt = gt_adj_diff, interval = interval, model = "CeSpGRN", bandwidth = bandwidth, truncate_param = truncate_param, lamb = lamb)
                score = calc_scores_para(thetas_inf = thetas_diff, thetas_gt = gt_adj_diff, interval = interval, model = "CeSpGRN", bandwidth = bandwidth, truncate_param = truncate_param, lamb = lamb, diff = True)
                scores = pd.concat([scores, score], axis = 0, ignore_index = True)

                # fig = plt.figure(figsize = (10,7))
                # X_umap = umap_op.fit_transform(thetas.reshape(ntimes * nsample, -1))
                # ax = fig.add_subplot()
                # ax.scatter(X_umap[:,0], X_umap[:,1], c = sim_time, s = 5)
                # fig.savefig(result_dir + "infer_G_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_0.png", bbox_inches = "tight")
    print("")

    # save results
    scores.to_csv(result_dir + "score_diff.csv")


if __name__ == "__main__":
    # assert len(sys.argv) == 2
    # seeds = [eval(sys.argv[1])]
    # seeds = [0,1,2]
    # settings = []
    # for interval in [5, 25]:
    #     for (ngenes, ntfs) in [(50, 20), (200, 20)]:
    #         for seed in seeds:
    #             settings.append({
    #                 "ntimes": ntimes,
    #                 "interval": interval,
    #                 "ngenes": ngenes,
    #                 "ntfs": ntfs,
    #                 "seed": seed
    #             })


    # pool = Pool(6) 
    # pool.map(summarize_scores_diff, [x for x in settings])
    # pool.close()
    # pool.join()

    intervals = [5, 25]
    ngenes = [50, 200]
    seeds = [0, 1, 2]

    setting = {
        "ntimes": ntimes,
        "interval": intervals[eval(sys.argv[1])],
        "ngenes": ngenes[eval(sys.argv[2])],
        "ntfs": 20,
        "seed": seeds[eval(sys.argv[3])]
    }
    summarize_scores_diff(setting)

# In[] summarize the mean result in csv file

ntimes = 1000
nsample = 1
for interval in [5, 25]:
    for (ngenes, ntfs) in [(50, 20), (200, 20)]:
        for seed in range(3):
            print("bifur_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "_" + str(seed) + "_sergio")
            result_dir = result_path + "bifur_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "_" + str(seed) + "_sergio/"
            score = pd.read_csv(result_dir + "score.csv", index_col = 0)
            mean_score = score.groupby(by = ["model", "bandwidth", "truncate_param", "lambda"]).mean()
            mean_score = mean_score.drop(["time"], axis = 1)
            mean_score.to_csv(result_dir + "mean_score.csv")
            # display(mean_score)

print("\ndifferences\n")

for interval in [5, 25]:
    for (ngenes, ntfs) in [(50, 20), (200, 20)]: 
        for seed in range(3):
            print("bifur_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "_" + str(seed) + "_sergio")
            result_dir = result_path + "bifur_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "_" + str(seed) + "_sergio/"
            score = pd.read_csv(result_dir + "score_diff.csv", index_col = 0)
            mean_score = score.groupby(by = ["model", "bandwidth", "truncate_param", "lambda"]).mean()
            mean_score = mean_score.drop(["time"], axis = 1)
            mean_score.to_csv(result_dir + "mean_score_diff.csv")
            # display(mean_score)




#In[] 
# -------------------------------------------------------------------------------------------
# 
# Test on hyper-parameter
# 
# -------------------------------------------------------------------------------------------

ntimes = 1000

# How the bandwidth and truncate parameter is affected by the interval
for interval in [5, 25]:
    score_interval = pd.DataFrame(columns = ["interval", "ngenes", "time", "model", "bandwidth", "truncate_param", "lambda", "nmse","kendall-tau", "pearson", "spearman", "cosine similarity", "AUPRC Ratio (pos)", "AUPRC Ratio (neg)", "AUPRC Ratio (abs)", "Early Precision Ratio (pos)", "Early Precision Ratio (neg)", "Early Precision Ratio (abs)"])
    score_interval_diff = pd.DataFrame(columns = ["interval", "ngenes", "time", "model", "bandwidth", "truncate_param", "lambda", "nmse","kendall-tau", "pearson", "spearman", "cosine similarity", "AUPRC Ratio (pos)", "AUPRC Ratio (neg)", "AUPRC Ratio (abs)", "Early Precision Ratio (pos)", "Early Precision Ratio (neg)", "Early Precision Ratio (abs)"])

    for (ngenes, ntfs) in [(50, 20), (200, 20)]:
        for seed in range(3):
            result_dir = result_path + "bifur_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "_" + str(seed) + "_sergio/"

            score = pd.read_csv(result_dir + "score.csv", index_col = 0)
            score_diff = pd.read_csv(result_dir + "score_diff.csv", index_col = 0)

            # score for admm
            score = score[score["model"] == "CeSpGRN"]
            score_diff = score_diff[score_diff["model"] == "CeSpGRN"]
            score_interval = pd.concat([score_interval, score], axis = 0)
            score_interval_diff = pd.concat([score_interval_diff, score_diff], axis = 0)

    # Plot including lambda
    fig, big_axes = plt.subplots( figsize=(15.0, 10.0) , nrows=3, ncols=1, sharey=False) 
    lambdas = [0.001, 0.01, 0.1]
    for row, big_ax in enumerate(big_axes, start=1):
        big_ax.set_title("Lambda {:.4f}".format(lambdas[row - 1]), fontsize=20)

        # Turn off axis lines and ticks of the big subplot 
        # obs alpha is 0 in RGBA string!
        big_ax.tick_params(labelcolor=(1.,1.,1., 0.0), top='off', bottom='off', left='off', right='off')
        # removes the white frame
        big_ax._frameon = False

    for i in range(1,7):
        ax = fig.add_subplot(3,2,i)
        if i%2 == 1:
            box_plot = sns.boxplot(data = score_interval[score_interval["lambda"] == lambdas[(i-1)//2]], x = "bandwidth", y = "pearson", hue = "truncate_param", ax = ax)
            # add_median_labels(box_plot.axes)
            ax.set_xlabel("Bandwidth")
            ax.set_ylabel("Pearson")
            ax.get_legend().remove()
        else:
            box_plot = sns.boxplot(data = score_interval[score_interval["lambda"] == lambdas[(i-1)//2]], x = "bandwidth", y = "spearman", hue = "truncate_param", ax = ax)
            # add_median_labels(box_plot.axes)            
            ax.set_xlabel("Bandwidth")
            ax.set_ylabel("Spearman")
            ax.legend(loc="upper left", prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.01, 1), title = "Neighborhood size")
    
    fig.set_facecolor('w')
    fig.suptitle("score of edge detection, change interval: " + str(interval), fontsize = 25)
    plt.tight_layout()
    fig.savefig(result_path + "CeSpGRN_" + str(interval) + ".png", bbox_inches = "tight")


    # # Plot including lambda
    # fig, axes = plt.subplots( figsize=(20.0, 5.0) , nrows=1, ncols=3) 
    # lambdas = [0.001, 0.01, 0.1]
    # for col, ax in enumerate(axes, start=1):
    #     ax.set_title("Lambda: {:.4f}".format(lambdas[col - 1]), fontsize=20)
    #     box_plot = sns.boxplot(data = score_interval_diff[score_interval_diff["lambda"] == lambdas[col - 1]], x = "bandwidth", y = "AUPRC Ratio (signed)", hue = "truncate_param", ax = ax)
    #     ax.set_yscale('log')
    #     ax.set_yticks([0.5, 1, 1.5, 2, 10, 50, 100])
    #     ax.set_ylabel("AUPRC ratio\n (signed)")
    #     ax.set_xlabel("Bandwidth")
    #     # add_median_labels(box_plot.axes)
    #     if col < 3:
    #         ax.get_legend().remove()
    #     else:
    #         ax.legend(loc="upper left", prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.01, 1), title = "Neighborhood size")


    # fig.set_facecolor('w')
    # fig.suptitle("score of changing edges detection, change interval: " + str(interval), fontsize = 25)
    # plt.tight_layout()
    # fig.savefig(result_path + "CeSpGRN_diff_" + str(interval) + ".png", bbox_inches = "tight")

# In[] 
# ---------------------------------------------------------------------------------
#
#  Comparison between models, edge detection score
#
# ---------------------------------------------------------------------------------
ntimes = 1000
score_all = pd.DataFrame(columns = ["interval", "ngenes", "nmse","kendall-tau", "pearson", "spearman", "cosine similarity", "time", "model", "bandwidth", "truncate_param", "lambda"])
for interval in [5, 25]:
    for (ngenes, ntfs) in [(50, 20)]:
        for seed in range(3):
            result_dir = "../results_GGM/bifur_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "_" + str(seed) + "_sergio/"
            score = pd.read_csv(result_dir + "score.csv", index_col = 0)       
            score_all = pd.concat([score_all, score], axis = 0)


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
ax[0].set(xlabel=None)
ax[1].set(xlabel=None)
ax[2].set(xlabel=None)
ax[3].set(xlabel=None)
ax[2].set_ylabel("Pearson")
ax[3].set_ylabel("Spearman")
# ax[0,0].set_yscale('log')
# ax[0,1].set_yscale('log')
fig.set_facecolor('w')
fig.suptitle("Score of edge detection, number of genes 50", fontsize = 25)
plt.tight_layout()
fig.savefig(result_path + "compare_models_50.png", bbox_inches = "tight") 



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
ax[0].set(xlabel=None)
ax[1].set(xlabel=None)
ax[2].set(xlabel=None)
ax[3].set(xlabel=None)
ax[2].set_ylabel("Pearson")
ax[3].set_ylabel("Spearman")
ax[0].set_yscale('log')
ax[1].set_yscale('log')
fig.set_facecolor('w')
fig.suptitle("Score of edge detection, number of genes 50", fontsize = 25)
plt.tight_layout()
fig.savefig(result_path + "compare_models_50_ratio.png", bbox_inches = "tight") 



ntimes = 1000
score_all = pd.DataFrame(columns = ["interval", "ngenes", "nmse","kendall-tau", "pearson", "spearman", "cosine similarity", "time", "model", "bandwidth", "truncate_param", "lambda"])
for interval in [5, 25]:
    for (ngenes, ntfs) in [(200, 20)]:
        for seed in range(3):
            result_dir = "../results_GGM/bifur_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "_" + str(seed) + "_sergio/"
            score = pd.read_csv(result_dir + "score.csv", index_col = 0)       
            score_all = pd.concat([score_all, score], axis = 0)


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

ax[2].set_yticks([-0.3, 0.0, 0.3, 0.6, 0.9])

ax[0].set(xlabel=None)
ax[1].set(xlabel=None)
ax[2].set(xlabel=None)
ax[3].set(xlabel=None)
ax[2].set_ylabel("Pearson")
ax[3].set_ylabel("Spearman")
# ax[0,0].set_yscale('log')
# ax[0,1].set_yscale('log')
fig.set_facecolor('w')
fig.suptitle("Score of edge detection, number of genes 200", fontsize = 25)
plt.tight_layout()
fig.savefig(result_path + "compare_models_200.png", bbox_inches = "tight") 


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
ax[0].set(xlabel=None)
ax[1].set(xlabel=None)
ax[2].set(xlabel=None)
ax[3].set(xlabel=None)
ax[2].set_ylabel("Pearson")
ax[3].set_ylabel("Spearman")
ax[0].set_yscale('log')
ax[1].set_yscale('log')
fig.set_facecolor('w')
fig.suptitle("Score of edge detection, number of genes 200", fontsize = 25)
plt.tight_layout()
fig.savefig(result_path + "compare_models_200_ratio.png", bbox_inches = "tight") 

# In[]
# ---------------------------------------------------------------------------------
#
#  Comparison between models, change detection score
#
# ---------------------------------------------------------------------------------

ntimes = 1000
for interval in [5, 25]:
    score_all_diff = pd.DataFrame(columns = ["interval", "ngenes", "nmse","kendall-tau", "pearson", "spearman", "cosine similarity", "time", "model", "bandwidth", "truncate_param", "lambda"])
    for (ngenes, ntfs) in [(50, 20)]:
        for seed in range(3):
            result_dir = "../results_GGM/bifur_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "_" + str(seed) + "_sergio/"
            score_diff = pd.read_csv(result_dir + "score_diff.csv", index_col = 0)       
            score_all_diff = pd.concat([score_all_diff, score_diff], axis = 0)

    fig, ax = plt.subplots( figsize=(12.0, 4.0) , nrows=1, ncols=4) 
    boxplot1 = sns.boxplot(data = score_all_diff, x = "model", y = "Early Precision (signed)", ax = ax[0])
    boxplot2 = sns.boxplot(data = score_all_diff[score_all_diff["model"] != "CSN"], x = "model", y = "AUPRC (signed)", ax = ax[1])
    boxplot3 = sns.boxplot(data = score_all_diff, x = "model", y = "pearson", ax = ax[2])
    boxplot4 = sns.boxplot(data = score_all_diff, x = "model", y = "spearman", ax = ax[3])
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

    ax[1].set_yticks([0.0, 0.1, 0.2, 0.3])

    ax[0].set(xlabel=None)
    ax[1].set(xlabel=None)
    ax[2].set(xlabel=None)
    ax[3].set(xlabel=None)
    ax[2].set_ylabel("Pearson")
    ax[3].set_ylabel("Spearman")
    # ax[0,0].set_yscale('log')
    # ax[0,1].set_yscale('log')
    fig.set_facecolor('w')
    # fig.suptitle("Score of changing edge detection, number of genes 50, change interval: " + str(interval), fontsize = 25)
    plt.tight_layout()
    fig.savefig(result_path + "compare_models_diff_" + str(interval) + "_50.png", bbox_inches = "tight") 


ntimes = 1000
for interval in [5, 25]:
    score_all_diff = pd.DataFrame(columns = ["interval", "ngenes", "nmse","kendall-tau", "pearson", "spearman", "cosine similarity", "time", "model", "bandwidth", "truncate_param", "lambda"])
    for (ngenes, ntfs) in [(200, 20)]:
        for seed in range(3):
            result_dir = "../results_GGM/bifur_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "_" + str(seed) + "_sergio/"
            score_diff = pd.read_csv(result_dir + "score_diff.csv", index_col = 0)       
            score_all_diff = pd.concat([score_all_diff, score_diff], axis = 0)


    fig, ax = plt.subplots( figsize=(12.0, 4.0) , nrows=1, ncols=4) 
    boxplot1 = sns.boxplot(data = score_all_diff, x = "model", y = "Early Precision (signed)", ax = ax[0])
    boxplot2 = sns.boxplot(data = score_all_diff[score_all_diff["model"] != "CSN"], x = "model", y = "AUPRC (signed)", ax = ax[1])
    boxplot3 = sns.boxplot(data = score_all_diff, x = "model", y = "pearson", ax = ax[2])
    boxplot4 = sns.boxplot(data = score_all_diff, x = "model", y = "spearman", ax = ax[3])
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

    ax[0].set_yticks([0.00, 0.05, 0.10])

    ax[0].set(xlabel=None)
    ax[1].set(xlabel=None)
    ax[2].set(xlabel=None)
    ax[3].set(xlabel=None)
    ax[2].set_ylabel("Pearson")
    ax[3].set_ylabel("Spearman")
    # ax[0,0].set_yscale('log')
    # ax[0,1].set_yscale('log')
    fig.set_facecolor('w')
    # fig.suptitle("Score of changing edge detection, number of genes 200, change interval: " + str(interval), fontsize = 25)
    plt.tight_layout()
    fig.savefig(result_path + "compare_models_diff_" + str(interval) + "_200.png", bbox_inches = "tight") 





# In[] Find time points and hyper-parameter that produce the highest spearman/AUPRC/Early Precision score
interval = 5
ngenes = 50
ntfs = 20
score_all_diff = pd.DataFrame(columns = ["interval", "ngenes", "nmse","kendall-tau", "pearson", "spearman", "cosine similarity", "time", "model", "bandwidth", "truncate_param", "lambda"])
for seed in range(3):
    result_dir = "../results_GGM/bifur_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "_" + str(seed) + "_sergio/"
    score_diff = pd.read_csv(result_dir + "score_diff.csv", index_col = 0)       
    score_all_diff = pd.concat([score_all_diff, score_diff], axis = 0)

max_auprc = score_all_diff["AUPRC (signed)"].argmax()
max_ep = score_all_diff["Early Precision (signed)"].argmax()
max_sp = score_all_diff["spearman"].argmax()
# bandwidth: 1.0, truncate_param: 15, lambda: 0.1
display(score_all_diff.iloc[max_auprc,:])
# bandwidth: 1.0, truncate_param: 15, lambda: 0.01
display(score_all_diff.iloc[max_ep,:])
# bandwidth: 0.1, truncate_param: 15, lambda: 0.01
display(score_all_diff.iloc[max_sp,:])


# In[]
# read in one graph
ntimes = 1000
interval = 5
seed = 0
ngenes = 50
result_path = "../results_GGM/"
path = "../../data/GGM_bifurcate/"

result_dir = result_path + "bifur_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "_" + str(seed) + "_sergio/"
bandwidth = 1
truncate_param = 15
lamb = 0.1
data = str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param)
thetas_gt = np.load(file = path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "_" + str(seed) + "_sergio/Gs.npy")
sim_time = np.load(file = path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "_" + str(seed) + "_sergio/sim_time.npy")
sim_time = sim_time/np.max(sim_time)
thetas_inf = np.load(file = result_dir + "thetas_" + data + "_kt.npy")
# thetas_gt = thetas_gt/np.max(thetas_gt)
# thetas_inf = thetas_inf/np.max(thetas_inf)


# In[]
def draw_graph(theta, threshold, ax):
    import networkx as nx 
    theta_norm = theta.copy()
    np.fill_diagonal(theta_norm, 0)
    theta_norm = theta/np.max(np.abs(theta))

    theta_bin = (theta_norm > threshold)|(theta_norm < - threshold)
    print("number of edges: {:d}".format(np.sum(theta_bin)))
    G = nx.Graph()
    G.add_nodes_from(np.arange(theta_norm.shape[0]))
    for row in range(theta_norm.shape[0]):
        for col in range(theta_norm.shape[1]):
            if (theta_bin[row, col] != 0) & (row != col):
                if theta_norm[row, col] > 0:
                    G.add_edge(row, col, color = 'blue', weight = 5 * np.abs(theta_norm[row, col]))
                else:
                    G.add_edge(row, col, color = 'red', weight = 5 * np.abs(theta_norm[row, col]))

    # print(G.edges)
    # node will be integer correspond to the row/column index
    # G = nx.from_numpy_array(theta_bin)
    pos = nx.circular_layout(G)
    colors = [G[u][v]['color'] for u,v in G.edges]
    weights = [G[u][v]['weight'] for u,v in G.edges]
    nx.draw(G, pos = pos, ax = ax, edge_color=colors, width=weights, with_labels = False)
    return ax

t1 = 300
t2 = 320
theta_gt1 = thetas_gt[t1,:,:]
theta_gt2 = thetas_gt[t2,:,:]
theta_inf1 = thetas_inf[t1,:,:]
theta_inf2 = thetas_inf[t2,:,:]

thetas_gt_diff = theta_gt2 - theta_gt1
thetas_inf_diff = theta_inf2 - theta_inf1

threshold = 0.0
threshold_norm = threshold * np.max(np.abs(thetas_gt_diff))
thetas_gt_diff_bin = np.where((thetas_gt_diff > threshold_norm), 1, thetas_gt_diff)
thetas_gt_diff_bin = np.where(thetas_gt_diff_bin < - threshold_norm, -1, thetas_gt_diff_bin)
thetas_gt_diff_bin = np.where((thetas_gt_diff_bin < threshold_norm) & (thetas_gt_diff_bin > - threshold_norm), 0, thetas_gt_diff_bin)
indeg = np.sum(thetas_gt_diff_bin, axis = 0)
idx = np.argsort(indeg)
thetas_gt_diff = thetas_gt_diff[idx,:][:,idx]
thetas_inf_diff = thetas_inf_diff[idx,:][:,idx]


fig = plt.figure(figsize = (15, 7))
axes = fig.subplots(nrows = 1, ncols = 2)
ax = draw_graph(thetas_gt_diff * (thetas_gt_diff != 0), threshold = threshold, ax = axes[0])
ax.set_title("changing edges between time {:.2f} and {:.2f} \n(ground truth)".format(sim_time[t1], sim_time[t2]))
# ax = draw_graph(thetas_inf_diff * (thetas_gt_diff != 0), threshold = threshold, ax = axes[1])
ax = draw_graph(thetas_inf_diff * (thetas_gt_diff != 0), threshold = threshold, ax = axes[1])
ax.set_title("changing edges between time {:.2f} and {:.2f} \n(CeSpGRN)".format(sim_time[t1], sim_time[t2]))
fig.savefig("../results_GGM/change_graphs.png", bbox_inches = "tight")

# t1 = 101
# t2 = 121

# theta_gt1 = thetas_gt[t1,:,:]
# theta_gt2 = thetas_gt[t2,:,:]
# theta_inf1 = thetas_inf[t1,:,:]
# theta_inf2 = thetas_inf[t2,:,:]

# thetas_inf_diff = theta_inf2 - theta_inf1
# thetas_gt_diff = theta_gt2 - theta_gt1

# fig = plt.figure(figsize = (10, 7))
# ax = fig.add_subplot()
# ax.hist(theta_gt1.reshape(-1), bins = 50)

# threshold = 0.0
# fig = plt.figure(figsize = (20, 7))
# axes = fig.subplots(nrows = 1, ncols = 2)
# ax = draw_graph(theta_gt1 * (thetas_inf_diff != 0), threshold = threshold, ax = axes[0])
# ax.set_title("ground truth graph at time " + str(sim_time[t1]))
# ax = draw_graph(theta_inf1 * (thetas_inf_diff != 0), threshold = threshold, ax = axes[1])
# ax.set_title("inferred graph at time " + str(sim_time[t1]))
# fig.savefig("../results_GGM/graph1.png", bbox_inches = "tight")

# fig = plt.figure(figsize = (10, 7))
# ax = fig.add_subplot()
# ax.hist(theta_gt2.reshape(-1), bins = 50)

# fig = plt.figure(figsize = (20, 7))
# axes = fig.subplots(nrows = 1, ncols = 2)
# ax = draw_graph(theta_gt2 * (thetas_inf_diff != 0), threshold = threshold, ax = axes[0])
# ax.set_title("ground truth graph at time " + str(sim_time[t2]))
# ax = draw_graph(theta_inf2 * (thetas_inf_diff != 0), threshold = threshold, ax = axes[1])
# ax.set_title("inferred graph at time " + str(sim_time[t2]))
# fig.savefig("../results_GGM/graph2.png", bbox_inches = "tight")

# fig = plt.figure(figsize = (10, 7))
# ax = fig.add_subplot()
# ax.hist(theta_gt2.reshape(-1) - theta_gt1.reshape(-1), bins = 50)






# In[]
'''
from sklearn.decomposition import PCA
plt.rcParams["font.size"] = 20

interval = 5
ngenes = 50
seed = 2
bandwidth = 10
lamb = 0.01
truncate_param = 30
data = str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param)
result_dir = "../results_GGM/bifur_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "_" + str(seed) + "_sergio/"

X = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "_" + str(seed) + "_sergio/expr.npy")
sim_time = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "_" + str(seed) + "_sergio/sim_time.npy")
sim_time = sim_time/np.max(sim_time)
thetas = np.load(file = result_dir + "thetas_" + data + "_kt.npy")

pca_op = PCA()
X_pca = pca_op.fit_transform(X)
theta_pca = pca_op.fit_transform(thetas.reshape(ntimes, -1))

fig = plt.figure(figsize = (10, 7))
ax = fig.add_subplot()
pic = ax.scatter(X_pca[:, 0], X_pca[:, 1], c = sim_time, s = 5)
fig.colorbar(pic)
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")


fig = plt.figure(figsize = (10, 7))
ax = fig.add_subplot()
pic = ax.scatter(theta_pca[:, 0], theta_pca[:, 1], c = sim_time, s = 5)
fig.colorbar(pic)
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")

'''
