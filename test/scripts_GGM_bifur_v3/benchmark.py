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

ntimes = 1000
nsample = 1
path = "../../data/GGM_bifurcate_test/"
result_path = "../results_GGM_test/"
intervals = [5, 25, 100]

umap_op = UMAP(n_components = 2, min_dist = 0.8, n_neighbors = 30, random_state = 0)

# In[] benchmark accuracy
print("------------------------------------------------------------------")
print("benchmark accuracy")
print("------------------------------------------------------------------")

def test_scores(interval):
    for interval in [interval]:
        for (ngenes, ntfs) in [(50, 20)]:
            for seed in range(1):            
                print("ntimes: " + str(ntimes) + ", interval: " + str(interval) + ", ngenes: " + str(ngenes) + ", seed: " + str(seed) + ", initial graph: sergio")
                scores = pd.DataFrame(columns = ["interval", "ngenes", "time", "model", "bandwidth", "truncate_param", "lambda", "nmse","kendall-tau", "pearson", "spearman", "cosine similarity", "AUPRC Ratio (pos)", "AUPRC Ratio (neg)", "AUPRC Ratio (abs)", "Early Precision Ratio (pos)", "Early Precision Ratio (neg)", "Early Precision Ratio (abs)"])
                result_dir = result_path + "bifur_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "_" + str(seed) + "_sergio/"
                
                # the data smapled from GGM is zero-mean
                X = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "_" + str(seed) + "_sergio/expr.npy")
                gt_adj = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "_" + str(seed) + "_sergio/Gs.npy")
                sim_time = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "_" + str(seed) + "_sergio/sim_time.npy")

                # CSN
                thetas = np.load(file = result_dir + "theta_CSN.npy")
                score = calc_scores(thetas_inf = thetas, thetas_gt = np.abs(gt_adj), interval = interval, model = "CSN")
                scores = pd.concat([scores, score], axis = 0, ignore_index = True)
                
                # admm, hyper-parameter
                for kt in [True, False]:
                    for bandwidth in [0.1, 1, 10]:
                        for truncate_param in [15, 30, 100]:
                            for lamb in [0.001, 0.01, 0.1]:
                                data = str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param)
                                if kt:
                                    thetas = np.load(result_dir + "thetas_" + data + "_kt.npy") 
                                    score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "CeSpGRN-kt", bandwidth = bandwidth, truncate_param = truncate_param, lamb = lamb)
                                
                                else:
                                    thetas = np.load(result_dir + "thetas_" + data + ".npy")
                                    score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "CeSpGRN", bandwidth = bandwidth, truncate_param = truncate_param, lamb = lamb)
                                
                                scores = pd.concat([scores, score], axis = 0, ignore_index = True)

                                print("\tHyper-parameter: bandwidth = {:.2f}, truncate_param = {:.2f}, lambda = {:.3f}".format(bandwidth, truncate_param, lamb)) 
                                print("\tmean nmse: {:.4f}, mean pearson: {:.4f}, mean kt: {:.4f}, mean spearman: {:.4f}, mean cosine: {:.4f}, mean AUPRC: {:4f}".format(score["nmse"].mean(), score["pearson"].mean(), score["kendall-tau"].mean(), score["spearman"].mean(), score["cosine similarity"].mean(), (score["AUPRC Ratio (pos)"].mean() + score["AUPRC Ratio (neg)"].mean())/2 ))
                                print()

                                fig = plt.figure(figsize = (10,7))
                                X_umap = umap_op.fit_transform(thetas.reshape(ntimes * nsample, -1))
                                ax = fig.add_subplot()
                                ax.scatter(X_umap[:,0], X_umap[:,1], c = sim_time, s = 5)
                                if kt:
                                    fig.savefig(result_dir + "infer_G_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_kt.png", bbox_inches = "tight")
                                else:
                                    fig.savefig(result_dir + "infer_G_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + ".png", bbox_inches = "tight")
                                
                # save results
                scores.to_csv(result_dir + "score.csv")
    
pool = Pool(3) 
pool.map(test_scores, intervals)
pool.close()
pool.join()


# In[] 
print("------------------------------------------------------------------")
print("benchmark differences")
print("------------------------------------------------------------------")
ntimes = 1000
nsample = 1
def test_scores_diff(interval):
    for interval in [interval]:
        for (ngenes, ntfs) in [(50, 20)]:
            for seed in range(1):
                print("ntimes: " + str(ntimes) + ", interval: " + str(interval) + ", ngenes: " + str(ngenes) + ", seed: " + str(seed) + ", initial graph: sergio")
                scores = pd.DataFrame(columns = ["interval", "ngenes", "time", "model", "bandwidth", "truncate_param", "lambda", "nmse","kendall-tau", "pearson", "spearman", "cosine similarity", "AUPRC Ratio (pos)", "AUPRC Ratio (neg)", "AUPRC Ratio (abs)", "Early Precision Ratio (pos)", "Early Precision Ratio (neg)", "Early Precision Ratio (abs)"])
                result_dir = result_path + "bifur_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "_" + str(seed) + "_sergio/"
                
                # the data smapled from GGM is zero-mean
                X = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "_" + str(seed) + "_sergio/expr.npy")
                gt_adj = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "_" + str(seed) + "_sergio/Gs.npy")
                step = 400
                gt_adj_diff = np.concatenate((gt_adj[400:600,:,:] - gt_adj[:200,:,:], gt_adj[800:,:,:] - gt_adj[:200,:,:]), axis = 0)

                sim_time = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "_" + str(seed) + "_sergio/sim_time.npy")

                # CSN
                thetas = np.load(file = result_dir + "theta_CSN.npy")
                thetas_diff = np.concatenate((thetas[400:600,:,:] - thetas[:200,:,:], thetas[800:,:,:] - thetas[:200,:,:]), axis = 0)
                score = calc_scores(thetas_inf = thetas_diff, thetas_gt = np.abs(gt_adj_diff), interval = interval, model = "CSN")
                scores = pd.concat([scores, score], axis = 0, ignore_index = True)

                # admm, hyper-parameter
                for kt in [True, False]:
                    for bandwidth in [0.1, 1, 10]:
                        for truncate_param in [15, 30, 100]:
                            for lamb in [0.001, 0.01, 0.1]:
                                data = str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param)
                                if kt:
                                    thetas = np.load(result_dir + "thetas_" + data + "_kt.npy")
                                    thetas_diff = np.concatenate((thetas[400:600,:,:] - thetas[:200,:,:], thetas[800:,:,:] - thetas[:200,:,:]), axis = 0)
                                    score = calc_scores(thetas_inf = thetas_diff, thetas_gt = gt_adj_diff, interval = interval, model = "CeSpGRN-kt", bandwidth = bandwidth, truncate_param = truncate_param, lamb = lamb)
                                
                                else:
                                    thetas = np.load(result_dir + "thetas_" + data + ".npy")
                                    thetas_diff = np.concatenate((thetas[400:600,:,:] - thetas[:200,:,:], thetas[800:,:,:] - thetas[:200,:,:]), axis = 0)
                                    score = calc_scores(thetas_inf = thetas_diff, thetas_gt = gt_adj_diff, interval = interval, model = "CeSpGRN", bandwidth = bandwidth, truncate_param = truncate_param, lamb = lamb)
                                
                                scores = pd.concat([scores, score], axis = 0, ignore_index = True)

                                print("\tHyper-parameter: bandwidth = {:.2f}, truncate_param = {:.2f}, lambda = {:.3f}".format(bandwidth, truncate_param, lamb)) 
                                print("\tmean nmse: {:.4f}, mean pearson: {:.4f}, mean kt: {:.4f}, mean spearman: {:.4f}, mean cosine: {:.4f}, mean AUPRC: {:4f}".format(score["nmse"].mean(), score["pearson"].mean(), score["kendall-tau"].mean(), score["spearman"].mean(), score["cosine similarity"].mean(), (score["AUPRC Ratio (pos)"].mean() + score["AUPRC Ratio (neg)"].mean())/2 ))
                                print()
    
                # save results
                scores.to_csv(result_dir + "score_diff.csv")

pool = Pool(3) 
pool.map(test_scores_diff, intervals)
pool.close()
pool.join()

# In[] summarize the mean result in csv file

ntimes = 1000
nsample = 1
for interval in [5, 100]:
    for (ngenes, ntfs) in [(50, 20)]:
        for seed in range(1):
            print("bifur_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "_" + str(seed) + "_sergio")
            result_dir = result_path + "bifur_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "_" + str(seed) + "_sergio/"
            score = pd.read_csv(result_dir + "score.csv", index_col = 0)
            mean_score = score.groupby(by = ["model", "bandwidth", "truncate_param", "lambda"]).mean()
            mean_score = mean_score.drop(["time"], axis = 1)
            mean_score.to_csv(result_dir + "mean_score.csv")
            # display(mean_score)

print("\ndifferences\n")

for interval in [5, 100]:
    for (ngenes, ntfs) in [(50, 20)]:
        for seed in range(1):
            print("bifur_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "_" + str(seed) + "_sergio")
            result_dir = result_path + "bifur_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "_" + str(seed) + "_sergio/"
            score = pd.read_csv(result_dir + "score_diff.csv", index_col = 0)
            mean_score = score.groupby(by = ["model", "bandwidth", "truncate_param", "lambda"]).mean()
            mean_score = mean_score.drop(["time"], axis = 1)
            mean_score.to_csv(result_dir + "mean_score_diff.csv")
            # display(mean_score)




#In[] Plot boxplots, rom above, we are able to find the setting with the best performance for each dataset already
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

ntimes = 1000
# ----------------------------------- without TF information ---------------------------------#
# How the bandwidth and truncate parameter is affected by the interval
for interval in [5, 25, 100]:
    score_interval = pd.DataFrame(columns = ["interval", "ngenes", "time", "model", "bandwidth", "truncate_param", "lambda", "nmse","kendall-tau", "pearson", "spearman", "cosine similarity", "AUPRC Ratio (pos)", "AUPRC Ratio (neg)", "AUPRC Ratio (abs)", "Early Precision Ratio (pos)", "Early Precision Ratio (neg)", "Early Precision Ratio (abs)"])
    score_interval_diff = pd.DataFrame(columns = ["interval", "ngenes", "time", "model", "bandwidth", "truncate_param", "lambda", "nmse","kendall-tau", "pearson", "spearman", "cosine similarity", "AUPRC Ratio (pos)", "AUPRC Ratio (neg)", "AUPRC Ratio (abs)", "Early Precision Ratio (pos)", "Early Precision Ratio (neg)", "Early Precision Ratio (abs)"])

    for (ngenes, ntfs) in [(50, 20)]:
        for seed in range(1):
            result_dir = result_path + "bifur_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "_" + str(seed) + "_sergio/"

            score = pd.read_csv(result_dir + "score.csv", index_col = 0)
            score_diff = pd.read_csv(result_dir + "score_diff.csv", index_col = 0)

            # score for admm
            score = score[score["model"] == "CeSpGRN"]
            score_diff = score_diff[score_diff["model"] == "CeSpGRN"]
            score_interval = pd.concat([score_interval, score], axis = 0)
            score_interval_diff = pd.concat([score_interval_diff, score_diff], axis = 0)

    score_interval["AUPRC Ratio (signed)"] = (score_interval["AUPRC Ratio (pos)"].values + score_interval["AUPRC Ratio (neg)"].values)/2
    score_interval_diff["AUPRC Ratio (signed)"] = (score_interval_diff["AUPRC Ratio (pos)"].values + score_interval_diff["AUPRC Ratio (neg)"].values)/2

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
            box_plot = sns.boxplot(data = score_interval[score_interval["lambda"] == lambdas[(i-1)//2]], x = "bandwidth", y = "AUPRC Ratio (signed)", hue = "truncate_param", ax = ax)
            # add_median_labels(box_plot.axes)            
            ax.set_xlabel("Bandwidth")
            ax.set_ylabel("AUPRC Ratio \n (signed)")
            ax.set_yscale('log')
            ax.legend(loc="upper left", prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.01, 1), title = "Neighborhood size")
    
    fig.set_facecolor('w')
    fig.suptitle("score of edge detection, change interval: " + str(interval), fontsize = 25)
    plt.tight_layout()
    fig.savefig(result_path + "CeSpGRN_" + str(interval) + ".png", bbox_inches = "tight")


    # Plot including lambda
    fig, axes = plt.subplots( figsize=(20.0, 5.0) , nrows=1, ncols=3) 
    lambdas = [0.001, 0.01, 0.1]
    for col, ax in enumerate(axes, start=1):
        ax.set_title("Lambda: {:.4f}".format(lambdas[col - 1]), fontsize=20)
        box_plot = sns.boxplot(data = score_interval_diff[score_interval_diff["lambda"] == lambdas[col - 1]], x = "bandwidth", y = "AUPRC Ratio (signed)", hue = "truncate_param", ax = ax)
        ax.set_yscale('log')
        ax.set_yticks([0.5, 1, 1.5, 2, 10, 50, 100])
        ax.set_ylabel("AUPRC ratio\n (signed)")
        ax.set_xlabel("Bandwidth")
        # add_median_labels(box_plot.axes)
        if col < 3:
            ax.get_legend().remove()
        else:
            ax.legend(loc="upper left", prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.01, 1), title = "Neighborhood size")


    fig.set_facecolor('w')
    fig.suptitle("score of changing edges detection, change interval: " + str(interval), fontsize = 25)
    plt.tight_layout()
    fig.savefig(result_path + "CeSpGRN_diff_" + str(interval) + ".png", bbox_inches = "tight")    

# In[]
for interval in [5, 25, 100]:
    score_interval = pd.DataFrame(columns = ["interval", "ngenes", "time", "model", "bandwidth", "truncate_param", "lambda", "nmse","kendall-tau", "pearson", "spearman", "cosine similarity", "AUPRC Ratio (pos)", "AUPRC Ratio (neg)", "AUPRC Ratio (abs)", "Early Precision Ratio (pos)", "Early Precision Ratio (neg)", "Early Precision Ratio (abs)"])
    score_interval_diff = pd.DataFrame(columns = ["interval", "ngenes", "time", "model", "bandwidth", "truncate_param", "lambda", "nmse","kendall-tau", "pearson", "spearman", "cosine similarity", "AUPRC Ratio (pos)", "AUPRC Ratio (neg)", "AUPRC Ratio (abs)", "Early Precision Ratio (pos)", "Early Precision Ratio (neg)", "Early Precision Ratio (abs)"])

    for (ngenes, ntfs) in [(50, 20)]:
        for seed in range(1):
            result_dir = result_path + "bifur_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "_" + str(seed) + "_sergio/"

            score = pd.read_csv(result_dir + "score.csv", index_col = 0)
            score_diff = pd.read_csv(result_dir + "score_diff.csv", index_col = 0)

            # score for admm
            score = score[score["model"] == "CeSpGRN-kt"]
            score_diff = score_diff[score_diff["model"] == "CeSpGRN-kt"]
            score_interval = pd.concat([score_interval, score], axis = 0)
            score_interval_diff = pd.concat([score_interval_diff, score_diff], axis = 0)

    score_interval["AUPRC Ratio (signed)"] = (score_interval["AUPRC Ratio (pos)"].values + score_interval["AUPRC Ratio (neg)"].values)/2
    score_interval_diff["AUPRC Ratio (signed)"] = (score_interval_diff["AUPRC Ratio (pos)"].values + score_interval_diff["AUPRC Ratio (neg)"].values)/2

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
            box_plot = sns.boxplot(data = score_interval[score_interval["lambda"] == lambdas[(i-1)//4]], x = "bandwidth", y = "pearson", hue = "truncate_param", ax = ax)
            # add_median_labels(box_plot.axes)
            ax.set_xlabel("Bandwidth")
            ax.set_ylabel("Pearson")
            ax.get_legend().remove()
        else:
            box_plot = sns.boxplot(data = score_interval[score_interval["lambda"] == lambdas[(i-1)//4]], x = "bandwidth", y = "AUPRC Ratio (signed)", hue = "truncate_param", ax = ax)
            # add_median_labels(box_plot.axes)            
            ax.set_xlabel("Bandwidth")
            ax.set_ylabel("AUPRC Ratio \n (signed)")
            ax.set_yscale('log')
            ax.legend(loc="upper left", prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.01, 1), title = "Neighborhood size")
    
    fig.set_facecolor('w')
    fig.suptitle("score of edge detection, change interval: " + str(interval), fontsize = 25)
    plt.tight_layout()
    fig.savefig(result_path + "CeSpGRN_" + str(interval) + "_kt.png", bbox_inches = "tight")


    # Plot including lambda
    fig, axes = plt.subplots( figsize=(20.0, 5.0) , nrows=1, ncols=3) 
    lambdas = [0.001, 0.01, 0.1]
    for col, ax in enumerate(axes, start=1):
        ax.set_title("Lambda: {:.4f}".format(lambdas[col - 1]), fontsize=20)
        box_plot = sns.boxplot(data = score_interval_diff[score_interval_diff["lambda"] == lambdas[col - 1]], x = "bandwidth", y = "AUPRC Ratio (signed)", hue = "truncate_param", ax = ax)
        ax.set_yscale('log')
        ax.set_yticks([0.5, 1, 1.5, 2, 10, 50, 100])
        ax.set_ylabel("AUPRC ratio\n (signed)")
        ax.set_xlabel("Bandwidth")
        add_median_labels(box_plot.axes)
        if col < 3:
            ax.get_legend().remove()
        else:
            ax.legend(loc="upper left", prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.01, 1), title = "Neighborhood size")


    fig.set_facecolor('w')
    fig.suptitle("score of changing edges detection, change interval: " + str(interval), fontsize = 25)
    plt.tight_layout()
    fig.savefig(result_path + "CeSpGRN_diff_" + str(interval) + "_kt.png", bbox_inches = "tight")    


# In[]
ntimes = 1000

# ----------------------------------- without TF information ---------------------------------#
# Bandwidth: 0.1, truncate parameter: 15, lambda: 0.1
bandwith = 0.1
truncate_param = 15
lamb = 0.1
score_all = pd.DataFrame(columns = ["interval", "ngenes", "nmse","kendall-tau", "pearson", "spearman", "cosine similarity", "time", "model", "bandwidth", "truncate_param", "lambda"])
score_all_diff = pd.DataFrame(columns = ["interval", "ngenes", "nmse","kendall-tau", "pearson", "spearman", "cosine similarity", "time", "model", "bandwidth", "truncate_param", "lambda"])


for interval in [5, 25, 100]:
    for (ngenes, ntfs) in [(50, 20)]:
        for seed in range(1):
            result_dir = result_path + "bifur_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "_" + str(seed) + "_sergio/"

            score = pd.read_csv(result_dir + "score.csv", index_col = 0)            
            score = score[(score["model"] == "CeSpGRN")|(score["model"] == "CeSpGRN-kt")|(score["model"] == "CSN")]
            score_all = pd.concat([score_all, score], axis = 0)


fig, ax = plt.subplots( figsize=(12.0, 5.0) , nrows=1, ncols=2) 
boxplot1 = sns.boxplot(data = score_all, x = "model", y = "pearson", ax = ax[0])
boxplot2 = sns.boxplot(data = score_all, x = "model", y = "spearman", ax = ax[1])

# ax[0].set_ylim([-1 , 1])
# ax[1].set_ylim([-1 , 1])
ax[0].set_xticklabels(boxplot1.get_xticklabels(), rotation = 45)
ax[1].set_xticklabels(boxplot2.get_xticklabels(), rotation = 45)


fig.set_facecolor('w')
fig.suptitle("score of edge detection")
plt.tight_layout()
fig.savefig(result_path + "compare_models.png", bbox_inches = "tight") 


# In[]
ntimes = 1000

# ----------------------------------- without TF information ---------------------------------#
# Bandwidth: 0.1, truncate parameter: 15, lambda: 0.1
bandwith = 0.1
truncate_param = 15
lamb = 0.1

for interval in [5, 25, 100]:
    score_all_diff = pd.DataFrame(columns = ["interval", "ngenes", "nmse","kendall-tau", "pearson", "spearman", "cosine similarity", "time", "model", "bandwidth", "truncate_param", "lambda"])
    for (ngenes, ntfs) in [(50, 20), (200, 20)]:
        for seed in range(5):
            result_dir = "../results_GGM/bifur_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "_" + str(seed) + "_sergio/"
            score_diff = pd.read_csv(result_dir + "score_diff.csv", index_col = 0)       
            score_diff = score_diff[(score_diff["model"] == "CeSpGRN")|(score_diff["model"] == "CeSpGRN-kt")|(score_diff["model"] == "CSN")]
            score_all_diff = pd.concat([score_all_diff, score_diff], axis = 0)

    score_all_diff.loc[score_all_diff["model"] != "CSN", "AUPRC Ratio (signed)"] = (score_all_diff.loc[score_all_diff["model"] != "CSN", "AUPRC Ratio (pos)"].values + score_all_diff.loc[score_all_diff["model"] != "CSN", "AUPRC Ratio (neg)"].values)/2
    score_all_diff.loc[score_all_diff["model"] == "CSN", "AUPRC Ratio (signed)"] = score_all_diff.loc[score_all_diff["model"] == "CSN", "AUPRC Ratio (abs)"].values
    score_all_diff.loc[score_all_diff["model"] != "CSN", "Early Precision Ratio (signed)"] = (score_all_diff.loc[score_all_diff["model"] != "CSN", "Early Precision Ratio (pos)"].values + score_all_diff.loc[score_all_diff["model"] != "CSN", "Early Precision Ratio (neg)"].values)/2
    score_all_diff.loc[score_all_diff["model"] == "CSN", "Early Precision Ratio (signed)"] = score_all_diff.loc[score_all_diff["model"] == "CSN", "Early Precision Ratio (abs)"].values


    fig, ax = plt.subplots( figsize=(12.0, 10.0) , nrows=2, ncols=2) 
    boxplot1 = sns.boxplot(data = score_all_diff, x = "model", y = "Early Precision Ratio (abs)", ax = ax[0,0])
    boxplot2 = sns.boxplot(data = score_all_diff, x = "model", y = "AUPRC Ratio (abs)", ax = ax[0,1])
    boxplot3 = sns.boxplot(data = score_all_diff, x = "model", y = "pearson", ax = ax[1,0])
    boxplot4 = sns.boxplot(data = score_all_diff, x = "model", y = "spearman", ax = ax[1,1])
    add_median_labels(boxplot3.axes)
    add_median_labels(boxplot4.axes)
    # ax[0].set_ylim([-1 , 1])
    # ax[1].set_ylim([-1 , 1])
    ax[0,0].set_xticklabels(boxplot1.get_xticklabels(), rotation = 45)
    ax[0,1].set_xticklabels(boxplot2.get_xticklabels(), rotation = 45)
    ax[1,0].set_xticklabels(boxplot3.get_xticklabels(), rotation = 45)
    ax[1,1].set_xticklabels(boxplot4.get_xticklabels(), rotation = 45)
    ax[0,0].set_yscale('log')
    ax[0,1].set_yscale('log')

    fig.set_facecolor('w')
    fig.suptitle("score of changing edges detection, change interval: " + str(interval), fontsize = 25)
    plt.tight_layout()
    fig.savefig(result_path + "compare_models_diff_" + str(interval) + ".png", bbox_inches = "tight") 

# In[]
ntimes = 1000

# ----------------------------------- without TF information ---------------------------------#
# Bandwidth: 0.1, truncate parameter: 15, lambda: 0.1
bandwith = 0.1
truncate_param = 15
lamb = 0.1

for interval in [5, 25, 100]:
    score_all_diff = pd.DataFrame(columns = ["interval", "ngenes", "nmse","kendall-tau", "pearson", "spearman", "cosine similarity", "time", "model", "bandwidth", "truncate_param", "lambda"])
    for (ngenes, ntfs) in [(50, 20), (200, 20)]:
        for seed in range(5):
            result_dir = "../results_GGM/bifur_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "_" + str(seed) + "_sergio/"
            score_diff = pd.read_csv(result_dir + "score_diff.csv", index_col = 0)          
            score_diff = score_diff[(score_diff["model"] == "CeSpGRN")|(score_diff["model"] == "CeSpGRN-kt")|(score_diff["model"] == "CSN")]            
            score_diff_CeSpGRN = score_diff[(score_diff["bandwidth"] == 10)&(score_diff["truncate_param"] == 15)&(score_diff["lambda"] == 0.01)&(score_diff["model"] == "CeSpGRN")]
            score_all_diff = pd.concat([score_all_diff, score_diff_other, score_diff_CeSpGRN], axis = 0)
            

    score_all_diff.loc[score_all_diff["model"] != "CSN", "AUPRC Ratio (signed)"] = (score_all_diff.loc[score_all_diff["model"] != "CSN", "AUPRC Ratio (pos)"].values + score_all_diff.loc[score_all_diff["model"] != "CSN", "AUPRC Ratio (neg)"].values)/2
    score_all_diff.loc[score_all_diff["model"] == "CSN", "AUPRC Ratio (signed)"] = score_all_diff.loc[score_all_diff["model"] == "CSN", "AUPRC Ratio (abs)"].values
    score_all_diff.loc[score_all_diff["model"] != "CSN", "Early Precision Ratio (signed)"] = (score_all_diff.loc[score_all_diff["model"] != "CSN", "Early Precision Ratio (pos)"].values + score_all_diff.loc[score_all_diff["model"] != "CSN", "Early Precision Ratio (neg)"].values)/2
    score_all_diff.loc[score_all_diff["model"] == "CSN", "Early Precision Ratio (signed)"] = score_all_diff.loc[score_all_diff["model"] == "CSN", "Early Precision Ratio (abs)"].values


    fig, ax = plt.subplots( figsize=(12.0, 10.0) , nrows=2, ncols=2) 
    boxplot1 = sns.boxplot(data = score_all_diff, x = "model", y = "Early Precision Ratio (abs)", ax = ax[0,0])
    boxplot2 = sns.boxplot(data = score_all_diff, x = "model", y = "AUPRC Ratio (abs)", ax = ax[0,1])
    boxplot3 = sns.boxplot(data = score_all_diff, x = "model", y = "pearson", ax = ax[1,0])
    boxplot4 = sns.boxplot(data = score_all_diff, x = "model", y = "spearman", ax = ax[1,1])
    add_median_labels(boxplot3.axes)
    add_median_labels(boxplot4.axes)
    # ax[0].set_ylim([-1 , 1])
    # ax[1].set_ylim([-1 , 1])
    ax[0,0].set_xticklabels(boxplot1.get_xticklabels(), rotation = 45)
    ax[0,1].set_xticklabels(boxplot2.get_xticklabels(), rotation = 45)
    ax[1,0].set_xticklabels(boxplot3.get_xticklabels(), rotation = 45)
    ax[1,1].set_xticklabels(boxplot4.get_xticklabels(), rotation = 45)
    ax[0,0].set_yscale('log')
    ax[0,1].set_yscale('log')

    fig.set_facecolor('w')
    fig.suptitle("score of changing edges detection, change interval: " + str(interval), fontsize = 25)
    plt.tight_layout()
    fig.savefig(result_path + "compare_models_diff_" + str(interval) + "_selected.png", bbox_inches = "tight") 


# %%
