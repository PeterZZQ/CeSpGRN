# In[0]
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

    scores = pd.DataFrame(columns = ["interval", "ngenes", "time", "model", "bandwidth", "truncate_param", "lambda", "nmse","kendall-tau", "pearson", "spearman", "cosine similarity", "AUPRC_pos", "AUPRC_neg"])

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

        scores = scores.append({"interval": interval,
                            "ngenes": ngenes,
                            "nmse": nmse, 
                            "pearson": pearson_val, 
                            "kendall-tau": kt,
                            "spearman": spearman_val,
                            "cosine similarity": cosine_sim, 
                            "AUPRC_pos": AUPRC_pos/AUPRC_pos_rand,
                            "AUPRC_neg": AUPRC_neg/AUPRC_neg_rand,
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
path = "../../data/GGM_bifurcate/"
umap_op = UMAP(n_components = 2, min_dist = 0.8, n_neighbors = 30, random_state = 0)

'''
def summarize_scores(setting):
    ntimes = setting["ntimes"]
    interval = setting["interval"]
    ngenes = setting["ngenes"]
    ntfs = setting["ntfs"]
    seed = setting["seed"]

    print("ntimes: " + str(ntimes) + ", interval: " + str(interval) + ", ngenes: " + str(ngenes) + ", seed: " + str(seed) + ", initial graph: sergio\n")
    scores = pd.DataFrame(columns = ["interval", "ngenes", "time", "model", "bandwidth", "truncate_param", "lambda", "nmse","kendall-tau", "pearson", "spearman", "cosine similarity", "AUPRC_pos", "AUPRC_neg"])
    result_dir = "../results_GGM/bifur_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "_" + str(seed) + "_sergio/"
    
    # the data smapled from GGM is zero-mean
    X = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "_" + str(seed) + "_sergio/expr.npy")
    gt_adj = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "_" + str(seed) + "_sergio/Gs.npy")
    sim_time = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "_" + str(seed) + "_sergio/sim_time.npy")
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
    score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "GENIE3-DYN")
    scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    print("genie3-dyn.")
    
    
    # genie3-dyn-tf 
    thetas = np.load(file = result_dir + "theta_genie_dyn_tf.npy")
    score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "GENIE3-DYN-TF")
    scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    print("genie3-dyn-tf.")


    # SCODE (True T)
    thetas = np.load(file = result_dir + "theta_scode_truet.npy")
    score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "SCODE (true time)")
    scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    print("SCODE (True T).")

    
    # SCODE-DYN (True T)
    thetas = np.load(file = result_dir + "theta_scode_dyn_truet.npy")
    score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "SCODE-DYN (true time)")
    scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    print("SCODE-DYN (True T).")


    print("Not using TF information")
    # admm, hyper-parameter
    for bandwidth in [0.1, 1, 10]:
        for truncate_param in [15, 30, 100]:
            for lamb in [0.001, 0.01, 0.1]:
                data = str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_0"
                thetas = np.load(file = result_dir + "thetas_" + data + ".npy")
                score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "Dyn-GRN", bandwidth = bandwidth, truncate_param = truncate_param, lamb = lamb)
                scores = pd.concat([scores, score], axis = 0, ignore_index = True)

                # print("\tHyper-parameter: bandwidth = {:.2f}, truncate_param = {:.2f}, lambda = {:.2f}".format(bandwidth, truncate_param, lamb)) 
                # print("\tmean nmse: {:.4f}, mean pearson: {:.4f}, mean kt: {:.4f}, mean spearman: {:.4f}, mean cosine: {:.4f}, mean AUPRC: {:4f}".format(score["nmse"].mean(), score["pearson"].mean(), score["kendall-tau"].mean(), score["spearman"].mean(), score["cosine similarity"].mean(), (score["AUPRC_pos"].mean() + score["AUPRC_neg"].mean())/2 ))
                # print()

                fig = plt.figure(figsize = (10,7))
                X_umap = umap_op.fit_transform(thetas.reshape(ntimes * nsample, -1))
                ax = fig.add_subplot()
                ax.scatter(X_umap[:,0], X_umap[:,1], c = sim_time, s = 5)
                fig.savefig(result_dir + "infer_G_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_0.png", bbox_inches = "tight")


    scores.to_csv(result_dir + "score.csv")


settings = []
for interval in [5, 25, 100]:
    for (ngenes, ntfs) in [(50, 20), (200, 20)]:
        for seed in range(5):
            settings.append({
                "ntimes": ntimes,
                "interval": interval,
                "ngenes": ngenes,
                "ntfs": ntfs,
                "seed": seed
            })


pool = Pool(15) 
pool.map(summarize_scores, [x for x in settings])
pool.close()
pool.join()


# In[] 
print("------------------------------------------------------------------")
print("benchmark differences")
print("------------------------------------------------------------------")


def summarize_scores_diff(setting):
    ntimes = setting["ntimes"]
    interval = setting["interval"]
    ngenes = setting["ngenes"]
    ntfs = setting["ntfs"]
    seed = setting["seed"]

    print("ntimes: " + str(ntimes) + ", interval: " + str(interval) + ", ngenes: " + str(ngenes) + ", seed: " + str(seed) + ", initial graph: sergio")
    scores = pd.DataFrame(columns = ["interval", "ngenes", "time", "model", "bandwidth", "truncate_param", "lambda", "nmse","kendall-tau", "pearson", "spearman", "cosine similarity", "AUPRC_pos", "AUPRC_neg"])
    result_dir = "../results_GGM/bifur_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "_" + str(seed) + "_sergio/"
    
    # the data smapled from GGM is zero-mean
    X = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "_" + str(seed) + "_sergio/expr.npy")
    gt_adj = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "_" + str(seed) + "_sergio/Gs.npy")
    sim_time = np.load(path + "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "_" + str(seed) + "_sergio/sim_time.npy")
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
    score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "GENIE3-DYN")
    scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    print("genie3-dyn.")
    
    # genie3-dyn-tf 
    thetas = np.load(file = result_dir + "theta_genie_dyn_tf.npy")
    thetas = thetas[step::,:,:] - thetas[:-step:,:,:]
    score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "GENIE3-DYN-TF")
    scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    print("genie3-dyn-tf.")

    # SCODE (True T)
    thetas = np.load(file = result_dir + "theta_scode_truet.npy")
    thetas = thetas[step::,:,:] - thetas[:-step:,:,:]
    score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "SCODE (true time)")
    scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    print("SCODE (True T).")
    
    # SCODE-DYN (True T)
    thetas = np.load(file = result_dir + "theta_scode_dyn_truet.npy")
    thetas = thetas[step::,:,:] - thetas[:-step:,:,:]
    score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "SCODE-DYN (true time)")
    scores = pd.concat([scores, score], axis = 0, ignore_index = True)
    print("SCODE-DYN (True T).")

    print("Not using TF information")
    # admm, hyper-parameter
    for bandwidth in [0.1, 1, 10]:
        for truncate_param in [15, 30, 100]:
            for lamb in [0.001, 0.01, 0.1]:
                data = str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_0"
                thetas = np.load(file = result_dir + "thetas_" + data + ".npy")
                thetas = thetas[step::,:,:] - thetas[:-step:,:,:]
                score = calc_scores(thetas_inf = thetas, thetas_gt = gt_adj, interval = interval, model = "Dyn-GRN", bandwidth = bandwidth, truncate_param = truncate_param, lamb = lamb)
                scores = pd.concat([scores, score], axis = 0, ignore_index = True)

                # print("\tHyper-parameter: bandwidth = {:.2f}, truncate_param = {:.2f}, lambda = {:.2f}".format(bandwidth, truncate_param, lamb)) 
                # print("\tmean nmse: {:.4f}, mean pearson: {:.4f}, mean kt: {:.4f}, mean spearman: {:.4f}, mean cosine: {:.4f}, mean AUPRC: {:4f}".format(score["nmse"].mean(), score["pearson"].mean(), score["kendall-tau"].mean(), score["spearman"].mean(), score["cosine similarity"].mean(), (score["AUPRC_pos"].mean() + score["AUPRC_neg"].mean())/2 ))
                # print()

    scores.to_csv(result_dir + "score_diff.csv")
    return result_dir, scores
    


ntimes = 1000
nsample = 1

settings = []
for interval in [5, 25, 100]:
    for (ngenes, ntfs) in [(200, 20)]:
        for seed in range(5):
            settings.append({
                "ntimes": ntimes,
                "interval": interval,
                "ngenes": ngenes,
                "ntfs": ntfs,
                "seed": seed
            })

pool = Pool(15)
result_dirs, scores = zip(*pool.map(summarize_scores_diff, [x for x in settings]))       
pool.close()
pool.join()
'''

# In[] summarize the mean result in csv file

ntimes = 1000
nsample = 1
for interval in [5, 25, 100]:
    for (ngenes, ntfs) in [(50, 20), (200, 20)]:
        for seed in range(5):
            print("ntimes: " + str(ntimes) + ", interval: " + str(interval) + ", ngenes: " + str(ngenes) + ", seed: " + str(seed) + ", initial graph: sergio")
            result_dir = "../results_GGM/bifur_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "_" + str(seed) + "_sergio/"
            score = pd.read_csv(result_dir + "score.csv", index_col = 0)
            mean_score = score.groupby(by = ["model", "bandwidth", "truncate_param", "lambda"]).mean()
            mean_score = mean_score.drop(["time"], axis = 1)
            mean_score.to_csv(result_dir + "mean_score.csv")
            # display(mean_score)

print("\ndifferences\n")

for interval in [5, 25, 100]:
    for (ngenes, ntfs) in [(50, 20), (200, 20)]:
        for seed in range(5):
            print("ntimes: " + str(ntimes) + ", interval: " + str(interval) + ", ngenes: " + str(ngenes) + ", seed: " + str(seed) + ", initial graph: sergio")
            result_dir = "../results_GGM/bifur_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "_" + str(seed) + "_sergio/"
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
    score_interval = pd.DataFrame(columns = ["interval", "ngenes", "time", "model", "bandwidth", "truncate_param", "lambda", "nmse","kendall-tau", "pearson", "spearman", "cosine similarity", "AUPRC_pos", "AUPRC_neg"])
    score_interval_diff = pd.DataFrame(columns = ["interval", "ngenes", "time", "model", "bandwidth", "truncate_param", "lambda", "nmse","kendall-tau", "pearson", "spearman", "cosine similarity", "AUPRC_pos", "AUPRC_neg"])

    for (ngenes, ntfs) in [(50, 20), (200, 20)]:
        for seed in range(5):
            result_dir = "../results_GGM/bifur_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "_" + str(seed) + "_sergio/"

            score = pd.read_csv(result_dir + "score.csv", index_col = 0)
            score_diff = pd.read_csv(result_dir + "score_diff.csv", index_col = 0)

            # score for admm
            score = score[score["model"] == "Dyn-GRN"]
            score_diff = score_diff[score_diff["model"] == "Dyn-GRN"]
            score_interval = pd.concat([score_interval, score], axis = 0)
            score_interval_diff = pd.concat([score_interval_diff, score_diff], axis = 0)

    score_interval["AUPRC_ratio"] = (score_interval["AUPRC_pos"].values + score_interval["AUPRC_neg"].values)/2
    score_interval_diff["AUPRC_ratio"] = (score_interval_diff["AUPRC_pos"].values + score_interval_diff["AUPRC_neg"].values)/2
    score_interval.loc[score_interval["model"] == "Dyn-GRN", "model"] = "CeSpGRN"
    score_interval_diff.loc[score_interval_diff["model"] == "Dyn-GRN", "model"] = "CeSpGRN"

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
            box_plot = sns.boxplot(data = score_interval[score_interval["lambda"] == lambdas[(i-1)//4]], x = "bandwidth", y = "spearman", hue = "truncate_param", ax = ax)
            # add_median_labels(box_plot.axes)            
            ax.set_xlabel("Bandwidth")
            ax.set_ylabel("Spearman")
            ax.legend(loc="upper left", prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.01, 1), title = "Neighborhood size")
    
    fig.set_facecolor('w')
    fig.suptitle("score of edge detection, change interval: " + str(interval), fontsize = 25)
    plt.tight_layout()
    fig.savefig("../results_GGM/DynGRN_bifur_score_full_interval_" + str(interval) + ".png", bbox_inches = "tight")


    # Plot including lambda
    fig, axes = plt.subplots( figsize=(20.0, 5.0) , nrows=1, ncols=3) 
    lambdas = [0.001, 0.01, 0.1]
    for col, ax in enumerate(axes, start=1):
        ax.set_title("Lambda: {:.4f}".format(lambdas[col - 1]), fontsize=20)
        box_plot = sns.boxplot(data = score_interval_diff[score_interval_diff["lambda"] == lambdas[col - 1]], x = "bandwidth", y = "AUPRC_ratio", hue = "truncate_param", ax = ax)
        ax.set_yscale('log')
        ax.set_yticks([0.5, 1, 1.5, 2, 10, 50, 100])
        ax.set_ylabel("AUPRC ratio (signed)")
        ax.set_xlabel("Bandwidth")
        add_median_labels(box_plot.axes)
        if col < 3:
            ax.get_legend().remove()
        else:
            ax.legend(loc="upper left", prop={'size': 15}, frameon = False, ncol = 1, bbox_to_anchor=(1.01, 1), title = "Neighborhood size")


    fig.set_facecolor('w')
    fig.suptitle("score of changing edges detection, change interval: " + str(interval), fontsize = 25)
    plt.tight_layout()
    fig.savefig("../results_GGM/DynGRN_bifur_score_full_diff_interval_" + str(interval) + ".png", bbox_inches = "tight")    


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
    for (ngenes, ntfs) in [(50, 20), (200, 20)]:
        for seed in range(5):
            result_dir = "../results_GGM/bifur_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "_" + str(seed) + "_sergio/"

            score = pd.read_csv(result_dir + "score.csv", index_col = 0)
            score_diff = pd.read_csv(result_dir + "score_diff.csv", index_col = 0)

            # score for admm
            # score_dyngrn = score[(score["model"] == "Dyn-GRN")&(score["lambda"] == 0.1)&(score["bandwidth"] == 0.1)&(score["truncate_param"] == 0.1)]
            # score_dyngrn_diff = score_diff[(score_diff["model"] == "Dyn-GRN")&(score_diff["lambda"] == 0.1)&(score_diff["bandwidth"] == 0.1)&(score_diff["truncate_param"] == 0.1)]
            # score_dyngrn_tf = score[(score["model"] == "Dyn-GRN-TF")&(score["lambda"] == 0.1)&(score["bandwidth"] == 0.1)&(score["truncate_param"] == 0.1)]
            # score_dyngrn_tf_diff = score_diff[(score_diff["model"] == "Dyn-GRN-TF")&(score_diff["lambda"] == 0.1)&(score_diff["bandwidth"] == 0.1)&(score_diff["truncate_param"] == 0.1)]

            # score_other = score[(score["model"] != "Dyn-GRN")|(score["model"] != "Dyn-GRN-TF")|(score["model"] != "GENIE3-TF")]
            # score_other_diff = score_diff[(score_diff["model"] != "Dyn-GRN")|(score_diff["model"] != "Dyn-GRN-TF")|(score_diff["model"] != "GENIE3-TF")]
            
            
            score = score[(score["model"] == "GENIE3")|(score["model"] == "GENIE3-DYN")|(score["model"] == "SCODE (true time)")|(score["model"] == "SCODE-DYN (true time)")|(score["model"] == "Dyn-GRN")]
            score_diff = score_diff[(score_diff["model"] == "GENIE3")|(score_diff["model"] == "GENIE3-DYN")|(score_diff["model"] == "SCODE (true time)")|(score_diff["model"] == "SCODE-DYN (true time)")|(score_diff["model"] == "Dyn-GRN")]
            score.loc[score["model"] == "SCODE (true time)", "model"] = "SCODE"
            score.loc[score["model"] == "SCODE-DYN (true time)", "model"] = "SCODE-Dyn"
            score.loc[score["model"] == "GENIE3-DYN", "model"] = "GENIE3-Dyn"
            score_all = pd.concat([score_all, score], axis = 0)
            score_all_diff = pd.concat([score_all_diff, score_diff], axis = 0)

            fig, ax = plt.subplots( figsize=(12.0, 7.0) , nrows=1, ncols=2) 
            boxplot1 = sns.boxplot(data = score, x = "model", y = "pearson", ax = ax[0])
            boxplot2 = sns.boxplot(data = score, x = "model", y = "spearman", ax = ax[1])

            # ax[0].set_ylim([-1 , 1])
            # ax[1].set_ylim([-1 , 1])
            ax[0].set_xticklabels(boxplot1.get_xticklabels(), rotation = 45)
            ax[1].set_xticklabels(boxplot2.get_xticklabels(), rotation = 45)

            
            fig.set_facecolor('w')
            fig.suptitle("score of edge detection, change interval: " + str(interval))
            plt.tight_layout()
            fig.savefig(result_dir + "compare_models.png", bbox_inches = "tight")


score_all.loc[score_all["model"] == "Dyn-GRN", "model"] = "CeSpGRN"
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
fig.savefig("../results_GGM/compare_models_bifur.png", bbox_inches = "tight")        

# %%
