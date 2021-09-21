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


# TODO: Compare between models, either using the best score or the average score, check if there are significant differences between models


result_dir = "../results/GGM/"
# baseline scores
scores_random = pd.read_csv(result_dir + "score_rand.csv")

# model 1, select the one with the highest median value
alpha = 1
rho = "adaptive"
best_nmse = [100, 0, 0]
best_pearson = [0, 0, 0]
best_cos = [0, 0, 0]
for bandwidth in [0.01, 0.1, 1, 10]:
    for lamb in [0.01, 0.1, 1]:
        data = str(bandwidth) + "_" + str(alpha) + "_" + str(lamb) + "_" + rho
        score = pd.read_csv(result_dir + "score_" + data + ".csv")
        # extract median score across times
        nmse_med = np.median(score["nmse"].values)
        # pos_med = np.median(score["probability of success"].values)
        pearson_med = np.median([eval(x)[0] for x in score["pearson"].values])
        cos_med = np.median(score["cosine similarity"].values)
        print("bandwidth: " + str(bandwidth) + ", alpha: " + str(alpha) + ", lambda: " + str(lamb) + ", rho: " + str(rho))
        print("nmse: " + str(nmse_med) + ", pearson: " + str(pearson_med) + ", cos simi: " + str(cos_med))
        if nmse_med < best_nmse[0]:
            best_nmse[0] = nmse_med
            best_nmse[1] = bandwidth
            best_nmse[2] = lamb
        if pearson_med > best_pearson[0]:
            best_pearson[0] = pearson_med
            best_pearson[1] = bandwidth
            best_pearson[2] = lamb
        if cos_med > best_cos[0]:
            best_cos[0] = cos_med
            best_cos[1] = bandwidth
            best_cos[2] = lamb
print()
print("best nmse: " + str(best_nmse[0]) + ", bandwidth: " + str(best_nmse[1]) + ", lamb: " + str(best_nmse[2]))
print("best pearson: " + str(best_pearson[0]) + ", bandwidth: " + str(best_pearson[1]) + ", lamb: " + str(best_pearson[2]))
print("best cos: " + str(best_cos[0]) + ", bandwidth: " + str(best_cos[1]) + ", lamb: " + str(best_cos[2]))
print()


# model 2
alpha = 2
rho = "1.7"
best_nmse = [100, 0, 0]
best_pearson = [0, 0, 0]
best_cos = [0, 0, 0]
for bandwidth in [0.01, 0.1, 1, 10]:
    for lamb in [0.01, 0.1, 1]:
        data = str(bandwidth) + "_" + str(alpha) + "_" + str(lamb) + "_" + rho
        score = pd.read_csv(result_dir + "score_" + data + ".csv")
        # extract median score across times
        nmse_med = np.median(score["nmse"].values)
        # pos_med = np.median(score["probability of success"].values)
        pearson_med = np.median([eval(x)[0] for x in score["pearson"].values])
        cos_med = np.median(score["cosine similarity"].values)
        print("bandwidth: " + str(bandwidth) + ", alpha: " + str(alpha) + ", lambda: " + str(lamb) + ", rho: " + str(rho))
        print("nmse: " + str(nmse_med) + ", pearson: " + str(pearson_med) + ", cos simi: " + str(cos_med))
        if nmse_med < best_nmse[0]:
            best_nmse[0] = nmse_med
            best_nmse[1] = bandwidth
            best_nmse[2] = lamb
        if pearson_med > best_pearson[0]:
            best_pearson[0] = pearson_med
            best_pearson[1] = bandwidth
            best_pearson[2] = lamb
        if cos_med > best_cos[0]:
            best_cos[0] = cos_med
            best_cos[1] = bandwidth
            best_cos[2] = lamb
print()
print("best nmse: " + str(best_nmse[0]) + ", bandwidth: " + str(best_nmse[1]) + ", lamb: " + str(best_nmse[2]))
print("best pearson: " + str(best_pearson[0]) + ", bandwidth: " + str(best_pearson[1]) + ", lamb: " + str(best_pearson[2]))
print("best cos: " + str(best_cos[0]) + ", bandwidth: " + str(best_cos[1]) + ", lamb: " + str(best_cos[2]))
print()

# model 3
alpha = 1
rho = "1.7"
best_nmse = [100, 0, 0]
best_pearson = [0, 0, 0]
best_cos = [0, 0, 0]
for bandwidth in [0.01, 0.1, 1, 10]:
    for lamb in [0.01, 0.1, 1]:
        data = str(bandwidth) + "_" + str(alpha) + "_" + str(lamb) + "_" + rho
        score = pd.read_csv(result_dir + "score_" + data + ".csv")
        # extract median score across times
        nmse_med = np.median(score["nmse"].values)
        # pos_med = np.median(score["probability of success"].values)
        pearson_med = np.median([eval(x)[0] for x in score["pearson"].values])
        cos_med = np.median(score["cosine similarity"].values)
        print("bandwidth: " + str(bandwidth) + ", alpha: " + str(alpha) + ", lambda: " + str(lamb) + ", rho: " + str(rho))
        print("nmse: " + str(nmse_med) + ", pearson: " + str(pearson_med) + ", cos simi: " + str(cos_med))
        if nmse_med < best_nmse[0]:
            best_nmse[0] = nmse_med
            best_nmse[1] = bandwidth
            best_nmse[2] = lamb
        if pearson_med > best_pearson[0]:
            best_pearson[0] = pearson_med
            best_pearson[1] = bandwidth
            best_pearson[2] = lamb
        if cos_med > best_cos[0]:
            best_cos[0] = cos_med
            best_cos[1] = bandwidth
            best_cos[2] = lamb
print()
print("best nmse: " + str(best_nmse[0]) + ", bandwidth: " + str(best_nmse[1]) + ", lamb: " + str(best_nmse[2]))
print("best pearson: " + str(best_pearson[0]) + ", bandwidth: " + str(best_pearson[1]) + ", lamb: " + str(best_pearson[2]))
print("best cos: " + str(best_cos[0]) + ", bandwidth: " + str(best_cos[1]) + ", lamb: " + str(best_cos[2]))
print()
print()
print("model using no mean")
print()

# model 4
alpha = 1
rho = "adaptive"
best_nmse = [100, 0, 0]
best_pearson = [0, 0, 0]
best_cos = [0, 0, 0]
for bandwidth in [0.01, 0.1, 1, 10]:
    for lamb in [0.01, 0.1, 1]:
        data = str(bandwidth) + "_" + str(alpha) + "_" + str(lamb) + "_" + rho
        score = pd.read_csv(result_dir + "score_" + data + "_nomean.csv")
        # extract median score across times
        nmse_med = np.median(score["nmse"].values)
        # pos_med = np.median(score["probability of success"].values)
        pearson_med = np.median([eval(x)[0] for x in score["pearson"].values])
        cos_med = np.median(score["cosine similarity"].values)
        print("bandwidth: " + str(bandwidth) + ", alpha: " + str(alpha) + ", lambda: " + str(lamb) + ", rho: " + str(rho))
        print("nmse: " + str(nmse_med) + ", pearson: " + str(pearson_med) + ", cos simi: " + str(cos_med))
        if nmse_med < best_nmse[0]:
            best_nmse[0] = nmse_med
            best_nmse[1] = bandwidth
            best_nmse[2] = lamb
        if pearson_med > best_pearson[0]:
            best_pearson[0] = pearson_med
            best_pearson[1] = bandwidth
            best_pearson[2] = lamb
        if cos_med > best_cos[0]:
            best_cos[0] = cos_med
            best_cos[1] = bandwidth
            best_cos[2] = lamb
print()
print("best nmse: " + str(best_nmse[0]) + ", bandwidth: " + str(best_nmse[1]) + ", lamb: " + str(best_nmse[2]))
print("best pearson: " + str(best_pearson[0]) + ", bandwidth: " + str(best_pearson[1]) + ", lamb: " + str(best_pearson[2]))
print("best cos: " + str(best_cos[0]) + ", bandwidth: " + str(best_cos[1]) + ", lamb: " + str(best_cos[2]))
print()

# model 5
alpha = 2
rho = "1.7"
best_nmse = [100, 0, 0]
best_pearson = [0, 0, 0]
best_cos = [0, 0, 0]
for bandwidth in [0.01, 0.1, 1, 10]:
    for lamb in [0.01, 0.1, 1]:
        data = str(bandwidth) + "_" + str(alpha) + "_" + str(lamb) + "_" + rho
        score = pd.read_csv(result_dir + "score_" + data + "_nomean.csv")
        # extract median score across times
        nmse_med = np.median(score["nmse"].values)
        # pos_med = np.median(score["probability of success"].values)
        pearson_med = np.median([eval(x)[0] for x in score["pearson"].values])
        cos_med = np.median(score["cosine similarity"].values)
        print("bandwidth: " + str(bandwidth) + ", alpha: " + str(alpha) + ", lambda: " + str(lamb) + ", rho: " + str(rho))
        print("nmse: " + str(nmse_med) + ", pearson: " + str(pearson_med) + ", cos simi: " + str(cos_med))
        if nmse_med < best_nmse[0]:
            best_nmse[0] = nmse_med
            best_nmse[1] = bandwidth
            best_nmse[2] = lamb
        if pearson_med > best_pearson[0]:
            best_pearson[0] = pearson_med
            best_pearson[1] = bandwidth
            best_pearson[2] = lamb
        if cos_med > best_cos[0]:
            best_cos[0] = cos_med
            best_cos[1] = bandwidth
            best_cos[2] = lamb
print()
print("best nmse: " + str(best_nmse[0]) + ", bandwidth: " + str(best_nmse[1]) + ", lamb: " + str(best_nmse[2]))
print("best pearson: " + str(best_pearson[0]) + ", bandwidth: " + str(best_pearson[1]) + ", lamb: " + str(best_pearson[2]))
print("best cos: " + str(best_cos[0]) + ", bandwidth: " + str(best_cos[1]) + ", lamb: " + str(best_cos[2]))
print()

# model 6
alpha = 1
rho = "1.7"
best_nmse = [100, 0, 0]
best_pearson = [0, 0, 0]
best_cos = [0, 0, 0]
for bandwidth in [0.01, 0.1, 1, 10]:
    for lamb in [0.01, 0.1, 1]:
        data = str(bandwidth) + "_" + str(alpha) + "_" + str(lamb) + "_" + rho
        score = pd.read_csv(result_dir + "score_" + data + "_nomean.csv")
        # extract median score across times
        nmse_med = np.median(score["nmse"].values)
        # pos_med = np.median(score["probability of success"].values)
        pearson_med = np.median([eval(x)[0] for x in score["pearson"].values])
        cos_med = np.median(score["cosine similarity"].values)
        print("bandwidth: " + str(bandwidth) + ", alpha: " + str(alpha) + ", lambda: " + str(lamb) + ", rho: " + str(rho))
        print("nmse: " + str(nmse_med) + ", pearson: " + str(pearson_med) + ", cos simi: " + str(cos_med))
        if nmse_med < best_nmse[0]:
            best_nmse[0] = nmse_med
            best_nmse[1] = bandwidth
            best_nmse[2] = lamb
        if pearson_med > best_pearson[0]:
            best_pearson[0] = pearson_med
            best_pearson[1] = bandwidth
            best_pearson[2] = lamb
        if cos_med > best_cos[0]:
            best_cos[0] = cos_med
            best_cos[1] = bandwidth
            best_cos[2] = lamb
print()
print("best nmse: " + str(best_nmse[0]) + ", bandwidth: " + str(best_nmse[1]) + ", lamb: " + str(best_nmse[2]))
print("best pearson: " + str(best_pearson[0]) + ", bandwidth: " + str(best_pearson[1]) + ", lamb: " + str(best_pearson[2]))
print("best cos: " + str(best_cos[0]) + ", bandwidth: " + str(best_cos[1]) + ", lamb: " + str(best_cos[2]))
print()

# %%
