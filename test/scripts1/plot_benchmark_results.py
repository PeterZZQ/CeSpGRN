# In[0]
from numpy.lib.function_base import disp
import pandas as pd
import numpy as np
import torch
import torch.nn
import matplotlib.pyplot as plt
import sys, os
sys.path.append('../../src/')

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform

import bmk_beeline as bmk
import warnings
warnings.filterwarnings("ignore")

from matplotlib import rcParams

labelsize = 16
rcParams['xtick.labelsize'] = labelsize
rcParams['ytick.labelsize'] = labelsize 


# TODO: Compare between models, either using the best score or the average score, check if there are significant differences between models
# In[1]
ntimes = 3000
interval = 200
ngenes = 20
result_dir = "../results/GGM_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "/"
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

fig = plt.figure(figsize = (10, 21))
ax = fig.subplots(nrows = 3, ncols = 1)
data = str(best_nmse[1]) + "_" + str(alpha) + "_" + str(best_nmse[2]) + "_" + rho
score = pd.read_csv(result_dir + "score_" + data + ".csv")
scores_random["pearson"] = [eval(x)[0] for x in scores_random["pearson"].values]
score["pearson"] = [eval(x)[0] for x in score["pearson"].values]
score_full = pd.concat([score, scores_random], axis = 0)
sns.boxplot(data = score_full, x = "model", y = "nmse", ax = ax[0])
sns.boxplot(data = score_full, x = "model", y = "pearson", ax = ax[1])
sns.boxplot(data = score_full, x = "model", y = "cosine similarity", ax = ax[2])
fig.savefig(result_dir + "score_boxplot1.png", bbox_inches = "tight")

# In[2]

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
fig = plt.figure(figsize = (10, 21))
ax = fig.subplots(nrows = 3, ncols = 1)
data = str(best_nmse[1]) + "_" + str(alpha) + "_" + str(best_nmse[2]) + "_" + rho
score = pd.read_csv(result_dir + "score_" + data + ".csv")
score["pearson"] = [eval(x)[0] for x in score["pearson"].values]
score_full = pd.concat([score, scores_random], axis = 0)
sns.boxplot(data = score_full, x = "model", y = "nmse", ax = ax[0])
sns.boxplot(data = score_full, x = "model", y = "pearson", ax = ax[1])
sns.boxplot(data = score_full, x = "model", y = "cosine similarity", ax = ax[2])
fig.savefig(result_dir + "score_boxplot2.png", bbox_inches = "tight")

# In[2]
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
fig = plt.figure(figsize = (10, 21))
ax = fig.subplots(nrows = 3, ncols = 1)
data = str(best_nmse[1]) + "_" + str(alpha) + "_" + str(best_nmse[2]) + "_" + rho
score = pd.read_csv(result_dir + "score_" + data + ".csv")
score["pearson"] = [eval(x)[0] for x in score["pearson"].values]
score_full = pd.concat([score, scores_random], axis = 0)
sns.boxplot(data = score_full, x = "model", y = "nmse", ax = ax[0])
sns.boxplot(data = score_full, x = "model", y = "pearson", ax = ax[1])
sns.boxplot(data = score_full, x = "model", y = "cosine similarity", ax = ax[2])
fig.savefig(result_dir + "score_boxplot3.png", bbox_inches = "tight")

# In[2]
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
fig = plt.figure(figsize = (10, 21))
ax = fig.subplots(nrows = 3, ncols = 1)
data = str(best_nmse[1]) + "_" + str(alpha) + "_" + str(best_nmse[2]) + "_" + rho
score = pd.read_csv(result_dir + "score_" + data + "_nomean.csv")
score["pearson"] = [eval(x)[0] for x in score["pearson"].values]
score_full = pd.concat([score, scores_random], axis = 0)
sns.boxplot(data = score_full, x = "model", y = "nmse", ax = ax[0])
sns.boxplot(data = score_full, x = "model", y = "pearson", ax = ax[1])
sns.boxplot(data = score_full, x = "model", y = "cosine similarity", ax = ax[2])
fig.savefig(result_dir + "score_boxplot4.png", bbox_inches = "tight")


# In[2]
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
fig = plt.figure(figsize = (10, 21))
ax = fig.subplots(nrows = 3, ncols = 1)
data = str(best_nmse[1]) + "_" + str(alpha) + "_" + str(best_nmse[2]) + "_" + rho
score = pd.read_csv(result_dir + "score_" + data + "_nomean.csv")
score["pearson"] = [eval(x)[0] for x in score["pearson"].values]
score_full = pd.concat([score, scores_random], axis = 0)
sns.boxplot(data = score_full, x = "model", y = "nmse", ax = ax[0])
sns.boxplot(data = score_full, x = "model", y = "pearson", ax = ax[1])
sns.boxplot(data = score_full, x = "model", y = "cosine similarity", ax = ax[2])
fig.savefig(result_dir + "score_boxplot5.png", bbox_inches = "tight")

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
fig = plt.figure(figsize = (10, 21))
ax = fig.subplots(nrows = 3, ncols = 1)
data = str(best_nmse[1]) + "_" + str(alpha) + "_" + str(best_nmse[2]) + "_" + rho
score = pd.read_csv(result_dir + "score_" + data + "_nomean.csv")
score["pearson"] = [eval(x)[0] for x in score["pearson"].values]
score_full = pd.concat([score, scores_random], axis = 0)
sns.boxplot(data = score_full, x = "model", y = "nmse", ax = ax[0])
sns.boxplot(data = score_full, x = "model", y = "pearson", ax = ax[1])
sns.boxplot(data = score_full, x = "model", y = "cosine similarity", ax = ax[2])
fig.savefig(result_dir + "score_boxplot6.png", bbox_inches = "tight")

# %%
