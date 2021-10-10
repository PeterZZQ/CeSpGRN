# In[]
import sys, os

sys.path.append('./')
import simulator_soft_ODE as simulator
import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.decomposition import PCA
import numpy as np

plt.rcParams["font.size"] = 20

def preprocess(counts):
    """\
    Input:
    counts = (ntimes, ngenes)

    Description:
    ------------
    Preprocess the dataset
    """
    # normalize according to the library size

    libsize = np.median(np.sum(counts, axis=1))
    counts = counts / np.sum(counts, axis=1)[:, None] * libsize
    counts = np.log1p(counts)
    return counts

import importlib
importlib.reload(simulator)

# In[] bifurcating
# the larger the stepsize, the noisier the trajectory is
# the smaller the m_gene is, the noisier the trajectory is
'''
stepsize = 0.0001
simu_setting = {"ncells": 1000, # number of cells
                "ntimes": 1000, # time length for euler simulation
                "integration_step_size": stepsize, # stepsize for each euler step
                # parameter for dyn_GRN
                "ngenes": 18, # number of genes 
                "mode": "TF-TF&target", # mode of the simulation, `TF-TF&target' or `TF-target'
                "ntfs": 12,  # number of TFs
                # nchanges also drive the trajectory, but even if don't change nchanges, there is still linear trajectory
                "nchanges": 10, # number of changing edges for each interval
                "change_stepsize": 100, # number of stepsizes for each change
                "density": 0.1, # number of edges
                "seed": 0, # random seed
                "dW": None,
                "backbone": np.array(["0_1"] * 300 + ["1_2"] * 350 + ["1_3"] * 350)
                }

results = simulator.run_simulator(**simu_setting)

fig = plt.figure(figsize = (10, 7))
ax = fig.add_subplot()
umap_op = UMAP(n_components = 2)
pca_op = PCA(n_components = 2)

X = results["true count"].T
pt = results["pseudotime"]
X = preprocess(X)
X_umap = pca_op.fit_transform(X)
ax.scatter(X_umap[:, 0], X_umap[:, 1], s = 5, c = pt)
fig.savefig("bifur/true_count_plot.png", bbox_inches = "tight")

fig = plt.figure(figsize = (10, 7))
ax = fig.add_subplot()
X = results["observed count"].T
pt = results["pseudotime"]
X = preprocess(X)
X_umap = pca_op.fit_transform(X)
ax.scatter(X_umap[:, 0], X_umap[:, 1], s = 5, c = pt)
fig.savefig("bifur/observed_count_plot.png", bbox_inches = "tight")

np.save("bifur/true_count.npy", results["true count"].T)
np.save("bifur/obs_count.npy", results["observed count"].T)
np.save("bifur/pseudotime.npy", results["pseudotime"])
'''
# In[1] simulate bifurcating data where the graph in/out degrees kept
path = "../../data/continuousODE/bifur_sample_keep_deg/"
sergio_path = "./sergio_data/Interaction_cID_8"

for stepsize in [0.0001, 0.0002]:
    for interval in [50, 100, 200]:
        for (ngenes, ntfs) in [(20, 5), (30, 10), (50, 20), (100, 50)]:

            # NEED TO KEEP DEGREE IN SERGIO INITIAL GRAPH
            G0 = simulator.load_sub_sergio(grn_init = sergio_path, sub_size = ngenes, ntfs = ntfs, seed = seed, init_size = 100)

            simu_setting = {"ncells": 1000, # number of cells
                            "ntimes": 1000, # time length for euler simulation
                            "integration_step_size": stepsize, # stepsize for each euler step
                            # parameter for dyn_GRN
                            "ngenes": ngenes, # number of genes 
                            "mode": "TF-TF&target", # mode of the simulation, `TF-TF&target' or `TF-target'
                            "ntfs": ntfs,  # number of TFs
                            # nchanges also drive the trajectory, but even if don't change nchanges, there is still linear trajectory
                            "nchanges": 4, # number of changing edges for each interval
                            "change_stepsize": interval, # number of stepsizes for each change
                            "density": 0.1, # number of edges
                            "seed": 0, # random seed
                            "dW": None,
                            # the changing point must be divided exactly by the change_stepsize, or there will be issue.
                            "backbone": np.array(["0_1"] * 200 + ["1_2"] * 400 + ["1_3"] * 400),
                            "keep_degree": False,
                            "G0": G0
                            }

            data = "ngenes_" + str(simu_setting["ngenes"]) + "_interval_" + str(simu_setting["change_stepsize"]) + "_stepsize_" + str(simu_setting["integration_step_size" ])
            if not os.path.exists(path + data):
                os.makedirs(path + data)

            results = simulator.run_simulator(**simu_setting)

            fig = plt.figure(figsize = (10, 7))
            ax = fig.add_subplot()
            umap_op = UMAP(n_components = 2)
            pca_op = PCA(n_components = 2)

            X = results["true count"].T
            pt = results["pseudotime"]
            GRNs = results["GRNs"]
            # make the graph symmetric, originally only one direction, so we can simply sum them up

            # don't calculate diagonal elment twice
            diag_values = np.concatenate([np.diag(np.diag(GRNs[x, :, :]))[None, :, :] for x in range(GRNs.shape[0])], axis = 0)
            GRNs = GRNs + np.transpose(GRNs, (0, 2, 1)) - diag_values

            X = preprocess(X)
            X_umap = pca_op.fit_transform(X)
            ax.scatter(X_umap[:, 0], X_umap[:, 1], s = 5, c = pt)
            fig.savefig(path + data + "/true_count_plot.png", bbox_inches = "tight")

            fig = plt.figure(figsize = (10, 7))
            ax = fig.add_subplot()
            X = results["observed count"].T
            pt = results["pseudotime"]
            X = preprocess(X)
            X_umap = pca_op.fit_transform(X)
            ax.scatter(X_umap[:, 0], X_umap[:, 1], s = 5, c = pt)
            fig.savefig(path + data + "/observed_count_plot.png", bbox_inches = "tight")

            np.save(path + data + "/true_count.npy", results["true count"].T)
            np.save(path + data + "/obs_count.npy", results["observed count"].T)
            np.save(path + data + "/pseudotime.npy", results["pseudotime"])
            np.save(path + data + "/GRNs.npy", GRNs)

# %%
