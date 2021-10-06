# In[]
import sys
sys.path.append('./')
import simulator_soft_ODE as simulator
import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.decomposition import PCA
import numpy as np

# In[] bifurcating
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

plt.rcParams["font.size"] = 20
stepsize = 0.00005
simu_setting = {"ncells": 1000, # number of cells
                "ntimes": 2000, # time length for euler simulation
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
                "backbone": np.array(["0_1"] * 600 + ["1_2"] * 700 + ["1_3"] * 700)
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

