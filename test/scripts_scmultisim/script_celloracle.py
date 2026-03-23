# In[]

import numpy as np 
import anndata
import scanpy as sc
import celloracle as co
import matplotlib.pyplot as plt
import pandas as pd
import os, sys


# In[]
# -------------------------------------
#
# read data
#
# -------------------------------------
# datasets = [
#     "simulated_8000_20_10_100_0.01_0_0_4",
#     "simulated_8000_20_10_100_0.01_1_0_4",
#     "simulated_8000_20_10_100_0.01_2_0_4",
#     "simulated_8000_20_10_100_0.1_0_0_4",
#     "simulated_8000_20_10_100_0.1_1_0_4",
#     "simulated_8000_20_10_100_0.1_2_0_4"
# ]
datasets = [
    "simulated_8000_3_10_100_0.01_0_0_4",
    "simulated_8000_3_10_100_0.01_1_0_4",
    "simulated_8000_3_10_100_0.01_2_0_4",
    "simulated_8000_3_10_100_0.1_0_0_4",
    "simulated_8000_3_10_100_0.1_1_0_4",
    "simulated_8000_3_10_100_0.1_2_0_4"
]
dataset = datasets[5]
nclusts = 10
path = f"../../data/scMultiSim/{dataset}/"
result_dir = f"../results_scmultisim/{dataset}/"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# count data
counts_rna = pd.read_csv(path + "counts_rna_true.txt", sep = "\t", index_col = 0).T.values
# normalization step
meta_cells = pd.read_csv(path + "meta_cells.txt", delimiter = "\t")
# create adata
adata = anndata.AnnData(X = counts_rna, obs = meta_cells)
adata.var.index = ["gene" + str(x + 1) for x in range(adata.shape[1])]



# --------------------------------------------------------------------
#
# Preprocessing
#
# --------------------------------------------------------------------

print("Preprocess the count matrix...")
# preprocessing, counts_per_cell_after cannot be too small, or import_anndata_as_raw_count will fail
sc.pp.normalize_per_cell(adata, counts_per_cell_after = 50)
adata.raw = adata
adata.layers["raw_count"] = adata.raw.X.copy()
# log transformation and scaling
sc.pp.log1p(adata)
sc.pp.scale(adata)
# dimension reduction
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=30, n_pcs=30)

# sc.tl.diffmap(adata)
# # Calculate neihbors again based on diffusionmap
# sc.pp.neighbors(adata, n_neighbors=30, use_rep='X_diffmap')
# sc.tl.paga(adata, groups='leiden')
# sc.tl.draw_graph(adata, init_pos='paga', random_state=123)

# sc.tl.umap(adata)
# # clustering
# sc.tl.leiden(adata, resolution=0.04)
# # visualization
# sc.pl.umap(adata, color = ["leiden", "pop"], save = "umap.png")

# ground truth clusters
for i in range(0, nclusts, 1):
    threshold = i * 1/nclusts
    adata.obs.loc[(adata.obs["depth"] >= threshold) & (adata.obs["depth"] <= threshold + 1/nclusts), "leiden"] = str(i)
# In[]
# base GRN matrix, from scATAC-seq information, of the shape (ngenes, ntfs)
grn_priors = []
# all grn_prior share the same column names/regulators and index
regulators = pd.read_csv(path + f"grn_prior_1.txt", sep = "\t",index_col = 0).columns.values
# celloracle accept only one baseGRN, average
for i in range(1, adata.shape[0] + 1):
    grn_priors.append(pd.read_csv(path + f"grn_prior_{i}.txt", sep = "\t",index_col = 0).values[None,:,:])
grn_priors = np.concatenate(grn_priors, axis = 0)
grn_priors = pd.DataFrame(data = np.mean(grn_priors, axis = 0)/np.max(np.mean(grn_priors, axis = 0)), columns = ["gene" + str(x) for x in regulators], index = adata.var.index.values)

# select threshold
fig = plt.figure(figsize = (7,5))
ax = fig.add_subplot()
_ = ax.hist(grn_priors.values.reshape(-1), bins = 30)
ax.set_yscale("log")
fig.savefig(result_dir + "histogram_grn_prior.png", bbox_inches = "tight")

# In[]
# --------------------------------------------------------------------
#
# Load count matrix and base GRN into CellOracle
#
# --------------------------------------------------------------------
print("Running cell oracle....")
# Instantiate Oracle object
oracle = co.Oracle()
# In this notebook, we use the unscaled mRNA count for the nput of Oracle object.
adata.X = adata.layers["raw_count"].copy()
# Instantiate Oracle object.
oracle.import_anndata_as_raw_count(adata=adata, cluster_column_name="leiden", embedding_name="X_pca", transform = "log2")

# Creat TFinfo_dictionary, each key is a target gene, the value is the list of TF connecting to the target gene in baseGRN
TF_to_TG_dictionary = {}
for tf_id, tf in enumerate(grn_priors.columns.values):
    # the input TF to TG is not weighted, if use 0, all are none 0, no different than using TF information
    # in addition, grn_prior cannot be negative
    threshold = 5e-2
    TF_to_TG_dictionary[tf] = [x for x in grn_priors.index.values[grn_priors.loc[:, tf] > threshold]] 

TG_to_TF_dictionary = co.utility.inverse_dictionary(TF_to_TG_dictionary)

# You can load TF info dataframe with the following code.
oracle.import_TF_data(TFdict=TG_to_TF_dictionary)


# In[]
# --------------------------------------------------------------------
#
# Data imputation
#
# --------------------------------------------------------------------
# knn imputation
# Perform PCA
oracle.perform_PCA()

# Select important PCs
plt.plot(np.cumsum(oracle.pca.explained_variance_ratio_)[:100])
n_comps = np.where(np.diff(np.diff(np.cumsum(oracle.pca.explained_variance_ratio_))>0.002))[0][0]
plt.axvline(n_comps, c="k")
plt.show()
print(n_comps)
n_comps = min(n_comps, 50)

n_cell = oracle.adata.shape[0]
k = int(0.025*n_cell)
oracle.knn_imputation(n_pca_dims=n_comps, k=k, balanced=True, b_sight=k*8, b_maxl=k*4, n_jobs=4)

# In[]
# --------------------------------------------------------------------
#
# Infer GRNs
#
# --------------------------------------------------------------------
links = oracle.get_links(cluster_name_for_GRN_unit="leiden", alpha=10, verbose_level=10)

# In[]
# --------------------------------------------------------------------
#
# Save results
#
# --------------------------------------------------------------------
print("Save results...")
grns_infer = np.zeros(shape = (adata.shape[0], adata.shape[1], adata.shape[1]))
# save grn inference result
for cluster_id in links.links_dict.keys():
    # direct output of celloracle
    grn_cluster = links.links_dict[cluster_id]
    # the coeff mean can be treated as the edge weight of the grn
    grn_coef_mean = pd.DataFrame(data = 0, index = adata.var.index.values, columns = adata.var.index.values)
    for i in range(grn_cluster.shape[0]):
        source = grn_cluster.loc[i, "source"]
        target = grn_cluster.loc[i, "target"]
        # row source, column target
        grn_coef_mean.loc[source, target] += grn_cluster.loc[i, "coef_mean"]
        # make symmetric
        grn_coef_mean.loc[target, source] += grn_cluster.loc[i, "coef_mean"]
    grns_infer[adata.obs["leiden"] == cluster_id,:,:] = grn_coef_mean.values[None,:,:]


# save results
np.save(file = result_dir + f"Gs_coef_mean_celloracle_{nclusts}.npy", arr = grns_infer)


# %%
