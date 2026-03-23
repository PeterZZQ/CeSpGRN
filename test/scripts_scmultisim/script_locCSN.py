# In[]
import locCSN
import scanpy as sc
import pandas as pd
import os
import anndata
import numpy as np 
import time
from sklearn.neighbors import NearestNeighbors

PROJECT_DIR = "/localscratch/ziqi/CeSpGRN/"

# # In[]
# # NOTE: Tutorial: 
# data_dir = PROJECT_DIR + "locCSN/DataStore/Velme/"
# data = sc.read_text(data_dir + 'Velme_log_mc_cpm_L.txt')
# data = data.transpose() 
# data.shape # 1778 metacells * 942 genes
# meta_L = pd.read_csv(data_dir + 'Velme_meta_mc_L.txt', sep = ' ') 
# meta_L.columns
# # Index(['sampleID', 'broad.cluster', 'cluster', 'diagnosis'], dtype='object')
# data.obs = meta_L
# # sc.pl.heatmap(data, data.var.index, groupby= ["cluster", "diagnosis"], dendrogram = False, swap_axes = True, 
# #               cmap='Wistia', figsize=(8,4))

# ct_name = "L4"
# data_L4 = data[data.obs.cluster == ct_name, :]
# data_L4.shape # 449 metacell * 942 genes
# mcknn = pd.read_csv(data_dir + 'mcknn100_' + ct_name + '.txt', sep = ' ')
# mcknn = mcknn.to_numpy()
# X_L4 = data_L4.X.transpose()

# start = time.time()
# csn_L4_sub = locCSN.csn_loc(X_L4[0:20, :], mcknn)
# end = time.time()
# print(end - start)
# # 25.824307203292847


# In[]
# Load the datasets
PROJECT_DIR = "/localscratch/ziqi/CeSpGRN/"
datasets = [
    "simulated_8000_20_10_100_0.01_0_0_4",
    "simulated_8000_20_10_100_0.01_1_0_4",
    "simulated_8000_20_10_100_0.01_2_0_4",
    "simulated_8000_20_10_100_0.1_0_0_4",
    "simulated_8000_20_10_100_0.1_1_0_4",
    "simulated_8000_20_10_100_0.1_2_0_4"
]
dataset = datasets[-1]
nclusts = 10
path = PROJECT_DIR + f"data/scMultiSim/{dataset}/"
result_dir = PROJECT_DIR + f"test/results_scmultisim/{dataset}/loccsn/"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# count scRNA-seq data
counts_rna = pd.read_csv(path + "counts_rna_true.txt", sep = "\t", index_col = 0).T.values
# normalization step
meta_cells = pd.read_csv(path + "meta_cells.txt", delimiter = "\t")
# create adata
adata_rna = anndata.AnnData(X = counts_rna, obs = meta_cells)
adata_rna.var.index = ["gene" + str(x + 1) for x in range(adata_rna.shape[1])]

# ground truth clusters
for i in range(0, nclusts, 1):
    threshold = i * 1/nclusts
    adata_rna.obs.loc[(adata_rna.obs["depth"] >= threshold) & (adata_rna.obs["depth"] <= threshold + 1/nclusts), "leiden"] = str(i)


# In[]
# Run locCSN for each cell group
loccsn_mtx = np.zeros((adata_rna.shape[0], adata_rna.shape[1], adata_rna.shape[1]))

for i in range(0, nclusts, 1):
    print(f"Running LocCSN for cluster {i}...")
    adata_clust = adata_rna[adata_rna.obs["leiden"] == str(i), :]
    # construct knn
    expr_clust = adata_clust.X.copy()
    # normalize the count
    expr_clust_norm = expr_clust/(expr_clust.sum(1, keepdims = True) + 1e-5) * 10e4
    expr_clust_norm = np.log1p(expr_clust_norm)

    k = 100  
    nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(expr_clust_norm)
    distances, indices = nbrs.kneighbors(expr_clust_norm)

    print("Main inference part")
    csn_i_sub = locCSN.csn_loc(expr_clust_norm.T, indices.T, ncore = 16)

    loccsn_mtx[(adata_rna.obs["leiden"] == str(i)).values, :, :] = np.concatenate([x.toarray()[None,:,:] for x in csn_i_sub], axis = 0)
    print("Done.")

np.save(result_dir + "Gs_loccsn.npy", loccsn_mtx)


