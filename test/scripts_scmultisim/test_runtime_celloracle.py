# In[]
import numpy as np 
import pandas as pd
import sys, os
import time
import celloracle as co
import anndata
import scanpy as sc
PROJECT_DIR = "/localscratch/ziqi/CeSpGRN/"

sys.path.append(PROJECT_DIR + 'src/')
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'


# In[]
def scale_dataset(counts, ncells_scale, ngenes_scale):

    if ncells_scale >= 1:
        counts_scale = np.repeat(counts, ncells_scale, axis = 0)
    else:
        ncells_new = int(ncells_scale * counts.shape[0])
        counts_scale = counts[:ncells_new, :]
    
    if ngenes_scale >= 1:
        counts_scale = np.repeat(counts_scale, ngenes_scale, axis = 1)
        if len(counts_scale.shape) == 2:
            pass
        elif len(counts_scale.shape) == 3:
            counts_scale = np.repeat(counts_scale, ngenes_scale, axis = 2)
        else:
            raise ValueError("shape not right")

    else:
        ngenes_new = int(ngenes_scale * counts_scale.shape[1])
        if len(counts_scale.shape) == 2:
            counts_scale = counts_scale[:, :ngenes_new]
        elif len(counts_scale.shape) == 3:
            counts_scale = counts_scale[:, :ngenes_new, :ngenes_new]
        else:
            raise ValueError("shape not right")
    

    print(f"number of cells after scaling: {counts_scale.shape[0]}")
    print(f"number of genes after scaling: {counts_scale.shape[1]}")
    return counts_scale


def scale_grnprior(grns, ncells_scale, ngenes_scale):
    if ncells_scale >= 1:
        grns_scale = np.repeat(grns, ncells_scale, axis = 0)
    else:
        ncells_new = int(ncells_scale * grns.shape[0])
        grns_scale = grns[:ncells_new, :]
    
    if ngenes_scale >= 1:
        grns_scale = np.repeat(grns_scale, ngenes_scale, axis = 1)
    else:
        ngenes_new = int(ngenes_scale * grns_scale.shape[1])
        grns_scale = grns_scale[:, :ngenes_new, :]

    return grns_scale

# In[]
dataset = "simulated_8000_20_10_100_0.01_0_0_4"
path = PROJECT_DIR + f"data/scMultiSim/{dataset}/"

counts_rna = pd.read_csv(path + "counts_rna_true.txt", sep = "\t", index_col = 0).T.values
meta_cells = pd.read_csv(path + "meta_cells.txt", delimiter = "\t")


# base GRN matrix, from scATAC-seq information, of the shape (ngenes, ntfs)
grn_priors = []
# all grn_prior share the same column names/regulators and index
regulators = pd.read_csv(path + f"grn_priors/grn_prior_1.txt", sep = "\t",index_col = 0).columns.values
# celloracle accept only one baseGRN, average
for i in range(1, counts_rna.shape[0] + 1):
    grn_priors.append(pd.read_csv(path + f"grn_priors/grn_prior_{i}.txt", sep = "\t",index_col = 0).values[None,:,:])
# (ncells, ngenes, ngenes)
grn_priors = np.concatenate(grn_priors, axis = 0)

# In[]

runtime_comb = pd.DataFrame(columns = ["ncells", "ngenes", "runtime (sec)"])

nclusts = 10
ngenes_scale = 1
for ncells_scale in [0.01, 0.02, 0.1, 0.25, 0.5, 1, 2]:
    counts_rna_scale = scale_dataset(counts_rna, ncells_scale = ncells_scale, ngenes_scale = ngenes_scale)
    grn_priors_scale = scale_grnprior(grn_priors, ncells_scale = ncells_scale, ngenes_scale = ngenes_scale)
    depth = scale_dataset(meta_cells[["depth"]].values, ncells_scale = ncells_scale, ngenes_scale = 1)
    depth = depth/np.max(depth)

    ncells, ngenes = counts_rna_scale.shape
    print(f"number of cells: {ncells}")
    print(f"number of genes: {ngenes}")

    result_dir = PROJECT_DIR + f"test/results_scmultisim/runtimes/runtime_{ncells}_{ngenes}/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # create adata
    adata = anndata.AnnData(X = counts_rna_scale)
    adata.var.index = ["gene" + str(x + 1) for x in range(adata.shape[1])]
    adata.obs["depth"] = depth

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

    # ground truth clusters
    for i in range(0, nclusts, 1):
        threshold = i * 1/nclusts
        adata.obs.loc[(adata.obs["depth"] >= threshold) & (adata.obs["depth"] <= threshold + 1/nclusts), "leiden"] = str(i)

    grn_priors_scale = pd.DataFrame(data = np.mean(grn_priors_scale, axis = 0)/np.max(np.mean(grn_priors_scale, axis = 0)), columns = ["gene" + str(x) for x in regulators], index = adata.var.index.values)

    start_time = time.time()
    
    print("Running cell oracle....")
    # Instantiate Oracle object
    oracle = co.Oracle()
    # In this notebook, we use the unscaled mRNA count for the nput of Oracle object.
    adata.X = adata.layers["raw_count"].copy()
    # Instantiate Oracle object.
    oracle.import_anndata_as_raw_count(adata=adata, cluster_column_name="leiden", embedding_name="X_pca", transform = "log2")

    # Creat TFinfo_dictionary, each key is a target gene, the value is the list of TF connecting to the target gene in baseGRN
    TF_to_TG_dictionary = {}
    for tf_id, tf in enumerate(grn_priors_scale.columns.values):
        # the input TF to TG is not weighted, if use 0, all are none 0, no different than using TF information
        # in addition, grn_prior cannot be negative
        threshold = 5e-2
        TF_to_TG_dictionary[tf] = [x for x in grn_priors_scale.index.values[grn_priors_scale.loc[:, tf] > threshold]] 

    TG_to_TF_dictionary = co.utility.inverse_dictionary(TF_to_TG_dictionary)

    # You can load TF info dataframe with the following code.
    oracle.import_TF_data(TFdict=TG_to_TF_dictionary)

   # knn imputation
    # Perform PCA
    oracle.perform_PCA()

    n_comps = np.where(np.diff(np.diff(np.cumsum(oracle.pca.explained_variance_ratio_))>0.002))[0][0]
    n_comps = min(n_comps, 50)

    n_cell = oracle.adata.shape[0]
    k = int(0.025*n_cell)
    oracle.knn_imputation(n_pca_dims=n_comps, k=k, balanced=True, b_sight=k*8, b_maxl=k*4, n_jobs=4)

    # Infer GRNs
    links = oracle.get_links(cluster_name_for_GRN_unit="leiden", alpha=10, verbose_level=10)


    end_time = time.time()
    runtime = end_time - start_time

    runtime_df = pd.DataFrame(columns = ["ncells", "ngenes", "runtime (sec)"])
    runtime_df["ncells"] = [ncells]
    runtime_df["ngenes"] = [ngenes]
    runtime_df["runtime (sec)"] = [runtime]
    runtime_comb = pd.concat([runtime_comb, runtime_df], axis = 0, ignore_index = True)

runtime_comb.to_csv(PROJECT_DIR + f"test/results_scmultisim/runtimes/runtime_ncells_celloracle.csv")

# In[]
# ------------------------------------------------------
#
# Runtime comparison for number of genes
#
# ------------------------------------------------------
# count data

runtime_comb = pd.DataFrame(columns = ["ncells", "ngenes", "runtime (sec)"])

nclusts = 10
ncells_scale = 0.05
for ngenes_scale in [0.25, 0.5, 1, 5, 7, 10]:
    counts_rna_scale = scale_dataset(counts_rna, ncells_scale = ncells_scale, ngenes_scale = ngenes_scale)
    grn_priors_scale = scale_grnprior(grn_priors, ncells_scale = ncells_scale, ngenes_scale = ngenes_scale)
    depth = scale_dataset(meta_cells[["depth"]].values, ncells_scale = ncells_scale, ngenes_scale = 1)
    depth = depth/np.max(depth)
    
    ncells, ngenes = counts_rna_scale.shape
    print(f"number of cells: {ncells}")
    print(f"number of genes: {ngenes}")

    result_dir = PROJECT_DIR + f"test/results_scmultisim/runtimes/runtime_{ncells}_{ngenes}/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # create adata
    adata = anndata.AnnData(X = counts_rna_scale)
    adata.var.index = ["gene" + str(x + 1) for x in range(adata.shape[1])]
    adata.obs["depth"] = depth

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

    # ground truth clusters
    for i in range(0, nclusts, 1):
        threshold = i * 1/nclusts
        adata.obs.loc[(adata.obs["depth"] >= threshold) & (adata.obs["depth"] <= threshold + 1/nclusts), "leiden"] = str(i)

    grn_priors_scale = pd.DataFrame(data = np.mean(grn_priors_scale, axis = 0)/np.max(np.mean(grn_priors_scale, axis = 0)), columns = ["gene" + str(x) for x in regulators], index = adata.var.index.values)

    start_time = time.time()
    
    print("Running cell oracle....")
    # Instantiate Oracle object
    oracle = co.Oracle()
    # In this notebook, we use the unscaled mRNA count for the nput of Oracle object.
    adata.X = adata.layers["raw_count"].copy()
    # Instantiate Oracle object.
    oracle.import_anndata_as_raw_count(adata=adata, cluster_column_name="leiden", embedding_name="X_pca", transform = "log2")

    # Creat TFinfo_dictionary, each key is a target gene, the value is the list of TF connecting to the target gene in baseGRN
    TF_to_TG_dictionary = {}
    for tf_id, tf in enumerate(grn_priors_scale.columns.values):
        # the input TF to TG is not weighted, if use 0, all are none 0, no different than using TF information
        # in addition, grn_prior cannot be negative
        threshold = 5e-2
        TF_to_TG_dictionary[tf] = [x for x in grn_priors_scale.index.values[grn_priors_scale.loc[:, tf] > threshold]] 

    TG_to_TF_dictionary = co.utility.inverse_dictionary(TF_to_TG_dictionary)

    # You can load TF info dataframe with the following code.
    oracle.import_TF_data(TFdict=TG_to_TF_dictionary)

   # knn imputation
    # Perform PCA
    oracle.perform_PCA()

    n_comps = np.where(np.diff(np.diff(np.cumsum(oracle.pca.explained_variance_ratio_))>0.002))[0][0]
    n_comps = min(n_comps, 50)

    n_cell = oracle.adata.shape[0]
    k = int(0.025*n_cell)
    oracle.knn_imputation(n_pca_dims=n_comps, k=k, balanced=True, b_sight=k*8, b_maxl=k*4, n_jobs=4)

    # Infer GRNs
    links = oracle.get_links(cluster_name_for_GRN_unit="leiden", alpha=10, verbose_level=10)


    end_time = time.time()
    runtime = end_time - start_time

    runtime_df = pd.DataFrame(columns = ["ncells", "ngenes", "runtime (sec)"])
    runtime_df["ncells"] = [ncells]
    runtime_df["ngenes"] = [ngenes]
    runtime_df["runtime (sec)"] = [runtime]
    runtime_comb = pd.concat([runtime_comb, runtime_df], axis = 0, ignore_index = True)


runtime_comb.to_csv(PROJECT_DIR + f"test/results_scmultisim/runtimes/runtime_ngenes_celloracle.csv")

# %%
