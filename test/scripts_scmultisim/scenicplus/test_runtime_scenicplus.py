# In[]
from scenicplus.cli.commands import infer_grn, infer_region_to_gene
import pandas as pd
import pathlib
import anndata 
import sys, os
import numpy as np
import time

sys.path.append(".")
from scenicplus_wrapper import *

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
PROJECT_DIR = "/localscratch/ziqi/CeSpGRN/"

nclusts = 10
dataset = "simulated_8000_20_10_100_0.01_0_0_4"
path = PROJECT_DIR + f"data/scMultiSim/{dataset}/scenicplus/"

# count scRNA-seq data
counts_rna = pd.read_csv(path + "counts_rna_true.txt", sep = "\t", index_col = 0).T.values
# normalization step
meta_cells = pd.read_csv(path + "meta_cells.txt", delimiter = "\t")

# base GRN matrix, from scATAC-seq information, of the shape (ngenes, ntfs)
grn_priors = []
region_tfs = []
# all grn_prior share the same column names/regulators and index
regulators = pd.read_csv(path + f"grn_priors/grn_prior_1.txt", sep = "\t",index_col = 0).columns.values
# celloracle accept only one baseGRN, average
for i in range(1, counts_rna.shape[0] + 1):
    grn_priors.append(pd.read_csv(path + f"grn_priors/grn_prior_{i}.txt", sep = "\t",index_col = 0).values[None,:,:])
    region_tfs.append(pd.read_csv(path + f"region_to_tfs/region_tf_{i}.txt", sep = "\t", index_col = 0).values[None,:,:])
grn_priors = np.concatenate(grn_priors, axis = 0)
region_tfs = np.concatenate(region_tfs, axis = 0)


region_gene = pd.read_csv(path + "region2gene.txt", sep = "\t").values
# region_gene = pd.DataFrame(data = region_gene, index = ["region" + str(x+1) for x in range(region_gene.shape[0])], columns = ["gene" + str(x+1) for x in range(region_gene.shape[1])])


# In[]

runtime_comb = pd.DataFrame(columns = ["ncells", "ngenes", "runtime (sec)"])

ngenes_scale = 1
for ncells_scale in [0.01, 0.02, 0.1, 0.25, 0.5, 1, 2]:
    counts_rna_scale = scale_dataset(counts_rna, ncells_scale = ncells_scale, ngenes_scale = ngenes_scale)
    region_tfs_scale = scale_grnprior(region_tfs, ncells_scale = ncells_scale, ngenes_scale = 1)
    if ngenes_scale >= 1:
        region_gene_scale = np.repeat(region_gene, ngenes_scale, axis = 1)
    else:
        ngenes_new = int(ngenes_scale * region_gene.shape[1])
        region_gene_scale = region_gene[:, :ngenes_new]
    region_gene_scale = pd.DataFrame(region_gene_scale, index = ["region" + str(x+1) for x in range(region_gene_scale.shape[0])], columns = ["gene" + str(x+1) for x in range(region_gene_scale.shape[1])])

    depth = scale_dataset(meta_cells[["depth"]].values, ncells_scale = ncells_scale, ngenes_scale = 1)
    depth = depth/np.max(depth)

    ncells, ngenes = counts_rna_scale.shape
    print(f"number of cells: {ncells}")
    print(f"number of genes: {ngenes}")

    result_dir = PROJECT_DIR + f"test/results_scmultisim/runtimes/runtime_{ncells}_{ngenes}/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    

    # create adata
    adata_rna = anndata.AnnData(X = counts_rna_scale)
    adata_rna.obs["depth"] = depth
    adata_rna.var.index = ["gene" + str(x + 1) for x in range(adata_rna.shape[1])]

    # ground truth clusters
    for i in range(0, nclusts, 1):
        threshold = i * 1/nclusts
        adata_rna.obs.loc[(adata_rna.obs["depth"] >= threshold) & (adata_rna.obs["depth"] <= threshold + 1/nclusts), "leiden"] = str(i)

    adata_rna.var["tf"] = False
    overlap_regulator = np.intersect1d(np.array(["gene" + str(x) for x in regulators]), adata_rna.var.index)
    adata_rna.var.loc[[x for x in overlap_regulator], "tf"] = True

    # calculate an average graph
    region_tfs_scale = pd.DataFrame(data = np.mean(region_tfs_scale, axis = 0)/np.max(np.mean(region_tfs_scale, axis = 0)), columns = ["gene" + str(x) for x in regulators], index = ["region" + str(x+1) for x in range(330)])

    region_to_gene_importances = {}
    region_to_gene_correlation = {}
    for gene in region_gene_scale.columns:
        region_acc = region_gene_scale.index[region_gene_scale[gene].values != 0]
        if len(region_acc) >= 1:
            # NOTE: NEED TO FILL IN THE CONTINUOUS VALUES, OR SCENIC WON'T FUNCTION
            region_to_gene_importances[gene] = pd.DataFrame(index = region_acc, columns = ["importance"], data = 0)
            region_to_gene_correlation[gene] = pd.DataFrame(index = region_acc, columns = ["correlation"], data = 0)
            for region in region_acc:
                acc_score = region_tfs_scale.loc[region, :].values
                region_to_gene_importances[gene].loc[region, "importance"] = np.mean(acc_score[acc_score != 0])
                region_to_gene_correlation[gene].loc[region, "correlation"] = np.mean(acc_score[acc_score != 0])

    # transform dictionaries to pandas dataframe
    region_gene_adj = pd.DataFrame(columns = ["target", "region", "importance", "rho"])
    for gene in region_to_gene_importances.keys():
        region_gene_adj = pd.concat([region_gene_adj, pd.DataFrame(data={'target': gene,
                                                    'region': region_to_gene_importances[gene].index.values[0],
                                                    'importance': region_to_gene_importances[gene].values[0],
                                                    'rho': region_to_gene_correlation[gene].loc[
                                                        region_to_gene_importances[gene].index.to_list()].values[0]})])

    region_gene_adj = region_gene_adj.reset_index()
    region_gene_adj = region_gene_adj.drop('index', axis=1)

    # no need to power
    region_gene_adj['importance_x_rho'] = region_gene_adj['rho']
    region_gene_adj['importance_x_abs_rho'] = abs(region_gene_adj['rho'])

    cistromes = anndata.AnnData(X = region_tfs_scale)

    fname_adj_tf_gene = result_dir + "tf_gene.csv" 
    dir_temp = result_dir + "tf_gene/" 
    if not os.path.exists(dir_temp):
        os.makedirs(dir_temp)

    start_time = time.time()
    infer_TF_to_gene(adata_rna = adata_rna,
                    temp_dir = dir_temp,
                    adj_out_fname = fname_adj_tf_gene,
                    method = "GBM",
                    n_cpu = 4,
                    seed = 666)

    tf_to_gene = pd.read_table(fname_adj_tf_gene)

    fname_grn = result_dir + "grn_infer.csv"
    dir_ranking_db = None # TODO: check what that is?
    infer_grn(tf_to_gene = tf_to_gene,
            region_to_gene = region_gene_adj,
            cistromes = cistromes,
            eRegulon_out_fname = fname_grn, 
            ranking_db_fname = dir_ranking_db,
            is_extended = False, 
            temp_dir = dir_temp,
            order_regions_to_genes_by = "importance",
            order_TFs_to_genes_by = "importance",
            gsea_n_perm = 1000,
            quantiles = (0.85, 0.90),
            top_n_regionTogenes_per_gene = (5, 10, 15),
            top_n_regionTogenes_per_region = (),
            binarize_using_basc = True, 
            min_regions_per_gene = 0,
            rho_dichotomize_tf2g = False, # whether to dichotomize tf-to-gene adj
            rho_dichotomize_r2g = False, # whether to dichotomize region-to-gene adj
            rho_dichotomize_eregulon = True, # whether to dichotomize eregulon
            keep_only_activating = True,
            rho_threshold = 0.03, # default
            min_target_genes = 5,
            n_cpu = 1,
            seed = 666)

    end_time = time.time()
    runtime = end_time - start_time

    runtime_df = pd.DataFrame(columns = ["ncells", "ngenes", "runtime (sec)"])
    runtime_df["ncells"] = [ncells]
    runtime_df["ngenes"] = [ngenes]
    runtime_df["runtime (sec)"] = [runtime]
    runtime_comb = pd.concat([runtime_comb, runtime_df], axis = 0, ignore_index = True)


runtime_comb.to_csv(PROJECT_DIR + f"test/results_scmultisim/runtimes/runtime_ncells_scenicplus.csv")


# In[]
runtime_comb = pd.DataFrame(columns = ["ncells", "ngenes", "runtime (sec)"])

ncells_scale = 0.05
for ngenes_scale in [0.5, 1, 5, 7, 10]:
    counts_rna_scale = scale_dataset(counts_rna, ncells_scale = ncells_scale, ngenes_scale = ngenes_scale)
    region_tfs_scale = scale_grnprior(region_tfs, ncells_scale = ncells_scale, ngenes_scale = 1)

    if ngenes_scale >= 1:
        region_gene_scale = np.repeat(region_gene, ngenes_scale, axis = 1)
    else:
        ngenes_new = int(ngenes_scale * region_gene.shape[1])
        region_gene_scale = region_gene[:, :ngenes_new]

    region_gene_scale = pd.DataFrame(region_gene_scale, index = ["region" + str(x+1) for x in range(region_gene_scale.shape[0])], columns = ["gene" + str(x+1) for x in range(region_gene_scale.shape[1])])


    depth = scale_dataset(meta_cells[["depth"]].values, ncells_scale = ncells_scale, ngenes_scale = 1)
    depth = depth/np.max(depth)

    ncells, ngenes = counts_rna_scale.shape
    print(f"number of cells: {ncells}")
    print(f"number of genes: {ngenes}")

    result_dir = PROJECT_DIR + f"test/results_scmultisim/runtimes/runtime_{ncells}_{ngenes}/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    

    # create adata
    adata_rna = anndata.AnnData(X = counts_rna_scale)
    adata_rna.obs["depth"] = depth
    adata_rna.var.index = ["gene" + str(x + 1) for x in range(adata_rna.shape[1])]

    # ground truth clusters
    for i in range(0, nclusts, 1):
        threshold = i * 1/nclusts
        adata_rna.obs.loc[(adata_rna.obs["depth"] >= threshold) & (adata_rna.obs["depth"] <= threshold + 1/nclusts), "leiden"] = str(i)

    adata_rna.var["tf"] = False
    overlap_regulator = np.intersect1d(np.array(["gene" + str(x) for x in regulators]), adata_rna.var.index)
    adata_rna.var.loc[[x for x in overlap_regulator], "tf"] = True

    # calculate an average graph
    # grn_priors_scale = pd.DataFrame(data = np.mean(grn_priors_scale, axis = 0)/np.max(np.mean(grn_priors_scale, axis = 0)), columns = ["gene" + str(x) for x in regulators], index = adata_rna.var.index.values)
    region_tfs_scale = pd.DataFrame(data = np.mean(region_tfs_scale, axis = 0)/np.max(np.mean(region_tfs_scale, axis = 0)), columns = ["gene" + str(x) for x in regulators], index = ["region" + str(x+1) for x in range(330)])

    # Step 1: infer the region to gene matrix

    region_to_gene_importances = {}
    region_to_gene_correlation = {}
    for gene in region_gene_scale.columns:
        region_acc = region_gene_scale.index[region_gene_scale[gene].values != 0]
        if len(region_acc) >= 1:
            # NOTE: NEED TO FILL IN THE CONTINUOUS VALUES, OR SCENIC WON'T FUNCTION
            region_to_gene_importances[gene] = pd.DataFrame(index = region_acc, columns = ["importance"], data = 0)
            region_to_gene_correlation[gene] = pd.DataFrame(index = region_acc, columns = ["correlation"], data = 0)
            for region in region_acc:
                acc_score = region_tfs_scale.loc[region, :].values
                region_to_gene_importances[gene].loc[region, "importance"] = np.mean(acc_score[acc_score != 0])
                region_to_gene_correlation[gene].loc[region, "correlation"] = np.mean(acc_score[acc_score != 0])

    # transform dictionaries to pandas dataframe
    region_gene_adj = pd.DataFrame(columns = ["target", "region", "importance", "rho"])
    for gene in region_to_gene_importances.keys():
        region_gene_adj = pd.concat([region_gene_adj, pd.DataFrame(data={'target': gene,
                                                    'region': region_to_gene_importances[gene].index.values[0],
                                                    'importance': region_to_gene_importances[gene].values[0],
                                                    'rho': region_to_gene_correlation[gene].loc[
                                                        region_to_gene_importances[gene].index.to_list()].values[0]})])

    region_gene_adj = region_gene_adj.reset_index()
    region_gene_adj = region_gene_adj.drop('index', axis=1)

    # no need to power
    region_gene_adj['importance_x_rho'] = region_gene_adj['rho']
    region_gene_adj['importance_x_abs_rho'] = abs(region_gene_adj['rho'])

    cistromes = anndata.AnnData(X = region_tfs_scale)

    fname_adj_tf_gene = result_dir + "tf_gene.csv" 
    dir_temp = result_dir + "tf_gene/" 
    if not os.path.exists(dir_temp):
        os.makedirs(dir_temp)

    start_time = time.time()

    infer_TF_to_gene(adata_rna = adata_rna,
                    temp_dir = dir_temp,
                    adj_out_fname = fname_adj_tf_gene,
                    method = "GBM",
                    n_cpu = 4,
                    seed = 666)

    tf_to_gene = pd.read_table(fname_adj_tf_gene)

    fname_grn = result_dir + "grn_infer.csv"
    dir_ranking_db = None # TODO: check what that is?

    infer_grn(tf_to_gene = tf_to_gene,
            region_to_gene = region_gene_adj,
            cistromes = cistromes,
            eRegulon_out_fname = fname_grn, 
            ranking_db_fname = dir_ranking_db,
            is_extended = False, 
            temp_dir = dir_temp,
            order_regions_to_genes_by = "importance",
            order_TFs_to_genes_by = "importance",
            gsea_n_perm = 1000,
            quantiles = (0.85, 0.90),
            top_n_regionTogenes_per_gene = (5, 10, 15),
            top_n_regionTogenes_per_region = (),
            binarize_using_basc = True, 
            min_regions_per_gene = 0,
            rho_dichotomize_tf2g = False, # whether to dichotomize tf-to-gene adj
            rho_dichotomize_r2g = False, # whether to dichotomize region-to-gene adj
            rho_dichotomize_eregulon = True, # whether to dichotomize eregulon
            keep_only_activating = True,
            rho_threshold = 0.03, # default
            min_target_genes = 5,
            n_cpu = 1,
            seed = 666)


    end_time = time.time()
    runtime = end_time - start_time

    runtime_df = pd.DataFrame(columns = ["ncells", "ngenes", "runtime (sec)"])
    runtime_df["ncells"] = [ncells]
    runtime_df["ngenes"] = [ngenes]
    runtime_df["runtime (sec)"] = [runtime]
    runtime_comb = pd.concat([runtime_comb, runtime_df], axis = 0, ignore_index = True)

runtime_comb.to_csv(PROJECT_DIR + f"test/results_scmultisim/runtimes/runtime_ngenes_scenicplus.csv")

# %%

