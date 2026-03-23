# In[]
from scenicplus.cli.commands import infer_grn, infer_region_to_gene
import pandas as pd
import pathlib
import anndata 
import sys, os
import numpy as np

sys.path.append(".")
from scenicplus_wrapper import *


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
dataset = datasets[5]
nclusts = 10
path = PROJECT_DIR + f"data/scMultiSim/{dataset}/scenicplus/"
result_dir = PROJECT_DIR + f"test/results_scmultisim/{dataset}/scenicplus/"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# count scRNA-seq data
counts_rna = pd.read_csv(path + "counts_rna_true.txt", sep = "\t", index_col = 0).T.values
# count scATAC-seq data
counts_atac = pd.read_csv(path + "counts_atac_true.txt", sep = "\t", index_col = 0).T.values
# normalization step
meta_cells = pd.read_csv(path + "meta_cells.txt", delimiter = "\t")
# create adata
adata_rna = anndata.AnnData(X = counts_rna, obs = meta_cells)
adata_rna.var.index = ["gene" + str(x + 1) for x in range(adata_rna.shape[1])]

# ground truth clusters
for i in range(0, nclusts, 1):
    threshold = i * 1/nclusts
    adata_rna.obs.loc[(adata_rna.obs["depth"] >= threshold) & (adata_rna.obs["depth"] <= threshold + 1/nclusts), "leiden"] = str(i)

# base GRN matrix, from scATAC-seq information, of the shape (ngenes, ntfs)
grn_priors = []
region_tfs = []
# all grn_prior share the same column names/regulators and index
regulators = pd.read_csv(path + f"grn_priors/grn_prior_1.txt", sep = "\t",index_col = 0).columns.values
# celloracle accept only one baseGRN, average
for i in range(1, adata_rna.shape[0] + 1):
    grn_priors.append(pd.read_csv(path + f"grn_priors/grn_prior_{i}.txt", sep = "\t",index_col = 0).values[None,:,:])
    region_tfs.append(pd.read_csv(path + f"region_to_tfs/region_tf_{i}.txt", sep = "\t", index_col = 0).values[None,:,:])
grn_priors = np.concatenate(grn_priors, axis = 0)
region_tfs = np.concatenate(region_tfs, axis = 0)

# calculate an average graph
grn_priors = pd.DataFrame(data = np.mean(grn_priors, axis = 0)/np.max(np.mean(grn_priors, axis = 0)), columns = ["gene" + str(x) for x in regulators], index = adata_rna.var.index.values)
region_tfs = pd.DataFrame(data = np.mean(region_tfs, axis = 0)/np.max(np.mean(region_tfs, axis = 0)), columns = ["gene" + str(x) for x in regulators], index = ["region" + str(x+1) for x in range(330)])

# assign tf
adata_rna.var["tf"] = False
adata_rna.var.loc[grn_priors.columns.values, "tf"] = True

# In[]
# Step 1: infer the region to gene matrix
region_gene = pd.read_csv(path + "region2gene.txt", sep = "\t").values
region_gene = pd.DataFrame(data = region_gene, index = ["region" + str(x+1) for x in range(region_gene.shape[0])], columns = ["gene" + str(x+1) for x in range(region_gene.shape[1])])

region_to_gene_importances = {}
region_to_gene_correlation = {}
for gene in region_gene.columns:
    region_acc = region_gene.index[region_gene[gene].values != 0]
    if len(region_acc) >= 1:
        # NOTE: NEED TO FILL IN THE CONTINUOUS VALUES, OR SCENIC WON'T FUNCTION
        region_to_gene_importances[gene] = pd.DataFrame(index = region_acc, columns = ["importance"], data = 0)
        region_to_gene_correlation[gene] = pd.DataFrame(index = region_acc, columns = ["correlation"], data = 0)
        for region in region_acc:
            acc_score = region_tfs.loc[region, :].values
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

# for idx in region_gene_adj.index.values:
#     region_gene_adj.loc[idx, "importance"] = [region_gene_adj.loc[idx, "importance"]]
#     region_gene_adj.loc[idx, "rho"] = [region_gene_adj.loc[idx, "rho"]]
#     region_gene_adj.loc[idx, "importance_x_rho"] = [region_gene_adj.loc[idx, "importance_x_rho"]]
#     region_gene_adj.loc[idx, "importance_x_abs_rho"] = [region_gene_adj.loc[idx, "importance_x_abs_rho"]]
    
# if add_distance:
#     search_space_rn = search_space.rename(
#         {'Name': 'region', 'Gene': 'target'}, axis=1).copy()
#     result_df = result_df.merge(search_space_rn, on=['region', 'target'])
#     #result_df['Distance'] = result_df['Distance'].map(lambda x: x[0])


# In[]
# Step 2: infer the tf to region matrix
# tf_region = pd.DataFrame(data = 0.0, index = grn_priors.columns.values, columns = ["region" + str(x+1) for x in range(region_gene.shape[0])])

# tf_region_dict = {tf: [] for tf in grn_priors.columns.values}
# for gene in region_gene.columns.values:
#     # non-zero values denote the regions that activate the gene
#     regions_act = region_gene.loc[:, gene].values
#     # non-zero values denote the tfs that activate the gene
#     tfs_act = grn_priors.loc[gene, :].values
#     for tf, val in zip(grn_priors.columns.values, tfs_act):
#         if val != 0:
#             open_region = region_gene.index[np.where(regions_act!=0)[0]].values
#             if len(open_region) > 0:
#                 for region in open_region:
#                     tf_region_dict[tf].append(region)
#                     tf_region.loc[tf, region] += 1

# cistromes columns are tfs, rows are region, should be a binary matrix
# cistromes = anndata.AnnData(X = tf_region.T.astype(int))

cistromes = anndata.AnnData(X = region_tfs)

# In[]
# Step 3: infer the tf to gene matrix
fname_adj_tf_gene = result_dir + "tf_gene.csv" 
dir_temp = result_dir + "tf_gene/" 
if not os.path.exists(dir_temp):
    os.makedirs(dir_temp)
infer_TF_to_gene(adata_rna = adata_rna,
                 temp_dir = dir_temp,
                 adj_out_fname = fname_adj_tf_gene,
                 method = "GBM",
                 n_cpu = 4,
                 seed = 666)

tf_to_gene = pd.read_table(fname_adj_tf_gene)

# In[]
# Step 4: get the GRNs
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

# In[]
# read in the inferred GRN
fname_grn = result_dir + "grn_infer.csv"
grn = pd.read_csv(fname_grn, sep = "\t")

grn_mtx = pd.DataFrame(index = grn_priors.index.values, columns = grn_priors.index.values, data = 0.0)
# TODO: transform into the grn matrix
for row in grn.index.values:
    gene_target = grn.loc[row, "Gene"]
    gene_tf = grn.loc[row, "TF"]
    grn_mtx.loc[gene_target, gene_tf] += grn.loc[row, "importance_TF2G"] * grn.loc[row, "rho_TF2G"]
grn_mtx = np.concatenate([grn_mtx.values[None, :, :]] * counts_rna.shape[0], axis = 0)

np.save(result_dir + "Gs_scenicplus.npy", grn_mtx)

# %%
