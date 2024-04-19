
# In[0]
import sys, os
sys.path.append('../../src/')

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import anndata
import scanpy as sc
import warnings
warnings.filterwarnings("ignore")

path = "../../data/nmp/"
# In[]

adata_rna_wt = anndata.read_h5ad(path + "adata_rna_wt.h5ad")
adata_rna_ko = anndata.read_h5ad(path + "adata_rna_ko.h5ad")
adata_atac_wt = anndata.read_h5ad(path + "adata_atac_wt.h5ad")
adata_atac_ko = anndata.read_h5ad(path + "adata_atac_ko.h5ad")
adata_atac_wt.X = (adata_atac_wt.X > 0)
adata_atac_ko.X = (adata_atac_ko.X > 0)

adata_rna = anndata.concat([adata_rna_wt, adata_rna_ko])
adata_atac = anndata.concat([adata_atac_wt, adata_atac_ko])


# In[]
# Infer mask matrix
peak_motif_ko = pd.read_csv(path + "peak_motif_ko.csv", index_col = 0)
peak_motif_ko = (peak_motif_ko > 0)
peak_motif_wt = pd.read_csv(path + "peak_motif_ko.csv", index_col = 0)
peak_motif_wt = (peak_motif_wt > 0)
motif_tf = pd.read_csv(path + "motif_tf.csv", index_col = 0)
TFs = np.unique(motif_tf["TF"].values)

peak_tf_ko = pd.DataFrame(data = 0, index = peak_motif_ko.index.values, columns = TFs)
for motif in peak_motif_ko.columns:
    tfs = motif_tf.loc[motif_tf["motif"] == motif, "TF"].values
    peak_tf_ko.loc[:, tfs] = peak_tf_ko.loc[:, tfs].values + peak_motif_ko.loc[:, [motif]].values

peak_tf_wt = pd.DataFrame(data = 0, index = peak_motif_wt.index.values, columns = TFs)
for motif in peak_motif_wt.columns:
    tfs = motif_tf.loc[motif_tf["motif"] == motif, "TF"].values
    peak_tf_wt.loc[:, tfs] = peak_tf_wt.loc[:, tfs].values + peak_motif_wt.loc[:, [motif]].values

# # on average, only 3.6 regions link to one gene (upstream 2kb)
# peak_target_df = pd.read_csv(path + "gact_upstream_2000.csv", index_col = 0)
# on average, 4.5 regions link to one gene (upstream 50kb)
peak_target_df = pd.read_csv(path + "gact_upstream_50000.csv", index_col = 0)
# # on average, 4.8 regions link to one gene (upstream 500kb)
# peak_target_df = pd.read_csv(path + "gact_upstream_50000.csv", index_col = 0)

peak_target = pd.DataFrame(data = 0, index = adata_atac.var.index.values, columns = adata_rna.var.index.values)
for row_idx in range(peak_target_df.shape[0]):
    if peak_target_df.iloc[row_idx, 1] in adata_rna.var.index.values:
        peak_target.loc[peak_target_df.iloc[row_idx, 0], peak_target_df.iloc[row_idx, 1]] += 1

# expand to include all regions
peak_tf_ko_full = pd.DataFrame(data = 0, index = adata_atac.var.index.values, columns = peak_tf_ko.columns.values)
peak_tf_ko_full.loc[peak_tf_ko.index.values,:] = peak_tf_ko.values
peak_tf_wt_full = pd.DataFrame(data = 0, index = adata_atac.var.index.values, columns = peak_tf_wt.columns.values)
peak_tf_wt_full.loc[peak_tf_wt.index.values,:] = peak_tf_wt.values

# In[]
# cell level mask
# find TF in the gene list
TFs_overlap = set([x for x in TFs]).intersection(set([x for x in adata_rna.var.index.values]))  
TFs_overlap = np.sort(np.array(list(TFs_overlap)))

# filter adata_rna, or no space for the calculation, 500 genes
sc.pp.filter_genes(adata_rna, min_counts = 20)
sc.pp.normalize_per_cell(adata_rna)
sc.pp.log1p(adata_rna)
sc.pp.highly_variable_genes(adata_rna, n_top_genes = 500)
genes = adata_rna.var.index.values[adata_rna.var["highly_variable"].values]
genes = np.concatenate([genes, TFs_overlap])
genes = np.unique(genes)
# adata_rna_ko = anndata.read_h5ad(path + "adata_rna_ko_filtered.h5ad")
# adata_rna_wt = anndata.read_h5ad(path + "adata_rna_wt_filtered.h5ad")
# genes = adata_rna_ko.var.index.values.squeeze()
# assert np.allclose(genes, adata_rna_wt.var.index.values.squeeze())

peak_target_sub = peak_target.loc[:,genes]
mask_cell_level = np.zeros((adata_rna_wt.shape[0], genes.shape[0], genes.shape[0]))
for celli in range(adata_rna_wt.shape[0]):
    print(celli)
    peak_tf_wt_i = peak_tf_wt_full.loc[:, TFs_overlap]
    accessible = adata_atac_wt.X[celli,:].toarray()
    peak_tf_wt_i = csr_matrix(peak_tf_wt_i.values * accessible.reshape(-1, 1))
    mask = pd.DataFrame(data = 0, index = genes, columns = genes)
    mask.loc[TFs_overlap,:] = (peak_tf_wt_i.T.dot(csr_matrix(peak_target_sub.values))).toarray()
    mask_cell_level[celli,:,:] = ((mask.values + mask.values.T) > 0)

np.save(path + "cell_specific_mask_wt_50000.npy", mask_cell_level)

mask_cell_level = np.zeros((adata_rna_ko.shape[0], genes.shape[0], genes.shape[0]))
for celli in range(adata_rna_ko.shape[0]):
    print(celli)
    peak_tf_ko_i = peak_tf_ko_full.loc[:, TFs_overlap]
    accessible = adata_atac_ko.X[celli,:].toarray()
    peak_tf_ko_i = csr_matrix(peak_tf_ko_i.values * accessible.reshape(-1, 1))
    mask = pd.DataFrame(data = 0, index = genes, columns = genes)
    mask.loc[TFs_overlap,:] = (peak_tf_ko_i.T.dot(csr_matrix(peak_target_sub.values))).toarray()
    mask_cell_level[celli,:,:] = ((mask.values + mask.values.T) > 0)

np.save(path + "cell_specific_mask_ko_50000.npy", mask_cell_level)

# In[]
# filtered scRNA-seq, mask and scRNA-seq are used for GRN inference in CeSpGRN
adata_rna_ko = adata_rna_ko[:, genes]
adata_rna_wt = adata_rna_wt[:, genes]
adata_rna_ko.write_h5ad(path + "adata_rna_ko_filtered_50000.h5ad")
adata_rna_wt.write_h5ad(path + "adata_rna_wt_filtered_50000.h5ad")


# In[]





