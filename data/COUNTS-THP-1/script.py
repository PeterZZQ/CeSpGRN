# In[0]
import numpy as np
import pandas as pd
from anndata import AnnData
import scanpy as sc

counts = pd.read_csv("counts.csv", sep = ",", index_col = 0)
anno = np.array(["0h"] * 120 + ["1h"] * 120 + ["6h"] * 120 + ["12h"] * 120 + ["24h"] * 120 + ["48h"] * 120 + ["72h"] * 120 + ["96h"] * 120)[:,None]
anno = pd.DataFrame(data = anno, columns = ["time"])
anno.index = counts.index.values
anno.to_csv("anno.csv")
genes = pd.DataFrame(data = counts.columns.values[:,None])
genes.to_csv("genes.csv", index = False, header = False)
counts.T.to_csv("expr.txt", header = False, index = False, sep = "\t")

adata = AnnData(X = counts.values)
adata.uns['iroot'] = 0
sc.pp.neighbors(adata, n_neighbors = 30)
sc.tl.diffmap(adata, n_comps = 5)
sc.tl.dpt(adata, n_dcs = 5)

pt_est = adata.obs["dpt_pseudotime"].values.squeeze()
pt_est = pd.DataFrame(data = pt_est[:,None])
pt_est.to_csv("dpt_time.txt", sep = "\t", index = True, header = False)
sc.pl.diffmap(adata, color = "dpt_pseudotime")


# In[] GRN 
gt = pd.read_excel("gt_matrix.xlsx")

grn_pos = {}
grn_neg = {}
for idx, row in gt.iterrows():
    if row["direction"] == "activate":
        if not pd.isna(row["Input gene"]):
            tf = row["Input gene"]
            if tf not in grn_pos.keys():
                grn_pos[tf] = []

        grn_pos[tf].extend([x for x in row[2:].values if not pd.isna(x)])

    elif row["direction"] == "suppress":
        if not pd.isna(row["Input gene"]):
            tf = row["Input gene"]
            if tf not in grn_neg.keys():
                grn_neg[tf] = []

        grn_neg[tf].extend([x for x in row[2:].values if not pd.isna(x)])       
    else:
        raise ValueError("either activate or suppress")

# In[]
tfs = sorted(list(set([x for x in grn_pos.keys()]).union([x for x in grn_neg.keys()])))
targets = []
for tf in grn_pos.keys():
    genes = grn_pos[tf]
    for gene in genes:
        if gene not in tfs:
            targets.append(gene)

for tf in grn_neg.keys():
    genes = grn_neg[tf]
    for gene in genes:
        if gene not in tfs:
            targets.append(gene)

targets = sorted(list(set(targets)))

grn = pd.DataFrame(index = tfs + targets, columns = tfs + targets, data = 0)
for tf in grn_pos.keys():
    genes = grn_pos[tf]
    grn.loc[tf, genes] += 1
print(np.max(grn.values))
for tf in grn_neg.keys():
    genes = grn_neg[tf]
    assert np.sum(grn.loc[tf, genes].values) == 0
    grn.loc[tf, genes] -= 1
print(np.min(grn.values))


# In[]

genes = [x for x in pd.read_csv("genes.csv", header = None).values.squeeze()]
overlap_genes = set(genes).intersection(set(tfs + targets))

assert len(overlap_genes) == len(genes)

grn = grn.loc[genes, genes]

grn.to_csv("GRN_static.csv")

# %%

