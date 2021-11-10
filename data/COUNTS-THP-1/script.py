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
# use raw count or log-transformed for scode?
counts.T.to_csv("expr.txt", header = False, index = False, sep = "\t")

# the distribution of count is log-normal
adata = AnnData(X = counts.values)
sc.pp.log1p(adata)
adata.uns['iroot'] = 0
sc.pp.neighbors(adata, n_neighbors = 30)
sc.tl.diffmap(adata, n_comps = 5)
sc.tl.dpt(adata, n_dcs = 5)

pt_est = adata.obs["dpt_pseudotime"].values.squeeze()
pt_est = pd.DataFrame(data = pt_est[:,None])
pt_est.to_csv("dpt_time.txt", sep = "\t", index = True, header = False)
sc.pl.diffmap(adata, color = "dpt_pseudotime")


# %%
