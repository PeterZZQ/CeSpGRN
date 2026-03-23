# In[]
import time
from sklearn.neighbors import NearestNeighbors
import sys, os
import numpy as np 
import scanpy as sc
import pandas as pd

PROJECT_DIR = "/localscratch/ziqi/CeSpGRN/"
sys.path.append(PROJECT_DIR + 'src/')

import genie3

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
nclusts = 10
dataset = "simulated_8000_20_10_100_0.01_0_0_4"
path = PROJECT_DIR + f"data/scMultiSim/{dataset}/"

counts_rna = pd.read_csv(path + "counts_rna_true.txt", sep = "\t", index_col = 0).T.values
meta_cells = pd.read_csv(path + "meta_cells.txt", delimiter = "\t")


# In[]
runtime_comb = pd.DataFrame(columns = ["ncells", "ngenes", "runtime (sec)"])

ngenes_scale = 1
for ncells_scale in [0.02, 0.1, 0.25, 0.5, 1, 2]:
    counts_rna_scale = scale_dataset(counts_rna, ncells_scale = ncells_scale, ngenes_scale = ngenes_scale)
    depth = scale_dataset(meta_cells[["depth"]].values, ncells_scale = ncells_scale, ngenes_scale = 1)
    depth = depth/np.max(depth)

    ncells, ngenes = counts_rna_scale.shape
    print(f"number of cells: {ncells}")
    print(f"number of genes: {ngenes}")

    result_dir = PROJECT_DIR + f"test/results_scmultisim/runtimes/runtime_{ncells}_{ngenes}/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    start_time = time.time()

    genie_theta = genie3.GENIE3(counts_rna_scale, gene_names=["gene_" + str(x+1) for x in range(ngenes)], regulators='all',tree_method='RF',K='sqrt',ntrees=1000,nthreads=1)

    end_time = time.time()
    runtime = end_time - start_time

    runtime_df = pd.DataFrame(columns = ["ncells", "ngenes", "runtime (sec)"])
    runtime_df["ncells"] = [ncells]
    runtime_df["ngenes"] = [ngenes]
    runtime_df["runtime (sec)"] = [runtime]
    runtime_comb = pd.concat([runtime_comb, runtime_df], axis = 0, ignore_index = True)

runtime_comb.to_csv(PROJECT_DIR + f"test/results_scmultisim/runtimes/runtime_ncells_genie3.csv")

# In[]
runtime_comb = pd.DataFrame(columns = ["ncells", "ngenes", "runtime (sec)"])

ncells_scale = 0.05
for ngenes_scale in [0.25, 0.5, 1, 5, 7, 10]:
    counts_rna_scale = scale_dataset(counts_rna, ncells_scale = ncells_scale, ngenes_scale = ngenes_scale)
    depth = scale_dataset(meta_cells[["depth"]].values, ncells_scale = ncells_scale, ngenes_scale = 1)
    depth = depth/np.max(depth)

    ncells, ngenes = counts_rna_scale.shape
    print(f"number of cells: {ncells}")
    print(f"number of genes: {ngenes}")

    result_dir = PROJECT_DIR + f"test/results_scmultisim/runtimes/runtime_{ncells}_{ngenes}/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    start_time = time.time()

    genie_theta = genie3.GENIE3(counts_rna_scale, gene_names=["gene_" + str(x+1) for x in range(ngenes)], regulators='all',tree_method='RF',K='sqrt',ntrees=1000,nthreads=1)

    end_time = time.time()
    runtime = end_time - start_time

    runtime_df = pd.DataFrame(columns = ["ncells", "ngenes", "runtime (sec)"])
    runtime_df["ncells"] = [ncells]
    runtime_df["ngenes"] = [ngenes]
    runtime_df["runtime (sec)"] = [runtime]
    runtime_comb = pd.concat([runtime_comb, runtime_df], axis = 0, ignore_index = True)

runtime_comb.to_csv(PROJECT_DIR + f"test/results_scmultisim/runtimes/runtime_ngenes_genie3.csv")