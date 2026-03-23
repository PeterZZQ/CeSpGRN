# In[]
import sys, os
PROJECT_DIR = "/localscratch/ziqi/CeSpGRN/"

sys.path.append(PROJECT_DIR + 'src/')
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from umap import UMAP
import g_admm, kernel, genie3, bmk, cov
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import seaborn as sns
import time

plt.rcParams.update(plt.rcParamsDefault)

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)
plt.rcParams["font.size"] = 20


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


# In[]
dataset = "simulated_8000_20_10_100_0.01_0_0_4"
path = f"../../data/scMultiSim/{dataset}/"


# default CeSpGRN setting
truncate_param = 100
bandwidths = [0.01, 0.1, 1, 10]
beta = 1
lamb_list = [0.01, 0.05, 0.1, 0.5]
use_tf = False

# In[]
# ------------------------------------------------------
#
# Runtime comparison for number of cells
#
# ------------------------------------------------------

counts_rna = pd.read_csv(path + "counts_rna_true.txt", sep = "\t", index_col = 0).T.values

runtime_comb = pd.read_csv(PROJECT_DIR + f"test/results_scmultisim/runtimes/runtime_ncells_cespgrn_{770}.csv", index_col = 0)


ngenes_scale = 7
# for ncells_scale in [0.02, 0.1, 0.25, 0.5, 1]:
for ncells_scale in [0.25, 0.5, 1]:
    
    counts_rna_scale = scale_dataset(counts_rna, ncells_scale = ncells_scale, ngenes_scale = ngenes_scale)
    ncells, ngenes = counts_rna_scale.shape
    print(f"number of cells: {ncells}")
    print(f"number of genes: {ngenes}")

    result_dir = PROJECT_DIR + f"test/results_scmultisim/runtimes/runtime_{ncells}_{ngenes}/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # normalization step
    libsize = np.median(np.sum(counts_rna_scale, axis = 1))
    counts_rna_norm = counts_rna_scale /(np.sum(counts_rna_scale, axis = 1, keepdims = True) + 1e-6) * libsize
    counts_rna_norm = np.log1p(counts_rna_norm)

    if (beta > 0) and use_tf:
        masks = np.load(PROJECT_DIR + f"test/results_scmultisim/{dataset}/masks_tf.npy")
        masks = scale_dataset(masks, ncells_scale = ncells_scale, ngenes_scale = ngenes_scale)
    elif beta > 0:
        masks = np.load(PROJECT_DIR + f"test/results_scmultisim/{dataset}/masks_atac.npy")
        masks = scale_dataset(masks, ncells_scale = ncells_scale, ngenes_scale = ngenes_scale)
    else:
        masks = None

    start_time = time.time()
    # select only one set of parameter
    bandwidth = bandwidths[0]
    lamb = lamb_list[0]

    X_pca = PCA(n_components = 20).fit_transform(counts_rna_norm)
    K, K_trun = kernel.calc_kernel_neigh(X_pca, bandwidth = bandwidth, truncate = True, truncate_param = truncate_param)

    empir_cov = cov.est_cov_ggm_para(X = counts_rna_norm, K_trun = K_trun, njobs = 8)

    print("scanning...")
    print("bandwidth: " + str(bandwidth))
    print("lambda: " + str(lamb))
    print("number of threads: " + str(torch.get_num_threads()))
    alpha = 2
    rho = 1.7
    max_iters = 100

    # test model without TF
    gadmm_batch = g_admm.G_admm_mask(X = counts_rna_norm[:, None, :], K = K, mask = masks, pre_cov = empir_cov, batchsize = 256, device = device)
    w_empir_cov = gadmm_batch.w_empir_cov.detach().cpu().numpy()
    Gs = gadmm_batch.train(max_iters = max_iters, n_intervals = 100, alpha = alpha, lamb = lamb, rho = rho, theta_init_offset = 0.1, beta = beta)
    end_time = time.time()

    runtime = end_time - start_time

    runtime_df = pd.DataFrame(columns = ["ncells", "ngenes", "runtime (sec)"])
    runtime_df["ncells"] = [ncells]
    runtime_df["ngenes"] = [ngenes]
    runtime_df["runtime (sec)"] = [runtime]
    runtime_comb = pd.concat([runtime_comb, runtime_df], axis = 0, ignore_index = True)


    sparsity = np.sum(Gs != 0)/(Gs.shape[0] * Gs.shape[1] * Gs.shape[2])
    print("sparsity: " + str(sparsity))

    runtime_comb.to_csv(PROJECT_DIR + f"test/results_scmultisim/runtimes/runtime_ncells_cespgrn_{ngenes}.csv")

assert False


# In[]
# ------------------------------------------------------
#
# Runtime comparison for number of genes
#
# ------------------------------------------------------
# count data
'''
counts_rna = pd.read_csv(path + "counts_rna_true.txt", sep = "\t", index_col = 0).T.values

runtime_comb = pd.DataFrame(columns = ["ncells", "ngenes", "runtime (sec)"])

ncells_scale = 0.05
for ngenes_scale in [0.25, 0.5, 1, 5, 7, 10]:
    counts_rna_scale = scale_dataset(counts_rna, ncells_scale = ncells_scale, ngenes_scale = ngenes_scale)
    ncells, ngenes = counts_rna_scale.shape
    # print(f"number of cells: {ncells}")
    # print(f"number of genes: {ngenes}")

    result_dir = PROJECT_DIR + f"test/results_scmultisim/runtimes/runtime_{ncells}_{ngenes}/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # normalization step
    libsize = np.median(np.sum(counts_rna_scale, axis = 1))
    counts_rna_norm = counts_rna_scale /(np.sum(counts_rna_scale, axis = 1, keepdims = True) + 1e-6) * libsize
    counts_rna_norm = np.log1p(counts_rna_norm)

    if (beta > 0) and use_tf:
        masks = np.load(PROJECT_DIR + f"test/results_scmultisim/{dataset}/masks_tf.npy")
        masks = scale_dataset(masks, ncells_scale = ncells_scale, ngenes_scale = ngenes_scale)
    elif beta > 0:
        masks = np.load(PROJECT_DIR + f"test/results_scmultisim/{dataset}/masks_atac.npy")
        masks = scale_dataset(masks, ncells_scale = ncells_scale, ngenes_scale = ngenes_scale)
    else:
        masks = None

    start_time = time.time()
    # select only one set of parameter
    bandwidth = bandwidths[0]
    lamb = lamb_list[0]

    X_pca = PCA(n_components = 20).fit_transform(counts_rna_norm)
    K, K_trun = kernel.calc_kernel_neigh(X_pca, bandwidth = bandwidth, truncate = True, truncate_param = truncate_param)

    if os.path.isfile(result_dir + "cov_" + str(bandwidth) + "_" + str(truncate_param) + ".npy"):
        print("load covariance matrix")
        empir_cov = np.load(result_dir + "cov_" + str(bandwidth) + "_" + str(truncate_param) + ".npy")
        print(f"covariance shape ({empir_cov.shape})")
    else:
        print("calculate covariance matrix")
        empir_cov = cov.est_cov_ggm_para(X = counts_rna_norm, K_trun = K_trun, njobs = 8)
        print(f"covariance shape ({empir_cov.shape})")
        np.save(result_dir + "cov_" + str(bandwidth) + "_" + str(truncate_param) + ".npy", empir_cov)

    print("scanning...")
    print("bandwidth: " + str(bandwidth))
    print("lambda: " + str(lamb))
    print("number of threads: " + str(torch.get_num_threads()))
    alpha = 2
    rho = 1.7
    max_iters = 100

    # test model without TF
    gadmm_batch = g_admm.G_admm_mask(X = counts_rna_norm[:, None, :], K = K, mask = masks, pre_cov = empir_cov, batchsize = 64, device = device)
    w_empir_cov = gadmm_batch.w_empir_cov.detach().cpu().numpy()
    Gs = gadmm_batch.train(max_iters = max_iters, n_intervals = 100, alpha = alpha, lamb = lamb, rho = rho, theta_init_offset = 0.1, beta = beta)
    end_time = time.time()

    runtime = end_time - start_time

    runtime_df = pd.DataFrame(columns = ["ncells", "ngenes", "runtime (sec)"])
    runtime_df["ncells"] = [ncells]
    runtime_df["ngenes"] = [ngenes]
    runtime_df["runtime (sec)"] = [runtime]
    runtime_comb = pd.concat([runtime_comb, runtime_df], axis = 0, ignore_index = True)


    sparsity = np.sum(Gs != 0)/(Gs.shape[0] * Gs.shape[1] * Gs.shape[2])
    print("sparsity: " + str(sparsity))
    np.save(file = result_dir + "precision_" + str(bandwidth) + "_" + str(lamb) + "_" + str(truncate_param) + "_"+ str(beta) + ".npy", arr = gadmm_batch.thetas) 


    runtime_comb.to_csv(PROJECT_DIR + f"test/results_scmultisim/runtimes/runtime_ngenes.csv")
'''

# In[]
# --------------------------------------------------------
#
# Plot the runtime curve with ncells
#
# --------------------------------------------------------
runtime_ncells_comb = pd.DataFrame()
for ngenes in [110, 220, 550, 770]:
    runtime_ncells = pd.read_csv(PROJECT_DIR + f"test/results_scmultisim/runtimes/runtime_ncells_cespgrn_{ngenes}.csv", index_col = 0)
    runtime_ncells_comb = pd.concat([runtime_ncells_comb, runtime_ncells], axis = 0)
runtime_ncells_comb["runtime (min)"] = runtime_ncells_comb["runtime (sec)"]/60
runtime_ncells_comb = runtime_ncells_comb.loc[runtime_ncells_comb["ncells"].isin([160, 800, 2000, 4000]), :]

sns.set_theme(font_scale = 1.5)
fig = plt.figure(figsize = (5, 4))
ax = fig.add_subplot()
sns.lineplot(data = runtime_ncells_comb, x = "ncells", y = "runtime (min)", markers=True, dashes=False, ax = ax, style = "ngenes", hue = "ngenes", palette = "tab10", marker_size = 10)
# ax.get_legend().remove()
ax.legend(
    bbox_to_anchor=(1.02, 1),   # position outside
    loc="upper left",
    borderaxespad=0,
    frameon=False,
    title="Number of Genes",

)
# ax.set_title("Total runtime")
ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))  # force scientific notation
fig.savefig(PROJECT_DIR + "test/results_scmultisim/runtimes/runtime_ncells.png", dpi = 300, bbox_inches = "tight")

# In[]
# --------------------------------------------------------
#
# Plot the runtime curve with ngenes
#
# --------------------------------------------------------
runtime_ncells_comb = pd.DataFrame()
for run in [0]:
    runtime_ncells = pd.read_csv(PROJECT_DIR + f"test/results_scmultisim/runtimes/runtime_ngenes.csv", index_col = 0)
    runtime_ncells["run"] = 0
    runtime_ncells_comb = pd.concat([runtime_ncells_comb, runtime_ncells], axis = 0)

    # runtime_ngenes = pd.read_csv(PROJECT_DIR + f"test/results_scmultisim/runtimes/runtime_ngenes_0.csv", index_col = 0)


sns.set_theme(font_scale = 2)
fig = plt.figure(figsize = (10, 7))
ax = fig.add_subplot()
sns.lineplot(data = runtime_ncells_comb, x = "ngenes", y = "runtime (sec)", markers=True, dashes=False, ax = ax, style = "ncells")
ax.get_legend().remove()
ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))  # force scientific notation
fig.savefig(PROJECT_DIR + "test/results_scmultisim/runtimes/runtime_ngenes.png", dpi = 300, bbox_inches = "tight")


# In[]
# NOTE: Compare the running time across different methods
result_dir = PROJECT_DIR + f"test/results_scmultisim/runtimes/"
runtime_cells_loccsn = pd.read_csv(result_dir + "runtime_ncells_loccsn.csv", sep = ",", index_col = 0)
runtime_cells_loccsn["method"] = "LocCSN"
# infer 10 graphs
# runtime_cells_loccsn["runtime (sec)"] /= 10
runtime_cells_celloracle = pd.read_csv(result_dir + "runtime_ncells_celloracle.csv", sep = ",", index_col = 0)
runtime_cells_celloracle["method"] = "CellOracle"
# infer 10 graphs
# runtime_cells_celloracle["runtime (sec)"] /= 10
runtime_cells_genie3 = pd.read_csv(result_dir + "runtime_ncells_genie3.csv", sep = ",", index_col = 0)
runtime_cells_genie3["method"] = "GENIE3"
runtime_cells_scenicplus = pd.read_csv(result_dir + "runtime_ncells_scenicplus.csv", sep = ",", index_col = 0)
runtime_cells_scenicplus["method"] = "SCENIC+"
runtime_cells_scmultiomeGRN = pd.read_csv(result_dir + "runtime_ncells_scmultiomeGRN.csv", sep = ",", index_col = 0)
runtime_cells_scmultiomeGRN["method"] = "scmultiomeGRN"

runtime_cells_scmtni = pd.read_csv(result_dir + "runtime_ncells_scmtni.csv", sep = ",", index_col = 0)
runtime_cells_scmtni["method"] = "scMTNI"
# infer 10 graphs
# runtime_cells_scmtni["runtime (sec)"] /= 10
runtime_cells_scode = pd.read_csv(result_dir + "runtime_ncells_scode.csv", sep = ",", index_col = 0)
runtime_cells_scode["method"] = "SCODE"
runtime_cells_cesp = pd.read_csv(result_dir + "runtime_ncells_cespgrn_110.csv", sep = ",", index_col = 0)
runtime_cells_cesp["method"] = "CeSpGRN"
# infer 
# runtime_cells_cesp["runtime (sec)"] /=runtime_cells_cesp["ncells"]

runtime_cells_comb = pd.concat([runtime_cells_loccsn, runtime_cells_celloracle,
                                runtime_cells_genie3, runtime_cells_scenicplus,
                                runtime_cells_scmultiomeGRN, runtime_cells_scmtni,
                                runtime_cells_scode, runtime_cells_cesp], axis = 0, ignore_index = True)
runtime_cells_comb = runtime_cells_comb.loc[runtime_cells_comb["ncells"].isin([160, 800, 2000, 4000, 8000, 16000]), :]
runtime_cells_comb["runtime (min)"] = runtime_cells_comb["runtime (sec)"]/60

sns.set_theme(font_scale = 2)
fig = plt.figure(figsize = (7, 5))
ax = fig.add_subplot()
sns.lineplot(data = runtime_cells_comb, x = "ncells", y = "runtime (min)", markers=True, dashes=False, ax = ax, hue = "method", style = "method", linewidth=2, markersize=10)
ax.set_yscale("log")
ax.legend(
    bbox_to_anchor=(1.02, 1),   # position outside
    loc="upper left",
    borderaxespad=0,
    frameon=False               # remove bounding box
)
ax.set_title("Total runtime")

fig.savefig(PROJECT_DIR + "test/results_scmultisim/runtimes/runtime_ncells_compare.png", dpi = 300, bbox_inches = "tight")

# In[]
# NOTE: Compare the running time across different methods for single grn
result_dir = PROJECT_DIR + f"test/results_scmultisim/runtimes/"
runtime_cells_loccsn = pd.read_csv(result_dir + "runtime_ncells_loccsn.csv", sep = ",", index_col = 0)
runtime_cells_loccsn["method"] = "LocCSN"
# infer 10 graphs
runtime_cells_loccsn["runtime (sec)"] /= 10
runtime_cells_celloracle = pd.read_csv(result_dir + "runtime_ncells_celloracle.csv", sep = ",", index_col = 0)
runtime_cells_celloracle["method"] = "CellOracle"
# infer 10 graphs
runtime_cells_celloracle["runtime (sec)"] /= 10
runtime_cells_genie3 = pd.read_csv(result_dir + "runtime_ncells_genie3.csv", sep = ",", index_col = 0)
runtime_cells_genie3["method"] = "GENIE3"
runtime_cells_scenicplus = pd.read_csv(result_dir + "runtime_ncells_scenicplus.csv", sep = ",", index_col = 0)
runtime_cells_scenicplus["method"] = "SCENIC+"
runtime_cells_scmultiomeGRN = pd.read_csv(result_dir + "runtime_ncells_scmultiomeGRN.csv", sep = ",", index_col = 0)
runtime_cells_scmultiomeGRN["method"] = "scmultiomeGRN"

runtime_cells_scmtni = pd.read_csv(result_dir + "runtime_ncells_scmtni.csv", sep = ",", index_col = 0)
runtime_cells_scmtni["method"] = "scMTNI"
# infer 10 graphs
runtime_cells_scmtni["runtime (sec)"] /= 10
runtime_cells_scode = pd.read_csv(result_dir + "runtime_ncells_scode.csv", sep = ",", index_col = 0)
runtime_cells_scode["method"] = "SCODE"
runtime_cells_cesp = pd.read_csv(result_dir + "runtime_ncells.csv", sep = ",", index_col = 0)
runtime_cells_cesp["method"] = "CeSpGRN"
# infer 
runtime_cells_cesp["runtime (sec)"] /=runtime_cells_cesp["ncells"]

runtime_cells_comb = pd.concat([runtime_cells_loccsn, runtime_cells_celloracle,
                                runtime_cells_genie3, runtime_cells_scenicplus,
                                runtime_cells_scmultiomeGRN, runtime_cells_scmtni,
                                runtime_cells_scode, runtime_cells_cesp], axis = 0, ignore_index = True)

runtime_cells_comb["runtime (min)"] = runtime_cells_comb["runtime (sec)"]/60
runtime_cells_comb = runtime_cells_comb.loc[runtime_cells_comb["ncells"].isin([160, 800, 2000, 4000, 8000, 16000]), :]

sns.set_theme(font_scale = 2)
fig = plt.figure(figsize = (7, 5))
ax = fig.add_subplot()
sns.lineplot(data = runtime_cells_comb, x = "ncells", y = "runtime (min)", markers=True, dashes=False, ax = ax, hue = "method", style = "method", linewidth=2, markersize=10)
ax.set_yscale("log")
ax.legend(
    bbox_to_anchor=(1.02, 1),   # position outside
    loc="upper left",
    borderaxespad=0,
    frameon=False               # remove bounding box
)
ax.set_title("Averaged runtime per GRN")

fig.savefig(PROJECT_DIR + "test/results_scmultisim/runtimes/runtime_ncells_compare_singlegrn.png", dpi = 300, bbox_inches = "tight")

# In[]
# NOTE: Compare the running time across different methods
result_dir = PROJECT_DIR + f"test/results_scmultisim/runtimes/"
runtime_genes_loccsn = pd.read_csv(result_dir + "runtime_ngenes_loccsn.csv", sep = ",", index_col = 0)
runtime_genes_loccsn["method"] = "LocCSN"
# infer 10 graphs
# runtime_genes_loccsn["runtime (sec)"] /= 10
runtime_genes_celloracle = pd.read_csv(result_dir + "runtime_ngenes_celloracle.csv", sep = ",", index_col = 0)
runtime_genes_celloracle["method"] = "CellOracle"
# infer 10 graphs
# runtime_genes_celloracle["runtime (sec)"] /= 10
runtime_genes_genie3 = pd.read_csv(result_dir + "runtime_ngenes_genie3.csv", sep = ",", index_col = 0)
runtime_genes_genie3["method"] = "GENIE3"
runtime_genes_scenicplus = pd.read_csv(result_dir + "runtime_ngenes_scenicplus.csv", sep = ",", index_col = 0)
runtime_genes_scenicplus["method"] = "SCENIC+"
runtime_genes_scmultiomeGRN = pd.read_csv(result_dir + "runtime_ngenes_scmultiomeGRN.csv", sep = ",", index_col = 0)
runtime_genes_scmultiomeGRN["method"] = "scmultiomeGRN"

runtime_genes_scmtni = pd.read_csv(result_dir + "runtime_ngenes_scmtni.csv", sep = ",", index_col = 0)
runtime_genes_scmtni["method"] = "scMTNI"
# infer 10 graphs
# runtime_genes_scmtni["runtime (sec)"] /= 10
runtime_genes_scode = pd.read_csv(result_dir + "runtime_ngenes_scode.csv", sep = ",", index_col = 0)
runtime_genes_scode["method"] = "SCODE"
runtime_genes_cesp = pd.read_csv(result_dir + "runtime_ngenes.csv", sep = ",", index_col = 0)
runtime_genes_cesp["method"] = "CeSpGRN"
# infer 
# runtime_genes_cesp["runtime (sec)"] /=runtime_genes_cesp["ncells"]

runtime_genes_comb = pd.concat([runtime_genes_loccsn, runtime_genes_celloracle,
                                runtime_genes_genie3, runtime_genes_scenicplus,
                                runtime_genes_scmultiomeGRN, runtime_genes_scmtni,
                                runtime_genes_scode, runtime_genes_cesp], axis = 0, ignore_index = True)

# runtime_cells_comb = runtime_cells_comb.loc[runtime_cells_comb["ngenes"].isin([27, 55, 110, 550, 770, 1100]), :]
runtime_genes_comb["runtime (min)"] = runtime_genes_comb["runtime (sec)"]/60
runtime_genes_comb = runtime_genes_comb.loc[runtime_genes_comb["ngenes"].isin([27, 55, 110, 550, 770]), :]

sns.set_theme(font_scale = 2)
fig = plt.figure(figsize = (7, 5))
ax = fig.add_subplot()
sns.lineplot(data = runtime_genes_comb, x = "ngenes", y = "runtime (min)", markers=True, dashes=False, ax = ax, hue = "method", style = "method", linewidth=2, markersize=10)
ax.set_yscale("log")
ax.legend(
    bbox_to_anchor=(1.02, 1),   # position outside
    loc="upper left",
    borderaxespad=0,
    frameon=False               # remove bounding box
)
ax.set_title("Total runtime")
fig.savefig(PROJECT_DIR + "test/results_scmultisim/runtimes/runtime_ngenes_compare.png", dpi = 300, bbox_inches = "tight")

# In[]
# NOTE: Compare the running time across different methods
result_dir = PROJECT_DIR + f"test/results_scmultisim/runtimes/"
runtime_genes_loccsn = pd.read_csv(result_dir + "runtime_ngenes_loccsn.csv", sep = ",", index_col = 0)
runtime_genes_loccsn["method"] = "LocCSN"
# infer 10 graphs
runtime_genes_loccsn["runtime (sec)"] /= 10
runtime_genes_celloracle = pd.read_csv(result_dir + "runtime_ngenes_celloracle.csv", sep = ",", index_col = 0)
runtime_genes_celloracle["method"] = "CellOracle"
# infer 10 graphs
runtime_genes_celloracle["runtime (sec)"] /= 10
runtime_genes_genie3 = pd.read_csv(result_dir + "runtime_ngenes_genie3.csv", sep = ",", index_col = 0)
runtime_genes_genie3["method"] = "GENIE3"
runtime_genes_scenicplus = pd.read_csv(result_dir + "runtime_ngenes_scenicplus.csv", sep = ",", index_col = 0)
runtime_genes_scenicplus["method"] = "SCENIC+"
runtime_genes_scmultiomeGRN = pd.read_csv(result_dir + "runtime_ngenes_scmultiomeGRN.csv", sep = ",", index_col = 0)
runtime_genes_scmultiomeGRN["method"] = "scmultiomeGRN"

runtime_genes_scmtni = pd.read_csv(result_dir + "runtime_ngenes_scmtni.csv", sep = ",", index_col = 0)
runtime_genes_scmtni["method"] = "scMTNI"
# infer 10 graphs
runtime_genes_scmtni["runtime (sec)"] /= 10
runtime_genes_scode = pd.read_csv(result_dir + "runtime_ngenes_scode.csv", sep = ",", index_col = 0)
runtime_genes_scode["method"] = "SCODE"
runtime_genes_cesp = pd.read_csv(result_dir + "runtime_ngenes.csv", sep = ",", index_col = 0)
runtime_genes_cesp["method"] = "CeSpGRN"
# infer 
runtime_genes_cesp["runtime (sec)"] /=runtime_genes_cesp["ncells"]

runtime_genes_comb = pd.concat([runtime_genes_loccsn, runtime_genes_celloracle,
                                runtime_genes_genie3, runtime_genes_scenicplus,
                                runtime_genes_scmultiomeGRN, runtime_genes_scmtni,
                                runtime_genes_scode, runtime_genes_cesp], axis = 0, ignore_index = True)

# runtime_cells_comb = runtime_cells_comb.loc[runtime_cells_comb["ngenes"].isin([27, 55, 110, 550, 770, 1100]), :]
runtime_genes_comb["runtime (min)"] = runtime_genes_comb["runtime (sec)"]/60
runtime_genes_comb = runtime_genes_comb.loc[runtime_genes_comb["ngenes"].isin([27, 55, 110, 550, 770]), :]

sns.set_theme(font_scale = 2)
fig = plt.figure(figsize = (7, 5))
ax = fig.add_subplot()
sns.lineplot(data = runtime_genes_comb, x = "ngenes", y = "runtime (min)", markers=True, dashes=False, ax = ax, hue = "method", style = "method", linewidth=2, markersize=10)
ax.set_yscale("log")
ax.legend(
    bbox_to_anchor=(1.02, 1),   # position outside
    loc="upper left",
    borderaxespad=0,
    frameon=False               # remove bounding box
)
ax.set_title("Averaged runtime for single GRN")
fig.savefig(PROJECT_DIR + "test/results_scmultisim/runtimes/runtime_ngenes_compare_singlegrn.png", dpi = 300, bbox_inches = "tight")

# %%
