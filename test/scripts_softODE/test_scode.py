# In[0]
from numpy.core.fromnumeric import repeat
import pandas as pd 
import numpy as np
from anndata import AnnData
import scanpy as sc
import sys, os
sys.path.append('../../src/')
import genie3


ntimes = 1000
data_dir = "../../data/continuousODE/"
# ----------------------------------------------------------------------------------
#
#           Organize and save input data
#
# ----------------------------------------------------------------------------------

# In[] Organize and save input data, static

for sub_dir in ["linear_ngenes_20_ncell_1000", "bifurc_ngenes_20_ncell_1000"]:
    for seed in range(5):
        counts = np.load(data_dir + sub_dir + "_seed_" + str(seed) + "/true_count.npy")
        counts_df = pd.DataFrame(data = counts.T)
        counts_df.to_csv(data_dir + sub_dir + "_seed_" + str(seed) + "/expr.txt", sep ="\t", index = False, header = False)
        sim_time = np.load(data_dir + sub_dir + "_seed_" + str(seed) + "/pseudotime.npy")
        sim_time = pd.DataFrame(data = sim_time)
        sim_time.to_csv(data_dir + sub_dir + "_seed_" + str(seed) + "/sim_time.txt", sep = "\t", header = False)

        adata = AnnData(X = counts)
        sc.pp.normalize_per_cell(adata)
        sc.pp.log1p(adata)
        sc.pp.neighbors(adata)
        sc.tl.diffmap(adata)
        adata.uns['iroot'] = 0
        sc.tl.dpt(adata)
        dpt = adata.obs.loc[:, ["dpt_pseudotime"]]
        dpt.to_csv(data_dir + sub_dir + "_seed_" + str(seed) + "/dpt_time.txt", sep = "\t", header = False)
        counts_df = pd.DataFrame(data = adata.X.T)
        counts_df.to_csv(data_dir + sub_dir + "_seed_" + str(seed) + "/expr_norm.txt", sep ="\t", index = False, header = False)
        

# In[] Organize and save input data, dynamic
for sub_dir in ["bifurc_ngenes_20_ncell_1000", "linear_ngenes_20_ncell_1000"]:
    for seed in range(5):
        counts = pd.read_csv(data_dir + sub_dir + "_seed_" + str(seed) + "/expr_norm.txt", sep = "\t", header = None)
        sim_time = pd.read_csv(data_dir + sub_dir + "_seed_" + str(seed) + "/sim_time.txt", sep = "\t", header = None, index_col = 0)
        dpt_time = pd.read_csv(data_dir + sub_dir + "_seed_" + str(seed) + "/dpt_time.txt", sep = "\t", header = None, index_col = 0)
        backbone = np.load(data_dir + sub_dir + "_seed_" + str(seed) + "/backbone.npy").astype(np.object)
        interval_size = 100
        
        count = 0
        for branch in np.sort(np.unique(backbone)):
            idx = np.where(backbone == branch)[0]
            counts_branch = counts.iloc[:, idx]
            simtime_branch = sim_time.iloc[idx, :]
            dpttime_branch = dpt_time.iloc[idx, :]
            branchsize = idx.shape[0]
            for i in range(np.int(branchsize/interval_size)):
                if i != np.int(branchsize/interval_size - 1):
                    counts_sub = counts_branch.iloc[:, i*interval_size:(i+1)*interval_size]
                    simtime_sub = simtime_branch.iloc[i*interval_size:(i+1)*interval_size, :]
                    dpttime_sub = dpttime_branch.iloc[i*interval_size:(i+1)*interval_size, :]
                    idx_sub = idx[i*interval_size:(i+1)*interval_size]
                else:
                    counts_sub = counts_branch.iloc[:, i*interval_size:]
                    simtime_sub = simtime_branch.iloc[i*interval_size:, :]
                    dpttime_sub = dpttime_branch.iloc[i*interval_size:, :]
                    idx_sub = idx[i*interval_size:]

                simtime_sub.index = np.arange(simtime_sub.shape[0])
                counts_sub.to_csv(data_dir + sub_dir + "_seed_" + str(seed) + "/expr_norm_" + str(count) + ".txt", sep = "\t", index = False, header = False)
                simtime_sub.to_csv(data_dir + sub_dir + "_seed_" + str(seed) + "/simtime_" + str(count) + ".txt", sep = "\t", index = True, header = False)
                dpttime_sub.to_csv(data_dir + sub_dir + "_seed_" + str(seed) + "/dpttime_" + str(count) + ".txt", sep = "\t", index = True, header = False)   
                np.save(data_dir + sub_dir + "_seed_" + str(seed) + "/idx_" + str(count) + ".npy", idx_sub)             
                count += 1

for sub_dir in ["bifurc_ngenes_20_ncell_1000", "linear_ngenes_20_ncell_1000"]:
    for seed in range(5):
        counts = pd.read_csv(data_dir + sub_dir + "_seed_" + str(seed) + "/expr.txt", sep = "\t", header = None)
        sim_time = pd.read_csv(data_dir + sub_dir + "_seed_" + str(seed) + "/sim_time.txt", sep = "\t", header = None, index_col = 0)
        dpt_time = pd.read_csv(data_dir + sub_dir + "_seed_" + str(seed) + "/dpt_time.txt", sep = "\t", header = None, index_col = 0)
        backbone = np.load(data_dir + sub_dir + "_seed_" + str(seed) + "/backbone.npy").astype(np.object)
        interval_size = 100
        
        count = 0
        for branch in np.sort(np.unique(backbone)):
            idx = np.where(backbone == branch)[0]
            counts_branch = counts.iloc[:, idx]
            simtime_branch = sim_time.iloc[idx, :]
            dpttime_branch = dpt_time.iloc[idx, :]
            branchsize = idx.shape[0]
            for i in range(np.int(branchsize/interval_size)):
                if i != np.int(branchsize/interval_size) - 1:
                    counts_sub = counts_branch.iloc[:, i*interval_size:(i+1)*interval_size]
                    simtime_sub = simtime_branch.iloc[i*interval_size:(i+1)*interval_size, :]
                    dpttime_sub = dpttime_branch.iloc[i*interval_size:(i+1)*interval_size, :]
                    idx_sub = idx[i*interval_size:(i+1)*interval_size]
                else:
                    counts_sub = counts_branch.iloc[:, i*interval_size:]
                    simtime_sub = simtime_branch.iloc[i*interval_size:, :]
                    dpttime_sub = dpttime_branch.iloc[i*interval_size:, :]
                    idx_sub = idx[i*interval_size:]

                simtime_sub.index = np.arange(simtime_sub.shape[0])
                counts_sub.to_csv(data_dir + sub_dir + "_seed_" + str(seed) + "/expr_" + str(count) + ".txt", sep = "\t", index = False, header = False)
                simtime_sub.to_csv(data_dir + sub_dir + "_seed_" + str(seed) + "/simtime_" + str(count) + ".txt", sep = "\t", index = True, header = False)
                dpttime_sub.to_csv(data_dir + sub_dir + "_seed_" + str(seed) + "/dpttime_" + str(count) + ".txt", sep = "\t", index = True, header = False)                
                # np.save(data_dir + sub_dir + "_seed_" + str(seed) + "/idx_" + str(count) + ".npy", idx_sub)             
                count += 1



# ----------------------------------------------------------------------------------
#
#           Summarize results
#
# ----------------------------------------------------------------------------------

# In[] Summarize results, static
ngenes = 20
result_dir = "../results_softODE_v2/"
print(result_dir)
for subfolder in ["linear_ngenes_20_ncell_1000", "bifurc_ngenes_20_ncell_1000"]:
    for seed in range(5):
        theta = pd.read_csv(result_dir + subfolder  + "_seed_" + str(seed) + "/SCODE/meanA.txt", sep = "\t", header = None).T.values
        theta = np.repeat(theta[None, :, :], repeats = 1000, axis = 0)
        assert theta.shape[0] == 1000
        assert theta.shape[1] == ngenes
        assert theta.shape[2] == ngenes
        np.save(file = result_dir + subfolder + "_seed_" + str(seed) + "/theta_scode.npy", arr = theta)




ngenes = 20
result_dir = "../results_softODE_v2/"
print(result_dir)
for subfolder in ["linear_ngenes_20_ncell_1000", "bifurc_ngenes_20_ncell_1000"]:
    for seed in range(5):
        theta = pd.read_csv(result_dir + subfolder + "_seed_" + str(seed) + "/SCODE_norm/meanA.txt", sep = "\t", header = None).T.values
        theta = np.repeat(theta[None, :, :], repeats = 1000, axis = 0)
        assert theta.shape[0] == 1000
        assert theta.shape[1] == ngenes
        assert theta.shape[2] == ngenes
        np.save(file = result_dir + subfolder + "_seed_" + str(seed) + "/theta_scode_norm.npy", arr = theta)



# In[] Summarize results, dynamic for bifurcate
ngenes = 20
data_dir = "../../data/continuousODE/"
result_dir = "../results_softODE_v2/"
print(result_dir)
for subfolder in ["bifurc_ngenes_20_ncell_1000"]:
    for seed in range(5):
        thetas = np.zeros((1000, 20, 20))
        t = 0
        while True:
            try:
                counts = pd.read_csv(data_dir + subfolder + "_seed_" + str(seed) + "/expr_" + str(t) + ".txt", sep = "\t", header = None)
                idx = np.load(data_dir + subfolder + "_seed_" + str(seed) + "/idx_" + str(t) + ".npy")
            except: 
                break

            theta = pd.read_csv(result_dir + subfolder  + "_seed_" + str(seed) +  "/SCODE/" + str(t) + "/meanA.txt", sep = "\t", header = None).T.values
            theta = np.repeat(theta[None, :, :], repeats = counts.shape[1], axis = 0)
            thetas[idx,:,:] = theta
            t += 1

        assert thetas.shape[0] == 1000
        assert thetas.shape[1] == ngenes
        assert thetas.shape[2] == ngenes
        np.save(file = result_dir + subfolder + "_seed_" + str(seed) + "/theta_scode_dyn.npy", arr = thetas)


ngenes = 20
data_dir = "../../data/continuousODE/"
result_dir = "../results_softODE_v2/"
print(result_dir)
for subfolder in ["bifurc_ngenes_20_ncell_1000"]:
    for seed in range(5):
        thetas = np.zeros((1000, 20, 20))
        t = 0
        while True:
            try:
                counts = pd.read_csv(data_dir + subfolder + "_seed_" + str(seed) + "/expr_norm_" + str(t) + ".txt", sep = "\t", header = None)
                idx = np.load(data_dir + subfolder + "_seed_" + str(seed) + "/idx_" + str(t) + ".npy")
            except: 
                break

            theta = pd.read_csv(result_dir + subfolder  + "_seed_" + str(seed) +  "/SCODE_norm/" + str(t) + "/meanA.txt", sep = "\t", header = None).T.values
            theta = np.repeat(theta[None, :, :], repeats = counts.shape[1], axis = 0)
            thetas[idx,:,:] = theta
            t += 1

        assert thetas.shape[0] == 1000
        assert thetas.shape[1] == ngenes
        assert thetas.shape[2] == ngenes
        np.save(file = result_dir + subfolder + "_seed_" + str(seed) + "/theta_scode_dyn_norm.npy", arr = thetas)

# In[] Summarize results, dynamic for linear

result_dir = "../results_softODE_v2/"
print(result_dir)
for sub_dir in ["linear_ngenes_20_ncell_1000"]:
    for seed in range(5):
        thetas = []
        for t in range(0, 10):
            try:
                theta = pd.read_csv(result_dir + sub_dir  + "_seed_" + str(seed) +  "/SCODE/" + str(t) + "/meanA.txt", sep = "\t", header = None).T.values
            except:
                break
            theta = np.repeat(theta[None, :, :], repeats = 100, axis = 0)
            thetas.append(theta)
            assert theta.shape[0] == 100
            assert theta.shape[1] == ngenes
            assert theta.shape[2] == ngenes
        if len(thetas) == 10:
            thetas = np.concatenate(thetas, axis = 0)
            assert thetas.shape[0] == 1000
            assert thetas.shape[1] == ngenes
            assert thetas.shape[2] == ngenes
            np.save(file = result_dir + sub_dir + "_seed_" + str(seed) + "/theta_scode_dyn.npy", arr = thetas)
        else:
            print(len(thetas))
            print("no result")

result_dir = "../results_softODE_v2/"
print(result_dir)
for sub_dir in ["linear_ngenes_20_ncell_1000"]:
    for seed in range(5):
        thetas = []
        for t in range(0, 10):
            try:
                theta = pd.read_csv(result_dir + sub_dir + "_seed_" + str(seed) + "/SCODE_norm/" + str(t) + "/meanA.txt", sep = "\t", header = None).T.values
            except:
                break
            theta = np.repeat(theta[None, :, :], repeats = 100, axis = 0)
            thetas.append(theta)
            assert theta.shape[0] == 100
            assert theta.shape[1] == ngenes
            assert theta.shape[2] == ngenes
        if len(thetas) == 10:
            thetas = np.concatenate(thetas, axis = 0)
            assert thetas.shape[0] == 1000
            assert thetas.shape[1] == ngenes
            assert thetas.shape[2] == ngenes
            np.save(file = result_dir + sub_dir + "_seed_" + str(seed) + "/theta_scode_dyn_norm.npy", arr = thetas)
        else:
            print(len(thetas))
            print("no result")

#In[]
# ----------------------------------------------------------------------------------
#
#           Test genie
#
# ----------------------------------------------------------------------------------
ngenes = 20
result_dir = "../results_softODE_v2/"

for sub_dir in ["linear_ngenes_20_ncell_1000", "bifurc_ngenes_20_ncell_1000"]:
    for seed in range(5):
        counts = pd.read_csv(data_dir + sub_dir + "_seed_" + str(seed) + "/expr.txt", sep = "\t", header = None).values.T
        genie_theta = genie3.GENIE3(counts, gene_names=["gene_" + str(x) for x in range(ngenes)], regulators='all',tree_method='RF',K='sqrt',ntrees=1000,nthreads=1)
        genie_theta = np.repeat(genie_theta[None, :, :], counts.shape[0], axis = 0)
        np.save(file = result_dir + sub_dir + "_seed_" + str(seed) + "/theta_genie.npy", arr = genie_theta)
        

ngenes = 20
result_dir = "../results_softODE_v2/"
for sub_dir in ["linear_ngenes_20_ncell_1000", "bifurc_ngenes_20_ncell_1000"]:
    for seed in range(5):
        counts = pd.read_csv(data_dir + sub_dir + "_seed_" + str(seed) + "/expr_norm.txt", sep = "\t", header = None).values.T
        genie_theta = genie3.GENIE3(counts, gene_names=["gene_" + str(x) for x in range(ngenes)], regulators='all',tree_method='RF',K='sqrt',ntrees=1000,nthreads=1)
        genie_theta = np.repeat(genie_theta[None, :, :], counts.shape[0], axis = 0)
        np.save(file = result_dir + sub_dir + "_seed_" + str(seed) + "/theta_genie_norm.npy", arr = genie_theta)

# In[]
ngenes = 20
result_dir = "../results_softODE_v2/"

for sub_dir in ["linear_ngenes_20_ncell_1000", "bifurc_ngenes_20_ncell_1000"]:
    for seed in range(5):
        counts = pd.read_csv(data_dir + sub_dir + "_seed_" + str(seed) + "/expr.txt", sep = "\t", header = None).values.T
        genie_theta = genie3.GENIE3(counts, gene_names=["gene_" + str(x) for x in range(ngenes)], regulators=["gene_" + str(x) for x in range(5)],tree_method='RF',K='sqrt',ntrees=1000,nthreads=1)
        genie_theta = np.repeat(genie_theta[None, :, :], counts.shape[0], axis = 0)
        np.save(file = result_dir + sub_dir + "_seed_" + str(seed) + "/theta_genie_tf.npy", arr = genie_theta)
        

ngenes = 20
result_dir = "../results_softODE_v2/"
for sub_dir in ["linear_ngenes_20_ncell_1000", "bifurc_ngenes_20_ncell_1000"]:
    for seed in range(5):
        counts = pd.read_csv(data_dir + sub_dir + "_seed_" + str(seed) + "/expr_norm.txt", sep = "\t", header = None).values.T
        genie_theta = genie3.GENIE3(counts, gene_names=["gene_" + str(x) for x in range(ngenes)], regulators=["gene_" + str(x) for x in range(5)],tree_method='RF',K='sqrt',ntrees=1000,nthreads=1)
        genie_theta = np.repeat(genie_theta[None, :, :], counts.shape[0], axis = 0)
        np.save(file = result_dir + sub_dir + "_seed_" + str(seed) + "/theta_genie_tf_norm.npy", arr = genie_theta)
 
# In[]
# for sub_dir in ["bifurc_ngenes_20_ncell_1000"]:
for sub_dir in ["linear_ngenes_20_ncell_1000"]:
    for seed in range(5):
        thetas = np.zeros((1000, ngenes, ngenes))
        count = 0
        while True:
            try:
                counts = pd.read_csv(data_dir + sub_dir + "_seed_" + str(seed) + "/expr_" + str(count) + ".txt", sep = "\t", header = None)
                idx = np.load(data_dir + sub_dir + "_seed_" + str(seed) + "/idx_" + str(count) + ".npy")
            except:
                break

            genie_theta = genie3.GENIE3(counts.values.T, gene_names=["gene_" + str(x) for x in range(ngenes)], regulators='all',tree_method='RF',K='sqrt',ntrees=1000,nthreads=1)
            thetas[idx,:,:] = genie_theta
            count += 1

        np.save(file = result_dir + sub_dir + "_seed_" + str(seed) + "/theta_genie_dyn.npy", arr = thetas)
        

ngenes = 20
result_dir = "../results_softODE_v2/"
# for sub_dir in ["bifurc_ngenes_20_ncell_1000"]:
for sub_dir in ["linear_ngenes_20_ncell_1000"]:
    for seed in range(5):
        thetas = np.zeros((1000, ngenes, ngenes))
        count = 0
        while True:
            try:
                counts = pd.read_csv(data_dir + sub_dir + "_seed_" + str(seed) + "/expr_norm_" + str(count) + ".txt", sep = "\t", header = None)
                idx = np.load(data_dir + sub_dir + "_seed_" + str(seed) + "/idx_" + str(count) + ".npy")
            except:
                break

            genie_theta = genie3.GENIE3(counts.values.T, gene_names=["gene_" + str(x) for x in range(ngenes)], regulators='all',tree_method='RF',K='sqrt',ntrees=1000,nthreads=1)
            thetas[idx,:,:] = genie_theta
            count += 1
        
        np.save(file = result_dir + sub_dir + "_seed_" + str(seed) + "/theta_genie_dyn_norm.npy", arr = thetas)

# In[]
for sub_dir in ["linear_ngenes_20_ncell_1000","bifurc_ngenes_20_ncell_1000"]:
    for seed in range(5):
        thetas = np.zeros((1000, ngenes, ngenes))
        count = 0
        while True:
            try:
                counts = pd.read_csv(data_dir + sub_dir + "_seed_" + str(seed) + "/expr_" + str(count) + ".txt", sep = "\t", header = None)
                idx = np.load(data_dir + sub_dir + "_seed_" + str(seed) + "/idx_" + str(count) + ".npy")
            except:
                break

            genie_theta = genie3.GENIE3(counts.values.T, gene_names=["gene_" + str(x) for x in range(ngenes)], regulators=["gene_" + str(x) for x in range(5)],tree_method='RF',K='sqrt',ntrees=1000,nthreads=1)
            thetas[idx,:,:] = genie_theta
            count += 1

        np.save(file = result_dir + sub_dir + "_seed_" + str(seed) + "/theta_genie_dyn_tf.npy", arr = thetas)
        

ngenes = 20
result_dir = "../results_softODE_v2/"
for sub_dir in ["linear_ngenes_20_ncell_1000","bifurc_ngenes_20_ncell_1000"]:
    for seed in range(5):
        thetas = np.zeros((1000, ngenes, ngenes))
        count = 0
        while True:
            try:
                counts = pd.read_csv(data_dir + sub_dir + "_seed_" + str(seed) + "/expr_norm_" + str(count) + ".txt", sep = "\t", header = None)
                idx = np.load(data_dir + sub_dir + "_seed_" + str(seed) + "/idx_" + str(count) + ".npy")
            except:
                break

            genie_theta = genie3.GENIE3(counts.values.T, gene_names=["gene_" + str(x) for x in range(ngenes)], regulators=["gene_" + str(x) for x in range(5)],tree_method='RF',K='sqrt',ntrees=1000,nthreads=1)
            thetas[idx,:,:] = genie_theta
            count += 1
        
        np.save(file = result_dir + sub_dir + "_seed_" + str(seed) + "/theta_genie_dyn_tf_norm.npy", arr = thetas)

# %%
