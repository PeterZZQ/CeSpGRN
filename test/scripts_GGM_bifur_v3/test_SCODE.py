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
data_dir = "../../data/GGM_bifurcate/"
# In[1]
# for use_init in ["sergio", "random"]:
#     for (ngenes, ntfs) in [(50, 20), (200, 20)]:
#         for interval in [5, 25, 100]:
#             for seed in range(5):
#                 data = "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "_" + str(seed) + "_" + use_init
#                 counts = pd.read_csv(data_dir + data + "/expr.txt", sep = "\t", header = None)
#                 sim_time = pd.read_csv(data_dir + data + "/sim_time.txt", sep = "\t", header = None, index_col = 0)
#                 dpt_time = pd.read_csv(data_dir + data + "/dpt_time.txt", sep = "\t", header = None, index_col = 0)
#                 interval_size = 100
#                 for i in range(np.int(ntimes/interval_size)):
#                     if i != np.int(ntimes/interval_size) - 1:
#                         counts_sub = counts.iloc[:, i*interval_size:(i+1)*interval_size]
#                         simtime_sub = sim_time.iloc[i*interval_size:(i+1)*interval_size, :]
#                         dpttime_sub = dpt_time.iloc[i*interval_size:(i+1)*interval_size, :]
#                     else:
#                         counts_sub = counts.iloc[:, i*interval_size:]
#                         simtime_sub = sim_time.iloc[i*interval_size:, :]
#                         dpttime_sub = dpt_time.iloc[i*interval_size:, :]
#                     assert counts_sub.shape[1] == interval_size
#                     assert simtime_sub.shape[0] == interval_size
#                     assert dpttime_sub.shape[0] == interval_size
#                     simtime_sub.index = np.arange(simtime_sub.shape[0])
#                     dpttime_sub.index = np.arange(dpttime_sub.shape[0])
#                     counts_sub.to_csv(data_dir + data + "/expr_" + str(i) + ".txt", sep = "\t", index = False, header = False)
#                     simtime_sub.to_csv(data_dir + data + "/simtime_" + str(i) + ".txt", sep = "\t", index = True, header = False)
#                     dpttime_sub.to_csv(data_dir + data + "/dpttime_" + str(i) + ".txt", sep = "\t", index = True, header = False)


# In[] Organize and save input data, dynamic
# ----------------------------------------------------------------------------------
#
#           Organize and save input data
#
# ----------------------------------------------------------------------------------
for interval in [5, 25]:
    for (ngenes, ntfs) in [(50, 20), (200, 20)]:
        for seed in range(3):
            data = "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "_" + str(seed) + "_sergio"
            counts = pd.read_csv(data_dir + data + "/expr.txt", sep = "\t", header = None)
            sim_time = pd.read_csv(data_dir + data + "/sim_time.txt", sep = "\t", header = None, index_col = 0)
            backbone = np.load(data_dir + data + "/backbone.npy").astype(np.object)
            interval_size = 100
            
            count = 0
            for branch in np.sort(np.unique(backbone)):
                idx = np.where(backbone == branch)[0]
                counts_branch = counts.iloc[:, idx]
                simtime_branch = sim_time.iloc[idx, :]
                branchsize = idx.shape[0]
                for i in range(np.int(branchsize/interval_size)):
                    if i != np.int(branchsize/interval_size - 1):
                        counts_sub = counts_branch.iloc[:, i*interval_size:(i+1)*interval_size]
                        simtime_sub = simtime_branch.iloc[i*interval_size:(i+1)*interval_size, :]
                        idx_sub = idx[i*interval_size:(i+1)*interval_size]
                    else:
                        counts_sub = counts_branch.iloc[:, i*interval_size:]
                        simtime_sub = simtime_branch.iloc[i*interval_size:, :]
                        idx_sub = idx[i*interval_size:]

                    simtime_sub.index = np.arange(simtime_sub.shape[0])
                    counts_sub.to_csv(data_dir + data + "/expr_" + str(count) + ".txt", sep = "\t", index = False, header = False)
                    simtime_sub.to_csv(data_dir + data + "/simtime_" + str(count) + ".txt", sep = "\t", index = True, header = False)
                    np.save(data_dir + data + "/idx_" + str(count) + ".npy", idx_sub)             
                    count += 1


# In[1]
# ----------------------------------------------------------------------------------
#
#           Summarize results
#
# ----------------------------------------------------------------------------------

result_dir = "../results_GGM/"
for (ngenes, ntfs) in [(50, 20), (200, 20)]:
    for interval in [5, 25]:
        for seed in range(3):
            subfolder = "bifur_1000_" + str(interval) + "_" + str(ngenes) + "_" + str(seed) + "_sergio"
            try:
                theta = pd.read_csv(result_dir + subfolder + "/SCODE/meanA.txt", sep = "\t", header = None).T.values
            except:
                print("no result: " + subfolder)
                continue
            theta = np.repeat(theta[None, :, :], repeats = 1000, axis = 0)
            assert theta.shape[0] == 1000
            assert theta.shape[1] == ngenes
            assert theta.shape[2] == ngenes
            np.save(file = result_dir + subfolder + "/theta_scode.npy", arr = theta)



for (ngenes, ntfs) in [(50, 20), (200, 20)]:
    for interval in [5, 25]:
        for seed in range(3):
            data_sub = "ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "_" + str(seed) + "_sergio"
            subfolder = "bifur_1000_" + str(interval) + "_" + str(ngenes) + "_" + str(seed) + "_sergio"
            thetas = np.zeros((1000, ngenes, ngenes))
            t = 0
            while True:
                try:
                    counts = pd.read_csv(data_dir + data_sub + "/expr_" + str(t) + ".txt", sep = "\t", header = None)
                    idx = np.load(data_dir + data_sub + "/idx_" + str(t) + ".npy")
                except:
                    print(t) 
                    break

                theta = pd.read_csv(result_dir + subfolder + "/SCODE/" + str(t) + "/meanA.txt", sep = "\t", header = None).T.values
                theta = np.repeat(theta[None, :, :], repeats = counts.shape[1], axis = 0)
                thetas[idx,:,:] = theta
                t += 1

            assert thetas.shape[0] == 1000
            assert thetas.shape[1] == ngenes
            assert thetas.shape[2] == ngenes
            np.save(file = result_dir + subfolder + "/theta_scode_dyn.npy", arr = thetas)

# %%
