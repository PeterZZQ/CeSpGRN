# In[0]
from numpy.core.fromnumeric import repeat
import pandas as pd 
import numpy as np

# In[1]
# ntimes = 1000
# data_dir = "../data/GGM_bifurcate/"

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


# In[1]

result_dir = "results_GGM/"
print(result_dir)
for use_init in ["sergio", "random"]:
    for pt in ["truet", "dpt"]:
        for (ngenes, ntfs) in [(50, 20), (200, 20)]:
            for interval in [5, 25, 100]:
                for seed in range(5):
                    subfolder = "bifur_1000_" + str(interval) + "_" + str(ngenes) + "_" + str(seed) + "_" + use_init
                    try:
                        theta = pd.read_csv(result_dir + subfolder + "/SCODE_" + pt + "/meanA.txt", sep = "\t", header = None).T.values
                    except:
                        print("no result: " + subfolder)
                        continue
                    theta = np.repeat(theta[None, :, :], repeats = 1000, axis = 0)
                    assert theta.shape[0] == 1000
                    assert theta.shape[1] == ngenes
                    assert theta.shape[2] == ngenes
                    np.save(file = result_dir + subfolder + "/theta_scode_" + pt + ".npy", arr = theta)


for use_init in ["sergio", "random"]:
    for pt in ["truet", "dpt"]:
        for (ngenes, ntfs) in [(50, 20), (200, 20)]:
            for interval in [5, 25, 100]:
                for seed in range(5):
                    subfolder = "bifur_1000_" + str(interval) + "_" + str(ngenes) + "_" + str(seed) + "_" + use_init
                    thetas = []
                    for t in range(0, 10):
                        try:
                            theta = pd.read_csv(result_dir + subfolder + "/SCODE_" + pt + "/" + str(t) + "/meanA.txt", sep = "\t", header = None).T.values
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
                        np.save(file = result_dir + subfolder + "/theta_scode_dyn_" + pt + ".npy", arr = thetas)
                    else:
                        print(len(thetas))
                        print("no result: ntimes_1000_interval_" + str(interval) + "_ngenes_" + str(ngenes))


# %%
