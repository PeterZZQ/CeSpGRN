# In[0]
import numpy as np 
import pandas as pd 


ntimes = 3000
interval = 50 
ngenes = 20
result_dir = "../results/GGM_" + str(ntimes) + "_" + str(interval) + "_" + str(ngenes) + "/"
bandwidth = 0.1
lamb = 0.01
alpha = 2
rho = 1.7
thetas = np.load(result_dir + "thetas_" + str(bandwidth) + "_" + str(alpha) + "_" + str(lamb) + "_" + str(rho) + ".npy") 
print(thetas.shape)
# %%
