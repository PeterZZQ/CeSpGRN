# In[0]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import sys, os
sys.path.append('../../src/')
import bmk_beeline as bmk
import genie3

plt.rcParams["font.size"] = 16


def preprocess(counts): 
    """\
    Input:
    counts = (ntimes, ngenes)
    
    Description:
    ------------
    Preprocess the dataset
    """
    # normalize according to the library size
    
    libsize = np.median(np.sum(counts, axis = 1))
    counts = counts / np.sum(counts, axis = 1)[:,None] * libsize
        
    counts = np.log1p(counts)
    return counts

# In[1] run model
ntimes = 1000
path = "../../data/continuousODE/sergio_dense/"

# for ngenes, ntfs in [(20, 5), (30, 10), (50, 20), (100, 50)]:
for ngenes, ntfs in [(20, 5), (30, 5), (50, 5)]:
    for interval in [50, 100, 200]:
        for stepsize in [0.0001, 0.0002]:
            result_dir = "../results_softODE_sergio/results_ngenes_" + str(ngenes) + "_interval_" + str(interval) + "_stepsize_" + str(stepsize) + "/"
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            # ---------------------------------------------------- #
            # first round, using standard scaler as initialization
            # ---------------------------------------------------- #
            X = np.load(path + "ngenes_" + str(ngenes) + "_interval_" + str(interval) + "_stepsize_" + str(stepsize) + "/true_count.npy")
            gt_adj = np.load(path + "ngenes_" + str(ngenes) + "_interval_" + str(interval) + "_stepsize_" + str(stepsize) + "/GRNs.npy")

            # sort the genes
            print("Raw TimePoints: {}, no.Genes: {}".format(X.shape[0],X.shape[1]))
            X = StandardScaler().fit_transform(X)

            # make sure the dimensions are correct
            assert X.shape[0] == ntimes
            assert X.shape[1] == ngenes
            assert gt_adj.shape[0] == ntimes
            assert gt_adj.shape[1] == ngenes
            assert gt_adj.shape[2] == ngenes

            # genie_theta of the shape (ntimes, ngenes, ngenes)
            genie_theta = genie3.GENIE3(X, gene_names=["gene_" + str(i) for i in range(ngenes)], regulators='all',tree_method='RF',K='sqrt',ntrees=1000,nthreads=1)
            genie_theta = np.repeat(genie_theta[None, :, :], ntimes, axis=0)

            # include TF information
            genie_theta_tf = genie3.GENIE3(X, gene_names=["gene_" + str(i) for i in range(ngenes)], regulators=["gene_" + str(i) for i in range(ntfs)], tree_method='RF',K='sqrt',ntrees=1000,nthreads=1)
            genie_theta_tf = np.repeat(genie_theta_tf[None, :, :], ntimes, axis=0)            

            np.save(file = result_dir + "genie_theta_standardscaler.npy", arr = genie_theta)
            np.save(file = result_dir + "genie_theta_standardscaler_tf.npy", arr = genie_theta_tf)

            # ---------------------------------------------------- #
            # second round, not using standard scaler as initialization
            # ---------------------------------------------------- #

            X = np.load(path + "ngenes_" + str(ngenes) + "_interval_" + str(interval) + "_stepsize_" + str(stepsize) + "/true_count.npy")
            gt_adj = np.load(path + "ngenes_" + str(ngenes) + "_interval_" + str(interval) + "_stepsize_" + str(stepsize) + "/GRNs.npy")

            # sort the genes
            print("Raw TimePoints: {}, no.Genes: {}".format(X.shape[0],X.shape[1]))

            # make sure the dimensions are correct
            assert X.shape[0] == ntimes
            assert X.shape[1] == ngenes
            assert gt_adj.shape[0] == ntimes
            assert gt_adj.shape[1] == ngenes
            assert gt_adj.shape[2] == ngenes

            # genie_theta of the shape (ntimes, ngenes, ngenes)
            genie_theta = genie3.GENIE3(X, gene_names=["gene_" + str(i) for i in range(ngenes)], regulators='all',tree_method='RF',K='sqrt',ntrees=1000,nthreads=1)
            genie_theta = np.repeat(genie_theta[None, :, :], ntimes, axis=0)

            # include TF information
            genie_theta_tf = genie3.GENIE3(X, gene_names=["gene_" + str(i) for i in range(ngenes)], regulators=["gene_" + str(i) for i in range(ntfs)], tree_method='RF',K='sqrt',ntrees=1000,nthreads=1)
            genie_theta_tf = np.repeat(genie_theta_tf[None, :, :], ntimes, axis=0)            

            np.save(file = result_dir + "genie_theta.npy", arr = genie_theta)
            np.save(file = result_dir + "genie_theta_tf.npy", arr = genie_theta_tf)

            # ---------------------------------------------------- #
            # second round, not using standard scaler as initialization
            # ---------------------------------------------------- #

            X = np.load(path + "ngenes_" + str(ngenes) + "_interval_" + str(interval) + "_stepsize_" + str(stepsize) + "/true_count.npy")
            gt_adj = np.load(path + "ngenes_" + str(ngenes) + "_interval_" + str(interval) + "_stepsize_" + str(stepsize) + "/GRNs.npy")

            # sort the genes
            print("Raw TimePoints: {}, no.Genes: {}".format(X.shape[0],X.shape[1]))

            # make sure the dimensions are correct
            assert X.shape[0] == ntimes
            assert X.shape[1] == ngenes
            assert gt_adj.shape[0] == ntimes
            assert gt_adj.shape[1] == ngenes
            assert gt_adj.shape[2] == ngenes

            # genie_theta of the shape (ntimes, ngenes, ngenes)
            genie_theta = genie3.GENIE3(X, gene_names=["gene_" + str(i) for i in range(ngenes)], regulators='all',tree_method='RF',K='sqrt',ntrees=1000,nthreads=1)
            genie_theta = np.repeat(genie_theta[None, :, :], ntimes, axis=0)

            # include TF information
            genie_theta_tf = genie3.GENIE3(X, gene_names=["gene_" + str(i) for i in range(ngenes)], regulators=["gene_" + str(i) for i in range(ntfs)], tree_method='RF',K='sqrt',ntrees=1000,nthreads=1)
            genie_theta_tf = np.repeat(genie_theta_tf[None, :, :], ntimes, axis=0)            

            np.save(file = result_dir + "genie_theta.npy", arr = genie_theta)
            np.save(file = result_dir + "genie_theta_tf.npy", arr = genie_theta_tf)

            # ---------------------------------------------------- #
            # second round, not using standard scaler as initialization
            # ---------------------------------------------------- #            

            X = np.load(path + "ngenes_" + str(ngenes) + "_interval_" + str(interval) + "_stepsize_" + str(stepsize) + "/obs_count.npy")
            gt_adj = np.load(path + "ngenes_" + str(ngenes) + "_interval_" + str(interval) + "_stepsize_" + str(stepsize) + "/GRNs.npy")
            X = preprocess(X)

            # sort the genes
            print("Raw TimePoints: {}, no.Genes: {}".format(X.shape[0],X.shape[1]))

            # make sure the dimensions are correct
            assert X.shape[0] == ntimes
            assert X.shape[1] == ngenes
            assert gt_adj.shape[0] == ntimes
            assert gt_adj.shape[1] == ngenes
            assert gt_adj.shape[2] == ngenes

            # genie_theta of the shape (ntimes, ngenes, ngenes)
            genie_theta = genie3.GENIE3(X, gene_names=["gene_" + str(i) for i in range(ngenes)], regulators='all',tree_method='RF',K='sqrt',ntrees=1000,nthreads=1)
            genie_theta = np.repeat(genie_theta[None, :, :], ntimes, axis=0)

            # include TF information
            genie_theta_tf = genie3.GENIE3(X, gene_names=["gene_" + str(i) for i in range(ngenes)], regulators=["gene_" + str(i) for i in range(ntfs)], tree_method='RF',K='sqrt',ntrees=1000,nthreads=1)
            genie_theta_tf = np.repeat(genie_theta_tf[None, :, :], ntimes, axis=0)            

            np.save(file = result_dir + "genie_theta_obs.npy", arr = genie_theta)
            np.save(file = result_dir + "genie_theta_obs_tf.npy", arr = genie_theta_tf)


# In[2] benchmark
print("test results:")
ntimes = 1000
path = "../../data/continuousODE/sergio_dense/"
# for ngenes, ntfs in [(20, 5), (30, 10), (50, 20), (100, 50)]:
for ngenes, ntfs in [(20, 5), (30, 5), (50, 5)]:
    for interval in [50, 100, 200]:
        for stepsize in [0.0001, 0.0002]:
            # read in the data
            X = np.load(path + "ngenes_" + str(ngenes) + "_interval_" + str(interval) + "_stepsize_" + str(stepsize) + "/true_count.npy")
            gt_adj = np.load(path + "ngenes_" + str(ngenes) + "_interval_" + str(interval) + "_stepsize_" + str(stepsize) + "/GRNs.npy")

            score = pd.DataFrame(columns =["model", "nmse","probability of success", "pearson", "cosine similarity", "time"])
            score_rand = pd.DataFrame(columns =["model", "nmse","probability of success", "pearson", "cosine similarity", "time"])
            result_dir = "../results_softODE_sergio/results_ngenes_" + str(ngenes) + "_interval_" + str(interval) + "_stepsize_" + str(stepsize) + "/"
            thetas1 = np.load(file = result_dir + "genie_theta_standardscaler.npy")
            thetas2 = np.load(file = result_dir + "genie_theta_standardscaler_tf.npy")
            thetas3 = np.load(file = result_dir + "genie_theta.npy")
            thetas4 = np.load(file = result_dir + "genie_theta_tf.npy")   
            thetas5 = np.load(file = result_dir + "genie_theta_obs.npy")
            thetas6 = np.load(file = result_dir + "genie_theta_obs_tf.npy")            
            
            print("ngenes_" + str(ngenes) + "_interval_" + str(interval) + "_stepsize_" + str(stepsize))

            for time in range(0, ntimes):
                np.random.seed(0)
                # benchmark random baseline
                thetas_rand = np.random.randn(ngenes,ngenes)
                # TODO: include testing method, check glad paper
                nmse_rand = bmk.NMSE(G_inf = thetas_rand, G_true = gt_adj[time])
                ps_rand = bmk.PS(G_inf = thetas_rand, G_true = gt_adj[time])

                pearson_rand_val, pval = bmk.pearson(G_inf = thetas_rand, G_true = gt_adj[time])
                cosine_sim_rand = bmk.cossim(G_inf = thetas_rand, G_true = gt_adj[time])        
                score = score.append({"nmse": nmse_rand, "probability of success":ps_rand, "pearson": pearson_rand_val, "cosine similarity": cosine_sim_rand, 
                                      "model":"random", "time":time}, ignore_index=True)

            print()
            print("model: random")
            print("mean nmse: " + str(np.mean(score.loc[score["model"] == "random", "nmse"].values)))
            print("mean pearson: " + str(np.mean(score.loc[score["model"] == "random", "pearson"].values)))
            print("mean cosine similarity: " + str(np.mean(score.loc[score["model"] == "random", "cosine similarity"].values)))
            print()

            for time in range(0, ntimes):
                # benchmark genie
                nmse = bmk.NMSE(G_inf = thetas1[time], G_true = gt_adj[time])
                ps = bmk.PS(G_inf = thetas1[time], G_true = gt_adj[time])
                pearson_val, pval = bmk.pearson(G_inf = thetas1[time], G_true = gt_adj[time])
                cosine_sim = bmk.cossim(G_inf = thetas1[time], G_true = gt_adj[time])            
                score = score.append({"nmse": nmse, "probability of success":ps, "pearson": pearson_val, "cosine similarity": cosine_sim, 
                                    "model":"genie_standardscaler", "time":time}, ignore_index=True)

            print()
            print("model: genie_standardscaler")
            print("mean nmse: " + str(np.mean(score.loc[score["model"] == "genie_standardscaler", "nmse"].values)))
            print("mean pearson: " + str(np.mean(score.loc[score["model"] == "genie_standardscaler", "pearson"].values)))
            print("mean cosine similarity: " + str(np.mean(score.loc[score["model"] == "genie_standardscaler", "cosine similarity"].values)))
            print()

            for time in range(0, ntimes):
                nmse = bmk.NMSE(G_inf = thetas2[time], G_true = gt_adj[time])
                ps = bmk.PS(G_inf = thetas2[time], G_true = gt_adj[time])
                pearson_val, pval = bmk.pearson(G_inf = thetas2[time], G_true = gt_adj[time])
                cosine_sim = bmk.cossim(G_inf = thetas2[time], G_true = gt_adj[time])            
                score = score.append({"nmse": nmse, "probability of success":ps, "pearson": pearson_val, "cosine similarity": cosine_sim, 
                                    "model":"genie_standardscaler_tf", "time":time}, ignore_index=True)

            print()
            print("model: genie_standardscaler_tf")
            print("mean nmse: " + str(np.mean(score.loc[score["model"] == "genie_standardscaler_tf", "nmse"].values)))
            print("mean pearson: " + str(np.mean(score.loc[score["model"] == "genie_standardscaler_tf", "pearson"].values)))
            print("mean cosine similarity: " + str(np.mean(score.loc[score["model"] == "genie_standardscaler_tf", "cosine similarity"].values)))
            print()

            for time in range(0, ntimes):
                nmse = bmk.NMSE(G_inf = thetas3[time], G_true = gt_adj[time])
                ps = bmk.PS(G_inf = thetas3[time], G_true = gt_adj[time])
                pearson_val, pval = bmk.pearson(G_inf = thetas3[time], G_true = gt_adj[time])
                cosine_sim = bmk.cossim(G_inf = thetas3[time], G_true = gt_adj[time])            
                score = score.append({"nmse": nmse, "probability of success":ps, "pearson": pearson_val, "cosine similarity": cosine_sim, 
                                    "model":"genie", "time":time}, ignore_index=True)

            print()
            print("model: genie")
            print("mean nmse: " + str(np.mean(score.loc[score["model"] == "genie", "nmse"].values)))
            print("mean pearson: " + str(np.mean(score.loc[score["model"] == "genie", "pearson"].values)))
            print("mean cosine similarity: " + str(np.mean(score.loc[score["model"] == "genie", "cosine similarity"].values)))
            print()

            for time in range(0, ntimes):
                nmse = bmk.NMSE(G_inf = thetas4[time], G_true = gt_adj[time])
                ps = bmk.PS(G_inf = thetas4[time], G_true = gt_adj[time])
                pearson_val, pval = bmk.pearson(G_inf = thetas4[time], G_true = gt_adj[time])
                cosine_sim = bmk.cossim(G_inf = thetas4[time], G_true = gt_adj[time])            
                score = score.append({"nmse": nmse, "probability of success":ps, "pearson": pearson_val, "cosine similarity": cosine_sim, 
                                    "model":"genie_tf", "time":time}, ignore_index=True)

            print()
            print("model: genie_tf")
            print("mean nmse: " + str(np.mean(score.loc[score["model"] == "genie_tf", "nmse"].values)))
            print("mean pearson: " + str(np.mean(score.loc[score["model"] == "genie_tf", "pearson"].values)))
            print("mean cosine similarity: " + str(np.mean(score.loc[score["model"] == "genie_tf", "cosine similarity"].values)))
            print()

            for time in range(0, ntimes):
                nmse = bmk.NMSE(G_inf = thetas5[time], G_true = gt_adj[time])
                ps = bmk.PS(G_inf = thetas5[time], G_true = gt_adj[time])
                pearson_val, pval = bmk.pearson(G_inf = thetas5[time], G_true = gt_adj[time])
                cosine_sim = bmk.cossim(G_inf = thetas5[time], G_true = gt_adj[time])            
                score = score.append({"nmse": nmse, "probability of success":ps, "pearson": pearson_val, "cosine similarity": cosine_sim, 
                                    "model":"genie_obs", "time":time}, ignore_index=True)

            print()
            print("model: genie_obs")
            print("mean nmse: " + str(np.mean(score.loc[score["model"] == "genie_obs", "nmse"].values)))
            print("mean pearson: " + str(np.mean(score.loc[score["model"] == "genie_obs", "pearson"].values)))
            print("mean cosine similarity: " + str(np.mean(score.loc[score["model"] == "genie_obs", "cosine similarity"].values)))
            print()

            for time in range(0, ntimes):
                nmse = bmk.NMSE(G_inf = thetas6[time], G_true = gt_adj[time])
                ps = bmk.PS(G_inf = thetas6[time], G_true = gt_adj[time])
                pearson_val, pval = bmk.pearson(G_inf = thetas6[time], G_true = gt_adj[time])
                cosine_sim = bmk.cossim(G_inf = thetas6[time], G_true = gt_adj[time])            
                score = score.append({"nmse": nmse, "probability of success":ps, "pearson": pearson_val, "cosine similarity": cosine_sim, 
                                    "model":"genie_obs_tf", "time":time}, ignore_index=True)

            print()
            print("model: genie_obs_tf")
            print("mean nmse: " + str(np.mean(score.loc[score["model"] == "genie_obs_tf", "nmse"].values)))
            print("mean pearson: " + str(np.mean(score.loc[score["model"] == "genie_obs_tf", "pearson"].values)))
            print("mean cosine similarity: " + str(np.mean(score.loc[score["model"] == "genie_obs_tf", "cosine similarity"].values)))
            print()




# In[]
# for ngenes, ntfs in [(20, 5), (30, 10), (50, 20), (100, 50)]:
for ngenes, ntfs in [(20, 5), (30, 5), (50, 5)]:
    for interval in [50, 100, 200]:
        for stepsize in [0.0001, 0.0002]:
            result_dir = "../results_softODE_sergio/results_ngenes_" + str(ngenes) + "_interval_" + str(interval) + "_stepsize_" + str(stepsize) + "/"
            score = pd.read_csv(result_dir + "score.csv", index_col = 0)
            fig = plt.figure(figsize = (25,7))
            ax = fig.add_subplot()
            sns.boxplot(data = score, x = "model", y = "nmse", ax = ax)
            ax.set_title("NMSE: ngenes: {:d}, interval: {:d}, stepsize: {:f}".format(int(ngenes), int(interval), stepsize))
            fig.savefig(result_dir + "nmse.png", bbox_inches = "tight")

            fig = plt.figure(figsize = (25,7))
            ax = fig.add_subplot()
            sns.boxplot(data = score, x = "model", y = "pearson", ax = ax)
            ax.set_title("Pearson: ngenes: {:d}, interval: {:d}, stepsize: {:f}".format(int(ngenes), int(interval), stepsize))            
            fig.savefig(result_dir + "pearson.png", bbox_inches = "tight")

            fig = plt.figure(figsize = (25,7))
            ax = fig.add_subplot()
            sns.boxplot(data = score, x = "model", y = "cosine similarity", ax = ax)
            ax.set_title("Cosine: ngenes: {:d}, interval: {:d}, stepsize: {:f}".format(int(ngenes), int(interval), stepsize))
            fig.savefig(result_dir + "cosine.png", bbox_inches = "tight")







          

# In[]
# shows that the performance of genie with TF is much better than without TF
# shows that the performance of genie with standard scalar preprocessing and no preprocessing on true data are similar, 
# and they are better than log-normalize on observed data
'''  
score_models = pd.DataFrame(columns = ["model", "nmse", "pearson", "cosine similarity"])
# for ngenes, ntfs in [(20, 5), (30, 10), (50, 20), (100, 50)]:
for ngenes, ntfs in [(20, 5), (30, 5), (50, 5)]:
    for interval in [50, 100, 200]:
        for stepsize in [0.0001, 0.0002]:
            print("ngenes_" + str(ngenes) + "_interval_" + str(interval) + "_stepsize_" + str(stepsize))
            
            result_dir = "../results_softODE_sergio/results_ngenes_" + str(ngenes) + "_interval_" + str(interval) + "_stepsize_" + str(stepsize) + "/"
            score = pd.read_csv(result_dir + "score.csv", index_col = 0)
            score_random = score[score["model"] == "random"]
            score_genie_standardscaler = score[score["model"] == "genie_standardscaler"]
            score_genie_standardscaler_tf = score[score["model"] == "genie_standardscaler_tf"]
            score_genie = score[score["model"] == "genie"]
            score_genie_tf = score[score["model"] == "genie_tf"]
            score_genie_obs = score[score["model"] == "genie_obs"]
            score_genie_obs_tf = score[score["model"] == "genie_obs_tf"]   
            score_models = score_models.append({"nmse": score_random["nmse"].mean(), "pearson": score_random["pearson"].mean(), "cosine similarity": score_random["cosine similarity"].mean(), 
                                    "model":"random", "ngenes": ngenes, "interval": interval, "stepsize": stepsize}, ignore_index=True)         
            score_models = score_models.append({"nmse": score_genie_standardscaler["nmse"].mean(), "pearson": score_genie_standardscaler["pearson"].mean(), 
                                    "cosine similarity": score_genie_standardscaler["cosine similarity"].mean(), "model":"genie_standardscaler", "ngenes": ngenes, 
                                    "interval": interval, "stepsize": stepsize}, ignore_index=True)    
            score_models = score_models.append({"nmse": score_genie_standardscaler_tf["nmse"].mean(), "pearson": score_genie_standardscaler_tf["pearson"].mean(), 
                                    "cosine similarity": score_genie_standardscaler_tf["cosine similarity"].mean(), "model":"genie_standardscaler_tf", 
                                    "ngenes": ngenes, "interval": interval, "stepsize": stepsize}, ignore_index=True)    
            score_models = score_models.append({"nmse": score_genie["nmse"].mean(), "pearson": score_genie["pearson"].mean(), "cosine similarity": score_genie["cosine similarity"].mean(), 
                                    "model":"score_genie", "ngenes": ngenes, "interval": interval, "stepsize": stepsize}, ignore_index=True)    
            score_models = score_models.append({"nmse": score_genie["nmse"].mean(), "pearson": score_genie["pearson"].mean(), "cosine similarity": score_genie["cosine similarity"].mean(), 
                                    "model":"score_genie", "ngenes": ngenes, "interval": interval, "stepsize": stepsize}, ignore_index=True)
            score_models = score_models.append({"nmse": score_genie_tf["nmse"].mean(), "pearson": score_genie_tf["pearson"].mean(), "cosine similarity": score_genie_tf["cosine similarity"].mean(), 
                                    "model":"score_genie_tf", "ngenes": ngenes, "interval": interval, "stepsize": stepsize}, ignore_index=True)
            score_models = score_models.append({"nmse": score_genie_obs["nmse"].mean(), "pearson": score_genie_obs["pearson"].mean(), "cosine similarity": score_genie_obs["cosine similarity"].mean(), 
                                    "model":"score_genie_obs", "ngenes": ngenes, "interval": interval, "stepsize": stepsize}, ignore_index=True)
            score_models = score_models.append({"nmse": score_genie_obs_tf["nmse"].mean(), "pearson": score_genie_obs_tf["pearson"].mean(), "cosine similarity": score_genie_obs_tf["cosine similarity"].mean(), 
                                    "model":"score_genie_obs_tf", "ngenes": ngenes, "interval": interval, "stepsize": stepsize}, ignore_index=True)


# shows that the stepsize 0.0001 should generally have better performance than 0.0002 for genie3 with standard scalar initialization
print("genie_standardscaler")
display(score_models[score_models["model"] == "genie_standardscaler"])
print("genie_standardscaler_tf")
display(score_models[score_models["model"] == "genie_standardscaler_tf"])
fig = plt.figure(figsize = (25,7))
ax = fig.add_subplot()
sns.boxplot(data = score_models, x = "model", y = "nmse", ax = ax)
ax.set_title("NMSE")
fig.savefig("../results_softODE_sergio/nmse.png", bbox_inches = "tight")       
fig = plt.figure(figsize = (25,7))
ax = fig.add_subplot()
sns.boxplot(data = score_models, x = "model", y = "pearson", ax = ax)
ax.set_title("Pearson")
fig.savefig("../results_softODE_sergio/cosine.png", bbox_inches = "tight")       
fig = plt.figure(figsize = (25,7))
ax = fig.add_subplot()
sns.boxplot(data = score_models, x = "model", y = "cosine similarity", ax = ax)
ax.set_title("Cosine Sim")
fig.savefig("../results_softODE_sergio/cosine.png", bbox_inches = "tight") 

# In[] check how the number of genes and interval affect the result of genie with standardscalar
# the larger the number of gene is the lower the accuracy is. The faster the changing speed, the lower the accuracy is.
# The number of gene has major effect. The changing speed has minor effect.
score_models = pd.DataFrame(columns = ["model", "nmse","probability of success", "pearson", "cosine similarity", "time", "ngenes", "interval", "stepsize"])
# for ngenes, ntfs in [(20, 5), (30, 10), (50, 20), (100, 50)]:
for ngenes, ntfs in [(20, 5), (30, 5), (50, 5)]:
    for interval in [50, 100, 200]:
        for stepsize in [0.0001, 0.0002]:
            print("ngenes_" + str(ngenes) + "_interval_" + str(interval) + "_stepsize_" + str(stepsize))
            
            result_dir = "../results_softODE_sergio/results_ngenes_" + str(ngenes) + "_interval_" + str(interval) + "_stepsize_" + str(stepsize) + "/"
            score = pd.read_csv(result_dir + "score.csv", index_col = 0)
            score["ngenes"] = ngenes
            score["interval"] = interval
            score["stepsize"] = stepsize
            score_models = pd.concat([score_models, score], axis = 0)

score_models_genie_standardscalar = score_models[score_models["model"] == "genie_standardscaler"]
score_models_genie_standardscaler_tf = score_models[score_models["model"] == "genie_standardscaler_tf"]
fig = plt.figure(figsize = (20,7))
axs = fig.subplots(1,2)
sns.boxplot(data = score_models_genie_standardscalar, x = "interval", y = "pearson", hue = "ngenes", ax = axs[0])
sns.boxplot(data = score_models_genie_standardscalar, x = "interval", y = "cosine similarity", hue = "ngenes", ax = axs[1])
fig.savefig("../results_softODE_sergio/genie_standardscalar.png", bbox_inches = "tight") 

fig = plt.figure(figsize = (20,7))
axs = fig.subplots(1,2)
sns.boxplot(data = score_models_genie_standardscaler_tf, x = "interval", y = "pearson", hue = "ngenes", ax = axs[0])
sns.boxplot(data = score_models_genie_standardscaler_tf, x = "interval", y = "cosine similarity", hue = "ngenes", ax = axs[1])
fig.savefig("../results_softODE_sergio/genie_standardscaler_tf.png", bbox_inches = "tight") 
'''


# %%
