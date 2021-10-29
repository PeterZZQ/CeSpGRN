# In[0]
# (main) simulate the GGM graph using diagonal domiance, following the glad implementation
# (optional) simulate the GGM graph using partial orthogonal, following the paper On generating random Gaussian graphical models, seems to have certain advantages??
# The key is to make the graph Symm+Positive definite
import numpy as np
from scanpy.neighbors import neighbors
from scipy.linalg import eigh
import os
import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.decomposition import PCA
plt.rcParams["font.size"] = 20
import pandas as pd
import scanpy as sc
from anndata import AnnData


def load_sub_sergio(grn_init, sub_size, ntfs, seed = 0, init_size = 100, mode = None):
    """\
    Parameters:
    -----------
        grn_init: 
            the directory of grn graph
        sub_size: 
            the size (number of genes) of the sampled grn graph, should be smaller than the total number of genes
        ntfs:
            the number of tfs of the sampled grn graph, should be smaller than the total number of tfs
        seed:
            random seed for sampling
        init_size:
            the graph size of the input grn
        mode:
            random selection or select the sub graph that is the most dense
    """
    import csv
    np.random.seed(seed)
    G_init = np.zeros((init_size,init_size))
    with open(grn_init+".txt","r") as f:
        reader = csv.reader(f, delimiter=",")
        target_list = []
        for row in reader:
            target = int(float(row[0]))
            n_tf = int(float(row[1]))
            target_list.append(target)
            
            # only consider activating regulation
            for tfId, K in zip(row[2: 2 + n_tf], row[2+n_tf : 2+2*n_tf]):
                if float(K) > 0:
                    tf = int(float(tfId))
                    G_init[tf,target] = float(K)

    tf_list = np.sort(np.unique((np.nonzero(G_init)[0])))
    target_list = np.sort(np.unique(np.array(target_list)))
    gene_list = np.sort(np.unique(np.concatenate((tf_list, target_list), axis = 0)))
    density = np.sum(G_init != 0)/(init_size ** 2)
    print("number of tfs: {:d}, number of target: {:d}, total number of genes: {:d}, edge density: {:4f}".format(tf_list.shape[0], target_list.shape[0], gene_list.shape[0], density))
    assert ntfs <= tf_list.shape[0]
    # make sure the target list doesn't have tfs
    target_list = np.sort(np.array(list(set([x for x in target_list]) - set([x for x in tf_list]))))
    G0 = np.zeros((sub_size,sub_size))



    # randomly select sub graph from SERGIO initial graph
    if mode == "random":
        selected_tf = np.random.choice(tf_list, ntfs, replace = False)
        selected_tf = np.sort(selected_tf)
        selected_target = np.random.choice(list(set(target_list)), sub_size-ntfs, replace = False)
        selected_target = np.sort(selected_target)
        G0[:ntfs,ntfs:] = G_init[np.ix_(selected_tf, selected_target)]
        G0[:ntfs,:ntfs] = G_init[np.ix_(selected_tf, selected_tf)]
        # make symmetric
        G0 = G0 + G0.T 

        # print("Subgraph of SERGIO: TF = {}\tTarget = {}\tDensity = {:.4f}".format(list(selected_tf), list(selected_target), np.sum(G0 != 0)/(sub_size ** 2)))
        print("subgraph with {:d} TFs and {:d} targets, the density is {:.4f}".format(selected_tf.shape[0], selected_target.shape[0], np.sum(G0 != 0)/(sub_size ** 2)))
    
    # select the most desnse (large number of edges) sub graph from SERGIO initial graph
    else:
        # select the tf with the largest out degree
        tf_oudegree = np.sum(G_init[np.ix_(tf_list, target_list)], axis = 1)
        selected_tf = tf_list[np.argsort(tf_oudegree)[-ntfs:]]
        # select the target with the highest number of in degree
        target_indegree = np.sum(G_init[np.ix_(selected_tf, target_list)], axis = 0)
        selected_target =  target_list[np.argsort(target_indegree)[-(sub_size-ntfs):]]
        G0[:ntfs,ntfs:] = G_init[np.ix_(selected_tf, selected_target)]
        G0[:ntfs,:ntfs] = G_init[np.ix_(selected_tf, selected_tf)]
        # make symmetric
        G0 = G0 + G0.T 

        # print("Subgraph of SERGIO: TF = {}\tTarget = {}\tDensity = {:.4f}".format(list(selected_tf), list(selected_target), np.sum(G0 != 0)/(sub_size ** 2)))
        print("subgraph with {:d} TFs and {:d} targets, the density is {:.4f}".format(selected_tf.shape[0], selected_target.shape[0], np.sum(G0 != 0)/(sub_size ** 2)))
    
    # n_edges = len(np.nonzero(G0)[0])
    # print("Number of edges: {}, Density: {}".format(n_edges, n_edges/(sub_size**2)))
            
    return G0

def make_PD(G, bias = 0.1):
    """\
    Description:
    ------------
        Make sure the graph is positive definite by diagonal domiance method
    Parameters:
    ------------
        G:
            Input graph adjacency matrix
        bia:
            Bias on diagonal value
    Return:
    ------------
        precision_mat: 
            Positive definite version of G
    """

    smallest_eigval = np.min(np.linalg.eigvals(G))
    # if make 0, then the smallest eigenvalue might be negative after inverse (numerical issue)
    if smallest_eigval <= 0.0:
        precision_mat = G + np.eye(G.shape[0])*(np.abs(smallest_eigval)+ bias)
    else:
        precision_mat = G
    return precision_mat

def isPSD(A, tol=1e-7):
    # E,V = eigh(A)
    E = np.linalg.eigvals(A)
    # print('min_eig = ', np.min(E) , 'max_eig = ', np.max(E), ' min_diag = ', np.min(np.diag(A)))
    # make sure symmetric positive definite
    return np.all(E > -tol) & np.allclose(A, A.T, atol = tol)

def dyn_GRN(setting):
    """\
    Description:
    ------------
        Generate graphs: precision matrices and the covariance matrices. Need to make sure the matrices are Symm PD
    Returns:
    ------------
        Gs: 
            precision matrices, of the shape (ntimes, ngenes, ngenes)
        Covs:
            sample covariance matrices 
    """
    # set parameters
    _setting = {"ngenes": 20, # number of genes
                "ntimes": 1000, # number of time steps
                "mode": "TF-TF&target", # mode, include: "TF-target" (only edges between TFs and targets), "TF-TF&target" (edges between TFs or edges between TFs and targets)
                "ntfs": 5, # ntfs: number of tfs
                "nchanges": 2,     # nchanges: how many edges to permute every ``change_stepsize'' steps
                "change_stepsize": 10,     # change_stepsize: permute edges every ``change_stepsize'' steps
                "backbone": np.array(["0_1"] * 1000), # the backbone branch belonging of each cell, of the form "A_B", where A and B are node ids starting from 0
                "G0": None,
                "seed": 0}
    _setting.update(setting)    
    ngenes, ntimes, mode, ntfs, nchanges, change_stepsize = _setting["ngenes"], _setting["ntimes"], \
        _setting["mode"], _setting["ntfs"], _setting["nchanges"], _setting["change_stepsize"]    


    np.random.seed(_setting["seed"])
    # stores the ground truth graphs
    Gs = [None] * ntimes
    # stores the covariance matrices
    Covs = [None] * ntimes
    # Initialize
    if _setting["G0"] is None:
        G0 = np.random.uniform(low = -1, high = 1, size = (ngenes, ngenes))
        # make symmetric
        G0 = G0 + G0.T

        # sparsity of initial graph
        threshold = 0.7
        M = (np.abs(G0) > threshold).astype(np.int)
        if mode ==  "TF-target":
            # assume the first ntfs are tf
            M[:ntfs,:ntfs] = 0
            M[ntfs:,ntfs:] = 0
        elif mode == "TF-TF&target":
            # make sure no interaction between target and target
            M[ntfs:,ntfs:] = 0
        
        G0 = G0 * M

        not_regulated = np.where(np.sum(G0, axis = 0) == 0)[0]
        # include self-regulation
        for i in not_regulated:
            G0[i, i] = np.random.uniform(low = 0, high = 1, size = 1)

    else:
        G0 = _setting["G0"].copy()


    # make sure the precision matrix is positive definite
    G0 = make_PD(G0)
    cov0 = np.linalg.inv(G0)
    Gs[0] = G0
    Covs[0] = cov0

    # check symm positive definite
    assert isPSD(G0)
    assert isPSD(cov0)


    if mode == "TF-TF&target":
        active_area = np.arange(ngenes**2).reshape(ngenes,ngenes)
        active_area[ntfs:,:] = -1
        active_area[np.tril_indices(ntfs)] = -1
    
    elif mode == "TF-target":
        active_area = np.arange(ngenes**2).reshape(ngenes,ngenes)
        active_area[ntfs:,:] = -1
        active_area[:ntfs,:ntfs] = -1

    # backbone, find all possible branches
    branches = sorted(list(set(setting["backbone"])))
    node_times = {}
    # root node
    node_times["0"] = 0
    for branch in branches:
        try:
            start_node, end_node = branch.split("_")
        except:
            raise ValueError("backbone should be of the form A_B, where A is the start node, and B is the end node")
        
        # find branching time of current branch, end nodes are unique in tree
        branch_times = np.where(setting["backbone"] == branch)[0]
        # assign branching time for the end_node
        if end_node not in node_times.keys():
            node_times[end_node] = np.max(branch_times)

    while len(branches) != 0:
        for branch in branches:
            start_node, end_node = branch.split("_")
            
            if Gs[node_times[start_node]] is not None:
                # the GRN for the starting point is already initialized
                # remove branch from branches
                branches.remove(branch)
                # use the corresponding branch
                break
        
        # find the time point correspond to current branch
        branch_times = np.where(setting["backbone"] == branch)[0]
        if Gs[node_times[start_node]] is not None:
            # initial graph in the branch, G0 will be updated this way.
            pre_G = Gs[node_times[start_node]]
            # graph changes
            for i, time in enumerate(branch_times):
                if i%change_stepsize == 0:
                    # some values are not exactly 0, numerical issue
                    pre_G = np.where(np.abs(pre_G) < 1e-6, 0, pre_G)
                    # some values are not exactly 0, numerical issue
                    Gt = pre_G.reshape(-1).copy()
                    if i != 0:
                        diff = Gs[branch_times[i - 1]] - Gs[branch_times[i - change_stepsize]]
                        diff = diff - np.diag(np.diag(diff))
                        assert len([(x,y) for x,y in zip(* np.where(del_mask!=0))]) == nchanges * 2
                        assert len([(x,y) for x,y in zip(* np.where(add_mask!=0))]) == nchanges * 2
                        assert len([(x,y) for x,y in zip(* np.where(np.abs(diff) > 1e-6))]) == nchanges * 4

                    # delete, reduce to 0
                    del_candid = np.array(sorted(list(set(np.where(Gt != 0)[0]).intersection(set(active_area.reshape(-1))))))
                    del_idx = np.random.choice(del_candid, nchanges, replace = False)
                    assert len(del_idx) == nchanges
                    # add, increase to
                    add_candid = np.array(sorted(list(set(np.where(Gt == 0)[0]).intersection(set(active_area.reshape(-1))))))
                    add_idx = np.random.choice(add_candid, nchanges, replace = False)
                    assert len(add_idx) == nchanges

                    # make del_idx and add_idx symm
                    del_mask = np.zeros_like(Gt)
                    del_mask[del_idx] = 1
                    add_mask = np.zeros_like(Gt)
                    # add_mask[add_idx] = np.random.uniform(low = -4, high = 4, size = nchanges) / change_stepsize
                    add_mask[add_idx[:int(nchanges/2)]] = np.random.uniform(low = 3.5, high = 4, size = int(nchanges/2)) / change_stepsize
                    add_mask[add_idx[int(nchanges/2):]] = np.random.uniform(low = -4, high = -3.5, size = nchanges - int(nchanges/2))/ change_stepsize

                    Gt = Gt.reshape((ngenes, ngenes))
                    del_mask = del_mask.reshape((ngenes, ngenes))
                    add_mask = add_mask.reshape((ngenes, ngenes))
                    del_mask = del_mask + del_mask.T
                    del_mask = Gt * del_mask / change_stepsize
                    assert np.allclose(del_mask, del_mask.T, atol = 1e-7)
                    # print(Gt)
                    # print()

                    # print(del_mask)
                    # print()

                    add_mask = add_mask + add_mask.T
                    assert np.allclose(add_mask, add_mask.T, atol = 1e-7)
                    # print(add_mask)
                    # print()
                else:
                    Gt = pre_G.copy()

                # update values
                Gt = Gt + add_mask
                Gt = Gt - del_mask

                # make sure the genes that are not regulated by any genes are self-regulating
                not_regulated = np.where(np.sum(Gt, axis = 0) == 0)[0]
                # include self-regulation
                for i in not_regulated:
                    Gt[i, i] = np.random.uniform(low = 0, high = 1, size = 1) 

                # make sure the precision matrix is positive definite
                Gt = make_PD(Gt)
                covt = np.linalg.inv(Gt)
                Gs[time] = Gt
                Covs[time] = covt
                # for the next loop, don't need to copy, thresholding pre_G with update Gs too
                pre_G = Gs[time]

                # check symm positive definite
                # print(time)
                assert isPSD(Gt)
                assert isPSD(covt)

    Gs = np.concatenate([G[None, :, :] for G in Gs], axis = 0)
    Covs = np.concatenate([G[None, :, :] for G in Covs], axis = 0)
    return Gs, Covs


def gen_samples(Covs, nsamples = 1, seed = 0):
    """\
    Description:
    ------------
        Generate samples from Gaussian Graphical Model using Covariance matrices
    Parameters:
    ------------
        Covs:
            Sample covariance matrix, of the shape (ntimes, ngenes, ngenes)
    Return:
    ------------
        Generated samples

    """
    if seed != None:
        np.random.seed(seed)

    ntimes, ngenes, ngenes = Covs.shape
    datas = []
    for t in range(ntimes):
        cov = Covs[t, :, :]
        assert isPSD(cov)
        data = np.random.multivariate_normal(mean = np.zeros(ngenes), cov = cov, size = nsamples)

        datas.append(data[None, :, :])
    
    # datas of the shape ntimes, nsamples, ngenes
    datas = np.concatenate(datas, axis = 0)
    if nsamples == 1:
        datas = datas.squeeze()

    return datas

def gen_samples_backbone(Covs, backbone, nsamples = 1, seed = 0):
    """\
    Description:
    ------------
        Generate samples from Gaussian Graphical Model using Covariance matrices, instead of zero-mean, mean is changing with time.
    Parameters:
    ------------
        Covs:
            Sample covariance matrix, of the shape (ntimes, ngenes, ngenes)
    Return:
    ------------
        Generated samples

    """
    assert backbone.shape[0] == Covs.shape[0]
    if seed != None:
        np.random.seed(seed)

    ntimes, ngenes, ngenes = Covs.shape
    datas = []
    sim_time = np.zeros((ntimes, nsamples))

    # unique backbones
    branches = sorted(list(set([x for x in backbone])))
    node_expr = {}
    node_time = {}
    node_expr["0"] = np.zeros(ngenes)
    node_time["0"] = 0
    while len(branches) != 0:
        for branch in branches:
            start_node, end_node = branch.split("_")
            if start_node in node_expr.keys():
                # remove branch from branches
                branches.remove(branch)
                # use the corresponding branch
                break

        if start_node in node_expr.keys():
            # find branching time of current branch, end nodes are unique in tree
            branch_idx = np.where(backbone == branch)[0]
            # simulation time for the branch
            pre_expr = node_expr[start_node]
            pre_time = node_time[start_node]
            for i, idx in enumerate(branch_idx):
                # get the covariance matrix correspond to the idx
                cov = Covs[idx, :, :]
                data = np.random.multivariate_normal(mean = pre_expr, cov = cov, size = nsamples)
                # add generated data sample to the dataset
                datas.append(data[None, :, :])
                # update the simulation time, assume the stepsize is 1
                time = pre_time + 1
                sim_time[idx, :] = time
                # get the mean for next sampling procedure
                pre_expr = np.mean(datas[-1], axis = 1).squeeze()
                pre_time = time
            # the last expression data
            node_expr[end_node] = pre_expr
            node_time[end_node] = pre_time            
    
    # datas of the shape ntimes, nsamples, ngenes
    datas = np.concatenate(datas, axis = 0)
    if nsamples == 1:
        datas = datas.squeeze()
        sim_time = sim_time.squeeze()

    return datas, sim_time    

def simulate_data(setting):
    # set parameters
    _setting = {"ngenes": 20, # number of genes
                "ntimes": 1000, # number of time steps
                "nsamples": 1,
                "mode": "TF-TF&target", # mode, include: "TF-target" (only edges between TFs and targets), "TF-TF&target" (edges between TFs or edges between TFs and targets)
                "ntfs": 5, # ntfs: number of tfs
                "nchanges": 2,     # nchanges: how many edges to permute every ``change_stepsize'' steps
                "change_stepsize": 10,     # change_stepsize: permute edges every ``change_stepsize'' steps
                "backbone": np.array(["0_1"] * 1000), # the backbone branch belonging of each cell, of the form "A_B", where A and B are node ids starting from 0
                "seed": 0,
                "G0": None
                }
    _setting.update(setting)   
    # generate dynamic grn 
    Gs, Covs = dyn_GRN(setting)
    # simulate data from grn
    samples, pt = gen_samples_backbone(Covs, backbone = _setting["backbone"], nsamples = _setting["nsamples"], seed = _setting["seed"])
    # summarize the result
    result = {
        "samples": samples,
        "sim time": pt,
        "GRNs": np.repeat(Gs[:, None, :], repeats = _setting["nsamples"], axis = 1),
        "Covs": np.repeat(Covs[:, None, :], repeats = _setting["nsamples"], axis = 1),
    }

    return result

# In[0] generate data with bifurcating backbone

umap_op = UMAP(n_components = 2, min_dist = 0.9, n_neighbors = 30, random_state = 0)
pca_op = PCA(n_components = 2)

for interval in [5, 25, 100]:
    for (ngenes, ntfs) in [(50, 20), (200, 20)]:
        for seed in range(5):
            ntimes = 1000
            backbone = np.array(["0_1"] * 200 + ["1_2"] * 400 + ["1_3"] * 400)
            nsamples = 1
            nchanges = 2

            if not os.path.exists("../../data/GGM_bifurcate/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) +  "_" + str(seed) + "_sergio/"):
                os.makedirs("../../data/GGM_bifurcate/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) +  "_" + str(seed) + "_sergio/")

            # Interaction_cID_5 is a sergio data which has 400 genes in total
            sergio_path = "../soft_boolODE/sergio_data/Interaction_cID_5"
            G0 = load_sub_sergio(grn_init = sergio_path, 
                                 sub_size = ngenes, 
                                 ntfs = ntfs, 
                                 seed = seed, 
                                 init_size = 400, 
                                 mode = "dense"
                                 )

            setting = {"ngenes": ngenes, 
                    "ntimes": ntimes, 
                    "nsamples": nsamples,
                    "mode": "TF-TF&target", 
                    "ntfs": ntfs, 
                    "nchanges": nchanges, 
                    "change_stepsize": interval, 
                    "seed": seed,
                    "backbone": backbone,
                    "G0": G0
                    }
            # run simulation
            results = simulate_data(setting)
            # obtain result
            samples = results["samples"].reshape(setting["ntimes"] * setting["nsamples"], setting["ngenes"])
            pt = results["sim time"].reshape(setting["ntimes"] * setting["nsamples"])
            Gs = results["GRNs"].reshape(setting["ntimes"] * setting["nsamples"], setting["ngenes"], setting["ngenes"])

            np.save(file = "../../data/GGM_bifurcate/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) +  "_" + str(seed) + "_sergio/Gs.npy", arr = Gs)
            np.save(file = "../../data/GGM_bifurcate/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes)  +  "_" + str(seed) + "_sergio/expr.npy", arr = samples)
            np.save(file = "../../data/GGM_bifurcate/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) +  "_" + str(seed) + "_sergio/sim_time.npy", arr = pt)
            # save for SCODE format, row correspond to gene, column correspond to cell, no index and header, \t separated txt file
            samples_tf = pd.DataFrame(data = samples.T)
            samples_tf.to_csv("../../data/GGM_bifurcate/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) +  "_" + str(seed) + "_sergio/expr.txt", sep = "\t", index = False, header = False)
            # save for SCODE format, row correspond to cell, column one is the index, column two is the pseudotime
            pt_df = pd.DataFrame(data = pt[:,None])
            pt_df.to_csv("../../data/GGM_bifurcate/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) +  "_" + str(seed) + "_sergio/sim_time.txt", sep = "\t", index = True, header = False)


            X_umap = umap_op.fit_transform(samples.reshape(-1, ngenes))
            fig = plt.figure(figsize = (10, 7))
            ax = fig.add_subplot()
            ax.scatter(X_umap[:, 0], X_umap[:, 1], c = pt, s = 5)
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.set_title("UMAP plot")
            fig.savefig("../../data/GGM_bifurcate/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) +  "_" + str(seed) + "_sergio/umap.png", bbox_inches = "tight")

            X_pca = pca_op.fit_transform(samples.reshape(-1, ngenes))
            fig = plt.figure(figsize = (10, 7))
            ax = fig.add_subplot()
            ax.scatter(X_pca[:, 0], X_pca[:, 1], c = pt, s = 5)
            ax.set_xlabel("PCA 1")
            ax.set_ylabel("PCA 2")
            ax.set_title("PCA plot")
            fig.savefig("../../data/GGM_bifurcate/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) +  "_" + str(seed) + "_sergio/pca.png", bbox_inches = "tight")


            fig = plt.figure(figsize = (10, 7))
            ax = fig.add_subplot()
            X = Gs.reshape(-1,setting["ngenes"] ** 2)
            G_umap = pca_op.fit_transform(X)
            ax.scatter(G_umap[:, 0], G_umap[:, 1], s = 5, c = pt)
            fig.savefig("../../data/GGM_bifurcate/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) +  "_" + str(seed) + "_sergio/grn_plot.png", bbox_inches = "tight")

            # diffusion pseudotime
            adata = AnnData(X = samples)
            adata.obsm["X_pca"] = X_pca
            adata.uns['iroot'] = 0
            sc.pp.neighbors(adata, n_neighbors = 30)
            sc.tl.diffmap(adata, n_comps = 5)
            sc.tl.dpt(adata, n_dcs = 5)

            pt_est = adata.obs["dpt_pseudotime"].values.squeeze()
            fig = plt.figure(figsize = (10, 7))
            ax = fig.add_subplot()
            ax.scatter(X_umap[:, 0], X_umap[:, 1], c = pt_est, s = 5)
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.set_title("UMAP plot (dpt)")
            fig.savefig("../../data/GGM_bifurcate/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) +  "_" + str(seed) + "_sergio/umap_dpt.png", bbox_inches = "tight")

            pt_df = pd.DataFrame(data = pt_est[:,None])
            pt_df.to_csv("../../data/GGM_bifurcate/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) +  "_" + str(seed) + "_sergio/dpt_time.txt", sep = "\t", index = True, header = False)


for interval in [5, 25, 100]:
    for (ngenes, ntfs) in [(50, 20), (200, 20)]:
        for seed in range(5):
            ntimes = 1000
            backbone = np.array(["0_1"] * 200 + ["1_2"] * 400 + ["1_3"] * 400)
            nsamples = 1
            nchanges = 2

            if not os.path.exists("../../data/GGM_bifurcate/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) +  "_" + str(seed) + "_random/"):
                os.makedirs("../../data/GGM_bifurcate/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) +  "_" + str(seed) + "_random/")

            # Interaction_cID_5 is a sergio data which has 400 genes in total
            sergio_path = "../soft_boolODE/sergio_data/Interaction_cID_5"
            G0 = load_sub_sergio(grn_init = sergio_path, 
                                 sub_size = ngenes, 
                                 ntfs = ntfs, 
                                 seed = seed, 
                                 init_size = 400, 
                                 mode = "dense"
                                 )

            setting = {"ngenes": ngenes, 
                    "ntimes": ntimes, 
                    "nsamples": nsamples,
                    "mode": "TF-TF&target", 
                    "ntfs": ntfs, 
                    "nchanges": nchanges, 
                    "change_stepsize": interval, 
                    "seed": seed,
                    "backbone": backbone,
                    "G0": None
                    }
            # run simulation
            results = simulate_data(setting)
            # obtain result
            samples = results["samples"].reshape(setting["ntimes"] * setting["nsamples"], setting["ngenes"])
            pt = results["sim time"].reshape(setting["ntimes"] * setting["nsamples"])
            Gs = results["GRNs"].reshape(setting["ntimes"] * setting["nsamples"], setting["ngenes"], setting["ngenes"])

            np.save(file = "../../data/GGM_bifurcate/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) +  "_" + str(seed) + "_random/Gs.npy", arr = Gs)
            np.save(file = "../../data/GGM_bifurcate/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes)  +  "_" + str(seed) + "_random/expr.npy", arr = samples)
            np.save(file = "../../data/GGM_bifurcate/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) +  "_" + str(seed) + "_random/sim_time.npy", arr = pt)
            # save for SCODE format, row correspond to gene, column correspond to cell, no index and header, \t separated txt file
            samples_tf = pd.DataFrame(data = samples.T)
            samples_tf.to_csv("../../data/GGM_bifurcate/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) +  "_" + str(seed) + "_random/expr.txt", sep = "\t", index = False, header = False)
            # save for SCODE format, row correspond to cell, column one is the index, column two is the pseudotime
            pt_df = pd.DataFrame(data = pt[:,None])
            pt_df.to_csv("../../data/GGM_bifurcate/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) +  "_" + str(seed) + "_random/sim_time.txt", sep = "\t", index = True, header = False)


            X_umap = umap_op.fit_transform(samples.reshape(-1, ngenes))
            fig = plt.figure(figsize = (10, 7))
            ax = fig.add_subplot()
            ax.scatter(X_umap[:, 0], X_umap[:, 1], c = pt, s = 5)
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.set_title("UMAP plot")
            fig.savefig("../../data/GGM_bifurcate/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) +  "_" + str(seed) + "_random/umap.png", bbox_inches = "tight")

            X_pca = pca_op.fit_transform(samples.reshape(-1, ngenes))
            fig = plt.figure(figsize = (10, 7))
            ax = fig.add_subplot()
            ax.scatter(X_pca[:, 0], X_pca[:, 1], c = pt, s = 5)
            ax.set_xlabel("PCA 1")
            ax.set_ylabel("PCA 2")
            ax.set_title("PCA plot")
            fig.savefig("../../data/GGM_bifurcate/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) +  "_" + str(seed) + "_random/pca.png", bbox_inches = "tight")


            fig = plt.figure(figsize = (10, 7))
            ax = fig.add_subplot()
            X = Gs.reshape(-1,setting["ngenes"] ** 2)
            G_umap = pca_op.fit_transform(X)
            ax.scatter(G_umap[:, 0], G_umap[:, 1], s = 5, c = pt)
            fig.savefig("../../data/GGM_bifurcate/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) +  "_" + str(seed) + "_random/grn_plot.png", bbox_inches = "tight")

            # diffusion pseudotime
            adata = AnnData(X = samples)
            adata.obsm["X_pca"] = X_pca
            adata.uns['iroot'] = 0
            sc.pp.neighbors(adata, n_neighbors = 30)
            sc.tl.diffmap(adata, n_comps = 5)
            sc.tl.dpt(adata, n_dcs = 5)

            pt_est = adata.obs["dpt_pseudotime"].values.squeeze()
            fig = plt.figure(figsize = (10, 7))
            ax = fig.add_subplot()
            ax.scatter(X_umap[:, 0], X_umap[:, 1], c = pt_est, s = 5)
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.set_title("UMAP plot (dpt)")
            fig.savefig("../../data/GGM_bifurcate/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) +  "_" + str(seed) + "_random/umap_dpt.png", bbox_inches = "tight")

            pt_df = pd.DataFrame(data = pt_est[:,None])
            pt_df.to_csv("../../data/GGM_bifurcate/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) +  "_" + str(seed) + "_random/dpt_time.txt", sep = "\t", index = True, header = False)


# In[] generate with linear trajectory
'''
umap_op = UMAP(n_components = 2, min_dist = 0.9, n_neighbors = 30, random_state = 0)
pca_op = PCA(n_components = 2)

# for interval in [5, 10, 25, 50, 100, 200]:
for interval in [2, 5, 10, 25, 50]:
    for (ngenes, ntfs) in [(20, 5), (30, 10), (50, 20), (100, 50)]:
        # ntimes = 1000
        # nsamples = 1
        # nchanges = 2
        # backbone = np.array(["0_1"] * ntimes)

        ntimes = 250
        nsamples = 10
        nchanges = 2   
        backbone = np.array(["0_1"] * ntimes)
        if not os.path.exists("../../data/GGM_linear/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/"):
            os.makedirs("../../data/GGM_linear/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/")

        setting = {"ngenes": ngenes, 
                "ntimes": ntimes, 
                "nsamples": nsamples,
                "mode": "TF-TF&target", 
                "ntfs": ntfs, 
                "nchanges": nchanges, 
                "change_stepsize": interval, 
                "seed": 0,
                "backbone": backbone
                }
        # run simulation
        results = simulate_data(setting)
        # obtain result
        samples = results["samples"].reshape(setting["ntimes"] * setting["nsamples"], setting["ngenes"])
        pt = results["sim time"].reshape(setting["ntimes"] * setting["nsamples"])
        Gs = results["GRNs"].reshape(setting["ntimes"] * setting["nsamples"], setting["ngenes"], setting["ngenes"])
        
        # samples_old = np.load("../../data/GGM_linear/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/expr.npy")
        # Gs_old = np.load("../../data/GGM_linear/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/Gs.npy")

        # assert np.allclose(samples_old, samples)
        # assert np.allclose(Gs_old, Gs)
        # print("the same")
        
        np.save(file = "../../data/GGM_linear/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/Gs.npy", arr = Gs)
        np.save(file = "../../data/GGM_linear/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/expr.npy", arr = samples)
        np.save(file = "../../data/GGM_linear/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/sim_time.npy", arr = pt)
        # save for SCODE format, row correspond to gene, column correspond to cell, no index and header, \t separated txt file
        samples_tf = pd.DataFrame(data = samples.T)
        samples_tf.to_csv("../../data/GGM_linear/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/expr.txt", sep = "\t", index = False, header = False)
        # save for SCODE format, row correspond to cell, column one is the index, column two is the pseudotime
        pt_df = pd.DataFrame(data = pt[:,None])
        pt_df.to_csv("../../data/GGM_linear/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/sim_time.txt", sep = "\t", index = True, header = False)

        X_umap = umap_op.fit_transform(samples.reshape(-1, ngenes))
        fig = plt.figure(figsize = (10, 7))
        ax = fig.add_subplot()
        ax.scatter(X_umap[:, 0], X_umap[:, 1], c = pt, s = 5)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title("UMAP plot")
        fig.savefig("../../data/GGM_linear/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/umap.png", bbox_inches = "tight")

        X_pca = pca_op.fit_transform(samples.reshape(-1, ngenes))
        fig = plt.figure(figsize = (10, 7))
        ax = fig.add_subplot()
        ax.scatter(X_pca[:, 0], X_pca[:, 1], c = pt, s = 5)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_title("PCA plot")
        fig.savefig("../../data/GGM_linear/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/pca.png", bbox_inches = "tight")


        fig = plt.figure(figsize = (10, 7))
        ax = fig.add_subplot()
        X = Gs.reshape(-1,setting["ngenes"] ** 2)
        G_umap = pca_op.fit_transform(X)
        ax.scatter(G_umap[:, 0], G_umap[:, 1], s = 5, c = pt)
        fig.savefig("../../data/GGM_linear/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/grn_plot.png", bbox_inches = "tight")

        # diffusion pseudotime
        adata = AnnData(X = samples)
        adata.obsm["X_pca"] = X_pca
        adata.uns['iroot'] = 0
        sc.pp.neighbors(adata, n_neighbors = 30)
        sc.tl.diffmap(adata, n_comps = 5)
        sc.tl.dpt(adata, n_dcs = 5)

        pt_est = adata.obs["dpt_pseudotime"].values.squeeze()
        fig = plt.figure(figsize = (10, 7))
        ax = fig.add_subplot()
        ax.scatter(X_umap[:, 0], X_umap[:, 1], c = pt_est, s = 5)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title("UMAP plot (dpt)")
        fig.savefig("../../data/GGM_linear/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/umap_dpt.png", bbox_inches = "tight")

        pt_df = pd.DataFrame(data = pt_est[:,None])
        pt_df.to_csv("../../data/GGM_linear/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/dpt_time.txt", sep = "\t", index = True, header = False)


# In[0] generate data with changing mean

import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.decomposition import PCA
plt.rcParams["font.size"] = 20

umap_op = UMAP(n_components = 2, min_dist = 0.5)
pca_op = PCA(n_components = 2)

for (ngenes, ntfs) in [(20, 5), (30, 10), (50, 20), (100, 50)]:
    ntimes = 1000
    interval = 200
    nchanges = 2
    if not os.path.exists("../../data/GGM_changing_mean/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/"):
        os.makedirs("../../data/GGM_changing_mean/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/")

    Gs, Covs = dyn_GRN(setting = {"ngenes": ngenes, "ntimes": ntimes, "mode": "TF-TF&target", "ntfs": ntfs, "nchanges": nchanges, "change_stepsize": interval, "connected_acyclic": False, "seed": 0})
    samples = gen_samples_dynmean(Covs, nsamples = 1, seed = 0)
    X_umap = umap_op.fit_transform(samples.reshape(-1, ngenes))
    fig = plt.figure(figsize = (10, 7))
    ax = fig.add_subplot()
    ax.scatter(X_umap[:, 0], X_umap[:, 1], c = np.arange(X_umap.shape[0]), s = 5)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("UMAP plot")
    fig.savefig("../../data/GGM_changing_mean/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/umap.png", bbox_inches = "tight")

    fig = plt.figure(figsize = (10, 7))
    ax = fig.add_subplot()
    X = Gs.reshape(1000, -1)
    X_umap = pca_op.fit_transform(X)
    ax.scatter(X_umap[:, 0], X_umap[:, 1], s = 5, c = np.arange(X_umap.shape[0]))
    fig.savefig("../../data/GGM_changing_mean/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/grn_plot.png", bbox_inches = "tight")

    np.save(file = "../../data/GGM_changing_mean/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/Gs.npy", arr = Gs)
    np.save(file = "../../data/GGM_changing_mean/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/expr.npy", arr = samples)
    break
'''
# In[1] generate data with zero mean
'''
for (ngenes, ntfs) in [(20, 5), (30, 10), (50, 20), (100, 50)]:
    # ngenes = 100
    # ntfs = 5
    ntimes = 3000
    interval = 50
    nchanges = 2
    Gs, Covs = dyn_GRN(setting = {"ngenes": ngenes, "ntimes": ntimes, "mode": "TF-TF&target", "ntfs": ntfs, "nchanges": nchanges, "change_stepsize": interval, "connected_acyclic": False, "seed": 0})
    samples = gen_samples(Covs, nsamples = 1, seed = 0)

    if not os.path.exists("../../data/GGM/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/"):
        os.makedirs("../../data/GGM/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/")

    np.save(file = "../../data/GGM/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/Gs.npy", arr = Gs)
    np.save(file = "../../data/GGM/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/expr.npy", arr = samples)
'''
# %%


'''
def dyn_GRN_stable(setting = {}):
    """\
    Description:
    ------------
        Generate graphs: precision matrices and the covariance matrices. Need to make sure the matrices are Symm PD
    Returns:
    ------------
        Gs: 
            precision matrices, of the shape (ntimes, ngenes, ngenes)
        Covs:
            sample covariance matrices 
    """
    # set parameters
    _setting = {"ngenes": 20, "ntimes": 1000, "mode": "TF-TF&target", "ntfs": 5, "nchanges": 2, "change_stepsize": 10, "connected_acyclic": False, "seed": 0}
    _setting.update(setting)
    # number of genes
    ngenes = _setting["ngenes"]
    # number of time steps
    ntimes = _setting["ntimes"]
    # mode, include: "TF-target" (only edges between TFs and targets), "TF-TF&target" (edges between TFs or edges between TFs and targets)
    mode = _setting["mode"]
    # ntfs: number of tfs
    ntfs = _setting["ntfs"]
    # nchanges: how many edges to permute every ``change_stepsize'' steps
    nchanges = _setting["nchanges"]
    # change_stepsize: permute edges every ``change_stepsize'' steps
    change_stepsize = _setting["change_stepsize"]
    # connected_acyclic: if the graph include cycles, under development
    connected_acyclic = _setting["connected_acyclic"]

    np.random.seed(_setting["seed"])
    # stores the ground truth graphs
    Gs = np.zeros((ntimes, ngenes, ngenes))
    # stores the covariance matrices
    Covs = np.zeros_like(Gs)
    # Initialize
    G0 = np.random.uniform(low = -1, high = 1, size = (ngenes, ngenes))
    # make symmetric
    G0 = G0 + G0.T

    # sparsity of initial graph
    threshold = 0.7
    M = (np.abs(G0) > threshold).astype(np.int)
    if mode ==  "TF-target":
        # assume the first ntfs are tf
        M[:ntfs,:ntfs] = 0
        M[ntfs:,ntfs:] = 0
    elif mode == "TF-TF&target":
        # make sure no interaction between target and target
        M[ntfs:,ntfs:] = 0
    
    G0 = G0 * M

    not_regulated = np.where(np.sum(G0, axis = 0) == 0)[0]
    # include self-regulation
    for i in not_regulated:
        G0[i, i] = np.random.uniform(low = 0, high = 1, size = 1)

    # make sure the precision matrix is positive definite
    G0 = make_PD(G0)
    cov0 = np.linalg.inv(G0)
    Gs[0, :, :] = G0
    Covs[0, :, :] = cov0

    # check symm positive definite
    assert isPSD(G0)
    assert isPSD(cov0)


    if mode == "TF-TF&target":
        active_area = np.arange(ngenes**2).reshape(ngenes,ngenes)
        active_area[ntfs:,ntfs:] = -1
        active_area[np.tril_indices(ntfs)] = -1
    
    elif mode == "TF-target":
        active_area = np.arange(ngenes**2).reshape(ngenes,ngenes)
        active_area[ntfs:,ntfs:] = -1
        active_area[:ntfs,:ntfs] = -1

    # graph changes
    for time in range(1, ntimes):
        if (time - 1)%change_stepsize == 0:

            # some values are not exactly 0, numerical issue
            Gs[time - 1, :, :] = np.where(np.abs(Gs[time - 1, :, :]) < 1e-6, 0, Gs[time - 1, :, :])
            Gt = Gs[time - 1, :, :].reshape(-1)

            # delete, reduce to 0
            del_candid = np.array(list(set(np.where(Gt != 0)[0]).intersection(set(active_area.reshape(-1)))))
            del_idx = np.random.choice(del_candid, nchanges, replace = False)
            assert len(del_idx) == nchanges
            # add, increase to
            add_candid = np.array(list(set(np.where(Gt == 0)[0]).intersection(set(active_area.reshape(-1)))))
            add_idx = np.random.choice(add_candid, nchanges, replace = False)
            assert len(add_idx) == nchanges

            # make del_idx and add_idx symm
            del_mask = np.zeros_like(Gt)
            del_mask[del_idx] = 1
            add_mask = np.zeros_like(Gt)
            add_mask[add_idx] = np.random.uniform(low = -2, high = 2, size = nchanges) / change_stepsize
            Gt = Gt.reshape((ngenes, ngenes))
            del_mask = del_mask.reshape((ngenes, ngenes))
            add_mask = add_mask.reshape((ngenes, ngenes))
            del_mask = del_mask + del_mask.T
            del_mask = Gt * del_mask / change_stepsize
            assert np.allclose(del_mask, del_mask.T, atol = 1e-7)
            # print(Gt)
            # print()

            # print(del_mask)
            # print()

            add_mask = add_mask + add_mask.T
            assert np.allclose(add_mask, add_mask.T, atol = 1e-7)
            # print(add_mask)
            # print()
        else:
            Gt = Gs[time - 1, :, :]

        # update values
        Gt = Gt + add_mask
        Gt = Gt - del_mask

        # make sure the genes that are not regulated by any genes are self-regulating
        not_regulated = np.where(np.sum(Gt, axis = 0) == 0)[0]
        # include self-regulation
        for i in not_regulated:
            Gt[i, i] = np.random.uniform(low = 0, high = 1, size = 1) 

        # make sure the precision matrix is positive definite
        Gt = make_PD(Gt)
        covt = np.linalg.inv(Gt)
        Gs[time, :, :] = Gt
        Covs[time, :, :] = covt

        # check symm positive definite
        # print(time)
        assert isPSD(Gt)
        assert isPSD(covt)

    return Gs, Covs

def gen_samples_dynmean(Covs, nsamples = 1, seed =0, change_interval = 1):
    """\
    Description:
    ------------
        Generate samples from Gaussian Graphical Model using Covariance matrices, instead of zero-mean, mean is changing with time.
    Parameters:
    ------------
        Covs:
            Sample covariance matrix, of the shape (ntimes, ngenes, ngenes)
    Return:
    ------------
        Generated samples

    """
    if seed != None:
        np.random.seed(seed)

    ntimes, ngenes, ngenes = Covs.shape
    datas = []
    for t in range(ntimes):
        if t == 0:
            mean_expr = np.zeros(ngenes)
        else:
            mean_expr = np.mean(datas[-change_interval], axis = 1).squeeze()
        cov = Covs[t, :, :]
        assert isPSD(cov)
        data = np.random.multivariate_normal(mean = mean_expr, cov = cov, size = nsamples)

        datas.append(data[None, :, :])
    
    # datas of the shape ntimes, nsamples, ngenes
    datas = np.concatenate(datas, axis = 0)
    if nsamples == 1:
        datas = datas.squeeze()

    return datas  

'''
