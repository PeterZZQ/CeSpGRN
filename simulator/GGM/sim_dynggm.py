# In[0]
# (main) simulate the GGM graph using diagonal domiance, following the glad implementation
# (optional) simulate the GGM graph using partial orthogonal, following the paper On generating random Gaussian graphical models, seems to have certain advantages??
# The key is to make the graph Symm+Positive definite
import numpy as np
from scipy.linalg import eigh
import os


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
    E,V = eigh(A)
    # print('min_eig = ', np.min(E) , 'max_eig = ', np.max(E), ' min_diag = ', np.min(np.diag(A)))
    # make sure symmetric positive definite
    return np.all(E > -tol) & np.allclose(A, A.T, atol = tol)


def dyn_GRN(setting = {}):
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

    Gs, Covs = dyn_GRN(setting = {"ngenes": ngenes, "ntimes": ntimes, "mode": "TF-TF&target", "ntfs": ntfs, "nchanges": nchanges, "change_stepsize": interval, "connected_acyclic": False, "seed": 0})
    samples = gen_samples_dynmean(Covs, nsamples = 1, seed = 0)
    X_umap = umap_op.fit_transform(samples.reshape(-1, ngenes))
    fig = plt.figure(figsize = (10, 7))
    ax = fig.add_subplot()
    ax.scatter(X_umap[:, 0], X_umap[:, 1], c = np.arange(X_umap.shape[0]), s = 5)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("UMAP plot")
    if not os.path.exists("../../data/GGM_changing_mean/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/"):
        os.makedirs("../../data/GGM_changing_mean/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/")

    np.save(file = "../../data/GGM_changing_mean/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/Gs.npy", arr = Gs)
    np.save(file = "../../data/GGM_changing_mean/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/expr.npy", arr = samples)
    fig.savefig("../../data/GGM_changing_mean/ntimes_" + str(ntimes) + "_interval_" + str(interval) + "_ngenes_" + str(ngenes) + "/umap.png", bbox_inches = "tight")

# In[1] generate data with zero mean
"""
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
"""
# %%

