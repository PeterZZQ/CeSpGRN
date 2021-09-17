# (main) simulate the GGM graph using diagonal domiance, following the glad implementation
# (optional) simulate the GGM graph using partial orthogonal, following the paper On generating random Gaussian graphical models, seems to have certain advantages??
# The key is to make the graph Symm+Positive definite
# In[0]
import numpy as np
from scipy.linalg import eigh


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
    if smallest_eigval <= 0:
        precision_mat = G + np.eye(G.shape[0])*(np.abs(smallest_eigval)+ bias)
    else:
        precision_mat = G
    return precision_mat

def isPSD(A, tol=1e-7):
    E,V = eigh(A)
    print('min_eig = ', np.min(E) , 'max_eig = ', np.max(E), ' min_diag = ', np.min(np.diag(A)))
    return np.all(E > -tol)


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
    ntfs = _setting["ntfs"]
    nchanges = _setting["nchanges"]
    change_stepsize = _setting["change_stepsize"]
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
        # M = np.zeros_like(G0)
        M[:ntfs,:ntfs] = 0
        M[ntfs:,:] = 0
    elif mode == "TF-TF&target":
        # M = np.zeros_like(G0)
        M[ntfs:,:] = 0
        M[np.tril_indices(ntfs)] = 0
        # M[:,ntfs:] = 0
        # # make sure connected acyclic? mimum spanning tree
        # if connected_acyclic:
        #     graph = nx.from_numpy_matrix(G0[:ntfs, :ntfs])
        #     # check connectivity, if not, restart with different seed
        #     assert nx.is_connected(graph)
        #     tree = nx.minimum_spanning_tree(graph)
        #     M[:ntfs, :ntfs] = (nx.to_numpy_array(tree) != 0)
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
        active_area[ntfs:,:] = -1
        active_area[np.tril_indices(ntfs)] = -1
    
    elif mode == "TF-target":
        active_area = np.arange(ngenes**2).reshape(ngenes,ngenes)
        active_area[ntfs:,:] = -1
        active_area[:ntfs,:][:,:ntfs] = -1

    # graph changes
    for time in range(1, ntimes):
        if (time - 1)%change_stepsize == 0:

            # some values are not exactly 0, numerical issue
            Gs[time - 1, :, :] = np.where(np.abs(Gs[time - 1, :, :]) < 1e-6, 0, Gs[time - 1, :, :])
            Gt = Gs[time - 1, :, :].reshape(-1)

            # delete, reduce to 0
            del_candid = np.array(list(set(np.where(Gt != 0)[0]).intersection(set(active_area.reshape(-1)))))
            del_idx = np.random.choice(del_candid, nchanges, replace = False)
            # add, increase to

            add_candid = np.array(list(set(np.where(Gt == 0)[0]).intersection(set(active_area.reshape(-1)))))
            add_idx = np.random.choice(add_candid, nchanges, replace = False)
            # add_value = np.random.normal(loc = 0, scale = 1, size = nchanges) / change_stepsize

            ### "Change" ###
            add_value = np.random.uniform(low = -1, high = 1, size = nchanges) / change_stepsize
            del_value = Gt[del_idx] / change_stepsize
        else:
            Gt = Gs[time - 1, :, :].reshape(-1)

        # update values
        Gt[add_idx] = Gt[add_idx] + add_value
        Gt[del_idx] = Gt[del_idx] - del_value
        Gt = Gt.reshape((ngenes, ngenes))

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
        assert isPSD(Gt)

        Gt = Gt.reshape(-1)

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
        data = np.random.multivariate_normal(mean = np.zeros(ngenes), cov = cov, size = nsamples)
        print(data.shape)
        datas.append(data[None, :, :])
    
    # datas of the shape ntimes, nsamples, ngenes
    datas = np.concatenate(datas, axis = 0)
    if nsamples == 1:
        datas = datas.squeeze()

    return datas

# In[1]
Gs, Covs = dyn_GRN(setting = {"ntimes": 100})
samples = gen_samples(Covs, nsamples = 1, seed = 0)

# %%
