import numpy as np
from scipy.spatial.distance import pdist, squareform
import networkx as nx

def calc_kernel_neigh(X, k = 5, bandwidth = 1, truncate = False, truncate_param = 30):
    """\
    Description:
    ------------
        Calculate the similarity kernel using gene expression data
    Parameter:
    ------------
        X: gene expression data
        k: the number of nearest neighbor
        bandwidth: the bandwidth of gaussian kernel function
        truncate: truncate out small values in Gaussian kernel function or not
    Return:
    ------------
        K: the weighted kernel function, the sum of each row is normalized to 1
        K_trun: the truncated weighted kernel function
    """
    # calculate pairwise distance
    D = squareform(pdist(X))
    for k in range(k, D.shape[0]):
        # calculate the knn graph from pairwise distance
        knn_index = np.argpartition(D, kth = k - 1, axis=1)[:, (k-1)]
        # find the value of the k-th distance
        kth_dist = np.take_along_axis(D, knn_index[:,None], axis = 1)
        # construct KNN graph
        knn = (D <= kth_dist)
        # make the graph symmetric
        knn = ((knn + knn.T) > 0).astype(np.int)
        # knn = knn * knn.T
        # construct nextworkx graph from weighted knn graph, should make sure the G is connected
        G = nx.from_numpy_array(D * knn)
        # make sure that G is undirected
        assert ~nx.is_directed(G)
        # break while loop if connected
        if nx.is_connected(G):
            break
        else:
            k += 1
        print("number of nearest neighbor: " + str(k))

    print("final number of nearest neighbor (make connected): " + str(k))
    # return a matrix of shortest path distances between nodes. Inf if no distances between nodes (should be no Inf, because the graph is connected)
    D = np.array(nx.floyd_warshall_numpy(G))
    assert np.max(D) < np.inf
    # scale the distance to between 0 and 1, similar to time-series kernel, values are still too large
    D = D/np.max(D)
    # calculate the bandwidth used in Gaussian kernel function
    mdis = 0.5 * bandwidth * np.median(D)    
    # transform the distances into similarity kernel value, better remove the identity, which is too large
    K = np.exp(-(D ** 2)/mdis) # + np.identity(D.shape[0])
    # if truncate the function
    if truncate == True:
        # trancate with the number of neighbors
        knn_index = np.argpartition(- K, kth = truncate_param - 1, axis=1)[:, (truncate_param-1)]
        kth_dist = np.take_along_axis(K, knn_index[:,None], axis = 1)
        mask = (K >= kth_dist).astype(np.int)
        K_trun = K * mask
    else:
        K_trun = None
        
    # make the weight on each row sum up to 1
    return K/np.sum(K, axis = 1, keepdims = True), K_trun/np.sum(K_trun, axis = 1, keepdims = True)

def calc_kernel(X, k = 5, bandwidth = 1, truncate = False, truncate_param = 1):
    """\
    Description:
    ------------
        Calculate the similarity kernel using gene expression data
    Parameter:
    ------------
        X: gene expression data
        k: the number of nearest neighbor
        bandwidth: the bandwidth of gaussian kernel function
        truncate: truncate out small values in Gaussian kernel function or not
    Return:
    ------------
        K: the weighted kernel function, the sum of each row is normalized to 1
        K_trun: the truncated weighted kernel function
    """
    # calculate pairwise distance
    D = squareform(pdist(X))
    for k in range(k, D.shape[0]):
        # calculate the knn graph from pairwise distance
        knn_index = np.argpartition(D, kth = k - 1, axis=1)[:, (k-1)]
        # find the value of the k-th distance
        kth_dist = np.take_along_axis(D, knn_index[:,None], axis = 1)
        # construct KNN graph
        knn = (D <= kth_dist)
        # make the graph symmetric
        knn = ((knn + knn.T) > 0).astype(np.int)
        # knn = knn * knn.T
        # construct nextworkx graph from weighted knn graph, should make sure the G is connected
        G = nx.from_numpy_array(D * knn)
        # make sure that G is undirected
        assert ~nx.is_directed(G)
        # break while loop if connected
        if nx.is_connected(G):
            break
        else:
            k += 1
        print("number of nearest neighbor: " + str(k))

    print("final number of nearest neighbor (make connected): " + str(k))
    # return a matrix of shortest path distances between nodes. Inf if no distances between nodes (should be no Inf, because the graph is connected)
    D = nx.floyd_warshall_numpy(G)
    assert np.max(D) < np.inf
    # scale the distance to between 0 and 1, similar to time-series kernel, values are still too large
    D = D/np.max(D)
    # calculate the bandwidth used in Gaussian kernel function
    mdis = 0.5 * bandwidth * np.median(D)    
    # transform the distances into similarity kernel value, better remove the identity, which is too large
    K = np.exp(-(D ** 2)/mdis) # + np.identity(D.shape[0])
    # if truncate the function
    if truncate == True:
        print(mdis)
        print(np.sqrt(mdis))
        cutoff = np.sqrt(mdis) * truncate_param
        mask = (D < cutoff).astype(np.int)
        K_trun = K * mask
    else:
        K_trun = None
        
    # make the weight on each row sum up to 1
    return K/np.sum(K, axis = 1, keepdims = True), K_trun/np.sum(K_trun, axis = 1, keepdims = True)



def kernel_band(bandwidth, ntimes, truncate=False):
    """\
    Description:
    -------------
        Calculate the similarity kernel using time-series data
    Parameter:
    -------------
        ntimes: number of time points
        bandwidth: the bandwidth of gaussian kernel function, bandwidth decide the shape (width), no matter the length ntimes
        truncate: truncate out small values in Gaussian kernel function or not
    Return:
    -------------
        the weighted kernel function, the sum of each row is normalized to 1
    """
    # scale the t to be between 0 and 1
    t = (np.arange(ntimes)/ntimes).reshape(ntimes, 1)
    # calculate the pairwise-distance between time pointes
    D = np.square(pdist(t))
    # calculate the bandwidth used in Gaussian kernel function
    mdis = 0.5 * bandwidth * np.median(D)
    # calculate the gaussian kernel function
    K = squareform(np.exp(-D/mdis))+np.identity(ntimes)

    # if truncate the function
    if truncate == True:
        cutoff = mdis * 1.5
        mask = (squareform(D) < cutoff).astype(np.int)
        K_trun = K * mask
    return K/np.sum(K, axis=1)[:, None], K_trun/np.sum(K_trun, axis = 1)[:, None]

'''
def calc_diffu_kernel(X, t = 10, k = 10, n_eign = None, bandwidth = 1, truncate = False):
    """\
    Description:
    ------------
        Calculate the similarity kernel using gene expression data, self implemented
    Parameter:
    ------------
        X: gene expression data
        k: the number of nearest neighbor
        bandwidth: the bandwidth of gaussian kernel function
        truncate: truncate out small values in Gaussian kernel function or not
    Return:
    ------------
        K: the weighted kernel function, the sum of each row is normalized to 1
        K_trun: the truncated weighted kernel function
    """
    from scipy.linalg import eigh
    # calculate pairwise distance
    D = squareform(pdist(X))
    # transform the distance into the similarity metric
    # calculate the knn graph from pairwise distance
    knn_index = np.argpartition(D, kth = k - 1, axis=1)[:, (k-1)]
    # find the value of the k-th distance
    kth_dist = np.take_along_axis(D, knn_index[:,None], axis = 1)
    # divide the distance by bandwidth * kth_distance, such that the K value is within the range of (0 (itself), 1/bandwidth (k-th neighbor)]
    K = D/(bandwidth * kth_dist + 1e-6) 
    K = np.exp(-K ** 2)
    # transition probability
    P = K/np.sum(K, axis = 1)[:, None]
    
    # diffuse for t steps, vanilla method, should give the same result as n_eign = None
    Pt = P ** t
    # Pt = -np.log(Pt + 1e-6)
    M = squareform(pdist(Pt))
    
    
    # # following diffusion map, https://towardsdatascience.com/unwrapping-the-swiss-roll-9249301bd6b7
    # # Di^{-1} @ K, make K asymmetric, make it symmetric again
    # D_right = np.diag(np.sum(K, axis = 1) ** 0.5)
    # D_left = np.diag(np.sum(K, axis = 1) ** -0.5)
    # P_symm = np.matmul(D_right, np.matmul(P, D_left))
    # # eigenvalue decomposition, and remove small eigenvalue terms, as these term will vanish when t increase.
    # eigenValues, eigenVectors = eigh(P_symm)
    # idx = eigenValues.argsort()[::-1]
    # eigenValues = eigenValues[idx]
    # eigenVectors = eigenVectors[:,idx]
    # U = np.matmul(D_left, eigenVectors)
    # if n_eign is None:
    #     X_diff = (eigenValues ** t)[None, :n_eign] * U[:,:n_eign]
    # else:
    #     X_diff = (eigenValues ** t)[None, :] * U
    # M = squareform(pdist(X_diff))

    # calculate the bandwidth used in Gaussian kernel function
    M = M/np.max(M)
    mdis = 0.5 * bandwidth * np.median(M)    
    K = np.exp(-(M ** 2)/mdis) # + np.identity(M.shape[0])
    # if truncate the function
    if truncate == True:
        cutoff = mdis * 1.5
        mask = (M < cutoff).astype(np.int)
        K_trun = K * mask
    else:
        K_trun = None

    # make the weight on each row sum up to 1
    return M, K/np.sum(K, axis = 1, keepdims = True), K_trun/np.sum(K_trun, axis = 1, keepdims = True)

        



def calc_diffu_kernel2(X, k = 10, bandwidth = 1, truncate = False):
    """\
    Description:
    ------------
        Calculate the similarity kernel using gene expression data
    Parameter:
    ------------
        X: gene expression data
        k: the number of nearest neighbor
        bandwidth: the bandwidth of gaussian kernel function
        truncate: truncate out small values in Gaussian kernel function or not
    Return:
    ------------
        K: the weighted kernel function, the sum of each row is normalized to 1
        K_trun: the truncated weighted kernel function
    """
    # calculate pairwise distance
    import graphtools as gt
    import scipy
    G = gt.Graph(X, n_pca=None, knn = k, use_pygsp=True)
    # G.diff_op is a diffusion operator, return similarity matrix calculated from diffusion operation
    W, V = scipy.sparse.linalg.eigs(G.diff_op, k=1)
    # Remove first eigenspace
    T_tilde = G.diff_op.toarray() - (V[:,0] @ V[:,0].T)
    
    # Calculate M
    I = np.eye(T_tilde.shape[1])
    M = np.linalg.inv(I - T_tilde) - I
    M = np.real(M)    

    # calculate the bandwidth used in Gaussian kernel function
    mdis = 0.5 * bandwidth * np.median(M)    
    # transform the distances into similarity kernel value
    K = np.exp(-(M ** 2)/mdis) + np.identity(M.shape[0])
    # if truncate the function
    if truncate == True:
        cutoff = mdis * 1.5
        mask = (M < cutoff).astype(np.int)
        K_trun = K * mask
    else:
        K_trun = None
        
    # make the weight on each row sum up to 1
    return K/np.sum(K, axis = 1, keepdims = True), K_trun/np.sum(K_trun, axis = 1, keepdims = True)
'''

