#In[0]

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import numpy as np
import time
from operator import itemgetter
from multiprocessing import Pool
from scipy.spatial.distance import pdist, squareform

#In[1]

def GENIE3(expr_data,gene_names=None,regulators='all',tree_method='RF',K='sqrt',ntrees=1000,nthreads=1):
    
    '''Computation of tree-based scores for all putative regulatory links.
    
    Parameters
    ----------
    
    expr_data: numpy array
        Array containing gene expression values. Each row corresponds to a condition and each column corresponds to a gene.
        
    gene_names: list of strings, optional
        List of length p, where p is the number of columns in expr_data, containing the names of the genes. The i-th item of gene_names must correspond to the i-th column of expr_data.
        default: None
        
    regulators: list of strings, optional
        List containing the names of the candidate regulators. When a list of regulators is provided, the names of all the genes must be provided (in gene_names). When regulators is set to 'all', any gene can be a candidate regulator.
        default: 'all'
        
    tree-method: 'RF' or 'ET', optional
        Specifies which tree-based procedure is used: either Random Forest ('RF') or Extra-Trees ('ET')
        default: 'RF'
        
    K: 'sqrt', 'all' or a positive integer, optional
        Specifies the number of selected attributes at each node of one tree: either the square root of the number of candidate regulators ('sqrt'), the total number of candidate regulators ('all'), or any positive integer.
        default: 'sqrt'
         
    ntrees: positive integer, optional
        Specifies the number of trees grown in an ensemble.
        default: 1000
    
    nthreads: positive integer, optional
        Number of threads used for parallel computing
        default: 1
        
        
    Returns
    -------
    An array in which the element (i,j) is the score of the edge directed from the i-th gene to the j-th gene. All diagonal elements are set to zero (auto-regulations are not considered). When a list of candidate regulators is provided, the scores of all the edges directed from a gene that is not a candidate regulator are set to zero.
        
    '''

    print('Tree method: {}, K: {}, Number of trees: {} \n'.format(tree_method,K,ntrees))
    ngenes = expr_data.shape[1]

    # Get the indices of the candidate regulators
    if regulators == 'all':
        input_idx = list(np.arange(ngenes))
    else:
        input_idx = [i for i, gene in enumerate(gene_names) if gene in regulators]
    
    # Learn an ensemble of trees for each target gene, and compute scores for candidate regulators
    VIM = np.zeros((ngenes,ngenes))
    
    if nthreads > 1:
        #print('running jobs on %d threads' % nthreads)

        input_data = list()
        for i in range(ngenes):
            input_data.append( [expr_data,i,input_idx,tree_method,K,ntrees] )

        pool = Pool(nthreads)
        alloutput = pool.map(wr_GENIE3_single, input_data)
    
        for (i,vi) in alloutput:
            VIM[i,:] = vi

    else:
        #print('running single threaded jobs')
        for i in range(ngenes):
            if (i+1)%20==0:
                print('Gene %d/%d...' % (i+1,ngenes))
            
            vi = GENIE3_single(expr_data,i,input_idx,tree_method,K,ntrees)
            VIM[i,:] = vi

    return np.transpose(VIM)
        

def wr_GENIE3_single(args):
    return([args[1], GENIE3_single(args[0], args[1], args[2], args[3], args[4], args[5])])


def GENIE3_single(expr_data,output_idx,input_idx,tree_method,K,ntrees):
    ngenes = expr_data.shape[1]
    
    # Expression of target gene
    output = expr_data[:,output_idx]
    
    # Normalize output data
    if np.std(output) == 0:
        output = np.zeros(len(output))
    else:
        output = output / np.std(output)
    
    # Remove target gene from candidate regulators
    input_idx = input_idx[:]
    if output_idx in input_idx:
        input_idx.remove(output_idx)

    expr_data_input = expr_data[:,input_idx]
    
    # Parameter K of the tree-based method
    if (K == 'all') or (isinstance(K,int) and K >= len(input_idx)):
        max_features = "auto"
    else:
        max_features = K
    
    if tree_method == 'RF':
        treeEstimator = RandomForestRegressor(n_estimators=ntrees,max_features=max_features)
    elif tree_method == 'ET':
        treeEstimator = ExtraTreesRegressor(n_estimators=ntrees,max_features=max_features)
        
    # Learn ensemble of trees
    treeEstimator.fit(expr_data_input,output)
    
    # Compute importance scores
    
    importances = [e.tree_.compute_feature_importances(normalize=False) for e in treeEstimator.estimators_]
    importances = np.asarray(importances)
    feature_importances = np.sum(importances,axis=0) / len(treeEstimator)
    
    vi = np.zeros(ngenes)
    vi[input_idx] = feature_importances
       
    return vi