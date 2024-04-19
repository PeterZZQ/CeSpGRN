# In[0]
import numpy as np
from multiprocessing import Pool, cpu_count

def dyn_GRN_degree(argdict):
    # Keep the indegree and outdegree
    # set parameters for simulator
    _argdict = {"ngenes": 20, # number of genes 
                "ntimes": 1000, # number of time steps
                "mode": "TF-TF&target", # mode of the simulation, `TF-TF&target' or `TF-target'
                "ntfs": 5,  # number of TFs
                "nchanges": 5, # number of changing edges for each interval
                "change_stepsize": 10, # number of stepsizes for each change
                "density": 0.1, # number of edges
                "seed": 0, # random seed
                "backbone": np.array(["0_1"] * 1000), # the backbone branch belonging of each cell, of the form "A_B", where A and B are node ids starting from 0
                "G0": None
                }
    _argdict.update(argdict)

    # set random seed
    np.random.seed(_argdict["seed"])

    # set parameter values
    ngenes, ntfs, ntimes, mode, nchanges, change_stepsize = _argdict["ngenes"], _argdict["ntfs"], \
        _argdict["ntimes"], _argdict["mode"], _argdict["nchanges"], _argdict["change_stepsize"]
    
    Gs = [None] * ntimes
    # initialization: only consider activator & set edge strength: [0,1]
    if _argdict["G0"] is None:
        G0 = np.zeros((ngenes,ngenes))
        nedges = int((ngenes**2)*_argdict["density"])

        if mode ==  "TF-target": 
            # TF is always self-regulated
            for tf_in in range(ntfs):
                G0[tf_in,tf_in] = np.random.uniform(low=0, high=1, size=1)
                
        elif mode ==  "TF-TF&target":
            for tf_in in range(ntfs):
                tf_out = np.random.choice(tf_in+1)
                G0[tf_out,tf_in] = np.random.uniform(low=0, high=1, size=1)
        
        idx = 0
        while len(np.nonzero(G0)[0]) < nedges:
            target = idx%(ngenes-ntfs) + ntfs
            tf = np.random.choice(ntfs)
            G0[tf,target] = np.random.uniform(low=0, high=1, size=1)
            idx += 1

        print("number of non-zero values in the network: {:d}".format(len(np.nonzero(G0)[0])))
        Gs[0] = G0

        # make sure the genes that are not regulated by any genes are self-regulating
        not_regulated = np.where(np.sum(Gs[0], axis = 0) == 0)[0]
        for i in not_regulated:
            if i < ntfs:
                Gs[0][i, i] = np.random.uniform(low = 0, high = 1, size = 1)
            else:
                tf = np.random.choice(ntfs)
                Gs[0][tf, i] = np.random.uniform(low = 0, high = 1, size = 1)

    else:
        Gs[0] = _argdict["G0"]

    # check initial out degree and in degree of the graph
    init_outdeg = np.sum((Gs[0] > 0).astype(np.int), axis = 1)
    init_indeg = np.sum((Gs[0] > 0).astype(np.int), axis = 0) 

    # active (index, 1d) & non-active (-1) area, row is tf and column is target
    if mode == "TF-target":
        # maker sure no interactions for target -> tf, target -> target, tf -> tf
        active_area = np.arange(ngenes**2).reshape(ngenes,ngenes)
        active_area[ntfs:,:] = -1
        active_area[np.triu_indices(ntfs,1)] = -1
        active_area[np.tril_indices(ntfs,-1)] = -1

    elif mode == "TF-TF&target":
        # maker sure no interactions for target -> tf, target -> target
        active_area = np.arange(ngenes**2).reshape(ngenes,ngenes)
        active_area[ntfs:,:] = -1
        active_area[np.tril_indices(ntfs, -1)] = -1

    # backbone, find all possible branches
    branches = list(set(_argdict["backbone"]))
    node_times = {}
    # root node
    node_times["0"] = 0
    for branch in branches:
        try:
            start_node, end_node = branch.split("_")
        except:
            raise ValueError("backbone should be of the form A_B, where A is the start node, and B is the end node")
        
        # find branching time of current branch, end nodes are unique in tree
        branch_times = np.where(_argdict["backbone"] == branch)[0]
        try:
            assert branch_times.shape[0] % change_stepsize == 0
        except:
            raise ValueError("the backbone length should be divided exactly by the stepsize")
        # assign branching time for the end_node
        if end_node not in node_times.keys():
            node_times[end_node] = np.max(branch_times)

    # loop through all branches
    while len(branches) != 0:
        for branch in branches:
            start_node, end_node = branch.split("_")
            if Gs[node_times[start_node]] is not None:
                # remove branch from branches
                branches.remove(branch)
                # use the corresponding branch
                break
        
        branch_times = np.where(_argdict["backbone"] == branch)[0]
        if Gs[node_times[start_node]] is not None:
            # initial graph in the branch, G0 will be updated this way. 
            # This node node_times[start_node] must be divided exactly by change_stepsize.
            pre_G = Gs[node_times[start_node]]
            for i, time in enumerate(branch_times):
                # graph change point
                if i%change_stepsize == 0:
                    # some values are not exactly 0, numerical issue
                    pre_G = np.where(pre_G < 1e-6, 0, pre_G)
                    
                    # make sure the out-degree and in-degree of each nodes are kept
                    outdeg = np.sum((pre_G > 0).astype(np.int), axis = 1)
                    indeg = np.sum((pre_G > 0).astype(np.int), axis = 0) 
                    assert np.all((outdeg - init_outdeg == 0))
                    assert np.all((indeg - init_indeg == 0))

                    Gt = pre_G.reshape(-1)
                    # find the target and TF that is regulated by at least one gene (make sure no isolated target gene is created)
                    rm_target = np.where(indeg > 1)[0]

                    # the interaction that can be removed, including the TF->TF and TF->Target
                    removable = np.arange(ngenes**2).reshape(ngenes,ngenes)
                    rm_matrix = removable[:ntfs,rm_target]
                    removable = set(rm_matrix.reshape(-1))

                    # choose the deleted edges (idx)
                    del_candid = set(np.where(Gt != 0)[0]).intersection(set(active_area.reshape(-1)))
                    del_candid = del_candid.intersection(removable)

                    for idx,max_count in enumerate(indeg[rm_target]):
                        idx_candid = del_candid.intersection(set(rm_matrix[:,idx]))

                        while len(idx_candid) >= max_count:
                            del_candid = del_candid - set([np.random.choice(list(idx_candid))])        
                            idx_candid = del_candid.intersection(set(rm_matrix[:,idx]))

                    add_candid = np.array(list(set(np.where(Gt == 0)[0]).intersection(set(active_area.reshape(-1)))))
                    
                    # find corresponding add_idx
                    while True:                    
                        del_idx = np.random.choice(list(del_candid), nchanges, replace = False)
                        
                        # make sure the add edges does not change the overall node degree
                        # find the corresponding sources of the deleted edges
                        del_row = del_idx//ngenes
                        # make sure the sources are all TFs
                        assert np.all(np.array([x < ntfs for x in del_row]))
                        # find the corresponding ends for the deleted edges, ends can be both targets and TFs.
                        del_col = del_idx%ngenes
                        # set the sources of the add edges to be the sources of the delted edges
                        # for the TFs where we remove edges, we need to add new edges too.
                        add_row = del_row
                        # select the targets for the TFs by permuting the deleted targets
                        # issue: the edges might already exist, the TF-TF regulation might not be in the active area (should be reverse)
                        for i in range(len(del_col)):
                            # in addition, we may have duplicated edges after permutation
                            add_col = np.append(del_col[i+1:], del_col[:i+1])
                            edges = [(x,y) for x,y in zip(add_row, add_col)]
                            # if there are no duplicated edges
                            if len(edges) == len(set(edges)):
                                break
                        # calculate add_idx
                        add_idx = [add_row[x] * ngenes + add_col[x] for x in range(len(add_row))]
                        if set([x for x in add_idx]).issubset(set([x for x in add_candid])):
                            break                  
                    
                    add_value = np.random.uniform(low = 0.1, high = 1, size = nchanges) / change_stepsize
                    del_value = Gt[del_idx] / change_stepsize

                else:
                    Gt = pre_G.reshape(-1)

                # update values
                Gt[add_idx] = Gt[add_idx] + add_value
                Gt[del_idx] = Gt[del_idx] - del_value
                Gs[time] = Gt.reshape((ngenes, ngenes))
                # give the next pre_G
                pre_G = Gs[time]

                # make sure no isolated gene
                not_regulated = np.where(np.sum(Gs[time], axis = 0) == 0)[0]
                assert len(not_regulated) == 0
        else:
            raise ValueError("no branch with initial grn assigned")
    Gs = np.concatenate([G[None, :, :] for G in Gs], axis = 0)
    return Gs

def dyn_GRN(argdict):
    # randomly delete edges, for each delete edges, generate one edges for the curresponding TFs.
    # set parameters for simulator
    _argdict = {"ngenes": 20, # number of genes 
                "ntimes": 1000, # number of time steps
                "mode": "TF-TF&target", # mode of the simulation, `TF-TF&target' or `TF-target'
                "ntfs": 5,  # number of TFs
                "nchanges": 5, # number of changing edges for each interval
                "change_stepsize": 10, # number of stepsizes for each change
                "density": 0.1, # number of edges
                "seed": 0, # random seed
                "backbone": np.array(["0_1"] * 1000), # the backbone branch belonging of each cell, of the form "A_B", where A and B are node ids starting from 0
                "G0": None
                }
    _argdict.update(argdict)
    
    # set random seed
    np.random.seed(_argdict["seed"])

    # set parameter values
    ngenes, ntfs, ntimes, mode, nchanges, change_stepsize = _argdict["ngenes"], _argdict["ntfs"], \
        _argdict["ntimes"], _argdict["mode"], _argdict["nchanges"], _argdict["change_stepsize"]
    
    Gs = [None] * ntimes
    # initialization: only consider activator & set edge strength: [0,1]
    # G0 = np.random.uniform(low = 0, high = 1, size = (ngenes, ngenes))
    if _argdict["G0"] is None:
        G0 = np.zeros((ngenes,ngenes))
        nedges = int((ngenes**2)*_argdict["density"])

        if mode ==  "TF-target": 
            # TF is always self-regulated
            for tf_in in range(ntfs):
                G0[tf_in,tf_in] = np.random.uniform(low=0, high=1, size=1)
                
        elif mode ==  "TF-TF&target":
            for tf_in in range(ntfs):
                tf_out = np.random.choice(tf_in+1)
                G0[tf_out,tf_in] = np.random.uniform(low=0, high=1, size=1)
        
        idx = 0
        while len(np.nonzero(G0)[0]) < nedges:
            target = idx%(ngenes-ntfs) + ntfs
            tf = np.random.choice(ntfs)
            G0[tf,target] = np.random.uniform(low=0, high=1, size=1)
            idx += 1
        
        # ref: LL sparsity: 95%

        # M = (np.abs(G0) > threshold).astype(int)

        # # direction of regulate: (tf, target): tf -> target
        # if mode ==  "TF-target":
        #     # assume the first ntfs are tf. [Top-Right] has non-zero value
        #     M[:ntfs,:ntfs] = 0
        #     M[ntfs:,:] = 0
        # elif mode == "TF-TF&target":
        #     # assume the first ntfs are tf. Upper triangular of [Top-Left] & [Top-Right] has non-zero value
        #     M[ntfs:,:] = 0
        #     M[np.tril_indices(ntfs,-1)] = 0

        # G0 = G0 * M
        print("number of non-zero values in the network: {:d}".format(len(np.nonzero(G0)[0])))
        Gs[0] = G0

        # make sure the genes that are not regulated by any genes are self-regulating
        not_regulated = np.where(np.sum(Gs[0], axis = 0) == 0)[0]
        for i in not_regulated:
            if i < ntfs:
                Gs[0][i, i] = np.random.uniform(low = 0, high = 1, size = 1)
            else:
                tf = np.random.choice(ntfs)
                Gs[0][tf, i] = np.random.uniform(low = 0, high = 1, size = 1)

    else:
        Gs[0] = _argdict["G0"]
    
    # active (index, 1d) & non-active (-1) area
    if mode == "TF-target":
        active_area = np.arange(ngenes**2).reshape(ngenes,ngenes)
        active_area[ntfs:,:] = -1
        active_area[np.triu_indices(ntfs,1)] = -1
        active_area[np.tril_indices(ntfs,-1)] = -1

    elif mode == "TF-TF&target":
        active_area = np.arange(ngenes**2).reshape(ngenes,ngenes)
        active_area[ntfs:,:] = -1
        active_area[np.tril_indices(ntfs, -1)] = -1

    # backbone, find all possible branches
    branches = list(set(_argdict["backbone"]))
    node_times = {}
    # root node
    node_times["0"] = 0
    for branch in branches:
        try:
            start_node, end_node = branch.split("_")
        except:
            raise ValueError("backbone should be of the form A_B, where A is the start node, and B is the end node")
        
        # find branching time of current branch, end nodes are unique in tree
        branch_times = np.where(_argdict["backbone"] == branch)[0]
        # assign branching time for the end_node
        if end_node not in node_times.keys():
            node_times[end_node] = np.max(branch_times)

    while len(branches) != 0:
        # select the first branch within the branches
        for branch in branches:
            start_node, end_node = branch.split("_")
            if Gs[node_times[start_node]] is not None:
                # remove branch from branches
                branches.remove(branch)
                # use the corresponding branch
                break
        
        branch_times = np.where(_argdict["backbone"] == branch)[0]
        if Gs[node_times[start_node]] is not None:
            # initial graph in the branch, G0 will be updated this way.
            pre_G = Gs[node_times[start_node]]
            for i, time in enumerate(branch_times):
                # graph change point
                if i%change_stepsize == 0:
                    # some values are not exactly 0, numerical issue
                    pre_G = np.where(pre_G < 1e-6, 0, pre_G)
                    Gt = pre_G.reshape(-1)

                    # delete, decrease to 0 # avoid isolated gene
                    edge_cnt = np.sum((pre_G>0), axis = 0)
                    rm_target = np.where(edge_cnt > 1)[0]

                    removable = np.arange(ngenes**2).reshape(ngenes,ngenes)
                    rm_matrix = removable[:ntfs,rm_target]
                    removable = set(rm_matrix.reshape(-1))

                    del_candid = set(np.where(Gt != 0)[0]).intersection(set(active_area.reshape(-1)))
                    del_candid = del_candid.intersection(removable)

                    for idx,max_count in enumerate(edge_cnt[rm_target]):
                        idx_candid = del_candid.intersection(set(rm_matrix[:,idx]))

                        while len(idx_candid) >= max_count:
                            del_candid = del_candid - set([np.random.choice(list(idx_candid))])        
                            idx_candid = del_candid.intersection(set(rm_matrix[:,idx]))

                    del_idx = np.random.choice(list(del_candid), nchanges, replace = False)

                    # TODO: to be updated. add, increase to [0,1]
                    add_candid = np.array(list(set(np.where(Gt == 0)[0]).intersection(set(active_area.reshape(-1)))))
                    add_idx = np.random.choice(add_candid, nchanges, replace = False)

                    add_value = np.random.uniform(low = 0, high = 1, size = nchanges) / change_stepsize
                    del_value = Gt[del_idx] / change_stepsize

                else:
                    Gt = pre_G.reshape(-1).copy()

                # update values
                Gt[add_idx] = Gt[add_idx] + add_value
                Gt[del_idx] = Gt[del_idx] - del_value
                Gs[time] = Gt.reshape((ngenes, ngenes)).copy()
                # give the next pre_G
                pre_G = Gs[time].copy()

                # make sure no isolated gene
                not_regulated = np.where(np.sum(Gs[time], axis = 0) == 0)[0]
                assert len(not_regulated) == 0
        else:
            raise ValueError("no branch with initial grn assigned")
    Gs = np.concatenate([G[None, :, :] for G in Gs], axis = 0)
    return Gs

def deltaW(N, m, h):
    """\
    Description:
    -------------
        Generate random matrix of the size (N, m), with 0 mean and h standard deviation
        
    Parameter:
    -------------
        N: first dimension
        m: second dimension
        h: standard deviation
        seed: seed  
    """
    # scale affect the noise level
    scale = 1
    return np.random.normal(0.0, scale * h, (N, m))

def noise(x):
    c = 10.#4.
    return (c*np.sqrt(abs(x)))

def soft_boolODE(G, xt, argdict):
    """\
    Description:
    -------------
        Generate gene expression based on Hill activation function. Do not consider the joint effect when multiple tf bind
        
    Parameter:
    -------------
        G: Ground-truth graph (weighted edges)
        xt: Gene expression
        argdict["k_gene"]: hill threshold
        argdict["n_gene"]: hill coefficient
        argdict["m_gene"]: mRNA transription
        argdict["l_gene"]: mRNA degradation        
    """  
    H = np.array([(exp/argdict["k_gene"][idx])**argdict["n_gene"][idx] for idx, exp in enumerate(xt)])
    dx = np.zeros_like(xt)

    for idx in range(len(G)):
        regul = np.nonzero(G[:,idx])[0]

        H_prod = np.prod(H[regul])
        alpha_1 = np.sum(G[:,idx][regul])/argdict["ntfs"]
        # alpha_1 = np.sum(G[:,idx][regul])/len(regul)
        alpha_0 = 1- alpha_1
        dx[idx] = argdict["m_gene"][idx]*(alpha_0+alpha_1*H_prod)/(1+H_prod) - argdict["l_gene"][idx]*xt[idx]
        
    return dx

def eulersde(argdict):
    """\
    Description:
    ------------
        Using Euler method to simulate the gene expression differential equation
        
    Parameters:
    ------------
        argdict: the arguments
        dW: the noise term, can be set to 0 by given a (time, gene) zero matrix
    """
    _argdict = {"dW": None}
    _argdict.update(argdict)
    # set differnt random seed for different cell
    np.random.seed(_argdict["cell_idx"])
    # print("cell id: " + str(_argdict["cell_idx"]))

    # dt is the time interval = intergration stepsize
    # dt = (_argdict["tspan"][_argdict["ntimes"] - 1] - _argdict["tspan"][0])/(_argdict["ntimes"] - 1)
    dt = _argdict["integration_step_size"]
    # allocate space for gene expression (ntimes, ngenes)
    y = np.zeros((_argdict["ntimes"], _argdict["ngenes"]), dtype=_argdict["init"].dtype)
    # allocate space for simulation time
    sim_time = np.zeros((_argdict["ntimes"],))

    if _argdict["dW"] is None:
        # noise, pre-generate Wiener increments (for d independent Wiener processes), sampled from normal distribution
        dW = deltaW(_argdict["ntimes"], _argdict["ngenes"], dt)
    else:
        dW = deltaW(_argdict["ntimes"], _argdict["ngenes"], _argdict["dW"])
    
    branches = list(set(_argdict["backbone"]))
    node_expr = {}
    node_time = {}
    node_expr["0"] = _argdict["init"]
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
            branch_idx = np.where(_argdict["backbone"] == branch)[0]
            # simulation time for the branch
            pre_expr = node_expr[start_node]
            pre_time = node_time[start_node]
            for idx in branch_idx:
                dWn = dW[idx,:]
                expr = pre_expr + soft_boolODE(_argdict["GRN"][idx, :, :], pre_expr, _argdict) * dt + np.multiply(noise(pre_expr), dWn)
                time = pre_time + dt
                indice = np.where(expr < 0)
                expr[indice] = pre_expr[indice]
                y[idx] = expr
                sim_time[idx] = time
                pre_expr = expr
                pre_time = time
            # the last expression data
            node_expr[end_node] = pre_expr
            node_time[end_node] = pre_time

        else:
            raise ValueError("no initial expression assigned")
    return y.T, sim_time

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"" This part is to add technical noise to dynamics data
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def outlier_effect_dynamics(Expr, outlier_prob, mean, scale):
    """
    This function
    """
    ncells, ngenes = Expr.shape
    out_indicator = np.random.binomial(n = 1, p = outlier_prob, size = ngenes)
    outlierGenesIndx = np.where(out_indicator == 1)[0]
    numOutliers = len(outlierGenesIndx)

    #### generate outlier factors ####
    outFactors = np.random.lognormal(mean = mean, sigma = scale, size = numOutliers)
    ##################################

    Expr = np.concatenate(Expr, axis = 1)
    for i, gIndx in enumerate(outlierGenesIndx):
        Expr[gIndx,:] = Expr[gIndx,:] * outFactors[i]

    return Expr

def lib_size_effect_dynamics(Expr, mean, scale):
    """
    Description:
    -------------
        This functions adjusts the mRNA levels in each cell seperately to mimic
        the library size effect. To adjust mRNA levels, cell-specific factors are sampled
        from a log-normal distribution with given mean and scale.

    Parameters:
    -------------
        Expr: 
            expression count of the size (ngenes, ncells)
        mean: 
            mean for log-normal distribution
        var: 
            var for log-normal distribution

    Returns:
    -------------
        libFactors ( np.array(nBin, nCell) )
        modified single cell data ( np.array(nBin, nGene, nCell) )
    """

    ngenes, ncells = Expr.shape

    libFactors = np.random.lognormal(mean = mean, sigma = scale, size = ncells)

    # calculate the current library size
    normalizFactors = np.sum(Expr, axis = 0)
    # divide libFactors by normalizFactors, calculate the scaling factor for each cell
    scalingFactors = np.true_divide(libFactors, normalizFactors)
    scalingFactors = scalingFactors.reshape(1, ncells)
    scalingFactors = np.repeat(scalingFactors, ngenes, axis = 0)

    Expr = np.multiply(Expr, scalingFactors)

    return libFactors, Expr

def dropout_indicator_dynamics(Expr, shape = 1, percentile = 65):
    """
    Description:
    ------------
        Simulate the dropout effect
    Parameters:
    ------------
        Expr:
            expression count, of the size (ngenes, ncells)
        shape:
            the shape of the logistic function
        percentile:
            the mid-point of logistic functions is set to the given percentile of the input scData
    Return:
    ------------
        binary_ind, further get the observed count by: 
        ``
            Expr_obs = np.multiply(binary_ind, Expr)
        ''
    """
    Expr_log = np.log(np.add(Expr,1))
    log_mid_point = np.percentile(Expr_log, percentile)
    prob_ber = np.true_divide (1, 1 + np.exp( -1*shape * (Expr_log - log_mid_point) ))

    binary_ind = np.random.binomial( n = 1, p = prob_ber)

    return binary_ind

def convert_to_UMIcounts_dynamics (Expr):
    """
    Input: scData can be the output of simulator or any refined version of it
    (e.g. with technical noise)
    """

    return np.random.poisson(Expr)

def run_simulator(**setting):

    # Basic setting
    _setting = {"ncells": 1, # number of cells
                "ntimes": 2000, # time length for euler simulation
                "integration_step_size": 0.01, # stepsize for each euler step
                # parameter for dyn_GRN
                "ngenes": 18, # number of genes 
                "mode": "TF-TF&target", # mode of the simulation, `TF-TF&target' or `TF-target'
                "ntfs": 12,  # number of TFs
                "nchanges": 10, # number of changing edges for each interval
                "change_stepsize": 1500, # number of stepsizes for each change
                "density": 0.1, # number of edges
                "seed": 0, # random seed
                "backbone": np.array(["0_1"] * 2000)
                }
    _setting.update(setting)
    # simulation time
    _setting["tspan"] = np.array([_setting["integration_step_size"] * x for x in range(1, _setting["ntimes"] + 1)])
    
    # generate GRN
    if _setting["keep_degree"]:
        GRNs = dyn_GRN_degree(_setting)
    else:
        GRNs = dyn_GRN(_setting)
    print("GRN generated.")
    _setting["GRN"] = GRNs
    assert GRNs.shape[0] == _setting["ntimes"]
    assert GRNs.shape[1] == _setting["ngenes"]
    
    # initial gene expressions, and kinetic parameters
    _setting["init"] = np.random.uniform(low = 0, high = 50, size = _setting["ngenes"])
    # original: 20, affect the trajectory a lot.
    _setting["m_gene"] = {gene:200. for gene in np.arange(_setting["ngenes"])}
    _setting["l_gene"] = {gene:10. for gene in np.arange(_setting["ngenes"])}
    _setting["k_gene"] = {gene:10. for gene in np.arange(_setting["ngenes"])}
    _setting["n_gene"] = {gene:10. for gene in np.arange(_setting["ngenes"])}

    # simulate gene expression dynamics for cells
    # setting for each cells simulation
    setting_euler = []
    for cell in range(_setting["ncells"]):
        setting_euler.append(_setting.copy())
        setting_euler[-1]["cell_idx"] = cell

    print("conduct experiment")
    # MULTI-THREADING, use machine core count for parallel calculation
    pool = Pool(cpu_count()) 
    Ps, sim_times = zip(*pool.map(eulersde, [x for x in setting_euler]))   
    
    # # SINGLE-THREADING
    # Ps = []
    # sim_times = []
    # for cell in range(_setting["ncells"]):
    #     # euler method (differential equation)
    #     P, sim_time = eulersde(setting_euler[cell])
    #     # obtained gene expression matrix (one experiment)
    #     Ps.append(P)
    #     sim_times.append(sim_time)
    assert sim_times[0].shape[0] == _setting["ntimes"]
    assert Ps[0].shape[1] == _setting["ntimes"]
    assert Ps[0].shape[0] == _setting["ngenes"]
    # Sampling pseudotime for cells, and generate true gene expression data
    # seed is set to original value
    np.random.seed(_setting["seed"])
    pseudotime_index = np.random.choice(np.arange(_setting["ntimes"]), _setting["ncells"], replace = True)
    # pseudotime, (ncells,)
    GRNs = _setting["GRN"][pseudotime_index, :, :]
    pseudotime = sim_times[0][pseudotime_index]
    assert pseudotime.shape[0] == _setting["ncells"]
    # Generate true Expression, (ngenes, ncells)
    Expr = np.concatenate([Ps[cell][:,pseudotime_index[cell]:(pseudotime_index[cell] + 1)] for cell in range(_setting["ncells"])], axis = 1)
    assert Expr.shape[1] == _setting["ncells"]
    # sort pseudotime and true expression data from early to late
    Expr = Expr[:, np.argsort(pseudotime)]
    GRNs = GRNs[np.argsort(pseudotime), :, :]
    pseudotime = pseudotime[np.argsort(pseudotime)]

    # Include technical effect, from Sergio
    # no outlier genes
    # Expr = outlier_effect_dynamics(Expr, outlier_prob = 0.01, mean = 0.8, scale = 1)
    # library size effect
    _, Expr_obs = lib_size_effect_dynamics(Expr, mean = 4.6, scale = 0.4)
    # add dropout
    # binary_ind = dropout_indicator_dynamics(Expr_obs, shape = 6.5, percentile = 82)
    # Expr_obs = np.multiply(binary_ind, Expr_obs)
    # convert to UMI (ngenes, ncells)
    # Expr_obs = convert_to_UMIcounts_dynamics(Expr_obs)
    # ps_Expr_obs = convert_to_UMIcounts_dynamics(ps_Expr_obs)

    results = {"true count": Expr, 
               "observed count": Expr_obs, 
               "pseudotime": pseudotime, 
               "experiment": Ps,
               "GRNs": GRNs
               }

    return results


def load_sub_sergio(grn_init, sub_size, ntfs, seed = 0, init_size = 100, mode = None):
    import csv
    np.random.seed(seed)
    G_init = np.zeros((init_size,init_size))
    with open(grn_init+".txt","r") as f:
        reader = csv.reader(f, delimiter=",")

        for row in reader:
            target = int(float(row[0]))
            n_tf = int(float(row[1]))
            
            # only consider activating regulation
            for tfId, K in zip(row[2: 2 + n_tf], row[2+n_tf : 2+2*n_tf]):
                if float(K) > 0:
                    tf = int(float(tfId))
                    G_init[tf,target] = float(K)

    tf_list = np.unique((np.nonzero(G_init)[0]))
    print("Original SERGIO: TF = {}".format(list(tf_list)))
    G0 = np.zeros((sub_size,sub_size))

    ntfs = ntfs
    sub_tf = np.random.choice(tf_list, ntfs, replace = False)
    sub_tf = np.sort(sub_tf)

    # randomly select sub graph from SERGIO initial graph
    if mode == "random":
        tf_ids = np.nonzero(G_init[sub_tf])[0]
        target_candid = np.nonzero(G_init[sub_tf])[1]

        target_genes = np.random.choice(list(set(target_candid)), sub_size-ntfs, replace = False)
        target_dict = {v:i for i,v in enumerate(np.sort(target_genes))}

        print("Subgraph of SERGIO: TF = {}\tTarget = {}".format(list(sub_tf), list(target_dict.keys())))

        for edge in zip(tf_ids, target_candid):
            tf_id, target = edge
            if target in target_dict.keys():
                G0[tf_id, ntfs+target_dict[target]] = G_init[sub_tf[tf_id],target]
                
    # select the most desnse (large number of edges) sub graph from SERGIO initial graph
    else:
        target_indegree = np.sum(G_init[sub_tf], axis = 0)
        target_genes =  np.argsort(target_indegree)[-(sub_size-ntfs):]
        print("Subgraph of SERGIO: TF = {}\tTarget = {}".format(list(sub_tf), list(np.sort(target_genes))))
        G0[:ntfs,ntfs:] = G_init[sub_tf][:,np.sort(target_genes)]
    
    # tune edge weight between 0 and 1
    max_val = np.max(G0)
    G0 = G0/max_val
            
    for tf in range(ntfs):
        G0[tf,tf] = np.random.uniform(low = 0, high = 1)

    n_edges = len(np.nonzero(G0)[0])
    print("Number of edges: {}, Density: {}".format(n_edges, n_edges/(sub_size**2)))
            
    return G0



# In[0] Check experiments (each experiments should return different results)
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from umap import UMAP
    from sklearn.decomposition import PCA
    sergio_path = "./sergio_data/Interaction_cID_8"
    ngenes = 20
    ntfs = 5
    seed = 0
    G0 = load_sub_sergio(grn_init = sergio_path, sub_size = ngenes, ntfs = ntfs, seed = seed, init_size = 100)

    plt.rcParams["font.size"] = 20
    stepsize = 0.001
    simu_setting = {"ncells": 4, # number of cells
                    "ntimes": 400, # time length for euler simulation
                    "integration_step_size": stepsize, # stepsize for each euler step
                    # parameter for dyn_GRN
                    "ngenes": 20, # number of genes 
                    "mode": "TF-TF&target", # mode of the simulation, `TF-TF&target' or `TF-target'
                    "ntfs": 5,  # number of TFs
                    # nchanges also drive the trajectory, but even if don't change nchanges, there is still linear trajectory
                    "nchanges": 10, # number of changing edges for each interval
                    "change_stepsize": 10, # number of stepsizes for each change
                    "density": 0.1, # number of edges
                    "seed": 0, # random seed
                    "dW": None,
                    # the changing point must be divided exactly by the change_stepsize, or there will be issue.
                    "backbone": np.array(["0_1"] * 40 + ["1_2"] * 180 + ["1_3"] * 180),
                    "keep_degree": False,
                    "G0": G0
                    }
    results = run_simulator(**simu_setting)

    def preprocess(counts):
        """\
        Input:
        counts = (ntimes, ngenes)

        Description:
        ------------
        Preprocess the dataset
        """
        # normalize according to the library size

        libsize = np.median(np.sum(counts, axis=1))
        counts = counts / np.sum(counts, axis=1)[:, None] * libsize
        counts = np.log1p(counts)
        return counts

    fig = plt.figure(figsize = (10, 7))
    ax = fig.add_subplot()
    umap_op = UMAP(n_components = 2)
    pca_op = PCA(n_components = 2)

    for i in range(len(results["experiment"])):
        X = results["experiment"][i].T
        # X = preprocess(X)
        if i == 0:
            X_umap = pca_op.fit_transform(X)
        else:
            X_umap = pca_op.transform(X)

        ax.scatter(X_umap[:, 0], X_umap[:, 1], label = "cell (experiment run) " + str(i), s = 5)
    fig.legend()
    # fig.savefig("experiments_plot.png", bbox_inches = "tight")


# In[1] Check simulation-parameters
if __name__ == "__main__":
    stepsize = 0.0005
    simu_setting = {"ncells": 1, # number of cells
                    "tmax": int(stepsize * 2000), # time length for euler simulation
                    "integration_step_size": stepsize, # stepsize for each euler step
                    # parameter for dyn_GRN
                    "ngenes": 18, # number of genes 
                    "mode": "TF-TF&target", # mode of the simulation, `TF-TF&target' or `TF-target'
                    "ntfs": 12,  # number of TFs

                    # nchanges also drive the trajectory, 
                    # but even if don't change nchanges, there is still linear trajectory, 
                    # which also make sense according to the differential equation
                    "nchanges": 0, # number of changing edges for each interval
                    "change_stepsize": 100, # number of stepsizes for each change
                    "density": 0.1, # number of edges
                    "seed": 0, # random seed

                    # dW and integration_step_size jointly affect the trajectory continuity, 
                    # dW would be too noisy under "integration_step_size": 0.01, 
                    # the test shows that "integration_step_size" around 0.0005~0.002 produce good result.
                    "dW": None
                    }
    results = run_simulator(**simu_setting)
    fig = plt.figure(figsize = (10, 7))
    ax = fig.add_subplot()
    for i in range(len(results["experiment"])):
        X = results["experiment"][i].T
        X = preprocess(X)
        X_umap = pca_op.fit_transform(X)
        ax.scatter(X_umap[:, 0], X_umap[:, 1], label = "changes: " + str(simu_setting["nchanges"]), s = 5, c = np.arange(X_umap.shape[0]))
    fig.legend()
    stepsize = 0.0005
    simu_setting = {"ncells": 1, # number of cells
                    "tmax": int(stepsize * 2000), # time length for euler simulation
                    "integration_step_size": stepsize, # stepsize for each euler step
                    # parameter for dyn_GRN
                    "ngenes": 18, # number of genes 
                    "mode": "TF-TF&target", # mode of the simulation, `TF-TF&target' or `TF-target'
                    "ntfs": 12,  # number of TFs
                    "nchanges": 10, # number of changing edges for each interval
                    "change_stepsize": 100, # number of stepsizes for each change
                    "density": 0.1, # number of edges
                    "seed": 0, # random seed
                    "dW": None
                    }
    results = run_simulator(**simu_setting)
    fig = plt.figure(figsize = (10, 7))
    ax = fig.add_subplot()
    for i in range(len(results["experiment"])):
        X = results["experiment"][i].T
        X = preprocess(X)
        X_umap = pca_op.fit_transform(X)
        ax.scatter(X_umap[:, 0], X_umap[:, 1], label = "changes: " + str(simu_setting["nchanges"]), s = 5, c = np.arange(X_umap.shape[0]))
    fig.legend()

# In[1] Generate simulation data for multiple cells
if __name__ == "__main__":    
    stepsize = 0.0005
    simu_setting = {"ncells": 1000, # number of cells
                    "tmax": int(stepsize * 2000), # time length for euler simulation
                    "integration_step_size": stepsize, # stepsize for each euler step
                    # parameter for dyn_GRN
                    "ngenes": 18, # number of genes 
                    "mode": "TF-TF&target", # mode of the simulation, `TF-TF&target' or `TF-target'
                    "ntfs": 12,  # number of TFs

                    # nchanges also drive the trajectory, if nchanges is larger than 0, then there is trajectory.
                    "nchanges": 2, # number of changing edges for each interval
                    "change_stepsize": 100, # number of stepsizes for each change
                    "density": 0.1, # number of edges
                    "seed": 0, # random seed

                    # dW and integration_step_size jointly affect the trajectory continuity, 
                    # dW would be too noisy under "integration_step_size": 0.01, 
                    # the test shows that "integration_step_size" around 0.0005~0.002 produce good result.
                    "dW": None
                    }
    results = run_simulator(**simu_setting)

    fig = plt.figure(figsize = (10, 7))
    ax = fig.add_subplot()
    umap_op = UMAP(n_components = 2)
    pca_op = PCA(n_components = 2)

    X = results["true count"].T
    pt = results["pseudotime"]
    X = preprocess(X)
    X_umap = pca_op.fit_transform(X)
    ax.scatter(X_umap[:, 0], X_umap[:, 1], s = 5, c = pt)
    fig.savefig("true_count_plot.png", bbox_inches = "tight")

    fig = plt.figure(figsize = (10, 7))
    ax = fig.add_subplot()
    X = results["observed count"].T
    pt = results["pseudotime"]
    X = preprocess(X)
    X_umap = pca_op.fit_transform(X)
    ax.scatter(X_umap[:, 0], X_umap[:, 1], s = 5, c = pt)
    fig.savefig("observed_count_plot.png", bbox_inches = "tight")

    np.save("true_count.npy", results["true count"].T)
    np.save("obs_count.npy", results["observed count"].T)
    np.save("pseudotime.npy", results["pseudotime"])


# In[2] check the grn signal strength
if __name__ == "__main__":
    from scipy.stats import spearmanr, pearsonr
    import seaborn as sns
    stepsize = 0.0005
    simu_setting = {"ncells": 1, # number of cells
                    "tmax": int(stepsize * 2000), # time length for euler simulation
                    "integration_step_size": stepsize, # stepsize for each euler step
                    # parameter for dyn_GRN
                    "ngenes": 18, # number of genes 
                    "mode": "TF-target", # mode of the simulation, `TF-TF&target' or `TF-target'
                    "ntfs": 12,  # number of TFs

                    # nchanges also drive the trajectory, if nchanges is larger than 0, then there is trajectory.
                    "nchanges": 0, # number of changing edges for each interval
                    "change_stepsize": 500, # number of stepsizes for each change
                    "density": 0.1, # number of edges
                    "seed": 0, # random seed

                    # dW and integration_step_size jointly affect the trajectory continuity, 
                    # dW would be too noisy under "integration_step_size": 0.01, 
                    # the test shows that "integration_step_size" around 0.0005~0.002 produce good result.
                    "dW": None
                    }
    results = run_simulator(**simu_setting)

    fig = plt.figure(figsize = (10, 7))
    ax = fig.add_subplot()
    umap_op = UMAP(n_components = 2)
    pca_op = PCA(n_components = 2)

    for i in range(len(results["experiment"])):
        X = results["experiment"][i].T
        X = preprocess(X)
        if i == 0:
            X_umap = pca_op.fit_transform(X)
        else:
            X_umap = pca_op.transform(X)

        ax.scatter(X_umap[:, 0], X_umap[:, 1], label = "changes: " + str(simu_setting["nchanges"]), s = 5, c = np.arange(X_umap.shape[0]))
    fig.legend()

    # clustermap
    X_mean = np.mean(X, axis = 0)
    corr = (X - X_mean).T @ (X - X_mean)
    TF = np.array(['b'] * simu_setting["ngenes"])
    TF[:simu_setting["ntfs"]] = 'r'
    sns.clustermap(corr, row_colors = TF, col_colors = TF, row_cluster = False, col_cluster = False)
    ave_GRN = np.mean(results["GRNs"] + results["GRNs"], axis = 0)
    ave_GRN = ave_GRN + ave_GRN.T
    sns.clustermap(ave_GRN, row_colors = TF, col_colors = TF, row_cluster = False, col_cluster = False)
    score, p_val = pearsonr(corr.reshape(-1), ave_GRN.reshape(-1))
    print("pearson between correlationship and ground truth: {:.4f}".format(score))
# %%
