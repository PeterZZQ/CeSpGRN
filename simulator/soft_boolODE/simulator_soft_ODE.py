import numpy as np

def dyn_GRN(setting = {}):
    # set parameters
    _setting = {"ngenes": 20, "ntimes": 1000, "mode": "TF-TF&target", "ntfs": 5, "nchanges": 5, "change_stepsize": 10, "density": 0.1, \
        "connected_acyclic": False, "seed": 0}
    _setting.update(setting)
    
    # number of genes, number of TFs, number of time steps
    ngenes, ntfs, ntimes = _setting["ngenes"], _setting["ntfs"], _setting["ntimes"]
    # mode, include: "TF-target" (only edges between TFs and targets), "TF-TF&target" (edges between TFs or edges between TFs and targets)
    mode = _setting["mode"]
    # number of change, change interval
    nchanges, change_stepsize = _setting["nchanges"], _setting["change_stepsize"]
    connected_acyclic = _setting["connected_acyclic"]
    np.random.seed(_setting["seed"])
    
    Gs = np.zeros((ntimes, ngenes, ngenes))

    # initialization: only consider activator & set edge strength: [0,1]
    # G0 = np.random.uniform(low = 0, high = 1, size = (ngenes, ngenes))
    G0 = np.zeros((ngenes,ngenes))
    nedges = int((ngenes**2)*_setting["density"])

    tf_list = np.arange(ntfs)
    target_list = np.arange(ngenes)

    if mode ==  "TF-target": # TF is always self-regulated
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
    print(len(np.nonzero(G0)[0]))
    Gs[0,:,:] = G0

    # make sure the genes that are not regulated by any genes are self-regulating
    not_regulated = np.where(np.sum(Gs[0, :, :], axis = 0) == 0)[0]
    for i in not_regulated:
        if i < ntfs:
            Gs[0, i, i] = np.random.uniform(low = 0, high = 1, size = 1)
        else:
            tf = np.random.choice(ntfs)
            Gs[0, tf, i] = np.random.uniform(low = 0, high = 1, size = 1)

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

    # graph changes
    for time in range(1, ntimes):
        # graph change point
        if (time - 1)%change_stepsize == 0:

            # some values are not exactly 0, numerical issue
            Gs[time - 1, :, :] = np.where(Gs[time - 1, :, :] < 1e-6, 0, Gs[time - 1, :, :])
            Gt = Gs[time - 1, :, :].reshape(-1)

            # delete, decrease to 0 # avoid isolated gene
            edge_cnt = np.sum((Gs[time-1, :, :]>0), axis = 0)
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

            # add, increase to [0,1]
            add_candid = np.array(list(set(np.where(Gt == 0)[0]).intersection(set(active_area.reshape(-1)))))
            add_idx = np.random.choice(add_candid, nchanges, replace = False)

            add_value = np.random.uniform(low = 0, high = 1, size = nchanges) / change_stepsize
            del_value = Gt[del_idx] / change_stepsize

        else:
            Gt = Gs[time - 1, :, :].reshape(-1)

        # update values
        Gt[add_idx] = Gt[add_idx] + add_value
        Gt[del_idx] = Gt[del_idx] - del_value
        Gs[time, :, :] = Gt.reshape((ngenes, ngenes))

        # make sure no isolated gene
        not_regulated = np.where(np.sum(Gs[time, :, :], axis = 0) == 0)[0]
        assert len(not_regulated) == 0
        
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
    return np.random.normal(0.0, h, (N, m))

def noise(x,t):
    c = 10.#4.
    return (c*np.sqrt(abs(x)))

def soft_boolODE(G, xt, argdict):
    """\
    Description:
    -------------
        Generate gene expression based on Hill activation function. Only consider activator with "or" condition
        
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
        alpha_1 = np.sum(G[:,idx][regul])/argdict["tf_size"]
#         alpha_1 = np.sum(G[:,idx][regul])/len(regul)
        alpha_0 = 1- alpha_1
        dx[idx] = argdict["m_gene"][idx]*(alpha_0+alpha_1*H_prod)/(1+H_prod) - argdict["l_gene"][idx]*xt[idx]
        
    return dx

def eulersde(argdict, dW=None):
    """\
    Description:
    ------------
        Using Euler method to simulate the gene expression differential equation
        
    Parameters:
    ------------
        argdict: the arguments
        y0: the initial value
        dW: the noise term, can be set to 0 by given a (time, gene) zero matrix
    """
    # span of pseudotime [0, step, ..., tmax]
    tspan = argdict['tspan']
    ntimes = len(tspan)
    # dt is the time interval, the same as intergration stepsize
    dt = (tspan[ntimes - 1] - tspan[0])/(ntimes - 1)
    # ground truth GRN, of the shape (ntimes, ngenes, ngenes)
    Gs = argdict["GRN"]
    y0 = argdict["init"]
    ngenes = len(y0)

    # allocate space for result, (ntimes, ngenes)
    y = np.zeros((ntimes + 1, ngenes), dtype=type(y0[0]))
    y[0] = argdict["init"]

    if dW is None:
        # pre-generate Wiener increments (for d independent Wiener processes):
        dW = deltaW(ntimes, ngenes, dt)
        
    # simulation process
    for time, p_time in enumerate(tspan):
        # iterate through all steps
        dWn = dW[time,:]
        
        y[time + 1] = y[time] + soft_boolODE(Gs[time, :, :], y[time], argdict) * dt + np.multiply(noise(y[time], p_time), dWn)

        # make sure y is always above 0
        indice = np.where(y[time + 1] < 0)
        y[time + 1][indice] = y[time][indice]

    # drop the first data point
    y = y[1:, :]

    return y

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

def run_simulator(ncells=1, ngenes=18,ntfs=12,tmax=75, mode = "TF-TF&target", nchanges=10, change_stepsize=1500,  density = 0.1, integration_step_size = 0.01):

    argdict = dict()
    # range of pseudotime
    argdict["tspan"] = np.linspace(0, tmax, int(tmax/integration_step_size))

    # Generate ground truth GRNs, ntimes, ngenes (TF), ngenes (Target)
    GRNs = dyn_GRN(setting = {"ngenes": ngenes, "ntfs": ntfs, "ntimes": argdict["tspan"].shape[0], \
        "mode": mode, "nchanges": nchanges, "change_stepsize": change_stepsize, "density": density})
    argdict["GRN"] = GRNs
    argdict["tf_size"] = ntfs
    
    # make sure the time span is the same
    assert argdict["tspan"].shape[0] == argdict["GRN"].shape[0]

    (ntimes, ngenes, _) = argdict["GRN"].shape

    # (ncells, ngenes, ntimes)
    Ps = []
    
    # initial gene expressions
    argdict["init"] = np.random.uniform(low = 0, high = 50, size = ngenes)
    
    # kinetic hyper-parameter
    argdict["m_gene"] = {gene:20. for gene in np.arange(ngenes)}
    argdict["l_gene"] = {gene:10. for gene in np.arange(ngenes)}
    argdict["k_gene"] = {gene:10. for gene in np.arange(ngenes)}
    argdict["n_gene"] = {gene:10. for gene in np.arange(ngenes)}

    # One cell differentiation process
    count = 0
    for cell in range(ncells):
        # set random seed
        seed = count
        np.random.seed(seed)

        # Given parameters and initial expression y0_exp (ngenes,), simulate P of the dimension (ntimes, ngenes)
        P = eulersde(argdict)
        # (ngenes, ntimes)
        P = P.T
        count += 1
            
        Ps.append(P)

    # Pseudotime, ntimes should be larger than ncells (ncells,)
    pseudotime_index = np.random.choice(np.arange(ntimes), ncells, replace = False)
    pseudotime = argdict["tspan"][pseudotime_index]
    assert pseudotime.shape[0] == ncells

    # True Expression, (ngenes, ncells)
    Expr = np.concatenate([Ps[cell][:,pseudotime_index[cell]:(pseudotime_index[cell] + 1)] for cell in range(ncells)], axis = 1)
    assert Expr.shape[1] == ncells

    # Include technical effect, from Sergio
    # no outlier genes
    # Expr = outlier_effect_dynamics(Expr, outlier_prob = 0.01, mean = 0.8, scale = 1)

    # library size effect
    # libFactor, Expr_obs = lib_size_effect_dynamics(Expr, mean = 4.6, scale = 0.4)
    libFactor, ps_Expr_obs = lib_size_effect_dynamics(Ps[0], mean = 4.6, scale = 0.4)

    # add dropout
    # binary_ind = dropout_indicator_dynamics(Expr_obs, shape = 6.5, percentile = 82)
    # Expr_obs = np.multiply(binary_ind, Expr_obs)

    # convert to UMI (ngenes, ncells)
    # Expr_obs = convert_to_UMIcounts_dynamics(Expr_obs)
    # ps_Expr_obs = convert_to_UMIcounts_dynamics(ps_Expr_obs)

    pseudotime = pseudotime[np.argsort(pseudotime)]
    Expr = Expr[:, np.argsort(pseudotime)]
    # Expr_obs = Expr_obs[:, np.argsort(pseudotime)]


    # In[2] Sanity check
    # import matplotlib.pyplot as plt
    # raw count, alpha and beta should be adjusted in case of value eplosion
    # _ = plt.hist(np.sum(Expr, axis = 0), bins = 20)
    # transformed count
    # _ = plt.hist(np.sum(Expr_obs, axis = 0), bins = 20)

    # In[3]
    # np.save("GT_graphs.npy",argdict["GRN"])
    # np.save("Single_Exp.npy", ps_Expr_obs)

    # np.save("Expr_true.npy", Expr)
    # np.save("Expr_obs.npy", Expr_obs)
    # np.save("pseudotime.npy", pseudotime)

    return ps_Expr_obs, argdict["GRN"]