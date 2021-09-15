# In[0]
import numpy as np 
import networkx as nx


def dyn_GRN(setting = {}):
    # rank-based methods for comparison, e.g. RBO, kendall-tau (sklearn)
    # extend on keller simulator:
    # 1. Negative regulation.
    # 2. Nonlinear change (optional)
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

    Gs = np.zeros((ntimes, ngenes, ngenes))
    # Initialization
    # G0 = np.random.randn((ngenes, ngenes))
    # G0 = np.random.normal(loc = 0, scale = 1, size = (ngenes, ngenes))

    ### "Change" ###
    G0 = np.random.uniform(low = -1, high = 1, size = (ngenes, ngenes))

    # M = np.random.uniform(low = 0, high = 1, size = (ngenes, ngenes))
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
    Gs[0,:,:] = G0

    not_regulated = np.where(np.sum(Gs[0, :, :], axis = 0) == 0)[0]
    # include self-regulation
    for i in not_regulated:
        # Gs[time, i, i] = np.random.uniform(low = 0, high = 2, size = 1)

        ### "Change" ###
        Gs[0, i, i] = np.random.uniform(low = 0, high = 1, size = 1)

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
            # if time != 1:
            #     print(Gt[del_idx])

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
        Gs[time, :, :] = Gt.reshape((ngenes, ngenes))

    # make sure the genes that are not regulated by any genes are self-regulating
        not_regulated = np.where(np.sum(Gs[time, :, :], axis = 0) == 0)[0]
        # include self-regulation
        for i in not_regulated:
            # Gs[time, i, i] = np.random.uniform(low = 0, high = 2, size = 1)

            ### "Change" ###
            Gs[time, i, i] = np.random.uniform(low = 0, high = 1, size = 1) 

    return Gs


def deltaW(N, m, h):
    """\
    Description:
    -------------
        Generate random matrix of the size (N, m), with 0 mean and h standard deviation
    Parameter:
    -------------
        N:
            first dimension
        m:
            second dimension
        h:
            standard deviation
        seed:
            seed  
    """
    return np.random.normal(0.0, h, (N, m))

def noise(x,t):
    c = 10.#4.
    return (c*np.sqrt(abs(x)))


def linear_model(G, xt, alpha, beta):
    """\
    Description:
    ------------
        The gene expression model using GRN, linear model
    
    Parameter:
    ------------
        G: GRN graph
        xt: gene expression vector
        alpha: the transcription rate
        beta: the degradation rate
    """
    return alpha * G @ xt - beta * xt

def eulersde(argdict, alp = 0.1, bet = 0.05, dW=None):
    """\
    Description:
    ------------
        Using Euler method to simulate the gene expression differential equation
    Parameters:
    ------------
        argdict:
            the arguments
        y0: 
            the initial value
        dW:
            the noise term, can be set to 0 by given a (time, gene) zero matrix
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

        # kinetic parameter
        # generation rate
        alpha = alp * np.ones((ngenes,))
        # degradation rate
        beta = bet * np.ones((ngenes,))
        # y[t+1] = y[t] + alpha * G @ y[t] * dt - beta * y[t] * dt + n
        y[time + 1] = y[time] + linear_model(Gs[time, :, :], y[time], alpha = alpha, beta = beta) * dt + np.multiply(noise(y[time], p_time), dWn)

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



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"" Benchmarking
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def kendalltau(pt_pred, pt_true):
    """\
    Description
        kendall tau correlationship
    
    Parameters
    ----------
    pt_pred
        inferred pseudo-time
    pt_true
        ground truth pseudo-time
    Returns
    -------
    tau
        returned score
    """
    from scipy.stats import kendalltau
    pt_true = pt_true.squeeze()
    pt_pred = pt_pred.squeeze()
    tau, p_val = kendalltau(pt_pred, pt_true)
    return tau





# In[1]
# initial setting
def run_simulator(ncells, ngenes=18,ntfs=12,tmax=75, mode = "TF-TF&target", nchanges=10,change_stepsize=1500, alp = 0.1, bet = 0.05, integration_step_size = 0.01):
    # simulation steps != ncells (ncells << simulation steps)


    argdict = dict()
    # range of pseudotime
    argdict["tspan"] = np.linspace(0, tmax, int(tmax/integration_step_size))

    # Generate ground truth GRNs, ntimes, ngenes (TF), ngenes (Target)
    GRNs = dyn_GRN(setting = {"ngenes": ngenes, "ntfs": ntfs, "ntimes": argdict["tspan"].shape[0], \
        "mode": mode, "nchanges": nchanges, "change_stepsize": change_stepsize})
    argdict["GRN"] = GRNs
    
    # # using the same graph
    # argdict["GRN"] = GRNs[0:1,:,:].repeat(GRNs.shape[0], axis = 0)
    # print(argdict["GRN"].shape)
    # print(GRNs.shape)

    # make sure the time span is the same
    assert argdict["tspan"].shape[0] == argdict["GRN"].shape[0]

    ntimes = argdict["GRN"].shape[0]
    ngenes = argdict["GRN"].shape[1]
    # (ncells, ngenes, ntimes)
    Ps = []
    # initial gene expressions
    argdict["init"] = np.random.uniform(low = 0, high = 50, size = ngenes)

    # One cell differentiation process
    count = 0
    for cell in range(ncells):
        retry = True

        while retry:
            # set random seed
            seed = count
            np.random.seed(seed)
            
            # Given parameters and initial expression y0_exp (ngenes,), simulate P of the dimension (ntimes, ngenes)
            P = eulersde(argdict = argdict, alp = alp, bet = bet)
            # (ngenes, ntimes)
            P = P.T
            retry = False
            
            # If the maximum value of gene expression within one time point is smaller than 0.1 of x_max, then restart
            """
            # how to decide x_max
            x_max = np.max(P)
            for time in range(P.shape[1]):
                max_expr = np.max(P[:, time])
                if max_expr < 0.1 * x_max:
                    retry = True
                    print("retry")
                    break
            """
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
    import matplotlib.pyplot as plt
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
# %%
