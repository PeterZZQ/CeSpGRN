import sys, os
sys.path.append('./')

import pandas as pd
import numpy as np
from pathlib import Path
from importlib.machinery import SourceFileLoader


def assignDefaultParameterValues(parameterNamePrefixAndDefaultsAll,parameterNamePrefixAndDefaultsGenes, withRules, genelist):
    """
    Set each kinetic parameter to its default value.
    """
    print("Fixing rate parameters to defaults")
    par = dict()
    for parPrefix, parDefault in parameterNamePrefixAndDefaultsAll.items():
        for node in withRules:
            par[parPrefix + node] = parDefault

    for parPrefix, parDefault in parameterNamePrefixAndDefaultsGenes.items():
        for node in withRules:
            if node in genelist:
                par[parPrefix + node] = parDefault
    return par


def getRegulatorsInRule(rule, species, inputs):
    rhs = rule
    rhs = rhs.replace('(',' ')
    rhs = rhs.replace(')',' ')
    tokens = rhs.split(' ')

    allreg = set([t for t in tokens if (t in species or t in inputs)])
    regulatorySpecies = set([t for t in tokens if t in species])
    inputreg = set([t for t in tokens if t in inputs])

    return((allreg, regulatorySpecies, inputreg))


def createRegulatoryTerms(currgene, combinationOfRegulators, regSpecies, settings):

    strengthSeepecified = False

    if settings['modeltype'] == 'hill':
        # Create the hill function terms for each regulator
        hills = []
        for reg in combinationOfRegulators:
            hillThresholdName = 'k_' + reg

            if reg in regSpecies:
                hills.append('(p_'+ reg +'/'+hillThresholdName+')^n_'+ reg)
            else:
                # Note: Only proteins can be regulators
                hills.append('('+ reg +'/'+hillThresholdName+')^n_'+ reg)
        mult = '*'.join(hills)
        return mult
    elif settings['modeltype'] == 'heaviside':
        terms = []
        for reg in combinationOfRegulators:
            terms.append('p_' + reg)
        mult = '*'.join(terms)
    return mult        


def writeModelToFile(grn_num, ModelSpec, varmapper, path = "./"):
    path_to_ode_model = path + "model_"+str(grn_num)+".py"

    with open(path_to_ode_model,'w') as out:
        out.write('#####################################################\n')
        out.write('import numpy as np\n')
        out.write('# This file is created automatically\n')
        out.write('def Model(Y,t,pars):\n')
        out.write('    # Parameters\n')
        par_names = sorted(ModelSpec['pars'].keys())
        for i,p in enumerate(par_names):
            out.write('    ' + p + ' = pars[' + str(i) + ']\n')
        outstr = ''
        out.write('    # Variables\n')
        for i in range(len(varmapper.keys())):
            out.write('    ' + varmapper[i] + ' = Y[' + str(i) + ']\n')
            outstr += 'd' + varmapper[i] + ','
        for i in range(len(varmapper.keys())):
            vdef = ModelSpec['varspecs'][varmapper[i]]
            vdef = vdef.replace('^','**')
            out.write('    d' + varmapper[i] + ' = '+vdef+'\n')

        out.write('    dY = np.array([' + outstr+ '])\n')
        out.write('    return(dY)\n')
        out.write('#####################################################')
    return path_to_ode_model

def noise(x,t):
    c = 10.#4.
    return (c*np.sqrt(abs(x)))

def deltaW(N, m, h,seed=0):
    np.random.seed(seed)
    return np.random.normal(0.0, h, (N, m))

#def eulersde(f,G,y0,tspan,pars,seed=0.,dW=None):
def eulersde(argdict,noise,y0,seed=0,dW=None):
    # From sdeint implementation
    tspan = argdict['tspan']
    N = len(tspan)
    h = (tspan[N-1] - tspan[0])/(N - 1)
    maxtime = tspan[-1]
    # allocate space for result
    d = len(y0)
    y = np.zeros((N+1, d), dtype=type(y0[0]))

    if dW is None:
        # pre-generate Wiener increments (for d independent Wiener processes):
        dW = deltaW(N, d, h, seed=seed)
    y[0] = y0

    ts = len(tspan) // len(argdict['ModelSpecs'].keys()) # set denominator as the number of graphs
    prev_grn_num = -1
    
    for time, p_time in enumerate(tspan):
        grn_num = time // ts
        if grn_num > prev_grn_num:
            model = SourceFileLoader("model", argdict['Model'][grn_num]).load_module()

            allParameters = dict(argdict['ModelSpecs'][grn_num]['pars'])
            parNames = sorted(list(allParameters.keys()))
            pars = [argdict['ModelSpecs'][grn_num]['pars'][k] for k in parNames]

        dWn = dW[time,:]
        y[time+1] = y[time] + model.Model(y[time], p_time, pars)*h + np.multiply(noise(y[time], p_time), dWn)

        indice = np.where(y[time+1] < 0)
        y[time+1][indice] = y[time][indice]

        prev_grn_num = grn_num
    return y


def simulateAndSample(argdict, y0_exp, path = "./"):
    # Retained for debugging
    isStochastic = True

    ## Boolean to check if a simulation is going to a
    ## 0 steady state, with all genes/proteins dying out
    retry = True
    trys = 0
    ## timepoints
    tps = [i for i in range(1,len(argdict['tspan']))]
    ## gene ids
    gid = [i for i,n in argdict['varmapper'].items() if 'x_' in n]
    outPrefix = path + "simulations/"
    while retry:
        argdict['seed'] += 1000
        
        # given parameters and initial expression y0_exp (ngenes,), simulate P of the dimension (ntimes, ngenes)
        P = eulersde(argdict,noise,y0_exp,seed=argdict['seed'])
        # (ngenes, ntimes)
        P = P.T
        retry = False
        # Extract Time points, select subset of the gene expression matrix P given gid and tps
        subset = P[gid,:][:,tps]
        # df of the shape (ngenes, ntimes), name the time points to be E(cell_id)_(timepoint)
        df = pd.DataFrame(subset, index=pd.Index(argdict['genelist']), columns = ['E' + str(argdict['cellid']) +'_' +str(i) for i in tps])
        # write it out
        df.to_csv(outPrefix + 'E' + str(argdict['cellid']) + '.csv')        
        dfmax = df.max()
        
        # if the maximum value of gene expression within one time point is smaller than 0.1 of x_max, then restart
        for col in df.columns:
            colmax = df[col].max()
            if colmax < 0.1*argdict['x_max']:
                retry= True
                break
                            
        trys += 1
        # write to file
        df.to_csv(outPrefix + '/E' + str(argdict['cellid']) + '.csv')
        
        if trys > 1:
            print('try', trys)


def generateInputFiles(BoolDF, withoutRules, grn_num, path = "./"):
    refnet = []
    genes = set(BoolDF['Gene'].values)
    genes = genes.difference(set(withoutRules))
    inputs = withoutRules

    for g in genes:
        row = BoolDF[BoolDF['Gene'] == g]
        rhs = list(row['Rule'].values)[0]
        rule = list(row['Rule'].values)[0]
        rhs = rhs.replace('(',' ')
        rhs = rhs.replace(')',' ')
        tokens = rhs.split(' ')
        if len(withoutRules) == 0:
            inputs = []
            avoidthese = ['and','or', 'not', '']
        else:
            avoidthese = list(withoutRules)
            avoidthese.extend(['and','or', 'not', ''])

        regulators = [t for t in tokens if (t in genes or t in inputs) if t not in avoidthese]
        if 'not' in tokens:
            whereisnot = tokens.index('not')
        else:
            whereisnot = None
        for r in regulators:
            if whereisnot is None:
                ty = '+'
            else:
                if type(whereisnot) is int:
                    whereisnot = [whereisnot]
                
                if tokens.index(r) < whereisnot[0]:
                    ty = '+'
                else:
                    ty = '-'
             # Regulator is Gene1 and Target is Gene2
            refnet.append({'Gene2':g, 'Gene1':r, 'Type':ty})
    refNetDF = pd.DataFrame(refnet)
    refNetDF.drop_duplicates(inplace=True)
    
    filepath = Path(path + 'ground_truth/')
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    refNetDF.to_csv(path + 'ground_truth/refNetwork_'+str(grn_num)+'.csv',sep=',',index=False)

def minmaxnorm(X):
    """Scales the values in X

    :param X: Input list of values to be scaled
    :type X: list
    :returns: 
        - N : list of values scaled between min and max values in input list
    """
    mix = min(X)
    mx = max(X)
    N = [(x-mix)/(mx-mix) for x in X]
    return N

