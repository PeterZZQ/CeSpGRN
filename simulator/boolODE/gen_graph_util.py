import numpy as np
import pandas as pd
import csv
import copy
from itertools import combinations
import parsing_fnc as fnc

# Load initial graph - SERGIO
def load_sergio(grn_init):

    raw_activate, raw_inactivate, all_regul = dict(), dict(), list()
    genes = set()

    with open(grn_init+".txt","r") as f:
        reader = csv.reader(f, delimiter=",")

        for row in reader:
            # number of regulator
            nRegs = int(float(row[1])) 
            # target gene id
            target = int(float(row[0]))
            genes.add(target)

            # regId: regulator id, K: regulation strength
            for regId, K in zip(row[2: 2 + nRegs], row[2+nRegs : 2+2*nRegs]):
                regul = int(float(regId))
                genes.add(regul)

                if (len(all_regul) > 0) and (len(np.all(np.array(all_regul)[:,:2] == (target,regul),axis = 1).nonzero()[0]) > 0):
                    continue

                # activating target gene, when the regulation strength is above 0
                # raw_activate: dictionary, keys: target genes, values: list of regulator genes id for the target genes (activate)
                if float(K) > 0:       
                    act_regul = raw_activate.get(target,[])
                    act_regul.append(regul)
                    raw_activate[target] = list(set(act_regul))

                    all_regul.append([target,regul,1])

                # inactivating target gene, when the regulation strength is below 0
                # raw_inactivate: dictionary, keys: target genes, values: list of regulator genes id for the target genes (inactivate)
                elif float(K) < 0:     
                    inact_regul = raw_inactivate.get(target,[])
                    inact_regul.append(regul)
                    raw_inactivate[target] = list(set(inact_regul))

                    all_regul.append([target,regul,0])
                    
    return genes, raw_activate, raw_inactivate



# Load initial graph - Linear Long (LL)
def load_init_graph(df):
    
    genes = [int(k.split('g')[1]) for k in list(df["Gene"])]
    
    activate_dict = {int(gene.split('g')[1]):[] for gene in df["Gene"]}
    inactivate_dict = {int(gene.split('g')[1]):[] for gene in df["Gene"]}
    all_regul = list()

    for row in df.values:

        clean = row[1].replace("(","").replace(")","").replace("or","").replace("g","")
        clean_list = clean.split()

        if "and" in clean_list:   # there are both activator(s) and inactivator(s)
            act_genes = clean_list[:clean_list.index("and")]
            inact_genes = clean_list[clean_list.index("not")+1:]

        elif "not" in clean_list: # there are only inactivator(s)
            act_genes = []
            inact_genes = clean_list[clean_list.index("not")+1:]
        else:                     # there are only activator(s)
            act_genes = clean_list
            inact_genes = []

        act_genes = [int(i) for i in act_genes]
        inact_genes = [int(j) for j in inact_genes]

        target = int(row[0].split('g')[1])

        activate_dict[target].extend([int(i) for i in act_genes])
        inactivate_dict[target].extend([int(j) for j in inact_genes])

    for key in activate_dict.keys():
        for reg in activate_dict[key]:
            all_regul.append([key,reg,1])

    for key in inactivate_dict.keys():
        for reg in inactivate_dict[key]:
            all_regul.append([key,reg,0])

    return genes, all_regul, activate_dict, inactivate_dict


def graph_generator(genes, activate_dict, inactivate_dict):

    df = pd.DataFrame(columns =["Gene","Rule"])
    
    for gene in genes:
        activators = activate_dict.get(gene,[])
        inactivators = inactivate_dict.get(gene,[])

        postive_set =  " or ".join("g"+str(reg) for reg in activators)
        negative_set =  " or ".join("g"+str(reg) for reg in inactivators)

        if (postive_set == "") and (negative_set == ""):
            regul_rule = "( g" + str(gene) + " )" # self_activation

        elif postive_set == "":
            regul_rule = "not ( " + negative_set + " )"

        elif negative_set == "":
            regul_rule = "( " + postive_set + " )"

        else:
            regul_rule = "(( " + postive_set + " ) and not ( " + negative_set + " ))"

        df = df.append({"Gene":"g"+str(gene), "Rule":regul_rule}, ignore_index = True)
    return df



def graph_perturbation_discrete(activate_dict, inactivate_dict, all_regul, num_pert=10):
    
    genes = set(list(activate_dict.keys())).union(set(list(inactivate_dict.keys())))
        
    # size of activator & inactivator = keep activate/inactivate ratio
    active_size = np.sum([len(reg) for reg in list(activate_dict.values())])
    inactive_size = np.sum([len(reg) for reg in list(inactivate_dict.values())])
    
    active_ratio = active_size/(active_size+inactive_size)

    # deletion step
    idx = np.random.randint(len(all_regul),size = num_pert)
    num_gen = len(list(set(idx)))  # constant for stable number of edges
    
    del_regulator = np.array(all_regul)[list(set(idx))]

    for row in del_regulator:
        if row[2] == 1:
            activate_dict[row[0]].remove(row[1])
        else:
            inactivate_dict[row[0]].remove(row[1])

    all_regul = np.delete(all_regul,idx, axis = 0).tolist()

    # generation step
    regula_candidate = list(np.random.choice(list(genes),num_gen))
    target_candidate = list(np.random.choice(list(genes),num_gen))
    
    for (target,regul) in zip(target_candidate,regula_candidate):
        
        if len(np.all(np.array(all_regul)[:,:2] == (target,regul),axis = 1).nonzero()[0]) > 0:
            # replace to different target gene
            new_target = genes - set(np.array(all_regul)[np.array(all_regul)[:,1] == regul][:,0])
            target = np.random.choice(list(new_target),1)[0]
            
        sign = np.random.choice([1,0],1,p=[active_ratio,1-active_ratio])
        
        if sign == 1:
            act_regul = activate_dict.get(target,[])
            act_regul.append(regul)
            activate_dict[target] = list(set(act_regul))

            all_regul.append([target,regul,1])

        else:
            inact_regul = inactivate_dict.get(target,[])
            inact_regul.append(regul)
            inactivate_dict[target] = list(set(inact_regul))

            all_regul.append([target,regul,0])
    
    return activate_dict, inactivate_dict, all_regul


def graph_perturbation_del(activate_dict, inactivate_dict, all_regul, del_num):
    
    # deletion step
    idx = np.random.randint(len(all_regul), size = del_num)
    del_num = len(list(set(idx)))  # constant for stable number of edges
    del_regulator = np.array(all_regul)[list(set(idx))]

    for row in del_regulator:
        if row[2] == 1:
            activate_dict[row[0]].remove(row[1])
        else:
            inactivate_dict[row[0]].remove(row[1])

    all_regul = np.delete(all_regul,idx, axis = 0).tolist()
    
    return activate_dict, inactivate_dict, all_regul


def graph_perturbation_gen(activate_dict, inactivate_dict, all_regul, gen_num):
    
    genes = set(list(activate_dict.keys())).union(set(list(inactivate_dict.keys())))
        
    # size of activator & inactivator = keep activate/inactivate ratio
    active_size = np.sum([len(reg) for reg in list(activate_dict.values())])
    inactive_size = np.sum([len(reg) for reg in list(inactivate_dict.values())])
    
    active_ratio = active_size/(active_size+inactive_size)

    # generation step
    regula_candidate = list(np.random.choice(list(genes),gen_num))
    target_candidate = list(np.random.choice(list(genes),gen_num))
    
    for (target,regul) in zip(target_candidate,regula_candidate):
        
        if len(np.all(np.array(all_regul)[:,:2] == (target,regul),axis = 1).nonzero()[0]) > 0:
            # replace to different target gene
            new_target = genes - set(np.array(all_regul)[np.array(all_regul)[:,1] == regul][:,0])
            target = np.random.choice(list(new_target),1)[0]
            
        sign = np.random.choice([1,0],1,p=[active_ratio,1-active_ratio])
        
        if sign == 1:
            act_regul = activate_dict.get(target,[])
            act_regul.append(regul)
            activate_dict[target] = list(set(act_regul))

            all_regul.append([target,regul,1])

        else:
            inact_regul = inactivate_dict.get(target,[])
            inact_regul.append(regul)
            inactivate_dict[target] = list(set(inact_regul))

            all_regul.append([target,regul,0])
    
    return activate_dict, inactivate_dict, all_regul

def model_generate(df, settings, withRules, inputs, par, genelist, proteinlist, varspecs):
    boolodespace = {}
    par_each = copy.deepcopy(par)
    
    for node in withRules:
        # Initialize species to 0
        tempStr = node + " = 0"  
        exec(tempStr, boolodespace)

    # Basal expression: Execute the rule to figure out the value of alpha_0 or omega_0
    for i,row in df.iterrows():
        exec('booleval = ' + row['Rule'], boolodespace)
        if settings['modeltype'] == 'hill':
            par_each['alpha_'+row['Gene']] = int(boolodespace['booleval'])
        elif settings['modeltype'] == 'heaviside':
            par_each['omega_' + row['Gene']] = (1 if int(boolodespace['booleval']) == 1 else -1)

    for i,row in df.iterrows():
        ## Parse Boolean rule to get list of regulators
        allreg, regSpecies, regInputs = fnc.getRegulatorsInRule(row['Rule'],withRules,inputs)

        # Basal expression term
        currgene = row['Gene']
        if settings['modeltype'] == 'hill':
            num = '( alpha_' + currgene
            den = '( 1'
        elif settings['modeltype'] == 'heaviside':
            exponent = '- sigmaH_' + currgene +'*( omega_' + currgene

        # Loop over combinations of regulators        
        for i in range(1,len(allreg) + 1):
            for combinationOfRegulators in combinations(allreg,i):
                regulatorExpression = fnc.createRegulatoryTerms(currgene, combinationOfRegulators, regSpecies, settings)

                if settings['modeltype'] == 'hill':
                    # Create Numerator and Denominator
                    den += ' +' +  regulatorExpression
                    num += ' + a_' + currgene +'_'  + '_'.join(list(combinationOfRegulators)) + '*' + regulatorExpression
                elif settings['modeltype'] == 'heaviside':
                    exponent += ' + w_' + currgene + '_' + '_'.join(list(combinationOfRegulators)) +'*' + regulatorExpression

                # evaluate rule to assign values to parameters
                ##################################################
                for node in withRules:                 #
                    exec(node + ' = 0', boolodespace)  #                        
                # Set each regulator to ON, evaluate rule. we are looping over all such combinations of regulators
                for geneInList in combinationOfRegulators:     #
                    exec(geneInList + ' = 1', boolodespace)    #
                                                               #
                exec('boolval = ' + row['Rule'], boolodespace) #
                ##################################################

                if settings['modeltype'] == 'hill':
                    par_each['a_' + currgene +'_'  + '_'.join(list(combinationOfRegulators))] = int(boolodespace['boolval'])
                elif settings['modeltype'] == 'heaviside':
                    par_each['w_' + currgene +'_'  + '_'.join(list(combinationOfRegulators))] = \
                        kineticParameterDefaults['heavisideOmega']*(1 if int(boolodespace['boolval']) == 1 else -1)          

        # Close expressions
        if settings['modeltype'] == 'hill':
            num += ' )'
            den += ' )'
            f = '(' + num + '/' + den + ')'
        elif settings['modeltype'] == 'heaviside':
            # In the case of heaviside expressions, to prevent
            # numerical blowup, we trucate the magnitude of the
            # regulatory terms
            exponent += ')'
            maxexp = '10.' # '100'
            f = '(1./(1. + np.exp(np.sign('+exponent+')*min(' +maxexp +',abs(' + exponent+ ')))))'

        if currgene in proteinlist:
            Production =  f
            Degradation = 'p_' + currgene
            varspecs['p_' + currgene] = 'signalingtimescale*(y_max*' + Production \
                                       + '-' + Degradation + ')'
        else:
            Production = 'm_'+ currgene + '*' + f
            Degradation = 'l_x_'  + currgene + '*x_' + currgene
            varspecs['x_' + currgene] =  Production + '-' + Degradation
            # Create the corresponding translated protein equation
            varspecs['p_' + currgene] = 'r_'+currgene+'*'+'x_' +currgene + '- l_p_'+currgene+'*'+'p_' + currgene

    ##########################################################
    # Initialize variables between 0 and 1, Doesn't matter.
    xvals = [1. for _ in range(len(genelist))]
    pvals = [20. for _ in range(len(proteinlist))]    
    ics = {}

    for node, xv in zip(genelist, xvals):
        ics['x_' + node] = xv
        ics['p_' + node] = 0
    for node, pv in zip(proteinlist, pvals):
        ics['p_' + node] = pv

    ModelSpec = dict()
    varmapper = dict()
    parmapper = dict()

    ModelSpec['varspecs'] = varspecs
    ModelSpec['pars'] = par_each
    ModelSpec['ics'] = ics

    varmapper = {i:var for i,var in enumerate(ModelSpec['varspecs'].keys())}
    parmapper = {i:par for i,par in enumerate(ModelSpec['pars'].keys())}

    return ModelSpec, varmapper, parmapper