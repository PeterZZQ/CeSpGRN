# Creating markov networks given input Sigma
# set tabstop=4
import numpy as np
import pandas as pd
import scipy
import networkx as nx
print(nx.__version__)
#if nx.__version__ != '1.11':
#    print('Wrong NETWORKx version!!! Will lead to wrong results')
import copy
import subprocess
from os import listdir
from os.path import isfile, join
import os
import matplotlib.pyplot as plt

def isPSD(A, tol=1e-7):
    E,V = scipy.linalg.eigh(A)
    print('min_eig = ', np.min(E) , 'max_eig = ', np.max(E), ' min_diag = ', np.min(np.diag(A)))
    return np.all(E > -tol)

def check_sym(a, tol=1e-7):
    print('is PSD: ', isPSD(a))
    return np.allclose(a, a.T, atol=tol)


class create_MN_vary_w(object):
    def __init__(self, k_train, m, n, graph_type, batches, w, k_test, RAND_details, k_valid=None): 
        self.k_train = k_train # number of graph for training
        self.k_test = k_test # number of graph for testing
        self.m = m # number of samples
        self.n = n # number of features
        print('graph details: ',graph_type, ' random graph details:', RAND_details)
        self.prob, self.MAX_DEG, self.SIGNS = RAND_details
        self.batches = batches 
        self.w_min, self.w_max = w
        self.train_graphs = {}
        self.test_graphs = {}
        self.collect_w = []
        for k in range(self.k_train):# [graph, cov, data]
            precision_mat, cov = self.init_random_graph(graph_type)
            for b in range(batches):
                data = self.get_samples(cov)
                self.train_graphs[k*batches+b] = [precision_mat, data]
        print('Total valid graphs for edge recovery = ', k*batches+b)
        
        if k_valid!=None:
            self.k_valid = k_valid
            self.valid_graphs = {}
            for k in range(self.k_valid):# [graph, cov, data]
                precision_mat, cov = self.init_random_graph(graph_type)
                for b in range(batches):
                    data = self.get_samples(cov)
                    self.valid_graphs[k*batches+b] = [precision_mat, data]
            print('Total valid graphs for edge recovery = ', k*batches+b)

        for k in range(self.k_test):# [graph, cov, data]
            #np.random.seed(k*batches+b)
            print('k = ', k)
            precision_mat, cov = self.init_random_graph(graph_type, seed=k)
            for b in range(batches):
                data = self.get_samples(cov, seed=k*batches+b)
                self.test_graphs[k*batches+b] = [precision_mat, data]
        print('Total test graphs for edge recovery = ', k*batches+b)
        print('different w_values used: ', self.collect_w)

    def init_random_graph(self, graph_type, seed=None,u=1): # init random adjacency matrix with probability as p
        # create a graph of given type\
#        true_graph = nx.generators.random_graphs.gnp_random_graph(self.n, prob, seed=seed, directed=False)
        if graph_type == 'grid':
            s = int(np.sqrt(self.n))
            G = nx.generators.lattice.grid_2d_graph(s, s)
        elif graph_type == 'chain':
            G = nx.generators.classic.path_graph(self.n)
        elif graph_type == 'random_maxd':# random graph with max degree d
            G = self.get_rand_graph_maxd(self.prob, seed=seed)

        theta = nx.adjacency_matrix(G).todense() # adjacency matrix
#        print('Precision matrix size is DxD with D = ', len(theta))
        if seed != None:
            # NOTE: seeding the randomness
            np.random.seed(seed)
        if graph_type == 'random_maxd':
            
            # create a random matrix between [self.w_min, self.w_max]
            U = np.matrix(np.random.random((self.n, self.n)) * (self.w_max - self.w_min) + self.w_min)
            #print('err: ', theta, U)
            if self.SIGNS == 1:     
                print('creating a matrix of random +1/-1 ')
                if seed != None:
                    np.random.seed(seed)
                sign_matrix = np.random.choice([-1, 1], size=(self.n, self.n))
                U = U * sign_matrix
 
            theta = np.multiply(theta, U)
            # making it symmetric
            theta = (theta + theta.T)/2
            """ 
            # add a multiple of minimum eigenvalue to ensure PSD
            theta = theta * w_val
            """
            smallest_eigval = np.min(np.linalg.eigvals(theta))
            precision_mat = theta + np.eye(self.n)*(np.abs(smallest_eigval)+ u)# + 0.1 + u)
        else:
            w_val = np.random.uniform(self.w_min, self.w_max)
            precision_mat = w_val * theta + np.eye(self.n) #*(np.abs(smallest_eigval)+ u)# + 0.1 + u)
            self.collect_w.append(w_val)
        cov = np.linalg.inv(precision_mat) # avoiding the use of pinv as data not true representative of conditional independencies.
            

        if isPSD(precision_mat) == 'False':
            print('NOT A PSD matrix!!! CHECK!!!')
            print('CHECK whether cov is sym: ', check_sym(cov), ' w_val = ', w_val)
            print('CHECK whether precision is sym : ', check_sym(precision_mat), w_val)
        return precision_mat, cov

    # procedure of creating synthetic datasets in ISTA for glasso 2012 paper
    def get_samples(self, cov, seed=None):
        if seed != None:
            # NOTE: seeding the randomness
            np.random.seed(seed)
        data = np.random.multivariate_normal(mean=np.zeros(self.n), cov=cov, size=self.m)#.T
        return data# mxn
  
    # get random graph with max degree d
    def get_rand_graph_maxd(self, prob, seed=None):
        # create a erdos-renyi graph
        true_graph = nx.generators.random_graphs.gnp_random_graph(self.n, prob, seed=seed, directed=False)
        while(True):
            max_deg = max([d[1] for d in list(true_graph.degree())])
            if (max_deg < self.MAX_DEG):
                print('max deg = ', max_deg, ' graph generated')
                break
            else:
                print('max deg = ', max_deg , ' removing edges...')
                br
        return true_graph
# get a random graph with prob 
        



class create_MN_random(object):
    def __init__(self, k_train, m, n, graph_type, batches, prob, k_test, k_valid=None):
        self.k_train = k_train # number of graph for training
        self.k_test = k_test # number of graph for testing
        self.m = m # number of samples
        self.n = n # number of features
        self.batches = batches
        self.p_min, self.p_max = prob
        self.train_graphs = {}
        self.test_graphs = {}
        self.collect_p = []
        for k in range(self.k_train):# [graph, cov, data]
            precision_mat, G = self.init_random_graph_gene()
            for b in range(batches):
                data = self.get_samples_gene(G)
                self.train_graphs[k*batches+b] = [precision_mat, data]
        print('Total valid graphs for edge recovery = ', k*batches+b)

        if k_valid!=None:
            self.k_valid = k_valid
            self.valid_graphs = {}
            for k in range(self.k_valid):# [graph, cov, data]
                precision_mat, G = self.init_random_graph_gene()
                for b in range(batches):
                    data = self.get_samples_gene(G)
                    self.valid_graphs[k*batches+b] = [precision_mat, data]
            print('Total valid graphs for edge recovery = ', k*batches+b)

        for k in range(self.k_test):# [graph, cov, data]
            #np.random.seed(k*batches+b)
            print('k = ', k)
            precision_mat, G = self.init_random_graph_gene(seed=k)
            for b in range(batches):
                data = self.get_samples_gene(G, seed=k*batches+b)
                self.test_graphs[k*batches+b] = [precision_mat, data]
        print('Total test graphs for edge recovery = ', k*batches+b)
        print('different w_values used: ', self.collect_p)

    # procedure of getting samples from gene generator
    def init_random_graph_gene(self, u=1, w_val=0.5, seed=None):
        if seed != None:
            np.random.seed(seed)
        prob = np.random.uniform(self.p_min, self.p_max) # defining the graph sparsityi
        self.collect_p.append(prob)
        G = nx.generators.random_graphs.gnp_random_graph(self.n, prob, seed=seed, directed=False)
        
        # graph created, now get the edges 
        G = make_graph_connected(G)
        # Covariance matrix should be positive semi-definite & inverse of precision matrix
        theta = nx.adjacency_matrix(G).todense() # adjacency matrix
#        print('check sym: ', check_sym(theta), theta)
        # theta is currently all ones and symmetric
        if seed != None:
            np.random.seed(seed)
        w_val = np.random.uniform(0.1, 0.2)
        theta = theta * w_val + np.eye(self.n)
        smallest_eigval = np.min(np.linalg.eigvals(theta))
#        print('smallest eval before I: ', smallest_eigval)
        #precision_mat = theta + np.eye(self.n)*(np.abs(smallest_eigval)+ u)# + 0.1 + u)
        precision_mat = theta + np.eye(self.n)*(u - smallest_eigval)# + 0.1 + u)
#        print('check smallest eval = %.3f '%(np.min(np.linalg.eigvals(precision_mat))), ' sparsity = %.3f' %(1-np.count_nonzero(precision_mat==0)/(self.n*self.n)), ' condition number = %.3f ' %(np.linalg.cond(precision_mat)))
#        print('adjacency matrix:', theta, np.sum(theta), 'prob = ', prob)
#        print('precision matrix:', precision_mat)
#        cov = np.linalg.inv(precision_mat) # avoiding the use of pinv as data not true representative of conditional independencies.
        if isPSD(precision_mat) == 'False':
            print('NOT A PSD matrix!!! CHECK!!!')
            print('CHECK whether cov is sym: ', check_sym(cov), ' w_val = ', w_val)
            print('CHECK whether precision is sym : ', check_sym(precision_mat), w_val)
        #data = np.random.multivariate_normal(mean=np.zeros(self.n), cov=cov, size=self.m)#.T
        return precision_mat, G# mxn
#        return cov, data# mxn

    # procedure of creating synthetic datasets from syntren 
    def get_samples_gene(self, G, seed=None, u_min=0.01, u_max=0.1):
        if seed != None:
            # NOTE: seeding the randomness
            np.random.seed(seed)
        noise_level = np.random.uniform(u_min, u_max)
        # Save the edges of gene to an ini file
        # saving the sif file
        sif_filepath = 'expts_gene/syntren/data/myNMSEnetworks/'
        sif_filepath_short = './data/myNMSEnetworks/'
        sif_filename = 'nmse_expts.sif'
        df = pd.DataFrame(list(G.edges()))
        dummy_column = ['du']*len(df) 
        df.insert(loc=1, column='dummy', value=dummy_column)
#        print('saving the ini file')
        df.to_csv(sif_filepath+sif_filename, sep='\t', index=False, header=None)
        # sif files saved
        # run that INI files by calling a separate script
        # calling the syntren generator on the saved sif file
        sample_filepath = 'expts_gene/syntren/data/nmse_samples/'
        sample_filepath_short = './data/nmse_samples/'
#        print('Editing the number of samples in the ini file')
        ini_filepath = 'expts_gene/syntren/data/samples/'
        ini_filename = 'sampleIniFile.ini'
        with open(ini_filepath + ini_filename) as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
#            print(i, line)
            if 'nrExperiments =' in line:
#                print('modify samples', lines[i], ' by M= ', str(self.m))
                lines[i] = 'nrExperiments = '+ str(self.m)+'\n'
            if 'NetworkSIFFile =' in line:
#                print('modify sif filepath', lines[i], ' by path = ', sif_filepath,  sif_filename)
                #lines[i] = 'NetworkSIFFile = '+sif_filepath+sif_filename+'\n'
                lines[i] = 'NetworkSIFFile = '+sif_filepath_short+sif_filename+'\n'
                # will call the new sif file: set of edges
            if 'outputdir =' in line:
#                print('modify samples filepath', lines[i], ' by path = ', sample_filepath)
                lines[i] = 'outputdir = '+sample_filepath_short
            for noises in ['correlationNoise = ', 'bioNoise = ', 'inputNoise = ', 'expNoise = ']:
                if noises in line:
#                    print('modify samples filepath', lines[i], ' by path = ', sample_filepath)
                    lines[i] = noises+str(noise_level)+'\n'

#        print('Updated ini file: ', lines)

        new_ini_filepath = 'expts_gene/syntren/data/nmse/'
        new_ini_filename = 'nmse_expts.ini'
        with open(new_ini_filepath+new_ini_filename, 'w') as f:
            for item in lines:
                f.write("%s"%item)
#        new_ini_file.write(lines)
#        new_ini_file.close()

        print('script to check the call to the sample INI file')
        #bashCommand = 'bash syntren_cli.sh data/samples/sampleIniFile.ini'
#        print('dir path: ', os.path.dirname(os.path.realpath(__file__)))
        
        os.chdir('expts_gene/syntren/')
#        print('changed dir path: ', os.path.dirname(os.path.realpath(__file__)))
        
        bashCommand = 'bash syntren_cli.sh '+' ./data/nmse/'+new_ini_filename
        #bashCommand = 'bash expts_gene/syntren/syntren_cli.sh '+ini_filepath+new_ini_filename
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
#        print('CHECKK: Sample file created ', output, error)
        os.chdir('../../')
#        print('dir path: ', os.path.dirname(os.path.realpath(__file__)))

#        print('Loading the sample file')
        filetype = ['%0.2f'%noise_level, 'experiments_'+str(self.m)+'_', '_normalized_']
        print('noise level %0.2f'%noise_level, filetype)
        # grabbing all the files in the folder
        files = [f for f in listdir(sample_filepath) if isfile(join(sample_filepath, f))]
#        sample_file = [f for f in files if filetype in f]
        #print('samples files generated: ', files)
        sample_file = []
        for f in files:
            PRESENT = 1
            for ft in filetype:
                if ft not in f:
                    PRESENT=0
                    break
            if PRESENT:
                sample_file.append(f)

#        print('Files: ', files, sample_file, len(sample_file)==1)
        if len(sample_file)==1:
            df = pd.read_csv(sample_filepath+sample_file[0], sep='\t', index_col=0)
            data = np.array(df).T # M x N
        else:
            print('Some error in samples file: check ', sample_filepath)
#        print('Data read: ', df, data, data.shape)
        #read_path = 'expts_gene/syntren/data/nmse_samples/'
        #bash_command = 'bash syntren_cli.sh data/nmse_samples/sample_nmse.ini'
        # collect the samples from the file generated and take the transpose

#        data = np.random.multivariate_normal(mean=np.zeros(self.n), cov=cov, size=self.m)#.T
        return data# mxn

    # procedure of creating synthetic datasets in ISTA for glasso 2012 paper
    def get_samples(self, cov, seed=None):
        if seed != None:
            # NOTE: seeding the randomness
            np.random.seed(seed)
        data = np.random.multivariate_normal(mean=np.zeros(self.n), cov=cov, size=self.m)#.T
        return data# mxn

    # procedure of creating synthetic datasets in ISTA for glasso 2012 paper
    def init_random_graph_glasso(self, u=1, seed=None):
        if seed != None:
            np.random.seed(seed)
        prob = np.random.uniform(self.p_min, self.p_max) # defining the graph sparsityi
        self.collect_p.append(prob)
        G = nx.generators.random_graphs.gnp_random_graph(self.n, prob, seed=seed, directed=False)
        # Covariance matrix should be positive semi-definite & inverse of precision matrix
        theta = nx.adjacency_matrix(G).todense() # adjacency matrix
        # theta is currently all ones and symmetric
        if seed != None:
            np.random.seed(seed)
        # uniform [-1, 1]
        U = np.matrix(np.random.random((self.n, self.n)) * 2 - 1)
        #print('err: ', theta, U)
        theta = np.multiply(theta, U)
        # making it symmetric
        theta = (theta + theta.T)/2
        smallest_eigval = np.min(np.linalg.eigvals(theta))
#        print('smallest eval before I: ', smallest_eigval)
        #precision_mat = theta + np.eye(self.n)*(np.abs(smallest_eigval)+ u)# + 0.1 + u)
        precision_mat = theta + np.eye(self.n)*(u - smallest_eigval)# + 0.1 + u)
#        print('check smallest eval = %.3f '%(np.min(np.linalg.eigvals(precision_mat))), ' sparsity = %.3f' %(1-np.count_nonzero(precision_mat==0)/(self.n*self.n)), ' condition number = %.3f ' %(np.linalg.cond(precision_mat)))
#        print('adjacency matrix:', theta, np.sum(theta), 'prob = ', prob)
#        print('precision matrix:', precision_mat)
        cov = np.linalg.inv(precision_mat) # avoiding the use of pinv as data not true representative of conditional independencies.
        if isPSD(precision_mat) == 'False':
            print('NOT A PSD matrix!!! CHECK!!!')
            print('CHECK whether cov is sym: ', check_sym(cov), ' w_val = ', w_val)
            print('CHECK whether precision is sym : ', check_sym(precision_mat), w_val)
        #data = np.random.multivariate_normal(mean=np.zeros(self.n), cov=cov, size=self.m)#.T
        return precision_mat, cov# mxn
#        return cov, data# mxn

def make_graph_connected(G):
    total_nodes = len(G.nodes())
    for n in G.nodes():
#        print('n ', n)
        if len(G[n])==0 and n<total_nodes-1:# add a random node to its adjacency list if unconnected
            G.add_edge(n, np.random.randint(n+1, total_nodes)) # select node from unexplored set of nodes
        elif len(G[n])==0 and n==total_nodes-1: #for the last node
            G.add_edge(n, np.random.randint(0, n))

    return G


class create_MN_ecoli(object):
    def __init__(self, k_train, m, graph_type, k_valid=1, w_val=0.15, u=1):
        self.k_train = k_train # number of graph for training
        self.k_valid = k_valid
        self.m = m # number of samples
        self.train_graphs = {}
        self.valid_graphs = {}
        filetype = ['_'+str(self.m)+'_', '_normalized_']
        sample_filepath = 'expts_gene/syntren/data/samples/sample0/' 
        # grabbing all the files in the folder
        files = [f for f in listdir(sample_filepath) if isfile(join(sample_filepath, f))]
        sample_file = []
        for f in files:
            PRESENT = 1
            for ft in filetype:
                if ft not in f:
                    PRESENT=0
                    break
            if PRESENT:
                sample_file.append(f)
        df = pd.read_csv(sample_filepath+sample_file[0], sep='\t', index_col=0)
        data = np.array(df).T # M x N
        dummy_precision_mat = np.eye(data.shape[1])
        if graph_type == 'ecoli':
            edges = pd.read_csv('expts_gene/syntren/data/sourceNetworks/EColi_full.sif', sep='\t', header=None)
        else:
            edges = pd.read_csv('expts_gene/syntren/data/sourceNetworks/'+graph_type+'.sif', sep='\t', header=None)
        edges = np.array(edges[[0, 2]])
        G=nx.Graph()
        G.add_edges_from(edges)
        print('G_nodes: ', G.nodes(), len(G.nodes()))
        self.nodes = G.nodes()
        self.G_true = G
        
        theta = nx.adjacency_matrix(G).todense()
       
        # uniform [a, b ]
        a, b = 0.15, 0.15
        U = np.matrix(np.random.random((theta.shape[0], theta.shape[1])) * (b-a) + a)
        #print('err: ', theta, U)
        theta = np.multiply(theta, U)
        # making it symmetric
        theta = (theta + theta.T)/2

#        w_val = np.random.uniform(0.15, 0.2)
#        precision_mat = theta*w_val #+ np.eye(data.shape[1])*3
        # making it psd with min eig=1
        #smallest_eigval = np.real(np.min(np.linalg.eigvals(precision_mat)))
        #smallest_eigval = np.min(np.linalg.eigvals(theta))
        smallest_eigval = np.real(np.min(np.linalg.eigvals(theta)))
        print('smallest eval before I: ', smallest_eigval)#, np.real(smallest_eigval))
        #print("CHECKK: ", sum(precision_mat==precision_mat.T), precision_mat, sum(np.isreal(precision_mat)))
#        print("CHECKK: ", sum(theta==theta.T), theta, sum(np.isreal(theta)))
#        print("CHECKK: ", theta)
        #precision_mat = theta + np.eye(self.n)*(np.abs(smallest_eigval)+ u)# + 0.1 + u)
        precision_mat = theta + np.eye(data.shape[1])*(u - smallest_eigval)# + 0.1 + u)
        if isPSD(precision_mat) == 'False':
            print('NOT A PSD matrix!!! CHECK!!!')
            print('CHECK whether cov is sym: ', check_sym(cov), ' w_val = ', w_val)
            print('CHECK whether precision is sym : ', check_sym(precision_mat), w_val)
        self.train_graphs[0] = [precision_mat, data]
#        self.train_graphs[1] = [precision_mat, data]
        self.valid_graphs[0] = [precision_mat, data]
        print('Ecoli graph: ', data.shape, precision_mat.shape)


def main():
    print('code for generating samples from multivariate Gaussian distribution, given Sigma and assuming mu=0')
    for p in [0.1]: #, 0.2, 0.5, 0.9]:
        print(p)
        mn = create_MN(1, 10, 5, p, 1, 1) # features with prob of egde = 0.2
    print(mn.train_graphs)
    print('Checking the diagonal of sample covariance matrix')
    prec_mat, data = mn.train_graphs[0]
    print(np.matmul(data.T, data)/len(data))
    
#    plt.figure()
#    nx.draw_networkx(mn.true_graph, pos=nx.spring_layout(mn.true_graph), with_labels = True, name='true graph')
#    plt.savefig(true_graph+'.png')
#    plt.show()
    return

if __name__=="__main__":
    main()
