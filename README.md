# DynGRN 

## Description
Dynamical Gene Regulatory Network Inference. 

* `src` stores the inference algorithms.
* `test` stores the `scripts` (testing scripts) and `results` (testing results).
* `simulator` stores the simulation code, include two simulators: `boolODE`, `linearODE` and `GGM`.
* `data` stores the generated data
  * `boolODE_Sep13` stores the bool ODE data that we used to comprehensively test the model on Sep. 13th.
  * `linearODE_Sep13` stores the linear ODE data that we used to comprehensively test the model on Sep. 13th.
  * `GGM` stores the Gaussian Graphical Model data that we used to test the model.


## Data
We generated the data:

**boolODE_Sep13**

Initial Grpah:
- nodes = 18 (12 TFs, 6 Target genes)
- edges = 18 (TF regulates (1 or 2) TFs/Target genes, '1to2' topology)

Graph Evolving/Perturb rules:
- Total time points = 7,500
- Continuous change = Every 5 time points, 2 interactions are randomly exchanged
- Discrete change = Every 1,500 time points, 10 interactions are randomly exchanged

Data: 
- '*_sorted_exp_1to2.npy' = Gene expression data 
- '*_gt_adj_1to2.npy' = Ground-Truth adjacency matrix (Boolean)

* `continue_gt_adj_1to2.npy`(7500, 18, 18) and `continue_sorted_exp_1to2.npy`(7500, 18) 
* `discrete_gt_adj_1to2.npy`(7500, 18, 18) and `discrete_sorted_exp_1to2.npy`(7500, 18) 

**linearODE_Sep13**

Initial Grpah:
- nodes = 18 (12 TFs, 6 Target genes)
- edges = 18 (TF regulates (non-TF) Target genes)

Graph Evolving/Perturb rules:
- Total time points = 7,500
- Continuous change = Every 5 time points, 2 interactions are randomly exchanged
- Discrete change = Every 1,500 time points, 10 interactions are randomly exchanged

Data: 
- '*_exp_linear_sim.npy' = Gene expression data 
- '*_gt_adj_linear_sim.npy' = Ground-Truth adjacency matrix (Boolean)

* `continue_gt_adj_linear_sim.npy`(7500, 18, 18) and `continue_exp_linear_sim.npy`(7500, 18) 
* `discrete_gt_adj_linear_sim.npy`(7500, 18, 18) and `discrete_exp_linear_sim.npy`(7500, 18) 


**GGM**

* `ntimes_1000_interval_200`: Gaussian graphical model with 1000 time points and 200 intervals for each edge permutation.
* `ntimes_3000_interval_200`: Gaussian graphical model with 3000 time points and 200 intervals for each edge permutation.


## Result
**Test on GGM data**

*Simulation parameter:* ntimes = 3000, interval = 200, ngenes = 20, ntfs = 5

*Model hyper-parameters:* ADMM model, adaptive rho/fix rho, different alpha, bandwidth, lambda, using mean/not using mean for covariance calculation.

*Best:* bandwidth = 0.1, lambda = 0.01, fix rho = 1.7, alpha = 1/2

