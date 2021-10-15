# Test

### Data:

scMultiSim data:

* Cells: 1000

* ngenes, ntfs: (100, 6), cannot maintain trajectory structure for smaller graph with 20 genes.

* interval: 50, 100, 200

* Changing: 0.1% edges every interval. 

* Initial graph, 6 ntfs, 94 targets, no interaction between tfs.

  <img src = "../../data/scMultiSim/ngenes_100_interval_50/true_count_plot.png">

  <img src = "../../data/scMultiSim/ngenes_100_interval_50/obs_count_plot.png">

### Results:







### Data:

Dataset of bifurcating trajectory. Detailed parameters:
* ncells: 1000
* stepsize: 0.0001, 0.0002 (bad performance for genie3, didn't use it when testing admm)
* ngenes, ntfs: (20, 5), (30, 10), (50, 20), (100, 50)
* interval: 50, 100, 200
* Sergio initial graph

### Testing steps:
1. Test on genie3, genie3 don't need hyper-parameters. Test genie3 on cases using TF/not using TF, and benchmark the result. Genie3 should represent our performance somehow, if genie3 is good, our shouldn't be too bad.

   * Test with different preprocessing set-up, using observed count with log-normalized preprocessing step shows better performance (especially when there are 20 genes).
    <img src = "../results_softODE_sergio/cosine.png">
   * Test on different dataset. The number of genes significantly affect the performance, the changing speed as minorly affect the performance. Cosine similarity have higher score compared to pearson.
    <img src = "../results_softODE_sergio/genie_obs.png">
   *  Using TF compared to previous test. The trend is clearer.
    <img src = "../results_softODE_sergio/genie_obs_tf.png">
   * Additional, using stepsize = 0.0001 give genie3 better performance.

2. Test on ADMM model. For the model on softODE data, we don't know the suitable hyper-parameter yet, so first test on the data with 20 genes, the hyper-parameter should not be affected too much by the number of genes. Bandwidth should have correlation with the changing speed, and lambda should have correlation with the graph degree. We need to make sure the rho parameter (None, 1.7) effect on the final result.

    * Rho setting effect: alpha = 2, rho = 1.7 have better performance compared to alpha = 1, rho = None.
    <img src = "../results_softODE_sergio/results_ngenes_20_interval_50_stepsize_0.0001/boxplot_mode.png">
    <img src = "../results_softODE_sergio/results_ngenes_20_interval_100_stepsize_0.0001/boxplot_mode.png">
    <img src = "../results_softODE_sergio/results_ngenes_20_interval_200_stepsize_0.0001/boxplot_mode.png">
    * Lambda setting. Lambda choice [0.001, 0.001, 0.1]
    * Bandwidth setting: [0.01, 0.1]
