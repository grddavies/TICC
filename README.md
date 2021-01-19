# ticclib

ticclib adapts the TICC class (see below) into an estimator compatible with the scikit-learn API.

## TICC

TICC is a python solver for efficiently segmenting and clustering a multivariate time series. It is initialised with the number of clusters `k`, a window size `w`, a regularization parameter `lambda` and smoothness parameter `beta`. A TICC estimator can then be fit to a T-by-n data matrix. TICC breaks the T timestamps into segments where each segment belongs to one of the `k` clusters. The total number of segments is affected by the smoothness parameter `beta`. Segmentation and labelling is performed by running an expectation-maximisation algorithm where TICC alternately assigns points to clusters using a dynamic programming algorithm and updates the cluster parameters by solving a Toeplitz Inverse Covariance Estimation problem.

For details about the method and implementation see the paper [1].

## Download & Setup

1. Download the source code for this implementation by running in the terminal:

        git clone https://github.com/hedscan/TICC.git

2. Install dependencies with `pip`

        pip install -r TICC/requirements.txt 

    or with `conda`
        
        conda install -n <envname> --file TICC/requirements.txt
    
3. Install `ticclib` in editable mode using `pip` 

        pip install -e TICC/

## Using TICC

The `TICC`-constructor takes the following parameters:

* `number_of_clusters`: the number 'k' of underlying clusters  to fit.
* `window_size`: the size of the sliding window in samples.
* `lambda_parameter`: sparsity of the Markov Random Field (MRF) for each of the clusters. The sparsity of the inverse covariance matrix of each cluster.
* `beta`: The switching penalty used in the TICC algorithm. Same as the beta parameter described in the paper.
* `maxIters`: The maximum iterations of the TICC algorithm before convergence. Default value is 100.
* `n_jobs`: The maximum number of concurrently running jobs to be run via `joblib`.
* `cluster_reassignment`: The proportion of points (0, 1) to move from a valid cluster to an empty cluster during `fit`.
* `random_state` : The generator used to initialise assingments and randomize point shuffling during empty cluster reassignment.
* `verbose` : If true print out iteration number and log any empty cluster reassignments.

Running the fit method on an array of multivariate timeseries data 'X', with rows in ascending-time order will return a fitted TICC estimator. A fitted estimator will have a list of 'k' clusters fitted to X, each with a block Toeplitz inverse covariance matrix which defines the cross-feature correlations for that cluster, and a list of points in X assigned to that cluster.

## Example Usage

See `example.py` for usage, and comparison with a Gaussian Mixture Model. Requires `matplotlib` and `networkx` for visualisations.

## References

[1] D. Hallac, S. Vare, S. Boyd, and J. Leskovec [Toeplitz Inverse Covariance-Based Clustering of Multivariate Time Series Data](https://arxiv.org/abs/1706.03161) Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 215--223
