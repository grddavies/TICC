import multiprocessing as mp
import operator
import warnings

import numpy as np
from sklearn.utils import check_random_state
from sklearn import mixture
from sklearn.base import BaseEstimator, ClusterMixin

from ticclib.admm_solver import ADMMSolver


def _update_clusters(LLE_node_vals, beta=1) -> np.array:
    """Compute the minimum cost path given node costs in LLE_node_vals and edge
    cost beta.
    This is equivalent to assigment of T samples to K clusters with switching
    parameter beta.

    Parameters
    ----------
    LLE_node_vals : array (n_samples, num_clusters)
        Matrix of estimated log likelihood values of the assignment of each
        sample to a cluster. Note the LLE's are actually the negative of the
        true LLE's!

    beta : float >= 0
        switching penalty to impart temporal consistency to cluster assignments

    Returns
    -------
    path : array (n_samples, )
        equivalent of cluster assignments per sample
    """
    if beta < 0:
        raise ValueError("beta parameter should be >= 0 but got value  of %.3f"
                         % beta)
    if beta == 0:
        path = LLE_node_vals.argmin(axis=1)
        return path

    (T, num_clusters) = LLE_node_vals.shape
    future_cost_vals = np.zeros(LLE_node_vals.shape)

    # compute future costs
    for i in range(T-2, -1, -1):
        j = i+1
        future_costs = future_cost_vals[j, :]
        lle_vals = LLE_node_vals[j, :]
        for cluster in range(num_clusters):
            total_vals = future_costs + lle_vals + beta
            total_vals[cluster] -= beta
            future_cost_vals[i, cluster] = np.min(total_vals)

    # compute the best path
    path = np.zeros(T)

    # the first location
    curr_location = np.argmin(future_cost_vals[0, :] + LLE_node_vals[0, :])
    path[0] = curr_location

    # compute the path
    for i in range(T-1):
        j = i+1
        future_costs = future_cost_vals[j, :]
        lle_vals = LLE_node_vals[j, :]
        total_vals = future_costs + lle_vals + beta
        total_vals[int(path[i])] -= beta

        path[i+1] = np.argmin(total_vals)

    # return the computed path
    return path


class _TICCluster:
    """Class to represent clusters discovered using TICC

    Parameters:
    ----------
    label : int
        numeric label to identify cluster.

    Attributes:
    ----------
    MRF_ : array (n*w, n*w)}, n = n_features, w = window_size
        Block Toeplitz inverse covariance matrix that defines cluster.

    logdet : float
        log(det(MRF_))

    indices : list
        indices of points assigned to cluster

    inv_cov_matrix : array
        how is diff from MRF?
    """
    def __init__(self, label: int):
        self.label = label

    def __index__(self):
        return self.label

    def __len__(self):
        return len(self.indices)

    def get_indices(self, y: np.array) -> np.array:
        self.indices = np.where(y == self.label)[0]

    def get_mean(self, X_stacked):
        self.mean = X_stacked[self.indices, :].mean(axis=0)

    def get_logdet(self):
        self.logdet = np.log(np.linalg.det(self.computed_covariance))

    def get_inv_cov_matrix(self):
        self.inv_cov_matrix = np.linalg.inv(self.computed_covariance)


class TICC(BaseEstimator, ClusterMixin):
    """Toeplitz inverse-covariance based clustering of multivariate timeseries.

    Parameters:
    ----------
    n_clusters : int
        number of clusters/segments to form as well as number of
        inverse covariance matrices to fit.

    window_size : int
        size of the sliding window in samples.

    lambda_parameter : float
        sparsity parameter which determines the sparsity level in
        the Markov random fields (MRFs) characterizing each cluster.

    beta : float
        temporal consistency parameter (beta). A smoothness penalty that
        encourages adjacent subsequences to be assigned to the same cluster.

    max_iter : int
        Maximum number of iterations of the TICC algorithm for a single
        run.

    cluster_reassignment : int
        number of points to reassign to a 0 cluster

    random_state : int, RandomState instance, default=None
        A pseudo random number generator used for the initialization of the
        cluster initialisation by gaussian mixture model and to select a point
        to be added to empty clusters during cluster training. Use an int to
        make the randomness deterministic.

    copy_x : bool, default=True
        If copy_x is True (default), then the original data is
        not modified. If False, the original data is modified, and put back
        before the function returns, but small numerical differences may be
        introduced by subtracting and then adding the data mean. Note that if
        the original data is not C-contiguous, a copy will be made even if
        copy_x is False. If the original data is sparse, but not in CSR format,
        a copy will be made even if copy_x is False.

    Attributes
    ----------
    labels_ : array, shape (n_samples, 1)
        Cluster assignment of each point. Available only if after calling
        ``fit``.

    clusters_ : list (n_clusters)
        List of Cluster objects discovered.

    converged_ : bool
        True when convergence was reached in ``fit``, False otherwise.
    """
    # TODO: order cluster labels by norm to aid label consistency
    # between runs and allow scoring by label constancy during CV
    def __init__(self, *, n_clusters=5, window_size=10,  lambda_parameter=11e-2,
                 beta=400, max_iter=1000, num_proc=1, cluster_reassignment=20,
                 random_state=None, copy_x=True):
        self.window_size = window_size
        self.n_clusters = n_clusters
        self.lambda_parameter = lambda_parameter
        self.beta = beta
        self.max_iter = max_iter
        self.num_proc = np.clip(num_proc, 1, mp.cpu_count())
        self.cluster_reassignment = cluster_reassignment
        self.random_state = random_state
        self.copy_x = copy_x
        self.converged_ = False

    def _initialize(self, X_stacked):
        """Initialize the cluster assignments

        Parameters
        ----------
        X_stacked : array-like (n_samples, n_features*window_size)

        Returns
        -------
        clustered_points : array-like (n_samples, 1)
            cluster labels for each sample
        """
        random_state = check_random_state(self.random_state)
        gmm = mixture.GaussianMixture(n_components=self.n_clusters,
                                      covariance_type="full",
                                      random_state=random_state
                                      )

        clustered_points = gmm.fit_predict(X_stacked)
        return clustered_points

    def _e_step(self, X_stacked):
        """E step of EM algorithm.

        Assign points in X_stacked to clusters given current cluster params.
        Update cluster lengths.

        Parameters
        ----------
        X_stacked : array-like (n_samples, n_features*window_size)

        Returns
        -------
        clustered_points : array-like (n_samples,)
            cluster labels per point
        """
        for cluster in self.clusters_:
            cluster.get_mean(X_stacked)
            cluster.get_inv_cov_matrix()
            cluster.get_logdet()  # log(det(sigma2|1))

        LLEs = self._lle_point_assignments(X_stacked)

        # label points
        clustered_points = _update_clusters(LLEs, beta=self.beta)
        # Update cluster indices/lengths
        for cluster in self.clusters_:
            cluster.get_indices(clustered_points)
        return clustered_points

    def _lle_point_assignments(self, X_stacked) -> np.array:
        """Part-vectorised LLE calculation"""
        lle_cols = []
        for cluster in self.clusters_:
            cluster_stacked_mean = cluster.mean
            X_stacked_ = X_stacked - cluster_stacked_mean
            inv_cov_matrix = cluster.inv_cov_matrix
            log_det_cov = cluster.logdet
            lle_cols.append(
                # TODO: Does loopdot scale better than einsum?
                np.einsum('ij,ij->i',
                          X_stacked_@inv_cov_matrix,
                          X_stacked_,
                          ).reshape([-1, 1])
                + log_det_cov
            )

        return np.concatenate(lle_cols, axis=1)

    def _m_step(self, X_stacked, pool):
        """Update cluster parameters by solving Toeplitz graphical lasso problem
        Pass cluster points to ADMM solver to optimize, then update cluster
        parameters based on current assignments"""
        optRes = [None for i in range(self.n_clusters)]
        w = self.window_size
        n = self.n_features_in_
        for cluster in self.clusters_:
            if len(cluster) != 0:
                indices = cluster.indices
                D_train = X_stacked[indices, :]
                # Fit a model - OPTIMIZATION
                probSize = w * n
                lamb = np.zeros((probSize, probSize)) + self.lambda_parameter
                # S is the empirical covariance of points in cluster
                S = np.cov(D_train, rowvar=False)  # NOTE: data vars in cols
                solver = ADMMSolver(lamb, w, n, 1, S)
                # apply to process pool
                # solver(maxIters, eps_abs, eps_rel, verbose)
                optRes[cluster] = pool.apply_async(solver,
                                                   (1000, 1e-6, 1e-6, False,))
                # S is empirical covariance of points in cluster
                S_est = optRes[cluster].get()  # BUG: Hangs, scipy >=1.3
                cov_out = np.linalg.inv(S_est)

                # Store the log-det, covariance, inverse-covariance
                # TODO: Check! these look wrong
                cluster.logdet = np.log(np.linalg.det(cov_out))
                cluster.computed_covariance = cov_out
                cluster.norm = np.linalg.norm(cov_out)
                cluster.MRF_ = S_est
        return self

    def _empty_cluster_assign(self, clustered_points):
        """Reassign points to empty clusters"""
        random_state = check_random_state(self.random_state)
        n = self.cluster_reassignment
        # Sort clusters by norm of covariance matrix
        clusters_sorted = sorted(self.clusters_,
                                 key=operator.attrgetter('norm'),
                                 reverse=True)
        # clusters that are not 0 as sorted by norm
        valid_clusters = [c for c in self.clusters_ if len(c) != 0]

        # Add points to empty clusters
        # assuming more non empty clusters than empty ones
        counter = 0
        for cluster in self.clusters_:
            if len(cluster) == 0:
                source = self.clusters_[clusters_sorted[counter]]
                counter = (counter + 1) % len(valid_clusters)
                print(f"cluster {cluster.label} is empty, moving points from "
                      f"cluster {source.label}")
                # random points from that cluster
                cluster.indices = random_state.choice(source.indices, size=n)
                # Change point labels
                clustered_points[cluster.indices] = cluster.label
                # Clone source cluster covariance parameter into empty cluster
                cluster.computed_covariance = source.computed_covariance
        return clustered_points

    def fit(self, X, y=None, sample_weight=None):
        """Compute Toeplitz Inverse Covariance-Based Clustering.

        Parameters
        ----------
        X : {array-like, sparse matrix} (n_samples, n_features).
            Training timeseries to cluster (in ascending time order).

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : Ignored
            Feature not implemented yet, present for API consistency.

        Returns
        -------
        self
        """
        assert self.max_iter > 0,  "Must have at least one iteration"
        try:
            n_samples, n_cols = X.shape
            if n_cols != self.n_features_in_ * self.window_size:
                raise ValueError("Input dimensions incorrect! Did you call "
                                 "``stack_data()`` before ``fit``?")
            else:
                X_stacked = X

        except AttributeError:
            # Stack the data into a n_samples by n_features*window_size array
            X_stacked = self.stack_data(X)
            n_samples, _ = X_stacked.shape

        # Initialization - GMM to cluster
        clustered_points_old = self._initialize(X_stacked)
        self.clusters_ = [_TICCluster(k) for k in range(self.n_clusters)]
        for cluster in self.clusters_:
            cluster.get_indices(clustered_points_old)

        # PERFORM TRAINING ITERATIONS
        pool = mp.Pool(processes=self.num_proc)  # multi-threading
        for i in range(1, self.max_iter+1):
            print("ITERATION # %d" % i)
            # M-Step > E-Step
            clustered_points = (self._m_step(X_stacked, pool)
                                    ._e_step(X_stacked)
                                )

            # Reassign to empty clusters
            clustered_points = self._empty_cluster_assign(clustered_points)

            if np.array_equal(clustered_points_old, clustered_points):
                self.converged_ = True
                print("Converged - Breaking Early")
                break

            clustered_points_old = clustered_points.copy()
            for cluster in self.clusters_:
                cluster.get_indices(clustered_points)

        if pool is not None:
            pool.close()
            pool.join()

        self.labels_ = clustered_points
        if not self.converged_:
            warnings.warn("Model did not converge after %d iterations. Try "
                          "changing model parameters or increasing `max_iter`"
                          % self.max_iter, RuntimeWarning)
        return self

    def stack_data(self, X):
        """Stack input data into array (n_samples, n_features*window)
        Note that the timeseries X must be in ascending time order

        Rather than just looking at xt (the (n, 1) vector of at time t), we
        instead cluster a short subsequence of size w ≪ n_samples that ends at
        time t. The resultant n_features * window vector is made up of xt and
        the previous (w - 1) observations.
        """
        def zero_rows(X, i):
            """Set first i rows of X to zero"""
            X[:i, :] = 0
            return X
        X = self._validate_data(X, accept_sparse='csr',
                                dtype=[np.float64, np.float32],
                                order='C', copy=self.copy_x,
                                accept_large_sparse=False)
        return np.concatenate([zero_rows(np.roll(X, i, axis=0,), i)
                               for i in range(self.window_size)], axis=1)

    def predict(self, X_stacked):
        LLEs = self._lle_point_assignments(X_stacked)
        # label points
        clustered_points = _update_clusters(LLEs, beta=self.beta)
        return clustered_points
