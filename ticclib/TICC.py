import collections
import warnings
import multiprocessing as mp
import numpy as np
# import sklearn
# import pickle
from sklearn.utils import check_random_state
from sklearn import mixture
from sklearn.base import BaseEstimator, ClusterMixin
from typing import Dict, List, Tuple

from ticclib.admm_solver import ADMMSolver


def _upper_to_full(a, eps=0):
    ind = (a < eps) & (a > -eps)
    a[ind] = 0
    n = int((-1 + np.sqrt(1 + 8*a.shape[0]))/2)
    A = np.zeros([n, n])
    A[np.triu_indices(n)] = a
    temp = A.diagonal()
    A = np.asarray((A + A.T) - np.diag(temp))
    return A


def _update_clusters(LLE_node_vals, beta=1):
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
        # Easily avoid loops if no switching penalty
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


def _cluster_point_indices(clustered_points: np.ndarray, n_clusters: int
                           ) -> Tuple[Dict[int, np.ndarray], Dict[int, int]]:
    """
    Return dicts for the indices of points in each cluster, and of the
    number of points in each cluster
    """
    clusters_arr = collections.defaultdict(list)
    for point, cluster_num in enumerate(clustered_points):
        clusters_arr[cluster_num].append(point)
    # dict of number of points in each cluster
    len_clusters = {k: len(clusters_arr[k]) for k in range(n_clusters)}
    return (clusters_arr, len_clusters)


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

    clusters_ : dict {cluster_number: {parameter_name: value}
        Parameters:
            - cluster_MRF_ : array (n*w, n*w)}, n = n_features, w = window_size
            Block Toeplitz inverse covariance matrices that define cluster.
            - 

    converged_ : bool
        True when convergence was reached in fit(), False otherwise.
    """
    # TODO: order cluster labels by some method to aid label consistency
    # between runs and allow scoring by label constancy during CV
    def __init__(self, *, n_clusters=5, window_size=10,  lambda_parameter=11e-2,
                 beta=400, max_iter=1000, num_proc=1, cluster_reassignment=20,
                 random_state=None, copy_x=True):
        self.window_size = window_size
        self.n_clusters = n_clusters
        self.lambda_parameter = lambda_parameter
        self.beta = beta
        self.max_iter = max_iter
        self.num_proc = num_proc  # np.clip(num_proc, 1, mp.cpu_count())
        self.cluster_reassignment = cluster_reassignment
        self.random_state = random_state
        self.copy_x = copy_x

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
        # TODO: is there a better initialisation?
        gmm = mixture.GaussianMixture(n_components=self.n_clusters,
                                      covariance_type="full",
                                      random_state=random_state
                                      )

        clustered_points = gmm.fit_predict(X_stacked)
        return clustered_points

    def _e_step(self, X_stacked):
        """E step of EM algorithm.

        Assign points in X_stacked to clusters given dict of cluster params:
         - cluster_t_means
         - computed_covariance
         - cluster_stacked_means

        Parameters
        ----------
        X_stacked : array-like (n_samples, n_features*window_size)

        Returns
        -------
        clustered_point : array-like (n_samples,)
            cluster labels per point
        """
        pass

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
        assert self.max_iter > 0  # must have at least one iteration
        # Check input format
        X = self._validate_data(X, accept_sparse='csr',
                                dtype=[np.float64, np.float32],
                                order='C', copy=self.copy_x,
                                accept_large_sparse=False)
        n_samples, n_features = X.shape
        assert n_features == self.n_features_in_
        random_state = check_random_state(self.random_state)

        # Stack the data into a n_samples by n_features*window_size array
        X_stacked = self.stack_data(X)

        # Initialization
        clustered_points = self._initialize(X_stacked)

        self.clusterMRFs_ = {}
        log_det_values = {}  # log dets of the thetas
        computed_covariance = {}
        cluster_t_means = {}
        cluster_stacked_means = {}
        old_clustered_points = None  # points from last iteration

        empirical_covariances = {}

        # PERFORM TRAINING ITERATIONS
        pool = mp.Pool(processes=self.num_proc)  # multi-threading
        # pool = None  # If get rid of pool, get rid of .get()
        # Zeroth training iteration
        print("\n\n\nITERATION # 0")

        # clusters_arr : {cluster: [point indices]}
        clusters_arr, len_clusters = _cluster_point_indices(clustered_points,
                                                            self.n_clusters)

        opt_res = self.train_clusters(cluster_t_means,
                                      cluster_stacked_means,
                                      X_stacked,
                                      empirical_covariances,
                                      len_clusters,
                                      pool,
                                      clusters_arr)

        self.optimize_clusters(computed_covariance,
                               len_clusters,
                               log_det_values,
                               opt_res,
                               )

        # update old computed covariance
        old_computed_covariance = computed_covariance

        print("UPDATED THE OLD COVARIANCE")

        self.trained_model = {'cluster_t_means': cluster_t_means,
                              'computed_covariance': computed_covariance,
                              'cluster_stacked_means': cluster_stacked_means,
                              'n_features': n_features}

        # def assign_to_clusters(X_stacked):
        #     """Apply clustering to input based on current cluster params"""
        #     lle_all_points_clusters = self._smoothen_clusters(
        #         cluster_t_means, computed_covariance,
        #         cluster_stacked_means, X_stacked, self.n_features_in_
        #     )
        #     return _update_clusters(lle_all_points_clusters,
        #                             beta=self.beta)
        # clustered_points = assign_to_clusters(X_stacked)

        clustered_points = self.predict_clusters(X_stacked)
        # Recalculate lengths
        _, len_new_clusters = _cluster_point_indices(clustered_points,
                                                     self.n_clusters)

        before_empty_cluster_assign = clustered_points.copy()

        # for cluster_num in range(self.n_clusters):
        #     print("length of cluster #", cluster_num, "-------->", sum([x == cluster_num for x in clustered_points]))

        old_clustered_points = before_empty_cluster_assign

        for iters in range(1, self.max_iter + 1):
            # Get the points assigned to each cluster
            clusters_arr, len_clusters = _cluster_point_indices(
                clustered_points, self.n_clusters)

            # optimization result?
            opt_res = self.train_clusters(cluster_t_means, cluster_stacked_means, X_stacked,
                                          empirical_covariances, len_clusters, pool,
                                          clusters_arr)

            self.optimize_clusters(computed_covariance, len_clusters,
                                   log_det_values, opt_res,
                                   )

            # update old computed covariance
            old_computed_covariance = computed_covariance

            # print("UPDATED THE OLD COVARIANCE")

            self.trained_model = {'cluster_t_means': cluster_t_means,
                                  'computed_covariance': computed_covariance,
                                  'cluster_stacked_means': cluster_stacked_means,
                                  'n_features': n_features}
            clustered_points = self.predict_clusters(X_stacked)
            # clustered_points = assign_to_clusters(X_stacked)

            # recalculate lengths
            new_clusters, len_new_clusters = _cluster_point_indices(
                clustered_points, self.n_clusters
            )

            before_empty_cluster_assign = clustered_points.copy()

            cluster_norms = [(np.linalg.norm(old_computed_covariance[i]), i) for i in range(self.n_clusters)]
            norms_sorted = sorted(cluster_norms, reverse=True)
            # clusters that are not 0 as sorted by norm
            valid_clusters = [cp[1] for cp in norms_sorted if len_new_clusters[cp[1]] != 0]

            # Add points to empty clusters
            # assuming more non empty clusters than empty ones
            counter = 0
            for cluster_num in range(self.n_clusters):
                if len_new_clusters[cluster_num] == 0:
                    # Do if no points in cluster
                    cluster_selected = valid_clusters[counter]  # a cluster that is not len 0
                    counter = (counter + 1) % len(valid_clusters)
                    print("cluster that is zero is:", cluster_num, "selected cluster instead is:", cluster_selected)
                    # random point number from that cluster
                    start = random_state.choice(new_clusters[cluster_selected])
                    for i in range(self.cluster_reassignment):
                        # put cluster_reassignment points from point_num in this cluster
                        point_to_move = start + i
                        if point_to_move >= len(clustered_points):
                            break
                        # Swap label assignment
                        clustered_points[point_to_move] = cluster_num
                        # Do something with computed covariance of cluster
                        computed_covariance[cluster_num] = old_computed_covariance[cluster_selected]
                        # Move point FIXME: this smells
                        cluster_stacked_means[cluster_num] = X_stacked[point_to_move, :]
                        cluster_t_means[cluster_num] = X_stacked[point_to_move, :][(self.window_size - 1) * n_features:self.window_size * n_features]

            if np.array_equal(old_clustered_points, clustered_points):
                self.converged = True
                # print("\n\n\n\nConverged - Breaking Early")
                break
            old_clustered_points = before_empty_cluster_assign
            # end of training

        if pool is not None:
            pool.close()
            pool.join()

        self.labels_ = clustered_points

        if not self.converged:
            warnings.warn("Model did not converge after %d iterations. Try"
                          " changing model parameters or increasing `max_iter`"
                          % self.max_iter, RuntimeWarning)
        return self

    def _smoothen_clusters(self, X_stacked, cluster_t_means,
                           computed_covariance, cluster_stacked_means):
        """
        Return log-likelihood estimate of cluster assignments at each
        timepoint
        """
        n_samples = len(X_stacked)
        inv_cov_dict = {}  # cluster : inv_cov
        log_det_dict = {}  # cluster : log_det
        w = self.window_size
        n = self.n_features_in_
        # Get cluster parameters
        for cluster in range(self.n_clusters):
            cov_matrix = computed_covariance[cluster]
            inv_cov_matrix = np.linalg.inv(cov_matrix)
            log_det_cov = np.log(np.linalg.det(cov_matrix))  # log(det(sigma2|1))
            inv_cov_dict[cluster] = inv_cov_matrix
            log_det_dict[cluster] = log_det_cov

        # For each point estimate the log-likelihood (LLE)
        assignment_LLEs = np.zeros([n_samples, self.n_clusters])
        for point in range(n_samples):
            for cluster in range(self.n_clusters):
                # cluster_t_mean = cluster_t_means[cluster]
                cluster_stacked_mean = cluster_stacked_means[cluster]
                x = X_stacked[point, :] - cluster_stacked_mean
                inv_cov_matrix = inv_cov_dict[cluster]
                log_det_cov = log_det_dict[cluster]
                # lle = np.dot(x.reshape([1, w * n]),
                #              inv_cov_matrix@(x.reshape([n*w, 1]))
                #              ) + log_det_cov
                lle = x@inv_cov_matrix@x + log_det_cov
                assignment_LLEs[point, cluster] = lle

        # def lle_point_assignments(X_stacked, inv_cov_dict, log_det_dict):
        #     # thetas = np.array()
        #     lle_cols = []
        #     for cluster in range(self.n_clusters):
        #         cluster_stacked_mean = cluster_stacked_means[cluster]
        #         X_stacked = X_stacked - cluster_stacked_mean
        #         inv_cov_matrix = inv_cov_dict[cluster]
        #         log_det_cov = log_det_dict[cluster]
        #         lle_cols.append(

        #         )

        return assignment_LLEs

    def optimize_clusters(self, computed_covariance, len_clusters,
                          log_det_values, optRes):
        for cluster in range(self.n_clusters):
            if optRes[cluster] is None:
                # Skip clusters with no inverse cov matrix
                continue
            val = optRes[cluster].get()  # BUG: Hangs here with scipy >=1.3
            # val = optRes[cluster]
            print(f"Optimisation for Cluster #{cluster} DONE")
            # THIS IS THE SOLUTION
            # S is empirical covariance of cluster
            S_est = _upper_to_full(val, 0)
            # BUG: non-finite S_est
            if not np.isfinite(S_est).all():
                warnings.warn("Empirical covariance matrix for cluster"
                              f" {cluster} was not finite", RuntimeWarning)
                S_est = np.nan_to_num(S_est)
            u, _ = np.linalg.eig(S_est)  # eigenvalues of the covariance matrix
            cov_out = np.linalg.inv(S_est)

            # Store the log-det, covariance, inverse-covariance, cluster means, stacked means
            log_det_values[cluster] = np.log(np.linalg.det(cov_out))
            computed_covariance[cluster] = cov_out
            self.clusterMRFs_[cluster] = S_est
        for cluster in range(self.n_clusters):
            print(f"length of cluster {cluster} -> {len_clusters[cluster]}")

    def train_clusters(self, cluster_t_means, cluster_stacked_means,
                       X_stacked, empirical_covariances, len_clusters, pool,
                       clusters_arr) -> List[np.ndarray]:
        """
        Update the cluster parameters for current point assignments
        """
        optRes = [None for i in range(self.n_clusters)]
        w = self.window_size
        n = self.n_features_in_
        for cluster in range(self.n_clusters):
            cluster_length = len_clusters[cluster]  # num points in cluster
            if cluster_length != 0:
                indices = clusters_arr[cluster]
                D_train = X_stacked[indices, :]

                # Mean feature value at time t
                cluster_t_means[cluster] = (D_train.mean(axis=0)[:n]
                                                   .reshape([1, n]))
                cluster_stacked_means[cluster] = D_train.mean(axis=0)

                # Fit a model - OPTIMIZATION
                probSize = w * n
                lamb = np.zeros((probSize, probSize)) + self.lambda_parameter
                # S is the empirical covariance of points in cluster
                S = np.cov(D_train, rowvar=False)  # NOTE: data - vars in cols
                empirical_covariances[cluster] = S

                solver = ADMMSolver(lamb, w, n, 1, S)
                # apply to process pool
                # tuple passed to solver (maxIters, eps_abs, eps_rel, verbose)
                optRes[cluster] = pool.apply_async(solver, (1000, 1e-6, 1e-6, False,))
                # # single-process version: (get rid of .get() call)
                # optRes[cluster] = solver(1000, 1e-6, 1e-6, False,)
        return optRes

    # def stack_data(self, X):
    #     n_samples, n_features = X.shape
    #     #  Flip because X is in ascending time order
    #     X = np.flip(X, axis=0)  # Do these flips break stuff later?
    #     X_stacked = np.zeros([n_samples, self.window_size * n_features])
    #     for i in range(n_samples):
    #         for k in range(self.window_size):
    #             if i + k < n_samples:
    #                 # idx_k = training_indices[i + k]
    #                 # X_stacked[i][k * n_features:(k + 1) * n_features] = X[idx_k][0:n_features]
    #                 X_stacked[i][k * n_features:(k + 1) * n_features] = X[i + k][:n_features]
    #     self.X_stacked = X_stacked
    #     return np.flip(X_stacked, axis=0)

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
        return np.concatenate([zero_rows(np.roll(X, i, axis=0,), i)
                               for i in range(self.window_size)], axis=1)

    def predict_clusters(self, X_stacked):
        """
        Given the current trained model, predict clusters.  If the cluster
        segmentation has not been optimized yet, then this will be part of
        the iterative process.

        Args:
            numpy array of data for which to predict clusters.  Columns are
            dimensions of the data, each row is a different timestamp

        Returns:
            vector of predicted cluster for the points
        """
        # SMOOTHENING
        lle_all_points_clusters = self._smoothen_clusters(
            X_stacked, self.trained_model['cluster_t_means'],
            self.trained_model['computed_covariance'],
            self.trained_model['cluster_stacked_means'],
                                                         )

        # Update cluster points - using NEW smoothening
        clustered_points = _update_clusters(lle_all_points_clusters,
                                            beta=self.beta)
        return clustered_points
