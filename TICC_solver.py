import numpy as np
import collections
from sklearn import mixture
from sklearn.base import BaseEstimator, ClusterMixin
import multiprocessing as mp
from typing import List

import src.TICC_helper as Th
from src.admm_solver import ADMMSolver


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
        lobpcg eigen vectors decomposition when ``eigen_solver='amg'`` and by
        the K-Means initialization. Use an int to make the randomness
        deterministic.

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

    cluster_MRFs_ : dict {cluster_number: array of shape (nw, nw)}, where n is
                    number of features, w is window size
        Block Toeplitz inverse covariance matrices that define clusters.
    """
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
        self.num_blocks = self.window_size + 1
        self.random_state = np.random.RandomState(random_state)
        self.copy_x = copy_x
        np.random.seed(random_state)

    def fit(self, X, y=None, sample_weight=None):
        """Compute Toeplitz Inverse Covariance-Based Clustering.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
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
        self.log_parameters()

        # Check input format
        X = self._validate_data(X, accept_sparse='csr',
                                dtype=[np.float64, np.float32],
                                order='C', copy=self.copy_x,
                                accept_large_sparse=False)

        n_samples, n_features = X.shape
        assert n_features == self.n_features_in_

        # Stack the data into a n_samples by n_features*window_size array
        X_stacked = self.stack_data(X)

        # Initialization
        # Gaussian Mixture - TODO: is there a better initialisation?
        gmm = mixture.GaussianMixture(n_components=self.n_clusters,
                                      covariance_type="full",
                                      # TODO: random_state
                                      )

        clustered_points = gmm.fit_predict(X_stacked)

        self.cluster_MRFs_ = {}
        log_det_values = {}  # log dets of the thetas
        computed_covariance = {}
        cluster_mean_info = {}
        cluster_mean_stacked_info = {}
        old_clustered_points = None  # points from last iteration

        empirical_covariances = {}

        # PERFORM TRAINING ITERATIONS
        pool = mp.Pool(processes=self.num_proc)  # multi-threading
        # pool = None  # If get rid of pool, get rid of .get()
        # Zeroth training iteration
        print("\n\n\nITERATION # 0")

        # clusters_arr : {cluster: [point indices]}
        clusters_arr = collections.defaultdict(list)
        for point, cluster_num in enumerate(clustered_points):
            clusters_arr[cluster_num].append(point)
        # dict of number of points in each cluster
        len_clusters = {k: len(clusters_arr[k]) for k in range(self.n_clusters)}

        # train_clusters holds the indices in complete_D_train
        # for each of the clusters
        opt_res = self.train_clusters(cluster_mean_info,
                                      cluster_mean_stacked_info,
                                      X_stacked,
                                      empirical_covariances,
                                      len_clusters,
                                      n_features,
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

        self.trained_model = {'cluster_mean_info': cluster_mean_info,
                              'computed_covariance': computed_covariance,
                              'cluster_mean_stacked_info': cluster_mean_stacked_info,
                              'complete_D_train': X_stacked,
                              'n_features': n_features}
        clustered_points = self.predict_clusters()

        # Recalculate lengths
        new_train_clusters = collections.defaultdict(list)  # {cluster: [point indices]}
        for point, cluster in enumerate(clustered_points):
            new_train_clusters[cluster].append(point)

        len_new_train_clusters = {k: len(new_train_clusters[k]) for k in range(self.n_clusters)}

        before_empty_cluster_assign = clustered_points.copy()

        for cluster_num in range(self.n_clusters):
            print("length of cluster #", cluster_num, "-------->", sum([x == cluster_num for x in clustered_points]))

        old_clustered_points = before_empty_cluster_assign

        for iters in range(1, self.max_iter):
            print("\n\n\nITERATION #", iters)
            # Get the train and test points
            clusters_arr = collections.defaultdict(list)  # {cluster: [point indices]}
            for point, cluster_num in enumerate(clustered_points):
                clusters_arr[cluster_num].append(point)

            len_clusters = {k: len(clusters_arr[k]) for k in range(self.n_clusters)}

            # train_clusters holds the indices in complete_D_train
            # for each of the clusters
            opt_res = self.train_clusters(cluster_mean_info, cluster_mean_stacked_info, X_stacked,
                                          empirical_covariances, len_clusters, n_features, pool,
                                          clusters_arr)

            self.optimize_clusters(computed_covariance, len_clusters,
                                   log_det_values, opt_res,
                                   )

            # update old computed covariance
            old_computed_covariance = computed_covariance

            print("UPDATED THE OLD COVARIANCE")

            self.trained_model = {'cluster_mean_info': cluster_mean_info,
                                  'computed_covariance': computed_covariance,
                                  'cluster_mean_stacked_info': cluster_mean_stacked_info,
                                  'complete_D_train': X_stacked,
                                  'n_features': n_features}
            clustered_points = self.predict_clusters()

            # recalculate lengths
            new_train_clusters = collections.defaultdict(list)  # {cluster: [point indices]}
            for point, cluster in enumerate(clustered_points):
                new_train_clusters[cluster].append(point)

            len_new_train_clusters = {k: len(new_train_clusters[k]) for k in range(self.n_clusters)}

            before_empty_cluster_assign = clustered_points.copy()

            cluster_norms = [(np.linalg.norm(old_computed_covariance[self.n_clusters, i]), i) for i in range(self.n_clusters)]
            norms_sorted = sorted(cluster_norms, reverse=True)
            # clusters that are not 0 as sorted by norm
            valid_clusters = [cp[1] for cp in norms_sorted if len_new_train_clusters[cp[1]] != 0]

            # Add a point to the empty clusters
            # assuming more non empty clusters than empty ones
            counter = 0
            for cluster_num in range(self.n_clusters):
                if len_new_train_clusters[cluster_num] == 0:
                    cluster_selected = valid_clusters[counter]  # a cluster that is not len 0
                    counter = (counter + 1) % len(valid_clusters)
                    print("cluster that is zero is:", cluster_num, "selected cluster instead is:", cluster_selected)
                    start_point = self.random_state.choice(
                        new_train_clusters[cluster_selected])  # random point number from that cluster
                    for i in range(0, self.cluster_reassignment):
                        # put cluster_reassignment points from point_num in this cluster
                        point_to_move = start_point + i
                        if point_to_move >= len(clustered_points):
                            break
                        clustered_points[point_to_move] = cluster_num
                        computed_covariance[self.n_clusters, cluster_num] = old_computed_covariance[
                            self.n_clusters, cluster_selected]
                        cluster_mean_stacked_info[self.n_clusters, cluster_num] = X_stacked[point_to_move, :]
                        cluster_mean_info[self.n_clusters, cluster_num] \
                            = X_stacked[point_to_move, :][
                                (self.window_size - 1) * n_features:self.window_size * n_features]

            for cluster_num in range(self.n_clusters):
                print("length of cluster #", cluster_num, "-------->", sum([x == cluster_num for x in clustered_points]))

            print("\n")

            if np.array_equal(old_clustered_points, clustered_points):
                print("\n\n\n\nConverged - Breaking Early")
                break
            old_clustered_points = before_empty_cluster_assign
            # end of training

        if pool is not None:
            pool.close()
            pool.join()

        self.labels_ = clustered_points
        return self

    def smoothen_clusters(self, cluster_mean_info, computed_covariance,
                          cluster_mean_stacked_info, complete_D_train, n):
        clustered_points_len = len(complete_D_train)
        inv_cov_dict = {}  # cluster : inv_cov
        log_det_dict = {}  # cluster : log_det
        for cluster in range(self.n_clusters):
            cov_matrix = computed_covariance[self.n_clusters, cluster][0:(self.num_blocks - 1) * n,
                                                                       0:(self.num_blocks - 1) * n]
            inv_cov_matrix = np.linalg.inv(cov_matrix)
            log_det_cov = np.log(np.linalg.det(cov_matrix))  # log(det(sigma2|1))
            inv_cov_dict[cluster] = inv_cov_matrix
            log_det_dict[cluster] = log_det_cov
        # For each point estimate the log-likelihood (LLE)
        # print("beginning the smoothening ALGORITHM")
        LLE_all_points_clusters = np.zeros([clustered_points_len, self.n_clusters])
        for point in range(clustered_points_len):
            if point + self.window_size - 1 < complete_D_train.shape[0]:
                for cluster in range(self.n_clusters):
                    # cluster_mean = cluster_mean_info[self.n_clusters, cluster]
                    cluster_mean_stacked = cluster_mean_stacked_info[self.n_clusters, cluster]
                    x = complete_D_train[point, :] - cluster_mean_stacked[0:(self.num_blocks - 1) * n]
                    inv_cov_matrix = inv_cov_dict[cluster]
                    log_det_cov = log_det_dict[cluster]
                    lle = np.dot(x.reshape([1, (self.num_blocks - 1) * n]),
                                 np.dot(inv_cov_matrix, x.reshape([n * (self.num_blocks - 1), 1]))) + log_det_cov
                    LLE_all_points_clusters[point, cluster] = lle

        return LLE_all_points_clusters

    def optimize_clusters(self, computed_covariance, len_clusters, log_det_values, optRes):
        for cluster in range(self.n_clusters):
            if optRes[cluster] is None:
                # Skip clusters with no inverse cov matrix
                continue
            val = optRes[cluster].get()  # BUG: Hangs here with scipy >=1.3
            # val = optRes[cluster]
            print(f"Optimisation for Cluster #{cluster} DONE")
            # THIS IS THE SOLUTION
            S_est = Th.upperToFull(val, 0)
            X2 = S_est  # TODO: what is the point/meaning of X2?
            u, _ = np.linalg.eig(S_est)  # eigenvalues of the covariance matrix
            cov_out = np.linalg.inv(X2)

            # Store the log-det, covariance, inverse-covariance, cluster means, stacked means
            log_det_values[self.n_clusters, cluster] = np.log(np.linalg.det(cov_out))
            computed_covariance[self.n_clusters, cluster] = cov_out
            self.cluster_MRFs_[cluster] = X2
        for cluster in range(self.n_clusters):
            print(f"length of cluster {cluster} -> {len_clusters[cluster]}")

    def train_clusters(self, cluster_mean_info, cluster_mean_stacked_info,
                       X_stacked, empirical_covariances, len_clusters, n, pool,
                       clusters_arr) -> List[np.ndarray]:
        '''
        Compute the parameters [theta_1 ... theta_k] for the point assignments
        stored in clusters_arr.
        '''
        optRes = [None for i in range(self.n_clusters)]
        for cluster in range(self.n_clusters):
            cluster_length = len_clusters[cluster]  # num points in cluster
            if cluster_length != 0:
                indices = clusters_arr[cluster]
                # D_train is the current points assigned, to update the cluster
                # parameter theta
                D_train = X_stacked[indices, :]

                # Keys are tuples of (n_clusters, cluster number)
                cluster_mean_info[self.n_clusters, cluster] = D_train.mean(axis=0)[(self.window_size - 1) * n:self.window_size * n].reshape([1, n])
                cluster_mean_stacked_info[self.n_clusters, cluster] = D_train.mean(axis=0)

                # Fit a model - OPTIMIZATION
                probSize = self.window_size * n
                lamb = np.zeros((probSize, probSize)) + self.lambda_parameter
                # S_i is the empirical covariance of theta_i
                S = np.cov(np.transpose(D_train))
                empirical_covariances[cluster] = S

                solver = ADMMSolver(lamb, self.window_size, n, 1, S)
                # apply to process pool
                # tuple passed to solver (maxIters, eps_abs, eps_rel, verbose)
                optRes[cluster] = pool.apply_async(solver, (1000, 1e-6, 1e-6, False,))
                # # single-process version: (get rid of .get() call)
                # optRes[cluster] = solver(1000, 1e-6, 1e-6, False,)
        return optRes

    # def stack_data(self, X):
    #     n_samples, n_features = X.shape
    #     # TODO: Optimise using numpy funcs
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
        '''
        Stack input data into array (n_samples, n_features*window)
        Note that the timeseries X must be in ascending time order

        Rather than just looking at xt (the (n, 1) vector of at time t), we
        instead cluster a short subsequence of size w ≪ n_samples that ends at
        time t. The resultant n_features * window vector is made up of xt and
        the previous (w - 1) observations.
        '''
        def zero_rows(X, i):
            '''Set first i rows of X to zero'''
            X[:i, :] = 0
            return X
        return np.concatenate([zero_rows(np.roll(X, i, axis=0,), i)
                               for i in range(self.window_size)], axis=1)

    def log_parameters(self):
        print("lam_sparse", self.lambda_parameter)
        print("beta", self.beta)
        print("num_cluster", self.n_clusters)
        print("num stacked", self.window_size)

    def predict_clusters(self, X=None):
        '''
        Given the current trained model, predict clusters.  If the cluster
        segmentation has not been optimized yet, then this will be part of
        the iterative process.

        Args:
            numpy array of data for which to predict clusters.  Columns are
            dimensions of the data, each row is a different timestamp

        Returns:
            vector of predicted cluster for the points
        '''
        if X is not None:
            if not isinstance(X, np.ndarray):
                raise TypeError("input must be a numpy array!")
        else:
            X = self.trained_model['complete_D_train']

        # SMOOTHENING
        lle_all_points_clusters = self.smoothen_clusters(self.trained_model['cluster_mean_info'],
                                                         self.trained_model['computed_covariance'],
                                                         self.trained_model['cluster_mean_stacked_info'],
                                                         X,
                                                         self.trained_model['n_features'])

        # Update cluster points - using NEW smoothening
        clustered_points = Th.updateClusters(lle_all_points_clusters,
                                             beta=self.beta)
        return clustered_points
