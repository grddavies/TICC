import numpy as np
import collections
import os
import errno
import matplotlib
import matplotlib.pyplot as plt
from sklearn import mixture
from sklearn.cluster import KMeans
# from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
import pandas as pd
import multiprocessing as mp

import src.TICC_helper as Th
from src.admm_solver import ADMMSolver
matplotlib.use('Agg')


class TICC:
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

    threshold : float?
        convergence threshold

    write_out_file : bool
        if true, prefix_string is output file dir

    prefix_string : str
        output directory if necessary

    cluster_reassignment : int
        number of points to reassign to a 0 cluster

    random_state : int, RandomState instance, default=None
        A pseudo random number generator used for the initialization of the
        lobpcg eigen vectors decomposition when ``eigen_solver='amg'`` and by
        the K-Means initialization. Use an int to make the randomness
        deterministic.
    """
    def __init__(self, *, n_clusters=5, window_size=10,  lambda_parameter=11e-2,
                 beta=400, max_iter=1000, threshold=2e-5, write_out_file=False,
                 prefix_string="", num_proc=1, compute_BIC=False, cluster_reassignment=20,
                 random_state=None):
        self.window_size = window_size
        self.n_clusters = n_clusters
        self.lambda_parameter = lambda_parameter
        self.switch_penalty = beta
        self.max_iter = max_iter
        self.threshold = threshold
        self.write_out_file = write_out_file
        self.prefix_string = prefix_string
        self.num_proc = np.clip(num_proc, 1, mp.cpu_count())
        self.compute_BIC = compute_BIC
        self.cluster_reassignment = cluster_reassignment
        self.num_blocks = self.window_size + 1
        self.random_state = random_state
        pd.set_option('display.max_columns', 500)
        np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
        np.random.seed(self.random_state)

    def fit(self, X):
        """Compute Toeplitz Inverse Covariance-Based Clustering.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster.

        Returns
        -------
        clustered_points : ndarray of shape (n_samples[-window + 1?], 1)
            cluster assignments of each sample

        cluster_MRF : dict {cluster_number: ndarray of shape (nw, nw)}
            Block Toeplitz inverse covariance matrices that define clusters
        """
        assert self.max_iter > 0  # must have at least one iteration
        self.log_parameters()

        # Check input format  # requires later sklearn - check unittest pass
        # X = self._validate_data(X, accept_sparse='csr',
        #                         dtype=[np.float64, np.float32],
        #                         order='C', copy=self.copy_x,
        #                         accept_large_sparse=False)
        time_series_rows_size, time_series_col_size = X.shape

        # The basic folder to be created
        str_NULL = self.prepare_out_directory()

        # Train test split
        training_indices = Th.getTrainTestSplit(time_series_rows_size,
                                                self.num_blocks,
                                                self.window_size)
        num_train_points = len(training_indices)

        # Stack the training data
        complete_D_train = self.stack_training_data(X,
                                                    time_series_col_size,
                                                    num_train_points,
                                                    training_indices)

        # Initialization

        # Gaussian Mixture
        gmm = mixture.GaussianMixture(n_components=self.n_clusters,
                                      covariance_type="full",
                                    #   random_state=102,
                                      )
        gmm.fit(complete_D_train)
        clustered_points = gmm.predict(complete_D_train)
        gmm_clustered_pts = clustered_points + 0
        # K-means
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(complete_D_train)
        clustered_points_kmeans = kmeans.labels_

        train_cluster_inverse = {}
        log_det_values = {}  # log dets of the thetas
        computed_covariance = {}
        cluster_mean_info = {}
        cluster_mean_stacked_info = {}
        old_clustered_points = None  # points from last iteration

        empirical_covariances = {}

        # PERFORM TRAINING ITERATIONS
        pool = mp.Pool(processes=self.num_proc)  # multi-threading
        # Zeroth training iteration
        print("\n\n\nITERATION # 0")

        # Get the train and test points
        train_clusters_arr = collections.defaultdict(list)  # {cluster: [point indices]}
        for point, cluster_num in enumerate(clustered_points):
            train_clusters_arr[cluster_num].append(point)

        len_train_clusters = {k: len(train_clusters_arr[k]) for k in range(self.n_clusters)}

        # train_clusters holds the indices in complete_D_train
        # for each of the clusters
        opt_res = self.train_clusters(cluster_mean_info,
                                      cluster_mean_stacked_info,
                                      complete_D_train,
                                      empirical_covariances,
                                      len_train_clusters,
                                      time_series_col_size,
                                      pool,
                                      train_clusters_arr)

        self.optimize_clusters(computed_covariance,
                               len_train_clusters,
                               log_det_values,
                               opt_res,
                               train_cluster_inverse)

        # update old computed covariance
        old_computed_covariance = computed_covariance

        print("UPDATED THE OLD COVARIANCE")

        self.trained_model = {'cluster_mean_info': cluster_mean_info,
                              'computed_covariance': computed_covariance,
                              'cluster_mean_stacked_info': cluster_mean_stacked_info,
                              'complete_D_train': complete_D_train,
                              'time_series_col_size': time_series_col_size}
        clustered_points = self.predict_clusters()

        # recalculate lengths
        new_train_clusters = collections.defaultdict(list)  # {cluster: [point indices]}
        for point, cluster in enumerate(clustered_points):
            new_train_clusters[cluster].append(point)

        len_new_train_clusters = {k: len(new_train_clusters[k]) for k in range(self.n_clusters)}

        before_empty_cluster_assign = clustered_points.copy()

        for cluster_num in range(self.n_clusters):
            print("length of cluster #", cluster_num, "-------->", sum([x == cluster_num for x in clustered_points]))

        self.write_plot(clustered_points, str_NULL, training_indices)

        # TEST SETS STUFF
        # LLE + swtiching_penalty
        # Segment length
        # Create the F1 score from the graphs from k-means and GMM
        # Get the train and test points
        train_confusion_matrix_EM = Th.compute_confusion_matrix(self.n_clusters,
                                                                clustered_points,
                                                                training_indices)
        train_confusion_matrix_GMM = Th.compute_confusion_matrix(self.n_clusters,
                                                                 gmm_clustered_pts,
                                                                 training_indices)
        train_confusion_matrix_kmeans = Th.compute_confusion_matrix(self.n_clusters,
                                                                    clustered_points_kmeans,
                                                                    training_indices)
        # compute the matchings
        matching_EM, matching_GMM, matching_Kmeans = self.compute_matches(train_confusion_matrix_EM,
                                                                          train_confusion_matrix_GMM,
                                                                          train_confusion_matrix_kmeans)

        print("\n")

        old_clustered_points = before_empty_cluster_assign

        for iters in range(1, self.max_iter):
            print("\n\n\nITERATION #", iters)
            # Get the train and test points
            train_clusters_arr = collections.defaultdict(list)  # {cluster: [point indices]}
            for point, cluster_num in enumerate(clustered_points):
                train_clusters_arr[cluster_num].append(point)

            len_train_clusters = {k: len(train_clusters_arr[k]) for k in range(self.n_clusters)}

            # train_clusters holds the indices in complete_D_train
            # for each of the clusters
            opt_res = self.train_clusters(cluster_mean_info, cluster_mean_stacked_info, complete_D_train,
                                          empirical_covariances, len_train_clusters, time_series_col_size, pool,
                                          train_clusters_arr)

            self.optimize_clusters(computed_covariance, len_train_clusters, log_det_values, opt_res,
                                   train_cluster_inverse)

            # update old computed covariance
            old_computed_covariance = computed_covariance

            print("UPDATED THE OLD COVARIANCE")

            self.trained_model = {'cluster_mean_info': cluster_mean_info,
                                  'computed_covariance': computed_covariance,
                                  'cluster_mean_stacked_info': cluster_mean_stacked_info,
                                  'complete_D_train': complete_D_train,
                                  'time_series_col_size': time_series_col_size}
            clustered_points = self.predict_clusters()

            # recalculate lengths
            new_train_clusters = collections.defaultdict(list)  # {cluster: [point indices]}
            for point, cluster in enumerate(clustered_points):
                new_train_clusters[cluster].append(point)

            len_new_train_clusters = {k: len(new_train_clusters[k]) for k in range(self.n_clusters)}

            before_empty_cluster_assign = clustered_points.copy()

            cluster_norms = [(np.linalg.norm(old_computed_covariance[self.n_clusters, i]), i) for i in
                                range(self.n_clusters)]
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
                    start_point = np.random.choice(
                        new_train_clusters[cluster_selected])  # random point number from that cluster
                    for i in range(0, self.cluster_reassignment):
                        # put cluster_reassignment points from point_num in this cluster
                        point_to_move = start_point + i
                        if point_to_move >= len(clustered_points):
                            break
                        clustered_points[point_to_move] = cluster_num
                        computed_covariance[self.n_clusters, cluster_num] = old_computed_covariance[
                            self.n_clusters, cluster_selected]
                        cluster_mean_stacked_info[self.n_clusters, cluster_num] = complete_D_train[
                                                                                            point_to_move, :]
                        cluster_mean_info[self.n_clusters, cluster_num] \
                            = complete_D_train[point_to_move, :][
                                (self.window_size - 1) * time_series_col_size:self.window_size * time_series_col_size]

            for cluster_num in range(self.n_clusters):
                print("length of cluster #", cluster_num, "-------->", sum([x == cluster_num for x in clustered_points]))

            self.write_plot(clustered_points, str_NULL, training_indices)

            # TEST SETS STUFF
            # LLE + swtiching_penalty
            # Segment length
            # Create the F1 score from the graphs from k-means and GMM
            # Get the train and test points
            train_confusion_matrix_EM = Th.compute_confusion_matrix(self.n_clusters, clustered_points,
                                                                    training_indices)
            train_confusion_matrix_GMM = Th.compute_confusion_matrix(self.n_clusters, gmm_clustered_pts,
                                                                     training_indices)
            train_confusion_matrix_kmeans = Th.compute_confusion_matrix(self.n_clusters, clustered_points_kmeans,
                                                                        training_indices)
            # compute the matchings
            matching_EM, matching_GMM, matching_Kmeans = self.compute_matches(train_confusion_matrix_EM,
                                                                              train_confusion_matrix_GMM,
                                                                              train_confusion_matrix_kmeans)

            print("\n")

            if np.array_equal(old_clustered_points, clustered_points):
                print("\n\n\n\nConverged - Breaking Early")
                break
            old_clustered_points = before_empty_cluster_assign
            # end of training

        if pool is not None:
            pool.close()
            pool.join()
        train_confusion_matrix_EM = Th.compute_confusion_matrix(self.n_clusters, clustered_points,
                                                                training_indices)
        train_confusion_matrix_GMM = Th.compute_confusion_matrix(self.n_clusters, gmm_clustered_pts,
                                                                 training_indices)
        train_confusion_matrix_kmeans = Th.compute_confusion_matrix(self.n_clusters, clustered_points_kmeans,
                                                                    training_indices)

        self.compute_f_score(matching_EM, matching_GMM, matching_Kmeans, train_confusion_matrix_EM,
                             train_confusion_matrix_GMM, train_confusion_matrix_kmeans)

        if self.compute_BIC:
            bic = Th.computeBIC(self.n_clusters,
                                time_series_rows_size,
                                clustered_points,
                                train_cluster_inverse,
                                empirical_covariances)
            return clustered_points, train_cluster_inverse, bic

        return clustered_points, train_cluster_inverse

    def compute_f_score(self, matching_EM, matching_GMM, matching_Kmeans, train_confusion_matrix_EM,
                        train_confusion_matrix_GMM, train_confusion_matrix_kmeans):
        f1_EM_tr = -1  # computeF1_macro(train_confusion_matrix_EM, matching_EM, num_clusters)
        f1_GMM_tr = -1  # computeF1_macro(train_confusion_matrix_GMM, matching_GMM, num_clusters)
        f1_kmeans_tr = -1  # computeF1_macro(train_confusion_matrix_kmeans, matching_Kmeans, num_clusters)
        print("\n\n")
        print("TRAINING F1 score:", f1_EM_tr, f1_GMM_tr, f1_kmeans_tr)
        correct_e_m = 0
        correct_g_m_m = 0
        correct_k_means = 0
        for cluster in range(self.n_clusters):
            matched_cluster__e_m = matching_EM[cluster]
            matched_cluster__g_m_m = matching_GMM[cluster]
            matched_cluster__k_means = matching_Kmeans[cluster]

            correct_e_m += train_confusion_matrix_EM[cluster, matched_cluster__e_m]
            correct_g_m_m += train_confusion_matrix_GMM[cluster, matched_cluster__g_m_m]
            correct_k_means += train_confusion_matrix_kmeans[cluster, matched_cluster__k_means]

    def compute_matches(self, train_confusion_matrix_EM, train_confusion_matrix_GMM, train_confusion_matrix_kmeans):
        matching_Kmeans = Th.find_matching(train_confusion_matrix_kmeans)
        matching_GMM = Th.find_matching(train_confusion_matrix_GMM)
        matching_EM = Th.find_matching(train_confusion_matrix_EM)
        correct_e_m = 0
        correct_g_m_m = 0
        correct_k_means = 0
        for cluster in range(self.n_clusters):
            matched_cluster_e_m = matching_EM[cluster]
            matched_cluster_g_m_m = matching_GMM[cluster]
            matched_cluster_k_means = matching_Kmeans[cluster]

            correct_e_m += train_confusion_matrix_EM[cluster, matched_cluster_e_m]
            correct_g_m_m += train_confusion_matrix_GMM[cluster, matched_cluster_g_m_m]
            correct_k_means += train_confusion_matrix_kmeans[cluster, matched_cluster_k_means]
        return matching_EM, matching_GMM, matching_Kmeans

    def write_plot(self, clustered_points, str_NULL, training_indices):
        # Save a figure of segmentation
        plt.figure()
        plt.plot(training_indices[0:len(clustered_points)], clustered_points, color="r")  # ,marker = ".",s =100)
        plt.ylim((-0.5, self.n_clusters + 0.5))
        if self.write_out_file:
            plt.savefig(f"{str_NULL}TRAINING_EM_lam_sparse={self.lambda_parameter}_switch_penalty={self.switch_penalty}.jpg")
        plt.close("all")
        print("Done writing the figure")

    def smoothen_clusters(self, cluster_mean_info, computed_covariance,
                          cluster_mean_stacked_info, complete_D_train, n):
        clustered_points_len = len(complete_D_train)
        inv_cov_dict = {}  # cluster to inv_cov
        log_det_dict = {}  # cluster to log_det
        for cluster in range(self.n_clusters):
            cov_matrix = computed_covariance[self.n_clusters, cluster][0:(self.num_blocks - 1) * n,
                                                                       0:(self.num_blocks - 1) * n]
            inv_cov_matrix = np.linalg.inv(cov_matrix)
            log_det_cov = np.log(np.linalg.det(cov_matrix))  # log(det(sigma2|1))
            inv_cov_dict[cluster] = inv_cov_matrix
            log_det_dict[cluster] = log_det_cov
        # For each point compute the LLE
        print("beginning the smoothening ALGORITHM")
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

    def optimize_clusters(self, computed_covariance, len_train_clusters, log_det_values, optRes, train_cluster_inverse):
        for cluster in range(self.n_clusters):
            if optRes[cluster] is None:
                continue
            val = optRes[cluster].get()  # TODO: Hangs here with scipy >=1.3
            print(f"Optimisation for Cluster #{cluster} DONE")
            # THIS IS THE SOLUTION
            S_est = Th.upperToFull(val, 0)
            X2 = S_est
            u, _ = np.linalg.eig(S_est)
            cov_out = np.linalg.inv(X2)

            # Store the log-det, covariance, inverse-covariance, cluster means, stacked means
            log_det_values[self.n_clusters, cluster] = np.log(np.linalg.det(cov_out))
            computed_covariance[self.n_clusters, cluster] = cov_out
            train_cluster_inverse[cluster] = X2
        for cluster in range(self.n_clusters):
            print(f"length of cluster {cluster} -> {len_train_clusters[cluster]}")

    def train_clusters(self, cluster_mean_info, cluster_mean_stacked_info, complete_D_train, empirical_covariances,
                       len_train_clusters, n, pool, train_clusters_arr) -> list:
        optRes = [None for i in range(self.n_clusters)]
        for cluster in range(self.n_clusters):
            cluster_length = len_train_clusters[cluster]
            if cluster_length != 0:
                size_blocks = n
                indices = train_clusters_arr[cluster]
                D_train = np.zeros([cluster_length, self.window_size * n])
                for i in range(cluster_length):
                    point = indices[i]
                    D_train[i, :] = complete_D_train[point, :]

                cluster_mean_info[self.n_clusters, cluster] = np.mean(D_train, axis=0)[
                                                                      (
                                                                          self.window_size - 1) * n:self.window_size * n].reshape(
                    [1, n])
                cluster_mean_stacked_info[self.n_clusters, cluster] = np.mean(D_train, axis=0)
                # Fit a model - OPTIMIZATION
                probSize = self.window_size * size_blocks
                lamb = np.zeros((probSize, probSize)) + self.lambda_parameter
                S = np.cov(np.transpose(D_train))
                empirical_covariances[cluster] = S

                rho = 1
                solver = ADMMSolver(lamb, self.window_size, size_blocks, 1, S)
                # apply to process pool
                optRes[cluster] = pool.apply_async(solver, (1000, 1e-6, 1e-6, False,))
        return optRes

    def stack_training_data(self, data, n, num_train_points, training_indices):
        complete_D_train = np.zeros([num_train_points, self.window_size * n])
        for i in range(num_train_points):
            for k in range(self.window_size):
                if i + k < num_train_points:
                    idx_k = training_indices[i + k]
                    complete_D_train[i][k * n:(k + 1) * n] = data[idx_k][0:n]
        return complete_D_train

    def prepare_out_directory(self):
        str_NULL = self.prefix_string + "lam_sparse=" + str(self.lambda_parameter) + "maxClusters=" + str(
            self.n_clusters + 1) + "/"
        if not os.path.exists(os.path.dirname(str_NULL)):
            try:
                os.makedirs(os.path.dirname(str_NULL), exist_ok=True)
            except OSError as exc:  # Guard against race condition of path already existing
                if exc.errno != errno.EEXIST:
                    raise

        return str_NULL

    def log_parameters(self):
        print("lam_sparse", self.lambda_parameter)
        print("switch_penalty", self.switch_penalty)
        print("num_cluster", self.n_clusters)
        print("num stacked", self.window_size)

    def predict_clusters(self, test_data=None):
        '''
        Given the current trained model, predict clusters.  If the cluster segmentation has not been optimized yet,
        than this will be part of the interative process.

        Args:
            numpy array of data for which to predict clusters.  Columns are dimensions of the data, each row is
            a different timestamp

        Returns:
            vector of predicted cluster for the points
        '''
        if test_data is not None:
            if not isinstance(test_data, np.ndarray):
                raise TypeError("input must be a numpy array!")
        else:
            test_data = self.trained_model['complete_D_train']

        # SMOOTHENING
        lle_all_points_clusters = self.smoothen_clusters(self.trained_model['cluster_mean_info'],
                                                         self.trained_model['computed_covariance'],
                                                         self.trained_model['cluster_mean_stacked_info'],
                                                         test_data,
                                                         self.trained_model['time_series_col_size'])

        # Update cluster points - using NEW smoothening
        clustered_points = Th.updateClusters(lle_all_points_clusters, switch_penalty=self.switch_penalty)

        return clustered_points