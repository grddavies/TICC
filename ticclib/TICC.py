import multiprocessing as mp
import operator
import warnings
import numpy as np
from scipy.special import logsumexp
from sklearn import mixture
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import ConvergenceWarning

from ticclib.admm_solver import ADMMSolver


def _update_clusters(LLE_node_vals, beta=1) -> np.array:
    """Compute the minimum cost path given node costs in LLE_node_vals and edge
    cost beta when swtiching to different cluster nodes.
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
    # TODO: Optimize - bottleneck in fit method
    if beta < 0:
        raise ValueError("beta parameter should be >= 0 but got value  of %.3f"
                         % beta)
    if beta == 0:
        path = LLE_node_vals.argmin(axis=1)
        return path

    (T, n_clusters) = LLE_node_vals.shape
    future_cost_vals = np.zeros(LLE_node_vals.shape)

    # compute future costs
    for i in range(T-2, -1, -1):
        j = i+1
        future_costs = future_cost_vals[j, :]
        lle_vals = LLE_node_vals[j, :]
        for cluster in range(n_clusters):
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
        Gaussian inverse covariance matrix that defines a Markov Random Field
        encoding the structural representation of the cluster. This is a
        symmetric block Toeplitz matrix

    logdet_theta : float
        log(det(MRF_))

    indices : list
        Indices of points assigned to cluster

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

    def get_mean(self, X):
        self.mean = X[self.indices, :].mean(axis=0)

    def get_logdet_theta(self):
        self.logdet_theta = np.log(np.linalg.det(self.computed_covariance))

    def get_inv_cov_matrix(self):
        self.inv_cov_matrix = np.linalg.inv(self.computed_covariance)

    def ll(self):
        """Log-likelihood of point assignments to cluster
            `|P| * (logdet(theta) + tr(S @ theta))`
        """
        return len(self)*(np.log(np.linalg.det(self.MRF_))
                          + np.trace(self._S @ self.MRF_))


class TICC(BaseEstimator):
    """Toeplitz inverse-covariance based clustering of multivariate timeseries.

    Parameters:
    ----------
    n_clusters : int
        Number of clusters/segments to form as well as number of
        inverse covariance matrices to fit.

    window_size : int
        Size of the sliding window in samples.

    lambda_parameter : float
        Sparsity parameter which determines the sparsity level in
        the Markov random fields (MRFs) characterizing each cluster.

    beta : float
        Temporal consistency parameter. A smoothness penalty that encourages
        adjacent subsequences to be assigned to the same cluster.

    max_iter : int
        Maximum number of iterations of the TICC algorithm for a single
        run.

    cluster_reassignment : int
        Number of points to reassign to empty clusters during ``fit``

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
        List of _TICCluster objects discovered.

    converged_ : bool
        True when convergence was reached in ``fit``, False otherwise.
    """
    def __init__(self, *, n_clusters=5, window_size=10,  lambda_parameter=11e-2,
                 beta=400, max_iter=1000, num_proc=1, cluster_reassignment=20,
                 random_state=None, copy_x=True):
        self.window_size = window_size
        self.n_clusters = n_clusters
        self.lambda_parameter = lambda_parameter
        self.beta = beta
        self.max_iter = max_iter
        # self.num_proc = np.clip(num_proc, 1, mp.cpu_count())
        self.cluster_reassignment = cluster_reassignment
        self.random_state = random_state
        self.copy_x = copy_x
        self.converged = False

    def _initialize(self, X) -> np.array:
        """Initialize the cluster assignments

        Parameters
        ----------
        X : array-like (n_samples, n_features*window_size)

        Returns
        -------
        clustered_points : array-like (n_samples, 1)
            cluster labels for each sample
        """
        random_state_ = check_random_state(self.random_state)
        gmm = mixture.GaussianMixture(n_components=self.n_clusters,
                                      covariance_type="full",
                                      random_state=random_state_
                                      )

        clustered_points = gmm.fit_predict(X)
        return clustered_points

    def _e_step(self, X) -> np.array:
        """E step of EM algorithm.

        Assign points in X to clusters given current cluster params.
        Update cluster lengths.

        Parameters
        ----------
        X : array-like (n_samples, n_features*window_size)

        Returns
        -------
        clustered_points : array-like (n_samples,)
            cluster labels per point
        """
        for cluster in self.clusters_:
            cluster.get_mean(X)
            cluster.get_inv_cov_matrix()
            cluster.get_logdet_theta()  # log(det(sigma2|1))

        LLEs = self._estimate_log_prob(X)

        # label points
        clustered_points = _update_clusters(LLEs, beta=self.beta)
        # Update cluster indices/lengths
        for cluster in self.clusters_:
            cluster.get_indices(clustered_points)
        return clustered_points

    def _m_step(self, X):
        """Update cluster parameters by solving Toeplitz graphical lasso problem.

        Pass cluster points to ADMM solver to optimize, then update cluster
        parameters based on current assignments
        """
        optRes = [None for i in range(self.n_clusters)]
        w = self.window_size
        n = self.n_features_in_
        for cluster in self.clusters_:
            if len(cluster) != 0:
                D_train = X[cluster.indices, :]
                # Fit a model - OPTIMIZATION
                probSize = w * n
                lamb = np.zeros((probSize, probSize)) + self.lambda_parameter
                # S is the empirical covariance of points in cluster
                S = np.cov(D_train, rowvar=False)  # NOTE: data vars in cols
                solver = ADMMSolver(lamb, w, n, 1, S)
                # solver(maxIters, eps_abs, eps_rel)
                optRes[cluster] = solver(1000, 1e-6, 1e-6)
                S_est = optRes[cluster]  # BUG: Hangs, scipy >=1.3
                # TODO: is this theta or inv(theta)? Check ADMMSolver
                theta = np.linalg.inv(S_est)

                # Store the log-det-theta, covariance, inverse-covariance
                # TODO: Check! these look wrong
                cluster._S = S
                cluster.logdet_theta = np.log(np.linalg.det(theta))
                cluster.computed_covariance = theta
                cluster.norm = np.linalg.norm(theta)
                cluster.MRF_ = S_est
        return self

    def _empty_cluster_assign(self, clustered_points) -> np.array:
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

    def _estimate_log_prob(self, X) -> np.array:
        """Estimate the log-probabilities log P(X | Z).
        Compute the log-probabilities per cluster for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features*window_size)

        Returns
        -------
        log_prob : array, shape (n_samples, n_clusters)
        """
        log_prob = np.zeros((len(X), self.n_clusters))
        for cluster in self.clusters_:
            X_ = X - cluster.mean
            inv_cov_matrix = cluster.inv_cov_matrix
            log_det_cov = cluster.logdet_theta
            log_prob[:, cluster] = (
                # TODO: Does sum(A*B, axis=1)
                # perform better than einsum?
                np.einsum('ij,ij->i',
                          X_@inv_cov_matrix,
                          X_,
                          )
                + log_det_cov
            )
        return log_prob

    def fit(self, X, y=None, sample_weight=None):
        """Compute Toeplitz Inverse Covariance-Based Clustering.

        Parameters
        ----------
        X : {array-like, sparse matrix} (n_samples, n_features) or
            (n_samples, n_features * window_size)
            Timeseries to cluster (in ascending time order).

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : Ignored
            Feature not implemented yet, present for API consistency.

        Returns
        -------
        self
        """
        if self.max_iter < 1:
            raise ValueError("Must have at least one iteration, increase "
                             "``max_iter``")
        try:
            n_samples, n_cols = X.shape
            if n_cols != self.n_features_in_ * self.window_size:
                raise ValueError("Input dimensions incorrect! Did you call "
                                 "``stack_data()`` before ``fit``?")

        except AttributeError:
            # Stack the data into a n_samples by n_features*window_size array
            X = self.stack_data(X)
            n_samples, _ = X.shape

        # Initialization - GMM to cluster
        clustered_points_old = self._initialize(X)
        self.clusters_ = [_TICCluster(k) for k in range(self.n_clusters)]
        for cluster in self.clusters_:
            cluster.get_indices(clustered_points_old)

        # PERFORM TRAINING ITERATIONS
        # TODO: joblib parallelization
        for i in range(1, self.max_iter+1):
            # print("ITERATION # %d" % i)
            # M-Step > E-Step
            clustered_points = (self._m_step(X)
                                    ._e_step(X)
                                )

            # Reassign to empty clusters
            clustered_points = self._empty_cluster_assign(clustered_points)

            if np.array_equal(clustered_points_old, clustered_points):
                self.converged = True
                # print("Converged - Breaking Early")
                break

            clustered_points_old = clustered_points.copy()
            for cluster in self.clusters_:
                cluster.get_indices(clustered_points)
        self.labels_ = clustered_points
        if not self.converged:
            warnings.warn("Model did not converge after %d iterations. Try "
                          "changing model parameters or increasing `max_iter`."
                          % self.max_iter, ConvergenceWarning)
        return self

    def stack_data(self, X_orig) -> np.array:
        """Stack input data into array (n_samples, n_features*window)
        Note that the timeseries X must be in ascending time order

        Rather than just looking at xt (the (n_features, 1) vector at time t),
        we instead cluster a short subsequence of size w ≪ n_samples that ends
        at time t. The resultant n_features * window vector is made up of xt
        and the previous (w - 1) observations.

        Parameters
        ---------
        X_orig : {array-like, sparse matrix} (n_samples, n_features)
            Timeseries to cluster (in ascending time order).

        Returns
        -------
        X : array (n_samples, n_features * window_size)
            Stacked timeseries.
        """
        def zero_rows(X, i):
            """Set first i rows of X to zero"""
            X[:i, :] = 0
            return X
        X_orig = self._validate_data(X_orig, accept_sparse='csr',
                                     dtype=[np.float64, np.float32],
                                     order='C', copy=self.copy_x,
                                     accept_large_sparse=False)
        n = self.n_features_in_
        X = np.zeros((X_orig.shape[0], n*self.window_size))
        for i in range(self.window_size):
            X[:, i*n:(i+1)*n] = zero_rows(np.roll(X_orig, i, axis=0,), i)
        return X

    def predict(self, X) -> np.array:
        """Predict the labels for the data samples in X using trained model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features * window_size)
            List of (n_features*window_size)-dimensional data points. Each row
            corresponds to a time window of window_size samples.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        check_is_fitted(self)
        LLEs = self._estimate_log_prob(X)
        # label points
        labels = _update_clusters(LLEs, beta=self.beta)
        return labels

    def score_samples(self, X) -> np.array:
        """Compute the weighted log probabilities for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        Returns
        -------
        log_prob : array, shape (n_samples,)
            Log probabilities of each data point in X.
        """
        check_is_fitted(self)
        return logsumexp(self._estimate_log_prob(X), axis=1)

    def score(self, X, y=None) -> float:
        """Compute the per-sample average log-likelihood of the given data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        Returns
        -------
        log_likelihood : float
            Log likelihood of the TICC model given X.
        """
        return self.score_samples(X).mean()

    def bic(self, X, threshold=2e-5) -> float:
        """Bayesian information criterion for the current model on the input X.

        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)

        Returns
        -------
        bic : float
            The lower the better.
        """
        lle_model = 0
        clusterParams = {}
        for c in self.clusters_:
            lle_model += c.ll()
            clusterParams[c.label] = np.sum(np.abs(c.MRF_) > threshold)
        curr_val = -1
        non_zero_params = 0
        for val in self.labels_:
            if val != curr_val:
                non_zero_params += clusterParams[val]
                curr_val = val
        return non_zero_params * np.log(len(X)) - 2*lle_model
