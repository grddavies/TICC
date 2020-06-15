import operator
import warnings
from joblib import Parallel, delayed
import numpy as np
from scipy.special import logsumexp
from sklearn import mixture
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import ConvergenceWarning

from ticclib.admm_solver import ADMMSolver


def _update_clusters(LLE_node_vals, beta=1) -> np.array:
    """Assign T samples to K clusters with switching parameter beta.

    This is equivalent to calculating the minimum cost path given node costs
    in LLE_node_vals and edge cost beta when switching clusters.

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
        raise ValueError("beta parameter should be >= 0 but got value  of %.3g"
                         % beta)
    elif beta == 0:
        path = LLE_node_vals.argmin(axis=1)
        return path

    T, k = LLE_node_vals.shape
    future_cost_vals = np.zeros(LLE_node_vals.shape)

    # compute future costs
    for i in range(T-2, -1, -1):
        j = i+1
        future_costs = future_cost_vals[j, :]
        lle_vals = LLE_node_vals[j, :]
        for cluster in range(k):
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
        encoding the conditional dependency structre of the cluster. This is
        takes the form of a symmetric block Toeplitz matrix.

    logdet_theta : float
        log(det(MRF_))

    indices : list
        Indices of points assigned to cluster

    inv_cov_matrix : array
        how is diff from MRF?

    mean : array (n_features * window_size, 1)
        The mean value of each column of the cluster.
    """
    def __init__(self, label: int):
        self.label = label

    def __index__(self):
        return self.label

    def __len__(self):
        return len(self.indices)

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)

    def get_indices(self, y: np.array) -> np.array:
        self.indices = np.where(y == self.label)[0]

    def get_points(self, X):
        return X[self.indices, :]

    def get_S(self, X):
        """Get covariance of points in cluster"""
        self._S = np.cov(self.get_points(X), rowvar=False)
        return self._S

    def get_mean(self, X):
        self.mean = X[self.indices, :].mean(axis=0)

    def get_logdet_theta(self):
        self.logdet_theta = np.log(np.linalg.det(self.computed_covariance))

    def get_inv_cov_matrix(self):
        self.inv_cov_matrix = np.linalg.inv(self.computed_covariance)

    def lle(self, X):
        """Log-likelihood of point assignments to cluster
            `|P| * (logdet(theta) + tr(S @ theta))`
        """
        return len(self)*(np.log(np.linalg.det(self.MRF_))
                          + np.trace(self.get_S(X) @ self.MRF_))


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
        Maximum number of iterations of the TICC E-M algorithm for a single run.

    cluster_reassignment : int
        Number of points to reassign to empty clusters during ``fit``

    random_state : int, RandomState instance or None, optional default=None
        A pseudo random number generator used for the initialization of the
        cluster initialisation by gaussian mixture model and to select a point
        to be added to empty clusters during cluster training. Use an int to
        make the randomness deterministic.

    copy_x : bool, default to True
        If copy_x is True (default), then the original data is
        not modified. If False, the original data is modified, and put back
        before the function returns, but small numerical differences may be
        introduced by subtracting and then adding the data mean. Note that if
        the original data is not C-contiguous, a copy will be made even if
        copy_x is False. If the original data is sparse, but not in CSR format,
        a copy will be made even if copy_x is False.

    verbose : bool, default to False
        If true print out iteration number and any empty cluster reassignments.

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
    def __init__(self, *, n_clusters=5, window_size=10, lambda_parameter=11e-2,
                 beta=400, max_iter=100, n_jobs=None, cluster_reassignment=0.2,
                 random_state=None, copy_x=True, verbose=True):
        self.window_size = window_size
        self.n_clusters = n_clusters
        self.lambda_parameter = lambda_parameter
        self.beta = beta
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.cluster_reassignment = cluster_reassignment
        self.random_state = random_state
        self.copy_x = copy_x
        self.verbose = verbose

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

        Parameters
        ----------
        X : array-like (n_samples, n_features*window_size)

        Returns
        -------
        clustered_points : array-like (n_samples,)
            cluster labels per point
        """
        # Update cluster parameters
        for cluster in self.clusters_:
            cluster.get_mean(X)
            cluster.get_inv_cov_matrix()
            cluster.get_logdet_theta()

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
        def solver(cluster, lam, w, n):
            """Call ADMMSolver instance.
            ADMMSolver args are (maxIters, eps_abs, eps_rel)
            """
            solver = ADMMSolver(lam, w, n, 1, cluster._S)
            return (cluster.label, solver(1000, 1e-6, 1e-6))
        w = self.window_size
        n = self.n_features
        lambda_ = np.zeros((w*n, w*n)) + self.lambda_parameter
        optRes = Parallel(n_jobs=self.n_jobs)(
            delayed(solver)(cluster, lambda_, w, n)
            for cluster in self.clusters_ if len(cluster) > 1)
        # # Non-parallel:
        # optRes = [solver(cluster, lambda_, w, n)
        #           for cluster in self.clusters_ if len(cluster) > 1]

        # Update cluster parameters
        for cluster in self.clusters_:
            try:
                index = next((i for i, v in enumerate(optRes)
                              if v[0] == cluster.label))
                S_est = optRes[index][1]
                cov_out = np.linalg.inv(S_est)
                # Store the log-det-theta, covariance, inverse-covariance
                cluster.logdet_theta = np.log(np.linalg.det(cov_out))
                cluster.computed_covariance = cov_out
                cluster.norm = np.linalg.norm(cov_out)
                cluster.MRF_ = S_est
            except StopIteration:
                if self.verbose:
                    print(f"Cluster {cluster.label} was empty!")
                # Cluster empty - no optimization result in list
                pass
        return self

    def _empty_cluster_assign(self, clustered_points, rng) -> np.array:
        """Reassign points to empty clusters"""
        p = self.cluster_reassignment
        # clusters that are not 0 as sorted by norm of their cov matrix
        valid_clusters = sorted([c for c in self.clusters_ if len(c) > 1],
                                key=operator.attrgetter('norm'),
                                reverse=True)

        # Add points to empty clusters
        counter = 0
        for cluster in self.clusters_:
            if len(cluster) < 2:
                # Select a source cluster
                source = self.clusters_[
                    self.clusters_.index(valid_clusters[counter])
                    ]
                counter = (counter + 1) % len(valid_clusters)
                if self.verbose:
                    print(f"Cluster {cluster.label} is empty, moving points "
                          f"from Cluster {source.label}")
                # Move random points from source to target cluster
                cluster.indices = rng.choice(source.indices,
                                             size=int(len(source)*p)
                                             )
                # Change point labels
                clustered_points[cluster.indices] = cluster.label
                # Update the affected clusters' lengths
                source.get_indices(clustered_points)
                cluster.get_indices(clustered_points)
                # Clone source covariance parameter into cluster
                cluster.computed_covariance = (source.computed_covariance
                                               + rng.random())
                if self.verbose:
                    print(f"Cluster {cluster.label} length = {len(cluster)} \n"
                          f"Cluster {source.label} length = {len(source)}")
        return clustered_points

    def _estimate_log_prob(self, X) -> np.array:
        """Estimate the log-probabilities log P(X | Z).
        Compute the log-probabilities per cluster for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features*window_size)
            Stacked multivariate timeseries data.

        Returns
        -------
        log_prob : array, shape (n_samples, n_clusters)
            Log-likelihood estimate for each sample belonging to each cluster.
        """
        log_prob = np.zeros((len(X), self.n_clusters))
        for cluster in self.clusters_:
            X_ = X - cluster.mean
            inv_cov_matrix = cluster.inv_cov_matrix
            log_det_cov = cluster.logdet_theta
            log_prob[:, cluster] = (
                # TODO: Does sum(A*B, axis=1) perform better than einsum?
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
            The fitted estimator
        """
        if self.max_iter < 1:
            raise ValueError("Must have at least one iteration, increase "
                             "``max_iter``")
        X = self.stack_data(X)

        X = self._validate_data(X, accept_sparse='csr',
                                dtype=[np.float64, np.float32],
                                order='C', copy=self.copy_x,
                                accept_large_sparse=False)
        # validate data creates .n_features_in_ which is = n_features * w
        self.n_features = self.n_features_in_//self.window_size
        self.n_samples, _ = X.shape

        # Initialization - GMM to cluster
        rng = np.random.default_rng(self.random_state)
        clustered_points_old = self._initialize(X)
        self.clusters_ = [_TICCluster(k) for k in range(self.n_clusters)]
        for cluster in self.clusters_:
            cluster.get_indices(clustered_points_old)
            cluster.get_S(X)

        # PERFORM TRAINING ITERATIONS
        for iters in range(1, self.max_iter+1):
            if self.verbose:
                print("Iteration %d" % iters)
            # M-Step > E-Step
            clustered_points = (self._m_step(X)
                                    ._e_step(X)
                                )

            if np.array_equal(clustered_points_old, clustered_points):
                # Converged - Break Early
                break

            # Reassign to empty clusters
            clustered_points = self._empty_cluster_assign(clustered_points,
                                                          rng)
            clustered_points_old = clustered_points.copy()
            for cluster in self.clusters_:
                cluster.get_indices(clustered_points)
                cluster.get_mean(X)
                cluster.get_S(X)
        self.labels_ = clustered_points.astype(int)
        if iters == self.max_iter:
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
            Timeseries to cluster (in ascending time order). For best
            performance, standardize observations before fitting TICC.

        Returns
        -------
        X : array (n_samples, n_features * window_size)
            Stacked timeseries.
        """
        def zero_rows(X, i):
            """Set first i rows of X to zero"""
            X[:i, :] = 0
            return X
        t_samples, n = X_orig.shape
        X = np.zeros((t_samples, n*self.window_size))
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
        X = self.stack_data(X)
        LLEs = self._estimate_log_prob(X)
        # label points
        labels = _update_clusters(LLEs, beta=self.beta)
        return labels

    def score_samples(self, X) -> np.array:
        """Compute the log probabilities for each sample.

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
        X = self.stack_data(X)
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

    def bic(self, X, thresh=2e-5) -> float:
        """Bayesian information criterion for the current model on the input X.

        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)

        Returns
        -------
        bic : float
            The lower the better.
        """
        X = self.stack_data(X)
        lle_model = sum([c.lle(X) for c in self.clusters_])
        clusterParams = [np.sum(np.abs(c.MRF_) > thresh)
                         for c in self.clusters_]
        curr_val = -1
        non_zero_params = 0
        for val in self.labels_:
            if val != curr_val:
                non_zero_params += clusterParams[val]
                curr_val = val
        return non_zero_params * np.log(len(X)) - 2*lle_model
