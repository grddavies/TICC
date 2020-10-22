import operator
import warnings
import numpy as np
from scipy.special import logsumexp
from sklearn import mixture
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state, extmath
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import ConvergenceWarning
from joblib import Parallel, delayed

from ticclib.admm_solver import ADMMSolver


def _update_clusters(LLE_node_vals, beta=1) -> np.array:
    """Assign T samples to K clusters with switching parameter beta.

    This is equivalent to calculating the minimum cost path given node costs
    in LLE_node_vals and edge cost beta when switching clusters. Here this is
    solved using the Viterbi path algorithm.

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
        raise ValueError("beta parameter should be => 0 but got value  of %.3g"
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
        raw_vals = future_costs + lle_vals
        unadj_min = np.min(future_costs + lle_vals + beta)
        for cluster in range(k):
            future_cost_vals[i, cluster] = min(unadj_min, raw_vals[cluster])

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
    return path.astype(int)


class _TICCluster:
    """Class to represent clusters discovered using TICC

    Parameters:
    ----------
    label : int
        numeric label to identify cluster. Used for indexing.

    Attributes:
    ----------
    MRF_ : array (n*w, n*w)}, n = n_features, w = window_size
        Gaussian inverse covariance matrix that defines a Markov Random Field
        encoding the conditional dependency structure of the cluster. This is
        takes the form of a symmetric block Toeplitz matrix.

    logdet_covar : float
        Natural log of the determinant of the cluster covariance matrix.
        log(det(covar)) = -log(det(precision))

    indices : list
        Indices of points in multivariate timeseries assigned to cluster

    mean : array (n_features * window_size, 1)
        The mean value of each column of the cluster.
    """
    def __init__(self, label: int):
        self.label = label
        self.MRF_ = np.array([])
        self.indices = np.array([])
        self._S = np.array([])
        self.mean = np.array([])
        self.norm = 0.0
        self.logdet_covar = 0.0

    def __index__(self):
        return self.label

    def __len__(self):
        return len(self.indices)

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__.keys())

    def change_label(self, x):
        self.label = x
        return self

    def get_indices(self, y: np.array):
        self.indices = np.where(y == self.label)[0]
        return self

    def get_points(self, X):
        return X[self.indices, :]

    def get_S(self, X):
        """Get empirical covariance of points in cluster"""
        self._S = np.cov(self.get_points(X), rowvar=False)
        return self._S

    def update_empirical_params(self, X, y):
        self.get_indices(y)
        if len(self) > 1:
            self.mean = X[self.indices, :].mean(axis=0)
            self.get_S(X)

    def update_computed_cov(self, cov_mat):
        """Store optimized covariance value"""
        self.computed_covariance = cov_mat
        self.norm = np.linalg.norm(cov_mat)
        self.logdet_covar = extmath.fast_logdet(cov_mat)

    def lle(self, X):
        """Log-likelihood of point assignments to cluster
            `|P| * (logdet(theta) + tr(S @ theta))`
        """
        # NOTE: det(covar) = 1/det(theta) -> -logdet(covar) = logdet(theta)
        return len(self)*(np.log(np.linalg.det(self.MRF_))
                          + np.trace(self.get_S(X) @ self.MRF_))

    def split_theta(self, w) -> np.array:
        """Split the first n columns of the the theta matrix (which defines
        a MRF) into w arrays of shape (n, n). Each of these arrays represents
        the conditional independence of the n variables across the window w.
        The first array represents relationship between concurrent values of
        the n features. Each subsequent array shows how a feature relates to
        future values of the other features.
        """
        return np.split(np.split(self.MRF_, w, axis=1)[0], w)


class TICC(BaseEstimator):
    """Toeplitz inverse-covariance based clustering of multivariate timeseries.

    Parameters:
    ----------
    n_clusters : int
        Number of clusters/segments to form, and number of inverse covariance
        matrices to fit.

    window_size : int
        Size of the sliding window in samples.

    lambda_parameter : float
        Sparsity parameter which determines the sparsity level in
        the Markov random fields (MRFs) characterizing each cluster.

    beta : float
        Temporal consistency parameter. A smoothness penalty that encourages
        consecutive subsequences to be assigned to the same cluster.

    max_iter : int
        Maximum number of iterations of the TICC expectation maximization
        algorithm for a single run.

    n_jobs : int, None (default)
        The maximum number of concurrently running jobs to be run via joblib.
        None is a marker for ‘unset’ that will be interpreted as n_jobs=1
        (sequential execution) unless the call is performed under a
        parallel_backend context manager that sets another value for n_jobs.

    cluster_reassignment : float (0, 1), default to 0.2
        The proportion of points to move from a valid cluster to an empty
        cluster during ``fit``.

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

    clusters_ : list (n_clusters)
        List of _TICCluster objects discovered.

    converged_ : bool
        True when convergence was reached in ``fit``, False otherwise.
    """
    def __init__(self, *, n_clusters=5, window_size=10, lambda_parameter=11e-2,
                 beta=400, max_iter=100, n_jobs=None, cluster_reassignment=0.2,
                 random_state=None, copy_x=True, verbose=False, n_init=1):
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
        self.n_init = n_init

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
                                      random_state=random_state_,
                                      n_init=self.n_init,
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
        labels : array-like (n_samples,)
            cluster labels per point
        """
        # Get log-likelihood estimates
        LLEs = self._estimate_log_lik(X)
        # label points
        labels = _update_clusters(LLEs, beta=self.beta)
        # Update cluster indices/lengths
        for cluster in self.clusters_:
            cluster.update_empirical_params(X, labels)
        return labels

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
        n = self.n_features_in_
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
                Theta_est = optRes[index][1]
                cov_out = np.linalg.inv(Theta_est)
                # Store the log-det-covar, covariance, inverse-covariance
                cluster.MRF_ = Theta_est  # inverse-covariance
                cluster.update_computed_cov(cov_out)
            except StopIteration:
                # Cluster empty - no optimization result in list
                if self.verbose:
                    print(f"Cluster {cluster.label} was empty!")
                pass

    def _empty_cluster_assign(self, clustered_points, rng):
        """Reassign points to empty clusters"""
        p = self.cluster_reassignment
        # clusters that are not 0 as sorted by norm of their cov matrix
        valid_clusters = sorted([c for c in self.clusters_ if len(c) > 2],
                                key=operator.attrgetter('norm'),
                                reverse=True)

        # Add points to empty clusters
        counter = 0
        for cluster in self.clusters_:
            if len(cluster) < 3:
                # Select a source cluster (must be valid)
                source = self.clusters_[
                    self.clusters_.index(valid_clusters[counter])
                    ]
                counter = (counter + 1) % len(valid_clusters)
                if self.verbose:
                    print(f"Cluster {cluster.label} is empty, moving points "
                          f"from Cluster {source.label}")
                # Move random points from source to target cluster
                frac = int(len(source)*p)
                start = rng.integers(len(source) - frac)
                cluster.indices = source.indices[start:start+frac]
                # Change point labels
                clustered_points[cluster.indices] = cluster.label
                # Update the affected clusters' lengths
                source.get_indices(clustered_points)
                cluster.get_indices(clustered_points)
                # Clone source covariance parameter into cluster
                # TODO: Should the source covar be modified/scaled?
                cluster.update_computed_cov(source.computed_covariance)
                if self.verbose:
                    print(f"Cluster {cluster.label} length = {len(cluster)}\n"
                          f"Cluster {source.label} length = {len(source)}")
        return clustered_points

    def _estimate_log_lik(self, X) -> np.array:
        """Estimate the log-likelihood log P(X | Z).
        Compute the log-likelihood of each sample being assigned to each
        cluster given the current cluster parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features*window_size)
            Stacked multivariate timeseries data.

        Returns
        -------
        log_likelihoods : array, shape (n_samples, n_clusters)
            Log-likelihood estimate for each sample belonging to each cluster.
        """
        log_lik = np.zeros((len(X), self.n_clusters))
        for cluster in self.clusters_:
            X_ = X - cluster.mean
            log_lik[:, cluster] = (
                # TODO: Does sum perform better than einsum?
                # np.sum((X_@inv_cov_matrix) * X_, axis=1)
                np.einsum('ij,ij->i', X_@cluster.MRF_, X_)
                + cluster.logdet_covar
            )
        return log_lik

    def fit(self, X, y=None, sample_weight=None):
        """Compute Toeplitz Inverse Covariance-Based Clustering and
        predict labels for data.

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
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
            if sample_weight.shape != (len(X), ):
                raise ValueError("If sample weights passed, length must match "
                                 "X input length")

        # Initialization
        self.converged_ = False
        rng = np.random.default_rng(self.random_state)
        self.clusters_ = [_TICCluster(k) for k in range(self.n_clusters)]
        old_labels = self._initialize(X)

        # PERFORM TRAINING ITERATIONS
        for iters in range(1, self.max_iter+1):
            if self.verbose:
                print("Iteration %d" % iters)
            # Get empirical cluster params based on last assignments
            for cluster in self.clusters_:
                cluster.update_empirical_params(X, old_labels)
            # M-Step: Calculate optimal cluster parameters
            self._m_step(X)
            # E-step: Make assignments based on optimal parameters
            labels = self._e_step(X)

            if np.array_equal(old_labels, labels):
                # Converged - Break Early
                self.converged_ = True
                if self.verbose:
                    print("Converged - Breaking early\n\n")
                break

            # Reassign points to any empty clusters
            # NOTE: This changes empirical and optimal params of empty clusters
            old_labels = self._empty_cluster_assign(labels, rng)
        if not self.converged_:
            warnings.warn("Model did not converge after %d iterations. Try "
                          "changing model parameters or increasing `max_iter`."
                          % self.max_iter, ConvergenceWarning)
        return self

    def fit_predict(self, X, y=None, sample_weight=None):
        """Compute Toeplitz Inverse Covariance parameters for clusters data X.

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
        labels : array, shape (n_samples,)
            Cluster labels for each timepoint in X.
        """
        return self.fit(X, sample_weight=sample_weight).predict(X)

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
        X_orig = self._validate_data(X_orig, accept_sparse=False,
                                     dtype='numeric', order='C',
                                     copy=self.copy_x,
                                     ensure_min_features=2)
        t_samples, _ = X_orig.shape
        n = self.n_features_in_
        X = np.zeros((t_samples, n*self.window_size))
        for i in range(self.window_size):
            X[:, i*n:(i+1)*n] = zero_rows(np.roll(X_orig, i, axis=0,), i)
        return X

    def predict(self, X) -> np.array:
        """Predict the labels for the data in X using trained model.

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
        LLEs = self._estimate_log_lik(X)
        # label points
        labels = _update_clusters(LLEs, beta=self.beta)
        return labels

    def score_samples(self, X) -> np.array:
        """Compute the total log likelihood for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single time point.

        Returns
        -------
        log_likelihoods : array, shape (n_samples,)
            Log likelihood of each data point in X.
        """
        check_is_fitted(self)
        X = self.stack_data(X)
        return logsumexp(self._estimate_log_lik(X), axis=1)

    def score(self, X, y=None) -> float:
        """Compute the average log-likelihood per cluster assignment.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_dimensions)
            List of n_features-dimensional data points. Each row
            corresponds to a single time point.
        Returns
        -------
        log_likelihood : float
            Log likelihood of the TICC model given X.
        """
        check_is_fitted(self)
        X_stacked = self.stack_data(X)
        return logsumexp(self._estimate_log_lik(X_stacked)[self.predict(X)])

    def bic(self, X, thresh=2e-5) -> float:
        """Bayesian information criterion for the current model on the input X.

        Parameters
        ----------
        X : array of shape (n_samples, n_dimensions)

        thresh : float
            The threshold above which a parameter in the theta matrix of a
            cluster is considered `nonzero` for the purposes of calculating
            the BIC

        Returns
        -------
        bic : float
            The lower the better.
        """
        cluster_assignments = self.predict(X)
        X = self.stack_data(X)
        lle_model = sum([c.lle(X) for c in self.clusters_])
        cluster_params = [np.sum(np.abs(c.MRF_) > thresh)
                          for c in self.clusters_]
        curr_val = -1
        non_zero_params = 0
        for val in cluster_assignments:
            if val != curr_val:
                non_zero_params += cluster_params[val]
                curr_val = val
        return non_zero_params * np.log(len(X)) - 2*lle_model
