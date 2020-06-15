import numpy as np
from networkx import nx


class RandomData:
    """Class to synthesise random multivariate time-series data.

    Attributes:
    ----------
    seed : int or (default) None
        Seed for random number generator.

    n_features : int, defaults to 5
        number of features in random data to be generated.

    window_size : int, defaults to 8
        Number of samples over with past points can exert an influence on
        future points. Ie the span over which inter-time correlations between
        features are considered.

    sparsity : float, (0, 1) defaults to 0.2
        The sparsity of the conditional independence networks generated.
    """
    def __init__(self, seed=None, n_features=5, window_size=8,
                 sparsity=0.2):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.n_features = n_features
        self.window_size = window_size
        self.sparsity = sparsity
        self.clusters = []

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)

    def genInvCov(self, low=0.3, upper=0.6, p=0.2, symmetric=True) -> np.array:
        """Generate inverse covariance matrices for n_features

        Parameters
        ----------
        low : float, default = 0.3
            Lower bound of inverse covariance values between features.

        upper : float, default = 0.6
            Upper bound of inverse covariance values between features.

        p : float > 0, default = 0.2
            Probability of edge between nodes in random graph, ie inverse
        covariance matrix sparsity.

        Returns
        -------
        S : array (n_features, n_features)
            Randomly generated covariance matrix.
        """
        n = self.n_features
        S = np.zeros((n, n))
        seed = int(self.rng.integers(1000))
        G = nx.erdos_renyi_graph(n, p, seed=seed)
        # Fill S with random values
        for e in G.edges:
            value = ((self.rng.integers(2)-0.5)
                     * 2 * (low+(upper-low)*self.rng.random()))
            S[e[0], e[1]] = value
        if symmetric:
            S = S + S.T
        return S

    def genRandInv(self, low=0.3, upper=0.6, p=0.2) -> np.array:
        n = self.n_features
        S = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if self.rng.random() < p:
                    value = ((self.rng.integers(2) - 0.5)*2
                             * (low + (upper-low)*self.rng.random()))
                    S[i, j] = value
        return S

    def GenerateInverse(self) -> np.array:
        """Generate block Toeplitz inverse covariance matrix.

        Each block is (`n_features`*`n_features`)
        Number of blocks is equal to window size

        Returns
        -------
        Theta : array (n * w, n * w)
            Block Toeplitz inverse covariance matrix
        """
        n = self.n_features
        w = self.window_size
        p = self.sparsity
        Theta = np.zeros((n*w, n*w))
        # w independent n*n matrices of independent Erdos-Renyi directed random
        # graphs with probability p of being selected.
        blocks = [self.genRandInv(p=p) for i in range(w)]
        # A0 block has symmetry enforced
        blocks[0] = self.genInvCov(p=p)
        for i in range(w):
            for j in range(w):
                block_num = np.abs(i - j)
                if i > j:
                    Theta[i*n:(i+1)*n, j*n:(j+1)*n] = blocks[block_num]
                else:
                    Theta[i*n:(i+1)*n, j*n:(j+1)*n] = blocks[block_num].T
        # Ensure Theta is Positive Definite (Paper step 4)
        lambda_ = np.abs(np.min(np.linalg.eigvals(Theta)))
        return Theta + (0.1 + lambda_)*np.identity(n*w)

    def generate_cluster_params(self, k_clusters):
        self.clusters = [self.GenerateInverse() for k in range(k_clusters)]

    def generate_points(self, t_samples, labels, break_points, recycle=False
                        ) -> np.array:
        """Generate n-dimensional timeseries coming from k states, which are
        represented by partial correlation matrices.

        Parameters
        ----------
        t_samples : int
            Number of samples of n-dimensional data to generate.

        labels : list of integers
            List of up to k labels to be attributed to each sample.

        break_points : list of integers
            points at which the sample labels change

        recycle : bool, defaults to False
            Reuse existing cluster parameters  if true. Requires prior call of
            `generate_cluster_params` to set `clusters` attribute.
        Returns
        -------
        X : array (t_samples, n_features*window_size)
            Synthetic multivariate timeseries data in n-dimensions.

        y : array (t_samples, 1)
            Vector containing the cluster labels for each time step.
        """
        # Check last break point
        if break_points[-1] != t_samples:
            raise ValueError("Last break at %d, there are %d points in "
                             "timeseries."
                             % (max(break_points), t_samples))

        # Generate/retrieve cluster parameters
        k_clusters = len(set(labels))
        if recycle:
            if len(self.clusters) < k_clusters:
                raise ValueError("Not enough clusters! %d Required to generate"
                                 " points but only %d in self.clusters"
                                 % (k_clusters, len(self.clusters)))
        else:
            self.clusters = [self.GenerateInverse() for k in range(k_clusters)]
        cluster_inverses = self.clusters
        cluster_covariances = [
            np.linalg.inv(k) for k in cluster_inverses
            ]
        n = self.n_features
        w = self.window_size
        rng = self.rng
        cluster_mean = np.zeros([n, 1])
        cluster_mean_stacked = np.zeros([n*w, 1])
        X = np.zeros((t_samples, self.n_features))
        start = 0
        for seg in range(len(break_points)):
            end = break_points[seg]
            cluster = labels[seg]
            for t in range(start, end):
                if t >= w:
                    cov_matrix = cluster_covariances[cluster]
                    Sig22 = cov_matrix[(w-1)*n:(w)*n, (w-1)*n:(w)*n]
                    Sig11 = cov_matrix[0:(w-1)*n, :(w-1)*n]
                    Sig21 = cov_matrix[(w-1)*n:w*n, :(w-1)*n]
                    Sig12 = np.transpose(Sig21)
                    cov_mat_tom = Sig22 - Sig21@np.linalg.inv(Sig11)@Sig12
                    a = np.zeros([(w-1)*n, 1])
                    for idx in range(w-1):
                        a[idx*n:(idx+1)*n, 0] = X[t - w+1+idx, :].reshape([n])
                    new_mean = cluster_mean + np.dot(Sig21@np.linalg.inv(Sig11),
                                                     (a-cluster_mean_stacked[:(w-1)*n, :]))
                    X[t, :] = rng.multivariate_normal(new_mean.reshape(n), cov_mat_tom)

                elif t == 0:  # first timepoint in seg
                    cov_matrix = cluster_covariances[cluster][:n, :n]
                    new_mean = cluster_mean_stacked[n*(w-1):n*w].reshape(n)
                    X[t, :] = rng.multivariate_normal(new_mean, cov_matrix)

                else:  # t < w
                    cov_matrix = cluster_covariances[cluster][:(t+1)*n, :(t+1)*n]
                    Sig22 = cov_matrix[t*n:(t+1)*n, t*n:(t+1)*n]
                    Sig11 = cov_matrix[:t*n, :t*n]
                    Sig21 = cov_matrix[t*n:(t+1)*n, :t*n]
                    Sig12 = np.transpose(Sig21)
                    cov_mat_tom = Sig22 - Sig21@np.linalg.inv(Sig11)@Sig12
                    a = np.zeros([t*n, 1])
                    for idx in range(t):
                        a[idx*n:(idx+1)*n, 0] = X[idx, :].reshape([n])
                    new_mean = (cluster_mean
                                + Sig21@np.linalg.inv(Sig11)
                                @ (a-cluster_mean_stacked[:t*n, :])
                                ).reshape(n)
                    X[t, :] = rng.multivariate_normal(new_mean, cov_mat_tom)
            start = break_points[seg-1]

        # Make label vector
        y = np.zeros((t_samples, ))
        # First label in sequence
        y[:break_points[0]] = labels[0]
        for i in range(len(labels)-1):
            y[break_points[i]:break_points[i+1]] = labels[i+1]
        return X, y.astype(int)
