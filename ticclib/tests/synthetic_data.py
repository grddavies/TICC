import numpy as np
from sklearn.utils import check_random_state
from networkx import nx


class RandomData:
    """Class to synthesise random time-series data with
    underlying graph properties
    """
    def __init__(self, random_state=None, n_features=5, window_size=8,
                 sparsity=0.2):
        self.random_state = random_state
        self.n_features = n_features
        self.window_size = window_size
        self.sparsity = sparsity

    def __repr__(self):
        return "%s(%r)" % (self.__class__, self.__dict__)

    def genInvCov(self, seed, low=0.3, upper=0.6, p=0.2, symmetric=True) -> np.array:
        """Generate inverse covariance matrices for n_features

        Atrributes
        ----------
        p : float > 0, default = 0.2
        Probability of edge between nodes in random graph, ie inverse
        covariance matrix sparsity
        """
        n = self.n_features
        random_state_ = check_random_state(seed)
        rng = np.random.default_rng(self.random_state)
        S = np.zeros((n, n))
        G = nx.erdos_renyi_graph(n, p, seed=random_state_)
        # Fill S with random values
        for e in G.edges:
            value = (rng.integers(2)-0.5)*2*(low+(upper-low)*rng.random())
            S[e[0], e[1]] = value
        if symmetric:
            S = S + S.T
        return S

    def genRandInv(self, seed, low=0.3, upper=0.6, p=0.2) -> np.array:
        n = self.n_features
        rng = np.random.default_rng(seed)
        S = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if rng.random() < p:
                    value = (rng.integers(2) - 0.5)*2*(low + (upper - low)*rng.random())
                    S[i, j] = value
        return S

    def GenerateInverse(self, seed=0) -> np.array:
        """Generate block Toeplitz inverse covariance matrix.

        Each block is (`n_features`*`n_features`)
        There are `window_size` number of blocks
        """
        if isinstance(self.random_state, int):
            seed += self.random_state
        rng = np.random.default_rng(seed)
        n = self.n_features
        w = self.window_size
        p = self.sparsity
        Theta = np.zeros((n*w, n*w))
        seeds = rng.choice(w, size=w, replace=False)
        # w independent n*n matrices of independent Erdos-Renyi directed random
        # graphs with probability p of being selected.
        blocks = [self.genRandInv(seeds[i], p=p) for i in seeds]
        # A0 block has symmetry enforced
        blocks[0] = self.genInvCov(seeds[0], p=p)
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

    def GeneratePoints(self, t_samples, segments, break_points) -> np.array:
        """Generate n-dimensional timeseries coming from k states, which are
        represented by partial correlation matrices.

        Parameters
        ----------
        t_samples : int
            Number of samples of n-dimensional data to generate.

        segments : list of integers
            List of up to k labels to be attributed to each sample.

        break_points : list of integers
            points at which the sample labels change
        """
        # Check input vals
        if break_points[-1] != t_samples:
            raise ValueError("Last break at %d, there are %d points in "
                             "timeseries."
                             % (max(break_points), t_samples))
        k_clusters = len(set(segments))
        # Generate cluster parameters
        cluster_inverses = [
            self.GenerateInverse(k) for k in range(k_clusters)
            ]
        cluster_covariances = [
            np.linalg.inv(k) for k in cluster_inverses
            ]
        n = self.n_features
        w = self.window_size
        rng = np.random.default_rng(self.random_state)
        cluster_mean = np.zeros([n, 1])
        cluster_mean_stacked = np.zeros([n*w, 1])
        X = np.zeros((t_samples, self.n_features))
        start = 0
        for seg in range(len(break_points)):
            end = break_points[seg]
            cluster = segments[seg]
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
                    new_mean = cluster_mean + np.dot(Sig21@np.linalg.inv(Sig11), (a-cluster_mean_stacked[:(w-1)*n, :]))
                    X[t, :] = rng.multivariate_normal(new_mean.reshape(n), cov_mat_tom)

                elif t == 0:  # this is the first timepoint in seg
                    cov_matrix = cluster_covariances[cluster][:n, :n]
                    new_mean = cluster_mean_stacked[n*(w-1):n*w].reshape(n)
                    X[t, :] = rng.multivariate_normal(new_mean, cov_matrix)

                elif t < w:
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
                else:
                    raise ValueError("What t is this?")
            start = break_points[seg-1]
        return X
