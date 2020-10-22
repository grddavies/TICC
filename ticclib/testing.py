import numpy as np
from itertools import permutations
from sklearn.datasets import make_sparse_spd_matrix
from sklearn.metrics import f1_score


def best_f1(y_true, y_pred, average='macro'):
    """Calculate an F1 score for each permutation of labels in y_pred and
    return the highest.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.

    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.

    average : str ['micro', 'macro' (default), 'samples', 'weighted']
        Keyword argument passed to sklearn f1_score function.
    """
    # Labels are arbitrary integers in y_pred. We want to find the permutation
    # of labels that minimises the distance between y_pred and y_true, and
    # return the associated f1 score of this labelling
    labels = np.unique(y_pred)
    # Make list of dicts to perform all permutations of label swaps
    dicts = [{label: p[ix] for ix, label in enumerate(labels)}
             for p in permutations(labels)]
    # F1 scores for each permutation of labels in y_pred
    results = [f1_score(y_true, np.vectorize(d.get)(y_pred), average=average)
               for d in dicts]
    return max(results)


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
        return "%s(%r)" % (self.__class__, self.__dict__.keys())

    def inv_cov(self, low=0.3, upper=0.6, p=0.2, symmetric=True) -> np.array:
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
        rs = self.rng.integers(10000)
        return make_sparse_spd_matrix(dim=self.n_features, alpha=1-p,
                                      smallest_coef=low, largest_coef=upper,
                                      random_state=rs)
        # # Old version: from paper
        # n = self.n_features
        # S = np.zeros((n, n))
        # seed = int(self.rng.integers(1000))
        # G = nx.fast_gnp_random_graph(n, p, seed=seed)
        # # Fill S with random values
        # for e in G.edges:
        #     value = ((self.rng.integers(2)-0.5)
        #              * 2 * (low+(upper-low)*self.rng.random()))
        #     S[e[0], e[1]] = value
        # if symmetric:
        #     S = S + S.T
        # return S

    def rand_inverse(self, low=0.3, upper=0.6, p=0.2) -> np.array:
        n = self.n_features
        S = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if self.rng.random() < p:
                    value = ((self.rng.integers(2) - 0.5)*2
                             * (low + (upper-low)*self.rng.random()))
                    S[i, j] = value
        return S

    def block_toeplitz(self) -> np.array:
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
        blocks = [self.rand_inverse(p=p) for i in range(w)]
        # A0 block has symmetry enforced
        blocks[0] = self.inv_cov(p=p)
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

    def generate_cluster_params(self, k_clusters, sort=True):
        self.clusters = [self.block_toeplitz() for k in range(k_clusters)]
        if sort:
            self.clusters = sorted(
                self.clusters,
                key=lambda x: np.linalg.norm(np.linalg.inv(x)),
                reverse=True)

    def generate_points(self, labels, break_points, recycle=False
                        ) -> np.array:
        """Generate n-dimensional timeseries coming from k states, which are
        represented by partial correlation matrices.

        Parameters
        ----------
        labels : list of integers
            List of up to k labels to be attributed to each sample.

        break_points : list of integers
            Points at which the sample labels change. Dataset ends at last
            break point.

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
        if len(break_points) != len(labels):
            raise ValueError(f"Specified {len(labels)} segments with `labels`"
                             f"argument, and {len(break_points)} break points "
                             "- ensure the same number of segments and break "
                             "points are passed")
        # Generate/retrieve cluster parameters
        k_clusters = len(set(labels))
        if recycle:
            if len(self.clusters) < k_clusters:
                raise ValueError("Not enough clusters! %d Required to generate"
                                 " points but only %d in self.clusters"
                                 % (k_clusters, len(self.clusters)))
        else:
            self.generate_cluster_params(k_clusters)
        cluster_inverses = self.clusters
        cluster_covariances = [
            np.linalg.inv(k) for k in cluster_inverses
            ]
        # Parameter aliases
        t_samples = break_points[-1]
        n = self.n_features
        w = self.window_size
        rng = self.rng
        # Initialize
        cluster_mean = np.zeros([n, 1])
        cluster_mean_stacked = np.zeros([n*w, 1])
        X = np.zeros((t_samples, self.n_features))
        start = 0
        for seg in range(len(break_points)):
            end = break_points[seg]
            cluster = labels[seg]
            for t in range(start, end):
                # print(f"seg {seg}, t {t}, cluster {cluster}, {(start, end)}")  # Debug
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

                elif t == 0:  # first timepoint
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
            start = break_points[seg]

        # Make label vector
        y = np.zeros((t_samples, ))
        # First label in sequence
        y[:break_points[0]] = labels[0]
        for i in range(len(labels)-1):
            y[break_points[i]:break_points[i+1]] = labels[i+1]
        return X, y.astype(int)
