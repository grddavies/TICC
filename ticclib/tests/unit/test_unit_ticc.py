import numpy as np
from numpy.testing import assert_array_equal
from ticclib.ticc import _update_clusters, _TICCluster, TICC
from ticclib.testing import RandomData


class TestUpdateClusters:
    def test_pass_if_output_dims_correct(self):
        test_cases = [
            (50, 5, 0, 0),
            (50, 5, 0, 1),
            (10, 3, 0, 10),
            (10, 3, 0, 0),
        ]
        for n_samples, k, seed, b in test_cases:
            X = (np.random.default_rng(seed)
                          .random((n_samples, k)))
            result = _update_clusters(X, beta=b)
        assert result.shape == (n_samples,)

    def test_equal_LLEs_case(self):
        # In the case of equal LLEs a path of zeros is returned
        test_cases = [
            (50, 5, 0),
            (10, 3, 100),
        ]
        for n_samples, k, b in test_cases:
            X = np.ones((n_samples, k))
            result = _update_clusters(X, beta=b)
            assert_array_equal(result, np.zeros(n_samples))

    def test_output_values_correct(self):
        test_cases = [
            (10, 5, 0, [3, 3, 1, 0, 0, 0, 2, 0, 1, 3]),
            (10, 5, 0.5, [3, 3, 3, 0, 0, 0, 0, 0, 3, 3]),
            (10, 8, 0.1, [3, 3, 4, 1, 0, 6, 5, 5, 5, 3]),
        ]
        for n_samples, k, b, path in test_cases:
            # path only correct for seed=0
            rng = np.random.default_rng(0)
            X = rng.random((n_samples, k))
            result = _update_clusters(X, beta=b)
            assert result.tolist() == path


class TestClusterMethods:
    def test_get_indices(self):
        test_cases = [
            ([0, 0, 1, 1], [2, 3]),
            ([1, 0, 1, 1], [0, 2, 3]),
            ([0, 0, 0, 0], []),
            ([1, 1, 1, 1], [0, 1, 2, 3]),
        ]
        for y, expected in test_cases:
            y = np.array(y)
            expected = np.array(expected)
            cluster = _TICCluster(1)
            result = cluster.get_indices(y).indices
            assert_array_equal(result, expected)

    def test_get_points(self):
        test_cases = [
            ([0, 0, 1, 1], [2, 3]),
            ([1, 0, 1, 1], [0, 2, 3]),
            ([0, 0, 0, 0], []),
            ([1, 1, 1, 1], [0, 1, 2, 3]),
        ]
        for y, indices in test_cases:
            y = np.array(y)
            cluster = _TICCluster(1)
            cluster.get_indices(y)
            rng = np.random.default_rng(0)
            X = rng.random((4, 4))
            result = cluster.get_points(X)
            assert_array_equal(result, X[indices, :])

    def test_split_theta(self):
        test_cases = [
            (5, 8)
        ]
        for n, w in test_cases:
            rdata = RandomData(0, n_features=n, window_size=w)
            cluster = _TICCluster(1)
            cluster.MRF_ = rdata.block_toeplitz()
            blocks = cluster.split_theta(w)
            assert len(blocks) == w
            for i, B in enumerate(blocks):
                assert B.shape == (n, n)
                A = cluster.MRF_[i*n:(i+1)*n, :n]
                assert_array_equal(B, A)


class TestTiccMethods:
    def test_stack_data(self):
        test_cases = [
            (5, 1),
            (5, 5),
            (2, 10),
        ]
        for n, w in test_cases:
            ticc = TICC(window_size=w)
            X = np.random.random((100, n))
            X_stacked = ticc.stack_data(X)
            assert X_stacked.shape == (100, n*w)
            for i in range(len(X)):
                assert_array_equal(X_stacked[i, :n], X[i, :])
                for j in range(w):
                    if i - j >= 0:
                        assert_array_equal(
                            X_stacked[i, j*n:(j+1)*n], X[i-j, :]
                            )
                    else:
                        assert_array_equal(
                            X_stacked[i, j*n:(j+1)*n], np.zeros((n,))
                            )
