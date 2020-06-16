import numpy as np
from ticclib import ticc


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
            result = ticc._update_clusters(X, beta=b)
        assert result.shape == (n_samples,)

    def test_equal_LLEs_case(self):
        # In the case of equal LLEs a path of zeros is returned
        test_cases = [
            (50, 5, 0),
            (10, 3, 100),
        ]
        for n_samples, k, b in test_cases:
            X = np.ones((n_samples, k))
            result = ticc._update_clusters(X, beta=b)
            np.testing.assert_array_equal(result, np.zeros(n_samples))

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
            result = ticc._update_clusters(X, beta=b)
            assert result.tolist() == path
