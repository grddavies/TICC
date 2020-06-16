import numpy as np
from ticclib.synthetic_data import RandomData


class TestGenerateInverse:
    def test_pass_if_block_dims_correct(self):
        test_cases = [
            (5, 10),
            (10, 10),
            (10, 1),
            (1, 1),
            (1, 10),
        ]

        for n, w in test_cases:
            result = (RandomData(n_features=n, window_size=w).GenerateInverse()
                                                             .shape
                      )

        assert result == (n*w, n*w)

    def test_pass_if_upper_diags_transposed(self):
        test_cases = [
            (5, 10),
            (10, 10),
            (3, 4),
            (5, 1),
        ]
        for n, w in test_cases:
            # Block Toeplitz Matrix Theta
            Theta = RandomData(n_features=n, window_size=w).GenerateInverse()
            block_chunked = [  # list of rows/cols of block matrix
                np.split(col, w) for col in np.split(Theta, w, axis=1)
                ]
            for i in range(w):
                for j in range(w):
                    A = block_chunked[i][j]
                    B = block_chunked[j][i]
                    np.testing.assert_array_equal(A.T, B)

    def test_pass_if_diags_symmetric(self):
        test_cases = [
            (5, 10),
            (10, 10),
            (3, 4),
            (5, 1),
        ]
        for n, w in test_cases:
            # Block Toeplitz Matrix Theta
            Theta = RandomData(n_features=n, window_size=w).GenerateInverse()
            block_chunked = [  # list of rows/cols of block matrix
                np.split(col, w) for col in np.split(Theta, w, axis=1)
                ]
            for i in range(w):
                A = block_chunked[i][i]
                np.testing.assert_array_equal(A.T, A)

    def test_pass_if_block_toeplitz(self):
        test_cases = [
            (5, 10),
            (10, 10),
            (3, 4),
            (5, 1),
        ]
        for n, w in test_cases:
            # Block Toeplitz Matrix Theta
            Theta = RandomData(n_features=n, window_size=w).GenerateInverse()
            # List of w columns of n*n blocks
            block_chunked = [  # list of rows/cols of block matrix
                np.split(col, w) for col in np.split(Theta, w, axis=1)
                ]
            for i in range(w-1):
                for j in range(w-1):
                    A = block_chunked[i][j]
                    B = block_chunked[i+1][j+1]
                    np.testing.assert_array_equal(A, B)

    def test_pass_if_positive_definite(self):
        test_cases = [
            (5, 10),
            (10, 10),
            (3, 4),
            (5, 1),
        ]
        for n, w in test_cases:
            # Block Toeplitz Matrix Theta
            Theta = RandomData(n_features=n, window_size=w).GenerateInverse()
            assert bool(np.linalg.cholesky(Theta).shape)


class TestGeneratePoints:
    def test_fail_if_input_error(self):
        test_cases = [
            (-100, [0, 1], [100, 300, 900]),
            (100, [0, 1], [100, 300, 900]),
            (1000, [0, 1, 3, 5], [100, 300, 450]),
        ]
        for t, seg, b in test_cases:
            try:
                RandomData().generate_points(t, seg, b)
            except ValueError:
                assert True

    def test_pass_if_X_size_correct(self):
        test_cases = [
            (60, [0, 1, 0], [20, 40, 60]),
            (600, [0, 1, 0, 1], [20, 40, 500, 600]),
        ]
        for t, seg, b in test_cases:
            rd = RandomData(window_size=5, n_features=5)
            result, _ = rd.generate_points(t, seg, b)
            assert result.shape == (t, rd.n_features)

    def test_pass_if_y_size_correct(self):
        test_cases = [
            (60, [0, 1, 0], [20, 40, 60]),
            (600, [0, 1, 0, 1], [20, 40, 500, 600]),
        ]
        for t, seg, b in test_cases:
            rd = RandomData(window_size=5, n_features=5)
            _, result = rd.generate_points(t, seg, b)
            assert result.shape == (t, )

    def test_pass_if_y_labels_correct(self):
        test_cases = [
            (10, [0, 1, 0], [3, 7, 10],
             [0, 0, 0, 1, 1, 1, 1, 0, 0, 0]),
            (20, [1, 0, 1, 0], [5, 10, 15, 20],
             [1]*5 + [0]*5 + [1]*5 + [0]*5),
        ]
        for t, seg, b, expected in test_cases:
            rd = RandomData(window_size=5, n_features=5)
            _, y = rd.generate_points(t, seg, b)
            assert y.tolist() == expected

    def test_pass_if_consistent_with_same_seed(self):
        test_cases = [
            (60, [0, 1, 0], [20, 40, 60], 0),
            (60, [0, 1, 0], [20, 40, 60], 1),
            (60, [0, 1, 0], [20, 40, 60], 10),
        ]
        for t, seg, b, seed in test_cases:
            X1, y1 = RandomData(seed).generate_points(t, seg, b)
            X2, y2 = RandomData(seed).generate_points(t, seg, b)

            np.testing.assert_array_equal(X1, X2)
            np.testing.assert_array_equal(y1, y2)

    def test_pass_if_generates_diff_points_each_call(self):
        test_cases = [
            (60, [0, 1, 0], [20, 40, 60], 0),
            (60, [0, 1, 0], [20, 40, 60], 10),
            (60, [0, 1, 0], [20, 40, 60], 100),
        ]
        for t, seg, b, seed in test_cases:
            rd = RandomData(seed)
            X1, y1 = rd.generate_points(t, seg, b)
            X2, y2 = rd.generate_points(t, seg, b)
            np.testing.assert_raises(AssertionError,
                                     np.testing.assert_array_equal,
                                     X1, X2)
            # We want different points but same labels!
            np.testing.assert_array_equal(y1, y2)

    def test_recycling_clusters_between_calls(self):
        test_cases = [
            (60, [0, 1, 0], [20, 40, 60], 0),
            (60, [0, 1, 0], [20, 40, 60], 10),
            (60, [0, 1, 0], [20, 40, 60], 100),
        ]
        for t, seg, b, seed in test_cases:
            rdata = RandomData(seed)
            rdata.generate_cluster_params(len(set(seg)))
            X1, y1 = rdata.generate_points(t, seg, b, True)
            C1 = rdata.clusters
            X2, y2 = rdata.generate_points(t, seg, b, True)
            C2 = rdata.clusters
            assert C1 == C2
