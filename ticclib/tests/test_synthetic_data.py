import pytest
import numpy as np

from .synthetic_data import RandomData


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

    def test_pass_if_upper_diags_tranposed(self):
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
            block_col = np.split(Theta, w, axis=1)[0]
            block_row = np.split(Theta, w, axis=0)[0]
            for block in range(w):
                A = np.split(block_col, w, axis=0)[block]
                B = np.split(block_row, w, axis=1)[block]
                np.testing.assert_array_equal(A.T, B)

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
                RandomData().GeneratePoints(t, seg, b)
            except ValueError:
                assert True

    def test_pass_if_right_num_points(self):
        test_cases = [
            (60, [0, 1, 0], [20, 40, 60]),
        ]
        for t, seg, b in test_cases:
            rd = RandomData(window_size=5, n_features=5)
            result = rd.GeneratePoints(t, seg, b)
            assert result.shape == (t, rd.n_features)