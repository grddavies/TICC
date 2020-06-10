import unittest
from TICC_solver import _update_clusters
import numpy as np


class TestUpdateClusters(unittest.TestCase):
    def test_update_clusters(self):
        rng = np.random.default_rng(42)
        X = rng.random((10, 3))
        _update_clusters(X, 0)


if __name__ == '__main__':
    unittest.main()
