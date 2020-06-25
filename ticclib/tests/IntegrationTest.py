import unittest
from ticclib import TICC
import numpy as np


class TestStringMethods(unittest.TestCase):
    def test_example(self):
        X = np.loadtxt('/Users/Gethin/Vestemi/code/TICC/ticclib/tests/test_data/example_data.txt', delimiter=",")
        ticc = TICC(n_clusters=8, window_size=1, lambda_parameter=11e-2,
                    beta=600, max_iter=100, n_jobs=4,
                    random_state=102, verbose=True)
        # X_stacked = ticc.stack_data(X)
        cluster_assignment = ticc.fit_predict(X)
        clusters = ticc.clusters_
        # np.savetxt("UnitTest_Data/Results.txt", cluster_assignment, fmt='%d', delimiter=",")
        assign = np.loadtxt("/Users/Gethin/Vestemi/code/TICC/ticclib/tests/test_data/Results.txt")
        val = abs(assign - cluster_assignment)
        self.assertEqual(sum(val), 0)
        # Test prediction works with batch of data outside of `fit` method. Perhaps there is a better way
        # to test this in parallel so these are more like unit tests rather than integration tests?
        batch_labels = ticc.predict(X[0:999, ])
        # np.savetxt("UnitTest_Data/batchLabels.txt", batch_labels, fmt="%d", delimiter=',')
        batch_val = abs(batch_labels - cluster_assignment[:999])
        self.assertEqual(sum(batch_val), 0)

        # Test streaming by passing in 5 row blocks at a time (current timestamp and previous 4)
        # I am causing data leakage by training on the whole set and then using the trained model while streaming,
        # but this is for testing the code, so it is ok
        # TODO: figure out why larger blocks don't improve predictions more. Reference:
        # https://github.com/davidhallac/TICC/issues/18#issuecomment-384514116

        def test_streaming(block_size):
            test_stream = np.zeros(1000)
            test_stream[0:block_size] = cluster_assignment[0:block_size]
            for i in range(block_size, 1000):
                point = X[i - block_size:i, ]
                test_stream[i] = ticc.predict(point)[block_size - 1]

            percent_correct_streaming = 100 * sum(cluster_assignment[:1000] == test_stream) / 1000.0
            self.assertGreater(percent_correct_streaming, 0.9)

        test_streaming(5)

        for i in range(8):
            # np.savetxt(f"UnitTest_Data/cluster_{i}.txt", clusterMRFs[i],  fmt='%.4e', delimiter=',')
            MRF = np.loadtxt(f"test_data/cluster_{i}.txt", delimiter=',')
            np.testing.assert_array_almost_equal(MRF, clusters[i].MRF_, decimal=3)

    def test_multiExample(self):
        X = np.loadtxt("test_data/example_data.txt", delimiter=",")
        ticc = TICC(n_clusters=5, window_size=5, lambda_parameter=11e-2, beta=600,
                    max_iter=100, n_jobs=4, random_state=102, verbose=True)
        # X_stacked = ticc.stack_data(X)
        cluster_assignment = ticc.fit_predict(X)
        clusters = ticc.clusters_
        # np.savetxt("UnitTest_Data/multiResults.txt", cluster_assignment, fmt='%d', delimiter=',')
        assign = np.loadtxt("test_data/multiResults.txt")
        val = abs(assign - cluster_assignment)
        self.assertEqual(sum(val), 0)

        for i in range(5):
            # np.savetxt(f"UnitTest_Data/multiCluster_{i}.txt", clusterMRFs[i], fmt='%.4e', delimiter=",")
            MRF = np.loadtxt(f"test_data/multiCluster_{i}.txt", delimiter=',')
            np.testing.assert_array_almost_equal(MRF, clusters[i].MRF_, decimal=3)

    def test_empty_cluster_handling(self):
        # We check if an error is thrown during handling of empty clusters
        X = np.load('test_data/example_empty_clusters.npy')
        ticc = TICC(n_clusters=4, window_size=5, n_jobs=4, random_state=0)
        # X_stacked = ticc.stack_data(X)
        ticc.fit(X)


if __name__ == '__main__':
    unittest.main()
