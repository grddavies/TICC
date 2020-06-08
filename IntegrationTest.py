import unittest
from TICC_solver import TICC
import numpy as np


class TestStringMethods(unittest.TestCase):

    def test_example(self):
        X = np.loadtxt("example_data.txt", delimiter=",")
        ticc = TICC(n_clusters=8, window_size=1, lambda_parameter=11e-2,
                    beta=600, max_iter=100, num_proc=4,
                    random_state=102)
        ticc.fit(X)
        cluster_assignment, cluster_MRFs = ticc.labels_, ticc.cluster_MRFs_
        # np.savetxt("UnitTest_Data/Results.txt", cluster_assignment, fmt='%d', delimiter=",")
        assign = np.loadtxt("UnitTest_Data/Results.txt")
        val = abs(assign - cluster_assignment)
        self.assertEqual(sum(val), 0)

        # Test prediction works with batch of data outside of `fit` method. Perhaps there is a better way
        # to test this in parallel so these are more like unit tests rather than integration tests?
        batch_labels = ticc.predict_clusters(ticc.trained_model['complete_D_train'][0:999, ])
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
                point = ticc.trained_model['complete_D_train'][i - block_size:i, ]
                test_stream[i] = ticc.predict_clusters(point)[block_size - 1]

            percent_correct_streaming = 100 * sum(cluster_assignment[:1000] == test_stream) / 1000.0
            self.assertGreater(percent_correct_streaming, 0.9)

        test_streaming(5)

        for i in range(8):
            # np.savetxt(f"UnitTest_Data/cluster_{i}.txt", cluster_MRFs[i],  fmt='%.4e', delimiter=',')
            mrf = np.loadtxt(f"UnitTest_Data/cluster_{i}.txt", delimiter=',')
            np.testing.assert_array_almost_equal(mrf, cluster_MRFs[i], decimal=3)

    def test_multiExample(self):
        X = np.loadtxt("example_data.txt", delimiter=",")
        ticc = TICC(n_clusters=5, window_size=5, lambda_parameter=11e-2, beta=600,
                    max_iter=100, num_proc=4, random_state=102)
        ticc.fit(X)
        cluster_assignment, cluster_MRFs = ticc.labels_, ticc.cluster_MRFs_
        # np.savetxt("UnitTest_Data/multiResults.txt", cluster_assignment, fmt='%d', delimiter=',')
        assign = np.loadtxt("UnitTest_Data/multiResults.txt")
        val = abs(assign - cluster_assignment)
        self.assertEqual(sum(val), 0)

        for i in range(5):
            # np.savetxt(f"UnitTest_Data/multiCluster_{i}.txt", cluster_MRFs[i], fmt='%.4e', delimiter=",")
            mrf = np.loadtxt(f"UnitTest_Data/multiCluster_{i}.txt", delimiter=',')
            np.testing.assert_array_almost_equal(mrf, cluster_MRFs[i], decimal=3)


if __name__ == '__main__':
    unittest.main()
