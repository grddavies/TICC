import pytest
import numpy as np
from itertools import combinations
from sklearn.utils.estimator_checks import check_estimator
from sklearn.metrics import f1_score
from ticclib.ticc import TICC
from ticclib.synthetic_data import RandomData


@pytest.mark.slow
def test_ticc_sklearn_compatibility():
    # NOTE: window size = 1 ensures input unchanged by stacking
    # Max iter and beta of 0 to speed up test completion
    ticc = TICC(n_clusters=3, window_size=1, max_iter=15, beta=0)
    return check_estimator(ticc)


class TestTicc:
    def test_matches_original_paper_macro_F1(self):
        test_cases = [
            # n_features, clusters, label order
            (5, [0, 1, 0], 0.9),
            (5, [0, 1, 2, 1, 0], 0.9),
            (5, [0, 1, 2, 3, 0, 1, 2, 3], 0.9),
            (5, [0, 1, 1, 0, 2, 2, 2, 0], 0.9),
        ]
        for n, labels, expected in test_cases:
            rdata = RandomData(0, n, window_size=5)
            samples_per_segment = 200
            k = len(set(labels))  # Num clusters
            t = samples_per_segment*k*len(labels)  # total ts length
            breaks = [i*t//len(labels) for i in range(1, len(labels) + 1)]
            X, y_tru = rdata.generate_points(labels, breaks)
            all_labels = list(set(labels))
            ticc = TICC(n_clusters=k, window_size=5, n_jobs=4, random_state=0)
            y = ticc.fit(X).labels_
            result = f1_score(y_tru, y, labels=all_labels, average='macro')
            assert result > expected

    def test_pass_if_consistent_on_similar_random_data(self):
        test_cases = [
            (5, [0, 1, 0], 5),
            (5, [0, 1, 2, 1, 0], 5),
        ]
        for n, labels, repeats in test_cases:
            rdata = RandomData(seed=0, n_features=n, window_size=5)
            k = len(set(labels))
            t = 200*k*len(labels)
            breaks = [i*t//len(labels) for i in range(1, len(labels) + 1)]
            rdata.generate_cluster_params(k)
            # Reuse same cluster parameters for each dataset
            data = [rdata.generate_points(labels, breaks, True)[0]
                    for i in range(repeats)]
            ticc = TICC(n_clusters=k, window_size=5, beta=300, n_jobs=4,
                        random_state=0, cluster_reassignment=0.3, verbose=True)
            models = [ticc.fit(X) for X in data]
            y_preds = [model.labels_ for model in models]
            for y1, y2 in combinations(y_preds, 2):
                result = np.sum(np.not_equal(y1, y2))/t
                assert result == 0

    def test_label_consistency(self):
        test_cases = [
            # n_features, clusters, label order
            (0, 1, 5, [0, 1, 2], 0.9),
            (3, 2, 5, [0, 1, 2, 1, 0], 0.9),
            (0, 1, 5, [0, 1, 2, 3, 0, 1, 2, 3], 0.9),
            # (5, [0, 1, 1, 0, 2, 2, 2, 0], 0.9),
        ]
        for s1, s2, n, labels, expected in test_cases:
            rdata = RandomData(0, n, window_size=5)
            k = len(set(labels))  # Num clusters
            t = 200*k*len(labels)  # total ts length
            breaks = [i*t//len(labels) for i in range(1, len(labels) + 1)]
            X, y_tru = rdata.generate_points(labels, breaks)
            all_labels = list(set(labels))
            ticc1 = TICC(n_clusters=k, window_size=5, n_jobs=4,
                         random_state=s1)
            ticc2 = TICC(n_clusters=k, window_size=5, n_jobs=4,
                         random_state=s2)
            y1 = ticc1.fit(X).labels_
            y2 = ticc2.fit(X).labels_
            result = f1_score(y1, y2, labels=all_labels, average='macro')
            assert result > expected
