import pytest
import numpy as np
from itertools import combinations
from sklearn.utils.estimator_checks import check_estimator
from sklearn.model_selection import ParameterGrid
from ticclib import TICC
from ticclib.testing import RandomData, best_f1
from joblib import Parallel, delayed


@pytest.mark.slow
def test_ticc_sklearn_compatibility():
    # NOTE: window size = 1 ensures input unchanged by stacking
    # Max iter and beta of 0 to speed up test completion
    ticc = TICC(n_clusters=3, window_size=1, max_iter=15, beta=0, n_jobs=-1)
    return check_estimator(ticc)


class TestTicc:
    def test_matches_original_paper_macro_F1(self):
        test_cases = [
            # n_features, label order, macro f1 to-beat
            (5, [0, 1, 0], 0.9),
            (5, [0, 1, 2, 1, 0], 0.9),
            (5, [0, 1, 2, 3, 0, 1, 2, 3], 0.9),
            (5, [0, 1, 1, 0, 2, 2, 2, 0], 0.9),
        ]
        for n, labels, expected in test_cases:
            rdata = RandomData(0, n, window_size=5)
            samples_per_segment = 100
            k = len(set(labels))  # Num clusters
            t = samples_per_segment*k*len(labels)  # total ts length
            breaks = [(i)*t//len(labels) for i, _ in enumerate(labels, 1)]
            X, y_tru = rdata.generate_points(labels, breaks)
            ticc = TICC(n_clusters=k, window_size=5, n_jobs=4, random_state=0)
            y = ticc.fit_predict(X)
            # We use best_f1 because label:segment assignments are arbitrary
            result = best_f1(y_tru, y, average='macro')
            assert result > expected

    def test_fit_predict(self):
        rdata = RandomData(0, 5, 5)
        labels = [0, 1, 2, 3, 4, 5]
        breaks = [(i)*1200//len(labels) for i, _ in enumerate(labels, 1)]
        X, _ = rdata.generate_points(labels, breaks)
        ticc = TICC(n_clusters=6, window_size=5, beta=0, random_state=0)
        A = ticc.fit(X).predict(X)
        B = ticc.fit_predict(X)
        np.testing.assert_array_equal(A, B)

    def test_pass_if_consistent_on_similar_random_data(self):
        test_cases = [
            (5, [0, 1, 0], 5),
            (5, [0, 1, 2, 1, 0], 5),
        ]
        for n, labels, repeats in test_cases:
            rdata = RandomData(seed=0, n_features=n, window_size=5)
            k = len(set(labels))
            t = 200*k*len(labels)
            breaks = [(i)*t//len(labels) for i, _ in enumerate(labels, 1)]
            rdata.generate_cluster_params(k)
            # Reuse same cluster parameters for each dataset
            data = [rdata.generate_points(labels, breaks, True)[0]
                    for i in range(repeats)]
            ticc = TICC(n_clusters=k, window_size=5, beta=300, n_jobs=4,
                        random_state=0, cluster_reassignment=0.3, verbose=True)
            y_preds = [ticc.fit_predict(X) for X in data]
            for y1, y2 in combinations(y_preds, 2):
                result = np.sum(np.not_equal(y1, y2))/t
                assert result < 0.02

    def test_label_consistency_w_different_seeds(self):
        test_cases = [
            # seed1, seed2, n_features, label order, expected
            (0, 1, 5, [0, 1, 2]),
            (3, 2, 5, [0, 1, 2, 1, 0]),
            (0, 1, 5, [0, 1, 2, 3, 0, 1, 2, 3]),
            (0, 9, 5, [0, 1, 1, 0, 2, 2, 2, 0]),
        ]
        for s1, s2, n, labels in test_cases:
            rdata = RandomData(0, n, window_size=5)
            k = len(set(labels))  # Num clusters
            t = 200*k*len(labels)  # total ts length
            breaks = [(i)*t//len(labels) for i, _ in enumerate(labels, 1)]
            X, _ = rdata.generate_points(labels, breaks)
            ticc1 = TICC(n_clusters=k, window_size=5, n_jobs=4,
                         random_state=s1)
            ticc2 = TICC(n_clusters=k, window_size=5, n_jobs=4,
                         random_state=s2)
            y1 = ticc1.fit_predict(X)
            y2 = ticc2.fit_predict(X)
            np.testing.assert_array_equal(y1, y2)

    def test_score_increases_with_f1_window_size(self):
        rdata = RandomData(0, 5, 5)
        samples_per_segment = 100
        labels = [0, 1, 2, 3, 1, 0, 1]
        k = len(set(labels))  # Num clusters
        t = samples_per_segment*k*len(labels)  # total ts length
        breaks = [(i)*t//len(labels) for i, _ in enumerate(labels, 1)]
        X, y_true = rdata.generate_points(labels, breaks)
        params = ParameterGrid({'beta': [0, 500, 1000],
                                'window_size': [1, 5, 10],
                                })
        models = (TICC(**p, random_state=0) for p in params)
        models = Parallel(n_jobs=-1)(delayed(
            lambda x: x.fit(X))(m) for m in models)
        scores = [model.score_samples(X) for model in models]
        f1_scores = [
            best_f1(y_true, y) for y in map(lambda x: x.predict(X), models)
            ]
        assert scores.index(max(scores)) == f1_scores.index(max(f1_scores))
