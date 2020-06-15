from sklearn.utils.estimator_checks import check_estimator
from ticclib.ticc import TICC


def test_ticc():
    # NOTE: window size = 1 ensures input unchanged by stacking
    # Max iter and beta of 0 to speed up test completion
    ticc = TICC(n_clusters=3, window_size=1, max_iter=10, beta=0)
    return check_estimator(ticc)
