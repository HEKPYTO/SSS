import numpy as np


def test_sffs_wrapper_returns_selected_features():
    from sss import SFFS

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [9, 9], [9, 8], [8, 9], [8, 8]], dtype=float)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    result = SFFS(forward_budget=2, backward_budget=1, cv_folds=0, warmup_probes=0).fit(
        X, y, target_size=1
    )
    assert len(result.features) == 1
