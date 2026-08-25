import sys
import types

import numpy as np


def _subset(size, *features):
    from src.sss.subset import Subset

    subset = Subset(size)
    subset.deselect_all()
    for feature in features:
        subset.select_raw(feature)
    return subset


def _wrapper_data():
    from src.sss.arff import ArffData
    from src.sss.split import Split

    data = ArffData(2, 2, [3, 3], np.array([0, 0, 0, 1, 1, 0, 10, 10, 10, 11, 11, 10], dtype=float))
    split = Split(2)
    split.train = [[0, 1], [0, 1]]
    split.test = [[2], [2]]
    return data, split


def test_wrapper_knn_uses_first_maximum_for_equal_distances():
    from src.sss.criteria import WrapperKnn

    data, split = _wrapper_data()
    data.data[:] = 0
    ok, value = WrapperKnn(data, [split], k=2).evaluate(_subset(2, 0, 1))
    assert ok and value == 0.5


def test_wrapper_uses_reference_neighbor_rule_without_sklearn(monkeypatch):
    from src.sss.criteria import WrapperKnn

    class WrongKnn:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, _X, _y):
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    sklearn = types.ModuleType("sklearn")
    neighbors = types.ModuleType("sklearn.neighbors")
    neighbors.KNeighborsClassifier = WrongKnn
    sklearn.neighbors = neighbors
    monkeypatch.setitem(sys.modules, "sklearn", sklearn)
    monkeypatch.setitem(sys.modules, "sklearn.neighbors", neighbors)
    data, split = _wrapper_data()
    ok, value = WrapperKnn(data, [split], k=1).evaluate(_subset(2, 0))
    assert ok and value == 1.0


def test_multinom_bhattacharyya_scores_singletons_and_subsets():
    from src.sss.arff import ArffData
    from src.sss.criteria import MultinomBhattacharyya

    data = ArffData(3, 2, [2, 2], np.array([2, 0, 1, 1, 0, 1, 0, 2, 1, 0, 1, 2], dtype=float))
    criterion = MultinomBhattacharyya(data, [[0, 1], [0, 1]])
    singleton = criterion.evaluate(_subset(3, 0))
    pair = criterion.evaluate(_subset(3, 0, 1))
    assert singleton[0] and singleton[1] >= 0
    assert pair[0] and pair[1] >= 0
