import numpy as np


def _subset(size, *features):
    from sss.subset import Subset

    subset = Subset(size)
    subset.deselect_all()
    for feature in features:
        subset.select_raw(feature)
    return subset


def _wrapper_data():
    from sss.arff import ArffData
    from sss.split import Split

    data = ArffData(2, 2, [3, 3], np.array([0, 0, 0, 1, 1, 0, 10, 10, 10, 11, 11, 10], dtype=float))
    split = Split(2)
    split.train = [[0, 1], [0, 1]]
    split.test = [[2], [2]]
    return data, split


def test_wrapper_knn_uses_first_maximum_for_equal_distances():
    from sss.criteria import WrapperKnn

    data, split = _wrapper_data()
    data.data[:] = 0
    ok, value = WrapperKnn(data, [split], k=2).evaluate(_subset(2, 0, 1))
    assert ok and value == 0.5


def test_multinom_bhattacharyya_scores_singletons_and_subsets():
    from sss.arff import ArffData
    from sss.criteria import MultinomBhattacharyya

    data = ArffData(3, 2, [2, 2], np.array([2, 0, 1, 1, 0, 1, 0, 2, 1, 0, 1, 2], dtype=float))
    criterion = MultinomBhattacharyya(data, [[0, 1], [0, 1]])
    singleton = criterion.evaluate(_subset(3, 0))
    pair = criterion.evaluate(_subset(3, 0, 1))
    assert singleton[0] and singleton[1] >= 0
    assert pair[0] and pair[1] >= 0
