from dataclasses import FrozenInstanceError

import numpy as np
import pytest


def _data():
    return np.array([[0, 0, 1], [0, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=float), np.array(
        [0, 0, 1, 1]
    )


def test_public_fit_returns_immutable_result():
    from sss import SFFS, Result

    X, y = _data()
    result = SFFS(forward_budget=2, backward_budget=1, warmup_probes=0, seed=1, cv_folds=0).fit(
        X, y, target_size=1
    )

    assert isinstance(result, Result)
    assert isinstance(result.features, tuple)
    assert len(result.features) == 1
    assert isinstance(result.value, float)
    assert result.evaluations > 0
    with pytest.raises(FrozenInstanceError):
        result.value = 0.0


def test_fit_arff_loads_path_without_mutating_caller_data(tmp_path):
    from sss import SFFS
    from sss.arff import load_arff

    path = tmp_path / "tiny.arff"
    path.write_text(
        "@RELATION tiny\n@ATTRIBUTE x NUMERIC\n@ATTRIBUTE y NUMERIC\n"
        "@ATTRIBUTE class {a,b}\n@DATA\n0,0,a\n0,1,a\n1,0,b\n1,1,b\n"
    )
    original = load_arff(str(path))
    before = original.data.copy()
    result = SFFS(forward_budget=2, backward_budget=1, warmup_probes=0).fit_arff(
        str(path), target_size=1, criterion="multinom-bhattacharyya"
    )

    assert len(result.features) == 1
    assert np.array_equal(original.data, before)


@pytest.mark.parametrize(
    ("X", "y"),
    [
        (np.empty((0, 2)), np.empty(0)),
        (np.empty(2), np.array([0, 1])),
        (np.ones((2, 2)), np.array([0])),
        (np.ones((2, 2)), np.array([0, 0])),
    ],
)
def test_invalid_fit_inputs_raise_value_error(X, y):
    from sss import SFFS

    with pytest.raises(ValueError):
        SFFS().fit(X, y, target_size=1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"forward_budget": -1},
        {"backward_budget": -1},
        {"exploration": -0.1},
        {"exploration": 1.1},
        {"horizon": 0},
        {"warmup_probes": -1},
        {"warmup_size": 0},
        {"frontier_delta": -1},
    ],
)
def test_invalid_configuration_raises_value_error(kwargs):
    from sss import SFFS

    with pytest.raises(ValueError):
        SFFS(**kwargs)
