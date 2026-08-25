from __future__ import annotations

from dataclasses import dataclass
from typing import TextIO

import numpy as np

from .arff import ArffData, load_arff, scale_to01
from .criteria import CountingCriterion, MultinomBhattacharyya, WrapperKnn
from .prng import ss_srand
from .sampled_step import SampledStep
from .split import Split, cv_folds, rr_split_class
from .subset import Subset
from .weighter import Weighter


@dataclass(frozen=True)
class Result:
    features: tuple[int, ...]
    value: float
    evaluations: int


def _emit_solution(stream: TextIO | None, value: float, sub: Subset) -> None:
    if stream is not None:
        features = ",".join(str(feature) for feature in sub.members())
        stream.write(
            f'{{"event":"solution","value":{value},"d":{sub.get_d()},"features":[{features}]}}\n'
        )
        stream.flush()


def sffs_search(
    target_size: int,
    delta: int,
    sub: Subset,
    criterion: CountingCriterion,
    step: SampledStep,
    stream: TextIO | None = None,
) -> Result | None:
    incumbents: list[tuple[float, list[int]] | None] = [None] * sub.n
    pivot = Subset(sub.n)
    maximum: tuple[float, list[int]] | None = None
    sub.set_forward_mode(True)
    sub.deselect_all()
    size = 0
    threshold = (
        target_size + delta if target_size and delta and target_size + delta < sub.n else sub.n
    )
    while size + 1 <= threshold:
        ok, value = step.Step(True, sub, criterion, stream)
        if not ok:
            return None
        size = sub.get_d()
        members = sub.bin.copy()
        if (
            maximum is None
            or value > maximum[0]
            or (value == maximum[0] and size < sum(v > 0 for v in maximum[1]))
        ):
            maximum = (value, members)
            _emit_solution(stream, value, sub)
        if incumbents[size - 1] is None or value > incumbents[size - 1][0]:
            incumbents[size - 1] = (value, members)
            _emit_solution(stream, value, sub)
        pivot.copy_members_from(sub)
        backtrack = True
        while backtrack and size >= 2:
            ok, value = step.Step(False, sub, criterion, stream)
            if not ok:
                return None
            size = sub.get_d()
            members = sub.bin.copy()
            if (
                maximum is None
                or value > maximum[0]
                or (value == maximum[0] and size < sum(v > 0 for v in maximum[1]))
            ):
                maximum = (value, members)
                _emit_solution(stream, value, sub)
            if incumbents[size - 1] is None:
                incumbents[size - 1] = (value, members)
                _emit_solution(stream, value, sub)
            elif value > incumbents[size - 1][0]:
                incumbents[size - 1] = (value, members)
                pivot.copy_members_from(sub)
                _emit_solution(stream, value, sub)
            else:
                backtrack = False
        sub.bin = pivot.bin.copy()
        sub.set_forward_mode(True)
        size = sub.get_d()
    candidate = incumbents[target_size - 1] if target_size else maximum
    if candidate is None:
        return None
    value, members = candidate
    return Result(
        tuple(index for index, selected in enumerate(members) if selected > 0),
        float(value),
        criterion.count,
    )


def _run(
    data: ArffData,
    *,
    target_size: int,
    criterion_name: str,
    forward_budget: int,
    backward_budget: int,
    exploration: float,
    temperature: float | None,
    horizon: int,
    warmup_probes: int,
    warmup_size: int,
    frontier_delta: int,
    seed: int,
    train_percent: int,
    test_percent: int,
    cv_folds_count: int,
    knn_k: int,
    scale: bool,
    cap_fraction: float = 0.0,
    sampler: str = "softmax",
    frozen: bool = False,
    stream: TextIO | None = None,
) -> Result:
    if target_size < 0 or target_size >= data.n_features:
        raise ValueError("target_size must be smaller than the number of features")
    if criterion_name not in {"wrapper-knn", "multinom-bhattacharyya"}:
        raise ValueError("criterion must be wrapper-knn or multinom-bhattacharyya")
    if (
        not 0 <= train_percent <= 100
        or not 0 <= test_percent <= 100
        or train_percent + test_percent > 100
    ):
        raise ValueError("train and test percentages must be in 0..100 and sum to at most 100")
    if knn_k < 1 or cv_folds_count < 0:
        raise ValueError("invalid criterion configuration")
    working = ArffData(data.n_features, data.n_classes, data.class_size.copy(), data.data.copy())
    if scale:
        scale_to01(working)
    ss_srand(seed)
    outer = Split(working.n_classes)
    for cls in range(working.n_classes):
        rr_split_class(
            working.class_size[cls], train_percent, test_percent, outer.train[cls], outer.test[cls]
        )
    if criterion_name == "wrapper-knn":
        folds = (
            cv_folds(outer, cv_folds_count, working.n_classes) if cv_folds_count > 1 else [outer]
        )
        inner = WrapperKnn(working, folds, knn_k)
    else:
        inner = MultinomBhattacharyya(working, outer.train)
    criterion = CountingCriterion(inner)
    weighter = Weighter(horizon)
    step = SampledStep(
        weighter,
        cap=forward_budget,
        cap_backward=backward_budget,
        cap_frac=cap_fraction,
        explore=exploration,
        tau=0.0 if temperature is None else temperature,
        sampler=sampler,
    )
    if warmup_probes:
        probe = Subset(working.n_features)
        weighter.reset(working.n_features)
        for _ in range(warmup_probes):
            probe.make_random_subset(min(warmup_size, working.n_features - 1))
            ok, value = criterion.evaluate(probe)
            if not ok:
                raise RuntimeError("warm-up probe evaluation failed")
            weighter.add(value, probe)
        weighter.flush_batch()
        if frozen:
            weighter.freeze()
    result = sffs_search(
        target_size, frontier_delta, Subset(working.n_features), criterion, step, stream
    )
    if result is None:
        raise RuntimeError("search not finished")
    return result


class SFFS:
    def __init__(
        self,
        forward_budget: int = 100,
        backward_budget: int = 50,
        exploration: float = 0.2,
        temperature: float | None = None,
        horizon: int = 100,
        warmup_probes: int = 0,
        warmup_size: int = 25,
        frontier_delta: int = 0,
        seed: int = 1,
        knn_k: int = 1,
        cv_folds: int = 3,
    ) -> None:
        if forward_budget < 0 or backward_budget < 0 or not 0 <= exploration <= 1:
            raise ValueError("invalid search budget or exploration")
        if temperature is not None and temperature <= 0:
            raise ValueError("temperature must be positive or None")
        if horizon < 1 or warmup_probes < 0 or warmup_size < 1 or frontier_delta < 0:
            raise ValueError("invalid search configuration")
        if knn_k < 1 or cv_folds < 0:
            raise ValueError("invalid criterion configuration")
        self.forward_budget, self.backward_budget = forward_budget, backward_budget
        self.exploration, self.temperature, self.horizon = exploration, temperature, horizon
        self.warmup_probes, self.warmup_size = warmup_probes, warmup_size
        self.frontier_delta, self.seed = frontier_delta, seed
        self.knn_k, self.cv_folds = knn_k, cv_folds

    def fit(self, X: np.ndarray, y: np.ndarray, *, target_size: int) -> Result:
        features, labels = np.asarray(X, dtype=np.float64), np.asarray(y)
        if features.ndim != 2 or not len(features) or len(labels) != len(features):
            raise ValueError("X must be a non-empty 2D array and y must have matching length")
        classes, encoded = np.unique(labels, return_inverse=True)
        if len(classes) < 2:
            raise ValueError("y must contain at least two classes")
        sizes = [int(np.sum(encoded == cls)) for cls in range(len(classes))]
        flat = (
            np.concatenate([features[encoded == cls] for cls in range(len(classes))]).ravel().copy()
        )
        return self._fit(
            ArffData(features.shape[1], len(classes), sizes, flat),
            target_size,
            "wrapper-knn",
            True,
            50,
            50,
        )

    def fit_arff(self, path: str, *, target_size: int, criterion: str) -> Result:
        return self._fit(
            load_arff(path),
            target_size,
            criterion,
            criterion == "wrapper-knn",
            50,
            50 if criterion == "wrapper-knn" else 40,
        )

    def _fit(
        self, data: ArffData, target_size: int, criterion: str, scale: bool, train: int, test: int
    ) -> Result:
        return _run(
            data,
            target_size=target_size,
            criterion_name=criterion,
            forward_budget=self.forward_budget,
            backward_budget=self.backward_budget,
            exploration=self.exploration,
            temperature=self.temperature,
            horizon=self.horizon,
            warmup_probes=self.warmup_probes,
            warmup_size=self.warmup_size,
            frontier_delta=self.frontier_delta,
            seed=self.seed,
            train_percent=train,
            test_percent=test,
            cv_folds_count=self.cv_folds,
            knn_k=self.knn_k,
            scale=scale,
        )
