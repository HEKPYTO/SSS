"""weighter.py — Online dependency-aware statistics, verbatim ssffs.cpp:638-705"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .subset import Subset


class Weighter:
    def __init__(self, horizon: int = 100):
        self.horizon = max(horizon, 1)
        self.n = 0
        self.frozen = False
        self._fs: list[dict] = []  # per feature: m_is, m_isnot, v_is, n_is, n_isnot
        self._batch: list[tuple[float, list[int]]] = []
        self._mark: list[int] = []

    def reset(self, n: int):
        self.n = n
        self._fs = [
            {"m_is": 0.0, "m_isnot": 0.0, "v_is": 0.0, "n_is": 0, "n_isnot": 0} for _ in range(n)
        ]
        self._mark = [0] * n
        self._batch.clear()
        self.frozen = False

    def add(self, value: float, sub: Subset):
        if self.frozen:
            return
        if self.n == 0:
            self.reset(sub.n)
        # ensure n matches
        if sub.n != self.n:
            # if reset not called after n change, reset
            self.reset(sub.n)
        selected = sub.members()
        self._batch.append((float(value), selected))

    def flush_batch(self):
        if self.frozen:
            self._batch.clear()
            return
        B = len(self._batch)
        if B == 0:
            return
        mu = sum(v for v, _ in self._batch) / B
        var = sum((v - mu) ** 2 for v, _ in self._batch) / B
        sd = math.sqrt(var)
        degenerate = not (sd > 1e-12)
        for value, selected in self._batch:
            z = 0.0 if degenerate else (value - mu) / sd
            for f in selected:
                self._mark[f] = 1
            for j in range(self.n):
                s = self._fs[j]
                if self._mark[j]:
                    s["n_is"] += 1
                    a = 1.0 / (min(self.horizon, s["n_is"]))
                    s["m_is"] += a * (z - s["m_is"])
                    s["v_is"] += a * (z * z - s["v_is"])
                else:
                    s["n_isnot"] += 1
                    a = 1.0 / (min(self.horizon, s["n_isnot"]))
                    s["m_isnot"] += a * (z - s["m_isnot"])
            for f in selected:
                self._mark[f] = 0
        self._batch.clear()

    def freeze(self):
        self.flush_batch()
        self.frozen = True

    def score(self, f: int) -> float:
        s = self._fs[f]
        return float(s["m_is"] - s["m_isnot"])

    def count_is(self, f: int) -> int:
        return int(self._fs[f]["n_is"])

    def count_isnot(self, f: int) -> int:
        return int(self._fs[f]["n_isnot"])

    # alias for test expectation n_is
    def n_is(self, f: int) -> int:
        return self.count_is(f)
