"""sss.py — sSFFS search + SFFS class, verbatim ssffs.cpp:983-1074 + main 1098-1208"""
from __future__ import annotations

import sys

import numpy as np

from .arff import ArffData, scale_to01
from .criteria import CountingCriterion, MultinomBhattacharyya, WrapperKnn
from .prng import ss_srand
from .sampled_step import SampledStep
from .split import Split, cv_folds, rr_split_class
from .subset import Subset
from .weighter import Weighter


def _emit_solution(os, value: float, sub: Subset):
    feats = sub.members()
    feats_str = ",".join(str(f) for f in feats)
    os.write(f'{{"event":"solution","value":{value},"d":{sub.get_d()},"features":[{feats_str}]}}\n')
    os.flush()


def sffs_search(
    target_d: int,
    delta: int,
    sub: Subset,
    crit: CountingCriterion,
    step: SampledStep,
    os,
) -> dict | None:
    n = sub.n
    # bsubs per size
    bsubs = [{"present": False, "bin": None, "critvalue": 0.0} for _ in range(n)]
    pivotsub = Subset(n)
    maxcritsub = Subset(n)
    maxcritval = 0.0
    havemax = False
    d_max = 0

    sub.set_forward_mode(True)
    sub.deselect_all()
    d = 0
    # forward threshold
    if target_d > 0 and delta > 0 and target_d + delta < n:
        forward_thr = target_d + delta
    else:
        forward_thr = n

    # per-size incumbent logic helpers
    def check_maxcrit(result, cur_sub, d_cur):
        nonlocal havemax, maxcritval, d_max
        if not havemax or result > maxcritval or (result == maxcritval and d_cur < d_max):
            maxcritsub.copy_members_from(cur_sub)
            maxcritval = result
            d_max = d_cur
            havemax = True
            _emit_solution(os, maxcritval, maxcritsub)

    while d + 1 <= forward_thr:
        ok, result = step.Step(True, sub, crit, os)
        if not ok:
            return None
        d = sub.get_d()
        check_maxcrit(result, sub, d)
        if not bsubs[d - 1]["present"]:
            bsubs[d - 1]["present"] = True
            bsubs[d - 1]["bin"] = sub.bin.copy()
            bsubs[d - 1]["critvalue"] = result
            _emit_solution(os, result, sub)
        elif result > bsubs[d - 1]["critvalue"]:
            bsubs[d - 1]["bin"] = sub.bin.copy()
            bsubs[d - 1]["critvalue"] = result
            _emit_solution(os, result, sub)
        pivotsub.copy_members_from(sub)
        backtrack = True
        while backtrack and d >= 2:
            ok, result = step.Step(False, sub, crit, os)
            if not ok:
                return None
            d = sub.get_d()
            check_maxcrit(result, sub, d)
            if not bsubs[d - 1]["present"]:
                bsubs[d - 1]["present"] = True
                bsubs[d - 1]["bin"] = sub.bin.copy()
                bsubs[d - 1]["critvalue"] = result
                _emit_solution(os, result, sub)
            elif result > bsubs[d - 1]["critvalue"]:
                bsubs[d - 1]["bin"] = sub.bin.copy()
                bsubs[d - 1]["critvalue"] = result
                pivotsub.copy_members_from(sub)
                _emit_solution(os, result, sub)
            else:
                backtrack = False
        # restore pivot
        sub.bin = pivotsub.bin.copy()
        # keep forward mode true? pivotsub was copied from sub which was forward mode after steps; ensure bin>0 semantics preserved
        # sub.forward already true after last Step restore, but after loop we restored bin which is hard +-1, need forward true
        sub.set_forward_mode(True)
        # but pivotsub bin is hard +-1; copy restores correct
        d = sub.get_d()

    if target_d > 0:
        # target_d is 1-indexed size; bsubs index target_d-1
        entry = bsubs[target_d - 1]
        if not entry["present"]:
            return None
        # reconstruct features from bin
        feats = [i for i, b in enumerate(entry["bin"]) if b > 0]
        return {"value": entry["critvalue"], "features": feats, "size": len(feats), "evaluations": crit.count}
    else:
        if not havemax:
            return None
        feats = maxcritsub.members()
        return {"value": maxcritval, "features": feats, "size": len(feats), "evaluations": crit.count}


class SFFS:
    """Stochastic SFFS (sSFFS) with budgeted steps. Defaults y=100/50."""

    def __init__(
        self,
        y: int = 100,
        y_back: int = 50,
        rho_u: float = 0.2,
        tau: float | None = None,
        temperature: float | None = None,
        T: float | None = None,
        sampler: str = "softmax",
        horizon: int = 100,
        warmup_probes: int = 200,
        warmup_card: int = 10,
        delta: int = 5,
        seed: int = 1,
        k: int = 1,
        cv_folds_n: int = 3,
        scaler: str = "to01",
        cap_frac: float = 0.0,
        step_frozen: bool = False,
    ):
        self.y = int(y)
        # aliases
        if tau is None and temperature is not None:
            tau = temperature
        if tau is None and T is not None:
            tau = T
        if tau is None:
            tau = 0.0
        self.tau = float(tau)
        self.y_back = int(y_back)
        self.rho_u = float(rho_u)
        self.sampler = sampler
        self.horizon = int(horizon)
        self.warmup_probes = int(warmup_probes)
        self.warmup_card = int(warmup_card)
        self.delta = int(delta)
        self.seed = int(seed)
        self.k = int(k)
        self.cv_folds_n = int(cv_folds_n)
        self.scaler = scaler
        self.cap_frac = float(cap_frac)
        self.step_frozen = bool(step_frozen)

    # support old names y vs step_cap, rho_u vs explore
    @property
    def explore(self):
        return self.rho_u

    def fit(self, X, y, d: int = 0):
        """Fit on numpy arrays, return selected feature indices list length d."""
        X = np.asarray(X, dtype=np.float64)
        yy = np.asarray(y)
        n_features = X.shape[1]
        # map labels to 0..C-1
        classes = np.unique(yy)
        label_to_idx = {lab: i for i, lab in enumerate(sorted(classes))}
        y_idx = np.array([label_to_idx[v] for v in yy], dtype=int)
        n_classes = len(classes)
        class_size = [int(np.sum(y_idx == c)) for c in range(n_classes)]
        total = X.shape[0]
        flat = np.zeros(total * n_features, dtype=np.float64)
        idx = 0
        for c in range(n_classes):
            rows = X[y_idx == c]
            for r in rows:
                row_f32 = r.astype(np.float32).astype(np.float64)
                for ff in range(n_features):
                    flat[idx] = float(row_f32[ff])
                    idx += 1
        data = ArffData(n_features, n_classes, class_size, flat)
        if self.scaler == "to01":
            scale_to01(data)
        return self._fit_arff(data, target_d=d, train_pct=50, test_pct=50, cv_folds_n=self.cv_folds_n, k=self.k)

    def fit_arff(self, data: ArffData, target_d: int, train_pct: int = 50, test_pct: int = 50, cv_folds_n: int | None = None, k: int | None = None):
        if cv_folds_n is None:
            cv_folds_n = self.cv_folds_n
        if k is None:
            k = self.k
        return self._fit_arff(data, target_d=target_d, train_pct=train_pct, test_pct=test_pct, cv_folds_n=cv_folds_n, k=k)

    def _fit_arff(self, data: ArffData, target_d: int, train_pct: int, test_pct: int, cv_folds_n: int, k: int):
        n = data.n_features
        if target_d >= n:
            raise ValueError("--target-d must be smaller than number of features")
        ss_srand(self.seed)
        # outer split
        outer = Split(data.n_classes)
        for c in range(data.n_classes):
            rr_split_class(data.class_size[c], train_pct, test_pct, outer.train[c], outer.test[c])
        # folds
        if cv_folds_n and cv_folds_n > 1:
            folds = cv_folds(outer, cv_folds_n, data.n_classes)
            inner = WrapperKnn(data, folds, k)
        else:
            # no CV: outer alone
            inner = WrapperKnn(data, [outer], k)
        crit = CountingCriterion(inner)
        weighter = Weighter(self.horizon)
        step = SampledStep(
            weighter,
            cap=self.y,
            cap_backward=self.y_back,
            cap_frac=self.cap_frac,
            explore=self.rho_u,
            tau=self.tau,
            sampler=self.sampler,
        )
        # warmup
        if self.warmup_probes > 0:
            r = self.warmup_card if self.warmup_card < n else n - 1
            weighter.reset(n)
            probe = Subset(n)
            for _ in range(self.warmup_probes):
                probe.make_random_subset(r)
                ok, pv = crit.evaluate(probe)
                if not ok:
                    raise RuntimeError("warm-up probe evaluation failed")
                weighter.add(pv, probe)
            weighter.flush_batch()
            if self.step_frozen:
                weighter.freeze()
        sub = Subset(n)
        res = sffs_search(target_d, self.delta, sub, crit, step, sys.stderr)
        if res is None:
            raise RuntimeError("search not finished")
        # store diagnostics
        self.evaluations_ = crit.count
        self.value_ = res["value"]
        self._all_evals = crit.count
        return res["features"]

    # convenience for filter criterion (multinom)
    def fit_filter(self, data: ArffData, target_d: int, train_pct: int = 50, test_pct: int = 40):
        """Filter criterion fit (multinomial Bhattacharyya) on ArffData."""
        n = data.n_features
        if target_d >= n:
            raise ValueError("--target-d must be smaller than number of features")
        ss_srand(self.seed)
        if self.scaler == "to01":
            # data already scaled? we copy? Assume caller handled; but if scaler to01 we scale in place
            scale_to01(data)
        outer_train = [[] for _ in range(data.n_classes)]
        outer_test = [[] for _ in range(data.n_classes)]
        for c in range(data.n_classes):
            rr_split_class(data.class_size[c], train_pct, test_pct, outer_train[c], outer_test[c])
        inner = MultinomBhattacharyya(data, outer_train)
        crit = CountingCriterion(inner)
        weighter = Weighter(self.horizon)
        step = SampledStep(
            weighter,
            cap=self.y,
            cap_backward=self.y_back,
            cap_frac=self.cap_frac,
            explore=self.rho_u,
            tau=self.tau,
            sampler=self.sampler,
        )
        if self.warmup_probes > 0:
            r = self.warmup_card if self.warmup_card < n else n - 1
            weighter.reset(n)
            probe = Subset(n)
            for _ in range(self.warmup_probes):
                probe.make_random_subset(r)
                ok, pv = crit.evaluate(probe)
                if not ok:
                    raise RuntimeError("warm-up probe evaluation failed")
                weighter.add(pv, probe)
            weighter.flush_batch()
            if self.step_frozen:
                weighter.freeze()
        sub = Subset(n)
        res = sffs_search(target_d, self.delta, sub, crit, step, sys.stderr)
        if res is None:
            raise RuntimeError("search not finished")
        self.evaluations_ = crit.count
        self.value_ = res["value"]
        return res["features"]
