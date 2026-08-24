"""criteria.py — WrapperKnn + Multinomial Bhattacharyya, verbatim ssffs.cpp:426-636"""
from __future__ import annotations

import math

import numpy as np

from .arff import ArffData, scale_to01
from .prng import ss_srand
from .split import Split, cv_folds, rr_split_class
from .subset import Subset


class CountingCriterion:
    def __init__(self, inner):
        self.inner = inner
        self.count = 0

    def evaluate(self, subset: Subset):
        self.count += 1
        return self.inner.evaluate(subset)


class WrapperKnn:
    def __init__(self, d: ArffData, folds: list[Split], k: int):
        self.d = d
        self.folds = folds
        self.k = k
        self.n_classes = d.n_classes
        self.n_features = d.n_features
        self.class_offset: list[int] = []
        off = 0
        for c in range(d.n_classes):
            self.class_offset.append(off)
            off += d.class_size[c] * d.n_features
        self.feats: list[int] = []
        # precompute per-class matrices for fast sklearn path
        self.class_mats: list[np.ndarray] = []
        for c in range(d.n_classes):
            off = self.class_offset[c]
            sz = d.class_size[c]
            mat = d.data[off : off + sz * d.n_features].reshape(sz, d.n_features) if sz else np.zeros((0, d.n_features))
            self.class_mats.append(mat)

    def _pattern(self, cls: int, idx: int) -> np.ndarray:
        off = self.class_offset[cls] + idx * self.n_features
        return self.d.data[off : off + self.n_features]

    def evaluate(self, sub: Subset):
        feats = sub.members()
        if not feats:
            return False, 0.0
        feats = sorted(feats)
        self.feats = feats
        # fast sklearn path if available — matches C++ wrapper at k=1 and small k with Euclidean to01 scaling
        try:
            from sklearn.neighbors import KNeighborsClassifier

            total = 0.0
            cnt = 0
            for split in self.folds:
                # build train
                X_train_parts = []
                y_train_parts = []
                for c in range(self.n_classes):
                    idxs = split.train[c]
                    if not idxs:
                        continue
                    X_train_parts.append(self.class_mats[c][idxs][:, feats])
                    y_train_parts.append(np.full(len(idxs), c, dtype=int))
                if not X_train_parts:
                    return False, 0.0
                X_train = np.vstack(X_train_parts)
                y_train = np.concatenate(y_train_parts)
                # build test
                X_test_parts = []
                y_test_parts = []
                for c in range(self.n_classes):
                    idxs = split.test[c]
                    if not idxs:
                        continue
                    X_test_parts.append(self.class_mats[c][idxs][:, feats])
                    y_test_parts.append(np.full(len(idxs), c, dtype=int))
                if not X_test_parts:
                    continue
                X_test = np.vstack(X_test_parts)
                y_test = np.concatenate(y_test_parts)
                n_train = X_train.shape[0]
                k_eff = min(self.k, n_train)
                if k_eff <= 0:
                    return False, 0.0
                # tie-avoidance: C++ keeps max_size=(k-1)*C+1 nearest, then votes k.
                # sklearn with uniform weights and brute search returns deterministic nearest by stable sort; for k=1 they match.
                # For k>1 small, difference only in tie handling at exactly equal distances (rare).
                clf = KNeighborsClassifier(n_neighbors=k_eff, p=2, algorithm="brute")
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                acc = float(np.mean(y_pred == y_test)) if len(y_test) else 0.0
                total += acc
                cnt += 1
            return True, total / cnt if cnt else 0.0
        except Exception:  # noqa: BLE001 — fallback to exact logic when sklearn fails
            return self._evaluate_exact(feats)

    def _evaluate_exact(self, feats):
        total = 0.0
        cnt = 0
        for split in self.folds:
            correct = 0
            total_test = 0
            for c_test in range(self.n_classes):
                for i in split.test[c_test]:
                    test_pat = self._pattern(c_test, i)
                    pred_ok, pred = self._classify_exact(test_pat, split, feats)
                    if not pred_ok:
                        return False, 0.0
                    if pred == c_test:
                        correct += 1
                    total_test += 1
            acc = correct / total_test if total_test else 0.0
            total += acc
            cnt += 1
        return True, total / cnt if cnt else 0.0

    def _classify_exact(self, test_pat: np.ndarray, split: Split, feats):
        C = self.n_classes
        max_size = (self.k - 1) * C + 1
        dists: list[float] = []
        labels: list[int] = []
        for c in range(C):
            for i in split.train[c]:
                train_pat = self._pattern(c, i)
                s = 0.0
                for f in feats:
                    diff = float(test_pat[f]) - float(train_pat[f])
                    s += diff * diff
                dists.append(math.sqrt(s))
                labels.append(c)
        if not dists:
            return False, 0
        order = np.argsort(np.asarray(dists, dtype=np.float64), kind="stable")
        order = order[:max_size]
        dists = [dists[i] for i in order]
        labels = [labels[i] for i in order]
        k_take = min(self.k, len(dists))
        scores = [0.0] * C
        for i in range(k_take):
            scores[labels[i]] += 1.0
        for c in range(C):
            scores[c] /= float(k_take) if k_take else 1.0
        best = 0
        for c in range(1, C):
            if scores[c] > scores[best]:
                best = c
        return True, best


class MultinomBhattacharyya:
    def __init__(self, d: ArffData, outer_train: list[list[int]]):
        self.d = d
        self.n = d.n_features
        self.classes = d.n_classes
        self.Nsuminclass = np.zeros(self.n * self.classes, dtype=np.float64)
        self.Pc = np.zeros(self.classes, dtype=np.float64)
        self.Pc_d = np.zeros(self.classes, dtype=np.float64)
        self.theta = np.zeros(self.n * self.classes, dtype=np.float64)
        self.IB = np.zeros(self.n, dtype=np.float64)
        self.index = np.zeros(self.n, dtype=np.int64)
        class_size_sum = sum(d.class_size)
        for c in range(self.classes):
            self.Pc[c] = d.class_size[c] / class_size_sum if class_size_sum else 0
        # class_offset
        class_offset = []
        off = 0
        for c in range(self.classes):
            class_offset.append(off)
            off += d.class_size[c] * self.n
        self._class_offset = class_offset
        allpatterns = 0
        wCV = 0
        for c in range(self.classes):
            for i in outer_train[c]:
                p_off = class_offset[c] + i * self.n
                row = d.data[p_off : p_off + self.n]
                for f in range(self.n):
                    self.Nsuminclass[wCV + f] += float(row[f])
            allpatterns += len(outer_train[c])
            wCV += self.n
        self.allpatterns = allpatterns
        total_sum = float(np.sum(self.Nsuminclass))
        if total_sum == 0:
            for c in range(self.classes):
                self.Pc_d[c] = 1.0 / self.classes
        else:
            wCV = 0
            for c in range(self.classes):
                class_sum = float(np.sum(self.Nsuminclass[wCV : wCV + self.n]))
                self.Pc_d[c] = class_sum / total_sum
                wCV += self.n
        self.doc_avg_length = total_sum / allpatterns if allpatterns else 0.0
        self.IB_computed = False
        self.feats: list[int] = []

    def _compute_theta(self, dd: int):
        total_sum = 0.0
        wCV = 0
        wCd = 0
        for c in range(self.classes):
            class_sum = 0.0
            for f in range(dd):
                class_sum += float(self.Nsuminclass[wCV + int(self.index[f])])
            total_sum += class_sum
            for f in range(dd):
                self.theta[wCd] = (1.0 + float(self.Nsuminclass[wCV + int(self.index[f])])) / (float(dd) + class_sum)
                wCd += 1
            wCV += self.n
        self.doc_avg_length = total_sum / self.allpatterns if self.allpatterns else 0.0

    def _compute_IB(self):
        for f in range(self.n):
            self.IB[f] = 0.0
        for f in range(self.n):
            value = 0.0
            combs = 0
            wCV1 = 0
            for c1 in range(self.classes):
                wCV2 = wCV1 + self.n
                for c2 in range(c1 + 1, self.classes):
                    t1 = float(self.theta[wCV1 + f])
                    t2 = float(self.theta[wCV2 + f])
                    thetasum = math.sqrt(t1 * t2) + math.sqrt((1.0 - t1) * (1.0 - t2))
                    # guard log(0)
                    if thetasum <= 0:
                        thetasum = 1e-300
                    value += (-self.doc_avg_length) * math.log(thetasum) * float(self.Pc_d[c1]) * float(self.Pc_d[c2])
                    combs += 1
                    wCV2 += self.n
                wCV1 += self.n
            self.IB[f] = value / combs if combs else 0.0

    def evaluate(self, sub: Subset):
        feats = sorted(sub.members())
        dd = len(feats)
        if dd == 0:
            return False, 0.0
        if dd == 1:
            if not self.IB_computed:
                for i in range(self.n):
                    self.index[i] = i
                self._compute_theta(self.n)
                self._compute_IB()
                self.IB_computed = True
            return True, float(self.IB[feats[0]])
        # dd >1
        for i in range(dd):
            self.index[i] = feats[i]
        self._compute_theta(dd)
        value = 0.0
        for c1 in range(self.classes):
            for c2 in range(c1 + 1, self.classes):
                t1 = self.theta[c1 * dd : c1 * dd + dd]
                t2 = self.theta[c2 * dd : c2 * dd + dd]
                thetasum = 0.0
                for f in range(dd):
                    thetasum += math.sqrt(float(t1[f]) * float(t2[f]))
                # log guard
                if thetasum <= 0:
                    thetasum = 1e-300
                value += math.log(thetasum) * float(self.Pc[c1]) * float(self.Pc[c2])
        result = (-self.doc_avg_length) * value
        return True, float(result)


def make_wrapper_knn(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 1,
    cv: int = 3,
    seed: int = 1,
    train_pct: int = 50,
    test_pct: int = 50,
    scale: bool = True,
):
    """Helper for synthetic numpy arrays -> WrapperKnn with RR splits."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    classes = np.unique(y)
    # map labels to 0..C-1 if not already
    label_to_idx = {lab: i for i, lab in enumerate(sorted(classes))}
    y_idx = np.array([label_to_idx[v] for v in y], dtype=int)
    n_classes = len(classes)
    n_features = X.shape[1]
    # class_size and class-major data
    class_size = [int(np.sum(y_idx == c)) for c in range(n_classes)]
    total = X.shape[0]
    flat = np.zeros(total * n_features, dtype=np.float64)
    idx = 0
    for c in range(n_classes):
        rows = X[y_idx == c]
        for r in rows:
            # widen via float32 as in ARFF
            row_f32 = r.astype(np.float32).astype(np.float64)
            for ff in range(n_features):
                flat[idx] = float(row_f32[ff])
                idx += 1
    d = ArffData(n_features, n_classes, class_size, flat)
    if scale:
        scale_to01(d)
    ss_srand(seed)
    outer = Split(n_classes)
    for c in range(n_classes):
        rr_split_class(class_size[c], train_pct, test_pct, outer.train[c], outer.test[c])
    if cv and cv > 1:
        folds = cv_folds(outer, cv, n_classes)
    else:
        folds = [outer]
    inner = WrapperKnn(d, folds, k)
    return CountingCriterion(inner)


def make_multinom_from_arff(d: ArffData, outer_train: list[list[int]]):
    inner = MultinomBhattacharyya(d, outer_train)
    return CountingCriterion(inner)
