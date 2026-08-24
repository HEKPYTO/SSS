"""split.py — RR splitter + CV folds, verbatim ssffs.cpp:332-390"""
from __future__ import annotations

from dataclasses import dataclass

from .prng import ss_rand


@dataclass
class Split:
    train: list[list[int]]  # per class
    test: list[list[int]]

    def __init__(self, n_classes: int = 0):
        self.train = [[] for _ in range(n_classes)]
        self.test = [[] for _ in range(n_classes)]


def rr_split_class(n: int, perctrain: int, perctest: int, train: list[int], test: list[int]) -> None:
    """Class-stratified RR(1) split, mutates train/test lists. ssffs.cpp:343-369"""
    id_empty, id_train, id_test = 0, 1, 2
    mark = [id_empty] * n
    trsiz = (n * perctrain) // 100
    tesiz = (n * perctest) // 100

    def fill_randomly(_from: int, _to: int, count: int):
        for _ in range(count):
            piv = ss_rand() % n
            while mark[piv] != _from:
                piv += 1
                if piv > n - 1:
                    piv = 0
            mark[piv] = _to

    def fill(_from: int, _to: int):
        for i in range(n):
            if mark[i] == _from:
                mark[i] = _to

    if perctrain <= 50:
        fill_randomly(id_empty, id_train, trsiz)
    else:
        fill(id_empty, id_train)
        fill_randomly(id_train, id_empty, n - trsiz)
    if perctrain + perctest == 100:
        fill(id_empty, id_test)
    elif perctest <= (100 - perctrain) // 2:
        fill_randomly(id_empty, id_test, tesiz)
    else:
        fill(id_empty, id_test)
        fill_randomly(id_test, id_empty, n - trsiz - tesiz)
    train.clear()
    test.clear()
    for i in range(n):
        if mark[i] == id_train:
            train.append(i)
        elif mark[i] == id_test:
            test.append(i)


def cv_folds(outer: Split, kfold: int, n_classes: int) -> list[Split]:
    """k-fold CV of outer.train, ssffs.cpp:373-389"""
    folds: list[Split] = []
    for _ in range(kfold):
        s = Split(n_classes)
        folds.append(s)
    for c in range(n_classes):
        tr = outer.train[c]
        remaining = len(tr)
        od = kfold
        start = 0
        for k in range(kfold):
            tcs = remaining // od if od else 0
            for i, val in enumerate(tr):
                if start <= i < start + tcs:
                    folds[k].test[c].append(val)
                else:
                    folds[k].train[c].append(val)
            start += tcs
            remaining -= tcs
            od -= 1
    return folds
