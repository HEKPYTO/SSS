"""subset.py — Feature subset, verbatim ssffs.cpp:391-424"""

from __future__ import annotations

from .prng import ss_rand


class Subset:
    def __init__(self, n: int):
        self.n = n
        self.bin: list[int] = [-1] * n
        self.forward: bool = True

    def id_sel(self) -> int:
        return 1 if self.forward else -1

    def id_desel(self) -> int:
        return -1 if self.forward else 1

    def id_traverse(self) -> int:
        return 3 if self.forward else -3

    def set_forward_mode(self, fwd: bool) -> None:
        self.forward = fwd

    def deselect_all(self) -> None:
        v = self.id_desel()
        for i in range(self.n):
            self.bin[i] = v

    def select_raw(self, f: int) -> None:
        self.bin[f] = self.id_sel()

    def deselect_raw(self, f: int) -> None:
        self.bin[f] = self.id_desel()

    def member(self, f: int) -> bool:
        return self.bin[f] > 0

    def get_d(self) -> int:
        return sum(1 for b in self.bin if b > 0)

    def members(self, out: list[int] | None = None) -> list[int]:
        lst = [i for i, b in enumerate(self.bin) if b > 0]
        if out is not None:
            out.clear()
            out.extend(lst)
            return out
        return lst

    def copy_members_from(self, other: Subset) -> None:
        for i in range(self.n):
            self.bin[i] = 1 if other.bin[i] > 0 else -1

    def make_random_subset(self, d: int) -> None:
        # reference PRNG consumption: d draws + linear probing
        self.deselect_all()
        for _ in range(d):
            piv = ss_rand() % self.n
            while self.bin[piv] != self.id_desel():
                piv += 1
                if piv > self.n - 1:
                    piv = 0
            self.bin[piv] = self.id_sel()
