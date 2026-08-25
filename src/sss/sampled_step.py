"""sampled_step.py — sADD/sRMV budgeted operators, verbatim ssffs.cpp:707-981"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .prng import frand
from .subset import Subset

if TYPE_CHECKING:
    from .weighter import Weighter


class SampledStep:
    def __init__(
        self,
        stats: Weighter,
        cap: int = 100,
        cap_backward: int = 50,
        cap_frac: float = 0.0,
        explore: float = 0.2,
        tau: float = 0.0,
        sampler: str = "softmax",
    ):
        self.stats = stats
        self.cap = int(cap)
        self.cap_backward = int(cap_backward)
        self.cap_frac = float(cap_frac)
        self.explore = max(0.0, min(1.0, float(explore)))
        self.tau = float(tau)
        self.sampler = sampler  # softmax|uniform|topk
        # instrumentation
        self.count_forward = 0
        self.count_backward = 0
        self.order_forward = 0.0
        self.order_backward = 0.0
        self.all_evals = 0

    def _note_step(self, forward: bool, order: int, cnt: int, pool: int, os):
        r = order / cnt if cnt else 0
        if forward:
            self.count_forward += 1
            self.order_forward = (
                (self.count_forward - 1) * self.order_forward + r
            ) / self.count_forward
            if os is None:
                return
            print(
                f"\nF order = {r}({order}) avgorder {self.order_forward} count = {self.count_forward} all_evals = {self.all_evals} pool = {pool} proposed = {cnt}",
                file=os,
                flush=True,
            )
        else:
            self.count_backward += 1
            self.order_backward = (
                (self.count_backward - 1) * self.order_backward + r
            ) / self.count_backward
            if os is None:
                return
            print(
                f"\nB order = {r}({order}) avgorder {self.order_backward} count = {self.count_backward} all_evals = {self.all_evals} pool = {pool} proposed = {cnt}",
                file=os,
                flush=True,
            )

    def _full_sweep(self, sub: Subset, crit, pool: int, os):
        forward = sub.forward
        bestval = None
        order = 0
        cnt = 0
        bestf = None
        # exhaustive over pool
        candidates = [f for f in range(sub.n) if (not sub.member(f) if forward else sub.member(f))]
        for f in candidates:
            # evaluate candidate: forward add, backward remove (via select_raw inversion)
            # C++ uses traversal marker (3/-3) then stats.add sees candidate; we replicate by adding before restore
            if forward:
                sub.select_raw(f)
            else:
                sub.select_raw(f)  # in backward mode removes
            ok, val = crit.evaluate(sub)
            if not ok:
                # restore before return
                sub.deselect_raw(f)
                return False, 0.0
            self.all_evals += 1
            tmp = Subset(sub.n)
            tmp.bin = sub.bin.copy()
            tmp.forward = sub.forward
            self.stats.add(val, tmp)
            cnt += 1
            if bestf is None or val > bestval:
                bestval = val
                order = cnt
                bestf = f
            sub.deselect_raw(f)
        if bestf is None:
            return False, 0.0
        self.stats.flush_batch()
        # apply winner
        sub.select_raw(bestf)
        self._note_step(forward, order, cnt, pool, os)
        return True, float(bestval)

    def _sampled_forward(self, sub: Subset, crit, pool: int, want: int, os):
        n = sub.n
        freef = [f for f in range(n) if not sub.member(f)]
        # proposal
        prop: list[int] = []
        taken = [0] * n
        if self.sampler == "topk":
            byscore = sorted(freef, key=lambda a: (-self.stats.score(a), a))
            for i in range(want):
                prop.append(byscore[i])
        else:
            u = int(self.explore * want + 0.5) if self.sampler != "uniform" else want
            u = min(u, want)
            if u > 0:
                bycnt = sorted(freef, key=lambda a: (self.stats.count_is(a), a))
                le = pool if self.sampler == "uniform" else min(pool, max(4 * u, u))
                # partial sort up to le already sorted; keep first le
                if le < pool:
                    # keep first le sorted, rest not considered for floor
                    bycnt = bycnt[:le]
                    le = len(bycnt)
                else:
                    le = len(bycnt)
                for k in range(u):
                    # uniform draw from [k, le)
                    if le - k <= 0:
                        break
                    j = k + int(frand() * (le - k))
                    # swap
                    bycnt[k], bycnt[j] = bycnt[j], bycnt[k]
                    prop.append(bycnt[k])
                    taken[bycnt[k]] = 1
            rest = want - u
            if rest > 0:
                cand = [f for f in freef if not taken[f]]
                sc = [self.stats.score(f) for f in cand]
                # tau
                t = self.tau
                if t <= 0:
                    tmp = sorted(sc)
                    if tmp:
                        q1 = len(tmp) // 4
                        q3 = (3 * len(tmp)) // 4
                        a = tmp[q1]
                        b = tmp[q3]
                        t = (b - a) / 1.349
                        if not (t > 1e-9):
                            t = 1e-9
                    else:
                        t = 1e-9
                if not sc:
                    pass
                else:
                    smax = max(sc)
                    w = [math.exp((s - smax) / t) if t != 0 else 0.0 for s in sc]
                    W = sum(w)
                    for _ in range(rest):
                        # C++ never breaks when W==0; it falls through to argmax
                        r = frand() * W
                        pick = None
                        acc = 0.0
                        for i, wi in enumerate(w):
                            if wi <= 0:
                                continue
                            acc += wi
                            if r < acc:
                                pick = i
                                break
                        if pick is None:
                            # fallback last positive
                            for i in range(len(w) - 1, -1, -1):
                                if w[i] > 0:
                                    pick = i
                                    break
                        if pick is None:
                            # underflow guard -> argmax (C++ complete-underflow)
                            best = None
                            best_sc = None
                            for i in range(len(sc)):
                                if w[i] >= 0 and (best is None or sc[i] > best_sc):
                                    best = i
                                    best_sc = sc[i]
                            pick = best
                            if pick is None:
                                break
                        prop.append(cand[pick])
                        if w[pick] > 0:
                            W -= w[pick]
                        w[pick] = -1.0
        # evaluate batch
        bestval = None
        bestf = None
        order = 0
        cnt = 0
        for f in prop:
            sub.select_raw(f)
            ok, val = crit.evaluate(sub)
            if not ok:
                sub.deselect_raw(f)
                return False, 0.0
            self.all_evals += 1
            tmp = Subset(sub.n)
            tmp.bin = sub.bin.copy()
            tmp.forward = sub.forward
            self.stats.add(val, tmp)
            cnt += 1
            if bestf is None or val > bestval:
                bestval = val
                bestf = f
                order = cnt
            sub.deselect_raw(f)
        self.stats.flush_batch()
        if bestf is None:
            return False, 0.0
        sub.select_raw(bestf)
        self._note_step(True, order, cnt, pool, os)
        return True, float(bestval)

    def _sampled_backward(self, sub: Subset, crit, pool: int, want: int, os):
        n = sub.n
        memb = [f for f in range(n) if sub.member(f)]
        prop: list[int] = []
        taken = [0] * n
        if self.sampler == "topk":
            byscore = sorted(memb, key=lambda a: (self.stats.score(a), a))
            for i in range(want):
                prop.append(byscore[i])
        else:
            u = int(self.explore * want + 0.5) if self.sampler != "uniform" else want
            u = min(u, want)
            if u > 0:
                bycnt = sorted(memb, key=lambda a: (self.stats.count_isnot(a), a))
                le = pool if self.sampler == "uniform" else min(pool, max(4 * u, u))
                if le < pool:
                    bycnt = bycnt[:le]
                    le = len(bycnt)
                else:
                    le = len(bycnt)
                for k in range(u):
                    if le - k <= 0:
                        break
                    j = k + int(frand() * (le - k))
                    bycnt[k], bycnt[j] = bycnt[j], bycnt[k]
                    prop.append(bycnt[k])
                    taken[bycnt[k]] = 1
            rest = want - u
            if rest > 0:
                cand = [f for f in memb if not taken[f]]
                sc = [self.stats.score(f) for f in cand]
                t = self.tau
                if t <= 0:
                    tmp = sorted(sc)
                    if tmp:
                        q1 = len(tmp) // 4
                        q3 = (3 * len(tmp)) // 4
                        a = tmp[q1]
                        b = tmp[q3]
                        t = (b - a) / 1.349
                        if not (t > 1e-9):
                            t = 1e-9
                    else:
                        t = 1e-9
                if sc:
                    smin = min(sc)
                    w = [math.exp((smin - s) / t) if t != 0 else 0.0 for s in sc]
                    W = sum(w)
                    for _ in range(rest):
                        r = frand() * W
                        pick = None
                        acc = 0.0
                        for i, wi in enumerate(w):
                            if wi <= 0:
                                continue
                            acc += wi
                            if r < acc:
                                pick = i
                                break
                        if pick is None:
                            for i in range(len(w) - 1, -1, -1):
                                if w[i] > 0:
                                    pick = i
                                    break
                        if pick is None:
                            best = None
                            best_sc = None
                            for i in range(len(sc)):
                                if w[i] >= 0 and (best is None or sc[i] < best_sc):
                                    best = i
                                    best_sc = sc[i]
                            pick = best
                            if pick is None:
                                break
                        prop.append(cand[pick])
                        if w[pick] > 0:
                            W -= w[pick]
                        w[pick] = -1.0
        # evaluate
        bestval = None
        bestf = None
        order = 0
        cnt = 0
        for f in prop:
            sub.select_raw(f)  # backward mode removes
            ok, val = crit.evaluate(sub)
            if not ok:
                sub.deselect_raw(f)
                return False, 0.0
            self.all_evals += 1
            tmp = Subset(sub.n)
            tmp.bin = sub.bin.copy()
            tmp.forward = sub.forward
            self.stats.add(val, tmp)
            cnt += 1
            if bestf is None or val > bestval:
                bestval = val
                bestf = f
                order = cnt
            sub.deselect_raw(f)
        self.stats.flush_batch()
        if bestf is None:
            return False, 0.0
        sub.select_raw(bestf)
        self._note_step(False, order, cnt, pool, os)
        return True, float(bestval)

    def evaluate_candidates(self, sub: Subset, crit, os):
        if self.stats.n != sub.n:
            self.stats.reset(sub.n)
        forward = sub.forward
        pool = (sub.n - sub.get_d()) if forward else sub.get_d()
        if pool == 0:
            return False, 0.0
        if forward:
            c = self.cap
            if self.cap_frac > 0:
                import math as _m

                c = _m.ceil(self.cap_frac * pool)
                c = max(c, 1)
            if c == 0 or pool <= c:
                return self._full_sweep(sub, crit, pool, os)
            return self._sampled_forward(sub, crit, pool, c, os)
        # backward
        if self.cap_backward == 0 or pool <= self.cap_backward:
            return self._full_sweep(sub, crit, pool, os)
        return self._sampled_backward(sub, crit, pool, self.cap_backward, os)

    def Step(self, forward: bool, sub: Subset, crit, os=None):
        mode_change = sub.forward != forward
        if mode_change:
            sub.set_forward_mode(forward)
        ok, val = self.evaluate_candidates(sub, crit, os)
        if mode_change:
            sub.set_forward_mode(not forward)
        return ok, val
