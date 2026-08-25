import io


class DummyCriterion:
    def __init__(self, n):
        self.n = n
        self.count = 0

    def evaluate(self, sub):
        self.count += 1
        # deterministic value proportional to max selected index (to make best predictable)
        m = sub.members()
        if not m:
            return False, 0.0
        v = max(m) / self.n
        return True, float(v)


def test_sampled_forward_respects_budget_and_floor():
    from sss.prng import ss_srand
    from sss.sampled_step import SampledStep
    from sss.subset import Subset
    from sss.weighter import Weighter

    ss_srand(1)
    n = 100
    w = Weighter(horizon=100)
    w.reset(n)
    crit = DummyCriterion(n)
    step = SampledStep(
        w, cap=25, cap_backward=0, cap_frac=0.0, explore=0.2, tau=0.0, sampler="softmax"
    )
    sub = Subset(n)
    sub.deselect_all()
    ok, _val = step.Step(True, sub, crit, io.StringIO())
    assert ok and sub.get_d() == 1
    assert crit.count == 25
    assert w.n == n
    # exactly 25 evals counted, explore floor u=5
    assert step.all_evals == 25
    assert step.count_forward == 1

    # second step should also respect budget
    ok, _val = step.Step(True, sub, crit, io.StringIO())
    assert ok and sub.get_d() == 2
    assert crit.count == 50


def test_full_sweep_fallback():
    from sss.prng import ss_srand
    from sss.sampled_step import SampledStep
    from sss.subset import Subset
    from sss.weighter import Weighter

    ss_srand(42)
    n = 10
    w = Weighter(horizon=100)
    w.reset(n)
    crit = DummyCriterion(n)
    # cap larger than pool -> full sweep
    step = SampledStep(
        w, cap=100, cap_backward=0, cap_frac=0.0, explore=0.2, tau=0.0, sampler="softmax"
    )
    sub = Subset(n)
    sub.deselect_all()
    # add 8 features so pool =2 <= cap => full sweep should evaluate pool=2 not cap
    for f in range(8):
        sub.select_raw(f)
    pool = n - sub.get_d()
    assert pool == 2
    ok, _val = step.Step(True, sub, crit, io.StringIO())
    assert ok
    assert crit.count == 2  # full sweep, not 100
    assert sub.get_d() == 9

    # cap=0 means full sweep always
    ss_srand(1)
    w2 = Weighter(horizon=100)
    w2.reset(n)
    crit2 = DummyCriterion(n)
    step2 = SampledStep(
        w2, cap=0, cap_backward=0, cap_frac=0.0, explore=0.2, tau=0.0, sampler="uniform"
    )
    sub2 = Subset(n)
    sub2.deselect_all()
    ok, _val = step2.Step(True, sub2, crit2, io.StringIO())
    assert ok
    assert crit2.count == n  # pool 10 -> full sweep 10
    assert step2.all_evals == 10


def test_sampled_backward():
    from sss.prng import ss_srand
    from sss.sampled_step import SampledStep
    from sss.subset import Subset
    from sss.weighter import Weighter

    ss_srand(7)
    n = 20
    w = Weighter(horizon=100)
    w.reset(n)
    DummyCriterion(n)

    class InvertedDummy:
        # for backward, best removal is smallest index (since we invert score)
        def __init__(self, n):
            self.count = 0
            self.n = n

        def evaluate(self, sub):
            self.count += 1
            m = sub.members()
            if not m:
                return False, 0.0
            # value higher when subset smaller? Use negative max to prefer removing large indices
            # For backward test, we want deterministic: prefer removing smallest score (lowest index)
            # So value = -min(m) -> larger when min larger? Let's just use sum
            return True, float(sum(m))

    crit2 = InvertedDummy(n)
    step = SampledStep(
        w, cap=10, cap_backward=5, cap_frac=0.0, explore=0.2, tau=1.0, sampler="softmax"
    )
    sub = Subset(n)
    sub.deselect_all()
    for f in range(10):
        sub.select_raw(f)
    assert sub.get_d() == 10
    ok, _val = step.Step(False, sub, crit2, io.StringIO())
    assert ok
    assert sub.get_d() == 9
    assert crit2.count == 5
    assert step.count_backward == 1


def test_uniform_sampler():
    from sss.prng import ss_srand
    from sss.sampled_step import SampledStep
    from sss.subset import Subset
    from sss.weighter import Weighter

    ss_srand(123)
    n = 50
    w = Weighter(horizon=100)
    w.reset(n)
    crit = DummyCriterion(n)
    step = SampledStep(w, cap=10, cap_backward=0, explore=0.2, tau=0.0, sampler="uniform")
    sub = Subset(n)
    sub.deselect_all()
    ok, _val = step.Step(True, sub, crit, io.StringIO())
    assert ok and sub.get_d() == 1
    assert crit.count == 10


def test_topk_sampler():
    from sss.prng import ss_srand
    from sss.sampled_step import SampledStep
    from sss.subset import Subset
    from sss.weighter import Weighter

    ss_srand(99)
    n = 30
    w = Weighter(horizon=100)
    w.reset(n)
    # give scores: feature 29 highest
    from sss.subset import Subset as S

    # need to populate stats so topk picks highest: add evaluations where subsets containing 29 have high value
    # simple: do a flush with known values
    # create two subsets: {29} high, {0} low
    s_high = S(n)
    s_high.deselect_all()
    s_high.select_raw(29)
    s_low = S(n)
    s_low.deselect_all()
    s_low.select_raw(0)
    w.add(1.0, s_high)
    w.add(0.0, s_low)
    w.flush_batch()
    assert w.score(29) > w.score(0)
    crit = DummyCriterion(n)
    step = SampledStep(w, cap=5, cap_backward=0, explore=0.0, tau=1.0, sampler="topk")
    sub = Subset(n)
    sub.deselect_all()
    ok, _val = step.Step(True, sub, crit, io.StringIO())
    assert ok
    # topk should have proposed highest scoring features; with Dummy value max index, 29 is best and should be among top 5
    # Since prop is sorted descending score, and 29 has highest score, it will be in prop, and maximize v => should pick 29
    assert 29 in sub.members()


def test_softmax_consumes_rng_after_weight_underflow():
    from sss.prng import ss_rand, ss_srand
    from sss.sampled_step import SampledStep
    from sss.subset import Subset
    from sss.weighter import Weighter

    class Criterion:
        def evaluate(self, sub):
            return True, float(sum(sub.members()))

    stats = Weighter()
    stats.reset(4)
    for feature in range(1, 4):
        stats._fs[feature]["m_is"] = -10_000.0
    step = SampledStep(stats, cap=3, explore=0.0, tau=1.0)
    subset = Subset(4)
    subset.deselect_all()
    ss_srand(1)

    assert step.Step(True, subset, Criterion(), io.StringIO())[0]
    assert ss_rand() == 26500
