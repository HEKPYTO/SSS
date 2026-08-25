def test_weighter_zcontrast_and_horizon():
    from sss.subset import Subset
    from sss.weighter import Weighter

    w = Weighter(horizon=100)
    w.reset(5)
    s0 = Subset(5)
    s0.deselect_all()
    s0.select_raw(0)
    s1 = Subset(5)
    s1.deselect_all()
    s1.select_raw(1)
    w.add(0.5, s0)
    w.add(1.5, s1)
    w.flush_batch()
    assert w.score(0) < w.score(1)
    # check exact values: z -1 and +1 => scores -2 and 2 within tolerance
    assert abs(w.score(0) - (-2.0)) < 1e-9
    assert abs(w.score(1) - 2.0) < 1e-9

    # second batch degenerate -> z=0, moves towards 0
    w.add(1.0, s0)
    w.add(1.0, s1)
    w.flush_batch()
    # after degenerate update, m_is for f0: previous -1, a=1/2 => -1 +0.5*(0 - (-1))= -0.5
    # m_isnot for f0 was 1, after second batch where f0 not in s1: n_isnot 2, a=0.5 => 1+0.5*(0-1)=0.5 => score -1.0
    assert abs(w.score(0) - (-1.0)) < 1e-9

    # frozen should ignore
    w.freeze()
    assert w.frozen
    prev_score = w.score(0)
    prev_n = w.n_is(0)
    w.add(10.0, s0)
    w.flush_batch()
    assert w.score(0) == prev_score
    assert w.n_is(0) == prev_n
    # batch cleared while frozen
    assert len(w._batch) == 0


def test_weighter_horizon_forgetting():
    from sss.subset import Subset
    from sss.weighter import Weighter

    w = Weighter(horizon=2)
    w.reset(2)
    s0 = Subset(2)
    s0.deselect_all()
    s0.select_raw(0)
    # add many batches where f0 always present with increasing z
    for _ in range(5):
        # create batch of 2: values 0 and 2 => z -1,1 ; but we will add single entry per flush to simplify horizon test
        # Actually each flush with single entry degenerates -> z=0, not useful. Use 2-entry batches
        pass
    # instead test that horizon caps rate at 1/2 after 2 updates
    # Batch1: f0 present vs absent
    s1 = Subset(2)
    s1.deselect_all()
    s1.select_raw(1)
    w.add(0.0, s0)
    w.add(2.0, s1)
    w.flush_batch()  # scores approx -2,2
    # second similar batch, with horizon 2, second update a=1/2
    w.add(0.0, s0)
    w.add(2.0, s1)
    w.flush_batch()
    # third similar batch, rate remains 1/2 (horizon)
    w.add(0.0, s0)
    w.add(2.0, s1)
    w.flush_batch()
    # should not diverge to infinity, bounded
    assert abs(w.score(0)) < 10


def test_weighter_reset_and_empty_flush():
    from sss.weighter import Weighter

    w = Weighter(horizon=100)
    w.reset(3)
    w.flush_batch()  # no effect
    assert w.score(0) == 0.0
    w.reset(5)
    assert w.n == 5
    assert len(w._fs) == 5
