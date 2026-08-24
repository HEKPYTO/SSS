def test_ssfs_wrapper_small():
    import sklearn.datasets

    from src.sss.sss import SFFS

    X, y = sklearn.datasets.make_classification(
        n_samples=200, n_features=20, n_informative=5, random_state=0
    )
    sel = SFFS(y=5, rho_u=0.2, tau=1.0, warmup_probes=20, warmup_card=5, delta=5, seed=1).fit(
        X, y, d=5
    )
    assert len(sel) == 5
    assert len(set(sel)) == 5
    assert all(0 <= f < 20 for f in sel)


def test_sss_cli_reuters_smoke():
    import json
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "-m",
        "src.sss.cli",
        "--data",
        "anc/data/reuters_apte.arff",
        "--rr-train",
        "50",
        "--rr-test",
        "40",
        "--scaler",
        "void",
        "--criterion",
        "multinom-bhattacharyya",
        "--target-d",
        "5",
        "--sffs-delta",
        "5",
        "--step-cap",
        "10",
        "--step-cap-backward",
        "5",
        "--warmup-probes",
        "10",
        "--warmup-card",
        "5",
        "--seed",
        "1",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr[-500:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["size"] == 5
    assert "evaluations" in out
    assert out["value"] > 0


def test_sss_filter_fit():
    from src.sss.arff import load_arff
    from src.sss.sss import SFFS

    d = load_arff("anc/data/reuters_apte.arff")
    # use SFFS filter path with tiny budget for speed
    sffs = SFFS(
        y=10,
        y_back=5,
        rho_u=0.2,
        tau=0.0,
        warmup_probes=5,
        warmup_card=5,
        delta=2,
        seed=1,
        scaler="void",
    )
    sel = sffs.fit_filter(d, target_d=3, train_pct=50, test_pct=40)
    assert len(sel) == 3


def test_sffs_search_emits_solution():
    import io

    import numpy as np

    from src.sss.prng import ss_srand
    from src.sss.sampled_step import SampledStep
    from src.sss.sss import sffs_search
    from src.sss.subset import Subset
    from src.sss.weighter import Weighter

    # tiny synthetic 10-d
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(40, 10))
    y = np.array([0] * 20 + [1] * 20)
    X[:20, 0] += 2
    # build Arff-like

    _n_features, _n_classes = 10, 2
    np.zeros(40 * 10, dtype=np.float64)
    # class-major already as X is not sorted but we use direct
    # simplify: make wrapper via helper but test low level
    from src.sss.criteria import make_wrapper_knn

    crit = make_wrapper_knn(X, y, k=1, cv=2, seed=1, train_pct=50, test_pct=50, scale=False)
    weighter = Weighter(100)
    step = SampledStep(weighter, cap=5, cap_backward=2, explore=0.2, tau=0.0, sampler="softmax")
    # warmup
    Subset(10)
    ss_srand(1)
    weighter.reset(10)
    probe = Subset(10)
    for _ in range(5):
        probe.make_random_subset(2)
        _ok, pv = crit.evaluate(probe)
        weighter.add(pv, probe)
    weighter.flush_batch()
    out_io = io.StringIO()
    res = sffs_search(target_d=3, delta=2, sub=Subset(10), crit=crit, step=step, os=out_io)
    assert res is not None
    assert res["size"] == 3
    assert "value" in res
    txt = out_io.getvalue()
    assert '"event":"solution"' in txt
