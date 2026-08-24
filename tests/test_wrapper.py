def test_sfffs_wrapper():
    import sklearn.datasets

    from src.sss.sss import SFFS

    X, y = sklearn.datasets.make_classification(
        n_samples=200, n_features=20, n_informative=5, random_state=0
    )
    sel = SFFS(y=5, rho_u=0.2, tau=1.0, warmup_probes=20, warmup_card=5, delta=5, seed=1).fit(
        X, y, d=5
    )
    assert len(sel) == 5
