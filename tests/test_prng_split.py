def test_prng_sequence():
    from sss.prng import ss_rand, ss_srand

    ss_srand(1)
    first5 = [ss_rand() for _ in range(5)]
    assert first5 == [41, 18467, 6334, 26500, 19169]


def test_arff_sparse_reuters():
    from sss.arff import load_arff

    d = load_arff("anc/data/reuters_apte.arff")
    assert d.n_features == 10105
    assert d.n_classes == 33
    assert sum(d.class_size) == 8941
    # data flat size
    assert d.data.size == 8941 * 10105
    # check class_size respects per-class counts (should sum)
    assert len(d.class_size) == 33


def test_arff_dense_roundtrip(tmp_path):
    # create tiny dense ARFF with class last
    p = tmp_path / "tiny.arff"
    p.write_text("""@RELATION tiny
@ATTRIBUTE f1 numeric
@ATTRIBUTE f2 numeric
@ATTRIBUTE class {c0,c1}
@DATA
1.0,2.0,c0
3.0,4.0,c1
5.0,6.0,c0
""")
    from sss.arff import load_arff

    d = load_arff(str(p))
    assert d.n_features == 2
    assert d.n_classes == 2
    assert d.class_size == [2, 1]
    # class-major: c0 samples (1,2) and (5,6) first, then c1 (3,4)
    import numpy as np

    # flat = [1,2, 5,6, 3,4] in float64
    assert np.allclose(d.data[:2], [1.0, 2.0])
    assert np.allclose(d.data[2:4], [5.0, 6.0])
    assert np.allclose(d.data[4:6], [3.0, 4.0])


def test_arff_rejects_unterminated_sparse_row(tmp_path):
    from sss.arff import load_arff

    path = tmp_path / "unterminated.arff"
    path.write_text(
        "@RELATION tiny\n@ATTRIBUTE x NUMERIC\n@ATTRIBUTE class {a,b}\n@DATA\n{0 1, 1 a\n"
    )
    import pytest

    with pytest.raises(ValueError, match="unterminated sparse data row"):
        load_arff(str(path))


def test_scaler_to01():
    import numpy as np

    from sss.arff import ArffData, scale_to01

    # 2 features, 2 samples class-major
    data = np.array([0.0, 10.0, 10.0, 20.0], dtype=np.float64)
    d = ArffData(2, 1, [2], data)
    scale_to01(d)
    # f0: 0->0, 10->1 ; f1:10->0,20->1
    assert np.allclose(d.data, [0.0, 0.0, 1.0, 1.0])


def test_rr_split_counts():
    from sss.prng import ss_srand
    from sss.split import rr_split_class

    ss_srand(1)
    train, test = [], []
    rr_split_class(1000, 50, 50, train, test)
    assert len(train) == 500 and len(test) == 500
    assert set(train).isdisjoint(test)
    # edge perctrain >50
    ss_srand(1)
    train, test = [], []
    rr_split_class(100, 80, 10, train, test)
    assert len(train) == 80 and len(test) == 10


def test_rr_split_deterministic():
    from sss.prng import ss_srand
    from sss.split import rr_split_class

    ss_srand(123)
    t1, e1 = [], []
    rr_split_class(10, 50, 50, t1, e1)
    ss_srand(123)
    t2, e2 = [], []
    rr_split_class(10, 50, 50, t2, e2)
    assert t1 == t2 and e1 == e2


def test_cv_folds():
    from sss.split import Split, cv_folds

    outer = Split(2)
    outer.train = [[0, 1, 2, 3, 4], [10, 11, 12]]
    outer.test = [[], []]
    folds = cv_folds(outer, 3, 2)
    assert len(folds) == 3
    # each training sample appears in exactly (k-1) folds test sets? Check per class
    # class 0: 5 samples -> folds sizes 1,2,2? remaining//od logic: 5//3=1, 4//2=2, 2//1=2
    assert len(folds[0].test[0]) == 1
    assert len(folds[1].test[0]) == 2
    assert len(folds[2].test[0]) == 2
    # class 1: 3 samples -> 1,1,1
    assert [len(f.test[1]) for f in folds] == [1, 1, 1]


def test_subset_random_and_members():
    from sss.prng import ss_srand
    from sss.subset import Subset

    ss_srand(1)
    s = Subset(500)
    s.make_random_subset(10)
    assert s.get_d() == 10
    m = s.members()
    assert len(m) == 10 and len(set(m)) == 10
    # copy
    t = Subset(500)
    t.copy_members_from(s)
    assert t.members() == s.members()
    # forward mode flip — note deselect_all uses id_desel which is inverted in backward mode
    s.set_forward_mode(False)
    assert s.id_sel() == -1
    assert s.id_desel() == 1
    s.deselect_all()
    assert s.get_d() == 500  # id_desel==1 means "selected" marker in backward mode
    s.set_forward_mode(True)
    s.deselect_all()
    assert s.get_d() == 0
    s.select_raw(5)
    assert s.member(5)


def test_subset_forward_backward_inversion():
    from sss.subset import Subset

    s = Subset(5)
    s.deselect_all()
    s.select_raw(1)
    s.select_raw(2)
    assert s.get_d() == 2
    s.set_forward_mode(False)
    # in backward mode, select_raw removes, deselect_raw restores
    s.select_raw(1)  # should remove 1
    assert not s.member(1)
    assert s.get_d() == 1
    s.deselect_raw(1)  # restore
    assert s.member(1)
