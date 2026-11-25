import json


def test_cli_prints_one_final_json_without_progress(tmp_path, capsys):
    from sss.cli import main

    path = tmp_path / "tiny.arff"
    path.write_text(
        "@RELATION tiny\n@ATTRIBUTE x NUMERIC\n@ATTRIBUTE y NUMERIC\n"
        "@ATTRIBUTE class {a,b}\n@DATA\n0,0,a\n0,1,a\n1,0,b\n1,1,b\n"
    )
    assert (
        main(
            [
                "--data",
                str(path),
                "--criterion",
                "multinom-bhattacharyya",
                "--target-d",
                "1",
                "--step-cap",
                "2",
                "--warmup-probes",
                "0",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 1
    assert json.loads(captured.out)["features"]
    assert captured.err == ""


def test_cli_progress_is_opt_in(tmp_path, capsys):
    from sss.cli import main

    path = tmp_path / "tiny.arff"
    path.write_text(
        "@RELATION tiny\n@ATTRIBUTE x NUMERIC\n@ATTRIBUTE y NUMERIC\n"
        "@ATTRIBUTE class {a,b}\n@DATA\n0,0,a\n1,1,b\n"
    )
    main(
        [
            "--data",
            str(path),
            "--criterion",
            "multinom-bhattacharyya",
            "--target-d",
            "1",
            "--step-cap",
            "1",
            "--warmup-probes",
            "0",
            "--progress",
        ]
    )
    assert '"event":"solution"' in capsys.readouterr().err


def test_cli_invalid_step_budget_returns_one(tmp_path, capsys):
    from sss.cli import main

    path = tmp_path / "tiny.arff"
    path.write_text(
        "@RELATION tiny\n@ATTRIBUTE x NUMERIC\n@ATTRIBUTE class {a,b}\n@DATA\n0,a\n1,b\n"
    )
    assert (
        main(["--data", str(path), "--criterion", "multinom-bhattacharyya", "--step-cap", "-1"])
        == 1
    )
    assert capsys.readouterr().err == "ssffs: budgets must be non-negative\n"


def test_cli_unreadable_data_returns_one_with_reference_prefix(tmp_path, capsys):
    from sss.cli import main

    missing = tmp_path / "missing.arff"
    assert main(["--data", str(missing), "--criterion", "multinom-bhattacharyya"]) == 1
    assert capsys.readouterr().err == f"ssffs: cannot open '{missing}'\n"


def test_cli_value_uses_reference_precision(tmp_path, monkeypatch, capsys):
    from sss.cli import main
    from sss.sss import Result

    path = tmp_path / "tiny.arff"
    path.write_text(
        "@RELATION tiny\n@ATTRIBUTE x NUMERIC\n@ATTRIBUTE class {a,b}\n@DATA\n0,a\n1,b\n"
    )
    monkeypatch.setattr("sss.cli._run", lambda *_args, **_kwargs: Result((0,), 1 / 3, 1))
    assert main(["--data", str(path), "--criterion", "multinom-bhattacharyya"]) == 0
    assert '"value": 0.333333333333,' in capsys.readouterr().out


def test_cli_clamps_explore_like_reference(tmp_path, capsys):
    from sss.cli import main

    path = tmp_path / "tiny.arff"
    path.write_text(
        "@RELATION tiny\n@ATTRIBUTE x NUMERIC\n@ATTRIBUTE y NUMERIC\n"
        "@ATTRIBUTE class {a,b}\n@DATA\n0,0,a\n0,1,a\n1,0,b\n1,1,b\n"
    )
    base = [
        "--data",
        str(path),
        "--criterion",
        "multinom-bhattacharyya",
        "--target-d",
        "1",
        "--step-cap",
        "1",
    ]
    assert main([*base, "--step-explore", "1.0"]) == 0
    clamped = capsys.readouterr().out
    assert main([*base, "--step-explore", "1.5"]) == 0
    assert capsys.readouterr().out == clamped


def test_cli_uses_auto_temperature_for_nonpositive_tau(tmp_path, monkeypatch):
    from sss.cli import main
    from sss.sss import Result

    path = tmp_path / "tiny.arff"
    path.write_text(
        "@RELATION tiny\n@ATTRIBUTE x NUMERIC\n@ATTRIBUTE class {a,b}\n@DATA\n0,a\n1,b\n"
    )
    received = {}

    def run(*_args, **kwargs):
        received.update(kwargs)
        return Result((0,), 1.0, 1)

    monkeypatch.setattr("sss.cli._run", run)
    for tau in ("0", "-1"):
        main(["--data", str(path), "--criterion", "multinom-bhattacharyya", "--step-tau", tau])
        assert received["temperature"] is None
