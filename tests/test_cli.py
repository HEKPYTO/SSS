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


def test_cli_rejects_invalid_step_budget(tmp_path):
    import pytest

    from sss.cli import main

    path = tmp_path / "tiny.arff"
    path.write_text(
        "@RELATION tiny\n@ATTRIBUTE x NUMERIC\n@ATTRIBUTE class {a,b}\n@DATA\n0,a\n1,b\n"
    )
    with pytest.raises(SystemExit):
        main(["--data", str(path), "--criterion", "multinom-bhattacharyya", "--step-cap", "-1"])


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
    main(["--data", str(path), "--criterion", "multinom-bhattacharyya", "--step-tau", "-1"])
    assert received["temperature"] is None
