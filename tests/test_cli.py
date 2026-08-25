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
