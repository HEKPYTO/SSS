import json
import os
import re
import subprocess
import sys

import pytest

PARITY_ARGS = [
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

_STEP = re.compile(
    r"^(?P<direction>[FB]) order = [^(]+\((?P<order>\d+)\).*all_evals = "
    r"(?P<evaluations>\d+) pool = (?P<pool>\d+) proposed = (?P<proposed>\d+)$"
)


def _run(command, *, env=None):
    result = subprocess.run(command, capture_output=True, text=True, timeout=90, env=env)
    assert result.returncode == 0, result.stderr
    events = []
    diagnostics = []
    for line in result.stderr.splitlines():
        if line.startswith('{"event":"solution"'):
            event = json.loads(line)
            events.append((event["d"], event["features"], event["value"]))
        elif match := _STEP.match(line):
            diagnostics.append(
                (
                    match["direction"],
                    int(match["order"]),
                    int(match["proposed"]),
                    int(match["pool"]),
                    int(match["evaluations"]),
                )
            )
    return json.loads(result.stdout), events, diagnostics


def test_reuters_trajectory_matches_cpp(tmp_path):
    oracle = tmp_path / "ssffs"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O2",
            "-funroll-loops",
            "-ffp-contract=off",
            "-DNDEBUG",
            "-o",
            str(oracle),
            "anc/ssffs/ssffs.cpp",
        ],
        check=True,
    )
    cpp_final, cpp_events, cpp_diagnostics = _run([str(oracle), *PARITY_ARGS])
    environment = os.environ | {"PYTHONPATH": "src"}
    python_final, python_events, python_diagnostics = _run(
        [sys.executable, "-m", "sss.cli", *PARITY_ARGS, "--progress"], env=environment
    )

    assert cpp_final["features"] == [71, 110, 114, 152, 6611]
    assert cpp_final["evaluations"] == 659
    assert python_final["features"] == cpp_final["features"]
    assert python_final["evaluations"] == cpp_final["evaluations"]
    assert python_final["value"] == pytest.approx(cpp_final["value"], abs=1e-11)
    assert [(d, features) for d, features, _value in python_events] == [
        (d, features) for d, features, _value in cpp_events
    ]
    assert [value for _d, _features, value in python_events] == pytest.approx(
        [value for _d, _features, value in cpp_events], abs=1e-6
    )
    assert python_diagnostics == cpp_diagnostics


def test_wrapper_knn_trajectory_matches_cpp_on_scaled_cv_fixture(tmp_path):
    data = tmp_path / "wrapper.arff"
    data.write_text(
        """@RELATION wrapper
@ATTRIBUTE signal NUMERIC
@ATTRIBUTE noise NUMERIC
@ATTRIBUTE shape NUMERIC
@ATTRIBUTE class {left,right}
@DATA
0,0,0,left
0.1,1,0.2,left
0.2,0,0.4,left
0.3,1,0.6,left
0.4,0,0.8,left
0.5,1,1,left
0.6,0,1.2,left
0.7,1,1.4,left
10,1,1,right
10.1,0,1.2,right
10.2,1,1.4,right
10.3,0,1.6,right
10.4,1,1.8,right
10.5,0,2,right
10.6,1,2.2,right
10.7,0,2.4,right
"""
    )
    args = [
        "--data",
        str(data),
        "--rr-train",
        "50",
        "--rr-test",
        "50",
        "--cv-folds",
        "2",
        "--scaler",
        "to01",
        "--criterion",
        "wrapper-knn",
        "--knn-k",
        "1",
        "--target-d",
        "2",
        "--sffs-delta",
        "1",
        "--step-cap",
        "3",
        "--step-cap-backward",
        "2",
        "--seed",
        "1",
    ]
    oracle = tmp_path / "ssffs"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O2",
            "-funroll-loops",
            "-ffp-contract=off",
            "-DNDEBUG",
            "-o",
            str(oracle),
            "anc/ssffs/ssffs.cpp",
        ],
        check=True,
    )
    cpp_final, cpp_events, cpp_diagnostics = _run([str(oracle), *args])
    python_final, python_events, python_diagnostics = _run(
        [sys.executable, "-m", "sss.cli", *args, "--progress"],
        env=os.environ | {"PYTHONPATH": "src"},
    )

    assert python_final["features"] == cpp_final["features"]
    assert python_final["evaluations"] == cpp_final["evaluations"]
    assert python_final["value"] == pytest.approx(cpp_final["value"], abs=1e-11)
    assert [(d, features) for d, features, _value in python_events] == [
        (d, features) for d, features, _value in cpp_events
    ]
    assert [value for _d, _features, value in python_events] == pytest.approx(
        [value for _d, _features, value in cpp_events], abs=1e-11
    )
    assert python_diagnostics == cpp_diagnostics
