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
        [sys.executable, "-m", "sss.cli", *PARITY_ARGS], env=environment
    )

    assert cpp_final["features"] == [71, 110, 114, 152, 6611]
    assert cpp_final["evaluations"] == 659
    assert python_final["features"] == cpp_final["features"]
    assert python_final["evaluations"] == cpp_final["evaluations"]
    assert python_events == pytest.approx(cpp_events, abs=1e-11)
    assert python_diagnostics == cpp_diagnostics
