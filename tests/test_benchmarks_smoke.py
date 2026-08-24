import json
import subprocess
import sys


def test_madelon_smoke():
    r = subprocess.run(
        [
            sys.executable,
            "scripts/madelon.py",
            "--y",
            "5",
            "--rho",
            "0.2",
            "--seed",
            "1",
            "--target-d",
            "5",
            "--synthetic",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr[-1000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert "features" in out and len(out["features"]) == 5
    assert out["evaluations"] > 0


def test_gisette_smoke():
    r = subprocess.run(
        [
            sys.executable,
            "scripts/gisette.py",
            "--y",
            "5",
            "--seed",
            "1",
            "--target-d",
            "5",
            "--synthetic",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr[-1000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert len(out["features"]) == 5


def test_reuters_smoke():
    r = subprocess.run(
        [
            sys.executable,
            "scripts/reuters.py",
            "--y",
            "5",
            "--target-d",
            "5",
            "--seed",
            "1",
            "--warmup-probes",
            "5",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr[-1000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["size"] == 5
    assert out["value"] > 0
