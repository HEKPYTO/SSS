import hashlib
import pathlib
import subprocess


def test_vendor_compiles_and_runs():
    p = pathlib.Path("anc/ssffs/ssffs.cpp")
    assert p.exists(), "vendored ssffs.cpp missing"
    assert p.stat().st_size > 40000
    r = subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O2",
            "-funroll-loops",
            "-ffp-contract=off",
            "-DNDEBUG",
            "-o",
            "/tmp/ssffs",
            "anc/ssffs/ssffs.cpp",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    r = subprocess.run(["/tmp/ssffs", "--help"], capture_output=True, text=True)
    assert "ssffs" in r.stdout.lower()


def test_official_ancillary_hashes():
    expected = {
        "anc/ssffs/ssffs.cpp": "acf6d17e01238fead9af18dfbe2f502e000df6ec372272daeb3ed99c286a9a24",
        "anc/data/reuters_apte.arff": "4b22e0e94f53993595f5fa80c7eca0b5dbda0ec80423ac2e31861e156ea1834a",
    }
    for path, digest in expected.items():
        with open(path, "rb") as source:
            assert hashlib.sha256(source.read()).hexdigest() == digest


def test_reuters_runs():
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O2",
            "-funroll-loops",
            "-ffp-contract=off",
            "-DNDEBUG",
            "-o",
            "/tmp/ssffs",
            "anc/ssffs/ssffs.cpp",
        ],
        check=True,
    )
    r = subprocess.run(
        [
            "/tmp/ssffs",
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
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert r.returncode == 0
    import json

    out = json.loads(r.stdout)
    assert out["size"] == 5 and "evaluations" in out and out["value"] > 0
