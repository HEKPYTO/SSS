import subprocess
import sys


def test_wheel_installs_cleanly(tmp_path):
    wheel_dir = tmp_path / "wheel"
    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", ".", "--no-deps", "--wheel-dir", str(wheel_dir)],
        check=True,
    )
    environment = tmp_path / "environment"
    subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
    python = environment / "bin" / "python"
    subprocess.run(
        [str(python), "-m", "pip", "install", str(next(wheel_dir.glob("*.whl")))], check=True
    )
    probe = "import sss; from sss import Result, SFFS; assert sss.__version__ == '0.1.0'"
    subprocess.run([str(python), "-c", probe], check=True, cwd=tmp_path)
    subprocess.run([str(environment / "bin" / "sss"), "--help"], check=True)
