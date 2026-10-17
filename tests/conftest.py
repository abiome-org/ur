import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"


class Ur:
    """Runs the ur CLI the way an agent would: as a command on PATH."""

    def __init__(self, tmp: Path):
        self.bin = tmp / "bin"
        self.bin.mkdir()
        shim = self.bin / "ur"
        shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" -m ur "$@"\n')
        shim.chmod(0o755)
        self.env = dict(os.environ)
        self.env["PATH"] = f"{self.bin}{os.pathsep}{self.env['PATH']}"
        self.env["PYTHONPATH"] = str(REPO / "src")
        self.env["HOME"] = str(tmp / "home")
        self.env.pop("XDG_DATA_HOME", None)
        (tmp / "home").mkdir()

    def __call__(self, *args: str, cwd: Path, check: bool = True, env: dict | None = None) -> subprocess.CompletedProcess:
        proc = subprocess.run(
            ["ur", *args], cwd=cwd, env={**self.env, **(env or {})}, capture_output=True, text=True, timeout=600
        )
        if check and proc.returncode != 0:
            raise AssertionError(f"ur {' '.join(args)} exited {proc.returncode}\n{proc.stdout}\n{proc.stderr}")
        return proc


@pytest.fixture
def ur(tmp_path: Path) -> Ur:
    return Ur(tmp_path)


def make_toy(tmp: Path, ur: Ur, extra_env: dict | None = None) -> dict:
    """The toy project, set up the way a setup agent would leave it before `ur check`."""
    project = tmp / "toy"
    shutil.copytree(REPO / "examples" / "toy", project)
    private = tmp / "private"
    make_data = REPO / "examples" / "toy-data" / "make_data.py"
    subprocess.run([sys.executable, str(make_data), "--project", str(project), "--private", str(private)], check=True)
    ur_dir = project / ".ur"
    ur_dir.mkdir()
    shutil.copytree(FIXTURES / "toy" / "eval", ur_dir / "eval")
    shutil.copytree(FIXTURES / "toy" / "checks", ur_dir / "checks")
    store = tmp / "store"
    config = (FIXTURES / "toy" / "config.toml").read_text()
    for key, value in {
        "@PRIVATE@": str(private),
        "@PYTHON@": sys.executable,
        "@AGENT@": str(Path(__file__).parent / "fake_agent.py"),
        "@STORE@": str(store),
    }.items():
        config = config.replace(key, value)
    (ur_dir / "config.toml").write_text(config)
    ur.env.update(extra_env or {})
    ur.env["FAKE_HELDOUT"] = str(private / "heldout.jsonl")
    return {"project": project, "private": private, "store": store}
