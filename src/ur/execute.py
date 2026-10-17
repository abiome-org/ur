"""Running commands under time limits, and starting and stopping agent sessions."""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from .util import now, wrap_command


_HAVE_PROC = Path("/proc/self/stat").exists()


def _proc_stat(pid: int) -> tuple[str, str] | None:
    """(state, start time) from /proc, or None where there's no /proc."""
    if not _HAVE_PROC:
        return None
    try:
        data = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return ("X", "")
    except OSError:
        return None
    fields = data[data.rindex(")") + 2 :].split()
    return fields[0], fields[19]


def identity(pid: int) -> dict[str, Any]:
    stat = _proc_stat(pid)
    return {"pid": pid, "start": stat[1] if stat else None}


def alive(proc: dict[str, Any] | None) -> bool:
    """Whether the process recorded in proc (pid plus start time) is still running."""
    if not proc or not proc.get("pid"):
        return False
    pid = proc["pid"]
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    stat = _proc_stat(pid)
    if stat is None:
        return True
    state, start = stat
    if state in ("Z", "X"):
        return False
    return not proc.get("start") or proc["start"] == start


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def kill_group(pgid: int, grace: float = 10.0) -> None:
    """SIGTERM a process group, then SIGKILL it if it's still there after grace seconds."""
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    end = time.time() + grace
    while time.time() < end:
        if not _group_alive(pgid):
            return
        time.sleep(0.2)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def stop(proc: dict[str, Any] | None, grace: float = 15.0) -> None:
    """Stop a recorded process (and its group, if it leads one)."""
    if not alive(proc):
        return
    pid = proc["pid"]
    try:
        if os.getpgid(pid) == pid:
            kill_group(pid, grace)
            return
    except ProcessLookupError:
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    end = time.time() + grace
    while time.time() < end and alive(proc):
        time.sleep(0.2)
    if alive(proc):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_logged(
    cmd: str,
    cwd: Path,
    env: dict[str, str],
    timeout: float | None,
    log_path: Path,
    wrap: str = "",
    on_start: Callable[[int], None] | None = None,
) -> tuple[str, int | None, float]:
    """Run cmd in its own process group, appending its output to log_path.

    Returns (status, exit code, seconds), where status is "ok", "failed" or "timeout".
    """
    full = wrap_command(wrap, cmd)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = now()
    with open(log_path, "ab") as log:
        log.write(f"$ {full}\n".encode())
        log.flush()
        proc = subprocess.Popen(
            full,
            shell=True,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            if on_start:
                on_start(proc.pid)
            code = proc.wait(timeout=timeout)
            status = "ok" if code == 0 else "failed"
        except subprocess.TimeoutExpired:
            kill_group(proc.pid)
            proc.wait()
            code, status = proc.returncode, "timeout"
        except BaseException:
            kill_group(proc.pid)
            proc.wait()
            raise
    return status, code, now() - start


def spawn(cmd: str, cwd: Path, env: dict[str, str], log_path: Path) -> dict[str, Any]:
    """Start a detached session; it keeps running after ur exits. Its exit code lands next to its log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    exit_file = log_path.with_suffix(".exit")
    exit_file.unlink(missing_ok=True)
    with open(log_path, "ab") as log:
        log.write(f"$ {cmd}\n".encode())
        log.flush()
        proc = subprocess.Popen(
            f"({cmd}); echo $? > {shlex.quote(str(exit_file))}",
            shell=True,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    proc.returncode = 0  # never waited on here; this keeps Python from warning about it
    return {**identity(proc.pid), "started": now(), "log": str(log_path), "exit_file": str(exit_file)}


def session_state(session: dict[str, Any] | None) -> str:
    """none, alive, done (exited 0) or died (anything else, including being killed)."""
    if not session:
        return "none"
    if alive(session):
        return "alive"
    try:
        code = Path(session.get("exit_file", "")).read_text().strip()
    except (FileNotFoundError, IsADirectoryError):
        return "died"
    return "done" if code == "0" else "died"


def tail(path: Path | str | None, lines: int = 30) -> str:
    if not path:
        return ""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 64 * 1024))
            data = f.read().decode(errors="replace")
    except FileNotFoundError:
        return ""
    return "\n".join(data.splitlines()[-lines:])
