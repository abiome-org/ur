"""Small helpers shared across ur."""

from __future__ import annotations

import contextlib
import datetime as _dt
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Iterator


class UrError(Exception):
    """An error whose message is meant for whoever ran ur."""


def now() -> float:
    return time.time()


def clock(ts: float) -> str:
    """Local wall-clock time, like 14:20."""
    return _dt.datetime.fromtimestamp(ts).strftime("%H:%M")


def human_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s" if minutes < 10 else f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def fmt_num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4g}"


def read_json(path: Path, default: Any = None) -> Any:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def write_json(path: Path, data: Any) -> None:
    """Write JSON atomically, so a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


@contextlib.contextmanager
def locked(path: Path) -> Iterator[None]:
    """An exclusive advisory lock on path, held for the with-block."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_dir(root: Path) -> dict[str, str]:
    """sha256 of every file under root, keyed by relative path (bytecode caches skipped)."""
    out: dict[str, str] = {}
    if not root.is_dir():
        return out
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for name in sorted(filenames):
            if name.endswith(".pyc"):
                continue
            path = Path(dirpath) / name
            out[path.relative_to(root).as_posix()] = sha256_file(path)
    return out


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        force_rmtree(path)


def make_read_only(path: Path) -> None:
    """Drop write permission on everything under path (links are left alone)."""
    if path.is_symlink() or not path.exists():
        return
    if path.is_file():
        path.chmod(path.stat().st_mode & ~0o222)
        return
    for dirpath, _, filenames in os.walk(path, topdown=False):
        for name in filenames:
            p = os.path.join(dirpath, name)
            if not os.path.islink(p):
                os.chmod(p, os.stat(p).st_mode & ~0o222)
        os.chmod(dirpath, os.stat(dirpath).st_mode & ~0o222)


def make_writable(path: Path) -> None:
    if path.is_symlink() or not path.exists():
        return
    if path.is_file():
        path.chmod(path.stat().st_mode | 0o200)
        return
    for dirpath, _, filenames in os.walk(path):
        os.chmod(dirpath, os.stat(dirpath).st_mode | 0o700)
        for name in filenames:
            p = os.path.join(dirpath, name)
            if not os.path.islink(p):
                os.chmod(p, os.stat(p).st_mode | 0o200)


def force_rmtree(path: Path) -> None:
    """rmtree that also removes read-only trees."""
    if not path.exists() or path.is_symlink():
        if path.is_symlink():
            path.unlink()
        return
    for dirpath, _, _ in os.walk(path):
        with contextlib.suppress(OSError):
            os.chmod(dirpath, os.stat(dirpath).st_mode | 0o700)
    shutil.rmtree(path, ignore_errors=True)


_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")


def fill(template: str, **values: Any) -> str:
    """Fill {name} placeholders with shell-quoted values. Other braces are left alone."""

    def sub(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            return match.group(0)
        value = values[key]
        return shlex.quote("" if value is None else str(value))

    return _PLACEHOLDER.sub(sub, template)


def wrap_command(wrap: str, cmd: str) -> str:
    """Put cmd inside the exec.wrap template: {cmd} as is, {qcmd} shell-quoted."""
    if not wrap:
        return cmd
    if "{cmd}" not in wrap and "{qcmd}" not in wrap:
        return f"{wrap} {cmd}"
    return wrap.replace("{qcmd}", shlex.quote(cmd)).replace("{cmd}", cmd)


def _glob_regex(pattern: str) -> str:
    out: list[str] = []
    i, n = 0, len(pattern)
    while i < n:
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("/**", i) and i + 3 == n:
            out.append("(?:/.*)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        elif pattern[i] == "[" and "]" in pattern[i + 1 :]:
            j = pattern.index("]", i + 1)
            body = pattern[i + 1 : j]
            if body.startswith("!"):
                body = "^" + body[1:]
            out.append("[" + body.replace("\\", "\\\\") + "]")
            i = j + 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return "".join(out)


class PathMatcher:
    """Paths matched by .gitignore-style patterns.

    A pattern without a slash matches a name at any depth; a leading or inner slash
    anchors it to the root; a trailing slash matches directories only. `**` crosses
    directories. Negations (`!`) aren't supported and are ignored.
    """

    def __init__(self, patterns: Iterable[str]):
        self.rules: list[tuple[re.Pattern[str], bool, bool]] = []
        for raw in patterns:
            pattern = raw.strip()
            if not pattern or pattern.startswith(("#", "!")):
                continue
            dir_only = pattern.endswith("/")
            pattern = pattern.rstrip("/")
            anchored = pattern.startswith("/") or "/" in pattern
            pattern = pattern.lstrip("/")
            if pattern:
                self.rules.append((re.compile(_glob_regex(pattern) + r"\Z"), anchored, dir_only))

    def match(self, rel: str, is_dir: bool) -> bool:
        name = rel.rsplit("/", 1)[-1]
        for regex, anchored, dir_only in self.rules:
            if dir_only and not is_dir:
                continue
            if regex.match(rel if anchored else name):
                return True
        return False

    def match_path(self, rel: str) -> bool:
        """Whether a file path, or any directory above it, matches."""
        parts = rel.split("/")
        for i in range(1, len(parts)):
            if self.match("/".join(parts[:i]), True):
                return True
        return self.match(rel, False)


def list_files(root: Path, matcher: PathMatcher) -> list[str]:
    """Relative paths of the files and symlinks under root that matcher doesn't exclude."""
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root).as_posix()
        prefix = "" if rel_dir == "." else rel_dir + "/"
        keep = []
        for name in dirnames:
            rel = prefix + name
            if matcher.match(rel, True):
                continue
            if (Path(dirpath) / name).is_symlink():
                out.append(rel)  # a link to a directory is stored as the link itself
                continue
            keep.append(name)
        dirnames[:] = sorted(keep)
        for name in filenames:
            rel = prefix + name
            if not matcher.match(rel, False):
                out.append(rel)
    return sorted(out)
