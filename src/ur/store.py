"""Ur's store: code snapshots in a bare git repo, run records with their checkpoints, and workspaces.

Layout under the store root:

    code.git/            every snapshot, kept alive by refs/ur/<node>
    runs/<node>/         run.json, logs, artifacts/
    meta/<workspace>.json
    ws/<workspace>/      the workspaces themselves
    prompts/<workspace>.md
    sessions/<workspace>.log
    project              path of the project this store belongs to
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from .util import PathMatcher, UrError, list_files, locked, read_json, write_json

MARKER = ".ur-workspace"
NODE_ID = re.compile(r"^n\d+$")

_GIT_ENV = {
    "GIT_AUTHOR_NAME": "ur",
    "GIT_AUTHOR_EMAIL": "ur@localhost",
    "GIT_COMMITTER_NAME": "ur",
    "GIT_COMMITTER_EMAIL": "ur@localhost",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_TERMINAL_PROMPT": "0",
}


class Store:
    def __init__(self, root: Path):
        self.root = root
        self.git_dir = root / "code.git"

    # Layout.

    def node_dir(self, nid: str) -> Path:
        return self.root / "runs" / nid

    def artifacts_dir(self, nid: str) -> Path:
        return self.node_dir(nid) / "artifacts"

    def ws_path(self, wid: str) -> Path:
        return self.root / "ws" / wid

    def prompt_path(self, wid: str) -> Path:
        return self.root / "prompts" / f"{wid}.md"

    def session_log(self, wid: str) -> Path:
        return self.root / "sessions" / f"{wid}.log"

    def tmp(self) -> Path:
        path = self.root / "tmp"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def exists(self) -> bool:
        return (self.git_dir / "HEAD").exists()

    def ensure(self, project: Path) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.exists():
            subprocess.run(
                ["git", "init", "--bare", "--quiet", str(self.git_dir)],
                check=True,
                env=self._env(),
                capture_output=True,
            )
            self.git("config", "gc.auto", "0")
        (self.root / "project").write_text(f"{project}\n")

    # Git.

    def _env(self, index: Path | None = None) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        env.update(_GIT_ENV)
        if index is not None:
            env["GIT_INDEX_FILE"] = str(index)
        return env

    def git(
        self,
        *args: str,
        work_tree: Path | None = None,
        index: Path | None = None,
        input: str | bytes | None = None,
        text: bool = True,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        cmd = ["git", f"--git-dir={self.git_dir}"]
        if work_tree is not None:
            cmd.append(f"--work-tree={work_tree}")
        cmd += args
        proc = subprocess.run(
            cmd,
            cwd=work_tree or self.root,
            env=self._env(index),
            input=input,
            capture_output=True,
            text=text,
        )
        if check and proc.returncode != 0:
            err = proc.stderr if text else proc.stderr.decode(errors="replace")
            raise UrError(f"git {args[0]} failed: {err.strip()}")
        return proc

    def snapshot(self, src: Path, matcher: PathMatcher, max_file_mb: float, max_total_mb: float) -> str:
        """Store src's files (minus what matcher excludes) and return the tree id."""
        files = list_files(src, matcher)
        too_big, total = [], 0
        for rel in files:
            path = src / rel
            if path.is_symlink():
                continue
            try:
                size = path.stat().st_size
            except FileNotFoundError:
                continue
            total += size
            if size > max_file_mb * 1e6:
                too_big.append(f"{rel} ({size / 1e6:.0f} MB)")
        if too_big:
            raise UrError(
                f"These files are over workspace.max_file_mb ({max_file_mb:g} MB). Delete them, or add them to "
                "run.artifacts, workspace.exclude or workspace.shared:\n  " + "\n  ".join(too_big)
            )
        if total > max_total_mb * 1e6:
            raise UrError(
                f"The code snapshot is {total / 1e6:.0f} MB, over workspace.max_total_mb ({max_total_mb:g} MB). "
                "Exclude or share the data directories in .ur/config.toml."
            )
        with tempfile.TemporaryDirectory(dir=self.tmp()) as d:
            index = Path(d) / "index"
            if files:
                self.git("update-index", "--add", "-z", "--stdin", work_tree=src, index=index, input="\0".join(files) + "\0")
            return self.git("write-tree", work_tree=src, index=index).stdout.strip()

    def commit(self, tree: str, parent: str | None, message: str) -> str:
        args = ["commit-tree", tree, "-m", message]
        if parent:
            args += ["-p", parent]
        return self.git(*args).stdout.strip()

    def keep(self, nid: str, commit: str) -> None:
        self.git("update-ref", f"refs/ur/{nid}", commit)

    def materialize(self, commit: str, dest: Path) -> None:
        """Write commit's files into dest."""
        dest.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.tmp()) as d:
            index = Path(d) / "index"
            self.git("read-tree", commit, work_tree=dest, index=index)
            self.git("checkout-index", "-a", "-f", work_tree=dest, index=index)

    def restore(self, dest: Path, commit: str, matcher: PathMatcher) -> None:
        """Make dest's snapshot files match commit: remove the extras, write the rest."""
        wanted = self.files(commit)
        for rel in list_files(dest, matcher):
            if rel not in wanted:
                (dest / rel).unlink()
        for dirpath, dirnames, filenames in os.walk(dest, topdown=False):
            path = Path(dirpath)
            if path != dest and not any(path.iterdir()) and not matcher.match_path(path.relative_to(dest).as_posix()):
                path.rmdir()
        self.materialize(commit, dest)

    def files(self, treeish: str) -> dict[str, tuple[str, str]]:
        """{path: (mode, blob id)} for every file in a commit or tree."""
        out = self.git("ls-tree", "-r", "-z", "--full-tree", treeish).stdout
        files = {}
        for entry in out.split("\0"):
            if not entry:
                continue
            meta, path = entry.split("\t", 1)
            mode, kind, sha = meta.split()
            if kind == "blob":
                files[path] = (mode, sha)
        return files

    def blob(self, sha: str) -> bytes:
        return self.git("cat-file", "blob", sha, text=False).stdout

    def diff(self, a: str, b: str, stat: bool = False) -> str:
        args = ["diff", "--no-color", "--no-ext-diff"]
        if stat:
            args.append("--stat")
        return self.git(*args, a, b).stdout

    def changed(self, a: str, b: str) -> list[tuple[str, str]]:
        """[(status letter, path)] between two commits or trees."""
        out = self.git("diff", "--name-status", "-z", "--no-renames", a, b).stdout
        parts = [p for p in out.split("\0") if p]
        return list(zip(parts[0::2], parts[1::2]))

    # Nodes: one per training run.

    def new_id(self) -> str:
        with locked(self.root / "lock"):
            path = self.root / "counter"
            n = int(path.read_text()) + 1 if path.exists() else 1
            tmp = path.with_suffix(".tmp")
            tmp.write_text(f"{n}\n")
            os.replace(tmp, path)
        return f"n{n}"

    def load_node(self, nid: str) -> dict[str, Any]:
        if not NODE_ID.match(nid or ""):
            raise UrError(f"{nid!r} isn't a node id (they look like n12).")
        data = read_json(self.node_dir(nid) / "run.json")
        if data is None:
            raise UrError(f"There's no node {nid}.")
        return data

    def save_node(self, node: dict[str, Any]) -> None:
        write_json(self.node_dir(node["id"]) / "run.json", node)

    # Workspaces.

    def load_ws(self, wid: str) -> dict[str, Any]:
        data = read_json(self.root / "meta" / f"{wid}.json")
        if data is None:
            raise UrError(f"There's no workspace {wid}.")
        return data

    def save_ws(self, ws: dict[str, Any]) -> None:
        write_json(self.root / "meta" / f"{ws['id']}.json", ws)

    def ws_lock(self, wid: str) -> AbstractContextManager[None]:
        return locked(self.root / "meta" / f"{wid}.lock")

    def all_ws(self) -> list[dict[str, Any]]:
        meta = self.root / "meta"
        if not meta.is_dir():
            return []
        return [read_json(p) for p in sorted(meta.glob("*.json"))]
