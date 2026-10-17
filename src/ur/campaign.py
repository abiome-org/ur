"""A campaign: the project's .ur/ directory, its config and state, and the store behind it."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any, Iterator

from . import config as config_mod
from .store import MARKER, Store
from .util import PathMatcher, UrError, hash_dir, locked, now, read_json, remove_path, sha256_file, write_json


def find_project(start: Path) -> Path | None:
    for d in (start, *start.parents):
        if (d / ".ur" / "config.toml").is_file():
            return d
    return None


def find_workspace(start: Path) -> tuple[Path, dict[str, Any]] | None:
    for d in (start, *start.parents):
        marker = d / MARKER
        if marker.is_file():
            return d, read_json(marker)
    return None


class Campaign:
    def __init__(self, project: Path):
        self.project = project.resolve()
        self.ur = self.project / ".ur"
        self.cfg = config_mod.load(self.project)
        self.store = Store(Path(self.cfg.store.root).expanduser())
        self.state_path = self.ur / "state" / "campaign.json"

    @classmethod
    def from_store(cls, root: Path) -> "Campaign":
        pointer = root / "project"
        if not pointer.exists():
            raise UrError("This workspace's campaign is gone.")
        return cls(Path(pointer.read_text().strip()))

    @property
    def brief_path(self) -> Path:
        return self.ur / "brief.md"

    @property
    def eval_dir(self) -> Path:
        return self.ur / "eval"

    # State.

    def load_state(self) -> dict[str, Any]:
        s = read_json(self.state_path) or {}
        defaults: dict[str, Any] = {
            "version": 1,
            "phase": "setup",
            "check": {},
            "sealed": {},
            "base": None,
            "round": 0,
            "rounds": [],
            "stalls": 0,
            "gpu_hours": 0.0,
            "stopped_reason": None,
        }
        for key, value in defaults.items():
            s.setdefault(key, value)
        return s

    @contextlib.contextmanager
    def update_state(self) -> Iterator[dict[str, Any]]:
        with locked(self.ur / "state" / "lock"):
            s = self.load_state()
            yield s
            write_json(self.state_path, s)

    def phase(self, s: dict[str, Any] | None = None) -> str:
        """setup, setup-probe, setup-brief, idle, round_open, round_closed or stopped."""
        s = s if s is not None else self.load_state()
        check = s.get("check") or {}
        if not check.get("passed"):
            probe = check.get("probe") or {}
            return "setup-probe" if probe.get("status") == "pending" else "setup"
        if not self.brief_path.exists():
            return "setup-brief"
        ph = s.get("phase") or "idle"
        if ph == "setup" or (ph == "stopped" and not self.stop_reason(s)):
            return "idle"
        return ph

    def stop_reason(self, s: dict[str, Any]) -> str | None:
        st = self.cfg.stop
        if st.patience and s["stalls"] >= st.patience:
            return f"no new base in {s['stalls']} round{'s' if s['stalls'] != 1 else ''}"
        if st.max_rounds and s["round"] >= st.max_rounds:
            return f"reached {st.max_rounds} rounds"
        if st.max_gpu_hours and s["gpu_hours"] >= st.max_gpu_hours:
            return f"used {s['gpu_hours']:.1f} of {st.max_gpu_hours:g} GPU-hours"
        return None

    # Seals: what can't change after setup.

    def seal(self, s: dict[str, Any]) -> None:
        s["sealed"] = {"eval": hash_dir(self.eval_dir), "config": config_mod.frozen_hash(self.cfg), "brief": None}

    def verify_seal(self, s: dict[str, Any] | None = None) -> None:
        s = s if s is not None else self.load_state()
        sealed = s.get("sealed") or {}
        if not sealed:
            return
        restart = "That's a new campaign: `ur archive`, then initialize again."
        if hash_dir(self.eval_dir) != sealed.get("eval"):
            raise UrError(f"The scorer in .ur/eval changed after setup. {restart}")
        if config_mod.frozen_hash(self.cfg) != sealed.get("config"):
            raise UrError(f"[objective], [splits], [eval] or [run] in .ur/config.toml changed after setup. {restart}")
        brief = sealed.get("brief")
        if brief and (not self.brief_path.exists() or sha256_file(self.brief_path) != brief):
            raise UrError(f"brief.md changed after the first round. {restart}")

    # Matching.

    def snapshot_matcher(self) -> PathMatcher:
        patterns = [".git", "/.ur", "/" + MARKER, *self.cfg.workspace.exclude]
        patterns += ["/" + a.strip("/") for a in self.cfg.run.artifacts]
        patterns += ["/" + s.strip("/") for s in self.cfg.workspace.shared]
        return PathMatcher(patterns)

    def editable_matcher(self) -> PathMatcher:
        return PathMatcher(self.cfg.workspace.editable)

    # Scores.

    @property
    def metric(self) -> str:
        return self.cfg.objective.metric

    def better(self, a: float | None, b: float | None) -> bool:
        """Whether score a is strictly better than score b."""
        if a is None:
            return False
        if b is None:
            return True
        return a > b if self.cfg.objective.direction == "max" else a < b

    def score(self, node: dict[str, Any]) -> float | None:
        """A node's held-out score: the mean over every time it was scored."""
        repeats = node.get("heldout_repeats") or []
        if repeats:
            return sum(repeats) / len(repeats)
        heldout = node.get("heldout") or {}
        return heldout.get(self.metric)

    def dev_score(self, node: dict[str, Any]) -> float | None:
        return (node.get("dev") or {}).get(self.metric)

    def rank(self, nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Nodes with a held-out score, best first."""
        scored = [n for n in nodes if self.score(n) is not None]
        return sorted(scored, key=self.score, reverse=self.cfg.objective.direction == "max")

    # Resources.

    def device_slices(self) -> list[list[str]]:
        devices, k = self.cfg.exec.devices, self.cfg.exec.devices_per_run
        if not devices:
            return [[]]
        return [devices[i : i + k] for i in range(0, len(devices) - k + 1, k)]

    def run_timeout(self, budget: float) -> float:
        if self.cfg.run.timeout:
            return float(self.cfg.run.timeout)
        return float(budget) + max(30.0, 0.1 * float(budget))

    def env(self, devices: list[str], **extra: Any) -> dict[str, str]:
        env = dict(os.environ)
        if devices:
            env[self.cfg.exec.device_env] = ",".join(devices)
        env.update({k: str(v) for k, v in extra.items()})
        return env

    # Workspaces.

    def link_artifacts(self, nid: str, dest: Path) -> None:
        """Link a node's checkpoints into dest at their usual paths."""
        for rel in self.cfg.run.artifacts:
            src = self.store.artifacts_dir(nid) / rel
            link = dest / rel
            if link.exists() or link.is_symlink():
                remove_path(link)
            if src.exists():
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(src)

    def link_shared(self, dest: Path) -> None:
        for rel in self.cfg.workspace.shared:
            link = dest / rel
            if link.exists() or link.is_symlink():
                remove_path(link)
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to((self.project / rel).resolve())

    def create_workspace(
        self,
        wid: str,
        base: str,
        purpose: str,
        round_n: int,
        deadline: float | None,
        visible: list[str],
        devices: list[str],
        budget: float | None = None,
        init_from: str | None = None,
    ) -> dict[str, Any]:
        """A fresh workspace holding base's code and checkpoints."""
        store = self.store
        path = store.ws_path(wid)
        if path.exists():
            remove_path(path)
        store.materialize(store.load_node(base)["commit"], path)
        self.link_shared(path)
        self.link_artifacts(base, path)
        write_json(path / MARKER, {"root": str(store.root), "id": wid})
        ws: dict[str, Any] = {
            "id": wid,
            "purpose": purpose,
            "round": round_n,
            "base": base,
            "head": base,
            "visible": sorted(set(visible) | {base}),
            "deadline": deadline,
            "budget": budget,
            "devices": devices,
            "closed": False,
            "candidates": [],
            "last_run": None,
            "runs": [],
            "session": None,
            "active": None,
            "created": now(),
            "last_activity": now(),
        }
        if init_from is not None:
            ws["init_from"] = init_from
        store.save_ws(ws)
        return ws
