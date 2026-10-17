"""Scoring with the sealed scorer in .ur/eval, always in a disposable copy."""

from __future__ import annotations

import json
import math
import queue
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import config as config_mod
from . import execute
from .util import UrError, fill

if TYPE_CHECKING:
    from .campaign import Campaign


class EvalFailed(UrError):
    def __init__(self, message: str, log: Path):
        super().__init__(message)
        self.log = log


def score_dir(c: "Campaign", workdir: Path, split: str, devices: list[str], log: Path) -> dict[str, float]:
    """Run the scorer on workdir for one split and return its metrics."""
    data = config_mod.resolve(c.project, c.cfg.splits[split])
    if data is None:
        raise UrError(f"splits.{split} isn't set.")
    out_dir = Path(tempfile.mkdtemp(dir=c.store.tmp(), prefix="out-"))
    out = out_dir / "metrics.json"
    try:
        cmd = fill(c.cfg.eval.command, eval_dir=c.eval_dir, workspace=workdir, data=data, split=split, out=out)
        # Nothing about the split goes in the environment: code the scorer imports can read that.
        env = c.env(devices)
        status, code, _ = execute.run_logged(cmd, workdir, env, c.cfg.eval.timeout, log, c.cfg.exec.wrap)
        if status == "timeout":
            raise EvalFailed(f"the scorer ran past eval.timeout ({c.cfg.eval.timeout:g}s)", log)
        if status != "ok":
            raise EvalFailed(f"the scorer exited with code {code}", log)
        try:
            raw = json.loads(out.read_text())
        except FileNotFoundError:
            raise EvalFailed("the scorer didn't write its metrics to {out}", log) from None
        except json.JSONDecodeError as e:
            raise EvalFailed(f"the scorer's output isn't JSON ({e})", log) from None
        if not isinstance(raw, dict):
            raise EvalFailed("the scorer's output isn't a JSON object", log)
        metrics = {
            k: float(v)
            for k, v in raw.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
        }
        if c.metric not in metrics:
            raise EvalFailed(f"the scorer reported no finite {c.metric}", log)
        return metrics
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def evaluate_node(c: "Campaign", node: dict[str, Any], split: str, devices: list[str] | None = None) -> dict[str, float]:
    """Score a node's code and checkpoints on a split, in a copy that's thrown away afterwards."""
    c.verify_seal()
    if devices is None:
        devices = node.get("devices") or []
    work = Path(tempfile.mkdtemp(dir=c.store.tmp(), prefix=f"{node['id']}-{split}-"))
    try:
        c.store.materialize(node["commit"], work)
        c.link_shared(work)
        c.link_artifacts(node["id"], work)
        return score_dir(c, work, split, devices, c.store.node_dir(node["id"]) / f"eval-{split}.log")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def evaluate_path(c: "Campaign", path: Path, split: str, log: Path) -> dict[str, float]:
    """Score a directory as it is (for check.expect), in a throwaway copy."""
    work = Path(tempfile.mkdtemp(dir=c.store.tmp(), prefix="expect-"))
    try:
        shutil.copytree(path, work, symlinks=True, dirs_exist_ok=True)
        return score_dir(c, work, split, c.device_slices()[0], log)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def score_nodes(c: "Campaign", nodes: list[dict[str, Any]], split: str) -> None:
    """Score nodes on a split in parallel, one per device slice, and save the results on each node.

    A node whose scoring fails gets the reason under errors[split] instead.
    """
    if not nodes:
        return
    slices: queue.Queue[list[str]] = queue.Queue()
    for devices in c.device_slices():
        slices.put(devices)

    def work(node: dict[str, Any]) -> None:
        devices = slices.get()
        try:
            try:
                metrics = evaluate_node(c, node, split, devices)
            except EvalFailed as e:
                node.setdefault("errors", {})[split] = str(e)
            else:
                node[split] = metrics
                if split == "heldout":
                    node["heldout_repeats"] = [metrics[c.metric]]
            c.store.save_node(node)
        finally:
            slices.put(devices)

    with ThreadPoolExecutor(max_workers=slices.qsize()) as pool:
        list(pool.map(work, nodes))
