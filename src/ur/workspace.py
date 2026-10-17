"""Commands that run inside a workspace: run, commit, reset, diff, apply, log, status."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import evaluate, execute
from .util import UrError, clock, fill, fmt_num, human_duration, make_read_only, now, remove_path

if TYPE_CHECKING:
    from .campaign import Campaign


def fmt_budget(budget: float) -> str:
    return str(int(budget)) if float(budget).is_integer() else f"{budget:g}"


def metrics_text(c: "Campaign", metrics: dict[str, float] | None) -> str:
    if not metrics:
        return "no scores"
    main = f"{c.metric} {fmt_num(metrics.get(c.metric))}"
    rest = [f"{k} {fmt_num(v)}" for k, v in sorted(metrics.items()) if k != c.metric]
    return main + (f" ({', '.join(rest)})" if rest else "")


def resolve_ref(c: "Campaign", ws: dict[str, Any], ref: str) -> str:
    """A node id from `base`, `head` or an id this workspace may see."""
    nid = {"base": ws["base"], "head": ws["head"]}.get(ref, ref)
    allowed = set(ws["visible"]) | set(ws.get("runs", [])) | {ws["base"], ws["head"]}
    if nid not in allowed:
        raise UrError(f"{ref} isn't visible from this workspace: you can see the base, your own runs, and nodes your prompt names.")
    c.store.load_node(nid)
    return nid


# Training runs.


def clear_stale_run(c: "Campaign", ws: dict[str, Any]) -> None:
    """If the last `ur run` here died without cleaning up, stop its training and mark its node failed."""
    active = ws.get("active")
    if not active or execute.alive(active):
        return
    if execute.alive(active.get("train")):
        execute.kill_group(active["train"]["pid"])
    try:
        node = c.store.load_node(active["node"])
    except UrError:
        node = None
    if node and node["status"] == "running":
        node.update(status="failed", error="the `ur run` that started it died")
        c.store.save_node(node)
    ws["active"] = None


def _require_idle(ws: dict[str, Any]) -> None:
    if execute.alive(ws.get("active")):
        raise UrError(f"A run is going in this workspace ({ws['active'].get('node')}). Wait for it to finish.")


def do_run(c: "Campaign", wid: str) -> dict[str, Any]:
    """Snapshot the workspace, train under the budget, collect the checkpoint and score it on dev."""
    store, cfg = c.store, c.cfg
    c.verify_seal()
    with store.ws_lock(wid):
        ws = store.load_ws(wid)
        if ws["closed"]:
            raise UrError("This workspace's round is closed, so nothing more counts here.")
        _require_idle(ws)
        clear_stale_run(c, ws)
        budget = ws.get("budget") or cfg.run.budget
        limit = c.run_timeout(budget)
        if ws.get("deadline") and now() + limit > ws["deadline"]:
            raise UrError(
                f"Not enough time left for another run: a run can take up to {human_duration(limit)}, and the "
                f"round ends in {human_duration(ws['deadline'] - now())}. Your last commit is what counts."
            )
        path = store.ws_path(wid)
        tree = store.snapshot(path, c.snapshot_matcher(), cfg.workspace.max_file_mb, cfg.workspace.max_total_mb)
        base = store.load_node(ws["base"])
        editable = c.editable_matcher()
        outside = [p for _, p in store.changed(base["commit"], tree) if not editable.match_path(p)]
        if outside:
            raise UrError(
                "These changes are outside the editable paths (" + ", ".join(cfg.workspace.editable) + "):\n  "
                + "\n  ".join(outside)
                + "\nUndo them (delete new files; `ur reset` restores your last commit), then run again."
            )
        head = store.load_node(ws["head"])
        commit = head["commit"] if tree == head["tree"] else store.commit(tree, head["commit"], wid)
        nid = store.new_id()
        if "init_from" in ws:
            init_from = ws["init_from"]
        else:
            init_from = {"scratch": None, "base": ws["base"], "parent": ws["head"]}[cfg.run.init]
        node: dict[str, Any] = {
            "id": nid,
            "workspace": wid,
            "round": ws["round"],
            "commit": commit,
            "tree": tree,
            "parent": ws["head"],
            "init_from": init_from,
            "budget": budget,
            "devices": ws["devices"],
            "started": now(),
            "seconds": None,
            "status": "running",
            "exit": None,
            "dev": None,
            "heldout": None,
            "heldout_repeats": [],
            "final": None,
        }
        store.save_node(node)
        store.keep(nid, commit)
        ws["active"] = {**execute.identity(os.getpid()), "node": nid}
        ws["last_activity"] = now()
        store.save_ws(ws)
    try:
        _train(c, ws, node, budget, limit)
    finally:
        with store.ws_lock(wid):
            ws = store.load_ws(wid)
            ws["active"] = None
            ws["last_run"] = nid
            ws.setdefault("runs", []).append(nid)
            ws["last_activity"] = now()
            store.save_ws(ws)
    return node


def _train(c: "Campaign", ws: dict[str, Any], node: dict[str, Any], budget: float, limit: float) -> None:
    store, cfg = c.store, c.cfg
    path = store.ws_path(ws["id"])
    for rel in cfg.run.artifacts:
        remove_path(path / rel)
    init = c.init_dir(node["init_from"])
    init_dir = str(init) if init else ""
    if node["init_from"] and not init:
        node["note"] = f"{node['init_from']} has no checkpoint, so this run trained from scratch"
    env = c.env(ws["devices"], UR_BUDGET=fmt_budget(budget), UR_INIT=init_dir, UR_RUN_ID=node["id"], UR_WORKSPACE=path)
    cmd = fill(cfg.run.command, budget=fmt_budget(budget), init=init_dir, workspace=path)
    log = store.node_dir(node["id"]) / "train.log"

    def on_start(pid: int) -> None:
        with store.ws_lock(ws["id"]):
            meta = store.load_ws(ws["id"])
            if meta.get("active"):
                meta["active"]["train"] = execute.identity(pid)
                store.save_ws(meta)

    status, code, seconds = "failed", None, 0.0
    try:
        status, code, seconds = execute.run_logged(cmd, path, env, limit, log, cfg.exec.wrap, on_start)
    finally:
        _collect_artifacts(c, path, node["id"])
        node.update(status=status, exit=code, seconds=seconds)
        if status == "timeout":
            node["error"] = f"training ran past its {human_duration(limit)} limit and was stopped"
        elif status == "failed":
            node["error"] = f"training exited with code {code}" if code is not None else "training was interrupted"
        node["error_log"] = str(log) if status != "ok" else None
        store.save_node(node)
    if status != "ok":
        return
    try:
        node["dev"] = evaluate.evaluate_node(c, node, "dev", ws["devices"])
    except evaluate.EvalFailed as e:
        node["status"] = "failed"
        node["error"] = f"scoring on dev failed: {e}"
        node["error_log"] = str(e.log)
    store.save_node(node)


def _collect_artifacts(c: "Campaign", path: Path, nid: str) -> None:
    """Move a run's outputs into the store and leave links to them in the workspace."""
    for rel in c.cfg.run.artifacts:
        src = path / rel
        if src.is_symlink():
            src.unlink()
            continue
        if not src.exists():
            continue
        dst = c.store.artifacts_dir(nid) / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        make_read_only(dst)
        src.symlink_to(dst)


def run_cmd(c: "Campaign", wid: str) -> int:
    ws = c.store.load_ws(wid)
    print(f"Training in {wid} ...", flush=True)
    node = do_run(c, wid)
    if node["status"] != "ok":
        print(f"{node['id']} failed: {node.get('error')}.")
        log_tail = execute.tail(node.get("error_log"))
        if log_tail:
            print(f"Last lines of {node.get('error_log')}:\n{log_tail}")
        return 1
    ws = c.store.load_ws(wid)
    base = c.store.load_node(ws["base"])
    head = c.store.load_node(ws["head"])
    print(f"{node['id']}: dev {metrics_text(c, node['dev'])}, trained {human_duration(node['seconds'])}.")
    if node.get("note"):
        print(f"  Note: {node['note']}.")
    compare = f"base {base['id']}: {fmt_num(c.dev_score(base))}"
    if head["id"] != base["id"]:
        compare += f", your last commit {head['id']}: {fmt_num(c.dev_score(head))}"
    print(f"  {compare}")
    print("`ur commit` keeps it; `ur reset` goes back to your last commit.")
    return 0


# Keeping and reverting.


def commit_cmd(c: "Campaign", wid: str, candidate: bool) -> int:
    store, cfg = c.store, c.cfg
    c.verify_seal()
    with store.ws_lock(wid):
        ws = store.load_ws(wid)
        if ws["closed"]:
            raise UrError("This workspace's round is closed, so nothing more counts here.")
        last = ws.get("last_run")
        if not last:
            raise UrError("Nothing to commit yet: `ur run` first.")
        node = store.load_node(last)
        if node["status"] != "ok":
            raise UrError(f"Your last run ({last}) failed, so there's nothing to commit. Fix it and `ur run` again.")
        tree = store.snapshot(store.ws_path(wid), c.snapshot_matcher(), cfg.workspace.max_file_mb, cfg.workspace.max_total_mb)
        if tree != node["tree"]:
            changed = [p for _, p in store.changed(node["tree"], tree)]
            raise UrError(
                f"The workspace changed since your last run ({last}):\n  " + "\n  ".join(changed)
                + "\nIf you changed them, `ur run` again, then commit. If training wrote them, delete them and commit."
            )
        limit = ws.get("max_candidates")
        if candidate and last not in ws["candidates"] and limit and len(ws["candidates"]) >= limit:
            raise UrError(f"You've already submitted {limit} candidate{'s' if limit != 1 else ''}, the most this round takes.")
        ws["head"] = last
        if candidate and last not in ws["candidates"]:
            ws["candidates"].append(last)
        ws["last_activity"] = now()
        store.save_ws(ws)
    print(f"Kept {last}: dev {metrics_text(c, node['dev'])}." + (" It's a candidate." if candidate else ""))
    return 0


def reset_cmd(c: "Campaign", wid: str, ref: str | None) -> int:
    store = c.store
    ws = store.load_ws(wid)
    _require_idle(ws)
    nid = resolve_ref(c, ws, ref or "head")
    node = store.load_node(nid)
    path = store.ws_path(wid)
    store.restore(path, node["commit"], c.snapshot_matcher())
    c.link_artifacts(nid, path)
    print(f"Reset the workspace to {nid}." + (" That's your last commit." if nid == ws["head"] else ""))
    return 0


# Looking at changes.


def _current_tree(c: "Campaign", wid: str) -> str:
    cfg = c.cfg
    return c.store.snapshot(c.store.ws_path(wid), c.snapshot_matcher(), cfg.workspace.max_file_mb, cfg.workspace.max_total_mb)


def diff_cmd(c: "Campaign", wid: str, a: str | None, b: str | None, stat: bool) -> int:
    store = c.store
    ws = store.load_ws(wid)
    left = store.load_node(resolve_ref(c, ws, a or "head"))["commit"]
    right = store.load_node(resolve_ref(c, ws, b))["commit"] if b else _current_tree(c, wid)
    print(store.diff(left, right, stat=stat), end="")
    return 0


def _is_binary(data: bytes | None) -> bool:
    return data is not None and b"\0" in data[:8192]


def _read_current(path: Path) -> bytes | None:
    if path.is_symlink():
        return os.readlink(path).encode()
    if path.is_file():
        return path.read_bytes()
    return None


def _write(path: Path, data: bytes, mode: str) -> None:
    if path.is_symlink() or path.exists():
        remove_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "120000":
        path.symlink_to(data.decode())
        return
    path.write_bytes(data)
    if mode == "100755":
        path.chmod(path.stat().st_mode | 0o111)


def _merge(c: "Campaign", ours: bytes, base: bytes, theirs: bytes, label: str) -> tuple[bytes, int]:
    with tempfile.TemporaryDirectory(dir=c.store.tmp()) as d:
        files = []
        for name, data in (("ours", ours), ("base", base), ("theirs", theirs)):
            p = Path(d) / name
            p.write_bytes(data)
            files.append(str(p))
        proc = subprocess.run(
            ["git", "merge-file", "-p", "-L", "yours", "-L", "base", "-L", label, *files],
            capture_output=True,
        )
        if proc.returncode < 0 or proc.returncode > 127:
            raise UrError(f"git merge-file failed: {proc.stderr.decode(errors='replace').strip()}")
        return proc.stdout, proc.returncode


def apply_cmd(c: "Campaign", wid: str, ref: str) -> int:
    """Three-way merge a node's changes (relative to this workspace's base) into the workspace."""
    store = c.store
    ws = store.load_ws(wid)
    if ws["closed"]:
        raise UrError("This workspace's round is closed, so nothing more counts here.")
    _require_idle(ws)
    nid = resolve_ref(c, ws, ref)
    base_commit = store.load_node(ws["base"])["commit"]
    their_commit = store.load_node(nid)["commit"]
    base_files, their_files = store.files(base_commit), store.files(their_commit)
    path = store.ws_path(wid)
    applied: list[str] = []
    conflicts: list[str] = []
    for _, rel in store.changed(base_commit, their_commit):
        target = path / rel
        cur = _read_current(target)
        old = store.blob(base_files[rel][1]) if rel in base_files else None
        if rel not in their_files:
            if cur is None:
                continue
            if cur == old:
                target.unlink()
                applied.append(f"{rel} (deleted)")
            else:
                conflicts.append(f"{rel}: deleted in {nid} but changed here; kept yours")
            continue
        mode, sha = their_files[rel]
        new = store.blob(sha)
        if cur == new:
            continue
        if cur is None:
            if old is None:
                _write(target, new, mode)
                applied.append(f"{rel} (new)")
            else:
                conflicts.append(f"{rel}: changed in {nid} but deleted here; left deleted")
            continue
        if cur == old:
            _write(target, new, mode)
            applied.append(rel)
            continue
        if mode == "120000" or target.is_symlink() or _is_binary(cur) or _is_binary(new) or _is_binary(old):
            conflicts.append(f"{rel}: binary or link changed on both sides; kept yours")
            continue
        merged, n = _merge(c, cur, old or b"", new, nid)
        target.write_bytes(merged)
        if n:
            conflicts.append(f"{rel}: {n} conflict{'s' if n != 1 else ''}, marked <<<<<<< yours ... >>>>>>> {nid}")
        else:
            applied.append(f"{rel} (merged)")
    if not applied and not conflicts:
        print(f"{nid}'s changes are already in your workspace.")
        return 0
    parts = []
    if applied:
        parts.append(f"{len(applied)} file{'s' if len(applied) != 1 else ''} merged cleanly")
    if conflicts:
        parts.append(f"{len(conflicts)} need{'s' if len(conflicts) == 1 else ''} you")
    print(f"Applied {nid}: {', '.join(parts)}" + (":" if conflicts else "."))
    for line in conflicts:
        print(f"  {line}")
    print("Then `ur run`." if not conflicts else "Resolve those, then `ur run`.")
    return 0


# Overview.


def log_cmd(c: "Campaign", wid: str) -> int:
    store = c.store
    ws = store.load_ws(wid)
    base = store.load_node(ws["base"])
    print(f"{base['id']}  base         dev {metrics_text(c, base.get('dev'))}")
    for nid in ws.get("runs", []):
        node = store.load_node(nid)
        mark = "commit" if nid == ws["head"] else ""
        if nid in ws.get("candidates", []):
            mark = "candidate"
        if node["status"] == "ok":
            what = f"dev {metrics_text(c, node.get('dev'))}"
        else:
            what = f"{node['status']}: {node.get('error', '')}"
        took = human_duration(node["seconds"]) if node.get("seconds") else ""
        print(f"{nid:<5} {mark:<12} {what}  {took}".rstrip())
    return 0


def status_cmd(c: "Campaign", wid: str, as_json: bool) -> int:
    import json

    store = c.store
    ws = store.load_ws(wid)
    base = store.load_node(ws["base"])
    head = store.load_node(ws["head"])
    last = store.load_node(ws["last_run"]) if ws.get("last_run") else None
    info = {
        "workspace": wid,
        "path": str(store.ws_path(wid)),
        "closed": ws["closed"],
        "deadline": ws.get("deadline"),
        "seconds_left": max(0.0, ws["deadline"] - now()) if ws.get("deadline") else None,
        "base": {"node": base["id"], "dev": base.get("dev")},
        "head": {"node": head["id"], "dev": head.get("dev")},
        "last_run": {"node": last["id"], "status": last["status"], "dev": last.get("dev")} if last else None,
        "running": execute.alive(ws.get("active")),
    }
    if as_json:
        print(json.dumps(info, indent=2))
        return 0
    if ws["closed"]:
        when = "closed"
    elif ws.get("deadline"):
        when = f"ends {clock(ws['deadline'])} (in {human_duration(ws['deadline'] - now())})"
    else:
        when = "no deadline"
    print(f"Workspace {wid} · {when}")
    print(f"Base {base['id']}: dev {metrics_text(c, base.get('dev'))}")
    if head["id"] == base["id"]:
        print("Your last commit: none yet")
    else:
        print(f"Your last commit: {head['id']}: dev {metrics_text(c, head.get('dev'))}")
    if info["running"]:
        print(f"Running: {ws['active'].get('node')}")
    elif last and last["id"] != head["id"]:
        detail = f"dev {metrics_text(c, last.get('dev'))}" if last["status"] == "ok" else f"{last['status']}: {last.get('error', '')}"
        print(f"Last run: {last['id']}: {detail} (not committed)")
    return 0
