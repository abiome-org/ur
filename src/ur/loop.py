"""Campaign commands run from the project: init, check, round, close, select, and the rest."""

from __future__ import annotations

import functools
import hashlib
import shutil
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from . import config as config_mod
from . import evaluate, execute, guards, prompts
from .campaign import Campaign
from .util import UrError, clock, fill, fmt_num, force_rmtree, human_duration, make_writable, now, sha256_file
from .workspace import clear_stale_run, do_run


def _exclusive(fn):
    """Commands that move the campaign between phases run one at a time."""

    @functools.wraps(fn)
    def wrapper(c: Campaign, *args: Any, **kwargs: Any) -> int:
        with c.exclusive():
            return fn(c, *args, **kwargs)

    return wrapper


# Setup.


def init_cmd(project: Path) -> int:
    ur = project / ".ur"
    if (ur / "config.toml").exists():
        raise UrError(f"{ur / 'config.toml'} already exists.")
    (ur / "eval").mkdir(parents=True, exist_ok=True)
    (ur / "checks").mkdir(exist_ok=True)
    (ur / ".gitignore").write_text("state/\narchive/\n")
    (ur / "config.toml").write_text(config_mod.template(project))
    print(f"Wrote {ur / 'config.toml'}, with empty .ur/eval/ and .ur/checks/.")
    print("Next: fill in the config with the user, put the scorer in .ur/eval/, then `ur check`.")
    return 0


def _stop_ws(c: Campaign, wid: str) -> None:
    """Close a workspace and stop whatever is still running in it."""
    store = c.store
    try:
        with store.ws_lock(wid):
            ws = store.load_ws(wid)
            ws["closed"] = True
            store.save_ws(ws)
    except UrError:
        return
    execute.stop(ws.get("session"))
    active = ws.get("active")
    if execute.alive(active):
        execute.stop(active)
    if active and execute.alive(active.get("train")):
        execute.kill_group(active["train"]["pid"])
    if active:
        with store.ws_lock(wid):
            ws = store.load_ws(wid)
            clear_stale_run(c, ws)
            store.save_ws(ws)


def _spawn(c: Campaign, wid: str) -> dict[str, Any]:
    store = c.store
    with store.ws_lock(wid):
        ws = store.load_ws(wid)
        if ws["closed"]:
            raise UrError(f"{wid} is closed.")
        if execute.alive(ws.get("session")):
            raise UrError(f"A session is already running in {wid} (pid {ws['session']['pid']}).")
        path, prompt_file = store.ws_path(wid), store.prompt_path(wid)
        cmd = fill(c.cfg.agents.command, workspace=path, prompt_file=prompt_file, prompt=prompt_file.read_text())
        # UR_SESSION keeps the session to its own workspace's commands.
        ws["session"] = execute.spawn(cmd, path, c.env(ws["devices"], UR_SESSION=wid), store.session_log(wid))
        store.save_ws(ws)
    return ws


def _start_sessions(c: Campaign, wids: list[str]) -> str:
    """Start sessions in command mode; otherwise say how to start them."""
    if c.cfg.agents.spawn == "command":
        for wid in wids:
            _spawn(c, wid)
        return f"ur started {len(wids)} session{'s' if len(wids) != 1 else ''} (logs in {c.store.root / 'sessions'})."
    if len(wids) == 1:
        return f"Start one fresh session with `ur prompt {wids[0]}` as its prompt, unchanged."
    return "Start one fresh session per workspace, with `ur prompt <workspace>` as its prompt, unchanged."


def _write_prompt(c: Campaign, wid: str, text: str) -> Path:
    path = c.store.prompt_path(wid)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _record_hashes(path: Path, limit: float | None) -> tuple[set[bytes], bool]:
    """Hashes of the records under path: lines of line-based text files, whole files otherwise.

    CSV/TSV header lines and very short lines are skipped, since they match across splits without
    meaning the same example.
    """
    line_suffixes = {".jsonl", ".ndjson", ".txt", ".csv", ".tsv"}
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    hashes: set[bytes] = set()
    total = 0
    for f in files:
        if f.suffix.lower() in line_suffixes:
            skip_header = f.suffix.lower() in (".csv", ".tsv")
            with open(f, "rb") as fh:
                for i, line in enumerate(fh):
                    total += len(line)
                    line = line.strip()
                    if len(line) >= 8 and not (skip_header and i == 0):
                        hashes.add(hashlib.blake2b(line, digest_size=16).digest())
                    if limit and total > limit:
                        return hashes, True
        else:
            hashes.add(hashlib.blake2b(f.read_bytes(), digest_size=16).digest())
            total += f.stat().st_size
            if limit and total > limit:
                return hashes, True
    return hashes, False


def _overlap(c: Campaign) -> tuple[bool, list[str]]:
    cfg = c.cfg
    mode = cfg.check.overlap
    if mode == "off":
        return True, ["Overlap: skipped (check.overlap is off)."]
    paths = {name: config_mod.resolve(c.project, cfg.splits[name]) for name in ("train", "dev", "heldout", "final")}
    if mode != "builtin":
        cmd = fill(mode, **{k: v or "" for k, v in paths.items()})
        proc = subprocess.run(cmd, shell=True, cwd=c.project, capture_output=True, text=True)
        detail = (proc.stdout + proc.stderr).strip()
        ok = proc.returncode == 0
        return ok, [f"Overlap: {'ok' if ok else 'FAILED'} (check.overlap command exited {proc.returncode})" + (f"\n{detail}" if detail else "")]
    protected = {name: _record_hashes(paths[name], None)[0] for name in ("heldout", "final") if paths[name]}
    lines, ok = [], True
    for name in ("train", "dev"):
        if not paths[name]:
            continue
        seen, truncated = _record_hashes(paths[name], cfg.check.overlap_max_mb * 1e6)
        for pname, phashes in protected.items():
            shared = len(seen & phashes)
            good = shared <= cfg.check.max_overlap
            ok = ok and good
            note = f" (read the first {cfg.check.overlap_max_mb:g} MB of {name})" if truncated else ""
            lines.append(f"Overlap {name}/{pname}: {shared} shared record{'s' if shared != 1 else ''}{note}{'' if good else '  FAILED'}")
    return ok, lines


@_exclusive
def check_cmd(c: Campaign) -> int:
    cfg, store = c.cfg, c.store
    s = c.load_state()
    if s["rounds"] or ((s.get("check") or {}).get("passed") and c.brief_path.exists()):
        raise UrError("Setup is complete, and `ur check` only runs during setup. To set up again, `ur archive` first.")
    errs = config_mod.problems(cfg, c.project)
    if not c.eval_dir.is_dir() or not any(c.eval_dir.iterdir()):
        errs.append(".ur/eval/ is empty: put the scorer there.")
    if errs:
        raise UrError("Fix these first:\n  " + "\n  ".join(errs))
    for w in config_mod.warnings(cfg, c.project):
        print(f"Warning: {w}")
    if store.exists():
        for ws in store.all_ws():
            if ws["purpose"] in ("probe", "baseline"):
                _stop_ws(c, ws["id"])
    store.ensure(c.project)
    with c.update_state() as s:
        s["check"] = {"passed": False, "started": now()}
        s["phase"] = "setup"
        s["base"] = None
        c.seal(s)
    print(f"Sealed the scorer ({len(s['sealed']['eval'])} file{'s' if len(s['sealed']['eval']) != 1 else ''}).", flush=True)

    # The project as it is now becomes the root of every lineage.
    tree = store.snapshot(c.project, c.snapshot_matcher(), cfg.workspace.max_file_mb, cfg.workspace.max_total_mb)
    root_id = store.new_id()
    commit = store.commit(tree, None, "project")
    store.save_node({"id": root_id, "workspace": None, "round": 0, "commit": commit, "tree": tree, "status": "source", "devices": []})
    store.keep(root_id, commit)

    # Baselines: proves the loop works end to end, measures noise and cost per run.
    slices = c.device_slices()
    n = cfg.check.baseline_runs
    print(f"Training the baseline {n} time{'s' if n != 1 else ''} ...", flush=True)
    wids = [c.create_workspace(f"setup-b{i + 1}", root_id, "baseline", 0, None, [], slices[i % len(slices)])["id"] for i in range(n)]
    parallel = len(slices) if cfg.exec.devices else 1
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        nodes = list(pool.map(lambda wid: do_run(c, wid), wids))
    for node in nodes:
        if node["status"] != "ok":
            continue
        after = store.snapshot(store.ws_path(node["workspace"]), c.snapshot_matcher(), cfg.workspace.max_file_mb, cfg.workspace.max_total_mb)
        stray = [p for _, p in store.changed(node["tree"], after)]
        if stray:
            raise UrError(
                "Training writes files outside run.artifacts:\n  " + "\n  ".join(stray)
                + "\nWorkers couldn't commit after such a run. Add them to run.artifacts (outputs to keep with the "
                "checkpoint) or workspace.exclude, then `ur check` again."
            )
    failed = [node for node in nodes if node["status"] != "ok"]
    if failed:
        node = failed[0]
        tail = execute.tail(node.get("error_log"))
        raise UrError(
            f"The baseline failed ({node['id']}: {node.get('error')}). Last lines of {node.get('error_log')}:\n{tail}\n"
            "Fix the training or scoring command, then `ur check` again."
        )
    evaluate.score_nodes(c, nodes, "heldout")
    if cfg.splits.final:
        evaluate.score_nodes(c, nodes, "final")
    broken = [node for node in nodes if c.score(node) is None]
    if broken:
        raise UrError(f"Scoring the baseline on held-out failed: {broken[0].get('errors', {}).get('heldout')}. Fix the scorer, then `ur check` again.")
    scores = [c.score(node) for node in nodes]
    mean = statistics.fmean(scores)
    noise = statistics.stdev(scores) if len(scores) > 1 else 0.0
    seconds = statistics.fmean(node["seconds"] for node in nodes)
    base = c.rank(nodes)[len(nodes) // 2]
    print(f"Baseline: held-out {c.metric} {fmt_num(mean)} ± {fmt_num(noise)}, {human_duration(seconds)} of training per run. Base is {base['id']}.")

    ok = True
    expect_results = []
    for i, entry in enumerate(cfg.check.expect):
        split = entry.get("split", "heldout")
        name = entry.get("name") or entry["workspace"]
        log = store.root / "check" / f"expect-{i + 1}.log"
        try:
            metrics = evaluate.evaluate_path(c, config_mod.resolve(c.project, entry["workspace"]), split, log)
            value = metrics.get(entry["metric"])
        except evaluate.EvalFailed as e:
            value, why = None, str(e)
        else:
            why = None if value is not None else f"no {entry['metric']} in the scores"
        good = value is not None and ("min" not in entry or value >= entry["min"]) and ("max" not in entry or value <= entry["max"])
        bounds = " and ".join(f"{k} {entry[k]:g}" for k in ("min", "max") if k in entry)
        line = f"Expect {name}: {split} {entry['metric']} {fmt_num(value)} (want {bounds})"
        print(line + ("" if good else f"  FAILED{f': {why}' if why else ''}"))
        expect_results.append({"name": name, "value": value, "ok": good})
        ok = ok and good

    overlap_ok, lines = _overlap(c)
    for line in lines:
        print(line)
    ok = ok and overlap_ok

    probe: dict[str, Any] = {"status": "skipped"}
    if ok and cfg.check.probe:
        budget = cfg.check.probe_budget or cfg.run.budget * 0.02
        if cfg.run.unit != "seconds" and isinstance(cfg.run.budget, int):
            budget = max(1, int(round(budget)))
        deadline = now() + cfg.check.probe_minutes * 60
        ws = c.create_workspace("probe", base["id"], "probe", 0, deadline, [], slices[0], budget=budget)
        path = store.ws_path("probe")
        _write_prompt(c, "probe", prompts.probe(path, prompts.budget_text(budget, cfg.run.unit), deadline))
        probe = {"status": "pending", "workspace": "probe", "deadline": deadline, "budget": budget}
    with c.update_state() as s:
        s["check"].update(
            base=base["id"],
            baseline={"runs": [node["id"] for node in nodes], "mean": mean, "noise": noise, "seconds": seconds},
            expect=expect_results,
            overlap=overlap_ok,
            probe=probe,
            passed=ok and probe["status"] == "skipped",
            finished=now(),
        )
        if s["check"]["passed"]:
            s["base"] = base["id"]
            s["phase"] = "idle"
    if not ok:
        print("Next: fix what FAILED, then `ur check` again.")
        return 1
    if probe["status"] == "pending":
        print(f"Cheat probe: workspace {store.ws_path('probe')}, until {clock(probe['deadline'])}.")
        print(_start_sessions(c, ["probe"]))
        print("Next: when it finishes (`ur wait`), `ur check --probe`.")
        return 0
    print("Setup checks passed. Next: write .ur/brief.md (`ur brief --draft` prints one).")
    return 0


@_exclusive
def check_probe_cmd(c: Campaign, accept: bool) -> int:
    store = c.store
    s = c.load_state()
    probe = (s.get("check") or {}).get("probe") or {}
    allowed = ("pending", "failed") if accept else ("pending",)
    if probe.get("status") not in allowed:
        raise UrError("No cheat probe is waiting to be scored. `ur check` starts one.")
    _stop_ws(c, "probe")
    ws = store.load_ws("probe")
    baseline = s["check"]["baseline"]
    mean, noise = baseline["mean"], baseline["noise"]
    head = ws["head"]
    if head == ws["base"]:
        verdict, detail = "passed", "The probe didn't commit anything."
    else:
        node = store.load_node(head)
        node["heldout"], node["heldout_repeats"] = None, []
        evaluate.score_nodes(c, [node], "heldout")
        score = c.score(node)
        margin = max(3 * noise, 1e-9 * max(1.0, abs(mean)))
        bar = mean + margin if c.cfg.objective.direction == "max" else mean - margin
        hole = c.better(score, bar)
        detail = (
            f"The probe's commit {head} scored {fmt_num(score)} on held-out (dev {fmt_num(c.dev_score(node))}) "
            f"with {prompts.budget_text(ws['budget'], c.cfg.run.unit)} per run. The baseline scores {fmt_num(mean)} ± {fmt_num(noise)}."
        )
        if score is None:
            verdict = "passed"
            detail = f"The probe's commit {head} couldn't be scored: {node.get('errors', {}).get('heldout')}."
        elif hole:
            verdict = "accepted" if accept else "failed"
        else:
            verdict = "passed"
    with c.update_state() as s:
        s["check"]["probe"].update(status=verdict, node=head, finished=now())
        if verdict in ("passed", "accepted"):
            s["check"]["passed"] = True
            s["base"] = s["check"]["base"]
            s["phase"] = "idle"
    print(detail)
    if verdict == "failed":
        print(
            "That's a hole: something other than training raised the score. See what it did with "
            f"`ur diff {ws['base']} {head}`, close the hole, then `ur check` again. If the gain is a legitimate "
            "improvement, `ur check --probe --accept`."
        )
        return 1
    print(("Accepted the probe's result. " if verdict == "accepted" else "No hole found. ") + "Setup checks passed.")
    print("Next: write .ur/brief.md (`ur brief --draft` prints one).")
    return 0


def brief_cmd(c: Campaign, draft: bool) -> int:
    if draft:
        print(prompts.brief_draft(c.cfg), end="")
        return 0
    if not c.brief_path.exists():
        raise UrError("There's no .ur/brief.md yet. `ur brief --draft` prints one built from the config.")
    print(c.brief_path.read_text(), end="")
    return 0


# Rounds.


def _require(c: Campaign, s: dict[str, Any], phase: str, what: str) -> None:
    current = c.phase(s)
    if current != phase:
        raise UrError(f"Can't {what} now: the campaign is in {current}. `ur status` says what's next.")


@_exclusive
def round_cmd(c: Campaign) -> int:
    cfg, store = c.cfg, c.store
    s = c.load_state()
    _require(c, s, "idle", "open a round")
    c.verify_seal(s)
    errs = config_mod.problems(cfg, c.project)
    if errs:
        raise UrError("Fix these first:\n  " + "\n  ".join(errs))
    brief = c.brief_path.read_text()
    with c.update_state() as s:
        if not s["sealed"].get("brief"):
            s["sealed"]["brief"] = sha256_file(c.brief_path)
        r = s["round"] + 1
        deadline = now() + cfg.round.minutes * 60
        slices = c.device_slices()
        wids = []
        for i in range(cfg.round.workers):
            wid = f"r{r}-w{i + 1}"
            c.create_workspace(wid, s["base"], "worker", r, deadline, [], slices[i % len(slices)])
            _write_prompt(c, wid, prompts.worker(brief, store.ws_path(wid), deadline))
            wids.append(wid)
        s["round"] = r
        s["phase"] = "round_open"
        s["rounds"].append(
            {
                "n": r,
                "base": s["base"],
                "opened": now(),
                "deadline": deadline,
                "workers": wids,
                "closed": None,
                "ranking": [],
                "construct": None,
                "construct_deadline": None,
                "selected": None,
            }
        )
    print(f"Round {r} is open until {clock(deadline)}, from base {s['base']}. Workspaces:")
    for wid in wids:
        print(f"  {wid}  {store.ws_path(wid)}")
    print(_start_sessions(c, wids))
    print("Next: wait for the deadline or for every worker to finish (`ur wait`), then `ur close`.")
    return 0


def prompt_cmd(c: Campaign, wid: str) -> int:
    path = c.store.prompt_path(wid)
    if not path.exists():
        raise UrError(f"There's no prompt for {wid}.")
    print(path.read_text(), end="")
    return 0


def spawn_cmd(c: Campaign, wid: str) -> int:
    if c.cfg.agents.spawn != "command":
        raise UrError(
            f'agents.spawn is "{c.cfg.agents.spawn}", so ur doesn\'t start sessions. Start a fresh session in '
            f"{c.store.ws_path(wid)} with `ur prompt {wid}` as its prompt."
        )
    ws = _spawn(c, wid)
    print(f"Started a session in {wid} (pid {ws['session']['pid']}, log {ws['session']['log']}).")
    return 0


@_exclusive
def close_cmd(c: Campaign) -> int:
    cfg, store = c.cfg, c.store
    s = c.load_state()
    _require(c, s, "round_open", "close the round")
    c.verify_seal(s)
    rnd = s["rounds"][-1]
    for wid in rnd["workers"]:
        _stop_ws(c, wid)
    heads: dict[str, str] = {}
    for wid in rnd["workers"]:
        ws = store.load_ws(wid)
        if ws["head"] != ws["base"]:
            heads[ws["head"]] = wid
    nodes = [store.load_node(nid) for nid in heads]
    print(f"Scoring {len(nodes)} committed worker{'s' if len(nodes) != 1 else ''} on held-out ...", flush=True)
    # Always score here, whatever a node's record already says: workers can write to their own records.
    for n in nodes:
        n["heldout"], n["heldout_repeats"] = None, []
        n.get("errors", {}).pop("heldout", None)
    evaluate.score_nodes(c, nodes, "heldout")
    base = store.load_node(rnd["base"])
    ranked = c.rank(nodes)
    broken = {n["id"]: guards.failures(cfg.objective.guards, n["heldout"], base.get("heldout")) for n in ranked}
    improved = [n for n in ranked if not broken[n["id"]] and c.better(c.score(n), c.score(base))]
    construct, construct_deadline = None, None
    if len(improved) >= 2:
        construct = f"r{rnd['n']}-c"
        construct_deadline = now() + cfg.round.construct_minutes * 60
        top = [n["id"] for n in improved[: cfg.round.top_m]]
        ws = c.create_workspace(construct, base["id"], "construct", rnd["n"], construct_deadline, top, c.device_slices()[0])
        ws["max_candidates"] = cfg.round.max_candidates
        store.save_ws(ws)
        _write_prompt(
            c,
            construct,
            prompts.construct(
                store.ws_path(construct), base["id"], top, cfg.round.max_candidates, construct_deadline, cfg.objective.guards
            ),
        )
    with c.update_state() as s:
        rnd = s["rounds"][-1]
        rnd.update(
            closed=now(),
            ranking=[{"node": n["id"], "workspace": heads[n["id"]], "score": c.score(n)} for n in ranked],
            construct=construct,
            construct_deadline=construct_deadline,
        )
        s["phase"] = "round_closed"
    print(f"Round {rnd['n']} closed. Base {base['id']}: held-out {c.metric} {fmt_num(c.score(base))}.")
    for n in ranked:
        note = "  beats the base" if n in improved else ""
        if broken[n["id"]]:
            note = f"  breaks a guard: {'; '.join(broken[n['id']])}"
        print(f"  {n['id']}  {heads[n['id']]:<8} held-out {fmt_num(c.score(n))}{note}")
    for n in nodes:
        if c.score(n) is None:
            print(f"  {n['id']}  {heads[n['id']]:<8} couldn't be scored: {n.get('errors', {}).get('heldout')}")
    if construct:
        print(f"{len(improved)} workers beat the base, so {construct} combines them, until {clock(construct_deadline)}.")
        print(_start_sessions(c, [construct]))
        print("Next: when it finishes (`ur wait`), `ur select`.")
    else:
        print("Next: `ur select`.")
    return 0


def _rescore(c: Campaign, s: dict[str, Any], node: dict[str, Any], rnd: dict[str, Any]) -> None:
    """Train a node's code once more and add the new held-out score to its average."""
    wid = f"r{rnd['n']}-again-{node['id']}-{len(node.get('heldout_repeats') or [])}"
    c.create_workspace(wid, node["id"], "rescore", rnd["n"], None, [], c.device_slices()[0], init_from=node.get("init_from"))
    rnd.setdefault("rescore", []).append(wid)
    again = do_run(c, wid)
    if again["status"] != "ok":
        return
    evaluate.score_nodes(c, [again], "heldout")
    value = c.score(again)
    if value is not None:
        node.setdefault("heldout_repeats", []).append(value)
        c.store.save_node(node)


def _cleanup(c: Campaign, s: dict[str, Any], rnd: dict[str, Any]) -> None:
    """Delete the round's workspaces, and the checkpoints nothing will need."""
    store = c.store
    keep = {r.get("base") for r in s["rounds"]} | {r.get("selected") for r in s["rounds"]} | {s.get("base")}
    keep |= {item["node"] for item in rnd.get("ranking", [])[: c.cfg.store.keep_top]}
    wids = list(rnd["workers"]) + ([rnd["construct"]] if rnd.get("construct") else []) + list(rnd.get("rescore", []))
    for wid in wids:
        try:
            ws = store.load_ws(wid)
        except UrError:
            continue
        for nid in ws.get("runs", []):
            if nid not in keep:
                force_rmtree(store.artifacts_dir(nid))
        force_rmtree(store.ws_path(wid))


def _gpu_hours(c: Campaign, rnd: dict[str, Any]) -> float:
    total = 0.0
    wids = list(rnd["workers"]) + ([rnd["construct"]] if rnd.get("construct") else []) + list(rnd.get("rescore", []))
    for wid in wids:
        try:
            ws = c.store.load_ws(wid)
        except UrError:
            continue
        for nid in ws.get("runs", []):
            node = c.store.load_node(nid)
            total += (node.get("seconds") or 0.0) * max(1, len(node.get("devices") or []))
    return total / 3600


@_exclusive
def select_cmd(c: Campaign) -> int:
    cfg, store = c.cfg, c.store
    s = c.load_state()
    _require(c, s, "round_closed", "select")
    c.verify_seal(s)
    rnd = s["rounds"][-1]
    if rnd.get("construct"):
        _stop_ws(c, rnd["construct"])
    base = store.load_node(rnd["base"])
    pool: dict[str, dict[str, Any]] = {base["id"]: base}
    for item in rnd["ranking"]:
        pool[item["node"]] = store.load_node(item["node"])
    candidates = []
    if rnd.get("construct"):
        submitted = store.load_ws(rnd["construct"])["candidates"][: cfg.round.max_candidates]
        candidates = [nid for nid in submitted if store.load_node(nid)["status"] == "ok"]
        for nid in candidates:
            pool[nid] = store.load_node(nid)
    for nid in candidates:  # scored here, never taken from their records
        pool[nid]["heldout"], pool[nid]["heldout_repeats"] = None, []
        pool[nid].get("errors", {}).pop("heldout", None)
    unscored = [n for n in pool.values() if c.score(n) is None and not (n.get("errors") or {}).get("heldout")]
    if unscored:
        print(f"Scoring {len(unscored)} candidate{'s' if len(unscored) != 1 else ''} on held-out ...", flush=True)
        evaluate.score_nodes(c, unscored, "heldout")
    eligible, rejected = [], []
    for node in pool.values():
        if c.score(node) is None:
            continue
        broken = [] if node["id"] == base["id"] else guards.failures(cfg.objective.guards, node["heldout"], base.get("heldout"))
        if broken:
            rejected.append((node["id"], broken))
        else:
            eligible.append(node)
    ranked = c.rank(eligible)
    noise = s["check"]["baseline"]["noise"]
    if len(ranked) > 1 and noise > 0 and abs(c.score(ranked[0]) - c.score(ranked[1])) < 2 * noise:
        print(f"{ranked[0]['id']} and {ranked[1]['id']} are within noise; training both once more ...", flush=True)
        for node in ranked[:2]:
            _rescore(c, s, node, rnd)
        ranked = c.rank(eligible)
    best = ranked[0]
    improved = best["id"] != base["id"] and c.better(c.score(best), c.score(base))
    new_base = best if improved else base
    if improved and cfg.splits.final and not new_base.get("final"):
        evaluate.score_nodes(c, [new_base], "final")
        new_base = store.load_node(new_base["id"])
    hours = _gpu_hours(c, rnd)
    ran = any(store.load_ws(wid).get("runs") for wid in rnd["workers"])
    with c.update_state() as s:
        r = s["rounds"][-1]
        r.update(
            selected=new_base["id"],
            score=c.score(new_base),
            improved=improved,
            candidates=candidates,
            rejected=[{"node": nid, "why": why} for nid, why in rejected],
            rescore=rnd.get("rescore", []),
            gpu_hours=hours,
            finished=now(),
        )
        s["base"] = new_base["id"]
        if improved:
            s["stalls"] = 0
        elif ran:
            s["stalls"] += 1
        s["empty_rounds"] = 0 if ran else s.get("empty_rounds", 0) + 1
        s["gpu_hours"] += hours
        reason = c.stop_reason(s)
        s["phase"] = "stopped" if reason else "idle"
        s["stopped_reason"] = reason
        rnd = r
    _cleanup(c, s, rnd)
    for nid, why in rejected:
        print(f"  {nid} broke a guard: {'; '.join(why)}")
    if not ran:
        print(
            f"No worker ran anything this round, so it doesn't count toward stop.patience. "
            f"Check the session logs in {store.root / 'sessions'}."
        )
    if improved:
        line = f"Round {rnd['n']}: base {base['id']} ({fmt_num(c.score(base))}) -> {new_base['id']} ({fmt_num(c.score(new_base))})"
        if new_base.get("final"):
            line += f", final {fmt_num(new_base['final'].get(c.metric))}"
    else:
        line = f"Round {rnd['n']}: base stays {base['id']} ({fmt_num(c.score(base))}); nothing beat it"
    budget = f"{s['gpu_hours']:.2f}" + (f" of {cfg.stop.max_gpu_hours:g}" if cfg.stop.max_gpu_hours else "")
    print(f"{line}. {budget} GPU-hours used.")
    print(f"Stopped: {reason}. `ur status` has the details." if reason else "Next: `ur round`.")
    return 0


def wait_cmd(c: Campaign, timeout: float | None) -> int:
    from .status import ready

    end = now() + timeout if timeout is not None else None
    while True:
        done, why = ready(c)
        if done:
            print(why)
            return 0
        if end is not None and now() >= end:
            print(f"Still waiting: {why}")
            return 1
        time.sleep(min(5.0, max(0.1, (end - now()) if end else 5.0)))


# Results and housekeeping.


def resolve_project_ref(c: Campaign, ref: str) -> str:
    if ref == "base":
        s = c.load_state()
        nid = s.get("base") or (s.get("check") or {}).get("base")
        if not nid:
            raise UrError("There's no base yet.")
        return nid
    c.store.load_node(ref)
    return ref


def diff_project_cmd(c: Campaign, a: str, b: str | None, stat: bool) -> int:
    left = c.store.load_node(resolve_project_ref(c, a))
    if b is None:
        right = c.store.load_node(left["parent"]) if left.get("parent") else None
        if right is None:
            raise UrError(f"{left['id']} has no parent; give two nodes.")
        left, right = right, left
    else:
        right = c.store.load_node(resolve_project_ref(c, b))
    print(c.store.diff(left["commit"], right["commit"], stat=stat), end="")
    return 0


def log_project_cmd(c: Campaign) -> int:
    s = c.load_state()
    check = s.get("check") or {}
    if check.get("base"):
        b = check["baseline"]
        print(f"setup    base {check['base']}  held-out {c.metric} {fmt_num(b['mean'])} ± {fmt_num(b['noise'])}")
    for r in s["rounds"]:
        if not r.get("selected"):
            state = "closed" if r.get("closed") else f"open until {clock(r['deadline'])}"
            print(f"round {r['n']:<3} {state}, from base {r['base']}")
            continue
        node = c.store.load_node(r["selected"])
        final = (node.get("final") or {}).get(c.metric)
        change = f"{r['base']} -> {r['selected']}" if r.get("improved") else f"{r['base']} stays"
        print(
            f"round {r['n']:<3} {change:<14} held-out {fmt_num(r.get('score'))}"
            + (f"  final {fmt_num(final)}" if final is not None else "")
            + f"  {r.get('gpu_hours', 0):.2f} GPU-h"
        )
    return 0


def export_cmd(c: Campaign, ref: str, dest: Path) -> int:
    nid = resolve_project_ref(c, ref)
    node = c.store.load_node(nid)
    if dest.exists() and any(dest.iterdir()):
        raise UrError(f"{dest} isn't empty.")
    c.store.materialize(node["commit"], dest)
    artifacts = c.store.artifacts_dir(nid)
    if artifacts.exists():
        shutil.copytree(artifacts, dest, dirs_exist_ok=True)
        make_writable(dest)
    print(f"Exported {nid} (code{' and checkpoints' if artifacts.exists() else ''}) to {dest}.")
    return 0


@_exclusive
def archive_cmd(c: Campaign) -> int:
    store = c.store
    for ws in store.all_ws():
        execute.stop(ws.get("session"))
        execute.stop(ws.get("active"))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = c.ur / "archive" / stamp
    dest.mkdir(parents=True)
    state_dir = c.ur / "state"
    if state_dir.exists():
        shutil.move(str(state_dir), str(dest / "state"))
    if c.brief_path.exists():
        shutil.move(str(c.brief_path), str(dest / "brief.md"))
    if store.root.exists():
        moved = store.root.with_name(f"{store.root.name}-archive-{stamp}")
        store.root.rename(moved)
        (dest / "store").write_text(f"{moved}\n")
    print(f"Archived the campaign to {dest}. The config and scorer stay, so setup can start from them.")
    print("Next: initialize again, with the user.")
    return 0


