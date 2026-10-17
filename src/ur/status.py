"""`ur status` from the project, and `ur wait`'s readiness test. Both say what to do next."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import config as config_mod
from . import execute
from .campaign import Campaign
from .util import UrError, clock, fmt_num, human_duration, now

NO_CAMPAIGN = "No .ur/ here or in any parent directory."


def none_status(as_json: bool) -> int:
    nxt = "initialize, with the user (`ur init` writes a config template)."
    if as_json:
        print(json.dumps({"phase": "none", "next": nxt}, indent=2))
    else:
        print(f"{NO_CAMPAIGN}\nNext: {nxt}")
    return 0


def _command_mode(c: Campaign) -> bool:
    return c.cfg.agents.spawn == "command"


def _sessions_done(c: Campaign, wids: list[str]) -> bool:
    for wid in wids:
        ws = c.store.load_ws(wid)
        if execute.alive(ws.get("session")) or execute.alive(ws.get("active")):
            return False
    return True


def _died(c: Campaign, wids: list[str]) -> list[str]:
    return [w for w in wids if execute.session_state(c.store.load_ws(w).get("session")) == "died"]


def _waiting(c: Campaign, s: dict[str, Any], ph: str) -> tuple[bool, str] | None:
    """For phases that wait on sessions: (ready, what's next). None for the others."""
    command = _command_mode(c)
    if ph == "setup-probe":
        probe = s["check"]["probe"]
        if now() >= probe["deadline"]:
            return True, "the probe's time is up. Next: `ur check --probe`."
        if command and _sessions_done(c, ["probe"]):
            return True, "the probe session has ended. Next: `ur check --probe`."
        if command:
            return False, (
                f"ur started the cheat probe; it has until {clock(probe['deadline'])}. "
                "When it finishes (`ur wait`), `ur check --probe`."
            )
        return False, "start one fresh session with `ur prompt probe` as its prompt; when it finishes, `ur check --probe`."
    if ph == "round_open":
        rnd = s["rounds"][-1]
        if now() >= rnd["deadline"]:
            return True, "the round's deadline has passed. Next: `ur close`."
        if command and _sessions_done(c, rnd["workers"]):
            return True, "every worker session has ended. Next: `ur close`."
        step = f"wait for the deadline ({clock(rnd['deadline'])}) or for every worker to finish (`ur wait`), then `ur close`."
        died = _died(c, rnd["workers"]) if command else []
        if died:
            step += " These sessions died and can restart: " + ", ".join(f"`ur spawn {w}`" for w in died) + "."
        return False, step
    if ph == "round_closed":
        rnd = s["rounds"][-1]
        wid = rnd.get("construct")
        if not wid:
            return True, "Next: `ur select`."
        if now() >= rnd["construct_deadline"]:
            return True, "the combining session's time is up. Next: `ur select`."
        if command and _sessions_done(c, [wid]):
            return True, "the combining session has ended. Next: `ur select`."
        if command:
            return False, f"ur started a session in {wid} to combine the best workers. When it finishes (`ur wait`), `ur select`."
        return False, f"start one fresh session with `ur prompt {wid}` as its prompt; when it finishes, `ur select`."
    return None


def ready(c: Campaign) -> tuple[bool, str]:
    """Whether the current phase has its next step ready, and what it is."""
    s = c.load_state()
    ph = c.phase(s)
    waiting = _waiting(c, s, ph)
    if waiting is None:
        return True, f"Nothing to wait for. Next: {next_step(c, s, ph)}"
    done, why = waiting
    return done, why[:1].upper() + why[1:]


def next_step(c: Campaign, s: dict[str, Any], ph: str) -> str:
    waiting = _waiting(c, s, ph)
    if waiting is not None:
        done, why = waiting
        return why.split("Next: ", 1)[1] if done else why
    if ph == "setup":
        if config_mod.problems(c.cfg, c.project):
            return "finish .ur/config.toml and the scorer in .ur/eval/ with the user, then `ur check`."
        if (s.get("check") or {}).get("started"):
            return "fix what `ur check` reported, then `ur check` again."
        return "`ur check`."
    if ph == "setup-brief":
        return "write .ur/brief.md (`ur brief --draft` prints one built from the config)."
    if ph == "idle":
        return "`ur round`."
    if ph == "stopped":
        return f"report the result to the user. `ur export {s['base']} <dir>` copies the model out."
    return "`ur status`."


def project_status(c: Campaign, as_json: bool) -> int:
    s = c.load_state()
    ph = c.phase(s)
    store = c.store
    info: dict[str, Any] = {"project": str(c.project), "phase": ph, "round": s["round"], "next": next_step(c, s, ph)}
    lines: list[str] = []
    label = {
        "setup": "setup",
        "setup-probe": "setup, cheat probe running",
        "setup-brief": "setup, brief missing",
        "idle": "idle",
        "round_open": f"round {s['round']} open",
        "round_closed": f"round {s['round']} closed",
        "stopped": "stopped",
    }[ph]
    lines.append(f"ur · {c.project.name} · {label}")

    if ph == "setup":
        problems = config_mod.problems(c.cfg, c.project)
        info["problems"] = problems
        for p in problems:
            lines.append(f"  - {p}")

    if s.get("base"):
        base = store.load_node(s["base"])
        final = (base.get("final") or {}).get(c.metric)
        info["base"] = {"node": base["id"], "heldout": c.score(base), "final": final, "dev": c.dev_score(base)}
        line = f"Base {base['id']}: held-out {c.metric} {fmt_num(c.score(base))}"
        if final is not None:
            line += f", final {fmt_num(final)}"
        lines.append(line)

    if ph in ("round_open", "round_closed") and s["rounds"]:
        rnd = s["rounds"][-1]
        workers = []
        if ph == "round_open":
            lines.append(f"Round {rnd['n']} ends {clock(rnd['deadline'])} (in {human_duration(rnd['deadline'] - now())}).")
        for wid in rnd["workers"]:
            ws = store.load_ws(wid)
            head = store.load_node(ws["head"])
            entry = {
                "workspace": wid,
                "path": str(store.ws_path(wid)),
                "head": ws["head"],
                "runs": len(ws.get("runs", [])),
                "session": None if not ws.get("session") else execute.session_state(ws["session"]),
                "last_activity": ws.get("last_activity"),
            }
            workers.append(entry)
            committed = "no commit" if ws["head"] == ws["base"] else f"commit {head['id']}"
            session = f", session {entry['session']}" if entry["session"] else ""
            lines.append(f"  {wid:<8} {entry['runs']} run{'s' if entry['runs'] != 1 else ''}, {committed}{session}")
        info["workers"] = workers
        if ph == "round_closed" and rnd.get("construct"):
            ws = store.load_ws(rnd["construct"])
            info["construct"] = {"workspace": ws["id"], "path": str(store.ws_path(ws["id"])), "candidates": ws["candidates"]}
            lines.append(
                f"  {ws['id']:<8} combining until {clock(rnd['construct_deadline'])}, "
                f"{len(ws['candidates'])} candidate{'s' if len(ws['candidates']) != 1 else ''}"
            )

    if ph == "stopped":
        info["stopped_reason"] = s.get("stopped_reason")
        lines.append(f"Stopped: {s.get('stopped_reason')}.")
        if s.get("base"):
            artifacts = store.artifacts_dir(s["base"])
            if artifacts.exists():
                lines.append(f"Checkpoint: {artifacts}")

    if s["round"] or ph in ("idle", "stopped"):
        st = c.cfg.stop
        budget = f"{s['gpu_hours']:.2f}" + (f" of {st.max_gpu_hours:g}" if st.max_gpu_hours else "")
        stalls = f"{s['stalls']} round{'s' if s['stalls'] != 1 else ''} without a new base"
        if st.patience:
            stalls += f" (stops at {st.patience})"
        lines.append(f"{budget} GPU-hours used; {stalls}.")
        info["gpu_hours"], info["stalls"] = s["gpu_hours"], s["stalls"]

    lines.append(f"Next: {info['next']}")
    if as_json:
        print(json.dumps(info, indent=2, default=str))
    else:
        print("\n".join(lines))
    return 0


def broken_config_status(project: Path, error: UrError, as_json: bool) -> int:
    nxt = "fix .ur/config.toml, then `ur status` again."
    if as_json:
        print(json.dumps({"project": str(project), "phase": "setup", "error": str(error), "next": nxt}, indent=2))
    else:
        print(f"ur · {project.name} · setup\n  - {error}\nNext: {nxt}")
    return 0
