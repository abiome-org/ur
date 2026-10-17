"""The `ur` command."""

from __future__ import annotations

import argparse
import os
import signal
import sys
from pathlib import Path

from . import __version__, loop, status, workspace
from .campaign import Campaign, find_project, find_workspace
from .install import install_cmd
from .util import UrError


def _project() -> Campaign:
    root = find_project(Path.cwd())
    if root is None:
        if find_workspace(Path.cwd()):
            raise UrError("This is a workspace. Campaign commands run from the project.")
        raise UrError(f"{status.NO_CAMPAIGN} `ur init` sets one up.")
    return Campaign(root)


def _workspace() -> tuple[Campaign, str]:
    found = find_workspace(Path.cwd())
    if not found:
        raise UrError("This isn't a ur workspace. `ur status` in the project lists them.")
    _, marker = found
    return Campaign.from_store(Path(marker["root"])), marker["id"]


def _in_workspace() -> bool:
    return find_workspace(Path.cwd()) is not None


def cmd_status(args: argparse.Namespace) -> int:
    if _in_workspace():
        c, wid = _workspace()
        return workspace.status_cmd(c, wid, args.json)
    root = find_project(Path.cwd())
    if root is None:
        return status.none_status(args.json)
    try:
        c = Campaign(root)
    except UrError as e:
        return status.broken_config_status(root, e, args.json)
    return status.project_status(c, args.json)


def cmd_init(args: argparse.Namespace) -> int:
    if _in_workspace():
        raise UrError("This is a workspace, not a project.")
    return loop.init_cmd(Path.cwd().resolve())


def cmd_check(args: argparse.Namespace) -> int:
    c = _project()
    if args.accept and not args.probe:
        raise UrError("--accept goes with --probe.")
    return loop.check_probe_cmd(c, args.accept) if args.probe else loop.check_cmd(c)


def cmd_run(args: argparse.Namespace) -> int:
    # Let a SIGTERM (from `ur close`) unwind, so the training process group gets stopped too.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    c, wid = _workspace()
    return workspace.run_cmd(c, wid)


def cmd_diff(args: argparse.Namespace) -> int:
    if _in_workspace():
        c, wid = _workspace()
        return workspace.diff_cmd(c, wid, args.a, args.b, args.stat)
    if not args.a:
        raise UrError("From the project, give one node (diffed against its parent) or two.")
    return loop.diff_project_cmd(_project(), args.a, args.b, args.stat)


def cmd_log(args: argparse.Namespace) -> int:
    if _in_workspace():
        c, wid = _workspace()
        return workspace.log_cmd(c, wid)
    return loop.log_project_cmd(_project())


COMMANDS = {
    "status": cmd_status,
    "init": cmd_init,
    "check": cmd_check,
    "brief": lambda a: loop.brief_cmd(_project(), a.draft),
    "round": lambda a: loop.round_cmd(_project()),
    "prompt": lambda a: loop.prompt_cmd(_project(), a.workspace),
    "spawn": lambda a: loop.spawn_cmd(_project(), a.workspace),
    "wait": lambda a: loop.wait_cmd(_project(), a.timeout),
    "close": lambda a: loop.close_cmd(_project()),
    "select": lambda a: loop.select_cmd(_project()),
    "run": cmd_run,
    "commit": lambda a: workspace.commit_cmd(*_workspace(), a.candidate),
    "reset": lambda a: workspace.reset_cmd(*_workspace(), a.node),
    "diff": cmd_diff,
    "apply": lambda a: workspace.apply_cmd(*_workspace(), a.node),
    "log": cmd_log,
    "export": lambda a: loop.export_cmd(_project(), a.node, Path(a.dir).resolve()),
    "archive": lambda a: loop.archive_cmd(_project()),
    "install": lambda a: install_cmd(a.dir, a.dry_run, a.uninstall),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ur",
        description="Hill-climb a model with fresh agent sessions against a sealed benchmark.",
        epilog="Start with `ur status`: it says what phase the campaign is in and what to do next.",
    )
    parser.add_argument("-C", dest="chdir", metavar="DIR", help="run as if ur was started in DIR")
    parser.add_argument("--version", action="version", version=f"ur {__version__}")
    sub = parser.add_subparsers(dest="cmd", metavar="<command>")

    p = sub.add_parser("status", help="campaign phase and the next step; in a workspace, your time left and scores")
    p.add_argument("--json", action="store_true")

    sub.add_parser("init", help="write a config template for this project")

    p = sub.add_parser("check", help="setup checks: baseline, noise, expected scores, overlap, cheat probe")
    p.add_argument("--probe", action="store_true", help="score the cheat probe")
    p.add_argument("--accept", action="store_true", help="with --probe: the probe's gain is legitimate")

    p = sub.add_parser("brief", help="print the brief")
    p.add_argument("--draft", action="store_true", help="print a brief built from the config")

    sub.add_parser("round", help="open a round: one workspace per worker, copied from the base")
    p = sub.add_parser("prompt", help="print a workspace's prompt, to pass unchanged to a fresh session")
    p.add_argument("workspace")
    p = sub.add_parser("spawn", help="start a fresh session in a workspace (agents.spawn = command)")
    p.add_argument("workspace")
    p = sub.add_parser("wait", help="block until the current phase has its next step ready")
    p.add_argument("--timeout", type=float, metavar="SECONDS", help="give up after this long (exit 1)")
    sub.add_parser("close", help="end the round and score the workers on held-out")
    sub.add_parser("select", help="score the candidates and set the new base")

    sub.add_parser("run", help="train and score this workspace under the per-run budget (dev split)")
    p = sub.add_parser("commit", help="keep the last run")
    p.add_argument("--candidate", action="store_true", help="also offer it to `ur select`")
    p = sub.add_parser("reset", help="put the workspace back to your last commit, or to a node")
    p.add_argument("node", nargs="?", help="a node id, `base` or `head` (default)")
    p = sub.add_parser("diff", help="code changes between nodes, or between a node and this workspace")
    p.add_argument("a", nargs="?", help="a node id, `base` or `head` (default)")
    p.add_argument("b", nargs="?", help="a node id (default: the workspace as it is)")
    p.add_argument("--stat", action="store_true", help="only list the files")
    p = sub.add_parser("apply", help="merge a node's changes into this workspace (3-way, conflict markers)")
    p.add_argument("node")
    sub.add_parser("log", help="runs in this workspace, or the campaign's base history")

    p = sub.add_parser("export", help="copy a node's code and checkpoints out")
    p.add_argument("node", help="a node id or `base`")
    p.add_argument("dir")
    sub.add_parser("archive", help="end this campaign and keep its records")

    p = sub.add_parser("install", help="install the ur skill for the agents on this machine")
    p.add_argument("--dir", action="append", default=[], help="a skills directory to link into (repeatable)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--uninstall", action="store_true")
    return parser


WORKSPACE_ONLY = {"run", "commit", "reset", "apply"}
EITHER = {"status", "diff", "log"}


def _isolate(cmd: str, started_in: tuple[Path, dict] | None) -> None:
    """Keep workspace sessions to their own workspace's commands."""
    if cmd == "install":
        return
    session = os.environ.get("UR_SESSION")
    here = find_workspace(Path.cwd())
    if cmd not in WORKSPACE_ONLY and not (cmd in EITHER and here):
        if session or started_in:
            raise UrError("Campaign commands aren't available from a workspace. Use them from the project.")
    if session and here and here[1].get("id") != session:
        raise UrError(f"This session works in workspace {session}, not {here[1].get('id')}.")
    if started_in and here and here[1].get("id") != started_in[1].get("id"):
        raise UrError("ur -C can't move from one workspace to another.")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        try:
            started_in = find_workspace(Path.cwd())
        except FileNotFoundError:
            raise UrError("The current directory no longer exists; its round may have ended.") from None
        if args.chdir:
            os.chdir(args.chdir)
        if not args.cmd:
            parser.print_help()
            return 0
        _isolate(args.cmd, started_in)
        return COMMANDS[args.cmd](args) or 0
    except UrError as e:
        print(f"ur: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    except FileNotFoundError:
        if args.chdir and not Path(args.chdir).is_dir():
            print(f"ur: {args.chdir} isn't a directory.", file=sys.stderr)
            return 1
        raise
