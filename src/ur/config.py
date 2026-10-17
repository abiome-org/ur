"""Loading and checking .ur/config.toml."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import tomllib
from pathlib import Path
from typing import Any

from . import guards
from .util import UrError

DEFAULTS: dict[str, Any] = {
    "objective": {"metric": "", "direction": "max", "guards": []},
    "splits": {"train": "", "dev": "", "heldout": "", "final": ""},
    "eval": {"command": "", "timeout": 3600},
    "run": {
        "command": "",
        "budget": 300,
        "unit": "seconds",
        "timeout": 0,
        "artifacts": [],
        "init": "scratch",
    },
    "workspace": {
        "editable": ["**"],
        "exclude": [],
        "shared": [],
        "max_file_mb": 50,
        "max_total_mb": 500,
    },
    "rules": {"network": False, "external_data": False, "external_weights": False, "notes": []},
    "round": {"workers": 4, "minutes": 120, "construct_minutes": 60, "top_m": 3, "max_candidates": 3},
    "stop": {"patience": 3, "max_rounds": 0, "max_gpu_hours": 0},
    "agents": {"spawn": "subagents", "command": "", "rerun": ""},
    "exec": {"devices": [], "devices_per_run": 1, "device_env": "CUDA_VISIBLE_DEVICES", "wrap": ""},
    "check": {
        "baseline_runs": 3,
        "probe": True,
        "probe_minutes": 20,
        "probe_budget": 0,
        "overlap": "builtin",
        "overlap_max_mb": 2000,
        "max_overlap": 0,
        "expect": [],
    },
    "store": {"root": "", "keep_top": 3},
}

# Sections that define the experiment. They can't change once setup is done.
FROZEN = ("objective", "splits", "eval", "run")
SPAWN_MODES = ("subagents", "tool", "command")
INIT_MODES = ("scratch", "base", "parent")


class Section(dict):
    """A dict whose keys can also be read as attributes."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


def _merge(defaults: dict, data: Any, prefix: str = "") -> Section:
    if not isinstance(data, dict):
        raise UrError(f".ur/config.toml: [{prefix.rstrip('.')}] should be a table.")
    unknown = sorted(set(data) - set(defaults))
    if unknown:
        raise UrError(".ur/config.toml has unknown settings: " + ", ".join(prefix + k for k in unknown))
    out = Section()
    for key, default in defaults.items():
        value = data.get(key, copy.deepcopy(default))
        if isinstance(default, dict):
            value = _merge(default, value, f"{prefix}{key}.")
        out[key] = value
    return out


def load(project: Path) -> Section:
    path = project / ".ur" / "config.toml"
    try:
        data = tomllib.loads(path.read_text())
    except FileNotFoundError:
        raise UrError(f"No {path}. `ur init` writes one.") from None
    except tomllib.TOMLDecodeError as e:
        raise UrError(f"{path}: {e}") from None
    cfg = _merge(DEFAULTS, data)
    if not cfg.store.root:
        cfg.store["root"] = str(default_store_root(project))
    cfg.exec["devices"] = [str(d) for d in cfg.exec.devices]
    return cfg


def default_store_root(project: Path) -> Path:
    base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "ur"
    digest = hashlib.sha1(str(project.resolve()).encode()).hexdigest()[:8]
    return base / f"{project.name}-{digest}"


def resolve(project: Path, value: str) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else project / path


def inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def problems(cfg: Section, project: Path) -> list[str]:
    """Everything that stops `ur check` from running, as instructions."""
    errs: list[str] = []
    o = cfg.objective
    if not o.metric:
        errs.append("objective.metric is empty: name the held-out metric selection should optimize.")
    if o.direction not in ("max", "min"):
        errs.append('objective.direction must be "max" or "min".')
    for g in o.guards:
        try:
            guards.parse(g)
        except UrError as e:
            errs.append(f"objective.guards: {e}")

    s = cfg.splits
    for name in ("dev", "heldout"):
        if not s[name]:
            errs.append(f"splits.{name} is empty.")
    for name in ("train", "dev", "heldout", "final"):
        path = resolve(project, s[name])
        if path is None:
            continue
        if not path.exists():
            errs.append(f"splits.{name}: {path} doesn't exist.")
        elif name in ("heldout", "final") and inside(path, project):
            errs.append(f"splits.{name} is inside the project, where workers can read it. Move it outside {project}.")

    if not cfg.eval.command:
        errs.append("eval.command is empty: give the command that scores a workspace.")
    if not _number(cfg.eval.timeout) or cfg.eval.timeout <= 0:
        errs.append("eval.timeout must be a positive number of seconds.")

    r = cfg.run
    if not r.command:
        errs.append("run.command is empty: give the command that trains in a workspace.")
    if not _number(r.budget) or r.budget <= 0:
        errs.append("run.budget must be a positive number.")
    if r.unit != "seconds" and not r.timeout:
        errs.append("run.timeout is required when run.unit isn't seconds, since ur can only enforce wall-clock limits.")
    if not _number(r.timeout) or r.timeout < 0:
        errs.append("run.timeout must be 0 or a positive number of seconds.")
    if r.init not in INIT_MODES:
        errs.append('run.init must be "scratch", "base" or "parent".')
    for a in r.artifacts:
        if Path(a).is_absolute() or ".." in Path(a).parts:
            errs.append(f"run.artifacts: {a!r} must be a path inside the workspace.")
    if r.init != "scratch" and not r.artifacts:
        errs.append("run.init needs run.artifacts: the checkpoint a run starts from.")

    w = cfg.workspace
    if not w.editable:
        errs.append('workspace.editable is empty, so workers could change nothing. Use ["**"] for everything.')
    for rel in w.shared:
        if not (project / rel).exists():
            errs.append(f"workspace.shared: {project / rel} doesn't exist.")

    rd = cfg.round
    if not isinstance(rd.workers, int) or rd.workers < 1:
        errs.append("round.workers must be at least 1.")
    for key in ("minutes", "construct_minutes"):
        if not _number(rd[key]) or rd[key] <= 0:
            errs.append(f"round.{key} must be a positive number.")
    if not isinstance(rd.top_m, int) or rd.top_m < 2:
        errs.append("round.top_m must be at least 2.")
    if not isinstance(rd.max_candidates, int) or rd.max_candidates < 1:
        errs.append("round.max_candidates must be at least 1.")

    a = cfg.agents
    if a.spawn not in SPAWN_MODES:
        errs.append('agents.spawn must be "subagents", "tool" or "command".')
    if a.spawn == "command" and not ("{prompt_file}" in a.command or "{prompt}" in a.command):
        errs.append("agents.command must start one fresh session and use {prompt_file} or {prompt}.")

    e = cfg.exec
    if not isinstance(e.devices_per_run, int) or e.devices_per_run < 1:
        errs.append("exec.devices_per_run must be at least 1.")
    elif e.devices and len(e.devices) < e.devices_per_run:
        errs.append("exec.devices has fewer devices than exec.devices_per_run.")

    c = cfg.check
    if not isinstance(c.baseline_runs, int) or c.baseline_runs < 1:
        errs.append("check.baseline_runs must be at least 1.")
    for i, entry in enumerate(c.expect):
        label = f"check.expect[{i}]"
        if not isinstance(entry, dict):
            errs.append(f"{label} should be a table.")
            continue
        unknown = set(entry) - {"name", "workspace", "split", "metric", "min", "max"}
        if unknown:
            errs.append(f"{label} has unknown keys: {', '.join(sorted(unknown))}.")
        if not entry.get("workspace") or not entry.get("metric"):
            errs.append(f"{label} needs a workspace and a metric.")
        elif not resolve(project, entry["workspace"]).is_dir():
            errs.append(f"{label}: {resolve(project, entry['workspace'])} isn't a directory.")
        if "min" not in entry and "max" not in entry:
            errs.append(f"{label} needs min, max, or both.")
        if entry.get("split", "heldout") not in ("dev", "heldout", "final"):
            errs.append(f"{label}: split must be dev, heldout or final.")
    return errs


def warnings(cfg: Section, project: Path) -> list[str]:
    out = []
    if not cfg.splits.final:
        out.append("splits.final is empty, so nothing will show when selection starts to overfit the held-out split.")
    if cfg.check.baseline_runs < 2:
        out.append("check.baseline_runs is 1, so ur can't measure noise or break near-ties.")
    if not cfg.exec.wrap:
        out.append(
            "exec.wrap is empty, so worker processes can read anything this user can, held-out data included. "
            "A sandbox wrapper closes that."
        )
    return out


def frozen_hash(cfg: Section) -> str:
    data = {key: cfg[key] for key in FROZEN}
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


def detect_devices() -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if out.returncode != 0:
        return []
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def gitignore_patterns(project: Path) -> list[str]:
    patterns = []
    path = project / ".gitignore"
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith(("#", "!")):
                patterns.append(line)
    for default in ("__pycache__/", "*.pyc"):
        if default not in patterns:
            patterns.append(default)
    return patterns


TEMPLATE = """\
# Ur campaign config for {name}.
# `ur init` wrote this. Finish it with the user, then run `ur check`.
# Relative paths are relative to the project.
# Once setup is done, [objective], [splits], [eval] and [run] are frozen:
# changing them starts a new campaign (`ur archive`).

[objective]
metric = ""             # the held-out metric selection optimizes, as the scorer names it
direction = "max"       # "max" or "min"
guards = []             # hard limits on held-out results, e.g. ["params <= 1.5e9", "latency_ms <= base * 1.05"]

[splits]
train = ""              # optional; only read by the overlap check
dev = ""                # workers see their scores on this split
heldout = ""            # outside the project. Only ur scores it, and workers never see those scores
final = ""              # outside the project. Never used for selection: it reports, and shows overfitting

[eval]
# The scorer. It runs in a disposable copy of a candidate's workspace (the current
# directory), checkpoints included, and must write a JSON object of numbers to {{out}}.
# Placeholders: {{eval_dir}} (.ur/eval) {{workspace}} {{data}} (the split's path) {{split}} {{out}}
command = "python3 {{eval_dir}}/score.py --workspace {{workspace}} --data {{data}} --out {{out}}"
timeout = 3600          # seconds

[run]
# Training. It runs in the workspace and must leave its outputs under `artifacts`.
# Placeholders: {{budget}} {{init}} (a checkpoint directory to start from, or empty) {{workspace}}
command = ""
budget = 300            # per run, in `unit`
unit = "seconds"        # ur enforces seconds itself. For any other unit the command must, and `timeout` is required
timeout = 0             # hard wall-clock limit per run in seconds; 0 means budget + 10% (at least 30s)
artifacts = []          # output paths, e.g. ["ckpt"]. Moved into ur's store after each run
init = "scratch"        # what {{init}} is: "scratch" (empty), "base" (the round's base checkpoint) or "parent" (the workspace's last commit)

[workspace]
editable = ["**"]       # globs workers may change. ur refuses runs that touch anything else
exclude = {exclude}  # left out of code snapshots, like .gitignore
shared = []             # read-only paths linked into each workspace instead of copied, e.g. ["data"]
max_file_mb = 50        # a bigger file in a snapshot is an error
max_total_mb = 500      # so is a bigger snapshot

[rules]                 # written into the brief. Enforce them with exec.wrap
network = false
external_data = false
external_weights = false
notes = []              # any other rules, one line each

[round]
workers = {workers}
minutes = 120           # how long workers have
construct_minutes = 60  # how long the session that combines the best workers has
top_m = 3               # how many of the best workers it gets
max_candidates = 3      # how many combinations it may submit

[stop]
patience = 3            # stop after this many rounds without a new base
max_rounds = 0          # 0 means no limit
max_gpu_hours = 0       # run time x devices per run; 0 means no limit

[agents]
spawn = "subagents"     # "subagents", "tool" (an orchestration tool) or "command" (ur starts sessions itself)
command = ""            # for "command": starts one fresh headless session. Placeholders: {{workspace}} {{prompt_file}} {{prompt}}
rerun = ""              # what re-runs `continue` on a schedule, if anything

[exec]
devices = {devices}       # accelerators to share out, e.g. ["0", "1"]
devices_per_run = 1
device_env = "CUDA_VISIBLE_DEVICES"
wrap = ""               # wraps every training and scoring command, e.g. a sandbox. {{cmd}} is the command, {{qcmd}} the command quoted

[check]
baseline_runs = 3       # baseline repeats, to measure noise and cost per run
probe = true            # run a cheat probe before setup is done
probe_minutes = 20
probe_budget = 0        # training budget per probe run; 0 means 2% of run.budget
overlap = "builtin"     # "builtin", "off", or a command (placeholders {{train}} {{dev}} {{heldout}} {{final}}; exit 0 means no overlap)
overlap_max_mb = 2000   # how much of train and dev the built-in check reads
max_overlap = 0         # records train or dev may share with heldout or final

# Expected scores, to catch a broken scorer. Each workspace is a directory scored as is.
# [[check.expect]]
# name = "constant predictor"
# workspace = ".ur/checks/constant"
# split = "heldout"
# metric = "accuracy"
# max = 0.12

[store]
root = ""               # runs, checkpoints and workspaces. Empty means {store}
keep_top = 3            # checkpoints kept per round besides the base's
"""


def template(project: Path) -> str:
    devices = detect_devices()
    return TEMPLATE.format(
        name=project.name,
        exclude=json.dumps(gitignore_patterns(project)),
        workers=max(4, len(devices)),
        devices=json.dumps(devices),
        store=default_store_root(project),
    )
