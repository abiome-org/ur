# Ur

Ur is a framework for hill-climbing a model with many agent sessions. It's designed for a kind of 'dark factory' approach to model development, where agents autonomously initialize research.

```
orchestrator ─► n workers ─► held-out benchmark ─► orchestrator combines the best ─► n workers ─► ...
```

Nothing written in prose passes from one worker agent to another, to contain memetics that might keep a configuration in a local minimum. Every worker agent starts in a fresh session with a copy of the current best model and a short brief.

The design follows two findings from [Fork-and-Flush](https://arxiv.org/abs/2610.07447):
- Autoresearch agents settle into "idea basins".
- Starting fresh sessions on top of the best workspace gets them out. Summarizing their history doesn't.

## Install

```sh
git clone https://github.com/abiome-org/ur && ./ur/install.sh
```

`install.sh` installs the `ur` command (with uv, pipx or pip) and runs `ur install`, which installs the skill. The skill is copied to `~/.config/agents/skills/ur` and linked from every other agent skills directory it finds under `~/.*/` and `~/.config/*/`. Ur needs Python 3.11+ and git, and has no other dependencies.

## Use (for humans)

In a project, tell your agent: *"I'd like to initialize abiome-org/ur for this project."*

## What's in `.ur/`

```
.ur/
  config.toml   objective, splits, scorer, training command, budgets, agents, devices
  brief.md      what every worker is told; frozen after the first round
  eval/         the scorer, sealed at setup
  checks/       directories that check.expect scores (trivial predictors, reference answers)
  state/        campaign state (gitignored)
```

The heavy files (code snapshots, runs, checkpoints, workspaces) live in a store outside the project, at `~/.local/share/ur/<project>-<hash>` by default. Code snapshots go in a bare git repo there, checkpoints in per-run directories. Each workspace is a plain copy with no history, so siblings can't reward-hack by peeking at each others work.

## Commands

```
ur status [--json]         campaign phase and the next step; in a workspace, your time left and scores
ur init                    write a config template for this project
ur check [--probe]         setup checks, then score the cheat probe (--accept: its gain is legitimate)
ur brief [--draft]         print the brief, or a draft built from the config
ur round                   open a round: one workspace per worker, copied from the base
ur prompt <workspace>      print a workspace's prompt, to pass unchanged to a fresh session
ur spawn <workspace>       start a fresh session in a workspace (agents.spawn = "command")
ur wait [--timeout S]      block until the current phase has its next step ready
ur close                   end the round and score the workers on held-out
ur select                  score the candidates and set the new base
ur run                     train and score this workspace under the per-run budget (dev split)
ur commit [--candidate]    keep the last run
ur reset [<node>]          put the workspace back to your last commit, or to a node
ur diff [<a>] [<b>]        code changes between nodes, or between a node and this workspace
ur apply <node>            three-way merge a node's changes into this workspace
ur log                     runs in this workspace, or the campaign's base history
ur export <node> <dir>     copy a node's code and checkpoints out
ur archive                 end this campaign and keep its records
ur install [--uninstall]   install the skill for the agents on this machine
```

## Development

```sh
uv run --with pytest pytest
```

The tests drive whole campaigns on `examples/toy`, a tiny regression problem in pure Python.
