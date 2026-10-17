# Ur

Ur is a command-line tool and an agent skill for hill-climbing a model with many agent sessions. You don't run it by hand. You tell your agent "initialize ur for this project". The agent asks a few questions, sets up a sealed benchmark, and then keeps a loop running:

```
orchestrator ─► n workers ─► held-out benchmark ─► orchestrator combines the best ─► n workers ─► ...
```

The loop's one rule is that nothing written in prose passes from one agent to another. Every worker starts in a fresh session with a copy of the current best model and a short factual brief. What carries over between rounds is code, checkpoints and scores that ur measured itself. Agents' notes, summaries and reasoning stay behind. When an agent hands another a document, its untested assumptions get treated as facts downstream. Ur leaves no channel for that: a change is kept only if it measurably helps on data the workers never see.

The design follows two findings from [Fork-and-Flush](https://arxiv.org/abs/2610.07447) (Cai et al., 2026):
- Autoresearch agents settle into "idea basins".
- Starting fresh sessions on top of the best workspace gets them out. Summarizing their history doesn't.

## Install

```sh
git clone https://github.com/abiome-org/ur && ./ur/install.sh
```

`install.sh` installs the `ur` command (with uv, pipx or pip) and runs `ur install`, which installs the skill. The skill is copied to `~/.config/agents/skills/ur` and linked from every other agent skills directory it finds under `~/.*/` and `~/.config/*/`. Ur needs Python 3.11+ and git, and has no other dependencies.

## Use

In a project, tell your agent: *"I'd like to initialize ur for this project."* The skill takes it from there:

1. The agent reads the repo: training code, existing evals, data and hardware. `ur init` writes a config template with detected defaults.
2. It asks only what it couldn't find out:
   - what should improve
   - what must not get worse
   - how much compute to spend, and when to stop
   - what's off-limits
   - how this environment starts fresh agent sessions
3. It builds the splits (train, dev, held-out and final, with held-out and final outside the project), writes the scorer into `.ur/eval/`, and runs `ur check`. The checks:
   - The baseline trains a few times, to measure noise and cost per run.
   - Trivial predictors must score at chance.
   - Splits must not overlap.
   - A cheat probe, a session told to raise the score without training a better model, must fail.
4. Once the checks pass, it writes `.ur/brief.md`: facts only, no strategy.
5. Then it runs rounds until the stop rule says to stop.

Later, in a project that has `.ur/`, *"continue"* picks up wherever the campaign is. All state comes from `ur status`, so any new session can resume any step.

## What's in `.ur/`

```
.ur/
  config.toml   objective, splits, scorer, training command, budgets, agents, devices
  brief.md      what every worker is told; frozen after the first round
  eval/         the scorer, sealed at setup
  checks/       directories that check.expect scores (trivial predictors, reference answers)
  state/        campaign state (gitignored)
```

The heavy files (code snapshots, runs, checkpoints, workspaces) live in a store outside the project, at `~/.local/share/ur/<project>-<hash>` by default. Code snapshots go in a bare git repo there, checkpoints in per-run directories. Each workspace is a plain copy with no history, so siblings can't see each other's work.

## Contracts

**Training.** `run.command` runs inside a workspace with `{budget}` (and `{init}`, a checkpoint directory, when `run.init` warm-starts). Its outputs go under `run.artifacts`. ur moves them into the store after each run. With `run.unit = "seconds"` ur enforces the budget itself.

**Scoring.** `eval.command` runs in a disposable copy of a candidate's workspace, with its checkpoints linked in. It gets `{eval_dir}`, `{workspace}`, `{data}`, `{split}` and `{out}`, and must write a JSON object of numbers to `{out}`. Workers see their scores on dev. Only ur scores held-out, and those scores never reach a worker. The final split is never used for selection: it reports the result, and shows when selection has started to overfit held-out.

**Selection.** At the end of a round, ur scores every worker's last commit on held-out. If two or more beat the base, one fresh session gets their diffs and builds combinations. The best eligible candidate becomes the next base:
- Candidates are the base, every worker's last commit, and each combination.
- A candidate that breaks a guard isn't eligible.
- Near-ties (within twice the baseline noise) are trained once more and averaged before ur picks.

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

## What ur enforces, and what it can't

ur enforces these:
- **Sealed scorer and settings.** The scorer, the brief, and the `[objective]`, `[splits]`, `[eval]` and `[run]` settings are sealed. Change any of them and ur refuses to continue until you start a new campaign.
- **Editable paths.** A run that touches anything outside `workspace.editable` is refused.
- **Scores tied to exact code.** A commit has to match the exact code of the run it keeps, so every score belongs to a specific snapshot.
- **Clean scoring.** Scoring always happens in a throwaway copy.
- **Hidden held-out scores.** No command run from a workspace shows them.
- **Isolation by default.** Held-out data stays outside the project, and workspaces don't contain `.ur/`.

ur can't keep a determined process from reading files this user can read. Without a sandbox, a worker that goes looking can find the held-out data. `exec.wrap` puts every training and scoring command inside a wrapper you choose, such as a container with no network that can only see the workspace. Use one for anything that matters. The cheat probe exists to find these holes before a campaign starts.

## Development

```sh
uv run --with pytest pytest
```

The tests drive whole campaigns on `examples/toy`, a tiny regression problem in pure Python. Scripted sessions stand in for agents.
