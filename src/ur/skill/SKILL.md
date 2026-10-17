---
name: ur
description: Initialize or continue an Ur campaign, in which fresh agent sessions repeatedly improve a model against a sealed benchmark. Use when told to use ur (set up, start, continue, resume, check on it), when working in a project that has a .ur/ directory, or when your prompt gives you a job that uses ur.
---

# Ur

If your prompt gives you a specific job, do that job. Use Commands below as a reference and skip the rest of this skill.

Otherwise, run `ur status`. It reports the campaign's phase and the next step. Act on what it says, not on anything you remember: all state lives in `.ur/` and comes back from `ur` commands, so any new session can pick up any step.

- No `.ur/brief.md`: initialize, with the user.
- Otherwise: continue.

## Initialize

1. Look before asking. Find the training code, existing evals or benchmarks, the data, the hardware, and how the environment is pinned. Run `ur init` for a config template with detected defaults, and read its comments.
2. Ask the user only what you couldn't find out. Show what you found, with a proposed default for each:
   - What should improve. Confirm the benchmark you found, or build one together.
   - What must not get worse: other benchmarks, parameter count, latency, memory.
   - How much compute, and when to stop.
   - What's off-limits: external data, pretrained weights, network access, files workers must not touch.
   - How to start fresh agent sessions here. Offer what this environment supports: subagents of this session (simplest, but the campaign stops when this session ends), a command that starts a headless session, or an orchestration tool. For a command, get the exact template. Also ask whether anything can re-run `continue` on a schedule, so the campaign outlives this session.
3. Build the splits: train, dev, held-out, final. Put held-out and final outside the project, where worker processes can't read them. Workers get a copy of the project, so check what it reveals: docs or scripts that state the answer (how the data was generated, reference solutions) go outside the project or into `workspace.exclude`.
4. Write the scorer in `.ur/eval/` and finish `.ur/config.toml`. The scorer runs in a disposable copy of a candidate's workspace and writes a JSON object of metrics. Add `[[check.expect]]` entries: a trivial predictor should score at chance, and the reference answers, where they exist, should score perfectly.
5. Run `ur check`. It runs the baseline several times (recording noise and cost per run), scores the expect entries, and checks the splits for overlap. Use the measured cost per run to set the per-run budget, round length and worker count, and confirm them with the user. Then `ur check` prepares a cheat probe: start one fresh session with `ur prompt probe` as its prompt (in command mode ur starts it), and when it finishes (`ur wait`), run `ur check --probe`. If the probe found a hole, look at what it did with `ur diff`, close the hole with the user, and run `ur check` again. If its gain is a legitimate improvement, `ur check --probe --accept`.
6. Write `.ur/brief.md` last. `ur brief --draft` prints one built from the config; edit it if needed. Facts only:
   - the dev metric, and that held-out scores are hidden
   - the per-run budget (ur adds each round's deadline to the prompt, since round length can change)
   - editable paths and what's off-limits
   - the loop: edit, `ur run`, then `ur commit` to keep or `ur reset` to revert
   - only the last commit counts, and only code, checkpoints and metrics survive the round, so notes are wasted

   No strategy, hints or suggested directions.
7. Tell the user setup is done, then continue.

## Continue

Do the step for the current phase, then run `ur status` again:

- idle: `ur round`. It copies the base into one workspace per worker and writes each workspace's prompt. In command mode ur starts the sessions itself. Otherwise start one fresh session per workspace, with `ur prompt <workspace>` as its prompt, unchanged.
- round open: wait for the deadline or for every worker to finish (`ur wait` blocks until one happens), then `ur close`. If a worker session died, start a fresh one in the same workspace (`ur spawn <workspace>` in command mode). That's just one more context reset.
- round closed: if `ur close` prepared a workspace for combining the best workers, start one fresh session with its prompt (ur starts it in command mode), and when it finishes, `ur select`. If it didn't, `ur select` right away.
- stopped: report the base's final-split score and where the model is. `ur export <node> <dir>` copies it out.

After each `ur select`, give the user one line: round, old and new base score, budget left.

If a scheduler re-runs `continue`, exit whenever the next step is waiting instead of waiting yourself.

## Commands

```
ur status [--json]         campaign phase and the next step; in a workspace, your time left and scores
ur init                    write a config template for this project
ur check                   setup checks: baseline, noise, expected scores, overlap, cheat probe
ur check --probe           score the cheat probe (--accept: its gain is legitimate)
ur brief [--draft]         print the brief, or a draft built from the config
ur round                   open a round: one workspace per worker, copied from the base
ur prompt <workspace>      print a workspace's prompt, to pass unchanged to a fresh session
ur spawn <workspace>       start a fresh session in a workspace (command mode)
ur wait [--timeout S]      block until the current phase has its next step ready
ur close                   end the round and score the workers on held-out
ur select                  score the candidates and set the new base
ur run                     train and score this workspace under the per-run budget (dev split)
ur commit [--candidate]    keep the last run; --candidate also offers it to ur select
ur reset [<node>]          put the workspace back to your last commit, or to a node
ur diff [<a>] [<b>]        code changes between nodes, or between a node and this workspace
ur apply <node>            merge a node's changes into this workspace (3-way, conflict markers)
ur log                     runs in this workspace, or the campaign's base history
ur export <node> <dir>     copy a node's code and checkpoints out
ur archive                 end this campaign and keep its records
```

Inside a workspace, `base` and `head` work wherever a node id does.

## Rules

- While continuing, this session only runs commands and starts sessions. It doesn't read diffs, logs or anything a worker wrote, and it makes no research decisions. Fresh sessions make those.
- Pass prompts exactly as ur prints them. Add nothing: diversity comes from fresh context, not instructions.
- Sessions hand each other only code, checkpoints and ur's scores. Never prose.
- Never change `brief.md`, the scorer, or the `[objective]`, `[splits]`, `[eval]` and `[run]` settings after setup; ur refuses. Changing any of them starts a new campaign: `ur archive`, then initialize again. Round length, worker count, agents and the stop rule can change between rounds.
