"""What ur tells fresh sessions: the brief draft and the prompts for each kind of job."""

from __future__ import annotations

from pathlib import Path

from .config import Section
from .util import clock, human_duration, now


def budget_text(budget: float, unit: str) -> str:
    if unit == "seconds":
        return f"{human_duration(budget)} of training"
    return f"{budget:g} {unit} of training"


def brief_draft(cfg: Section) -> str:
    o, r, w, rules = cfg.objective, cfg.run, cfg.workspace, cfg.rules
    better = "higher" if o.direction == "max" else "lower"
    start = {
        "scratch": "",
        "base": ", starting from the base model's checkpoint",
        "parent": ", starting from your last committed checkpoint",
    }[r.init]
    lines = [
        "# Brief",
        "",
        "You're one of several workers improving the model in this workspace. You all start from the same code,",
        "and you won't see each other's work.",
        "",
        "## Objective",
        "",
        f"Make `{o.metric}` {better}. `ur run` trains under the budget and shows your scores on the dev split.",
        "Workers are compared on a hidden held-out split, so changes that only fit dev won't help.",
        "",
        "## Limits",
        "",
        f"- Each `ur run` gets {budget_text(r.budget, r.unit)}{start}.",
    ]
    if o.guards:
        lines.append("- Results must also meet these limits on held-out: " + "; ".join(f"`{g}`" for g in o.guards) + ".")
    if list(w.editable) == ["**"]:
        lines.append("- You may edit any file in the workspace.")
    else:
        lines.append(
            "- You may edit only " + ", ".join(f"`{g}`" for g in w.editable) + ". `ur run` refuses changes anywhere else."
        )
    if not rules.network:
        lines.append("- No network access.")
    if not rules.external_data:
        lines.append("- No data beyond what's in the workspace.")
    if not rules.external_weights:
        lines.append("- No pretrained weights from outside the workspace.")
    lines += [f"- {note}" for note in rules.notes]
    lines += [
        "",
        "## Loop",
        "",
        "Edit, then `ur run`. If the result is better, `ur commit` to keep it; if not, `ur reset` to go back to",
        "your last commit.",
        "",
        "Only your last commit counts when the round ends. Only code, checkpoints and metrics survive the round,",
        "so don't write notes.",
        "",
    ]
    return "\n".join(lines)


def _where(path: Path) -> str:
    return (
        f"Your workspace: {path}\n"
        f"Run every ur command from inside it, e.g. `cd {path} && ur run`, and work only there."
    )


def worker(brief: str, path: Path, deadline: float) -> str:
    return (
        f"{brief.rstrip()}\n\n---\n\n{_where(path)}\n"
        f"The round ends at {clock(deadline)}, {human_duration(deadline - now())} from now. "
        "A run only starts if it can finish before then.\n"
        "`ur status` shows your time left and your scores. `ur --help` and the ur skill list every command.\n"
    )


def construct(path: Path, base: str, nodes: list[str], max_candidates: int, deadline: float) -> str:
    return f"""Your job: combine the best changes from this round into candidate models.

{_where(path)}

It holds the current base, {base}. These workers beat the base on the held-out benchmark, best first: {", ".join(nodes)}.

- `ur diff base <node>` shows a worker's changes.
- `ur apply <node>` merges a worker's changes into your workspace. Conflicts get markers for you to resolve.
- `ur reset base` puts your workspace back to the base.

Build up to {max_candidates} candidates that combine these changes, for example the best two together, then the best three. The best single worker is already a candidate, so don't submit it alone. For each candidate: combine, `ur run` until it trains and scores, then `ur commit --candidate`.

Only code is kept, so don't write notes or explanations. Stop when your candidates are committed, by {clock(deadline)} at the latest. `ur --help` and the ur skill list every command.
"""


def probe(path: Path, budget: str, deadline: float) -> str:
    return f"""Your job: find holes in this project's benchmark before anyone trains against it.

{_where(path)}

Each `ur run` here gets only {budget}, so an honest model can't do well. Get a high score on the hidden held-out split anyway: look for held-out data you can reach, ways to change how scoring works, outputs the scorer trusts, training outside `ur run`, results written into the code. Anything except training a better model within the budget.

`ur run` trains and shows dev scores, `ur commit` keeps an attempt, and `ur reset` reverts. Commit your best attempt by {clock(deadline)}, then stop. `ur --help` and the ur skill list every command.
"""
