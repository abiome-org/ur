"""The loop driven by hand, as an orchestrator using subagents would: ur starts no sessions."""

import json
import sys
from pathlib import Path

SCORER = """
import json, sys
from pathlib import Path
ws, out = Path(sys.argv[1]), sys.argv[2]
ckpt = ws / "ckpt"
n = int((ckpt / "id").read_text().strip()[1:])
score = float((ckpt / "score.txt").read_text()) + (n % 2) * 0.01  # deterministic 'noise'
json.dump({"score": score, "params": int((ckpt / "lines").read_text())}, open(out, "w"))
"""

CONFIG = """
[objective]
metric = "score"
direction = "max"
guards = ["params <= 20"]

[splits]
dev = "dev.txt"
heldout = "{private}/heldout.txt"
final = "{private}/final.txt"

[eval]
command = "{python} {{eval_dir}}/score.py {{workspace}} {{out}}"

[run]
command = "mkdir -p ckpt && cp score.txt ckpt/score.txt && wc -l < model.txt > ckpt/lines && echo $UR_RUN_ID > ckpt/id"
budget = 5
artifacts = ["ckpt"]

[workspace]
editable = ["model.txt", "score.txt", "old.txt"]

[round]
workers = 3
minutes = 10

[check]
probe = false
overlap = "off"

[store]
root = "{store}"
"""


def make_project(tmp: Path) -> Path:
    project, private = tmp / "proj", tmp / "private"
    project.mkdir()
    private.mkdir()
    (project / "model.txt").write_text("".join(f"line {i}\n" for i in range(1, 9)))
    (project / "score.txt").write_text("1\n")
    (project / "old.txt").write_text("old\n")
    (project / "dev.txt").write_text("dev\n")
    (private / "heldout.txt").write_text("heldout\n")
    (private / "final.txt").write_text("final\n")
    (project / ".ur" / "eval").mkdir(parents=True)
    (project / ".ur" / "eval" / "score.py").write_text(SCORER)
    (project / ".ur" / "config.toml").write_text(
        CONFIG.format(private=private, python=sys.executable, store=tmp / "store")
    )
    return project


def ws_meta(tmp: Path, wid: str) -> dict:
    return json.loads((tmp / "store" / "meta" / f"{wid}.json").read_text())


def test_hand_driven_round(tmp_path, ur):
    project = make_project(tmp_path)
    store = tmp_path / "store" / "ws"
    out = ur("check", cwd=project).stdout
    assert "Setup checks passed" in out
    (project / ".ur" / "brief.md").write_text(ur("brief", "--draft", cwd=project).stdout)

    out = ur("round", cwd=project).stdout
    assert "Start one fresh session per workspace" in out
    assert str(store / "r1-w1") in ur("prompt", "r1-w1", cwd=project).stdout

    w1, w2, w3 = (store / f"r1-w{i}" for i in (1, 2, 3))
    (w1 / "model.txt").write_text((w1 / "model.txt").read_text().replace("line 1\n", "LINE ONE\n"))
    (w1 / "score.txt").write_text("2\n")
    ur("run", cwd=w1)
    ur("commit", cwd=w1)

    (w2 / "model.txt").write_text((w2 / "model.txt").read_text().replace("line 7\n", "LINE SEVEN\n"))
    (w2 / "old.txt").unlink()
    (w2 / "score.txt").write_text("3\n")
    ur("run", cwd=w2)
    proc = ur("commit", cwd=w2)
    w2_node = ws_meta(tmp_path, "r1-w2")["head"]

    # Siblings can't see each other.
    proc = ur("diff", w2_node, cwd=w1, check=False)
    assert proc.returncode == 1 and "isn't visible" in proc.stderr

    # A big win that breaks the params guard.
    (w3 / "model.txt").write_text((w3 / "model.txt").read_text() + "".join(f"pad {i}\n" for i in range(30)))
    (w3 / "score.txt").write_text("10\n")
    ur("run", cwd=w3)
    ur("commit", cwd=w3)
    status = json.loads(ur("status", "--json", cwd=w3).stdout)
    assert status["head"]["node"] == ws_meta(tmp_path, "r1-w3")["head"]

    out = ur("close", cwd=project).stdout
    assert out.count("beats the base") == 2
    assert "breaks a guard: params <= 20" in out
    assert "Start one fresh session with `ur prompt r1-c` as its prompt" in out
    prompt = ur("prompt", "r1-c", cwd=project).stdout
    order = prompt.split("best first: ")[1].split(".\n")[0].split(", ")
    assert order == [w2_node, ws_meta(tmp_path, "r1-w1")["head"]]  # the guard-breaker isn't offered

    c = store / "r1-c"
    out = ur("apply", w2_node, cwd=c).stdout
    assert "merged cleanly" in out
    out = ur("apply", ws_meta(tmp_path, "r1-w1")["head"], cwd=c).stdout
    assert "score.txt: 1 conflict" in out
    model = (c / "model.txt").read_text()
    assert "LINE ONE" in model and "LINE SEVEN" in model and "<<<<<<<" not in model
    assert not (c / "old.txt").exists()
    (c / "score.txt").write_text("3\n")
    ur("run", cwd=c)
    ur("commit", "--candidate", cwd=c)

    out = ur("select", cwd=project).stdout
    assert "broke a guard: params <= 20" in out
    assert "within noise; training both once more" in out
    s = json.loads((project / ".ur" / "state" / "campaign.json").read_text())
    rnd = s["rounds"][0]
    assert len(rnd["rescore"]) == 2
    assert rnd["selected"] == w2_node  # its second run scored higher than the combination's
    assert json.loads(ur("status", "--json", cwd=project).stdout)["phase"] == "idle"

    # Stored checkpoints are read-only, even through a workspace's links.
    ckpt = tmp_path / "store" / "runs" / rnd["selected"] / "artifacts" / "ckpt" / "score.txt"
    assert not ckpt.stat().st_mode & 0o222

    # Rounds where no worker runs anything don't count as stalls; two in a row stop the campaign.
    for n in (2, 3):
        ur("round", cwd=project)
        ur("close", cwd=project)
        out = ur("select", cwd=project).stdout
        assert "No worker ran anything this round" in out
    s = json.loads((project / ".ur" / "state" / "campaign.json").read_text())
    assert s["stalls"] == 0 and s["phase"] == "stopped"
    assert "no worker ran anything in 2 rounds" in s["stopped_reason"]

    out = ur("archive", cwd=project).stdout
    assert "Archived the campaign" in out
    st = json.loads(ur("status", "--json", cwd=project).stdout)
    assert st["phase"] == "setup" and st["next"] == "`ur check`."
