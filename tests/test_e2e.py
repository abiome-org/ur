"""The whole loop on the toy project, with scripted sessions standing in for agents."""

import json
from pathlib import Path

from conftest import make_toy


def state(project: Path) -> dict:
    return json.loads((project / ".ur" / "state" / "campaign.json").read_text())


def status(ur, project: Path) -> dict:
    return json.loads(ur("status", "--json", cwd=project).stdout)


def test_full_campaign(tmp_path, ur):
    toy = make_toy(tmp_path, ur)
    project = toy["project"]

    assert status(ur, project)["next"] == "`ur check`."
    out = ur("check", cwd=project).stdout
    assert "Baseline: held-out mse" in out
    assert "Expect mean predictor" in out and "FAILED" not in out
    assert "Overlap train/heldout: 0 shared records" in out
    assert status(ur, project)["phase"] == "setup-probe"

    ur("wait", "--timeout", "120", cwd=project)
    out = ur("check", "--probe", cwd=project).stdout
    assert "No hole found" in out, out
    assert status(ur, project)["phase"] == "setup-brief"

    brief = ur("brief", "--draft", cwd=project).stdout
    assert "Make `mse` lower" in brief
    (project / ".ur" / "brief.md").write_text(brief)
    assert status(ur, project)["phase"] == "idle"
    base0 = state(project)["base"]

    # Round 1: w1 adds x^2, w2 adds sin(4x); the combining session merges both.
    out = ur("round", cwd=project).stdout
    assert "ur started 2 sessions" in out
    prompt = ur("prompt", "r1-w1", cwd=project).stdout
    assert prompt.startswith("# Brief") and "Your workspace:" in prompt
    ur("wait", "--timeout", "180", cwd=project)
    out = ur("close", cwd=project).stdout
    assert out.count("beats the base") == 2, out
    assert "r1-c combines them" in out
    ur("wait", "--timeout", "180", cwd=project)
    out = ur("select", cwd=project).stdout
    s = state(project)
    rnd = s["rounds"][0]
    assert rnd["improved"], out
    assert rnd["selected"] in rnd["candidates"], "the combination should beat either worker alone"
    workers_best = min(item["score"] for item in rnd["ranking"])
    assert rnd["score"] < workers_best
    assert s["base"] != base0

    # The workers' workspaces are gone, but every node is still on record.
    store = toy["store"]
    assert not (store / "ws" / "r1-w1").exists()
    assert (store / "runs" / rnd["selected"] / "artifacts" / "ckpt" / "weights.json").exists()

    # Round 2: nobody improves anything, so with patience 1 the campaign stops.
    ur("round", cwd=project)
    ur("wait", "--timeout", "180", cwd=project)
    out = ur("close", cwd=project).stdout
    assert "Next: `ur select`." in out
    out = ur("select", cwd=project).stdout
    assert "Stopped: no new base in 1 round" in out
    st = status(ur, project)
    assert st["phase"] == "stopped"
    assert st["base"]["final"] is not None

    exported = tmp_path / "exported"
    ur("export", "base", str(exported), cwd=project)
    assert "math.sin" in (exported / "model.py").read_text()
    assert (exported / "ckpt" / "weights.json").exists()
    log = ur("log", cwd=project).stdout
    assert "round 1" in log and "round 2" in log

    # Raising patience resumes the campaign.
    config = project / ".ur" / "config.toml"
    config.write_text(config.read_text().replace("patience = 1", "patience = 5"))
    assert status(ur, project)["phase"] == "idle"

    # The scorer is sealed.
    score = project / ".ur" / "eval" / "score.py"
    score.write_text(score.read_text() + "\n# changed\n")
    proc = ur("round", cwd=project, check=False)
    assert proc.returncode == 1 and "scorer in .ur/eval changed" in proc.stderr


def test_cheat_probe_is_caught(tmp_path, ur):
    toy = make_toy(tmp_path, ur, extra_env={"FAKE_MODE": "cheat"})
    project = toy["project"]
    ur("check", cwd=project)
    ur("wait", "--timeout", "120", cwd=project)
    proc = ur("check", "--probe", cwd=project, check=False)
    assert proc.returncode == 1
    assert "That's a hole" in proc.stdout, proc.stdout
    assert status(ur, project)["phase"] == "setup"
    out = ur("check", "--probe", "--accept", cwd=project).stdout
    assert "Accepted" in out
    assert status(ur, project)["phase"] == "setup-brief"


def test_status_without_campaign(tmp_path, ur):
    out = ur("status", cwd=tmp_path).stdout
    assert "Next: initialize, with the user" in out


def test_init_writes_a_template(tmp_path, ur):
    project = tmp_path / "proj"
    project.mkdir()
    (project / ".gitignore").write_text("data/\n# comment\n")
    ur("init", cwd=project)
    config = (project / ".ur" / "config.toml").read_text()
    assert '"data/"' in config
    st = json.loads(ur("status", "--json", cwd=project).stdout)
    assert st["phase"] == "setup"
    assert any("objective.metric is empty" in p for p in st["problems"])
    proc = ur("init", cwd=project, check=False)
    assert proc.returncode == 1
