"""One test per bug found in review, so they stay fixed."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from test_manual import make_project
from ur.loop import _record_hashes


def edit_config(project: Path, old: str, new: str) -> None:
    path = project / ".ur" / "config.toml"
    text = path.read_text()
    assert old in text, old
    path.write_text(text.replace(old, new))


def ready_project(tmp, ur) -> Path:
    project = make_project(tmp)
    ur("check", cwd=project)
    (project / ".ur" / "brief.md").write_text(ur("brief", "--draft", cwd=project).stdout)
    return project


def test_absolute_or_escaping_shared_paths_are_refused(tmp_path, ur):
    project = make_project(tmp_path)
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep.txt").write_text("precious\n")
    edit_config(project, 'editable = [', f'shared = ["{victim}"]\neditable = [')
    proc = ur("check", cwd=project, check=False)
    assert proc.returncode == 1 and "must be a path inside the project" in proc.stderr
    assert (victim / "keep.txt").read_text() == "precious\n"


def test_store_inside_the_project_is_refused_and_relative_roots_resolve(tmp_path, ur):
    project = make_project(tmp_path)
    edit_config(project, f'root = "{tmp_path / "store"}"', 'root = "inner-store"')
    proc = ur("check", cwd=project, check=False)
    assert "store.root" in proc.stderr and "inside the project" in proc.stderr
    edit_config(project, 'root = "inner-store"', 'root = "../relative-store"')
    ur("check", cwd=project)
    assert (tmp_path / "relative-store" / "code.git").is_dir()


def test_training_that_writes_outside_artifacts_fails_check(tmp_path, ur):
    project = make_project(tmp_path)
    edit_config(project, "echo $UR_RUN_ID > ckpt/id", "echo $UR_RUN_ID > ckpt/id && echo hi > train.log")
    proc = ur("check", cwd=project, check=False)
    assert proc.returncode == 1
    assert "Training writes files outside run.artifacts" in proc.stderr and "train.log" in proc.stderr


def test_phase_commands_take_turns(tmp_path, ur):
    project = ready_project(tmp_path, ur)
    holder = subprocess.Popen(
        [sys.executable, "-c", "import sys,time; from ur.campaign import Campaign; c=Campaign(__import__('pathlib').Path(sys.argv[1]));\n"
         "with c.exclusive():\n print('held', flush=True); time.sleep(30)", str(project)],
        env=ur.env, stdout=subprocess.PIPE, text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        proc = ur("round", cwd=project, check=False)
        assert proc.returncode == 1 and "Another ur command is changing the campaign" in proc.stderr
    finally:
        holder.kill()
        holder.wait()
    ur("round", cwd=project)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    return not stat.exists() or stat.read_text().rsplit(") ", 1)[1][0] not in "ZX"


def test_killed_run_doesnt_leave_training_behind(tmp_path, ur):
    project = make_project(tmp_path)
    (project / "train.sh").write_text(
        'if grep -q slow score.txt; then echo $$ > "$PIDFILE"; exec sleep 60; fi\n'
        "mkdir -p ckpt && cp score.txt ckpt/score.txt && wc -l < model.txt > ckpt/lines && echo $UR_RUN_ID > ckpt/id\n"
    )
    edit_config(project, 'command = "mkdir -p ckpt && cp score.txt ckpt/score.txt && wc -l < model.txt > ckpt/lines && echo $UR_RUN_ID > ckpt/id"',
                'command = "sh train.sh"')
    edit_config(project, "workers = 3", "workers = 1")
    ur("check", cwd=project)
    (project / ".ur" / "brief.md").write_text("brief\n")
    ur("round", cwd=project)
    ws = tmp_path / "store" / "ws" / "r1-w1"
    (ws / "score.txt").write_text("slow\n")
    pidfile = tmp_path / "train.pid"
    run = subprocess.Popen(["ur", "run"], cwd=ws, env={**ur.env, "PIDFILE": str(pidfile)},
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(200):
        if pidfile.exists() and pidfile.read_text().strip():
            break
        time.sleep(0.05)
    train = int(pidfile.read_text())
    run.send_signal(signal.SIGKILL)
    run.wait()
    time.sleep(0.2)
    assert _alive(train), "the scenario: training outlives a SIGKILLed ur run"
    ur("close", cwd=project)
    assert not _alive(train)
    meta = json.loads((tmp_path / "store" / "meta" / "r1-w1.json").read_text())
    node = json.loads((tmp_path / "store" / "runs" / meta["runs"][-1] / "run.json").read_text()) if meta["runs"] else None
    stuck = [p for p in (tmp_path / "store" / "runs").glob("*/run.json") if json.loads(p.read_text())["status"] == "running"]
    assert not stuck, stuck
    assert node is None or node["status"] == "failed"


def test_died_sessions_are_reported(tmp_path, ur):
    project = ready_project(tmp_path, ur)
    edit_config(project, "[check]", '[agents]\nspawn = "command"\ncommand = "test -f {prompt_file} && exit 3"\n\n[check]')
    ur("round", cwd=project)
    out = ur("wait", "--timeout", "30", cwd=project).stdout
    assert "died" in out and "ur spawn" in out
    st = json.loads(ur("status", "--json", cwd=project).stdout)
    assert {w["session"] for w in st["workers"]} == {"died"}


def test_workspaces_cant_reach_campaign_commands(tmp_path, ur):
    project = ready_project(tmp_path, ur)
    ur("round", cwd=project)
    ws = tmp_path / "store" / "ws" / "r1-w1"
    for args in (["-C", str(project), "status"], ["-C", str(project), "log"], ["-C", str(tmp_path / "store" / "ws" / "r1-w2"), "log"]):
        proc = ur(*args, cwd=ws, check=False)
        assert proc.returncode == 1, args
    proc = ur("status", cwd=project, check=False, env={"UR_SESSION": "r1-w1"})
    assert proc.returncode == 1 and "aren't available from a workspace" in proc.stderr
    proc = ur("log", cwd=tmp_path / "store" / "ws" / "r1-w2", check=False, env={"UR_SESSION": "r1-w1"})
    assert proc.returncode == 1 and "works in workspace r1-w1" in proc.stderr
    ur("status", cwd=ws, env={"UR_SESSION": "r1-w1"})


def test_wait_timeout_zero_returns_at_once(tmp_path, ur):
    project = ready_project(tmp_path, ur)
    ur("round", cwd=project)
    start = time.time()
    proc = ur("wait", "--timeout", "0", cwd=project, check=False)
    assert proc.returncode == 1 and time.time() - start < 10


def test_check_cant_rerun_mid_campaign(tmp_path, ur):
    project = ready_project(tmp_path, ur)
    ur("round", cwd=project)
    (project / ".ur" / "brief.md").unlink()
    proc = ur("check", cwd=project, check=False)
    assert proc.returncode == 1 and "`ur archive` first" in proc.stderr


def test_candidate_cap(tmp_path, ur):
    project = ready_project(tmp_path, ur)
    edit_config(project, "minutes = 10", "minutes = 10\nmax_candidates = 1")
    ur("round", cwd=project)
    for i, score in ((1, "2"), (2, "3")):
        ws = tmp_path / "store" / "ws" / f"r1-w{i}"
        (ws / "score.txt").write_text(score + "\n")
        (ws / "model.txt").write_text((ws / "model.txt").read_text() + f"w{i}\n")
        ur("run", cwd=ws)
        ur("commit", cwd=ws)
    ur("close", cwd=project)
    c = tmp_path / "store" / "ws" / "r1-c"
    (c / "score.txt").write_text("4\n")
    ur("run", cwd=c)
    ur("commit", "--candidate", cwd=c)
    (c / "score.txt").write_text("5\n")
    ur("run", cwd=c)
    proc = ur("commit", "--candidate", cwd=c, check=False)
    assert proc.returncode == 1 and "the most this round takes" in proc.stderr


def test_init_points_at_the_checkpoint(tmp_path, ur):
    project = make_project(tmp_path)
    (project / "train.py").write_text(
        "import os, shutil, sys\n"
        "init = sys.argv[1]\n"
        "os.makedirs('ckpt', exist_ok=True)\n"
        "shutil.copy('score.txt', 'ckpt/score.txt')\n"
        "open('ckpt/lines', 'w').write(str(sum(1 for _ in open('model.txt'))))\n"
        "open('ckpt/id', 'w').write(os.environ['UR_RUN_ID'])\n"
        "open('ckpt/init', 'w').write(open(os.path.join(init, 'id')).read() if init else 'none')\n"
    )
    edit_config(project, 'command = "mkdir -p ckpt && cp score.txt ckpt/score.txt && wc -l < model.txt > ckpt/lines && echo $UR_RUN_ID > ckpt/id"',
                f'command = "{sys.executable} train.py {{init}}"\ninit = "base"')
    edit_config(project, 'artifacts = ["ckpt"]\n', 'artifacts = ["ckpt"]\n')
    ur("check", cwd=project)
    (project / ".ur" / "brief.md").write_text("brief\n")
    ur("round", cwd=project)
    ws = tmp_path / "store" / "ws" / "r1-w1"
    ur("run", cwd=ws)
    base = json.loads((project / ".ur" / "state" / "campaign.json").read_text())["base"]
    run = json.loads((tmp_path / "store" / "meta" / "r1-w1.json").read_text())["last_run"]
    assert (tmp_path / "store" / "runs" / run / "artifacts" / "ckpt" / "init").read_text().strip() == base


def test_overlap_ignores_headers_and_short_lines(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_text("x,y,label\n0.123456,1.5,cat\n{\n")
    b.write_text("x,y,label\n0.654321,2.5,dog\n{\n")
    assert not (_record_hashes(a, None)[0] & _record_hashes(b, None)[0])


def test_a_worker_cant_fill_in_its_own_heldout_score(tmp_path, ur):
    project = ready_project(tmp_path, ur)
    ur("round", cwd=project)
    ws = tmp_path / "store" / "ws" / "r1-w1"
    (ws / "score.txt").write_text("2\n")
    ur("run", cwd=ws)
    ur("commit", cwd=ws)
    head = json.loads((tmp_path / "store" / "meta" / "r1-w1.json").read_text())["head"]
    record = tmp_path / "store" / "runs" / head / "run.json"
    node = json.loads(record.read_text())
    node["heldout"], node["heldout_repeats"] = {"score": 1e9, "params": 1}, [1e9]
    record.write_text(json.dumps(node))
    ur("close", cwd=project)
    s = json.loads((project / ".ur" / "state" / "campaign.json").read_text())
    assert s["rounds"][0]["ranking"][0]["score"] < 10
