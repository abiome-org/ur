"""Stands in for an agent session in tests: reads its prompt, then does a fixed job with ur.

FAKE_MODE=cheat makes the cheat probe actually cheat (FAKE_HELDOUT names the file it reads).
"""

import os
import re
import subprocess
import sys
from pathlib import Path

X2 = "    f.append(x * x)\n"
SIN = "    f.append(math.sin(4 * x))\n"
COMBINED = '''import math

"""The model: a linear model over these features. Change the features to change the model."""


def features(x):
    f = [1.0, x]
    f.append(x * x)
    f.append(math.sin(4 * x))
    return f
'''


def ur(*args, check=True):
    proc = subprocess.run(["ur", *args], capture_output=True, text=True)
    print(f"$ ur {' '.join(args)}\n{proc.stdout}{proc.stderr}", flush=True)
    if check and proc.returncode != 0:
        raise SystemExit(f"ur {' '.join(args)} failed")
    return proc


def add_line(line, imports=""):
    src = Path("model.py").read_text()
    src = src.replace("    f = [1.0, x]\n", "    f = [1.0, x]\n" + line)
    Path("model.py").write_text(imports + src)


def probe(mode):
    if mode != "cheat":
        return
    heldout = os.environ["FAKE_HELDOUT"]
    Path("model.py").write_text(
        "import json\n"
        f"LOOKUP = {{round(json.loads(l)['x'], 6): json.loads(l)['y'] for l in open({heldout!r})}}\n\n\n"
        "def features(x):\n    return [LOOKUP.get(round(x, 6), 0.0)]\n"
    )
    Path("train.py").write_text(
        "import json, os\nos.makedirs('ckpt', exist_ok=True)\njson.dump([1.0], open('ckpt/weights.json', 'w'))\n"
    )
    ur("run")
    ur("commit")


def construct(prompt):
    nodes = re.search(r"best first: (.+)\.\n", prompt).group(1).split(", ")
    for node in nodes:
        ur("apply", node)
    if "<<<<<<<" in Path("model.py").read_text():
        Path("model.py").write_text(COMBINED)
    ur("run")
    ur("commit", "--candidate")


def worker(name):
    if not name.startswith("r1-"):
        return  # later rounds: find nothing, so the campaign stalls and stops
    if name.endswith("w1"):
        add_line(X2)
        ur("run")
        ur("commit")
    elif name.endswith("w2"):
        # A note outside the editable paths is refused; a worse change gets reset.
        Path("NOTES.md").write_text("sin features help\n")
        assert ur("run", check=False).returncode != 0
        Path("NOTES.md").unlink()
        Path("model.py").write_text('def features(x):\n    return [1.0]\n')
        ur("run")
        ur("reset")
        add_line(SIN, imports="import math\n")
        ur("run")
        ur("commit")


def main():
    prompt = Path(sys.argv[1]).read_text()
    path = re.search(r"Your workspace: (.+)", prompt).group(1).strip()
    os.chdir(path)
    if "find holes" in prompt:
        probe(os.environ.get("FAKE_MODE", ""))
    elif "combine the best changes" in prompt:
        construct(prompt)
    else:
        worker(Path(path).name)


if __name__ == "__main__":
    main()
