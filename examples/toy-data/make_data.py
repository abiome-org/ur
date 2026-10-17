"""Make the toy project's data: y = 1.5x^2 - x + 0.3 sin(4x) + noise, for x in [-1, 1].

This lives outside the toy project on purpose: a worker that can read the formula
doesn't need to learn it. train and dev go to <project>/data, where workers can read
them; heldout and final go to --private, which should be outside the project too.
"""

import argparse
import json
import math
import os
import random


def rows(n, rng):
    out = []
    for _ in range(n):
        x = rng.uniform(-1, 1)
        y = 1.5 * x * x - x + 0.3 * math.sin(4 * x) + rng.gauss(0, 0.05)
        out.append({"x": round(x, 6), "y": round(y, 6)})
    return out


def write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="the toy project; train and dev go in its data/")
    parser.add_argument("--private", required=True, help="where heldout and final go; keep it outside the project")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    write(os.path.join(args.project, "data", "train.jsonl"), rows(2000, rng))
    write(os.path.join(args.project, "data", "dev.jsonl"), rows(500, rng))
    write(os.path.join(args.private, "heldout.jsonl"), rows(500, rng))
    write(os.path.join(args.private, "final.jsonl"), rows(500, rng))


if __name__ == "__main__":
    main()
