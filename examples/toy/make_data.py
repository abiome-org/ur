"""Make the toy dataset: y = 1.5x^2 - x + 0.3 sin(4x) + noise, for x in [-1, 1].

train and dev go in data/, where workers can read them. heldout and final go to
--private, which should be outside the project.
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
    parser.add_argument("--private", required=True, help="where heldout and final go; keep it outside the project")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    here = os.path.dirname(os.path.abspath(__file__))
    write(os.path.join(here, "data", "train.jsonl"), rows(2000, rng))
    write(os.path.join(here, "data", "dev.jsonl"), rows(500, rng))
    write(os.path.join(args.private, "heldout.jsonl"), rows(500, rng))
    write(os.path.join(args.private, "final.jsonl"), rows(500, rng))


if __name__ == "__main__":
    main()
