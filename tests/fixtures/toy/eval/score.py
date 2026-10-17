"""Score a workspace: mean squared error of its model on a split."""

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.workspace)
    import model

    with open(os.path.join(args.workspace, "ckpt", "weights.json")) as f:
        w = json.load(f)
    with open(args.data) as f:
        rows = [json.loads(line) for line in f]
    se = 0.0
    for r in rows:
        feats = model.features(r["x"])
        if len(feats) != len(w):
            raise SystemExit(f"the model has {len(feats)} features but the checkpoint has {len(w)} weights")
        se += (sum(wi * fi for wi, fi in zip(w, feats)) - r["y"]) ** 2
    with open(args.out, "w") as f:
        json.dump({"mse": se / len(rows), "params": len(w)}, f)


if __name__ == "__main__":
    main()
