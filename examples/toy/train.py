"""Train the model by minibatch gradient descent until the time budget runs out."""

import argparse
import json
import os
import random
import time

import model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, required=True, help="seconds of training")
    parser.add_argument("--init", default="", help="a checkpoint directory to start from")
    parser.add_argument("--out", default="ckpt")
    args = parser.parse_args()

    with open(os.path.join("data", "train.jsonl")) as f:
        rows = [json.loads(line) for line in f]
    xs = [model.features(r["x"]) for r in rows]
    ys = [r["y"] for r in rows]
    w = [0.0] * len(xs[0])
    if args.init and os.path.exists(os.path.join(args.init, "weights.json")):
        with open(os.path.join(args.init, "weights.json")) as f:
            start = json.load(f)
        if len(start) == len(w):
            w = start

    rng = random.Random(os.environ.get("UR_RUN_ID", "0"))
    lr, batch, steps = 0.05, 16, 0
    end = time.time() + args.budget
    while time.time() < end:
        grad = [0.0] * len(w)
        for _ in range(batch):
            i = rng.randrange(len(xs))
            err = sum(wj * xj for wj, xj in zip(w, xs[i])) - ys[i]
            for j, xj in enumerate(xs[i]):
                grad[j] += 2 * err * xj / batch
        w = [wj - lr * g for wj, g in zip(w, grad)]
        steps += 1

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "weights.json"), "w") as f:
        json.dump(w, f)
    print(f"{steps} steps, weights {w}")


if __name__ == "__main__":
    main()
