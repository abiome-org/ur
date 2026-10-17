# Toy project

A regression problem small enough to train in a second on a CPU, for trying ur end to end. The model is linear over the features in `model.py`, trained by minibatch gradient descent until its time budget runs out. The data is `y = 1.5x² − x + 0.3·sin(4x) + noise`, so there's a lot to gain from better features.

```sh
python3 make_data.py --private /tmp/toy-private   # train and dev in data/, held-out and final outside the project
```

Then tell your agent: *"I'd like to initialize ur for this project."* The held-out and final files are in `/tmp/toy-private`, `train.py --budget {budget} --init {init}` trains, and the checkpoint is `ckpt/weights.json`.
