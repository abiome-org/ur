# Toy project

A regression problem small enough to train in a second on a CPU, for trying ur end to end. The model is linear over the features in `model.py`, trained by minibatch gradient descent until its time budget runs out. The data is one noisy function of `x` on [-1, 1], and the starting features fit it badly.

Its data comes from a generator kept outside this directory, so workers can't read how it was made:

```sh
python3 ../toy-data/make_data.py --project . --private /tmp/toy-private
```

That writes train and dev into `data/`, and held-out and final into `/tmp/toy-private`. Then tell your agent: *"I'd like to initialize ur for this project."* `train.py --budget {budget} --init {init}` trains, and the checkpoint is `ckpt/weights.json`.
