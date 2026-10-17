import pytest

from ur import guards
from ur.util import PathMatcher, UrError, fill, wrap_command


def test_matcher_follows_gitignore_rules():
    m = PathMatcher(["__pycache__/", "*.pyc", "/data", "logs/*.txt", "**/tmp"])
    assert m.match("__pycache__", True)
    assert not m.match("__pycache__", False)  # trailing slash: directories only
    assert m.match("a/b/x.pyc", False)  # no slash: any depth
    assert m.match("data", True)
    assert not m.match("sub/data", True)  # leading slash: anchored
    assert m.match("logs/a.txt", False)
    assert not m.match("logs/deep/a.txt", False)  # * doesn't cross directories
    assert m.match("x/y/tmp", True)
    assert m.match_path("data/train.jsonl")
    assert m.match_path("pkg/__pycache__/m.cpython.pyc")
    assert not m.match_path("model.py")


def test_editable_globs():
    m = PathMatcher(["model/**", "train.py"])
    assert m.match_path("model/layers/attn.py")
    assert m.match_path("train.py")
    assert not m.match_path("eval.py")
    assert PathMatcher(["**"]).match_path("anything/at/all.py")


def test_guards():
    assert guards.failures(["params <= 10"], {"params": 5}, None) == []
    assert guards.failures(["latency <= base * 1.1"], {"latency": 10.5}, {"latency": 10}) == []
    (msg,) = guards.failures(["latency <= base * 1.1"], {"latency": 12}, {"latency": 10})
    assert "latency is 12" in msg
    (msg,) = guards.failures(["acc >= 0.5"], {}, {})
    assert "no acc" in msg
    (msg,) = guards.failures(["acc >= base - 1"], {"acc": 3}, {})
    assert "the base has no acc" in msg
    for bad in ["params", "params <= __import__('os')", "params <= base.x", "params <= True"]:
        with pytest.raises(UrError):
            guards.parse(bad)


def test_fill_quotes_values_and_leaves_other_braces():
    cmd = fill("run {budget} --out {out} awk '{print $1}'", budget=5, out="a b")
    assert cmd == "run 5 --out 'a b' awk '{print $1}'"
    assert fill("x {init}", init="") == "x ''"


def test_wrap_command():
    assert wrap_command("", "train") == "train"
    assert wrap_command("srun -n1", "train") == "srun -n1 train"
    assert wrap_command("sh -c {qcmd}", "a && b") == "sh -c 'a && b'"
    assert wrap_command("nice {cmd}", "train") == "nice train"
