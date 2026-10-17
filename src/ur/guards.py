"""Guards: hard limits a candidate's held-out scores must meet, like `params <= 1.5e9`."""

from __future__ import annotations

import ast
import operator
import re
from typing import Any

from .util import UrError

_GUARD = re.compile(r"^\s*([A-Za-z_][\w./:-]*)\s*(<=|>=|==|!=|<|>)\s*(.+?)\s*$")
_COMPARE = {
    "<=": operator.le,
    ">=": operator.ge,
    "<": operator.lt,
    ">": operator.gt,
    "==": operator.eq,
    "!=": operator.ne,
}
_ARITH = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def parse(text: str) -> tuple[str, str, ast.expr]:
    match = _GUARD.match(text)
    if not match:
        raise UrError(f"can't read guard {text!r}; write it like \"params <= 1.5e9\" or \"latency_ms <= base * 1.05\".")
    metric, op, rhs = match.groups()
    try:
        expr = ast.parse(rhs, mode="eval").body
    except SyntaxError:
        raise UrError(f"can't read the right side of guard {text!r}.") from None
    _check(expr, text)
    return metric, op, expr


def _check(node: ast.expr, text: str) -> None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return
    if isinstance(node, ast.Name) and node.id == "base":
        return
    if isinstance(node, ast.BinOp) and type(node.op) in _ARITH:
        _check(node.left, text)
        _check(node.right, text)
        return
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        _check(node.operand, text)
        return
    raise UrError(f"guard {text!r}: the right side may only use numbers, `base`, and + - * /.")


def _value(node: ast.expr, base: Any) -> float:
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.Name):
        if not isinstance(base, (int, float)):
            raise KeyError("base")
        return float(base)
    if isinstance(node, ast.BinOp):
        return _ARITH[type(node.op)](_value(node.left, base), _value(node.right, base))
    assert isinstance(node, ast.UnaryOp)
    value = _value(node.operand, base)
    return -value if isinstance(node.op, ast.USub) else value


def failures(guards: list[str], metrics: dict[str, float], base_metrics: dict[str, float] | None) -> list[str]:
    """The guards these metrics break, each with the reason."""
    out = []
    for text in guards:
        metric, op, expr = parse(text)
        value = metrics.get(metric)
        if not isinstance(value, (int, float)):
            out.append(f"{text}: the scorer reported no {metric}")
            continue
        try:
            bound = _value(expr, (base_metrics or {}).get(metric))
        except KeyError:
            out.append(f"{text}: the base has no {metric} to compare with")
            continue
        except ZeroDivisionError:
            out.append(f"{text}: divides by zero")
            continue
        if not _COMPARE[op](value, bound):
            out.append(f"{text}: {metric} is {value:.4g}, limit {bound:.4g}")
    return out
