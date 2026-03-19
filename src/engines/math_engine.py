"""Safe math evaluation without exposing raw eval to arbitrary code."""
from __future__ import annotations

import ast
import operator as op
import re
from typing import Any

_ALLOWED_BINOPS: dict[type[Any], Any] = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.FloorDiv: op.floordiv,
}
_ALLOWED_UNARYOPS: dict[type[Any], Any] = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}


class SafeMathError(ValueError):
    pass


def extract_expression(question: str) -> str:
    match = re.search(r'([-+/*().%^0-9\s]+)', question.replace(',', '.'))
    expression = (match.group(1) if match else question).strip()
    expression = expression.replace('^', '**')
    if not expression:
        raise SafeMathError('No mathematical expression found.')
    if not re.fullmatch(r'[0-9\s+\-*/().%*]+', expression):
        raise SafeMathError('Expression contains invalid characters.')
    return expression


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_eval_node(node.operand))
    raise SafeMathError('Unsupported math operation.')


def solve_math(question: str) -> str:
    expression = extract_expression(question)
    parsed = ast.parse(expression, mode='eval')
    value = _eval_node(parsed)
    if value.is_integer():
        return str(int(value))
    return f'{value:.10g}'
