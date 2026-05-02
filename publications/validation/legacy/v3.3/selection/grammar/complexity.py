"""Complexity scoring for formula ASTs."""

from .ast_node import ASTNode
from .atoms import ATOM_COSTS
from .operations import OP_COSTS


def complexity(node: ASTNode) -> float:
    if node.kind == "atom":
        return ATOM_COSTS.get(node.value, 2.0)
    elif node.kind == "int":
        n = abs(node.value)
        if n <= 5:
            return 0.5
        elif n <= 10:
            return 1.0
        return 1.5
    elif node.kind == "transcendental":
        return ATOM_COSTS.get(node.value, 5.0)
    elif node.kind == "op":
        return OP_COSTS.get(node.value, 2.0) + sum(complexity(c) for c in node.children)
    return 0.0


def complexity_with_depth_penalty(node: ASTNode) -> float:
    base = complexity(node)
    penalty = max(0, node.depth() - 3) * 2.0
    return base + penalty


def complexity_class(node: ASTNode) -> str:
    ops = _collect_ops(node)
    if ops & {"arctan", "arcsin"}:
        return "angle"
    elif ops & {"log", "exp"} or _has_transcendental_atom(node):
        return "transcendental"
    elif ops & {"sqrt"}:
        return "algebraic"
    return "rational"


def _collect_ops(node: ASTNode) -> set[str]:
    ops = set()
    if node.kind == "op":
        ops.add(node.value)
    for c in node.children:
        ops |= _collect_ops(c)
    return ops


def _has_transcendental_atom(node: ASTNode) -> bool:
    if node.kind == "transcendental":
        return True
    return any(_has_transcendental_atom(c) for c in node.children)
