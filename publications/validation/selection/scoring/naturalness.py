"""Naturalness scoring: penalize formulas violating physical constraints."""

import math
from ..grammar.ast_node import ASTNode


def check_range(value: float, obs_class: str) -> tuple[bool, float]:
    if math.isnan(value) or math.isinf(value):
        return False, 5.0
    if obs_class == "A":
        if value <= 0 or abs(value - round(value)) > 1e-6:
            return False, 5.0
    elif obs_class == "B":
        if value <= 0 or value >= 1:
            return False, 5.0
    elif obs_class == "C":
        if value <= 0:
            return False, 5.0
    elif obs_class == "D":
        if value <= 0 or value >= 360:
            return False, 5.0
    elif obs_class == "E":
        if value <= 0:
            return False, 5.0
    return True, 0.0


def check_small_coefficients(node: ASTNode, max_coeff: int = 5) -> tuple[bool, float]:
    penalty = sum(2.0 for v in _collect_int_leaves(node) if abs(v) > max_coeff)
    return penalty == 0, penalty


def check_type_consistency(node: ASTNode, obs_class: str) -> tuple[bool, float]:
    penalty = 0.0
    if obs_class in ("A", "B", "C"):
        trans = _collect_transcendental_atoms(node) - {"sqrt2"}
        if trans:
            penalty += 3.0
        if _has_op(node, {"arctan", "arcsin"}):
            penalty += 3.0
        if _has_op(node, {"log", "exp"}):
            penalty += 3.0
    return penalty == 0, penalty


def check_no_gratuitous_transcendentals(node: ASTNode, obs_class: str) -> tuple[bool, float]:
    if obs_class in ("D", "E"):
        return True, 0.0
    trans = _collect_transcendental_atoms(node) - {"sqrt2"}
    return (not trans), (4.0 if trans else 0.0)


def naturalness_score(node: ASTNode, obs_class: str, value: float, max_coeff: int = 5) -> float:
    total = 0.0
    _, p = check_range(value, obs_class); total += p
    _, p = check_small_coefficients(node, max_coeff); total += p
    _, p = check_type_consistency(node, obs_class); total += p
    _, p = check_no_gratuitous_transcendentals(node, obs_class); total += p
    return total


def _collect_int_leaves(node: ASTNode) -> list[int]:
    r = [node.value] if node.kind == "int" else []
    for c in node.children:
        r.extend(_collect_int_leaves(c))
    return r

def _has_op(node: ASTNode, ops: set[str]) -> bool:
    if node.kind == "op" and node.value in ops:
        return True
    return any(_has_op(c, ops) for c in node.children)

def _collect_transcendental_atoms(node: ASTNode) -> set[str]:
    r = {node.value} if node.kind == "transcendental" else set()
    for c in node.children:
        r |= _collect_transcendental_atoms(c)
    return r
