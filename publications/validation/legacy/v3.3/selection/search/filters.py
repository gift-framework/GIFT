"""Pre-enumeration filters for fast rejection during search."""

import math
from ..grammar.ast_node import ASTNode


def passes_range(value: float, obs_class: str, margin: float = 0.5) -> bool:
    if math.isnan(value) or math.isinf(value):
        return False
    if obs_class == "A":
        return 0 < value < 1000 and abs(value - round(value)) < 1e-6
    elif obs_class == "B":
        return -margin < value < 1 + margin
    elif obs_class == "C":
        return value > 0
    elif obs_class == "D":
        return -margin * 360 < value < 360 * (1 + margin)
    elif obs_class == "E":
        return value > 0
    return True


def passes_target_range(value: float, target: float, margin: float = 0.5) -> bool:
    if math.isnan(value) or math.isinf(value):
        return False
    if abs(target) < 1e-15:
        return abs(value) < margin
    return abs(value - target) / abs(target) <= margin


def passes_max_depth(node: ASTNode, max_depth: int = 3) -> bool:
    return node.depth() <= max_depth


def passes_coefficient_bound(node: ASTNode, max_coeff: int = 5) -> bool:
    return all(abs(v) <= max_coeff for v in _collect_ints(node))


def _collect_ints(node: ASTNode) -> list[int]:
    r = [node.value] if node.kind == "int" else []
    for c in node.children:
        r.extend(_collect_ints(c))
    return r
