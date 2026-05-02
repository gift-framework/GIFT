"""Fragility scoring: robustness under integer leaf perturbation."""

import math
from ..grammar.ast_node import ASTNode


def fragility_score(node: ASTNode, target: float,
                    invariants: dict, transcendentals: dict,
                    tolerance: float = 0.05) -> float:
    positions = _find_int_positions(node)
    if not positions:
        return 0.0
    n_total = len(positions) * 2
    n_robust = 0
    for path in positions:
        for delta in [-1, +1]:
            perturbed = _perturb_at(node, path, delta)
            try:
                val = perturbed.evaluate_float(invariants, transcendentals)
                if math.isfinite(val) and abs(target) > 1e-15:
                    if abs(val - target) / abs(target) <= tolerance:
                        n_robust += 1
            except Exception:
                pass
    return 1.0 - (n_robust / n_total) if n_total else 0.0


def _find_int_positions(node: ASTNode, path: tuple = ()) -> list[tuple]:
    r = [path] if node.kind == "int" else []
    for i, c in enumerate(node.children):
        r.extend(_find_int_positions(c, path + (i,)))
    return r


def _perturb_at(node: ASTNode, path: tuple, delta: int) -> ASTNode:
    if not path:
        return ASTNode("int", node.value + delta) if node.kind == "int" else node
    new_children = list(node.children)
    new_children[path[0]] = _perturb_at(node.children[path[0]], path[1:], delta)
    return ASTNode(node.kind, node.value, new_children)
