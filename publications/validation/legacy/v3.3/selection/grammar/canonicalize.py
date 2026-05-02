"""Symbolic canonicalization for formula deduplication."""

import sympy
from .ast_node import ASTNode


def canonical_value(node: ASTNode, invariants: dict, transcendentals: dict) -> sympy.Expr:
    expr = node.evaluate(invariants, transcendentals)
    try:
        return sympy.nsimplify(expr, rational=False)
    except Exception:
        return expr


def canonical_float(node: ASTNode, invariants: dict, transcendentals: dict) -> float:
    return node.evaluate_float(invariants, transcendentals)


def are_equivalent(a: ASTNode, b: ASTNode,
                   invariants: dict, transcendentals: dict,
                   tol: float = 1e-12) -> bool:
    fa = a.evaluate_float(invariants, transcendentals)
    fb = b.evaluate_float(invariants, transcendentals)
    if abs(fa - fb) > tol * max(abs(fa), abs(fb), 1.0):
        return False
    try:
        expr_a = a.evaluate(invariants, transcendentals)
        expr_b = b.evaluate(invariants, transcendentals)
        return sympy.simplify(expr_a - expr_b) == 0
    except Exception:
        return abs(fa - fb) < tol
