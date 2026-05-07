"""Null models for statistical significance testing."""

from __future__ import annotations
import random
import math
from ..grammar.ast_node import ASTNode, atom, integer, op
from ..grammar.atoms import get_allowed_atoms
from ..grammar.operations import get_binary_ops, get_unary_ops
from ..grammar.complexity import complexity
from ..config import INVARIANTS, TRANSCENDENTALS, MAX_COEFF
from ..scoring.error import error_score


def null_distribution_random(obs_class: str, y_exp: float, sigma_y: float,
                             n_random: int = 10000, max_complexity: float = 12.0,
                             max_depth: int = 3, seed: int = 42) -> dict:
    rng = random.Random(seed)
    allowed_atoms = get_allowed_atoms(obs_class)
    binary_ops = get_binary_ops(obs_class)
    unary_ops = get_unary_ops(obs_class)
    errors = []
    for _ in range(n_random):
        try:
            tree = _random_tree(rng, allowed_atoms, binary_ops, unary_ops,
                                max_depth, max_complexity)
            if tree is None:
                continue
            val = tree.evaluate_float(INVARIANTS, TRANSCENDENTALS)
            if math.isfinite(val) and val > 0:
                err = error_score(val, y_exp, sigma_y)
                if math.isfinite(err):
                    errors.append(err)
        except Exception:
            continue
    if not errors:
        return {"errors": [], "p_value": 1.0, "mean_error": float('inf'),
                "std_error": 0.0, "n_valid": 0, "n_total": n_random}
    mean_err = sum(errors) / len(errors)
    try:
        var_err = sum((e - mean_err) ** 2 for e in errors) / len(errors)
        std_err = math.sqrt(var_err)
    except (OverflowError, ValueError):
        std_err = float('inf')
    return {"errors": sorted(errors), "mean_error": mean_err,
            "std_error": std_err, "n_valid": len(errors), "n_total": n_random}


def compute_p_value(gift_error: float, null_errors: list[float]) -> float:
    if not null_errors:
        return 1.0
    return sum(1 for e in null_errors if e <= gift_error) / len(null_errors)


def null_distribution_shuffled(gift_ast: ASTNode, obs_class: str, y_exp: float,
                               sigma_y: float, n_shuffles: int = 10000,
                               seed: int = 42) -> dict:
    rng = random.Random(seed)
    allowed_atoms = [a for a in get_allowed_atoms(obs_class) if a in INVARIANTS]
    atom_positions = _find_atom_positions(gift_ast)
    if not atom_positions or not allowed_atoms:
        return {"errors": [], "p_value": 1.0, "mean_error": float('inf'),
                "std_error": 0.0, "n_valid": 0, "n_total": n_shuffles}
    errors = []
    for _ in range(n_shuffles):
        shuffled = _shuffle_atoms(gift_ast, atom_positions, allowed_atoms, rng)
        try:
            val = shuffled.evaluate_float(INVARIANTS, TRANSCENDENTALS)
            if math.isfinite(val) and val > 0:
                err = error_score(val, y_exp, sigma_y)
                if math.isfinite(err):
                    errors.append(err)
        except Exception:
            continue
    if not errors:
        return {"errors": [], "p_value": 1.0, "mean_error": float('inf'),
                "std_error": 0.0, "n_valid": 0, "n_total": n_shuffles}
    mean_err = sum(errors) / len(errors)
    try:
        var_err = sum((e - mean_err) ** 2 for e in errors) / len(errors)
        std_err = math.sqrt(var_err)
    except (OverflowError, ValueError):
        std_err = float('inf')
    return {"errors": sorted(errors), "mean_error": mean_err,
            "std_error": std_err, "n_valid": len(errors), "n_total": n_shuffles}


def _random_tree(rng, atoms, binary_ops, unary_ops, max_depth, max_comp, depth=0):
    if depth >= max_depth or rng.random() < 0.4:
        if rng.random() < 0.6 and atoms:
            name = rng.choice(atoms)
            if name in INVARIANTS:
                return atom(name)
            from ..grammar.ast_node import transcendental
            return transcendental(name)
        return integer(rng.randint(1, MAX_COEFF))
    if rng.random() < 0.7 and binary_ops:
        op_name = rng.choice(binary_ops)
        left = _random_tree(rng, atoms, binary_ops, unary_ops, max_depth, max_comp, depth + 1)
        right = _random_tree(rng, atoms, binary_ops, unary_ops, max_depth, max_comp, depth + 1)
        if left is None or right is None:
            return None
        result = op(op_name, left, right)
        return result if complexity(result) <= max_comp else None
    elif unary_ops:
        op_name = rng.choice(unary_ops)
        child = _random_tree(rng, atoms, binary_ops, unary_ops, max_depth, max_comp, depth + 1)
        if child is None:
            return None
        result = op(op_name, child)
        return result if complexity(result) <= max_comp else None
    return integer(rng.randint(1, MAX_COEFF))


def _find_atom_positions(node: ASTNode, path: tuple = ()) -> list[tuple]:
    r = [path] if node.kind == "atom" else []
    for i, c in enumerate(node.children):
        r.extend(_find_atom_positions(c, path + (i,)))
    return r


def _shuffle_atoms(node: ASTNode, positions: list[tuple],
                   allowed_atoms: list[str], rng) -> ASTNode:
    replacements = {pos: rng.choice(allowed_atoms) for pos in positions}
    return _replace_atoms(node, replacements, ())


def _replace_atoms(node: ASTNode, replacements: dict, path: tuple) -> ASTNode:
    if path in replacements:
        return atom(replacements[path])
    if not node.children:
        return node
    new_children = [_replace_atoms(c, replacements, path + (i,))
                    for i, c in enumerate(node.children)]
    return ASTNode(node.kind, node.value, new_children)
