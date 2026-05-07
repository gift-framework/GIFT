"""Bounded AST enumeration engine."""

import math
from ..grammar.ast_node import ASTNode, atom, integer, transcendental, op
from ..grammar.atoms import ATOM_COSTS, get_allowed_atoms
from ..grammar.operations import get_binary_ops, get_unary_ops, OP_COSTS
from ..grammar.complexity import complexity
from ..config import INVARIANTS, TRANSCENDENTALS, MAX_DEPTH, MAX_COEFF
from .filters import passes_range, passes_target_range
from .dedup import FormulaDeduplicator


def enumerate_formulas(
    obs_class: str,
    target_value: float,
    max_complexity: float = 12.0,
    max_depth: int = MAX_DEPTH,
    invariant_set: str = "primary",
    target_margin: float = 0.5,
) -> list[tuple[float, ASTNode]]:
    dedup = FormulaDeduplicator()
    leaves = _make_leaves(obs_class, invariant_set)
    binary_ops = get_binary_ops(obs_class)
    unary_ops = get_unary_ops(obs_class)

    # Level 0: evaluate leaves
    level0 = []
    for node in leaves:
        val = node.evaluate_float(INVARIANTS, TRANSCENDENTALS)
        if math.isfinite(val):
            level0.append((val, node))
            if passes_target_range(val, target_value, target_margin):
                if complexity(node) <= max_complexity:
                    dedup.add(node, val)

    # Level 1: binary on leaves + unary on leaves
    level1 = []
    for i, (va, na) in enumerate(level0):
        for j, (vb, nb) in enumerate(level0):
            if i == j:
                continue
            for op_name in binary_ops:
                new_node = op(op_name, na, nb)
                c = complexity(new_node)
                if c > max_complexity:
                    continue
                val = new_node.evaluate_float(INVARIANTS, TRANSCENDENTALS)
                if math.isfinite(val) and abs(val) < 1e12:
                    level1.append((val, new_node))
                    if passes_target_range(val, target_value, target_margin):
                        if passes_range(val, obs_class):
                            dedup.add(new_node, val)

    for val0, node0 in level0:
        for op_name in unary_ops:
            new_node = op(op_name, node0)
            c = complexity(new_node)
            if c > max_complexity:
                continue
            val = new_node.evaluate_float(INVARIANTS, TRANSCENDENTALS)
            if math.isfinite(val) and abs(val) < 1e12:
                level1.append((val, new_node))
                if passes_target_range(val, target_value, target_margin):
                    if passes_range(val, obs_class):
                        dedup.add(new_node, val)

    if max_depth < 2:
        return dedup.get_all()

    # Level 2: binary on (level0 x level1) + unary on level1
    for va, na in level0:
        for vb, nb in level1:
            for op_name in binary_ops:
                for left, right in [(na, nb), (nb, na)]:
                    new_node = op(op_name, left, right)
                    c = complexity(new_node)
                    if c > max_complexity or new_node.depth() > max_depth:
                        continue
                    val = new_node.evaluate_float(INVARIANTS, TRANSCENDENTALS)
                    if math.isfinite(val) and passes_target_range(val, target_value, target_margin):
                        if passes_range(val, obs_class):
                            dedup.add(new_node, val)

    for val1, node1 in level1:
        for op_name in unary_ops:
            new_node = op(op_name, node1)
            c = complexity(new_node)
            if c > max_complexity or new_node.depth() > max_depth:
                continue
            val = new_node.evaluate_float(INVARIANTS, TRANSCENDENTALS)
            if math.isfinite(val) and passes_target_range(val, target_value, target_margin):
                if passes_range(val, obs_class):
                    dedup.add(new_node, val)

    return dedup.get_all()


def _make_leaves(obs_class: str, invariant_set: str) -> list[ASTNode]:
    leaves = []
    allowed = get_allowed_atoms(obs_class)
    for name in allowed:
        if invariant_set == "primary" and ATOM_COSTS.get(name, 0) > 1:
            if name not in ("pi", "sqrt2", "phi", "ln2", "zeta3", "zeta5", "zeta11"):
                continue
        if name in INVARIANTS:
            leaves.append(atom(name))
        elif name in ("pi", "sqrt2", "phi", "ln2", "zeta3", "zeta5", "zeta11"):
            leaves.append(transcendental(name))
    for n in range(1, MAX_COEFF + 1):
        leaves.append(integer(n))
    return leaves
