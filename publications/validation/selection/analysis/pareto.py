"""Pareto frontier computation for error vs. complexity tradeoff."""

from __future__ import annotations


def pareto_frontier(formulas: list[dict]) -> list[dict]:
    if not formulas:
        return []
    sorted_f = sorted(formulas, key=lambda f: (f["comp"], f["err"]))
    frontier = []
    min_err = float('inf')
    for f in sorted_f:
        if f["err"] < min_err:
            frontier.append(f)
            min_err = f["err"]
    return frontier


def is_on_frontier(formula: dict, frontier: list[dict]) -> bool:
    for f in frontier:
        if (f.get("formula_str") == formula.get("formula_str") or
            (abs(f["comp"] - formula["comp"]) < 1e-10 and
             abs(f["err"] - formula["err"]) < 1e-10)):
            return True
    return False


def pareto_rank(formula: dict, all_formulas: list[dict]) -> int:
    remaining = list(all_formulas)
    rank = 0
    while remaining:
        frontier = pareto_frontier(remaining)
        if is_on_frontier(formula, frontier):
            return rank
        remaining = [f for f in remaining if not is_on_frontier(f, frontier)]
        rank += 1
    return rank
