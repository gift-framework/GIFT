"""Tests for the analysis module: Pareto, ranking, null models."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from selection.analysis.pareto import pareto_frontier, is_on_frontier, pareto_rank
from selection.analysis.ranking import rank_gift_formula
from selection.analysis.null_model import compute_p_value


def test_pareto_frontier_basic():
    """Simple Pareto frontier computation."""
    formulas = [
        {"comp": 1.0, "err": 0.5, "total": 1.5},
        {"comp": 2.0, "err": 0.3, "total": 1.0},
        {"comp": 3.0, "err": 0.1, "total": 0.8},
        {"comp": 1.5, "err": 0.6, "total": 2.0},  # dominated
        {"comp": 2.5, "err": 0.4, "total": 1.5},  # dominated
    ]

    frontier = pareto_frontier(formulas)
    # Frontier should contain (1, 0.5), (2, 0.3), (3, 0.1)
    assert len(frontier) == 3
    assert frontier[0]["comp"] == 1.0
    assert frontier[1]["comp"] == 2.0
    assert frontier[2]["comp"] == 3.0


def test_pareto_is_on_frontier():
    """Check if a formula is on the frontier."""
    formulas = [
        {"comp": 1.0, "err": 0.5, "total": 1.5, "formula_str": "a"},
        {"comp": 2.0, "err": 0.3, "total": 1.0, "formula_str": "b"},
        {"comp": 3.0, "err": 0.6, "total": 2.0, "formula_str": "c"},  # dominated
    ]

    frontier = pareto_frontier(formulas)
    assert is_on_frontier(formulas[0], frontier)
    assert is_on_frontier(formulas[1], frontier)
    assert not is_on_frontier(formulas[2], frontier)


def test_ranking():
    """Ranking computation."""
    gift = {"err": 0.2, "comp": 5.0, "total": 2.0, "formula_str": "gift"}
    all_scores = [
        {"err": 0.1, "comp": 10.0, "total": 3.0, "formula_str": "a"},
        {"err": 0.2, "comp": 5.0, "total": 2.0, "formula_str": "gift"},
        {"err": 0.5, "comp": 3.0, "total": 1.5, "formula_str": "c"},
        {"err": 0.8, "comp": 2.0, "total": 4.0, "formula_str": "d"},
    ]

    r = rank_gift_formula(gift, all_scores)
    assert r["rank_by_error"] == 2  # One formula has lower error
    assert r["total_formulas"] == 4


def test_p_value():
    """p-value computation."""
    gift_err = 0.1
    null_errors = [0.5, 0.3, 0.2, 0.15, 0.08, 0.4, 0.6, 0.25, 0.35, 0.45]

    p = compute_p_value(gift_err, null_errors)
    # Only 0.08 is <= 0.1, so p = 1/10 = 0.1
    assert abs(p - 0.1) < 0.01


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS: {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {passed + failed}")
