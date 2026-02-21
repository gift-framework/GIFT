"""Tests for the scoring module."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import math
from selection.config import INVARIANTS, TRANSCENDENTALS
from selection.grammar.ast_node import atom, integer, div, add, mul, sub
from selection.scoring.error import error_score, relative_error
from selection.scoring.naturalness import (
    check_range, check_small_coefficients, naturalness_score,
)
from selection.scoring.fragility import fragility_score
from selection.scoring.composite import composite_score


def test_error_score_with_uncertainty():
    """Z-score when uncertainty is provided."""
    err = error_score(0.231, 0.23122, 0.00003)
    # |0.231 - 0.23122| / 0.00003 = 0.00022 / 0.00003 = 7.33
    assert abs(err - 7.33) < 0.1


def test_error_score_relative():
    """Relative error when no uncertainty."""
    err = error_score(0.231, 0.23122, 0.0)
    assert abs(err - 0.00095) < 0.001


def test_range_class_B():
    """Class B: values in (0,1)."""
    passed, _ = check_range(0.5, "B")
    assert passed
    passed, _ = check_range(1.5, "B")
    assert not passed
    passed, _ = check_range(-0.1, "B")
    assert not passed


def test_range_class_D():
    """Class D: angles in (0, 360)."""
    passed, _ = check_range(197, "D")
    assert passed
    passed, _ = check_range(400, "D")
    assert not passed


def test_small_coefficients():
    """Integer leaves within [-5, 5]."""
    node = mul(integer(3), atom("b2"))
    passed, _ = check_small_coefficients(node, max_coeff=5)
    assert passed

    node = mul(integer(7), atom("dim_G2"))
    passed, _ = check_small_coefficients(node, max_coeff=5)
    assert not passed


def test_naturalness_weinberg():
    """Weinberg angle formula should be perfectly natural for class B."""
    weinberg = div(atom("b2"), add(atom("b3"), atom("dim_G2")))
    score = naturalness_score(weinberg, "B", 0.230769)
    assert score == 0.0  # No violations


def test_fragility_weinberg():
    """Weinberg formula has no integer leaves, so fragility = 0."""
    weinberg = div(atom("b2"), add(atom("b3"), atom("dim_G2")))
    frag = fragility_score(weinberg, 0.23122, INVARIANTS, TRANSCENDENTALS)
    assert frag == 0.0


def test_fragility_delta_cp():
    """delta_CP = 7*14 + 99 has integer leaf 7, so fragility > 0."""
    delta = add(mul(integer(7), atom("dim_G2")), atom("H_star"))
    frag = fragility_score(delta, 197.0, INVARIANTS, TRANSCENDENTALS)
    # Perturbing 7 to 6 or 8 gives 6*14+99=183 or 8*14+99=211
    # Neither is within 5% of 197 (187-207)
    # 183 < 187: not robust. 211 > 207: not robust.
    # So fragility should be 1.0 (both perturbations leave range)
    assert frag == 1.0


def test_composite_score_structure():
    """Composite score returns all expected fields."""
    weinberg = div(atom("b2"), add(atom("b3"), atom("dim_G2")))
    scores = composite_score(
        node=weinberg,
        y_exp=0.23122,
        sigma_y=0.00003,
        obs_class="B",
        invariants=INVARIANTS,
        transcendentals=TRANSCENDENTALS,
    )
    assert "err" in scores
    assert "comp" in scores
    assert "unnat" in scores
    assert "frag" in scores
    assert "total" in scores
    assert "predicted" in scores
    assert abs(scores["predicted"] - 0.230769) < 1e-4


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
