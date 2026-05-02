"""Tests for the search module: enumeration, dedup, filters."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import math
from selection.search.enumerator import enumerate_formulas
from selection.search.filters import passes_range, passes_target_range
from selection.config import INVARIANTS, TRANSCENDENTALS


def test_passes_range_class_B():
    assert passes_range(0.5, "B")
    assert not passes_range(1.5, "B")
    assert not passes_range(-0.6, "B")  # outside margin=0.5
    assert passes_range(0.001, "B")
    assert passes_range(-0.1, "B")  # within margin=0.5


def test_passes_target_range():
    assert passes_target_range(0.23, 0.23122, margin=0.5)
    assert not passes_target_range(0.5, 0.23122, margin=0.5)


def test_enumerate_class_A():
    """Class A enumeration for N_gen=3 should find small formulas."""
    formulas = enumerate_formulas(
        obs_class="A",
        target_value=3.0,
        max_complexity=8.0,
        max_depth=2,
        invariant_set="primary",
    )
    # Should find at least N_gen itself (atom)
    values = [v for v, _ in formulas]
    assert any(abs(v - 3.0) < 1e-10 for v in values), \
        f"N_gen=3 not found in {len(formulas)} formulas. Values near 3: {[v for v in values if abs(v-3) < 1]}"

    # Search space should be manageable
    assert len(formulas) < 5000, f"Too many formulas for class A: {len(formulas)}"


def test_enumerate_class_B_weinberg():
    """Class B enumeration should find the Weinberg angle formula."""
    formulas = enumerate_formulas(
        obs_class="B",
        target_value=0.230769,
        max_complexity=12.0,
        max_depth=3,
        invariant_set="primary",
    )

    # Should find 3/13 = 0.230769...
    values = [v for v, _ in formulas]
    assert any(abs(v - 3/13) < 1e-8 for v in values), \
        f"Weinberg 3/13 not found in {len(formulas)} formulas"


def test_enumerate_class_B_koide():
    """Class B enumeration should find the Koide formula."""
    formulas = enumerate_formulas(
        obs_class="B",
        target_value=0.666667,
        max_complexity=12.0,
        max_depth=3,
        invariant_set="primary",
    )

    values = [v for v, _ in formulas]
    assert any(abs(v - 2/3) < 1e-8 for v in values), \
        f"Koide 2/3 not found in {len(formulas)} formulas"


def test_no_duplicate_values():
    """Deduplication should prevent exact duplicates."""
    formulas = enumerate_formulas(
        obs_class="B",
        target_value=0.230769,
        max_complexity=8.0,
        max_depth=2,
        invariant_set="primary",
    )

    values = [round(v, 12) for v, _ in formulas]
    # Allow some near-duplicates due to float precision
    unique_count = len(set(values))
    assert unique_count >= len(values) * 0.95, \
        f"Too many duplicates: {len(values)} total, {unique_count} unique"


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
