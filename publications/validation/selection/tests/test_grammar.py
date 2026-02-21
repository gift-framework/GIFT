"""Tests for the grammar module: AST, complexity, canonicalization."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import math
import sympy
from selection.config import INVARIANTS, TRANSCENDENTALS
from selection.grammar.ast_node import (
    ASTNode, atom, integer, transcendental, op, add, sub, mul, div, sqrt, inv,
)
from selection.grammar.complexity import complexity, complexity_with_depth_penalty, complexity_class
from selection.grammar.canonicalize import canonical_value, are_equivalent


def test_ast_leaf_evaluation():
    """Atoms evaluate to their invariant values."""
    node = atom("b2")
    val = node.evaluate(INVARIANTS, TRANSCENDENTALS)
    assert val == 21

    node = integer(5)
    val = node.evaluate(INVARIANTS, TRANSCENDENTALS)
    assert val == 5

    node = transcendental("pi")
    val = node.evaluate(INVARIANTS, TRANSCENDENTALS)
    assert val == sympy.pi


def test_weinberg_angle():
    """sin²θ_W = b2/(b3 + dim_G2) = 21/91 = 3/13."""
    weinberg = div(atom("b2"), add(atom("b3"), atom("dim_G2")))
    val = weinberg.evaluate(INVARIANTS, TRANSCENDENTALS)
    assert val == sympy.Rational(21, 91)
    assert val == sympy.Rational(3, 13)
    assert abs(float(val) - 0.230769) < 1e-4


def test_koide():
    """Q_Koide = dim_G2/b2 = 14/21 = 2/3."""
    koide = div(atom("dim_G2"), atom("b2"))
    val = koide.evaluate(INVARIANTS, TRANSCENDENTALS)
    assert val == sympy.Rational(2, 3)


def test_delta_cp():
    """δ_CP = 7*dim_G2 + H* = 98 + 99 = 197."""
    delta = add(mul(integer(7), atom("dim_G2")), atom("H_star"))
    val = delta.evaluate(INVARIANTS, TRANSCENDENTALS)
    assert val == 197


def test_n_s():
    """n_s = ζ(11)/ζ(5)."""
    ns = div(transcendental("zeta11"), transcendental("zeta5"))
    val_f = ns.evaluate_float(INVARIANTS, TRANSCENDENTALS)
    assert abs(val_f - 0.9649) < 0.001


def test_to_str():
    """String representation is readable."""
    weinberg = div(atom("b2"), add(atom("b3"), atom("dim_G2")))
    s = weinberg.to_str()
    assert "b2" in s
    assert "b3" in s
    assert "dim_G2" in s


def test_json_roundtrip():
    """AST survives JSON serialization/deserialization."""
    weinberg = div(atom("b2"), add(atom("b3"), atom("dim_G2")))
    j = weinberg.to_json()
    restored = ASTNode.from_json(j)

    # Same structure
    assert restored.kind == weinberg.kind
    assert restored.value == weinberg.value

    # Same evaluation
    v1 = weinberg.evaluate_float(INVARIANTS, TRANSCENDENTALS)
    v2 = restored.evaluate_float(INVARIANTS, TRANSCENDENTALS)
    assert abs(v1 - v2) < 1e-10


def test_depth():
    """Depth computation."""
    leaf = atom("b2")
    assert leaf.depth() == 0

    binary = add(atom("b2"), atom("b3"))
    assert binary.depth() == 1

    nested = div(atom("b2"), add(atom("b3"), atom("dim_G2")))
    assert nested.depth() == 2


def test_complexity_basic():
    """Complexity scoring."""
    # Simple atom
    c = complexity(atom("b2"))
    assert c == 1.0  # primary invariant cost

    # Simple binary
    c = complexity(add(atom("b2"), atom("b3")))
    assert c == 1.0 + 1.0 + 1.0  # add + b2 + b3

    # Weinberg: div(b2, add(b3, dim_G2))
    weinberg = div(atom("b2"), add(atom("b3"), atom("dim_G2")))
    c = complexity(weinberg)
    # div(1.5) + b2(1) + add(1) + b3(1) + dim_G2(1) = 5.5
    assert abs(c - 5.5) < 0.01


def test_complexity_class():
    """Complexity class detection."""
    weinberg = div(atom("b2"), add(atom("b3"), atom("dim_G2")))
    assert complexity_class(weinberg) == "rational"

    alpha_s = div(transcendental("sqrt2"), sub(atom("dim_G2"), atom("p2")))
    assert complexity_class(alpha_s) == "transcendental"


def test_canonicalize_equivalent():
    """Two different ASTs for 3/13 are recognized as equivalent."""
    ast1 = div(atom("b2"), add(atom("b3"), atom("dim_G2")))
    ast2 = div(integer(3), integer(13))

    assert are_equivalent(ast1, ast2, INVARIANTS, TRANSCENDENTALS)


def test_canonicalize_different():
    """Different values are not equivalent."""
    ast1 = div(atom("b2"), atom("b3"))  # 21/77 = 3/11
    ast2 = div(atom("dim_G2"), atom("b2"))  # 14/21 = 2/3

    assert not are_equivalent(ast1, ast2, INVARIANTS, TRANSCENDENTALS)


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
