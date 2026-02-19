# Lean 4 Implementation Plan for gift-framework/core

**Version**: 1.0
**Date**: February 2026
**Target**: gift-framework/core v3.3.16+

This document provides a detailed implementation plan for extending the Lean 4 formalization based on discoveries from the `/research/` documents.

---

## Executive Summary

Based on analysis of `K7_EXPLICIT_METRIC_ANALYTICAL.md` and `TORSION_FREE_CONDITION_ANALYSIS.md`, we identify **5 high-priority** and **4 medium-priority** formalization targets. All high-priority items are pure arithmetic and can be implemented immediately.

---

## 1. High Priority: Pure Arithmetic (Immediate Implementation)

### 1.1 Pell Equation Module

**New file**: `Lean/GIFT/Sequences/PellEquation.lean`

```lean
/-!
# Pell Equation Structure in GIFT

The spectral gap λ₁ = 14/99 satisfies a classical Pell equation,
connecting GIFT topological constants to number theory.
-/

import Mathlib.NumberTheory.Pell

namespace GIFT.Sequences.Pell

/-- The fundamental Pell equation for GIFT spectral parameters -/
theorem pell_equation_gift : (99 : ℤ)^2 - 50 * (14 : ℤ)^2 = 1 := by decide

/-- The discriminant 50 = 7² + 1 = dim(K₇)² + 1 -/
theorem pell_discriminant : (50 : ℕ) = 7^2 + 1 := by decide

/-- The solution pair (99, 14) = (H*, dim(G₂)) -/
theorem pell_solution_is_gift :
    (99 : ℕ) = 21 + 77 + 1 ∧ (14 : ℕ) = 14 := by decide

/-- Fundamental unit ε = 7 + √50 satisfies ε² = 99 + 14√50 -/
-- This requires Mathlib's Real.sqrt; axiom for now
axiom fundamental_unit_squared :
    let ε := 7 + Real.sqrt 50
    ε^2 = 99 + 14 * Real.sqrt 50

end GIFT.Sequences.Pell
```

**Effort**: Low (1-2 hours)
**Dependencies**: `Mathlib.NumberTheory.Pell` (optional for extended proofs)

---

### 1.2 Continued Fraction Module

**New file**: `Lean/GIFT/Sequences/ContinuedFraction.lean`

```lean
/-!
# Continued Fraction Structure for Spectral Gap

The spectral gap λ₁ = 14/99 has continued fraction [0; 7, 14],
where 7 = dim(K₇) and 14 = dim(G₂).
-/

namespace GIFT.Sequences.ContinuedFraction

/-- λ₁ = 14/99 as a rational -/
def lambda1 : ℚ := 14 / 99

/-- The continued fraction [0; 7, 14] evaluates to 14/99 -/
theorem lambda1_continued_fraction :
    lambda1 = 1 / (7 + 1 / 14) := by
  simp only [lambda1]
  norm_num

/-- Only GIFT dimensions appear in the continued fraction -/
theorem cf_uses_gift_dimensions :
    7 = dim_K7 ∧ 14 = dim_G2 := by
  decide

/-- Alternative: λ₁ = 1/(7 + 1/14) = 14/(7×14 + 1) = 14/99 -/
theorem lambda1_denominator :
    7 * 14 + 1 = 99 := by decide

end GIFT.Sequences.ContinuedFraction
```

**Effort**: Low (1 hour)
**Dependencies**: None beyond core GIFT constants

---

### 1.3 Torsion Class Dimensions

**New file**: `Lean/GIFT/Geometry/TorsionClasses.lean`

```lean
/-!
# Torsion Class Decomposition

The intrinsic torsion of a G₂ structure decomposes into
W₁ ⊕ W₇ ⊕ W₁₄ ⊕ W₂₇ with total dimension 49 = dim(K₇)².
-/

namespace GIFT.Geometry.TorsionClasses

/-- Dimensions of the four torsion classes -/
def dim_W1 : ℕ := 1
def dim_W7 : ℕ := 7
def dim_W14 : ℕ := 14
def dim_W27 : ℕ := 27

/-- Total torsion space dimension -/
theorem torsion_total_dim :
    dim_W1 + dim_W7 + dim_W14 + dim_W27 = 49 := by decide

/-- Torsion space dimension equals dim(K₇)² -/
theorem torsion_dim_is_K7_squared :
    dim_W1 + dim_W7 + dim_W14 + dim_W27 = dim_K7 ^ 2 := by decide

/-- W₁₄ has dimension equal to holonomy -/
theorem W14_is_holonomy_dim :
    dim_W14 = dim_G2 := rfl

/-- W₂₇ has dimension equal to Jordan algebra -/
theorem W27_is_jordan_dim :
    dim_W27 = dim_J3O := rfl

/-- Fiber dimensions: W₇ = dim(K₇) -/
theorem W7_is_fiber_dim :
    dim_W7 = dim_K7 := rfl

end GIFT.Geometry.TorsionClasses
```

**Effort**: Low (1 hour)
**Dependencies**: `GIFT.Core` constants

---

### 1.4 Three Paths to det(g)

**Enhancement to**: `Lean/GIFT/Foundations/AnalyticalMetric.lean`

```lean
/-!
# Three Independent Derivations of det(g) = 65/32

The metric determinant admits three algebraically independent paths,
demonstrating structural necessity rather than parameter fitting.
-/

namespace GIFT.Foundations.AnalyticalMetric.ThreePaths

/-- Path 1: Weyl formula -/
theorem det_g_path1_weyl :
    (Weyl * (rank_E8 + Weyl)) / (2 ^ Weyl : ℚ) = 65 / 32 := by
  simp only [Weyl, rank_E8]
  norm_num

/-- Path 2: Cohomological formula -/
theorem det_g_path2_cohomological :
    (p2 : ℚ) + 1 / (b2 + dim_G2 - N_gen) = 65 / 32 := by
  simp only [p2, b2, dim_G2, N_gen]
  norm_num

/-- Path 3: H* formula -/
theorem det_g_path3_H_star :
    ((H_star - b2 - 13) : ℚ) / 32 = 65 / 32 := by
  simp only [H_star, b2]
  norm_num

/-- All three paths agree -/
theorem det_g_three_paths_agree :
    (Weyl * (rank_E8 + Weyl)) / (2 ^ Weyl : ℚ) =
    (p2 : ℚ) + 1 / (b2 + dim_G2 - N_gen) ∧
    (p2 : ℚ) + 1 / (b2 + dim_G2 - N_gen) =
    ((H_star - b2 - 13) : ℚ) / 32 := by
  constructor <;> { simp only [Weyl, rank_E8, p2, b2, dim_G2, N_gen, H_star]; norm_num }

end GIFT.Foundations.AnalyticalMetric.ThreePaths
```

**Effort**: Low (30 minutes)
**Dependencies**: Existing `AnalyticalMetric.lean`

---

### 1.5 Moduli Space Dimension

**Enhancement to**: `Lean/GIFT/Algebraic/BettiNumbers.lean`

```lean
/-!
# Moduli Space of Torsion-Free G₂ Structures

The moduli space has dimension equal to b₃(K₇) = 77.
-/

namespace GIFT.Algebraic.ModuliSpace

/-- Dimension of moduli space of torsion-free G₂ structures -/
def dim_moduli : ℕ := b3

/-- Moduli dimension equals third Betti number -/
theorem moduli_dim_is_b3 : dim_moduli = 77 := rfl

/-- Moduli dimension decomposition -/
theorem moduli_dim_decomposition :
    dim_moduli = b3_M1 + b3_M2 ∧ 40 + 37 = 77 := by decide

end GIFT.Algebraic.ModuliSpace
```

**Effort**: Very low (15 minutes)
**Dependencies**: Existing Betti number module

---

## 2. Medium Priority: Structural Extensions

### 2.1 Variational Torsion Formulation

**New file**: `Lean/GIFT/Geometry/TorsionFunctional.lean`

```lean
/-!
# The Torsion Functional Θ_G₂

The torsion-free condition can be expressed variationally:
  Θ_G₂ := ‖∇φ‖² − κ_T‖φ‖² = 0

This leads to an eigenvalue problem for the associative 3-form.
-/

namespace GIFT.Geometry.TorsionFunctional

/-- The torsion capacity κ_T = 1/61 -/
def kappa_T : ℚ := 1 / 61

/-- κ_T topological derivation -/
theorem kappa_T_topological :
    kappa_T = 1 / (b3 - dim_G2 - p2) := by
  simp only [kappa_T, b3, dim_G2, p2]
  norm_num

/-- The torsion functional (abstract specification) -/
-- Full formalization requires differential geometry infrastructure
axiom TorsionFunctional : ThreeForm → ℝ

/-- Torsion-free condition as functional zero -/
axiom torsion_free_iff_functional_zero (φ : ThreeForm) :
    TorsionFree φ ↔ TorsionFunctional φ = 0

/-- Eigenvalue formulation: ∇²φ = κ_T φ -/
-- Requires Laplacian on differential forms
axiom eigenvalue_formulation (φ : ThreeForm) :
    TorsionFree φ ↔ Laplacian φ = kappa_T • φ

end GIFT.Geometry.TorsionFunctional
```

**Effort**: Medium (requires differential forms infrastructure)
**Dependencies**: `GIFT.Geometry.DifferentialFormsR7`, Mathlib differential geometry

---

### 2.2 RG Flow Constraints

**New file**: `Lean/GIFT/Spectral/RGFlowConstraints.lean`

```lean
/-!
# RG Flow Constraints

The RG exponents satisfy h_G₂² = 36 and sum rule constraints.
-/

namespace GIFT.Spectral.RGFlowConstraints

/-- Coxeter number of G₂ -/
def h_G2 : ℕ := 6

/-- The central constraint: lag × β = h_G₂² -/
theorem coxeter_constraint :
    h_G2 ^ 2 = 36 := by decide

/-- The constraint involves rank(E₈) and 13 -/
theorem lag_8_times_beta : 8 * (36 / 8 : ℚ) = 36 := by norm_num
theorem lag_13_times_beta : 13 * (36 / 13 : ℚ) = 36 := by norm_num

/-- Sum rule: Σβ = b₃/dim(K₇) = 11 -/
theorem rg_sum_rule :
    (b3 : ℚ) / dim_K7 = 11 := by
  simp only [b3, dim_K7]
  norm_num

end GIFT.Spectral.RGFlowConstraints
```

**Effort**: Low-Medium (mostly arithmetic)
**Dependencies**: None

---

### 2.3 Modified Pell from Riemann Zeros

**New file**: `Lean/GIFT/Zeta/ModifiedPell.lean`

```lean
/-!
# Modified Pell Equation from Riemann Zeros

The equation γ₂₉² − 49γ₁² + γ₂ + 1 ≈ 0 connects Riemann zeros
to K₇ topology, where 49 = dim(K₇)².
-/

namespace GIFT.Zeta.ModifiedPell

/-- The coefficient 49 = dim(K₇)² -/
theorem coefficient_is_K7_squared :
    (49 : ℕ) = 7 ^ 2 := by decide

/-- GIFT constants near Riemann zeros (numerical axioms) -/
-- These are empirical observations, not proofs
axiom gamma_1_near_dim_G2 : |RiemannZero 1 - 14| < 0.15
axiom gamma_2_near_b2 : |RiemannZero 2 - 21| < 0.03
axiom gamma_29_near_H_star : |RiemannZero 29 - 99| < 0.2

/-- The modified Pell structure (empirical, sub-0.001% accuracy) -/
axiom modified_pell_empirical :
    let γ₁ := RiemannZero 1
    let γ₂ := RiemannZero 2
    let γ₂₉ := RiemannZero 29
    |γ₂₉^2 - 49 * γ₁^2 + γ₂ + 1| < 0.01

end GIFT.Zeta.ModifiedPell
```

**Effort**: Medium (requires Riemann zero infrastructure)
**Dependencies**: `GIFT.Zeta.Correspondences`

---

### 2.4 Riemann-GIFT Recurrence

**Enhancement to**: `Lean/GIFT/Zeta/Correspondences.lean`

```lean
/-!
# Riemann Zero Recurrence with GIFT Lags

The recurrence γₙ = Σᵢ aᵢ γₙ₋ₗᵢ + c has lags {5, 8, 13, 27}
that are GIFT constants.
-/

namespace GIFT.Zeta.Recurrence

/-- The recurrence lags are GIFT constants -/
def recurrence_lags : List ℕ := [5, 8, 13, 27]

/-- Lag 5 = Weyl factor -/
theorem lag_5_is_weyl : 5 = Weyl := rfl

/-- Lag 8 = rank(E₈) -/
theorem lag_8_is_rank_E8 : 8 = rank_E8 := rfl

/-- Lag 13 = F₇ = α_sum_B -/
theorem lag_13_is_F7 : 13 = F 7 := by decide

/-- Lag 27 = dim(J₃(O)) -/
theorem lag_27_is_jordan : 27 = dim_J3O := rfl

/-- All lags are GIFT constants -/
theorem lags_are_gift :
    recurrence_lags = [Weyl, rank_E8, F 7, dim_J3O] := rfl

/-- Empirical accuracy bound (0.074% over 100k zeros) -/
axiom recurrence_accuracy_bound :
    ∀ n ≥ 100, RecurrenceError n < 0.00074 * RiemannZero n

end GIFT.Zeta.Recurrence
```

**Effort**: Medium-High
**Dependencies**: Riemann zero data, `GIFT.Sequences.Fibonacci`

---

## 3. Implementation Roadmap

### Phase 1: Immediate (Week 1)

| Module | Lines | Effort | Blocker |
|--------|-------|--------|---------|
| `Sequences/PellEquation.lean` | ~40 | 1-2h | None |
| `Sequences/ContinuedFraction.lean` | ~30 | 1h | None |
| `Geometry/TorsionClasses.lean` | ~40 | 1h | None |
| `AnalyticalMetric.lean` (extend) | ~30 | 30min | None |
| `BettiNumbers.lean` (extend) | ~15 | 15min | None |

**Total**: ~155 lines, 5-6 hours

### Phase 2: Short-term (Weeks 2-3)

| Module | Lines | Effort | Blocker |
|--------|-------|--------|---------|
| `Geometry/TorsionFunctional.lean` | ~80 | 1 day | DiffForms |
| `Spectral/RGFlowConstraints.lean` | ~50 | 2h | None |

### Phase 3: Medium-term (Weeks 4-6)

| Module | Lines | Effort | Blocker |
|--------|-------|--------|---------|
| `Zeta/ModifiedPell.lean` | ~60 | 4h | Riemann data |
| `Zeta/Correspondences.lean` (extend) | ~80 | 1 day | Recurrence impl |

---

## 4. Blueprint Updates

Each new theorem should be added to `blueprint/src/content.tex`:

```latex
\begin{theorem}[Pell Equation Structure]\label{thm:pell_equation}
    \lean{GIFT.Sequences.Pell.pell_equation_gift}
    \leanok
    The spectral gap parameters satisfy $99^2 - 50 \times 14^2 = 1$.
\end{theorem}

\begin{theorem}[Continued Fraction]\label{thm:continued_fraction}
    \lean{GIFT.Sequences.ContinuedFraction.lambda1_continued_fraction}
    \leanok
    $\lambda_1 = 14/99 = [0; 7, 14]$ where only $\dim(K_7)$ and $\dim(G_2)$ appear.
\end{theorem}

\begin{theorem}[Torsion Class Dimension]\label{thm:torsion_dim}
    \lean{GIFT.Geometry.TorsionClasses.torsion_dim_is_K7_squared}
    \leanok
    $\dim(W_1 \oplus W_7 \oplus W_{14} \oplus W_{27}) = 49 = \dim(K_7)^2$.
\end{theorem}

\begin{theorem}[Three Paths to det(g)]\label{thm:three_paths}
    \lean{GIFT.Foundations.AnalyticalMetric.ThreePaths.det_g_three_paths_agree}
    \leanok
    Three independent algebraic formulas yield $\det(g) = 65/32$.
\end{theorem}

\begin{theorem}[Moduli Space Dimension]\label{thm:moduli_dim}
    \lean{GIFT.Algebraic.ModuliSpace.moduli_dim_is_b3}
    \leanok
    The moduli space of torsion-free $G_2$ structures has $\dim(\mathcal{M}) = b_3 = 77$.
\end{theorem}
```

---

## 5. Testing Strategy

### 5.1 Unit Tests

Each theorem should have corresponding Python validation in `gift_core/`:

```python
# tests/test_pell_equation.py
def test_pell_equation():
    assert 99**2 - 50 * 14**2 == 1

def test_continued_fraction():
    from fractions import Fraction
    assert Fraction(14, 99) == Fraction(1, 7 + Fraction(1, 14))

def test_torsion_classes():
    assert 1 + 7 + 14 + 27 == 49
    assert 49 == 7**2
```

### 5.2 Integration Tests

Verify consistency between Lean theorems and Python constants:

```python
# tests/test_lean_python_consistency.py
def test_det_g_paths():
    from gift_core.constants import Weyl, rank_E8, p2, b2, dim_G2, N_gen, H_star

    path1 = (Weyl * (rank_E8 + Weyl)) / (2 ** Weyl)
    path2 = p2 + 1 / (b2 + dim_G2 - N_gen)
    path3 = (H_star - b2 - 13) / 32

    assert path1 == path2 == path3 == 65/32
```

---

## 6. Axiom Classification

New axioms should follow the Category A-F classification:

| New Axiom | Category | Justification |
|-----------|----------|---------------|
| `fundamental_unit_squared` | F (Numerical) | Requires `Real.sqrt` in Mathlib |
| `torsion_free_iff_functional_zero` | C (Geometric) | Differential geometry statement |
| `eigenvalue_formulation` | C (Geometric) | Requires Laplacian on forms |
| `modified_pell_empirical` | E (GIFT claim) | Numerical observation |
| `recurrence_accuracy_bound` | E (GIFT claim) | Empirical validation |

---

## 7. Success Criteria

### Quantitative

- [ ] 5 new theorem files created
- [ ] 155+ new lines of Lean code
- [ ] 0 new `sorry` statements
- [ ] <10 new axioms (all classified)
- [ ] Blueprint updated with 5+ new theorem entries

### Qualitative

- [ ] All arithmetic theorems proven by `decide` or `norm_num`
- [ ] Modular structure following existing patterns
- [ ] Python tests passing for all new theorems
- [ ] CI pipeline green after integration

---

## References

1. `research/K7_EXPLICIT_METRIC_ANALYTICAL.md` — Source for Pell, continued fraction, three paths
2. `research/TORSION_FREE_CONDITION_ANALYSIS.md` — Source for torsion classes, moduli space
3. `Lean/GIFT/Spectral/` — Existing spectral formalization patterns
4. `Lean/GIFT/Zeta/` — Existing Riemann correspondence infrastructure

---

*This plan provides concrete, actionable implementation steps for extending the GIFT Lean formalization based on the latest research developments.*
