# TIER 3: Selection Coefficient (CONJECTURE)

## Status: CONJECTURE (open problem)

The exact value of κ = L²·λ₁/H* remains undetermined.

---

## The Question

Given:
- λ₁ = c₁/L² to c₂/L² (Tier 1: PROVEN)
- L² is constrained by H* (Tier 2: SUPPORTED)

What is the exact coefficient?

**GIFT claims:** κ = π²/14 = π²/dim(G₂)

**Status:** Conjecture, not proven

---

## The 14 vs 13 Problem

### Observations in Numerics

Different discretizations/normalizations give:
- κ ≈ π²/14 ≈ 0.705 (some setups)
- κ ≈ π²/13 ≈ 0.760 (other setups)

### Possible Explanations

**If 14:** κ = π²/dim(G₂)
- dim(G₂) = 14 (the holonomy group)
- Natural from representation theory

**If 13:** κ = π²/(dim(G₂) - 1)
- Could arise from "reduced" dimension
- Might be related to a quotient construction

**If neither:**
- The coefficient could depend on specific building blocks
- Might not have a "nice" closed form

---

## What Would Settle It

### Option A: Explicit G₂ Metric

1. Construct a fully explicit g_ij(x) on K7
2. Compute λ₁ numerically to high precision
3. Determine if λ₁ · L² · H* = π² · H*/14 or something else

**Challenge:** Full 7D eigenvalue computation is expensive.

### Option B: Canonical Normalization

1. Define a canonical way to normalize the TCS metric
2. Prove the coefficient is determined by this normalization
3. Compute it

**Challenge:** Multiple natural normalizations exist:
- Vol(M_L) = 1
- Vol(Y) = 1
- ||φ||_{L²} = 1 (G₂ 3-form)

### Option C: Variational Principle

1. Find a functional F[g] whose critical points give the selection
2. Prove F has a unique minimizer
3. Compute κ at the minimizer

**Challenge:** No canonical functional is known.

---

## The GIFT Ansatz

### Statement

**Conjecture 3.1 (GIFT Selection):**
$$\kappa = \frac{\pi^2}{\dim(G_2)} = \frac{\pi^2}{14}$$

### Motivation

1. **Dimensional analysis:** κ should involve dim(G₂) = 14
2. **π² from spectral theory:** First Neumann eigenvalue on [0,1] is π²
3. **Numerics:** Various computations give κ ≈ 0.70-0.71

### Equivalent Formulations

If κ = π²/14, then:
- L* = π√(H*/14) = π√(99/14) ≈ 8.354
- λ₁ = 14/99 ≈ 0.1414
- λ₁ · H* = 14

---

## Alternative Conjectures

### Conjecture 3.2 (Reduced Dimension)
$$\kappa = \frac{\pi^2}{13} = \frac{\pi^2}{\dim(G_2) - 1}$$

**Rationale:** The "1" might account for a gauge degree of freedom or a quotient by the center.

### Conjecture 3.3 (Betti Ratio)
$$\kappa = \frac{\pi^2 b_2}{b_3} = \frac{21\pi^2}{77}$$

**Rationale:** Direct topological formula without involving dim(G₂).

**Value:** ≈ 2.69 (seems too large)

### Conjecture 3.4 (Generic Coefficient)
$$\kappa = \frac{\pi^2}{c}$$

where c depends on specific TCS building blocks and is not universal.

---

## Discriminating Tests

### Test 1: Different Building Blocks

Compute κ for TCS with different ACyl CY3:
- Quintic + Quintic
- CI(2,2,2) + CI(2,2,2)
- Quintic + CI(2,2,2)

If κ varies, Conjecture 3.4 is favored.
If κ = π²/14 for all, Conjecture 3.1 is supported.

### Test 2: Numerical Precision

High-precision computation of λ₁ on explicit TCS:
- If λ₁·L²/π² = 1.000 ± 0.001, the coefficient is indeed π²
- If λ₁·L²/π² ≠ 1, need different formula

### Test 3: Analytic Continuation

Study λ₁ as function of continuous parameters:
- Moduli of K3
- Gluing angle
- ACyl deformation

Look for universal behavior in limits.

---

## Current Evidence

### For κ = π²/14

| Source | Evidence |
|--------|----------|
| Upper bound (test function) | c₂ = π² ± O(L⁻³) |
| GIFT predictions | sin²θ_W = 3/13 matches experiment |
| Dimensional analysis | 14 = dim(G₂) is natural |

### Against (or uncertain)

| Issue | Concern |
|-------|---------|
| Lower bound | c₁ only proven to be O(1), not = π² |
| Numerical tests | 1D model failed, no 7D validation |
| 14 vs 13 | Some normalizations suggest 13 |

---

## Path Forward

### Priority 1: Settle c₁ = c₂ = π²

The upper bound gives c₂ = π². Need to prove:
$$c_1 = \pi^2 - o(1)$$

This would establish λ₁ = π²/L² exactly (not just up to constants).

### Priority 2: Canonical Normalization

Define THE normalization for TCS metrics that:
- Is geometrically natural
- Fixes all scaling ambiguities
- Makes the coefficient computable

### Priority 3: Algebraic Proof of 14

Show that 14 appears through:
- Representation theory of G₂
- Character formulas
- Index theorem contributions

---

## Summary

### Status: CONJECTURE

The coefficient κ = π²/14 is:
- Plausible (dimensional analysis, GIFT consistency)
- Unproven (no rigorous derivation)
- Testable (needs explicit metric or better numerics)

### Open Problem

**Problem 3.5:** Determine whether κ = π²/dim(G₂) is universal for all TCS G₂ manifolds, or depends on building block data.

### Honest Statement

Until proven:

$$\kappa = \frac{\pi^2}{14 \pm 1}$$

with the exact value remaining an open question in G₂ spectral geometry.
