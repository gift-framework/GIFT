# Proper κ Test: Independent of GIFT Assumptions

**Goal**: Test whether κ = π²/14 without presupposing the answer.

---

## The Problem with Previous Test

### What We Did

1. Assumed λ₁ = 14/H* (GIFT prediction)
2. Computed κ = π²·H*/λ₁·H*² = π²/14
3. "Validated" that κ = π²/14 gives λ₁ = 14/H*

This is **circular** - we validated our own assumption.

---

## Proper Test Design

### Requirements

1. **Independent λ₁ measurement**: Compute λ₁(L) numerically for various L
2. **Independent L* selection**: Determine L* from a physical criterion (not λ₁ = 14/H*)
3. **Measure κ empirically**: κ = L*²/H*
4. **Check if κ ≈ π²/14**: Compare to candidate value

---

## Test 1: Numerical Eigenvalue Sweep

### Method

For a TCS model (or toy model of TCS):
1. Parameterize by neck length L ∈ [L_min, L_max]
2. Solve eigenvalue problem: find λ₁(L)
3. Plot λ₁ vs 1/L²
4. Extract coefficient: λ₁ ≈ c/L²

### Expected Outcome

If c ≈ π², then λ₁ ≈ π²/L² (supports heuristic).
If c ≠ π², we learn the actual coefficient.

### Implementation Note

This requires:
- Discretized Laplacian on TCS (FEM or spectral method)
- Or: Use DEC (Discrete Exterior Calculus) on mesh
- Or: PINN to approximate eigenfunction

---

## Test 2: Physical L* Selection

### Candidates for L* Criterion

| Criterion | How to Compute | L* Definition |
|-----------|----------------|---------------|
| Min torsion | ||T(φ_L)||² | ∂||T||²/∂L = 0 |
| Min correction | ||φ̃_L - φ_L|| | First L where < threshold |
| Hodge match | dim(ker Δ^(2)) | First L where = b₂ |
| Holonomy exact | ||Hol - G₂|| | First L where < threshold |

### Test Procedure

For each criterion:
1. Compute the functional F(L) for L ∈ [L_min, L_max]
2. Find L* = argmin F or threshold crossing
3. Compute κ_empirical = L*²/H*
4. Compare to π²/14

### Success Criterion

If **multiple independent criteria** give κ ≈ π²/14, this is evidence.
If they give different κ, the selection principle is criterion-dependent.

---

## Test 3: Multi-Manifold Universality

### Method

For G₂ manifolds with different (b₂, b₃):
1. Compute L*(M) using criterion C
2. Compute κ(M) = L*(M)²/H*(M)
3. Check if κ is constant across manifolds

### Manifold Sample

| Manifold | b₂ | b₃ | H* |
|----------|----|----|-----|
| K7 | 21 | 77 | 99 |
| CHNP_1 | 9 | 47 | 57 |
| CHNP_3 | 12 | 60 | 73 |
| Joyce_1 | 12 | 43 | 56 |

### Success Criterion

If κ(M) = π²/14 ± small error for all M: **universality confirmed**.
If κ varies with M: **universality fails**, need M-dependent selection.

---

## Test 4: Coefficient Check

### The Key Question

Is the coefficient 14 or something else?

### Sub-tests

| Hypothesis | Coefficient | κ value |
|------------|-------------|---------|
| dim(G₂) | 14 | 0.7050 |
| dim(SU(3)) | 8 | 1.234 |
| rank(E₈) | 8 | 1.234 |
| b₂ - 7 | varies | varies |

### Measurement

From numerical L*, compute:
```
c = π²·H*/L*²
```

Check if c is close to an integer (14, 8, etc.).

---

## Simplified Toy Model Test

### S¹ × S³ × S³ Model (Before TCS Upgrade)

For the simpler model:
- H* = 1 + 0 + 2 = 3 (for S³ × S³ × S¹)
- Check if selection principle applies

**Caveat**: This model is NOT G₂ holonomy, so it may not apply.

### TCS Toy Model

Build a simplified TCS:
- Two copies of T*S³ (cotangent bundle of 3-sphere)
- Glue along S³ × S² neck
- Vary neck length L

This is more tractable numerically.

---

## Implementation Plan

### Phase 1: Toy Model Eigenvalues

```python
# Discretize Laplacian on TCS toy model
# Sweep L from 5 to 20
# Compute λ₁(L) at each L
# Fit: λ₁ = c/L²
# Report c (compare to π²)
```

### Phase 2: Selection Functional

```python
# For same toy model
# Compute ||T(φ_L)||² as function of L
# Find L* = argmin
# Compute κ = L*²/H*
# Compare to π²/14
```

### Phase 3: Multi-Model

```python
# Repeat for 3-5 different TCS configurations
# Check if κ is constant
```

---

## Expected Outcomes

### Best Case

1. λ₁ ≈ π²/L² (heuristic confirmed)
2. L* from torsion minimization gives κ ≈ π²/14
3. κ is universal across models

**Conclusion**: Selection principle exists, κ = π²/14 validated independently.

### Intermediate Case

1. λ₁ ≈ c/L² with c ≠ π² but close
2. L* varies by criterion
3. κ varies by model

**Conclusion**: Selection is model/criterion-dependent, need refinement.

### Worst Case

1. λ₁ doesn't scale as 1/L²
2. No consistent L*
3. κ is meaningless

**Conclusion**: The TCS → spectral gap connection is more subtle than assumed.

---

## Summary

| Test | What It Checks | Independent of GIFT? |
|------|----------------|---------------------|
| Eigenvalue sweep | λ₁ ~ c/L² | ✅ Yes |
| L* from functional | κ = L*²/H* | ✅ Yes |
| Multi-manifold | κ universal? | ✅ Yes |
| Coefficient | c = 14 vs 8 vs other | ✅ Yes |

**All tests are independent of the GIFT prediction λ₁ = 14/H*.**

---

*Proper Test Design: 2026-01-26*
