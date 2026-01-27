# ERRATUM: Rigorous Status of κ = π²/14

**Date**: 2026-01-26
**Status**: CORRECTION

---

## What Was Overclaimed

The previous documents marked several items as "PROVEN" or "DISCOVERED" that are actually:

| Claim | Previous Status | Correct Status |
|-------|-----------------|----------------|
| λ₁ = π²/L² exactly | "PROVEN (separation of variables)" | **CONJECTURAL** (needs neck-stretching) |
| κ = π²/14 | "DISCOVERED" | **CANDIDATE VALUE** |
| λ₁·H* = 14 universal | "VALIDATED" | **HYPOTHESIS** (circular test) |

---

## Tier Classification (Corrected)

### Tier 1: Proven Bounds ✅

**Statement**: For TCS G₂-manifolds with neck length L >> 1:
```
c/L² ≤ λ₁ ≤ C/L²
```

**Status**: PROVABLE via Cheeger inequality (lower) and test functions (upper).

**References**: Standard spectral geometry, Cheeger-Buser.

### Tier 1.5: Asymptotic Behavior ⚠️

**Statement**: The lowest eigenmode is neck-dominated for large L:
```
λ₁(L) ~ π²/L² + O(e^{-δL})  as L → ∞
```

**Status**: PLAUSIBLE but requires:
- Proper neck-stretching/analytic surgery framework
- Hypotheses on cross-section spectrum (spectral gap of K3 × S¹)
- Exponential decay estimates

**NOT just "separation of variables"** - the TCS is a glued manifold, not a product.

**References**: Langlais (2024) for spectral density; needs adaptation for ground state.

### Tier 2: Selection Principle ❓

**Statement**: There exists a canonical L* such that:
```
L*² = κ · H*  where κ = π²/14
```

**Status**: **OPEN CONJECTURE** - requires:
1. An explicit functional F(L) whose minimum is L*
2. Proof that ∂F/∂L = 0 implies L² ∝ H*
3. Determination of the coefficient κ

**The value κ = π²/14 is a CANDIDATE**, not a discovery.

---

## The Circular Validation Problem

### What the Previous Test Did

1. **Assumed** λ₁ = 14/H* (the GIFT prediction)
2. **Computed** κ = π²/λ₁ · (1/H*) = π²/(14/H*) · (1/H*) = π²/14
3. **Validated** that κ = π²/14 gives λ₁ = 14/H*

This is **tautological** - we validated our own assumption.

### What a Proper Test Needs

1. **Independent computation** of λ₁(L) for various L (numerical eigenvalue solver)
2. **Independent determination** of L* from a physical principle
3. **Comparison** of λ₁(L*) with 14/H*

---

## Corrected Claims

### We CAN Claim (Solid)

1. TCS construction exists and gives (b₂, b₃) = (21, 77) for specific building blocks
2. Spectral bounds λ₁ ~ 1/L² hold (Tier 1)
3. The formula λ₁ = dim(G₂)/H* is **algebraically consistent** with other GIFT predictions
4. The numerical identity b₂ - 7 = dim(G₂) = 14 holds for K7

### We CANNOT Claim (Yet)

1. ~~λ₁ = π²/L² exactly~~ → Need neck-stretching proof
2. ~~κ = π²/14 is "discovered"~~ → It's a candidate value
3. ~~The selection principle is validated~~ → The test was circular

---

## Path Forward

### To Establish κ = π²/14

Need one of:

**Option A: Variational Principle**
- Define F(L) = torsion energy, mismatch functional, etc.
- Show F has minimum at L* with L*² ∝ H*
- Compute coefficient

**Option B: Direct Numerical Computation**
- Solve eigenvalue problem on TCS numerically
- Find L* where some physical criterion is satisfied
- Measure κ = L*²/H*

**Option C: Analytic Surgery**
- Apply Langlais-type spectral convergence to TCS
- Derive λ₁ asymptotics with explicit constants
- Relate to topological invariants

---

## Updated Status Table

| Component | Status | Confidence |
|-----------|--------|------------|
| TCS construction | PROVEN | High |
| Betti numbers | PROVEN | High |
| λ₁ ~ 1/L² bounds | PROVABLE | High |
| λ₁ = π²/L² + O(e^{-δL}) | CONJECTURAL | Medium |
| κ = π²/14 | CANDIDATE | Low (needs test) |
| Selection functional | OPEN | Unknown |

---

*Erratum: 2026-01-26*
*This corrects overclaims in previous documents*
