# Spectral Gap for TCS G₂ Manifolds

## Tiered Proof Structure

This proof is organized in 3 tiers with decreasing rigor:

| Tier | Content | Status | File |
|------|---------|--------|------|
| **1** | c₁/L² ≤ λ₁ ≤ c₂/L² | **THEOREM** | `TIER1_THEOREM.md` |
| **2** | L ↔ H* via harmonic forms | **SUPPORTED** | `TIER2_HARMONIC.md` |
| **3** | κ = π²/14 coefficient | **CONJECTURE** | `TIER3_CONJECTURE.md` |

---

## Tier 1: Spectral Bounds (THEOREM)

### Hypotheses

**(H1)** Neck structure: N_L ≃ [0,L] × Y with product metric + O(e^{-δL})
**(H2)** Volume: Vol(N_L) = L · Vol(Y) · (1 + o(1))
**(H3)** Cross-section gap: γ = λ₁(Y) > 0
**(H4)** Isoperimetric: Caps have controlled Poincaré constant
**(H5)** Regularity: Smooth metric

### Statement

**Theorem:** Under (H1)-(H5), there exist c₁, c₂ > 0 such that:

$$\frac{c_1}{L^2} \leq \lambda_1(M_L) \leq \frac{c_2}{L^2}$$

### Proof Sketch

- **Upper bound:** Test function cos(πt/L) → c₂ = π² + o(1)
- **Lower bound:** Localization + 1D Poincaré → c₁ > 0

**This is the core "indisputable" result.** Lean-friendly.

---

## Tier 2: L ↔ H* Connection (SUPPORTED)

The neck length L is constrained by the need to support H* = b₂ + b₃ + 1 harmonic forms.

### Idea

- Harmonic forms must extend through neck
- Neck "capacity" bounds number of quasi-harmonic modes
- Mayer-Vietoris gives b₂ = 21, b₃ = 77 for TCS

### Status

Strong geometric motivation, not rigorously proven.

**What would complete it:** Prove L² ~ f(H*, dim G₂) via Hodge theory.

---

## Tier 3: Coefficient (CONJECTURE)

### The Question

Given λ₁ ~ 1/L² and L ~ √H*, what is the exact coefficient κ?

**GIFT claims:** κ = π²/14 = π²/dim(G₂)

### The 14 vs 13 Problem

Different normalizations give κ ≈ π²/14 or κ ≈ π²/13.

**Resolution requires:**
- Explicit metric computation, OR
- Canonical normalization argument, OR
- Representation-theoretic proof

### Honest Status

$$\kappa = \frac{\pi^2}{14 \pm 1}$$

The exact value is **open**.

---

## Legacy Phase Documents

The original 6-phase proof provides detailed arguments:

| Phase | Content | File |
|-------|---------|------|
| 0 | Strategy | `ANALYTICAL_PROOF_STRATEGY.md` |
| 1 | Setup | `PHASE1_SETUP.md` |
| 2 | Cylinder | `PHASE2_CYLINDER.md` |
| 3 | Surgery | `PHASE3_SURGERY.md` |
| 4 | Asymptotics | `PHASE4_ASYMPTOTICS.md` |
| 5 | Errors | `PHASE5_ERRORS.md` |
| 6 | Selection | `PHASE6_SELECTION.md` |

These remain valid but should be read with the tiered status in mind.

---

## Summary Table

| Claim | Status | Evidence |
|-------|--------|----------|
| λ₁ ~ 1/L² | **THEOREM** | Variational bounds |
| Upper bound c₂ = π² | **THEOREM** | Test function explicit |
| Lower bound c₁ > 0 | **THEOREM** | Localization + Poincaré |
| Lower bound c₁ = π² | **OPEN** | Needs tighter analysis |
| L² ~ H*/dim(G₂) | **SUPPORTED** | Harmonic capacity |
| κ = π²/14 exactly | **CONJECTURE** | Dimensional analysis + GIFT |

---

## Path to Complete Proof

### For Tier 1 → Lean

1. Formalize neck manifold structure
2. Prove upper bound via test function
3. Prove lower bound via localization lemma

### For Tier 2 → Paper

1. Rigorous Hodge theory on TCS
2. Capacity bounds for harmonic forms
3. L ↔ H* inequality

### For Tier 3 → Resolution

1. Compute explicit G₂ metric on K7
2. High-precision eigenvalue calculation
3. Determine if 14, 13, or other

---

## References

- Mazzeo-Melrose: Surgery calculus
- Kovalev: TCS G₂ construction
- Cheeger: Spectral convergence
- Joyce: Compact manifolds with special holonomy

---

*CC BY 4.0 - GIFT Framework Research*
