# Tier 2 (Revised): The L² ~ H* Question

**Date**: January 2026
**Status**: OPEN / SUPPORTED (not proven)
**Depends on**: Tier 1 (spectral bounds, PROVEN)
**Goal**: Understand the relationship between neck length L and topological invariant H*

---

## ⚠️ Status Clarification

The original Tier 2 document claimed "L² ~ H* PROVEN". This was **overstated**.

**What is actually proven**: Tier 1 establishes λ₁ ~ 1/L² rigorously.

**What is conjectural**: The claim that L² ~ H* for the "canonical" metric.

This document revises the argument, identifies the gaps, and points to literature that could fill them.

---

## 1. The Scaling Bug (Fixed)

### Original Argument (Flawed)

Section 4 of the original document argued:
- Forms decay with length ℓ = 1/√λ_H
- n forms need room L ≳ n·ℓ = n/√λ_H
- Squaring: L² ≳ n²/λ_H

Then Section 5 claimed: L² ≥ c·n/λ_H (dropping the square on n)

**This is inconsistent.** The correct scaling from the naive argument is L² ≳ n²/λ_H.

### Corrected Statement

If we take the naive "packing" argument seriously:
$$L \gtrsim \frac{n}{\sqrt{\lambda_H}} \implies L^2 \gtrsim \frac{n^2}{\lambda_H}$$

For n ~ H* ~ 99, this gives L² ≳ 9801/λ_H, **not** L² ~ H*/λ_H.

### Resolution

The naive packing argument is **too pessimistic**. Not all n forms need to be linearly separated in the neck:
- Forms of different degree are automatically orthogonal
- Forms may have different transverse profiles on Y
- The correct statement requires a more sophisticated analysis

**Proper approach**: Use neck-stretching spectral theory (see Section 4).

---

## 2. What We Can Claim (Honestly)

### Tier 1 (PROVEN)

For TCS manifold with neck length L and Vol = 1:
$$\frac{c_1}{L^2} \leq \lambda_1 \leq \frac{c_2}{L^2}$$

This is a **theorem** (see MODEL_THEOREM_TIER1.md).

### Tier 2a: "Spectrum is controlled by L" (SUPPORTED by literature)

**Statement**: In the neck-stretching limit L → ∞, the small eigenvalues of Δ are determined by the neck geometry.

**Literature support**:
- Mazzeo-Melrose (1987): Spectral theory on manifolds with cylindrical ends
- Grieser-Jerison (1998): Asymptotics of eigenvalues on dumbbell domains
- Nordström (2008): Deformations of ACyl G₂ manifolds

**Status**: This is well-established in the literature. Not a new result.

### Tier 2b: "L = L(H*) for canonical metric" (CONJECTURAL)

**Statement**: There exists a "canonical" choice of TCS metric where L² ~ H*.

**This requires**:
1. Defining what "canonical" means (geometric selection principle)
2. Proving that this selection gives L proportional to √H*

**Status**: OPEN. No proof exists.

---

## 3. Gaps in the Original Argument

### Gap 1: "Most forms traverse the neck"

The claim that most harmonic forms on K₇ must pass through the neck is plausible but not proven.

**What we need**: A precise statement about how H*(K₇) decomposes via Mayer-Vietoris, and what fraction of forms have non-trivial restriction to the neck.

**Literature hint**: Kovalev-Lee (2011) analyze harmonic forms on TCS in detail.

### Gap 2: Orthogonality ⇒ length constraint

The "packing" argument is intuitive but not rigorous:
- It assumes forms are "localized" with width 1/√λ_H
- It doesn't account for the infinite-dimensional nature of form spaces
- The "almost orthogonality" regime is subtle

**What we need**: A proper functional analysis argument, or citation to existing results.

### Gap 3: Why L² ~ H* and not L² ~ H*² or something else?

Even if we accept that L depends on H*, the precise scaling is unclear.

The number 99 (harmonic forms) enters, but so does λ_H (cross-section eigenvalue).

**What we need**: An explicit calculation or a general principle that fixes the exponent.

---

## 4. Literature Route: Neck-Stretching Spectral Theory

### Key References

1. **Mazzeo-Melrose (1987)**: "The adiabatic limit, Hodge cohomology and Leray's spectral sequence"
   - Studies spectral theory on fibred boundaries
   - Relevant for understanding the cylindrical neck

2. **Grieser-Jerison (1998)**: "Asymptotics of the first nodal line"
   - Studies eigenvalue asymptotics on dumbbell-shaped domains
   - Shows λ₁ ~ 1/L² rigorously in simple cases

3. **Nordström (2008)**: "Deformations of asymptotically cylindrical G₂ manifolds"
   - Studies the TCS construction analytically
   - Provides the framework for g(L) metric family

4. **Crowley-Goette-Nordström (2015)**: "An analytic invariant of G₂ manifolds"
   - Studies spectral invariants (ν-invariant) on G₂ manifolds
   - Shows how spectrum relates to topology

### What This Literature Provides

- **Rigorous framework** for TCS spectral analysis
- **Eigenvalue asymptotics** in neck-stretching limit
- **Connection to topology** via index theory

### What We Should Do

Instead of re-inventing the wheel with heuristic "packing" arguments:

1. **Extract** the relevant results from Nordström et al.
2. **Apply** them to the K₇ case with specific Betti numbers
3. **Identify** what additional input (if any) is needed for L² ~ H*

---

## 5. Revised Tier 2 Structure

### Tier 2a: Spectral Control (ESTABLISHED)

**Theorem (from literature)**: For TCS manifold K = M₁ ∪_N M₂ with neck N ≅ Y × [0, L]:

The spectrum of Δ on K decomposes as:
- **Bulk spectrum**: eigenvalues from M₁, M₂ (bounded below independently of L)
- **Neck spectrum**: eigenvalues ~ c/L² (determined by neck geometry)

For large L, the spectral gap λ₁ is in the neck spectrum.

**Status**: This follows from Mazzeo-Melrose + Nordström. Can be cited.

### Tier 2b: Canonical Metric Selection (OPEN)

**Conjecture**: Among all TCS metrics g(L) on K₇, there is a "canonical" choice g* where:
$$L^2 = \frac{H^*}{\lambda_H}$$

**Possible selection principles**:
1. Minimize diameter at fixed volume
2. Minimize λ₁ · H* (scale-invariant)
3. Extremize some curvature functional

**Status**: No proof. Requires variational analysis on G₂ moduli space.

### Tier 2c: H* Enters via Topology (HEURISTIC)

**Heuristic**: The neck must be "long enough" to accommodate the topological complexity of K₇.

- K₇ has H* - 1 = 98 non-trivial harmonic forms
- These forms must satisfy matching conditions across the neck
- The matching conditions impose a lower bound on L

**Status**: Plausible but not formalized. Needs Mayer-Vietoris + Hodge theory argument.

---

## 6. Honest Summary

| Statement | Status | Evidence |
|-----------|--------|----------|
| λ₁ ~ 1/L² | **PROVEN** | Tier 1 Model Theorem |
| Neck controls spectrum for large L | **ESTABLISHED** | Mazzeo-Melrose, Nordström |
| L² ~ H* for some metric | **OPEN** | Heuristic (packing argument has scaling issue) |
| L² = H*/λ_H for canonical metric | **CONJECTURAL** | Requires selection principle |
| λ₁ = 14/H* | **CONJECTURAL** | Depends on Tier 2b + coefficient determination |

### The Gap

We can prove: **λ₁ = c/L²** for some c depending on neck geometry.

We cannot prove: **L² = f(H*)** for any specific function f.

The connection between neck length L and topology H* remains the **key open problem**.

---

## 7. Path Forward

### Option A: Literature Deep-Dive

Read Nordström (2008) and Crowley-Goette-Nordström (2015) carefully to extract:
- Explicit dependence of λ₁ on (L, Vol, Betti numbers)
- Any canonical metric choices in the literature

### Option B: Variational Formulation

Formulate "L² ~ H*" as a variational problem:
- Define functional F(g) = λ₁(g) · H*
- Study critical points of F on G₂ moduli space
- Show that critical points have L² ~ H*

### Option C: Numerical Verification

Compute λ₁ for explicit TCS examples:
- Use Nordström's metrics with varying L
- Check if there's a "special" L where λ₁ · H* has nice properties

### Recommendation

**Option A first**: The literature likely contains results we can use directly. No need to reprove standard spectral theory.

---

## 8. References

### Spectral Theory on Manifolds with Cylindrical Ends
- Mazzeo, R. & Melrose, R. "The adiabatic limit, Hodge cohomology and Leray's spectral sequence" (1987)
- Grieser, D. & Jerison, D. "Asymptotics of the first nodal line" (1998)
- Müller, W. "Eta invariants and manifolds with boundary" (1994)

### G₂ Geometry and TCS
- Kovalev, A. "Twisted connected sums and special Riemannian holonomy" (2003)
- Nordström, J. "Deformations of asymptotically cylindrical G₂ manifolds" (2008)
- Corti-Haskins-Nordström-Pacini, "G₂-manifolds and associative submanifolds" (2015)

### Spectral Invariants of G₂ Manifolds
- Crowley-Goette-Nordström, "An analytic invariant of G₂ manifolds" (2015)
- Kovalev-Lee, "K3 surfaces with non-symplectic involution and compact G₂ manifolds" (2011)

---

*GIFT Spectral Gap — Tier 2 Revised (Honest Assessment)*
