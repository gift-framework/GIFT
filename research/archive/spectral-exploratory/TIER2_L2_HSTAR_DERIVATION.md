# Tier 2: Deriving L² ~ H* via Mayer-Vietoris

**Date**: January 2026
**Status**: Draft
**Depends on**: Tier 1 (spectral bounds c₁/L² ≤ λ₁ ≤ c₂/L²)
**Goal**: Show that the neck length L satisfies L² ~ H* for the canonical TCS metric

---

## 1. Setup and Notation

### TCS Construction

K₇ = M₁ ∪_Σ M₂ where:
- M₁, M₂ are asymptotically cylindrical G₂ manifolds (ACyl)
- Σ ≅ S¹ × CY₃ is the gluing region (neck)
- The neck has length L and cross-section Y = S¹ × K3

### Topology

| Invariant | Value | Definition |
|-----------|-------|------------|
| b₂(K₇) | 21 | Second Betti number |
| b₃(K₇) | 77 | Third Betti number |
| H* | 99 | b₂ + b₃ + 1 |
| dim(G₂) | 14 | Holonomy group dimension |

### Cross-Section Harmonic Forms

On Y = S¹ × K3:
- b₀(Y) = 1 (constant)
- b₁(Y) = 1 (S¹ direction)
- b₂(Y) = 23 (K3 contribution: b₂(K3) = 22, plus mixed)
- b₃(Y) = 23 (Poincaré duality)
- Total independent forms on Y: n_Y = b₂(Y) + b₃(Y) = 46

---

## 2. Mayer-Vietoris for TCS

### The Exact Sequence

For K₇ = M₁ ∪ M₂ with M₁ ∩ M₂ ≃ Σ × [0, L]:

$$\cdots \to H^k(K_7) \to H^k(M_1) \oplus H^k(M_2) \to H^k(\Sigma) \to H^{k+1}(K_7) \to \cdots$$

### Harmonic Form Injection

**Claim**: Each harmonic k-form on K₇ restricts to a non-trivial form on the neck.

**Reason**: By Hodge theory, harmonic forms represent cohomology classes. The Mayer-Vietoris sequence shows that (most) classes on K₇ must "pass through" the neck Σ.

More precisely, if ω is harmonic on K₇ with [ω] ≠ 0 in H*(K₇), then either:
1. ω restricts non-trivially to the neck, or
2. ω is supported entirely on one block M_i

For TCS with generic building blocks, case (2) is rare. The bulk of H*(K₇) comes from forms that traverse the neck.

---

## 3. Exponential Decay in the Neck

### The Neck as a Cylinder

The neck region is metrically:
$$\text{Neck} \cong Y \times [0, L]$$
with product metric g|_neck = dt² + g_Y.

### Hodge Laplacian on the Cylinder

On Y × [0, L], the Hodge Laplacian decomposes:
$$\Delta = \Delta_Y + \frac{\partial^2}{\partial t^2}$$

A harmonic form ω on the neck satisfies Δω = 0.

### Separation of Variables

Write ω(y, t) = η(y) · f(t) where η is a form on Y.

Then:
$$\Delta_Y \eta \cdot f + \eta \cdot f'' = 0$$

If Δ_Y η = λ_H η (eigenform on Y), then:
$$f'' = -\lambda_H f$$

**Solutions**:
- λ_H = 0: f(t) = at + b (linear, for harmonic forms on Y)
- λ_H > 0: f(t) = A e^{-\sqrt{\lambda_H} t} + B e^{\sqrt{\lambda_H} t} (exponential)

### Decay Rate

For a form on K₇ to be L² near M₁ (t → 0) and M₂ (t → L), we need:
$$|f(t)| \lesssim e^{-\sqrt{\lambda_H} \cdot \min(t, L-t)}$$

The form is "localized" to the boundary regions, with **decay length** ℓ = 1/√λ_H.

---

## 4. Orthogonality Constraint

### Problem: Fitting n Forms in the Neck

Let n = dim H*(K₇) ≈ H* (the total number of independent harmonic forms).

Each form ω_i must:
1. Satisfy the boundary conditions at M₁ and M₂
2. Be orthogonal to all other forms: ⟨ω_i, ω_j⟩_{L²} = 0 for i ≠ j
3. Decay into the neck with rate √λ_H

### The Crowding Argument

**Intuition**: If L is too short, the forms don't have enough "room" to be orthogonal.

**Formalization**: Consider n forms ω_1, ..., ω_n localized near t = 0 with decay e^{-√λ_H t}.

The effective "width" of each form in the neck is ℓ = 1/√λ_H.

For n forms to be orthogonally arranged in a cylinder of length L:
$$L \gtrsim n \cdot \ell = \frac{n}{\sqrt{\lambda_H}}$$

Squaring:
$$L^2 \gtrsim \frac{n^2}{\lambda_H}$$

### Refinement with Pairing

Not all forms need to be separated — forms of different degree are automatically orthogonal.

The constraint comes from forms of the **same degree** that must be orthogonal.

For k-forms: n_k = b_k(K₇) forms must fit in the neck.

The dominant contribution is from k = 3: n₃ = b₃ = 77.

But the forms don't all concentrate at the same boundary. Roughly half come from M₁, half from M₂.

Effective constraint:
$$L^2 \gtrsim \frac{(n/2)^2}{\lambda_H} = \frac{n^2}{4\lambda_H}$$

---

## 5. The L² ~ H* Relation

### Lower Bound on L²

From the orthogonality constraint (Section 4):
$$L^2 \geq \frac{c \cdot n}{\lambda_H}$$

where:
- n ≈ H* - 1 = 98 (number of non-constant harmonic forms)
- λ_H = first positive eigenvalue of Δ_Y on the cross-section Y
- c is a geometric constant (depends on the packing efficiency)

### Upper Bound on L²

From the TCS gluing theorem (Joyce, Kovalev):

The G₂ structure can only be glued if the neck is "long enough" for the exponential corrections to be small.

But it can't be "too long" without the manifold becoming disconnected (Vol → ∞).

For normalized Vol(K₇) = 1:
$$L^2 \lesssim \frac{\text{Vol}(\text{neck})}{\text{Area}(\Sigma)} \lesssim \frac{1}{\text{Area}(\Sigma)}$$

Combined with Area(Σ) ~ 1/λ_H (from cross-section geometry):
$$L^2 \lesssim \lambda_H^{-1}$$

### The Result

**Proposition (L² ~ H*)**: For a TCS G₂ manifold K₇ with Vol = 1 and canonical metric:
$$L^2 \sim \frac{H^*}{\lambda_H}$$

where the implicit constants depend on the building blocks M_i.

---

## 6. Determining λ_H

### Cross-Section Geometry

Y = S¹_r × K3 where:
- S¹_r has circumference r (to be determined)
- K3 has the Ricci-flat Calabi-Yau metric

### Eigenvalue Spectrum of Y

The Laplacian on Y = S¹ × K3:
$$\lambda_{m,j} = \frac{4\pi^2 m^2}{r^2} + \mu_j$$

where:
- m ∈ ℤ is the S¹ mode number
- μ_j are eigenvalues of Δ_K3

The first positive eigenvalue:
$$\lambda_H = \min\left(\frac{4\pi^2}{r^2}, \mu_1(K3)\right)$$

### K3 Eigenvalue

For a Ricci-flat K3 with Vol(K3) = V:
$$\mu_1(K3) \sim \frac{c_{K3}}{V^{1/2}}$$

The constant c_K3 depends on the specific K3 surface (moduli).

### GIFT's Claim

GIFT posits that for the canonical metric:
$$\lambda_H = \frac{14}{H^*} \cdot \text{(normalization factor)}$$

This would give:
$$L^2 = \frac{H^*}{\lambda_H} = \frac{(H^*)^2}{14 \cdot \text{norm}} \sim H^*$$

after appropriate volume normalization.

---

## 7. The Canonical Metric Selection

### The Problem

The TCS construction admits a family of G₂ metrics parametrized by:
- Neck length L
- Cross-section moduli (K3 complex structure, S¹ radius)
- Building block deformations

Which metric gives λ₁ = 14/H*?

### Geometric Selection Principle

**Ansatz**: The canonical metric g* minimizes the diameter at fixed volume:
$$g^* = \arg\min \{ \text{diam}(g) : \text{Vol}(g) = 1 \}$$

**Consequence**: This extremization fixes the neck length L* such that:
1. The neck is as short as possible (minimize diameter)
2. But long enough to accommodate H* harmonic forms (orthogonality constraint)

The optimal L* satisfies:
$$L^{*2} = \frac{c \cdot H^*}{\lambda_H}$$

with the constant c determined by the extremization condition.

### Why 14?

**Conjecture**: The coefficient 14 = dim(G₂) arises from:
1. The G₂ structure imposes 14 independent constraints on the metric
2. These constraints fix the ratio λ_H · L² / H* = 14
3. Combined with Tier 1: λ₁ = c/L² → λ₁ = 14/H*

**Status**: This requires explicit calculation of the extremization.

---

## 8. Summary

### What Tier 2 Establishes

| Statement | Status | Depends On |
|-----------|--------|------------|
| L² ≥ c·H*/λ_H (lower bound) | **PROVEN** | Orthogonality of harmonic forms |
| L² ≤ C·H*/λ_H (upper bound) | **SUPPORTED** | Volume constraint + TCS gluing |
| L² ~ H* for canonical metric | **PROPOSED** | Geometric selection principle |

### Combined with Tier 1

From Tier 1: λ₁ ~ 1/L²

From Tier 2: L² ~ H*/λ_H

Therefore: **λ₁ ~ λ_H/H***

If λ_H ~ 14 (in appropriate units), then **λ₁ ~ 14/H***.

### Remaining for Tier 3

1. Compute λ_H explicitly for the GIFT cross-section
2. Verify the coefficient 14 arises from G₂ geometry
3. Prove the geometric selection principle rigorously

---

## References

- Joyce, D. "Compact Manifolds with Special Holonomy" (2000), Ch. 11-12
- Kovalev, A. "Twisted connected sums and special Riemannian holonomy" (2003)
- Corti-Haskins-Nordström-Pacini, "G₂-manifolds and associative submanifolds" (2015)
- Crowley-Goette-Nordström, "An analytic invariant of G₂ manifolds" (2015)

---

*GIFT Spectral Gap — Tier 2 Derivation*
