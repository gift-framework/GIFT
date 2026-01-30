# Laplacian Hierarchy: From Scalar to Yang-Mills

**Date**: January 2026
**Status**: Research Notes

---

## The Three Laplacians

```
┌──────────────────────────────────────────────────────────────────────┐
│                    LAPLACIAN HIERARCHY                                │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   Level 0: SCALAR LAPLACIAN  Δ₀                                      │
│   ─────────────────────────────                                      │
│   Acts on: Functions f : M → ℝ                                       │
│   Definition: Δ₀f = -div(grad f) = -gⁱʲ∇ᵢ∇ⱼf                        │
│   What we compute: K₇ spectral gap studies                           │
│   Status: ✓ Numerically validated                                    │
│                                                                       │
│                          ↓                                           │
│                                                                       │
│   Level 1: HODGE LAPLACIAN  Δₖ                                       │
│   ─────────────────────────────                                      │
│   Acts on: k-forms ω ∈ Ωᵏ(M)                                        │
│   Definition: Δₖ = dδ + δd  (d = exterior derivative, δ = d*)       │
│   Key case: k=1 (1-forms, related to gauge fields)                   │
│   Status: ✗ Not yet computed for K₇                                  │
│                                                                       │
│                          ↓                                           │
│                                                                       │
│   Level 2: CONNECTION LAPLACIAN  Δ_A                                 │
│   ─────────────────────────────────                                  │
│   Acts on: Connections A (gauge fields)                              │
│   Definition: Δ_A = D_A*D_A (covariant Laplacian)                   │
│   This IS Yang-Mills!                                                │
│   Status: ✗ Requires full gauge theory setup                         │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Why This Hierarchy Matters

### Clay Millennium Problem
The Yang-Mills mass gap conjecture asks about the **spectrum of Δ_A** (Level 2).

### Current K₇ Work
We compute **Δ₀** (Level 0) on a discretized approximation of K₇.

### The Bridge Question
**Does λ₁(Δ₀) relate to λ₁(Δ_A)?**

If yes, our numerical work has direct implications for Yang-Mills.
If no, we need to climb the hierarchy.

---

## Mathematical Relationships

### Bochner-Weitzenböck Formula
Relates Hodge Laplacian to scalar Laplacian + curvature:
```
Δ₁ = ∇*∇ + Ric
```
where Ric is the Ricci curvature operator.

For Ricci-flat manifolds (like Calabi-Yau): Δ₁ = ∇*∇
For G₂ manifolds: Ric = 0, so same simplification!

**Key insight**: On K₇ with G₂ holonomy, the Hodge Laplacian on 1-forms is closely related to the scalar Laplacian.

### Spectrum Comparison (Flat Case)

On T⁷ (flat torus):
```
λ₁(Δ₀) = 1      (scalar)
λ₁(Δ₁) = 1      (1-forms)
λ₁(Δ₂) = 1      (2-forms)
...
```
All equal! The spectrum is the same across all form degrees.

**Question for K₇**: Does this equality persist with G₂ holonomy?

---

## G₂ Holonomy Specifics

### Hodge Decomposition on K₇
```
Ω¹(K₇) = ker(Δ₁) ⊕ im(d) ⊕ im(δ)
        = H¹(K₇)  ⊕  exact  ⊕  coexact
```

For K₇: H¹ = 0 (first Betti number vanishes for compact G₂)

### Parallel Forms
G₂ holonomy admits:
- 1 parallel spinor (this is the "−1" in dim(G₂) - 1 = 13?)
- 1 parallel 3-form φ (the G₂ structure)
- 1 parallel 4-form ψ = *φ

**Speculation**: The parallel spinor may contribute a "−1" correction to spectral quantities.

---

## Numerical Path Forward

### Step 1: Validate Scalar (CURRENT)
- Calibrate Δ₀ pipeline on S³, T⁷ ✓
- Confirm K₇ spectral gap is well-defined

### Step 2: Hodge Laplacian on 1-forms (NEXT)
Tools:
- **PyDEC**: Discrete Exterior Calculus library
- **SEC**: Spectral Exterior Calculus
- Custom DEC implementation

Challenge: Need to discretize the exterior derivative d and codifferential δ.

### Step 3: Connection Laplacian (FUTURE)
Would require:
- Explicit gauge bundle over K₇
- Connection 1-forms as dynamical variables
- Full Yang-Mills functional

This is the "holy grail" but likely out of numerical reach.

---

## Alternative: Cheeger Bounds

Instead of climbing the Laplacian hierarchy, we can use **analytical bounds**.

### Cheeger Inequality
```
λ₁(Δ₀) ≥ h²/4
```
where h is the Cheeger (isoperimetric) constant.

### Cheeger-Buser (Reverse)
```
λ₁(Δ₀) ≤ Ch√(n-1)κ + h²
```
where κ is curvature bound.

### For G₂ Manifolds
If we can show h(K₇) ~ c/H* for some constant c, then:
```
λ₁ ≥ c²/(4H*²)
```
This gives a **rigorous lower bound** without computing Yang-Mills directly!

---

## Key References

### Hodge Laplacian Spectrum
- Colbois-Courtois: "A note on the first nonzero eigenvalue of the Laplacian acting on p-forms"
- Gallot-Meyer: "Opérateur de courbure et laplacien des formes différentielles"

### G₂ Geometry
- Joyce: "Compact Manifolds with Special Holonomy"
- Bryant-Salamon: "On the construction of some complete metrics with exceptional holonomy"

### Discrete Exterior Calculus
- Hirani: "Discrete Exterior Calculus" (PhD thesis, 2003)
- Desbrun et al.: "Discrete Differential Forms for Computational Modeling"

### Spectral Gap & Yang-Mills
- Uhlenbeck: "Connections with Lᵖ bounds on curvature"
- Donaldson-Kronheimer: "The Geometry of Four-Manifolds"

---

## Summary Table

| Laplacian | Acts On | K₇ Result | Status |
|-----------|---------|-----------|--------|
| Δ₀ (scalar) | Functions | λ₁×H* ≈ 13 | ✓ Computed |
| Δ₁ (Hodge 1-forms) | 1-forms | Unknown | TODO: PyDEC |
| Δ_A (connection) | Gauge fields | Unknown | Yang-Mills target |

---

## Open Questions

1. **Does λ₁(Δ₁) = λ₁(Δ₀) on K₇?**
   - True on flat spaces
   - Unknown for G₂ holonomy
   - Would simplify bridge to Yang-Mills

2. **What is h(K₇)?**
   - Cheeger constant from isoperimetric ratio
   - Bounds λ₁ analytically
   - No one has computed this for G₂ manifolds

3. **Why 13 instead of 14?**
   - Parallel spinor contribution?
   - Discretization artifact?
   - Calibration study will answer

---

*GIFT Spectral Gap Research Program — Theoretical Notes*
