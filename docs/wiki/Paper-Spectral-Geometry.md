---
title: "Paper Spectral Geometry"
layout: default
---

# Paper: Spectral Geometry

**Spectral Geometry of an Explicit G₂ Metric on a Compact 7-Manifold**

*Brieuc de La Fournière (2026)*
[Full text (markdown)](https://github.com/gift-framework/GIFT/blob/main/publications/papers/markdown/Spectral_Geometry.md) | [Zenodo DOI: 10.5281/zenodo.18920368](https://doi.org/10.5281/zenodo.18920368)

---

## Abstract

First explicit numerical computation of Kaluza-Klein spectrum on compact G₂ manifold. Adiabatic decomposition K₇ ≈ K3 × T² × I reduces 7D PDEs to 1D Sturm-Liouville ODEs. All Betti numbers confirmed spectrally: b₀=1, b₁=0, b₂=21, b₃=77. SD/ASD gap in K3 intersection matrix: 2210×.

---

## Key Results

### Scalar Spectrum

| Quantity | Value |
|----------|-------|
| Zero mode λ₀ | 3.47×10⁻¹³ (machine zero) |
| Spectral gap λ₁ | 0.1244 ± 0.0001 |
| Weyl law | λₙ = 0.125n², α = 1.998 (exact: 2.0) |

### Betti Number Confirmation

| Betti | Spectral | Gap ratio |
|-------|----------|-----------|
| b₀ = 1 | 1 zero mode | — |
| b₁ = 0 | no zero 1-forms | — |
| b₂ = 21 | 21 near-zero eigenvalues | 14,635× |
| b₃ = 77 | 77 near-zero eigenvalues | — |

### Mass Hierarchy (from SD/ASD gap)

| Ratio | Spectral | Exp. | Dev. |
|-------|----------|------|------|
| m₁/m₂ (τ/μ) | 16.5 | 16.82 | 1.9% |
| m₁/m₃ (τ/e) | 3400 | 3477 | 2.2% |
| SD/ASD gap | 2210× | — | — |

### Adiabatic Validation (5 tests)

| Test | Result |
|------|--------|
| Fiber flatness | < 0.002% max s-variation |
| Additivity error | 0.003–0.023% |
| Weyl law exponent | α = 1.998 (exact: 2.0) |
| T² isotropy | \|g^θθ − g^ψψ\| = 3×10⁻⁷ |
| K3 roundness | spread < 0.1% |

### KK Tower

- 1744 distinct eigenvalues (λ < 20)
- 4460 states with multiplicities
- Three-scale hierarchy: neck, T², K3

---

## Section Structure

1. **Introduction** — Context, adiabatic ansatz validation
2. **The Metric** — Chebyshev-Cholesky summary, certification
3. **Scalar Laplacian** — Spectral gap, Weyl law, KK tower
4. **Hodge Laplacian on 2-Forms** — b₂=21 confirmation, SD/ASD structure
5. **Harmonic Forms & Betti Numbers** — K3 forms, K₇ assembly, b₃=77
6. **1-Form Hodge Laplacian** — Spectral democracy to 10⁻⁴, b₁=0
7. **Singular Limits** — ADE singularity model, spectral stability
8. **Discussion** — G₂-MSSM, F-theory, string landscape
9. **Conclusion**

---

## Figures

1. Metric profiles: neck transition and ACyl decay
2. Scalar eigenvalue staircase (Weyl law)
3. First 5 scalar eigenfunctions
4. T² channel spectra (adiabatic additivity)
5. 2-form spectrum with 14,635× gap

---

## Related

- [Paper Explicit G2 Metric](Paper-Explicit-G2-Metric.html) — The metric this paper analyzes
- [Paper Main Framework](Paper-Main-Framework.html) — Physics predictions from this geometry
- [Paper S1 Foundations](Paper-S1-Foundations.html) — TCS construction theory
- [Observable Reference](Observable-Reference.html) — Predictions catalog
