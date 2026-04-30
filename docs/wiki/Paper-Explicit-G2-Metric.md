---
title: "Paper Explicit G2 Metric"
layout: default
---

# Paper: Explicit G₂ Metric

**An Explicit Approximate G₂ Metric on a Compact TCS 7-Manifold with Certified Torsion-Free Completion**

*Brieuc de La Fournière (2026)*
[Full text (markdown)](https://github.com/gift-framework/GIFT/blob/main/publications/papers/markdown/g2_certified_neck.md) | [Zenodo DOI: 10.5281/zenodo.19892350](https://doi.org/10.5281/zenodo.19892350)

---

## Abstract

Constructs explicit 169-parameter Chebyshev-Cholesky metric on compact TCS K₇. Newton-Kantorovich certificate proves unique torsion-free G₂ metric g* exists within distance 4.86×10⁻⁶. Initial torsion ‖T‖ = 8.94×10⁻² reduced to 2.98×10⁻⁵ in 5 Joyce iterations (3000× reduction).

---

## Key Results

### Certification Chain

| Quantity | Value |
|----------|-------|
| Initial torsion ‖T‖₀ | 8.936×10⁻² |
| Final torsion ‖T‖₅ | 2.984×10⁻⁵ |
| Reduction factor | 2995× |
| NK contraction h | 6.65×10⁻⁸ |
| NK threshold | 0.5 |
| Safety margin | ×7.5M |
| Distance to exact metric | ≤ 4.86×10⁻⁶ |

### Metric Properties

| Property | Value |
|----------|-------|
| Parameters | 169 (168 Chebyshev + 1 ACyl decay) |
| det(g) | 65/32 (exact) |
| \|φ\|² | 42 (error < 10⁻¹⁴) |
| Holonomy | Hol(g*) = G₂ |
| Torsion class | 99.6% in W₃, \|dφ\|²/\|d*φ\|² = 1/5 |

### Eigenvalue Hierarchy

Three-scale structure:
- **Neck** (seam): λ₀ ≈ 6.8
- **T²** (fiber): λ₁,₆ ≈ 2.9
- **K3** (fiber): λ₂₋₅ ≈ 1.1

---

## Section Structure

1. **Introduction**: Context, objective, scope & claims
2. **The Manifold**: TCS construction, topology (b₂=21, b₃=77)
3. **The Metric**: Model hierarchy, coordinates, Chebyshev-Cholesky parametrization
4. **Norm Definitions & Domain**: Metric distance, torsion norms, NK norm
5. **Torsion Analysis**: Initial approximation, K3 verification, Gauss-Newton reduction
6. **Certification**: NK convergence, interval arithmetic, holonomy proof
7. **Geometric Invariants**, det(g)=65/32, |φ|²=42, Hol(g*)=G₂
8. **Discussion**: Limitations, comparison with prior work
9. **Reproducibility**: Data files, companion notebook (< 1 min runtime)

---

## Figures

- TCS visualization with torsion intensity coloring
- Atlas chart schematic
- Eigenvalue profile (three-scale hierarchy)
- Torsion convergence (log scale, 5 iterations)

---

## Related

- [Paper Main Framework](Paper-Main-Framework.html): Physics application
- [Paper S1 Foundations](Paper-S1-Foundations.html): TCS construction theory
- [Paper Spectral Geometry](Paper-Spectral-Geometry.html): Spectral analysis of this metric
- [For Geometers](For-Geometers.html): Computational pipeline overview
