---
title: "Paper Main Framework"
layout: default
---

# Paper: Main Framework

**Geometric Information Field Theory: Topological Derivation of Standard Model Parameters from G₂ Holonomy Manifolds**

*Brieuc de La Fournière (2026)*

[**PDF (main, 44 pp.)**](https://github.com/gift-framework/GIFT/raw/main/publications/papers/pdf/gift_3.4_main.pdf) | [**DOI: 10.5281/zenodo.20070101**](https://doi.org/10.5281/zenodo.20070101) | [Markdown](https://github.com/gift-framework/GIFT/blob/main/publications/papers/markdown/GIFT_v3.4_main.md)

> **Companion papers (Zenodo):** [A: certified G₂](https://doi.org/10.5281/zenodo.19892350) · [B: spectral](https://doi.org/10.5281/zenodo.19893371) · [C: K3 NK](https://doi.org/10.5281/zenodo.19708916) · [D: Donaldson analytic](https://doi.org/10.5281/zenodo.20039066) · [v3.3 archive](https://doi.org/10.5281/zenodo.18837071)

---

## Abstract

Framework proposing that Standard Model dimensionless parameters emerge as topological invariants of a 7D G₂ holonomy manifold K₇ with Betti numbers (b₂=21, b₃=77) coupled to E₈×E₈ gauge structure. v3.4 catalogues 95 observables (35 Type I exact-target relations) with 0.39% mean deviation on Type I from experiment (PDG 2024 / NuFIT 6.0); 213 conjuncts certified in Lean 4 with 4 main-chain axioms. DUNE will test δ_CP = 197° falsification criterion.

---

## Key Results

| Observable | GIFT Formula | Value | Exp. | Dev. |
|------------|-------------|-------|------|------|
| sin²θ_W | b₂/(b₃+dim(G₂)) | 3/13 | 0.2312 | 0.19% |
| N_gen | rank(E₈)−Weyl | 3 | 3 | exact |
| Q_Koide | dim(G₂)/b₂ | 2/3 | 0.6667 | 0.001% |
| α_s(M_Z) | √2/12 | 0.1179 | 0.1179 | 0.04% |
| δ_CP | 7×14+99 | 197° | 177°±20° | 1σ |
| m_τ/m_e | 7+10×248+10×99 | 3477 | 3477.2 | 0.004% |
| n_s | ζ(11)/ζ(5) | 0.9649 | 0.9649 | 0.004% |

**Global** (v3.4): 95 observables in 4 types (33 Type I + 19 II + 21 III + 22 IV); 66 with experimental comparison; mean deviations 0.73% (Type I), 0.17% (Type II), 3.4% (Type III); 11 exact matches (<0.01%), 53 within 1%

---

## Section Structure

1. **Introduction**: Parameter problem, contemporary context, framework overview
2. **Mathematical Framework**: Octonions, E₈×E₈, K₇ hypothesis, G₂ structure
3. **Methodology & Epistemic Status**: Derivation principle, claims vs non-claims
4. **Derivation of 33 Predictions**: Gauge, lepton, quark, neutrino, Higgs, cosmological sectors
5. **Formal Verification & Statistics**: Lean 4 (290+ theorems), uniqueness among 192,349 alternatives
6. **G₂ Metric Program**: PINN atlas construction, metric quality results
7. **Falsifiable Predictions**, δ_CP via DUNE, fourth generation bounds
8. **Discussion**: M-theory connections, comparison with other approaches, limitations
9. **Conclusion**

---

## Statistical Validation

- (21,77) unique optimal among 192,349 configurations (p < 5×10⁻⁶)
- E₈×E₈ achieves 12.8× better agreement than next best gauge group
- G₂ holonomy achieves 13× better than Calabi-Yau (SU(3))
- Lean 4 verification: 290+ theorems, 0 sorry, 0 domain-specific axioms

---

## Related

- [Paper S1 Foundations](Paper-S1-Foundations.html): Mathematical foundations
- [Paper S2 Derivations](Paper-S2-Derivations.html): Complete derivations
- [Paper Explicit G2 Metric](Paper-Explicit-G2-Metric.html): Numerical G₂ metric
- [Paper Spectral Geometry](Paper-Spectral-Geometry.html): KK spectrum
- [Observable Reference](Observable-Reference.html): Full prediction catalog
- [Statistical Evidence](Statistical-Evidence.html), 7-component validation
