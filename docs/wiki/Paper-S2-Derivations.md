---
title: "Paper S2 Derivations"
layout: default
---

# Paper: S2: Complete Derivations

**Supplement S2: Complete Derivations (Dimensionless): All 33 Dimensionless Predictions**

*Brieuc de La Fournière (2026)*
[Full text (markdown, v3.5)](https://github.com/Arithmon/K7/blob/main/publications/papers/markdown/k7_framework_3_5_S2_derivations.md) | [v3.3 archive on Zenodo: 10.5281/zenodo.18837071](https://doi.org/10.5281/zenodo.18837071)

> **v3.5 update.** Canonical content lives in [`GIFT_v3.5_S2_derivations.md`](https://github.com/Arithmon/K7/blob/main/publications/papers/markdown/k7_framework_3_5_S2_derivations.md). The v3.5 LaTeX/PDF is being recompiled with the GIFT branding template.

---

## Abstract

Provides complete algebraic derivations for all 33 dimensionless predictions from topological invariants (b₂, b₃, dim(G₂), etc.). 18 core relations VERIFIED in Lean 4; 15 extended predictions use topological formulas. Includes expression counts showing structural redundancy.

---

## Key Results

### Deviation Distribution

| Range | Count | % |
|-------|-------|---|
| Exact (0%) | 4 | 22% |
| < 0.01% | 3 | 17% |
| < 0.1% | 4 | 22% |
| < 0.5% | 7 | 39% |

### Expression Counts (Top Observables)

| Observable | # Expressions | Status |
|------------|--------------|--------|
| Q_Koide = 2/3 | 27 | CANONICAL |
| N_gen = 3 | 24+ | CANONICAL |
| sin²θ₁₂ᴾᴹᴺˢ = 4/13 | 21 | CANONICAL |
| sin²θ_W = 3/13 | 19 | ROBUST |
| m_H/m_t = 56/77 | 16 | ROBUST |

### Gauge/Holonomy Comparison

| Config | Mean Dev. | Factor |
|--------|-----------|--------|
| E₈×E₈ | 0.26% | 1× (optimal) |
| E₇×E₈ | 8.80% | 34× worse |
| SU(3)/CY holonomy | 4.43% | 17× worse |

---

## Section Structure

- **Part 0**: Derivation Philosophy: Inputs vs outputs, claims vs non-claims
- **Part I**: Foundations: Status classification, notation
- **Part II**: Foundational Theorems: N_gen=3, τ=3472/891, κ_T=1/61, det(g)=65/32
- **Part III**: Gauge Sector, sin²θ_W=3/13, α_s=√2/12
- **Part IV**: Lepton Sector: Q_Koide=2/3, m_τ/m_e=3477, m_μ/m_e=27^φ
- **Part V**: Quark Sector, m_s/m_d=20, m_b/m_t=1/42, CKM angles
- **Part VI**: Neutrino Sector, δ_CP=197°, mixing angles
- **Part VII**: Higgs & Cosmology, λ_H=√17/32, Ω_DE, n_s, h, σ₈
- **Part VIII**: Summary (18 VERIFIED + 15 extended)
- **Part IX**: Observable Catalog

---

## Related

- [Paper Main Framework](Paper-Main-Framework.html): Main paper
- [Paper S1 Foundations](Paper-S1-Foundations.html): Mathematical foundations
- [Observable Reference](Observable-Reference.html): Complete observable catalog
- [Lean Formalization](Lean-Formalization.html): Machine-checked proofs
