---
title: "Home"
layout: default
---

# GIFT — Geometric Information Field Theory

**A zero-parameter framework deriving Standard Model constants from G₂ holonomy geometry.**

---

## At a Glance

| Property | Value |
|----------|-------|
| **Predictions** | 89 observables (33 dimensionless + 56 structural/cosmological) |
| **Mean deviation** | 0.24% across 32 well-measured observables (PDG 2024 / NuFIT 6.0) |
| **Free parameters** | 0 |
| **Lean 4 verification** | 126 files, 2636 build jobs, 0 sorry, 38 axioms |
| **Statistical significance** | p < 2×10⁻⁵ (σ > 4.2), unique among 3M+ configurations |
| **Monte Carlo null model** | P(algebraic) = 10⁻¹³³ over 4.2M formulas |

---

## Quick Links

| | |
|---|---|
| **New here?** | [Getting Started](Getting-Started.html) — Pick your path |
| **Read papers** | [Paper Main Framework](Paper-Main-Framework.html) — [Paper Explicit G2 Metric](Paper-Explicit-G2-Metric.html) — [Paper Spectral Geometry](Paper-Spectral-Geometry.html) |
| **Browse predictions** | [Observable Reference](Observable-Reference.html) — 89 observables with formulas |
| **Check proofs** | [Lean Formalization](Lean-Formalization.html) — 127-conjunct certificate |

---

## Key Results

| Observable | GIFT Formula | Value | Exp. | Dev. |
|------------|-------------|-------|------|------|
| sin²θ_W | b₂/(b₃+dim(G₂)) = 21/91 | 3/13 | 0.2312 | 0.19% |
| N_gen | rank(E₈) − Weyl | 3 | 3 | exact |
| Q_Koide | dim(G₂)/b₂ = 14/21 | 2/3 | 0.6667 | 0.001% |
| α_s(M_Z) | √2/(dim(G₂)−p₂) | √2/12 | 0.1179 | 0.04% |
| δ_CP | dim(K₇)×dim(G₂)+H* | 197° | 177°±20° | 1.0σ |
| m_τ/m_e | 7+10×248+10×99 | 3477 | 3477.2 | 0.004% |
| Ω_DE | ln(2)×98/99 | 0.686 | 0.685 | 0.21% |

## Exact Algebraic Relations

All predictions derive from topological constants of a compact G₂ manifold K₇ with Betti numbers b₂ = 21, b₃ = 77, coupled to E₈×E₈ gauge structure (dim = 496):

```
sin²θ_W = b₂ / (b₃ + dim(G₂))  = 21/91  = 3/13
Q_Koide  = dim(G₂) / b₂          = 14/21  = 2/3
N_gen    = |PSL(2,7)| / fund(E₇) = 168/56 = 3
κ_T      = 1/(b₃ − dim(G₂) − p₂) = 1/61
α        = e^K (geometric coupling, zero free params)
```

## Falsification Tests

| Prediction | Experiment | Timeline | Status |
|------------|------------|----------|--------|
| δ_CP = 197° ± 5° | DUNE | 2028–2039 | Awaiting |
| sin²θ_W = 3/13 | FCC-ee | 2040s | Awaiting |
| N_gen = 3 (no 4th gen) | LHC/FCC | Ongoing | Consistent |
| m_s/m_d = 20 | Lattice QCD | ~2030 | Consistent |

---

## Structure

This wiki consolidates GIFT documentation across three repositories:

- **[gift-framework/GIFT](https://github.com/gift-framework/GIFT)** — Theoretical documentation, papers, outreach
- **[gift-framework/core](https://github.com/gift-framework/core)** — Lean 4 proofs, Python package, blueprint
- **Zenodo deposits** — Archived data with DOIs

Browse the sidebar for full navigation, or see the [Site Map](Site-Map.html) for a complete index.

---

*GIFT Framework v3.3 | [GitHub](https://github.com/gift-framework/GIFT) | [Core](https://github.com/gift-framework/core) | [Blueprint](https://gift-framework.github.io/core/) | [Zenodo](https://doi.org/10.5281/zenodo.18837071) | MIT License*
