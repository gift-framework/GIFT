---
title: "Home"
layout: default
---

# GIFT: Geometric Information Field Theory

**A zero-parameter framework deriving Standard Model constants from G₂ holonomy geometry.**

---

## At a Glance

| Property | Value |
|----------|-------|
| **Predictions** | 95 observables (33 Type I + 19 Type II + 21 Type III + 22 Type IV) |
| **Mean deviation** | 0.99% on 33 Type I (exact-target) relations (NuFIT 6.1 / PDG 2024 / Planck 2018 / CODATA 2022) |
| **Free parameters** | 0 (3 integer primitives: N=3, r₈=8, r₂=2) |
| **Lean 4 verification** | 143 files, 8391 build jobs, 0 sorry, 15 axioms (4 main-chain + 11 interval-arithmetic) (K7-Lean v3.4.29) |
| **Statistical significance** | p < 2×10⁻⁵ (σ > 4.2), unique among 3M+ configurations |
| **Monte Carlo null model** | P(algebraic) = 10⁻¹³⁸ over 3M+ formulas |

---

## Quick Links

| | |
|---|---|
| **New here?** | [Getting Started](Getting-Started.html): Pick your path |
| **Read papers** | [Paper Main Framework](Paper-Main-Framework.html) ([Paper Explicit G2 Metric](Paper-Explicit-G2-Metric.html)) [Paper Spectral Geometry](Paper-Spectral-Geometry.html) |
| **Browse predictions** | [Observable Reference](Observable-Reference.html), 95 observables with formulas |
| **Check proofs** | [Lean Formalization](Lean-Formalization.html), 140-conjunct certificate |

---

## Key Results

| Observable | GIFT Formula | Value | Exp. | Dev. |
|------------|-------------|-------|------|------|
| sin²θ_W | b₂/(b₃+dim(G₂)) = 21/91 | 3/13 | 0.2312 | 0.19% |
| N_gen | rank(E₈) − Weyl | 3 | 3 | exact |
| Q_Koide | dim(G₂)/b₂ = 14/21 | 2/3 | 0.6667 | 0.001% |
| α_s(M_Z) | √2/(dim(G₂)−p₂) | √2/12 | 0.1179 | 0.04% |
| δ_CP | dim(K₇)×dim(G₂)+H* | 197° | ~207-212° (NuFIT 6.1) | ~1σ |
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
| δ_CP = 197° (window [182°, 212°]) | DUNE | 2028–2040 | Awaiting |
| sin²θ_W = 3/13 | FCC-ee | 2040s | Awaiting |
| N_gen = 3 (no 4th gen) | LHC/FCC | Ongoing | Consistent |
| m_s/m_d = 20 | Lattice QCD | ~2030 | Consistent |

---

## Outreach

Accessible writing on the ideas behind GIFT lives on the [Substack blog](https://arithmon.substack.com/); see the [Site Map blog index](Site-Map.html#blog-posts) for the full list. Latest: *[The Singing Glass](https://arithmon.substack.com/p/the-singing-glass)*, when a Burgundy wine glass explains the hidden dimensions of the universe.

---

## Structure

This wiki consolidates GIFT documentation across three repositories:

- **[Arithmon/K7](https://github.com/Arithmon/K7)**: Theoretical documentation, papers, outreach
- **[Arithmon/K7-Lean](https://github.com/Arithmon/K7-Lean)**: Lean 4 proofs, Python package, blueprint
- **Zenodo deposits**: Archived data with DOIs

Browse the sidebar for full navigation, or see the [Site Map](Site-Map.html) for a complete index.

---

*The K₇ Framework v3.5 | [GitHub](https://github.com/Arithmon/K7) | [Core](https://github.com/Arithmon/K7-Lean) | [Blueprint](https://arithmon.github.io/K7-Lean/) | [Zenodo (Paper A)](https://doi.org/10.5281/zenodo.19892350) · [Paper B](https://doi.org/10.5281/zenodo.19893371) · [Paper C](https://doi.org/10.5281/zenodo.19708916) | MIT License*
