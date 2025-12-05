# Geometric Information Field Theory v2.3

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-2.3-green.svg)](CHANGELOG.md)
[![Lean 4 Verified](https://img.shields.io/badge/Lean_4-Verified-blue)](https://github.com/gift-framework/core)
[![Coq Verified](https://img.shields.io/badge/Coq_8.18-Verified-orange)](https://github.com/gift-framework/core)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gift-framework/GIFT/main)

## Overview

| Metric | Value |
|--------|-------|
| **Precision** | 0.128% mean deviation across 39 observables |
| **Parameters** | Zero continuous adjustable (all structurally determined) |
| **Formally verified relations** | **35 proven** in Lean 4 + Coq (dual verification, zero axioms) |
| **Key results** | sin²θ_W = 3/13, κ_T = 1/61, det(g) = 65/32, τ = 3472/891, δ_CP = 197° |

The **Geometric Information Field Theory (GIFT)** derives Standard Model parameters from E₈×E₈ exceptional Lie algebras via dimensional reduction **E₈×E₈ → AdS₄×K₇ → Standard Model**. Version 2.3 achieves the **zero-parameter paradigm** with **formal verification**: all quantities derive from fixed topological structure, with **35 exact relations machine-verified** via both **Lean 4** and **Coq** proof assistants.

## Formal Verification (Lean 4 + Coq)

All 35 exact relations are **independently verified** in both **Lean 4** and **Coq**, providing dual proof-assistant validation (13 original + 12 topological extension + 10 Yukawa duality).

### Mathematical Core Repository

The formal proofs are maintained in a dedicated repository:

**[gift-framework/core](https://github.com/gift-framework/core)** — Exact rational and integer relations formally verified in two independent proof assistants.

| Proof Assistant | Modules | Status |
|-----------------|---------|--------|
| **Lean 4** (Mathlib 4.14+) | 17 modules | **0 sorry** · **0 domain axioms** |
| **Coq 8.18** | 21 modules | **0 Admitted** · **0 explicit axioms** |

The `core` repository contains:
- Complete Lean 4 formalization (Algebra, Geometry, Topology, Relations, Certificate)
- Complete Coq formalization (parallel structure)
- **K₇ metric pipeline** (giftpy v1.3.0) — G₂ geometry, harmonic forms, Yukawa extraction, Yukawa duality
- Continuous integration and verification

> **Note**: The original proofs were developed in this repository and have been migrated to `gift-framework/core` for independent verification. Historical versions are preserved in [`legacy/formal_proofs_v23_local/`](legacy/formal_proofs_v23_local/).

**Computational Validation:**
- Physics-informed neural networks (PINN) with 1M+ training samples
- Monte Carlo uncertainty propagation (10⁵ configurations)
- Mean precision: 0.128% across 39 observables

## Quick Start

### Local Installation

```bash
git clone https://github.com/gift-framework/GIFT.git
cd GIFT
pip install -r requirements.txt
```

**Requirements**: Python 3.11+

## Key Results

### 35 Lean-Verified Exact Relations

#### Original 13 Relations

| Relation | Value | Formula | Status |
|----------|-------|---------|--------|
| sin²θ_W | 3/13 | b₂/(b₃ + dim(G₂)) | **PROVEN (Lean + Coq)** |
| τ | 3472/891 | (496×21)/(27×99) | **PROVEN (Lean + Coq)** |
| det(g) | 65/32 | Topological formula | **PROVEN (Lean + Coq)** |
| κ_T | 1/61 | 1/(b₃ - dim(G₂) - p₂) | **PROVEN (Lean + Coq)** |
| δ_CP | 197° | 7×dim(G₂) + H* | **PROVEN (Lean + Coq)** |
| Q_Koide | 2/3 | dim(G₂)/b₂ | **PROVEN (Lean + Coq)** |
| m_s/m_d | 20 | p₂² × Weyl | **PROVEN (Lean + Coq)** |
| m_τ/m_e | 3477 | dim(K₇) + 10×dim(E₈) + 10×H* | **PROVEN (Lean + Coq)** |
| λ_H | √17/32 | √(dim(G₂)+N_gen)/2⁵ | **PROVEN (Lean + Coq)** |
| H* | 99 | b₂ + b₃ + 1 | **PROVEN (Lean + Coq)** |
| p₂ | 2 | dim(G₂)/dim(K₇) | **PROVEN (Lean + Coq)** |
| N_gen | 3 | rank(E₈) - Weyl | **PROVEN (Lean + Coq)** |
| E₈×E₈ | 496 | 2 × dim(E₈) | **PROVEN (Lean + Coq)** |

#### Topological Extension (12 New Relations)

| Relation | Value | Formula | Status |
|----------|-------|---------|--------|
| α_s denom | 12 | dim(G₂) - p₂ | **PROVEN (Lean + Coq)** |
| γ_GIFT | 511/884 | (2·rank(E₈) + 5·H*) / (10·dim(G₂) + 3·dim(E₈)) | **PROVEN (Lean + Coq)** |
| δ penta | 25 | Weyl² (pentagonal structure) | **PROVEN (Lean + Coq)** |
| θ₂₃ | 85/99 | (rank(E₈) + b₃) / H* | **PROVEN (Lean + Coq)** |
| θ₁₃ denom | 21 | b₂ (Betti number) | **PROVEN (Lean + Coq)** |
| α_s² denom | 144 | (dim(G₂) - p₂)² | **PROVEN (Lean + Coq)** |
| λ_H² | 17/1024 | (dim(G₂) + N_gen) / 32² | **PROVEN (Lean + Coq)** |
| θ₁₂ factor | 12775 | Weyl² × γ_num | **PROVEN (Lean + Coq)** |
| m_μ/m_e base | 27 | dim(J₃(O)) | **PROVEN (Lean + Coq)** |
| n_s indices | 11, 5 | D_bulk, Weyl_factor | **PROVEN (Lean + Coq)** |
| Ω_DE frac | 98/99 | (H* - 1) / H* | **PROVEN (Lean + Coq)** |
| α⁻¹ base | 137 | (dim(E₈) + rank(E₈))/2 + H*/11 | **PROVEN (Lean + Coq)** |

#### Yukawa Duality (10 New Relations - v1.3.0)

The Extended Koide formula exhibits a **duality** between two α² structures that are both topologically determined:

| Structure | α² values | Sum | Product+1 | Physical meaning |
|-----------|-----------|-----|-----------|------------------|
| **A** (Topological) | {2, 3, 7} | 12 = gauge_dim | 43 = visible | K3 signature origin |
| **B** (Dynamical) | {2, 5, 6} | 13 = rank+Weyl | 61 = κ_T⁻¹ | Exact mass fit |

| Relation | Value | Formula | Status |
|----------|-------|---------|--------|
| α²_A sum | 12 | 2 + 3 + 7 = dim(SM gauge) | **PROVEN (Lean + Coq)** |
| α²_A prod+1 | 43 | 2×3×7 + 1 = visible_dim | **PROVEN (Lean + Coq)** |
| α²_B sum | 13 | 2 + 5 + 6 = rank(E₈) + Weyl | **PROVEN (Lean + Coq)** |
| α²_B prod+1 | 61 | 2×5×6 + 1 = κ_T⁻¹ | **PROVEN (Lean + Coq)** |
| Duality gap | 18 | 61 - 43 = p₂ × N_gen² | **PROVEN (Lean + Coq)** |
| α²_up (B) | 5 | dim(K₇) - p₂ = Weyl | **PROVEN (Lean + Coq)** |
| α²_down (B) | 6 | dim(G₂) - rank(E₈) = 2×N_gen | **PROVEN (Lean + Coq)** |
| visible_dim | 43 | b₃ - hidden_dim | **PROVEN (Lean + Coq)** |
| hidden_dim | 34 | b₃ - visible_dim | **PROVEN (Lean + Coq)** |
| Jordan gap | 27 | 61 - 34 = dim(J₃(𝕆)) | **PROVEN (Lean + Coq)** |

Complete proofs: [gift-framework/core](https://github.com/gift-framework/core) | Paper proofs: [Supplement S4](publications/markdown/S4_complete_derivations_v23.md)

### Precision by Sector

| Sector | Observables | Best Result |
|--------|-------------|-------------|
| Gauge | 3 | α_s = √2/12 (0.04%) |
| Neutrino | 4 | δ_CP = 197° (0.00%) |
| Lepton | 4 | Q_Koide = 2/3 (0.001%) |
| Quark ratios | 10 | m_s/m_d = 20 (0.00%) |
| CKM | 6 | Mean 0.11% |
| Cosmology | 2 | n_s = ζ(11)/ζ(5) (0.00%) |

Full tables: [Main Paper Section 8](publications/markdown/gift_2_3_main.md)

## Documentation

### Reading Path

| Time | Document | Description |
|------|----------|-------------|
| 5 min | [Publications README](publications/README.md) | Executive briefing |
| 30 min | [Main Paper](publications/markdown/gift_2_3_main.md) | Complete framework |
| Deep dive | [Supplements S1-S7](publications/markdown/) | Mathematical details |

### Key Documents

- **[Main Paper](publications/markdown/gift_2_3_main.md)** - Complete theoretical framework (~1400 lines)
- **[Observable Reference](publications/references/GIFT_v23_Observable_Reference.md)** - All 39 observables
- **[Formal Proofs](https://github.com/gift-framework/core)** - Machine-verified in Lean 4 + Coq (gift-framework/core)
- **[Glossary](docs/GLOSSARY.md)** - Technical terms
- **[FAQ](docs/FAQ.md)** - Common questions
- **[Philosophy](docs/PHILOSOPHY.md)** - The philosophy behind GIFT

### Mathematical Supplements (S1-S7)

| Supplement | Title | Description |
|------------|-------|-------------|
| **[S1](publications/markdown/S1_mathematical_architecture_v23.md)** | Mathematical Architecture | E₈ exceptional Lie algebra foundations: root system (240 roots), Weyl group, Cartan matrix, Dynkin diagram. Establishes branching E₈ → E₇ → E₆ → SO(10) → SU(5) for Standard Model embedding. |
| **[S2](publications/markdown/S2_K7_manifold_construction_v23.md)** | K₇ Manifold Construction | Twisted connected sum (TCS) construction of the compact 7-manifold with G₂ holonomy. Derives Betti numbers b₂=21, b₃=77 via Mayer-Vietoris. Physics-informed neural networks validate metric invariants with Lean 4 formal verification. |
| **[S3](publications/markdown/S3_torsional_dynamics_v23.md)** | Torsional Dynamics | Torsion tensor from G₂ 3-form non-closure. Derives geodesic flow equation and connection to renormalization group. Establishes κ_T = 1/61 (topological) and ultra-slow flow velocity |v| ~ 0.015. |
| **[S4](publications/markdown/S4_complete_derivations_v23.md)** | Complete Derivations | Full mathematical proofs of all 13 exact relations and detailed calculations for 39 observables. Organized by sector (gauge, fermion, neutrino, cosmology) with error analysis. |
| **[S5](publications/markdown/S5_experimental_validation_v23.md)** | Experimental Validation | Comparison with PDG 2024, NuFIT 5.3, Planck 2020, DESI DR2. Chi-square analysis, pull distributions. Defines falsification criteria (Type A/B/C) and experimental timeline. |
| **[S6](publications/markdown/S6_theoretical_extensions_v23.md)** | Theoretical Extensions | Speculative extensions: M-theory embedding (11D → 10D → 4D), AdS/CFT correspondence, information-theoretic interpretations, number-theoretic patterns. Status: EXPLORATORY. |
| **[S7](publications/markdown/S7_dimensional_observables_v23.md)** | Dimensional Observables | Bridge from dimensionless ratios to absolute masses (GeV). Derives scale parameter Lambda_GIFT from b₂, e⁸, dim(E₈). Covers fermion masses, boson masses, cosmological parameters. |

## Repository Structure

```
gift/
├── publications/           # Theoretical documents
│   ├── markdown/          # Main paper + S1-S7 supplements
│   ├── references/        # Observable reference, geometric justifications
│   ├── tex/               # LaTeX sources
│   └── pdf/               # Generated PDFs
├── statistical_validation/ # Monte Carlo validation
├── tests/                 # Test suite
├── docs/                  # FAQ, glossary, guides
└── legacy/                # Archived: v1, v2.0, v2.1, formal proofs, G2_ML
```

**Core Library** ([gift-framework/core](https://github.com/gift-framework/core)):
- Formal proofs (Lean 4 + Coq) — 35 certified relations
- K₇ metric pipeline (`pip install giftpy`) — G₂ geometry, harmonic forms, physics extraction

See [STRUCTURE.md](STRUCTURE.md) for navigation guide.

## Connect

### Media

| Platform | Link |
|----------|------|
| YouTube | [@giftheory](https://youtube.com/@giftheory) |
| Substack | [giftheory.substack.com](https://substack.com/@giftheory) |
| X/Twitter | [@GIFTheory](https://x.com/GIFTheory) |
| Instagram | [@theory.gift](https://instagram.com/theory.gift) |

### Platforms

| Platform | Link |
|----------|------|
| Zenodo | [10.5281/zenodo.17751250](https://doi.org/10.5281/zenodo.17751250) |
| ResearchGate | [Author page](https://www.researchgate.net/profile/Brieuc-De-La-Fourniere) |
| SSRN | [Author page](https://papers.ssrn.com/sol3/cf_dev/AbsByAuth.cfm?per_id=7701350) |

## Falsification Tests

| Prediction | Test | Timeline | Criterion |
|------------|------|----------|-----------|
| δ_CP = 197° | DUNE | 2027-2030 | Outside [187°, 207°] |
| sin²θ_W = 3/13 | FCC-ee | 2040s | Outside [0.2295, 0.2320] |
| m_s/m_d = 20 | Lattice QCD | 2030 | Converges outside [19, 21] |
| N_gen = 3 | LHC | Ongoing | Fourth generation discovery |

Details: [S5](publications/markdown/S5_experimental_validation_v23.md), [Experimental Status](docs/EXPERIMENTAL_VALIDATION.md)

## Citation

```bibtex
@software{gift_framework_v23a_2025,
  title={GIFT Framework v2.3: Geometric Information Field Theory},
  author={{Brieuc de La Fournière}},
  year={2025},
  url={https://github.com/gift-framework/GIFT},
  version={2.3.0}
}
```

See [CITATION.md](CITATION.md) for additional formats.

## Contributing

Contributions welcome: mathematical refinements, experimental comparisons, visualizations, documentation.

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE)

## Contact

- **Repository**: https://github.com/gift-framework/GIFT
- **Issues**: https://github.com/gift-framework/GIFT/issues
- **Author**: [contact](mailto:brieuc@bdelaf.com)

---
> **Gift from bit**
---
