# Geometric Information Field Theory v2.3

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-2.3-green.svg)](CHANGELOG.md)
[![Lean 4 Verified](https://img.shields.io/badge/Lean_4-Verified-blue)](Lean/)
[![CI Status](https://github.com/gift-framework/GIFT/actions/workflows/lean.yml/badge.svg)](https://github.com/gift-framework/GIFT/actions/workflows/lean.yml)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gift-framework/GIFT/main)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gift-framework/GIFT/blob/main/assets/visualizations/precision_dashboard.ipynb)

## Overview

| Metric | Value |
|--------|-------|
| **Precision** | 0.128% mean deviation across 39 observables |
| **Parameters** | Zero continuous adjustable (all structurally determined) |
| **Lean-verified relations** | **13 formally proven** (zero domain axioms, zero sorry) |
| **Key results** | sin²θ_W = 3/13, κ_T = 1/61, det(g) = 65/32, τ = 3472/891, δ_CP = 197° |

The **Geometric Information Field Theory (GIFT)** derives Standard Model parameters from E₈×E₈ exceptional Lie algebras via dimensional reduction **E₈×E₈ → AdS₄×K₇ → Standard Model**. Version 2.3 achieves the **zero-parameter paradigm** with **formal verification**: all quantities derive from fixed topological structure, with **13 exact relations machine-verified** via Lean 4 theorem prover and Mathlib.

## Formal Verification (Lean 4)

The `/Lean/` directory contains a complete Lean 4 formalization with **17 modules**, proving all 13 exact relations from topological inputs:

| Module | Content |
|--------|---------|
| `GIFT.Algebra` | E₈ root system, Weyl group (4 modules) |
| `GIFT.Geometry` | G₂ holonomy, TCS construction (4 modules) |
| `GIFT.Topology` | Betti numbers, cohomology (3 modules) |
| `GIFT.Relations` | All 13 physical relations (7 modules) |
| `GIFT.Certificate` | Main theorem + zero-parameter proof (3 modules) |

**Verification status:**
- Lean 4.14.0 + Mathlib 4.14.0
- **0 domain-specific axioms** (only propext, Quot.sound)
- **0 sorry** (all proofs complete)

See [Lean/README.md](Lean/README.md) for build instructions.

**Computational Validation:**
- Physics-informed neural networks (PINN) with 1M+ training samples
- Monte Carlo uncertainty propagation (10⁵ configurations)
- Mean precision: 0.128% across 39 observables

## Quick Start

### Run in Browser (No Installation)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gift-framework/GIFT/main?filepath=assets/visualizations/) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gift-framework/GIFT/blob/main/assets/visualizations/precision_dashboard.ipynb)

### Local Installation

```bash
git clone https://github.com/gift-framework/GIFT.git
cd gift
pip install -r requirements.txt
jupyter notebook assets/visualizations/
```

**Requirements**: Python 3.11+

## Key Results

### 13 Lean-Verified Exact Relations

| Relation | Value | Formula | Status |
|----------|-------|---------|--------|
| sin²θ_W | 3/13 | b₂/(b₃ + dim(G₂)) | **PROVEN (Lean)** |
| τ | 3472/891 | (496×21)/(27×99) | **PROVEN (Lean)** |
| det(g) | 65/32 | Topological formula | **PROVEN (Lean)** |
| κ_T | 1/61 | 1/(b₃ - dim(G₂) - p₂) | **PROVEN (Lean)** |
| δ_CP | 197° | 7×dim(G₂) + H* | **PROVEN (Lean)** |
| Q_Koide | 2/3 | dim(G₂)/b₂ | **PROVEN (Lean)** |
| m_s/m_d | 20 | p₂² × Weyl | **PROVEN (Lean)** |
| m_τ/m_e | 3477 | dim(K₇) + 10×dim(E₈) + 10×H* | **PROVEN (Lean)** |
| λ_H | √17/32 | √(dim(G₂)+N_gen)/2⁵ | **PROVEN (Lean)** |
| H* | 99 | b₂ + b₃ + 1 | **PROVEN (Lean)** |
| p₂ | 2 | dim(G₂)/dim(K₇) | **PROVEN (Lean)** |
| N_gen | 3 | rank(E₈) - Weyl | **PROVEN (Lean)** |
| E₈×E₈ | 496 | 2 × dim(E₈) | **PROVEN (Lean)** |

Complete proofs: [Lean/](Lean/) | Paper proofs: [Supplement S4](publications/markdown/S4_complete_derivations_v23.md)

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
- **[Lean Formal Verification](Lean/)** - Machine-verified proofs (17 modules, 0 sorry)
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

### Interactive Tools

| Notebook | Description |
|----------|-------------|
| [Precision Dashboard](assets/visualizations/precision_dashboard.ipynb) | All 39 observables vs experiment |
| [E₈ Root System](assets/visualizations/e8_root_system_3d.ipynb) | 240-root 3D visualization |
| [Dimensional Flow](assets/visualizations/dimensional_reduction_flow.ipynb) | 496D → 99D → 4D animation |

## Repository Structure

```
gift/
├── publications/           # Theoretical documents
│   ├── markdown/          # Main paper + S1-S7 supplements
│   ├── references/        # Observable reference, geometric justifications
│   ├── tex/               # LaTeX sources
│   └── pdf/               # Generated PDFs
├── Lean/                  # Lean 4 formal verification (17 modules)
│   └── GIFT/              # Algebra, Geometry, Topology, Relations, Certificate
├── assets/visualizations/ # Interactive notebooks
├── statistical_validation/ # Monte Carlo validation
├── G2_ML/                 # Neural network for K₇ metrics
├── tests/                 # Test suite
├── docs/                  # FAQ, glossary, guides
└── legacy/                # Archived versions (v1, v2.0, v2.1)
```

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

Details: [S5](publications/markdown/S5_experimental_validation.md), [Experimental Status](docs/EXPERIMENTAL_VALIDATION.md)

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
