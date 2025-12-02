# Geometric Information Field Theory v2.3a

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-2.3.0-green.svg)](CHANGELOG.md)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gift-framework/GIFT/main)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gift-framework/GIFT/blob/main/assets/visualizations/precision_dashboard.ipynb)

## Overview

| Metric | Value |
|--------|-------|
| **Precision** | 0.128% mean deviation across 39 observables |
| **Parameters** | Zero continuous adjustable (all structurally determined) |
| **Exact relations** | 13 rigorously proven topological identities |
| **Lean certifications** | det(g)=65/32, Banach FP theorem, Joyce perturbation |
| **Key results** | sin²θ_W = 3/13, κ_T = 1/61, det(g) = 65/32, δ_CP = 197° |

The **Geometric Information Field Theory (GIFT)** derives Standard Model parameters from E₈×E₈ exceptional Lie algebras via dimensional reduction **E₈×E₈ → AdS₄×K₇ → Standard Model**. Version 2.3a achieves the **zero-parameter paradigm** with **formal verification**: all quantities derive from fixed topological structure, with key results **machine-verified** via Lean 4 theorem prover.

## Formal Verification Highlights

**Machine-Verified Results (Lean 4):**
- **det(g) = 65/32**: Metric determinant certified to 0.0001% precision
- **Banach Fixed Point Theorem**: Existence of torsion-free G₂ structures proven
- **Joyce Perturbation Theorem**: 20× safety margin for manifold existence
- **13 PROVEN Relations**: All exact topological identities formally verified

**Computational Validation:**
- Physics-informed neural networks (PINN) with 1M+ training samples
- Monte Carlo uncertainty propagation (10⁵ configurations)
- Mean precision: 0.128% across 39 observables
- Interval arithmetic with rigorous error bounds

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

### 13 Proven Exact Relations

| Relation | Value | Source |
|----------|-------|--------|
| sin²θ_W | 3/13 | b₂/(b₃ + dim(G₂)) |
| κ_T | 1/61 | 1/(b₃ - dim(G₂) - p₂) |
| det(g) | 65/32 | Topological formula |
| τ | 3472/891 | (496×21)/(27×99) |
| δ_CP | 197° | dim(K₇)×dim(G₂) + H* |
| Q_Koide | 2/3 | dim(G₂)/b₂ |
| m_s/m_d | 20 | p₂² × Weyl |
| m_τ/m_e | 3477 | dim(K₇) + 10×dim(E₈) + 10×H* |
| N_gen | 3 | Atiyah-Singer index theorem |
| p₂ | 2 | dim(G₂)/dim(K₇) |
| n_s | ζ(11)/ζ(5) | Bulk/Weyl zeta ratio |
| Ω_DE | ln(2)×98/99 | Binary information architecture |
| ξ | 5π/16 | (Weyl/p₂) × β₀ |

Complete proofs: [Supplement S4](publications/markdown/S4_complete_derivations_v23.md)

**🔬 Machine-Verified Results:**
- **det(g) = 65/32**: Certified via Lean 4 with 0.0001% precision (20× Joyce margin)
- **13 PROVEN Relations**: All topological identities formally verified
- **Banach Fixed Point**: Existence theorem for G₂ structures proven

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
- **[Glossary](docs/GLOSSARY.md)** - Technical terms
- **[FAQ](docs/FAQ.md)** - Common questions
- **[Philosophy](docs/PHILOSOPHY.md)** - The philosophy behind GIFT
- **[Lean Formal Verification](G2_ML/G2_Lean/)** - Machine-verified mathematical proofs

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
  title={GIFT Framework v2.3a: Geometric Information Field Theory},
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
