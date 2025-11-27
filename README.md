# Geometric Information Field Theory v2.2

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-2.2.0-green.svg)](CHANGELOG.md)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gift-framework/GIFT/main)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gift-framework/GIFT/blob/main/assets/visualizations/precision_dashboard.ipynb)

## Overview

| Metric | Value |
|--------|-------|
| **Precision** | 0.128% mean deviation across 39 observables |
| **Parameters** | Zero continuous adjustable (all structurally determined) |
| **Exact relations** | 13 rigorously proven topological identities |
| **Key results** | sin²θ_W = 3/13, κ_T = 1/61, det(g) = 65/32, δ_CP = 197° |

The **Geometric Information Field Theory (GIFT)** derives Standard Model parameters from E₈×E₈ exceptional Lie algebras via dimensional reduction **E₈×E₈ → AdS₄×K₇ → Standard Model**. Version 2.2 achieves the **zero-parameter paradigm**: all quantities derive from fixed topological structure.

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

Complete proofs: [Supplement S4](publications/supplements/S4_complete_derivations.md)

### Precision by Sector

| Sector | Observables | Best Result |
|--------|-------------|-------------|
| Gauge | 3 | α_s = √2/12 (0.04%) |
| Neutrino | 4 | δ_CP = 197° (0.00%) |
| Lepton | 4 | Q_Koide = 2/3 (0.001%) |
| Quark ratios | 10 | m_s/m_d = 20 (0.00%) |
| CKM | 6 | Mean 0.11% |
| Cosmology | 2 | n_s = ζ(11)/ζ(5) (0.00%) |

Full tables: [Main Paper Section 8](publications/gift_2_2_main.md)

## Documentation

### Reading Path

| Time | Document | Description |
|------|----------|-------------|
| 5 min | [summary.txt](publications/summary.txt) | Executive briefing |
| 30 min | [Main Paper](publications/gift_2_2_main.md) | Complete framework |
| Deep dive | [Supplements S1-S7](publications/supplements/) | Mathematical details |

### Key Documents

- **[Main Paper](publications/gift_2_2_main.md)** - Complete theoretical framework (~1400 lines)
- **[Observable Reference](publications/GIFT_v22_Observable_Reference.md)** - All 39 observables
- **[S4: Derivations](publications/supplements/S4_complete_derivations.md)** - Proofs of 13 exact relations
- **[S5: Validation](publications/supplements/S5_experimental_validation.md)** - Experimental comparison
- **[Glossary](docs/GLOSSARY.md)** - Technical terms
- **[FAQ](docs/FAQ.md)** - Common questions

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
│   ├── gift_2_2_main.md   # Core paper
│   ├── supplements/       # S1-S7 mathematical details
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

## Falsification Tests

| Prediction | Test | Timeline | Criterion |
|------------|------|----------|-----------|
| δ_CP = 197° | DUNE | 2027-2030 | Outside [187°, 207°] |
| sin²θ_W = 3/13 | FCC-ee | 2040s | Outside [0.2295, 0.2320] |
| m_s/m_d = 20 | Lattice QCD | 2030 | Converges outside [19, 21] |
| N_gen = 3 | LHC | Ongoing | Fourth generation discovery |

Details: [S5](publications/supplements/S5_experimental_validation.md), [Experimental Status](docs/EXPERIMENTAL_VALIDATION.md)

## Citation

```bibtex
@software{gift_framework_v22_2025,
  title={GIFT Framework v2.2: Geometric Information Field Theory},
  author={{Brieuc de La Fournière}},
  year={2025},
  url={https://github.com/gift-framework/GIFT},
  version={2.2.0}
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
