# Repository Structure

This repository contains the theoretical documentation for GIFT v3.3.

## Directory Layout

```
GIFT/
├── publications/                      # Published content & validation
│   ├── papers/                        # Scientific articles
│   │   ├── markdown/                  # Core documents (v3.3)
│   │   │   ├── GIFT_v3.3_main.md         # Main paper
│   │   │   ├── GIFT_v3.3_S1_foundations.md   # E₈, G₂, K₇ foundations
│   │   │   ├── GIFT_v3.3_S2_derivations.md   # 33 dimensionless derivations
│   │   │   ├── Explicit_G2_Metric.md          # Analytical G₂ metric construction
│   │   │   └── Spectral_Geometry.md           # KK spectrum, Yukawa, gauge unification
│   │   ├── tex/                       # LaTeX sources
│   │   ├── pdf/                       # Compiled PDFs
│   │   ├── figures/                   # Publication figures (PDF + PNG)
│   │   └── notebooks/                 # Companion Jupyter notebooks
│   │       ├── G2_Metric_Companion.ipynb
│   │       └── Spectral_Geometry_Companion.ipynb
│   ├── outreach/                      # Vulgarization & blog posts
│   │   └── (7 Substack posts)
│   ├── references/                    # Data & reference catalogs
│   │   ├── GIFT_ATLAS.json            # Canonical structured atlas (v3.3)
│   │   ├── observables.csv            # Machine-readable observables
│   │   ├── OBSERVABLE_REFERENCE.md    # Complete observable catalog
│   │   ├── STATISTICAL_EVIDENCE.md    # Rigorous statistical analysis
│   │   ├── INDEPENDENT_VALIDATIONS.md # External research converging with GIFT
│   │   └── Bibliography.md            # References
│   └── validation/                    # Monte Carlo validation (v3.3 only)
│       ├── validation_v33.py          # Core formulas & experimental data
│       ├── bulletproof_validation_v33.py    # 7-component bullet-proof validation
│       ├── exhaustive_validation_v33.py     # Exhaustive search (3M+ configs)
│       ├── comprehensive_statistics_v33.py  # Advanced statistical tests
│       └── selection/                 # Formula selection & Pareto analysis
│
├── docs/                              # User-facing documentation
│   ├── GIFT_FOR_EVERYONE.md           # Complete guide with everyday analogies
│   ├── FAQ.md                         # Common questions
│   ├── GLOSSARY.md                    # Technical terms
│   ├── GIFTPY_FOR_GEOMETERS.md        # Guide for geometers
│   ├── INFO_GEO_FOR_PHYSICISTS.md     # Guide for physicists
│   ├── LEAN_FOR_PHYSICS.md            # Guide for formalization
│   └── figures/                       # Lean blueprints, diagrams
│
├── README.md                          # Main repository overview
├── CHANGELOG.md                       # Version history
├── CITATION.md                        # How to cite
├── STRUCTURE.md                       # This file
└── LICENSE                            # MIT License
```

## Quick Navigation

| Looking for... | Go to |
|----------------|-------|
| Framework overview | `README.md` |
| Beginner-friendly guide | `docs/GIFT_FOR_EVERYONE.md` |
| Complete theory | `publications/papers/markdown/GIFT_v3.3_main.md` |
| All derivations | `publications/papers/markdown/GIFT_v3.3_S2_derivations.md` |
| Spectral geometry | `publications/papers/markdown/Spectral_Geometry.md` |
| Companion notebooks | `publications/papers/notebooks/` |
| Observables data | `publications/references/observables.csv` |
| Monte Carlo validation | `publications/validation/` |
| Blog posts & outreach | `publications/outreach/` |
| Formal verification | [gift-framework/core](https://github.com/gift-framework/core) |
| Technical definitions | `docs/GLOSSARY.md` |

## Core Documents (v3.3)

| Document | Content |
|----------|---------|
| GIFT_v3.3_main.md | Complete theoretical framework |
| GIFT_v3.3_S1_foundations.md | E₈, G₂, K₇ mathematical construction |
| GIFT_v3.3_S2_derivations.md | 33 dimensionless derivations with proofs |
| Explicit_G2_Metric.md | Analytical G₂ metric construction |
| Spectral_Geometry.md | KK spectrum, Yukawa, gauge unification from G₂ metric |

## Related Repositories

| Repository | Content |
|------------|---------|
| [gift-framework/core](https://github.com/gift-framework/core) | Formal proofs (Lean 4), K₇ metric pipeline, giftpy |

## Version

**Current**: v3.3.31 (2026-03-08)
**Relations**: 455+ certified (core v3.3.31)
**Predictions**: 33 predictions (**0.24% mean deviation** across 32 well-measured, 0.57% incl. δ_CP; PDG 2024 / NuFIT 6.0)
**Validation**: 3,070,396 configs exhaustive + 7-component bullet-proof (Westfall-Young, Bayesian, PPC)
**Key Result**: Analytical G₂ metric with T = 0 exactly
