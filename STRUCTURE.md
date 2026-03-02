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
│   │   │   └── Numerical_G2_Metric.md        # PINN-based G₂ metric construction
│   │   ├── tex/                       # LaTeX sources
│   │   └── pdf/                       # Generated PDFs
│   ├── outreach/                      # Vulgarization & blog posts
│   │   └── (7 Substack posts)
│   ├── references/                    # Data & reference catalogs
│   │   ├── GIFT_ATLAS.json            # Canonical structured atlas (v3.3)
│   │   ├── observables.csv            # Machine-readable observables
│   │   ├── OBSERVABLE_REFERENCE.md    # Complete observable catalog
│   │   ├── NUMBER_THEORETIC_STRUCTURES.md  # Fibonacci, Prime Atlas, Monster
│   │   ├── SPECULATIVE_PHYSICS.md     # Scale bridge, Yukawa, M-theory, QG
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
├── research/                          # Exploratory research (WIP)
│   ├── yang-mills/                    # Yang-Mills mass gap investigation
│   ├── riemann/                       # Riemann hypothesis connections
│   ├── heegner-riemann/               # Heegner points & Riemann
│   ├── spectral/                      # Spectral analysis
│   ├── pattern_recognition/           # Pattern discovery
│   ├── tcs/                           # TCS constructions
│   ├── notebooks/                     # Research-grade notebooks
│   ├── analysis/                      # Exploratory analysis documents
│   ├── archive/                       # Archived explorations
│   └── legacy/                        # Old research logs
│
├── notebooks/                         # Curated demo notebooks (11 key)
│   ├── GIFT_v3_Framework_Validation.ipynb
│   ├── GIFT_PINN_Training.ipynb
│   ├── K7_Metric_Formalization.ipynb
│   ├── Joyce_Formalization_Tutorial.ipynb
│   ├── Selberg_Complete_Verification.ipynb
│   └── ...
│
├── docs/                              # User-facing documentation
│   ├── GIFT_FOR_EVERYONE.md           # Complete guide with everyday analogies
│   ├── FAQ.md                         # Common questions
│   ├── GLOSSARY.md                    # Technical terms
│   ├── GIFTPY_FOR_GEOMETERS.md        # Guide for geometers
│   ├── INFO_GEO_FOR_PHYSICISTS.md     # Guide for physicists
│   ├── LEAN_FOR_PHYSICS.md            # Guide for formalization
│   ├── figures/                       # Lean blueprints, diagrams
│   ├── media/                         # Logos, images
│   └── legacy/                        # Archived v2.2/v2.3/v3.0/v3.1 documents
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
| Observables data | `publications/references/observables.csv` |
| Monte Carlo validation | `publications/validation/` |
| Blog posts & outreach | `publications/outreach/` |
| Formal verification | [gift-framework/core](https://github.com/gift-framework/core) |
| Technical definitions | `docs/GLOSSARY.md` |
| Yang-Mills research | `research/yang-mills/` |
| Demo notebooks | `notebooks/` |
| Legacy supplements | `docs/legacy/` |

## Core Documents (v3.3)

| Document | Content |
|----------|---------|
| GIFT_v3.3_main.md | Complete theoretical framework |
| GIFT_v3.3_S1_foundations.md | E₈, G₂, K₇ mathematical construction |
| GIFT_v3.3_S2_derivations.md | 33 dimensionless derivations with proofs |
| Numerical_G2_Metric.md | PINN-based G₂ metric construction |

## Exploratory References

| Document | Content |
|----------|---------|
| NUMBER_THEORETIC_STRUCTURES.md | Fibonacci, Prime Atlas, Monster, Moonshine |
| SPECULATIVE_PHYSICS.md | Scale bridge, Yukawa, M-theory, QG |

## Research (Work in Progress)

| Folder | Content |
|--------|---------|
| research/yang-mills/ | Spectral gap investigation, λ₁ ∝ 1/H* validation |
| research/riemann/ | Riemann hypothesis connections |
| research/analysis/ | Exploratory analysis documents |

## Related Repositories

| Repository | Content |
|------------|---------|
| [gift-framework/core](https://github.com/gift-framework/core) | Formal proofs (Lean 4), K₇ metric pipeline, giftpy |

## Version

**Current**: v3.3.24 (2026-03-02)
**Relations**: ~290 certified (core v3.3.24)
**Predictions**: 33 predictions (**0.24% mean deviation** across 32 well-measured, 0.57% incl. δ_CP; PDG 2024 / NuFIT 6.0)
**Validation**: 3,070,396 configs exhaustive + 7-component bullet-proof (Westfall-Young, Bayesian, PPC)
**Key Result**: Analytical G₂ metric with T = 0 exactly
