# Repository Structure

This repository contains the theoretical documentation for GIFT v3.3.

## Directory Layout

```
GIFT/
├── publications/                     # Theoretical documents
│   ├── README.md                    # Overview and reading guide
│   ├── markdown/                    # Core documents (v3.3)
│   │   ├── GIFT_v3.3_main.md         # Main paper
│   │   ├── GIFT_v3.3_S1_foundations.md   # E₈, G₂, K₇ foundations
│   │   ├── GIFT_v3.3_S2_derivations.md   # 33 dimensionless derivations
│   │   └── GIFT_v3.3_S3_dynamics.md      # Torsional flow, scale bridge
│   ├── references/                  # Exploratory & reference docs
│   │   ├── 39_observables.csv      # Machine-readable observables
│   │   ├── OBSERVABLE_REFERENCE.md # Complete observable catalog
│   │   ├── NUMBER_THEORETIC_STRUCTURES.md  # Fibonacci, Prime Atlas, Monster
│   │   └── SPECULATIVE_PHYSICS.md  # Scale bridge, Yukawa, M-theory, QG
│   ├── tex/                         # LaTeX sources
│   ├── pdf/                         # Generated PDFs
│   └── Lean/                        # Lean formalization docs
│
├── docs/                            # Supporting documentation
│   ├── GIFT_FOR_EVERYONE.md        # Complete guide with everyday analogies
│   ├── FAQ.md                       # Common questions
│   ├── GLOSSARY.md                  # Technical terms
│   ├── PHILOSOPHY.md                # Foundational perspective
│   ├── GIFTPY_FOR_GEOMETERS.md     # Guide for geometers
│   ├── INFO_GEO_FOR_PHYSICISTS.md  # Guide for physicists
│   ├── LEAN_FOR_PHYSICS.md         # Guide for formalization
│   ├── figures/                     # Lean blueprints
│   └── legacy/                      # Archived v2.3/v3.0 supplements
│
├── statistical_validation/          # Monte Carlo validation
│   ├── validation_v33.py           # v3.3 comprehensive validation
│   ├── comprehensive_statistics_v33.py  # Advanced statistical tests
│   └── validation_v33_results.json # v3.3 results
│
├── research/                        # Exploratory research (WIP)
│   ├── README.md                   # Research overview
│   └── yang-mills/                 # Yang-Mills mass gap investigation
│       ├── README.md               # Project summary
│       ├── UNIVERSALITY_RESULTS.md # Validated spectral gap scaling
│       ├── BLIND_VALIDATION_RECAP.md  # Methodology (bias elimination)
│       ├── DEEP_STRUCTURE.md       # H* = dim(G₂) × dim(K₇) + 1
│       ├── UNIVERSALITY_CONJECTURE.md # Universal vs GIFT-specific formulas
│       └── STATUS.md               # Project status log
│
├── README.md                        # Main repository overview
├── CHANGELOG.md                     # Version history
├── CITATION.md                      # How to cite
├── STRUCTURE.md                     # This file
└── LICENSE                          # MIT License
```

## Quick Navigation

| Looking for... | Go to |
|----------------|-------|
| Framework overview | `README.md` |
| Beginner-friendly guide | `docs/GIFT_FOR_EVERYONE.md` |
| Complete theory | `publications/markdown/GIFT_v3.3_main.md` |
| All derivations | `publications/markdown/GIFT_v3.3_S2_derivations.md` |
| Observables data | `publications/references/39_observables.csv` |
| Monte Carlo validation | `statistical_validation/validation_v33.py` |
| Formal verification | [gift-framework/core](https://github.com/gift-framework/core) |
| Technical definitions | `docs/GLOSSARY.md` |
| Yang-Mills research | `research/yang-mills/` |
| Legacy supplements | `docs/legacy/` |

## Core Documents (v3.3)

| Document | Content |
|----------|---------|
| GIFT_v3.3_main.md | Complete theoretical framework |
| GIFT_v3.3_S1_foundations.md | E₈, G₂, K₇ mathematical construction |
| GIFT_v3.3_S2_derivations.md | 33 dimensionless derivations with proofs |
| GIFT_v3.3_S3_dynamics.md | Torsional dynamics, scale bridge |

## Exploratory References

| Document | Content |
|----------|---------|
| NUMBER_THEORETIC_STRUCTURES.md | Fibonacci, Prime Atlas, Monster, Moonshine |
| SPECULATIVE_PHYSICS.md | Scale bridge, Yukawa, M-theory, QG |

## Research (Work in Progress)

| Folder | Content |
|--------|---------|
| research/yang-mills/ | Spectral gap investigation, λ₁ ∝ 1/H* validation |

## Related Repositories

| Repository | Content |
|------------|---------|
| [gift-framework/core](https://github.com/gift-framework/core) | Formal proofs (Lean 4), K₇ metric pipeline, giftpy |

## Version

**Current**: v3.3.14 (2026-01-28)
**Relations**: ~330 certified (core v3.3.14)
**Predictions**: 33 dimensionless (**0.21% mean deviation, PDG 2024**)
**Monte Carlo**: 192,349 configurations tested, 0 better than GIFT
**Key Result**: Analytical G₂ metric with T = 0 exactly
