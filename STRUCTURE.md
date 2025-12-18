# Repository Structure

This repository contains the theoretical documentation for GIFT v3.1.

## Directory Layout

```
GIFT/
├── publications/                     # Theoretical documents
│   ├── README.md                    # Overview and reading guide
│   ├── markdown/                    # Core documents (v3.1)
│   │   ├── GIFT_v3.1_main.md         # Main paper
│   │   ├── GIFT_v3.1_S1_foundations.md   # E₈, G₂, K₇ foundations
│   │   ├── GIFT_v3.1_S2_derivations.md   # 18 dimensionless derivations
│   │   └── GIFT_v3.1_S3_dynamics.md      # Torsional flow, scale bridge
│   ├── references/                  # Exploratory & reference docs
│   │   ├── 39_observables.csv      # Machine-readable observables
│   │   ├── NUMBER_THEORETIC_STRUCTURES.md  # Fibonacci, Prime Atlas, Monster
│   │   └── SPECULATIVE_PHYSICS.md  # Scale bridge, Yukawa, M-theory, QG
│   ├── tex/                         # LaTeX sources
│   ├── pdf/                         # Generated PDFs
│   └── Lean/                        # Lean formalization docs
│
├── docs/                            # Supporting documentation
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
| Complete theory | `publications/markdown/GIFT_v3.1_main.md` |
| All derivations | `publications/markdown/GIFT_v3.1_S2_derivations.md` |
| Observables data | `publications/references/39_observables.csv` |
| Formal verification | [gift-framework/core](https://github.com/gift-framework/core) |
| Technical definitions | `docs/GLOSSARY.md` |
| Legacy supplements | `docs/legacy/` |

## Core Documents (v3.1)

| Document | Content |
|----------|---------|
| GIFT_v3.1_main.md | Complete theoretical framework |
| GIFT_v3.1_S1_foundations.md | E₈, G₂, K₇ mathematical construction |
| GIFT_v3.1_S2_derivations.md | 18 dimensionless derivations with proofs |
| GIFT_v3.1_S3_dynamics.md | Torsional dynamics, scale bridge |

## Exploratory References

| Document | Content |
|----------|---------|
| NUMBER_THEORETIC_STRUCTURES.md | Fibonacci, Prime Atlas, Monster, Moonshine |
| SPECULATIVE_PHYSICS.md | Scale bridge, Yukawa, M-theory, QG |

## Related Repositories

| Repository | Content |
|------------|---------|
| [gift-framework/core](https://github.com/gift-framework/core) | Formal proofs (Lean 4 + Coq), K₇ metric pipeline, giftpy |

## Version

**Current**: v3.1.1 (2025-12-17)
**Relations**: 180+ certified
**Predictions**: 18 dimensionless (**0.087% mean deviation**)
**Key Result**: Analytical G₂ metric with T = 0 exactly
