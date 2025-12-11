# Repository Structure

This repository contains the theoretical documentation for GIFT v3.0.

## Directory Layout

```
GIFT/
├── publications/                     # Theoretical documents
│   ├── README.md                    # Overview and reading guide
│   ├── markdown/                    # Core documents (v3.0)
│   │   ├── GIFT_v3_main.md         # Main paper
│   │   ├── GIFT_v3_S1_foundations.md   # E₈, G₂, K₇ foundations
│   │   └── GIFT_v3_S2_derivations.md   # 18 dimensionless derivations
│   ├── references/                  # Exploratory & reference docs
│   │   ├── 39_observables.csv      # Machine-readable observables
│   │   ├── yukawa_mixing.md        # CKM/PMNS, Yukawa couplings
│   │   ├── sequences_prime_atlas.md # Fibonacci, Prime Atlas
│   │   ├── monster_moonshine.md    # Monster group, j-invariant
│   │   ├── dimensional_observables.md # Absolute masses
│   │   └── theoretical_extensions.md  # M-theory, QG
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
│   ├── legacy/                      # Archived v2.3/v3.0 supplements
│   └── wip/                         # Work in progress
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
| Complete theory | `publications/markdown/GIFT_v3_main.md` |
| All derivations | `publications/markdown/GIFT_v3_S2_derivations.md` |
| Observables data | `publications/references/39_observables.csv` |
| Formal verification | [gift-framework/core](https://github.com/gift-framework/core) |
| Technical definitions | `docs/GLOSSARY.md` |
| Legacy supplements | `docs/legacy/` |

## Core Documents (v3.0)

| Document | Content |
|----------|---------|
| GIFT_v3_main.md | Complete theoretical framework |
| GIFT_v3_S1_foundations.md | E₈, G₂, K₇ mathematical construction |
| GIFT_v3_S2_derivations.md | 18 dimensionless derivations with proofs |

## Exploratory References

| Document | Content |
|----------|---------|
| yukawa_mixing.md | CKM/PMNS matrices, Yukawa couplings |
| sequences_prime_atlas.md | Fibonacci embedding, Prime Atlas |
| monster_moonshine.md | Monster group, j-invariant connections |
| dimensional_observables.md | Absolute masses (heuristic) |
| theoretical_extensions.md | M-theory, quantum gravity |

## Related Repositories

| Repository | Content |
|------------|---------|
| [gift-framework/core](https://github.com/gift-framework/core) | Formal proofs (Lean 4 + Coq), K₇ metric pipeline, giftpy |

## Version

**Current**: v3.0.0 (2025-12-11)
**Relations**: 165+ certified
**Observables**: 18 dimensionless (0.197% mean deviation)
