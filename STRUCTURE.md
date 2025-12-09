# Repository Structure

This repository contains the theoretical documentation for GIFT v3.0.

## Directory Layout

```
GIFT/
├── publications/                     # Theoretical documents
│   ├── README.md                    # Overview and reading guide
│   ├── markdown/                    # Main documents
│   │   ├── gift_3_0_main.md        # Core paper (v3.0)
│   │   ├── S1-S7 supplements       # Mathematical details
│   │   ├── S8_sequences_prime_atlas_v30.md  # NEW: Fibonacci, Primes
│   │   └── S9_monster_moonshine_v30.md      # NEW: Monster, Moonshine
│   ├── references/                  # Quick reference documents
│   │   ├── GIFT_v30_Observable_Reference.md
│   │   ├── GIFT_v30_Geometric_Justifications.md
│   │   └── GIFT_v30_Statistical_Validation.md
│   ├── tex/                         # LaTeX sources
│   └── pdf/                         # Generated PDFs
│
├── docs/                            # Supporting documentation
│   ├── FAQ.md                       # Common questions
│   ├── GLOSSARY.md                  # Technical terms
│   ├── PHILOSOPHY.md                # Foundational perspective
│   ├── GIFTPY_FOR_GEOMETERS.md     # Guide for geometers
│   ├── INFO_GEO_FOR_PHYSICISTS.md  # Guide for physicists
│   ├── LEAN_FOR_PHYSICS.md         # Guide for formalization
│   └── figures/                     # Lean blueprints
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
| Complete theory | `publications/markdown/gift_3_0_main.md` |
| All 39 observables | `publications/references/GIFT_v30_Observable_Reference.md` |
| Proofs (165+ relations) | `publications/markdown/S4_complete_derivations_v30.md` |
| Fibonacci/Prime structure | `publications/markdown/S8_sequences_prime_atlas_v30.md` |
| Monster/Moonshine | `publications/markdown/S9_monster_moonshine_v30.md` |
| Formal verification | [gift-framework/core](https://github.com/gift-framework/core) |
| Technical definitions | `docs/GLOSSARY.md` |

## Supplements Overview

| Supplement | Content |
|------------|---------|
| S1 | E₈ algebra, Exceptional Chain, McKay correspondence |
| S2 | K₇ manifold construction, TCS, Betti numbers |
| S3 | Torsional dynamics, geodesic flow |
| S4 | Complete derivations, 165+ certified relations |
| S5 | Experimental validation, falsification criteria |
| S6 | Theoretical extensions (M-theory, AdS/CFT) |
| S7 | Dimensional observables, scale bridge |
| **S8** | **Fibonacci/Lucas embedding, Prime Atlas** |
| **S9** | **Monster group, Monstrous Moonshine** |

## Related Repositories

| Repository | Content |
|------------|---------|
| [gift-framework/core](https://github.com/gift-framework/core) | Formal proofs (Lean 4 + Coq), K₇ metric pipeline, validation code |

## Version

**Current**: v3.0.0 (2025-12-09)
**Relations**: 165+ certified
**Observables**: 39 with 0.198% mean deviation
