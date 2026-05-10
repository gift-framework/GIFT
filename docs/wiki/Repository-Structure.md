---
title: "Repository Structure"
layout: default
---


This repository contains the theoretical documentation for GIFT v3.4.

## Directory Layout

```
GIFT/
├── publications/                      # Published content & validation
│   ├── papers/                        # Scientific articles
│   │   ├── markdown/                  # Core documents (v3.4 — canonical sources)
│   │   │   ├── GIFT_v3.4_main.md              # Main paper
│   │   │   ├── GIFT_v3.4_S1_foundations.md    # E₈, G₂, K₇ foundations
│   │   │   ├── GIFT_v3.4_S2_derivations.md    # 33 Type I derivations
│   │   │   ├── GIFT_v3.4_S3_observables.md    # 95-observable catalog
│   │   │   ├── g2_certified_neck.md           # Paper A — certified G₂ structure
│   │   │   └── g2_spectral.md                 # Paper B — spectral geometry
│   │   ├── tex/                       # LaTeX sources (v3.4)
│   │   ├── pdf/                       # Compiled PDFs (all published)
│   │   │   ├── gift_3.4_main.pdf              # Framework (Zenodo 20070101)
│   │   │   ├── g2_certified_neck.pdf          # Paper A (Zenodo 19892350)
│   │   │   ├── g2_spectral.pdf                # Paper B (Zenodo 19893371)
│   │   │   ├── K3_NK_Certificate.pdf          # Paper C (Zenodo 19708916)
│   │   │   └── donaldson_analytic.pdf         # Paper D (Zenodo 20039066)
│   │   ├── legacy/v3.3/               # v3.3 framework PDFs + markdown (archived)
│   │   ├── figures/                   # Publication figures (PDF + PNG)
│   │   └── notebooks/                 # Companion Jupyter notebooks
│   │       ├── g2_certified_neck_companion.ipynb
│   │       └── g2_spectral_companion.ipynb
│   ├── outreach/                      # Vulgarization & blog posts
│   ├── references/                    # Data & reference catalogs
│   │   ├── GIFT_ATLAS.json            # Canonical structured atlas
│   │   ├── observables.csv            # Machine-readable observables
│   │   ├── OBSERVABLE_REFERENCE.md    # Complete observable catalog
│   │   ├── STATISTICAL_EVIDENCE.md    # Rigorous statistical analysis
│   │   ├── INDEPENDENT_VALIDATIONS.md # External research converging with GIFT
│   │   └── Bibliography.md            # References
│   └── validation/                    # Monte Carlo validation
│       └── legacy/v3.3/               # v3.3 validation scripts (archived;
│                                      # v3.4 stats refresh in core/private)
│
├── docs/                              # User-facing documentation + Jekyll site
│   ├── index.html                     # Landing page (gift-framework.github.io)
│   ├── _config.yml                    # Jekyll config
│   ├── GIFT_FOR_EVERYONE.md           # Complete guide with everyday analogies
│   ├── FAQ.md                         # Common questions
│   ├── GLOSSARY.md                    # Technical terms
│   ├── GIFTPY_FOR_GEOMETERS.md        # Guide for geometers
│   ├── INFO_GEO_FOR_PHYSICISTS.md     # Guide for physicists
│   ├── LEAN_FOR_PHYSICS.md            # Guide for formalization
│   ├── wiki/                          # GitHub Wiki mirror (EN + FR)
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
| Complete theory | `publications/papers/markdown/GIFT_v3.4_main.md` |
| All derivations | `publications/papers/markdown/GIFT_v3.4_S2_derivations.md` |
| Observable catalog | `publications/papers/markdown/GIFT_v3.4_S3_observables.md` |
| Paper A (certified G₂) | `publications/papers/pdf/g2_certified_neck.pdf` |
| Paper B (spectral geometry) | `publications/papers/pdf/g2_spectral.pdf` |
| Companion notebooks | `publications/papers/notebooks/` |
| Observables data | `publications/references/observables.csv` |
| Blog posts & outreach | `publications/outreach/` |
| Formal verification | [gift-framework/core](https://github.com/gift-framework/core) |
| v3.3 archive | `publications/papers/legacy/v3.3/` |
| Technical definitions | `docs/GLOSSARY.md` |

## Core Documents (v3.4)

| Document | Content |
|----------|---------|
| GIFT_v3.4_main.md | Complete theoretical framework |
| GIFT_v3.4_S1_foundations.md | E₈, G₂, K₇ mathematical construction |
| GIFT_v3.4_S2_derivations.md | 33 Type I derivations with proofs |
| GIFT_v3.4_S3_observables.md | 95-observable catalog (35 I + 19 II + 21 III + 22 IV) |
| g2_certified_neck.md (Paper A) | Computer-assisted G₂ existence proof |
| g2_spectral.md (Paper B) | Laplacian spectrum, harmonic forms |

## Wiki

The **[GitHub Wiki](https://github.com/gift-framework/GIFT/wiki)** provides a navigable multi-audience hub consolidating all documentation, paper summaries, reference data, blog posts, and project meta.

## Related Repositories

| Repository | Content |
|------------|---------|
| [gift-framework/core](https://github.com/gift-framework/core) | Formal proofs (Lean 4), K₇ metric pipeline, giftpy |

## Version

**Current**: v3.4.20 (2026-05-10)
**Relations**: 213 conjuncts certified (core v3.4.20, 4 main-chain axioms + 11 interval-arith certificates)
**Predictions**: 95 observables (35 Type I, exact targets, 0.39% mean deviation; PDG 2024 / NuFIT 6.0)
**Validation**: 3M+ configs exhaustive (log₁₀ p_algebraic = −138)
**Papers**: Framework v3.4 (Zenodo 20070101) + A (19892350) + B (19893371) + C (19708916) + D (20039066)
