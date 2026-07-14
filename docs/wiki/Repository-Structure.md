---
title: "Repository Structure"
layout: default
---


This repository contains the theoretical documentation for K₇ v3.5.

## Directory Layout

```
K₇/
├── publications/                      # Published content & validation
│   ├── papers/                        # Scientific articles
│   │   ├── markdown/                  # Core documents (v3.5 — canonical sources)
│   │   │   ├── k7_framework_3_5_main.md            # Main paper
│   │   │   ├── k7_framework_3_5_S1_foundations.md  # E₈, G₂, K₇ foundations
│   │   │   ├── k7_framework_3_5_S2_derivations.md  # 33 Type I derivations
│   │   │   ├── k7_framework_3_5_S3_observables.md  # 95-observable catalog
│   │   │   ├── k7_framework_3_5_S4_sieve_diagnostics.md  # Sieve diagnostics
│   │   │   ├── g2_certified_neck.md           # Paper A — certified G₂ structure
│   │   │   └── g2_spectral.md                 # Paper B — spectral geometry
│   │   ├── tex/                       # LaTeX sources (v3.5)
│   │   ├── pdf/                       # Compiled PDFs (all published)
│   │   │   ├── k7_framework_3_5_main.pdf      # Framework (Zenodo 21296168)
│   │   │   ├── k7_framework_3_5_S1_foundations.pdf … S4_sieve_diagnostics.pdf
│   │   │   ├── g2_certified_neck.pdf          # Paper A (Zenodo 19892350)
│   │   │   ├── g2_spectral.pdf                # Paper B (Zenodo 19893371)
│   │   │   ├── K3_NK_Certificate.pdf          # Paper C (Zenodo 19708916)
│   │   │   ├── donaldson_analytic.pdf         # Paper D (Zenodo 20039066)
│   │   │   └── rank_one_branched_adiabatic.pdf (+ _supplement)  # Paper E (Zenodo 21209413)
│   │   ├── legacy/v3.4/               # v3.4 framework PDFs + markdown + tex (archived)
│   │   ├── legacy/v3.3/               # v3.3 framework PDFs + markdown (archived)
│   │   ├── figures/                   # Publication figures (PDF + PNG)
│   │   └── notebooks/                 # Companion Jupyter notebooks
│   │       ├── g2_certified_neck_companion.ipynb
│   │       └── g2_spectral_companion.ipynb
│   ├── outreach/                      # Vulgarization & blog posts
│   ├── references/                    # Data & reference catalogs
│   │   ├── K₇_ATLAS.json            # Canonical structured atlas
│   │   ├── observables.csv            # Machine-readable observables
│   │   ├── OBSERVABLE_REFERENCE.md    # Complete observable catalog
│   │   ├── STATISTICAL_EVIDENCE.md    # Rigorous statistical analysis
│   │   ├── INDEPENDENT_VALIDATIONS.md # External research converging with K₇
│   │   └── Bibliography.md            # References
│   └── validation/                    # Monte Carlo validation
│       └── legacy/v3.3/               # v3.3 validation scripts (archived;
│                                      # v3.4 stats refresh in K7-Lean/private)
│
├── docs/                              # User-facing documentation + Jekyll site
│   ├── index.html                     # Landing page (arithmon.github.io/K7)
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
| Complete theory | `publications/papers/markdown/k7_framework_3_5_main.md` |
| All derivations | `publications/papers/markdown/k7_framework_3_5_S2_derivations.md` |
| Observable catalog | `publications/papers/markdown/k7_framework_3_5_S3_observables.md` |
| Paper A (certified G₂) | `publications/papers/pdf/g2_certified_neck.pdf` |
| Paper B (spectral geometry) | `publications/papers/pdf/g2_spectral.pdf` |
| Companion notebooks | `publications/papers/notebooks/` |
| Observables data | `publications/references/observables.csv` |
| Blog posts & outreach | `publications/outreach/` |
| Formal verification | [Arithmon/K7-Lean](https://github.com/Arithmon/K7-Lean) |
| v3.3 archive | `publications/papers/legacy/v3.3/` |
| Technical definitions | `docs/GLOSSARY.md` |

## Core Documents (v3.5)

| Document | Content |
|----------|---------|
| k7_framework_3_5_main.md | Complete theoretical framework |
| k7_framework_3_5_S1_foundations.md | E₈, G₂, K₇ mathematical construction |
| k7_framework_3_5_S2_derivations.md | 33 Type I derivations with proofs |
| k7_framework_3_5_S3_observables.md | 95-observable catalog (33 I + 19 II + 21 III + 22 IV) |
| k7_framework_3_5_S4_sieve_diagnostics.md | Sieve diagnostics supplement |
| g2_certified_neck.md (Paper A) | Computer-assisted G₂ existence proof |
| g2_spectral.md (Paper B) | Laplacian spectrum, harmonic forms |

## Wiki

The **[GitHub Wiki](https://github.com/Arithmon/K7/wiki)** provides a navigable multi-audience hub consolidating all documentation, paper summaries, reference data, blog posts, and project meta.

## Related Repositories

| Repository | Content |
|------------|---------|
| [Arithmon/K7-Lean](https://github.com/Arithmon/K7-Lean) | Formal proofs (Lean 4), K₇ metric pipeline, giftpy |

## Version

**Current**: v3.5 (2026-07-10)
**Relations**: 140 conjuncts certified (core v3.4.29, 15 axioms (4 main-chain + 11 interval-arithmetic))
**Predictions**: 95 observables (33 Type I, exact targets, 0.99% mean deviation; NuFIT 6.1 / PDG 2024 / Planck 2018 / CODATA 2022)
**Validation**: 3,000,000 random algebraic formula sets, none reproduces the joint profile (set-level bound ~10⁻⁶, no independence assumption)
**Papers**: Framework v3.5 (Zenodo 21296168) + A (19892350) + B (19893371) + C (19708916) + D (20039066) + E (21209413)
