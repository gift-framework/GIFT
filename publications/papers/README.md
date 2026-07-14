# The K₇ Framework v3.5 — Publications

[![Lean 4 Verified](https://img.shields.io/badge/Lean_4-Verified-blue)](https://github.com/Arithmon/K7-Lean/tree/main/Lean)

The K₇ Framework (formerly GIFT): Standard Model parameters from G₂ holonomy geometry coupled to E₈×E₈.

---

## Documentation Structure

```
publications/
├── papers/
│   ├── markdown/                    # Canonical sources (v3.5)
│   │   ├── k7_framework_3_5_main.md             # Main paper
│   │   ├── k7_framework_3_5_S1_foundations.md   # E₈, G₂, K₇ foundations
│   │   ├── k7_framework_3_5_S2_derivations.md   # Type I derivations
│   │   ├── k7_framework_3_5_S3_observables.md   # Observable catalog
│   │   ├── k7_framework_3_5_S4_sieve_diagnostics.md # Sieve diagnostics
│   │   ├── g2_certified_neck.md         # Paper A — certified G₂
│   │   └── g2_spectral.md               # Paper B — spectral
│   ├── pdf/                         # Compiled PDFs (all published)
│   │   ├── k7_framework_3_5_main.pdf              # Framework main (Zenodo 21296168)
│   │   ├── k7_framework_3_5_S1_foundations.pdf    # Framework S1  (Zenodo 21296168)
│   │   ├── k7_framework_3_5_S2_derivations.pdf    # Framework S2  (Zenodo 21296168)
│   │   ├── k7_framework_3_5_S3_observables.pdf    # Framework S3  (Zenodo 21296168)
│   │   ├── k7_framework_3_5_S4_sieve_diagnostics.pdf # Framework S4 (Zenodo 21296168)
│   │   ├── g2_certified_neck.pdf        # Paper A (Zenodo 19892350)
│   │   ├── g2_spectral.pdf              # Paper B (Zenodo 19893371)
│   │   ├── K3_NK_Certificate.pdf        # Paper C (Zenodo 19708916)
│   │   ├── donaldson_analytic.pdf       # Paper D (Zenodo 20039066)
│   │   └── rank_one_branched_adiabatic.pdf # Paper E (Zenodo 21209413)
│   ├── tex/                         # LaTeX sources (v3.5)
│   ├── figures/                     # Publication figures
│   ├── notebooks/                   # Companion Jupyter notebooks
│   ├── legacy/v3.3/                 # v3.3 framework PDFs + markdown + tex (archived)
│   └── legacy/v3.4/                 # v3.4 framework PDFs + markdown + tex (archived)
│
├── outreach/                        # Substack mirror (essays archive)
│
├── references/
│   ├── observables.csv              # Machine-readable data
│   ├── OBSERVABLE_REFERENCE.md
│   ├── STATISTICAL_EVIDENCE.md
│   └── Bibliography.md
│
└── validation/
    ├── README.md                    # v3.4 headline + pointer to legacy v3.3
    └── legacy/v3.3/                 # v3.3 validation pipeline (archived)
```

---

## Framework v3.5 (PDF + TeX + markdown, published Zenodo [10.5281/zenodo.21296168](https://doi.org/10.5281/zenodo.21296168))

| Document | PDF | Markdown |
|----------|-----|----------|
| Main Paper (47 pp.) | [k7_framework_3_5_main.pdf](pdf/k7_framework_3_5_main.pdf) | [k7_framework_3_5_main.md](markdown/k7_framework_3_5_main.md) |
| S1: Foundations | [k7_framework_3_5_S1_foundations.pdf](pdf/k7_framework_3_5_S1_foundations.pdf) | [k7_framework_3_5_S1_foundations.md](markdown/k7_framework_3_5_S1_foundations.md) |
| S2: Derivations | [k7_framework_3_5_S2_derivations.pdf](pdf/k7_framework_3_5_S2_derivations.pdf) | [k7_framework_3_5_S2_derivations.md](markdown/k7_framework_3_5_S2_derivations.md) |
| S3: Observables | [k7_framework_3_5_S3_observables.pdf](pdf/k7_framework_3_5_S3_observables.pdf) | [k7_framework_3_5_S3_observables.md](markdown/k7_framework_3_5_S3_observables.md) |
| S4: Sieve Diagnostics | [k7_framework_3_5_S4_sieve_diagnostics.pdf](pdf/k7_framework_3_5_S4_sieve_diagnostics.pdf) | [k7_framework_3_5_S4_sieve_diagnostics.md](markdown/k7_framework_3_5_S4_sieve_diagnostics.md) |

> LaTeX sources: [`tex/`](tex/). Concept DOI (always latest version): [10.5281/zenodo.16891489](https://doi.org/10.5281/zenodo.16891489).

## Companion Papers (peer-reviewable, Zenodo)

| Paper | Description | DOI |
|-------|-------------|-----|
| [A — Certified G₂ Structure](pdf/g2_certified_neck.pdf) | First computer-assisted existence proof for a metric with special holonomy | [10.5281/zenodo.19892350](https://doi.org/10.5281/zenodo.19892350) |
| [B — Spectral Geometry](pdf/g2_spectral.pdf) | Laplacian spectrum, harmonic forms, λ₁ ≈ 6π²/475 | [10.5281/zenodo.19893371](https://doi.org/10.5281/zenodo.19893371) |
| [C — K3 Newton-Kantorovich](pdf/K3_NK_Certificate.pdf) | NK diagnostics on Donaldson K3 (CI(2,2,2)) | [10.5281/zenodo.19708916](https://doi.org/10.5281/zenodo.19708916) |
| [D — Donaldson Analytic Note](pdf/donaldson_analytic.pdf) | Closed-form G₂ ansatz on K3-coassociative neck, 5-layer Wirtinger cert | [10.5281/zenodo.20039066](https://doi.org/10.5281/zenodo.20039066) |
| [E — Rank-1 Branched Adiabatic](pdf/rank_one_branched_adiabatic.pdf) | Neck-level analytic machinery closed at D₀ (open: (J) + H_global) | [10.5281/zenodo.21209413](https://doi.org/10.5281/zenodo.21209413) |

---

## Headline Results

| # | Relation | Value | Status |
|---|----------|-------|--------|
| 1 | N_gen | 3 | **PROVEN** (Lean) |
| 2 | τ | 3472/891 | **PROVEN** |
| 3 | det(g) | 65/32 | **PROVEN** (construction) |
| 4 | κ_T | 1/61 | **PROVEN** |
| 5 | sin²θ_W | 3/13 | **PROVEN** |
| 6 | α_s(M_Z) | √2/12 | TOPOLOGICAL |
| 7 | Q_Koide | 2/3 | **PROVEN** |
| 8 | m_τ/m_e | 3477 | **PROVEN** |
| 9 | m_s/m_d | 20 | **PROVEN** |
| 10 | δ_CP | 197° | **PROVEN** |

**Zero continuous adjustable parameters. 3 integer primitives (N=3, r₈=8, r₂=2). Mean deviation 0.99% on 33 Type I exact-target relations (PDG 2024 / NuFIT 6.1).**

---

## Statistical Validation

| Metric | Value |
|--------|-------|
| Configurations tested | 3,000,000+ |
| Better alternatives | 0 |
| Algebraic null model | set-level ~10⁻⁶ (assumption-free); log₁₀ p = −134 |
| Lean certificate | 140 conjuncts, 15 axioms (4 main-chain + 11 interval-arithmetic), 0 sorry |

See [`validation/`](../validation/) and [`STATISTICAL_EVIDENCE.md`](../references/STATISTICAL_EVIDENCE.md) for methodology. The v3.3 validation pipeline (3,070,396-config exhaustive + 7-component bullet-proof) is archived in [`validation/legacy/v3.3/`](../validation/legacy/v3.3/).

---

## Formal Verification

143 Lean 4 files, 140 conjuncts certified, 15 axioms (4 main-chain + 11 interval-arithmetic certificates), 0 sorry, 8391 build jobs.

See [Arithmon/K7-Lean](https://github.com/Arithmon/K7-Lean) for proofs (v3.4.29).

---

## Legacy

- `legacy/v3.3/` — v3.3 framework PDFs, markdown and tex (archived 2026-05-02)
- `legacy/v3.4/` — v3.4 framework PDFs, markdown and tex (archived 2026-07-14, Zenodo [10.5281/zenodo.20070101](https://doi.org/10.5281/zenodo.20070101))
- `../validation/legacy/v3.3/` — v3.3 validation pipeline + Pareto formula selection

---

**Version**: 3.5 (2026-07-10)
**Repository**: https://github.com/Arithmon/K7
