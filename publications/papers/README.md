# GIFT Framework v3.4 — Publications

[![Lean 4 Verified](https://img.shields.io/badge/Lean_4-Verified-blue)](https://github.com/gift-framework/core/tree/main/Lean)

Geometric Information Field Theory: Standard Model parameters from G₂ holonomy geometry coupled to E₈×E₈.

---

## Documentation Structure

```
publications/
├── papers/
│   ├── markdown/                    # Canonical sources (v3.4)
│   │   ├── GIFT_v3.4_main.md            # Main paper
│   │   ├── GIFT_v3.4_S1_foundations.md  # E₈, G₂, K₇ foundations
│   │   ├── GIFT_v3.4_S2_derivations.md  # 33 Type I derivations
│   │   ├── GIFT_v3.4_S3_observables.md  # 95-observable catalog
│   │   ├── g2_certified_neck.md         # Paper A — certified G₂
│   │   └── g2_spectral.md               # Paper B — spectral
│   ├── pdf/                         # Compiled PDFs (all published)
│   │   ├── gift_3.4_main.pdf            # Framework main (Zenodo 20070101)
│   │   ├── gift_3.4_S1.pdf              # Framework S1  (Zenodo 20070101)
│   │   ├── gift_3.4_S2.pdf              # Framework S2  (Zenodo 20070101)
│   │   ├── gift_3.4_S3.pdf              # Framework S3  (Zenodo 20070101)
│   │   ├── g2_certified_neck.pdf        # Paper A (Zenodo 19892350)
│   │   ├── g2_spectral.pdf              # Paper B (Zenodo 19893371)
│   │   ├── K3_NK_Certificate.pdf        # Paper C (Zenodo 19708916)
│   │   └── donaldson_analytic.pdf       # Paper D (Zenodo 20039066)
│   ├── tex/                         # LaTeX sources (v3.4)
│   ├── figures/                     # Publication figures
│   ├── notebooks/                   # Companion Jupyter notebooks
│   └── legacy/v3.3/                 # v3.3 framework PDFs + markdown (archived)
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

## Framework v3.4 (PDF + TeX + markdown, published Zenodo [10.5281/zenodo.20070101](https://doi.org/10.5281/zenodo.20070101))

| Document | PDF | Markdown |
|----------|-----|----------|
| Main Paper (44 pp.) | [gift_3.4_main.pdf](pdf/gift_3.4_main.pdf) | [GIFT_v3.4_main.md](markdown/GIFT_v3.4_main.md) |
| S1: Foundations (27 pp.) | [gift_3.4_S1.pdf](pdf/gift_3.4_S1.pdf) | [GIFT_v3.4_S1_foundations.md](markdown/GIFT_v3.4_S1_foundations.md) |
| S2: Derivations (42 pp.) | [gift_3.4_S2.pdf](pdf/gift_3.4_S2.pdf) | [GIFT_v3.4_S2_derivations.md](markdown/GIFT_v3.4_S2_derivations.md) |
| S3: Observables (10 pp.) | [gift_3.4_S3.pdf](pdf/gift_3.4_S3.pdf) | [GIFT_v3.4_S3_observables.md](markdown/GIFT_v3.4_S3_observables.md) |

> LaTeX sources: [`tex/`](tex/).

## Companion Papers (peer-reviewable, Zenodo)

| Paper | Description | DOI |
|-------|-------------|-----|
| [A — Certified G₂ Structure](pdf/g2_certified_neck.pdf) | First computer-assisted existence proof for a metric with special holonomy | [10.5281/zenodo.19892350](https://doi.org/10.5281/zenodo.19892350) |
| [B — Spectral Geometry](pdf/g2_spectral.pdf) | Laplacian spectrum, harmonic forms, λ₁ ≈ 6π²/475 | [10.5281/zenodo.19893371](https://doi.org/10.5281/zenodo.19893371) |
| [C — K3 Newton-Kantorovich](pdf/K3_NK_Certificate.pdf) | NK diagnostics on Donaldson K3 (CI(2,2,2)) | [10.5281/zenodo.19708916](https://doi.org/10.5281/zenodo.19708916) |
| [D — Donaldson Analytic Note](pdf/donaldson_analytic.pdf) | Closed-form G₂ ansatz on K3-coassociative neck, 5-layer Wirtinger cert | [10.5281/zenodo.20039066](https://doi.org/10.5281/zenodo.20039066) |

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

## Statistical Validation (v3.4)

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

See [gift-framework/core](https://github.com/gift-framework/core) for proofs (v3.4.26).

---

## Legacy

- `legacy/v3.3/` — v3.3 framework PDFs and markdown (archived 2026-05-02)
- `../validation/legacy/v3.3/` — v3.3 validation pipeline + Pareto formula selection

---

**Version**: 3.4.26 (2026-06-03)
**Repository**: https://github.com/gift-framework/GIFT
