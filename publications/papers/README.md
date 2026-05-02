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
│   ├── pdf/                         # Compiled PDFs (Papers A, B published)
│   │   ├── g2_certified_neck.pdf        # Paper A (Zenodo 19892350)
│   │   └── g2_spectral.pdf              # Paper B (Zenodo 19893371)
│   ├── tex/                         # LaTeX sources (v3.4 recompile pending)
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

## Framework v3.4 (canonical in markdown)

| Document | Description |
|----------|-------------|
| [GIFT_v3.4_main.md](markdown/GIFT_v3.4_main.md) | Complete theoretical framework |
| [GIFT_v3.4_S1_foundations.md](markdown/GIFT_v3.4_S1_foundations.md) | E₈, G₂, K₇ mathematical construction |
| [GIFT_v3.4_S2_derivations.md](markdown/GIFT_v3.4_S2_derivations.md) | 33 Type I derivations with proofs |
| [GIFT_v3.4_S3_observables.md](markdown/GIFT_v3.4_S3_observables.md) | 95-observable catalog (35 I + 19 II + 21 III + 22 IV) |

> v3.4 PDFs are being recompiled with the GIFT branding template. Markdown is canonical until then.

## Triptyque (peer-reviewable, Zenodo)

| Paper | Description | DOI |
|-------|-------------|-----|
| [A — Certified G₂ Structure](pdf/g2_certified_neck.pdf) | First computer-assisted existence proof for a metric with special holonomy | [10.5281/zenodo.19892350](https://doi.org/10.5281/zenodo.19892350) |
| [B — Spectral Geometry](pdf/g2_spectral.pdf) | Laplacian spectrum, harmonic forms, λ₁ ≈ 6π²/475 | [10.5281/zenodo.19893371](https://doi.org/10.5281/zenodo.19893371) |
| C — Newton-Kantorovich on K3 | NK diagnostics on Donaldson K3 (CI(2,2,2)) | [10.5281/zenodo.19708916](https://doi.org/10.5281/zenodo.19708916) |

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

**Zero continuous adjustable parameters. 3 integer primitives (N=3, r₈=8, r₂=2). Mean deviation 0.39% on 35 Type I exact-target relations (PDG 2024 / NuFIT 6.0).**

---

## Statistical Validation (v3.4)

| Metric | Value |
|--------|-------|
| Configurations tested | 3,000,000+ |
| Better alternatives | 0 |
| Algebraic null model | log₁₀ p = −138 |
| Lean certificate | 213 conjuncts, 4 main-chain axioms, 0 sorry |

See [`validation/`](../validation/) and [`STATISTICAL_EVIDENCE.md`](../references/STATISTICAL_EVIDENCE.md) for methodology. The v3.3 validation pipeline (3,070,396-config exhaustive + 7-component bullet-proof) is archived in [`validation/legacy/v3.3/`](../validation/legacy/v3.3/).

---

## Formal Verification

132 Lean 4 files, 213 conjuncts certified, 4 main-chain axioms (15 total incl. interval-arithmetic certificates), 0 sorry.

See [gift-framework/core](https://github.com/gift-framework/core) for proofs (v3.4.13).

---

## Legacy

- `legacy/v3.3/` — v3.3 framework PDFs and markdown (archived 2026-05-02)
- `../validation/legacy/v3.3/` — v3.3 validation pipeline + Pareto formula selection

---

**Version**: 3.4.13 (2026-04-29)
**Repository**: https://github.com/gift-framework/GIFT
