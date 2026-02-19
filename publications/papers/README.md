# GIFT Framework v3.3 - Publications

[![Lean 4 Verified](https://img.shields.io/badge/Lean_4-Verified-blue)](https://github.com/gift-framework/core/tree/main/Lean)

Geometric Information Field Theory: Deriving Standard Model parameters from E₈×E₈ topology.

---

## Documentation Structure

```
publications/
├── papers/                        # Scientific articles
│   ├── markdown/                  # Core documents (v3.3)
│   │   ├── GIFT_v3.3_main.md         # Main paper
│   │   ├── GIFT_v3.3_S1_foundations.md   # E₈, G₂, K₇ foundations
│   │   ├── GIFT_v3.3_S2_derivations.md   # 33 dimensionless derivations
│   │   └── GIFT_v3.3_S3_dynamics.md      # RG flow, torsional dynamics
│   ├── tex/                       # LaTeX sources
│   ├── pdf/                       # Generated PDFs
│   ├── FoP/                       # Foundations of Physics submission
│   └── Lean/                      # G₂ Lean formalization paper
│
├── outreach/                      # Blog posts & vulgarization
│   └── (7 Substack posts)
│
├── references/                    # Data & reference catalogs
│   ├── 39_observables.csv         # Machine-readable data
│   ├── OBSERVABLE_REFERENCE.md    # Complete observable catalog
│   ├── NUMBER_THEORETIC_STRUCTURES.md  # Fibonacci, Prime Atlas, Monster
│   ├── SPECULATIVE_PHYSICS.md     # Scale bridge, Yukawa, M-theory, QG
│   ├── STATISTICAL_EVIDENCE.md    # Rigorous statistical analysis
│   └── Bibliography.md            # References
│
└── validation/                    # Monte Carlo validation
    ├── validation_v33.py          # v3.3 comprehensive validation
    ├── comprehensive_statistics_v33.py  # Advanced statistical tests
    ├── paper1_*.py / paper2_*.py  # Paper-specific validations
    └── results/                   # Validation results
```

---

## Core Documents

### [GIFT_v3.3_main.md](markdown/GIFT_v3.3_main.md)
Complete theoretical framework - the main paper.

### [GIFT_v3.3_S1_foundations.md](markdown/GIFT_v3.3_S1_foundations.md)
Mathematical foundations: E₈ exceptional algebra, G₂ holonomy, K₇ manifold construction.

### [GIFT_v3.3_S2_derivations.md](markdown/GIFT_v3.3_S2_derivations.md)
All 33 dimensionless derivations with complete proofs.

### [GIFT_v3.3_S3_dynamics.md](markdown/GIFT_v3.3_S3_dynamics.md)
RG flow, torsional dynamics, scale bridge.

---

## Key Results

| # | Relation | Value | Status |
|---|----------|-------|--------|
| 1 | N_gen | 3 | **PROVEN** |
| 2 | τ | 3472/891 | **PROVEN** |
| 3 | det(g) | 65/32 | **PROVEN** |
| 4 | κ_T | 1/61 | **PROVEN** |
| 5 | sin²θ_W | 3/13 | **PROVEN** |
| 6 | α_s | √2/12 | TOPOLOGICAL |
| 7 | Q_Koide | 2/3 | **PROVEN** |
| 8 | m_τ/m_e | 3477 | **PROVEN** |
| 9 | m_s/m_d | 20 | **PROVEN** |
| 10 | δ_CP | 197° | **PROVEN** |

**Zero continuous adjustable parameters. Mean deviation 0.26% (PDG 2024).**

---

## Statistical Validation (v3.3)

Comprehensive Monte Carlo validation across 192,349 configurations:

| Metric | Value |
|--------|-------|
| Configurations tested | 192,349 |
| Better alternatives | 0 |
| p-value | < 5×10⁻⁶ |
| Significance | > 4.5σ |

See [`validation/`](../validation/) for methodology and results.

---

## Exploratory References

| Document | Content | Status |
|----------|---------|--------|
| [NUMBER_THEORETIC_STRUCTURES.md](../references/NUMBER_THEORETIC_STRUCTURES.md) | Fibonacci, Prime Atlas, Monster, Moonshine | Observation |
| [SPECULATIVE_PHYSICS.md](../references/SPECULATIVE_PHYSICS.md) | Scale bridge, Yukawa, M-theory, QG | Speculative |

---

## Formal Verification

**~290 relations verified** in Lean 4 (core v3.3.17).

See [gift-framework/core](https://github.com/gift-framework/core) for proofs.

---

## Legacy Documents

Historical supplements (S1-S9 v2.2/v3.0) are archived in `../../docs/legacy/`.

---

**Version**: 3.3.17 (2026-02-14)
**Repository**: https://github.com/gift-framework/GIFT
