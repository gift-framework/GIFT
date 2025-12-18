# GIFT Framework v3.1 - Publications

[![Lean 4 Verified](https://img.shields.io/badge/Lean_4-Verified-blue)](https://github.com/gift-framework/core/tree/main/Lean)
[![Coq Verified](https://img.shields.io/badge/Coq_8.18-Verified-orange)](https://github.com/gift-framework/core/tree/main/COQ)

Geometric Information Field Theory: Deriving Standard Model parameters from E₈×E₈ topology.

---

## Documentation Structure

```
publications/
├── markdown/                    # Core documents (v3.1)
│   ├── GIFT_v3_main.md         # Main paper
│   ├── GIFT_v3_S1_foundations.md   # E₈, G₂, K₇ foundations
│   ├── GIFT_v3_S2_derivations.md   # 18 dimensionless derivations
│   └── GIFT_v3_S3_dynamics.md      # RG flow, torsional dynamics
│
├── references/                  # Exploratory & reference docs
│   ├── 39_observables.csv      # Machine-readable data
│   ├── NUMBER_THEORETIC_STRUCTURES.md  # Fibonacci, Prime Atlas, Monster
│   └── SPECULATIVE_PHYSICS.md  # Scale bridge, Yukawa, M-theory, QG
│
├── Lean/                        # Lean formalization docs
├── tex/                         # LaTeX sources
└── pdf/                         # Generated PDFs
```

---

## Core Documents

### [GIFT_v3_main.md](markdown/GIFT_v3_main.md)
Complete theoretical framework - the main paper.

### [GIFT_v3_S1_foundations.md](markdown/GIFT_v3_S1_foundations.md)
Mathematical foundations: E₈ exceptional algebra, G₂ holonomy, K₇ manifold construction.

### [GIFT_v3_S2_derivations.md](markdown/GIFT_v3_S2_derivations.md)
All 18 dimensionless derivations with complete proofs.

### [GIFT_v3_S3_dynamics.md](markdown/GIFT_v3_S3_dynamics.md)
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

**Zero continuous adjustable parameters. Mean deviation 0.087%.**

---

## Exploratory References

⚠️ **Note**: Content in `references/` beyond the CSV is exploratory/speculative.

| Document | Content | Status |
|----------|---------|--------|
| [NUMBER_THEORETIC_STRUCTURES.md](references/NUMBER_THEORETIC_STRUCTURES.md) | Fibonacci, Prime Atlas, Monster, Moonshine | Observation |
| [SPECULATIVE_PHYSICS.md](references/SPECULATIVE_PHYSICS.md) | Scale bridge, Yukawa, M-theory, QG | Speculative |

---

## Formal Verification

**180+ relations verified** in Lean 4 + Coq (dual verification).

See [gift-framework/core](https://github.com/gift-framework/core) for proofs.

---

## Legacy Documents

Historical supplements (S1-S9 v2.3/v3.0) are archived in `../docs/legacy/`.

---

**Version**: 3.1.1 (2025-12-17)
**Repository**: https://github.com/gift-framework/GIFT
