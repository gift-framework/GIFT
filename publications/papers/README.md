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
│   │   └── Numerical_G2_Metric.md        # PINN-based G₂ metric construction
│   ├── tex/                       # LaTeX sources
│   └── pdf/                       # Generated PDFs
│
├── outreach/                      # Blog posts & vulgarization
│   └── (7 Substack posts)
│
├── references/                    # Data & reference catalogs
│   ├── observables.csv            # Machine-readable data
│   ├── OBSERVABLE_REFERENCE.md    # Complete observable catalog
│   ├── NUMBER_THEORETIC_STRUCTURES.md  # Fibonacci, Prime Atlas, Monster
│   ├── SPECULATIVE_PHYSICS.md     # Scale bridge, Yukawa, M-theory, QG
│   ├── STATISTICAL_EVIDENCE.md    # Rigorous statistical analysis
│   └── Bibliography.md            # References
│
└── validation/                    # Monte Carlo validation (v3.3 only)
    ├── validation_v33.py          # Core formulas & experimental data
    ├── bulletproof_validation_v33.py    # 7-component bullet-proof validation
    ├── exhaustive_validation_v33.py     # Exhaustive search (3M+ configs)
    ├── comprehensive_statistics_v33.py  # Advanced statistical tests
    └── selection/                 # Formula selection & Pareto analysis
```

---

## Core Documents

### [GIFT_v3.3_main.md](markdown/GIFT_v3.3_main.md)
Complete theoretical framework - the main paper.

### [GIFT_v3.3_S1_foundations.md](markdown/GIFT_v3.3_S1_foundations.md)
Mathematical foundations: E₈ exceptional algebra, G₂ holonomy, K₇ manifold construction.

### [GIFT_v3.3_S2_derivations.md](markdown/GIFT_v3.3_S2_derivations.md)
All 33 dimensionless derivations with complete proofs.

### [Numerical_G2_Metric.md](markdown/Numerical_G2_Metric.md)
PINN-based G₂ metric construction (companion numerical paper).

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

**Zero continuous adjustable parameters. Mean deviation 0.24% across 32 well-measured observables (0.57% incl. δ_CP; PDG 2024 / NuFIT 6.0).**

---

## Statistical Validation (v3.3)

Exhaustive search (3,070,396 configs) + seven-component bullet-proof analysis:

| Metric | Value |
|--------|-------|
| Configurations tested | 3,070,396 |
| Better alternatives | 0 |
| Null model p-value | < 2×10⁻⁵ (σ > 4.2) |
| Westfall-Young maxT | 11/33 significant (global p = 0.008) |
| Bayes factor | 304–4,738 (decisive) |

See [`validation/`](../validation/) and [`STATISTICAL_EVIDENCE.md`](../references/STATISTICAL_EVIDENCE.md) for methodology.

---

## Exploratory References

| Document | Content | Status |
|----------|---------|--------|
| [NUMBER_THEORETIC_STRUCTURES.md](../references/NUMBER_THEORETIC_STRUCTURES.md) | Fibonacci, Prime Atlas, Monster, Moonshine | Observation |
| [SPECULATIVE_PHYSICS.md](../references/SPECULATIVE_PHYSICS.md) | Scale bridge, Yukawa, M-theory, QG | Speculative |

---

## Formal Verification

**2400+ theorems verified** in Lean 4 (core v3.3.24).

See [gift-framework/core](https://github.com/gift-framework/core) for proofs.

---

## Legacy Documents

Historical supplements (S1-S9 v2.2/v3.0) are archived in `../../docs/legacy/`.

---

**Version**: 3.3.24 (2026-03-02)
**Repository**: https://github.com/gift-framework/GIFT
