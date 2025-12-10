# GIFT Framework v3.0 - Publications

[![Lean 4 Verified](https://img.shields.io/badge/Lean_4-Verified-blue)](https://github.com/gift-framework/core/tree/main/Lean)
[![Coq Verified](https://img.shields.io/badge/Coq_8.18-Verified-orange)](https://github.com/gift-framework/core/tree/main/COQ)

Geometric Information Field Theory: Deriving Standard Model parameters from E₈×E₈ topology.

---

## What's New in v3.0: Publication Restructuring

Version 3.0 restructures publications to clearly distinguish:

1. **Zenodo-ready** (`zenodo/`): Rigorous dimensionless predictions with formal verification
2. **Exploratory** (`exploratory/`): Speculative content (transparent but not for publication)
3. **Archive** (`markdown/`): Historical versions preserved for reference

---

## Documentation Structure

```
publications/
├── README.md                           # This file
│
├── zenodo/                             # ★ OFFICIAL PUBLICATIONS ★
│   ├── GIFT_v3_main.md                # Core paper (~25 pages, 18 relations)
│   ├── GIFT_v3_S1_foundations.md      # Mathematical foundations (E₈+G₂+K₇)
│   └── GIFT_v3_S2_derivations.md      # Complete derivations (dimensionless)
│
├── exploratory/                        # ⚠️ SPECULATIVE CONTENT ⚠️
│   ├── theoretical_extensions.md      # QG, info theory, M-theory
│   ├── dimensional_observables.md     # Absolute masses (heuristic)
│   ├── sequences_prime_atlas.md       # Fibonacci, Lucas, primes
│   └── monster_moonshine.md           # Monster group connections
│
├── markdown/                           # Archive (v23/v30 versions)
│   ├── gift_2_3_main.md               # Historical
│   ├── gift_3_0_main.md               # Historical
│   └── S1-S9_*_v23/v30.md             # Historical supplements
│
├── references/                         # Quick reference docs
└── Lean/                               # Formal proofs (unchanged)
```

---

## Key Results (v3.0)

### 18 PROVEN Dimensionless Relations

| # | Relation | Formula | Value | Status |
|---|----------|---------|-------|--------|
| 1 | N_gen | Atiyah-Singer | 3 | **PROVEN** |
| 2 | τ | 496×21/(27×99) | 3472/891 | **PROVEN** |
| 3 | det(g) | p₂ + 1/32 | 65/32 | **PROVEN** |
| 4 | κ_T | 1/(b₃-dim(G₂)-p₂) | 1/61 | **PROVEN** |
| 5 | sin²θ_W | b₂/(b₃+dim(G₂)) | 3/13 | **PROVEN** |
| 6 | α_s | √2/(dim(G₂)-p₂) | √2/12 | TOPOLOGICAL |
| 7 | Q_Koide | dim(G₂)/b₂ | 2/3 | **PROVEN** |
| 8 | m_τ/m_e | 7+10×248+10×99 | 3477 | **PROVEN** |
| 9 | m_s/m_d | p₂²×Weyl | 20 | **PROVEN** |
| 10 | δ_CP | dim(K₇)×dim(G₂)+H* | 197° | **PROVEN** |
| 11 | θ₁₃ | π/b₂ | π/21 | TOPOLOGICAL |
| 12 | θ₂₃ | (rank+b₃)/H* | 85/99 rad | TOPOLOGICAL |
| 13 | λ_H | √(dim(G₂)+N_gen)/2^Weyl | √17/32 | **PROVEN** |
| 14 | Ω_DE | ln(p₂)×(b₂+b₃)/H* | ln(2)×98/99 | **PROVEN** |
| 15 | n_s | ζ(D_bulk)/ζ(Weyl) | ζ(11)/ζ(5) | **PROVEN** |
| 16 | m_μ/m_e | dim(J₃(O))^φ | 27^φ | TOPOLOGICAL |
| 17 | θ₁₂ | arctan(√(δ/γ)) | 33.42° | TOPOLOGICAL |
| 18 | α⁻¹ | 128+9+det(g)×κ_T | 137.033 | TOPOLOGICAL |

**Zero continuous adjustable parameters. Mean deviation 0.197%.**

---

## Reading Guide

### For Quick Overview (5 min)

Read this README + [zenodo/GIFT_v3_main.md](zenodo/GIFT_v3_main.md) Abstract and Section 10.

### For Understanding the Framework (2 hrs)

1. [zenodo/GIFT_v3_main.md](zenodo/GIFT_v3_main.md) - Full paper
2. [zenodo/GIFT_v3_S2_derivations.md](zenodo/GIFT_v3_S2_derivations.md) - All 18 proofs

### For Mathematical Details (Half day)

1. [zenodo/GIFT_v3_S1_foundations.md](zenodo/GIFT_v3_S1_foundations.md) - E₈, G₂, K₇ construction
2. Lean proofs at [gift-framework/core](https://github.com/gift-framework/core)

### For Exploratory Content (Research)

⚠️ **Warning**: Content in `exploratory/` is speculative and not peer-reviewed.

- [exploratory/sequences_prime_atlas.md](exploratory/sequences_prime_atlas.md) - Fibonacci patterns
- [exploratory/monster_moonshine.md](exploratory/monster_moonshine.md) - Monster group
- [exploratory/dimensional_observables.md](exploratory/dimensional_observables.md) - Absolute masses
- [exploratory/theoretical_extensions.md](exploratory/theoretical_extensions.md) - QG, M-theory

---

## Zenodo vs Exploratory: What's the Difference?

### Zenodo Publications (✓ Rigorous)

| Content | Status | Publication |
|---------|--------|-------------|
| 18 dimensionless relations | **PROVEN (Lean)** | Zenodo-ready |
| E₈×E₈ → K₇ architecture | Established math | Zenodo-ready |
| Joyce theorem application | **PROVEN** | Zenodo-ready |
| Experimental falsification | Defined | Zenodo-ready |

### Exploratory Content (⚠️ Speculative)

| Content | Status | Publication |
|---------|--------|-------------|
| Absolute masses (GeV/MeV) | Heuristic | Repo only |
| Fibonacci/Lucas patterns | Observation | Repo only |
| Monster group connections | Speculative | Repo only |
| M-theory embedding | Theoretical | Repo only |
| Quantum gravity | Speculative | Repo only |

---

## Important Limitations

### What GIFT Predicts vs. Assumes

**Predicted** (dimensionless):
- All mass ratios
- Gauge couplings at M_Z
- Mixing angles and phases
- Cosmological ratios

**Assumed** (structural choices):
- E₈×E₈ gauge group
- K₇ with b₂=21, b₃=77
- G₂ holonomy

### Epistemic Status

| Layer | Status | Confidence |
|-------|--------|------------|
| Core predictions (18 relations) | Falsifiable | High |
| Structural relations | Derived | Medium |
| Number-theoretic patterns | Exploratory | Low |
| Monster/Moonshine | Highly speculative | Very low |

---

## Falsification Protocol

| Prediction | Test | Timeline | Criterion |
|------------|------|----------|-----------|
| δ_CP = 197° | DUNE | 2027-2030 | Outside [187°, 207°] |
| sin²θ_W = 3/13 | FCC-ee | 2040s | Outside [0.2295, 0.2320] |
| m_s/m_d = 20 | Lattice QCD | 2030 | Converges outside [19, 21] |
| N_gen = 3 | LHC | Ongoing | Fourth generation |

---

## Formal Verification

**165+ relations verified** in Lean 4 + Coq (dual verification).

```
Lean 4.14.0 + Mathlib 4.14.0: 0 sorry, 0 domain axioms
Coq 8.18: 0 Admitted, 0 explicit axioms
```

See [gift-framework/core](https://github.com/gift-framework/core) for proofs.

---

## Changelog v3.0

- **Restructured** publications into zenodo/ and exploratory/
- **Condensed** main paper to focus on 18 PROVEN dimensionless relations
- **Merged** S1 (E₈ architecture) + S2 (K₇ construction) into S1_foundations
- **Refactored** S4 (derivations) to exclude dimensional masses
- **Moved** S6, S7, S8, S9 to exploratory/ with clear warnings
- **Added** explicit status headers to all exploratory content

---

**Version**: 3.0
**Last Updated**: 2025-12-10
**Repository**: https://github.com/gift-framework/GIFT
**Formal Proofs**: https://github.com/gift-framework/core
