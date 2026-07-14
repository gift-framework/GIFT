---
title: "Lean Formalization"
layout: default
---

# Lean Formalization

## Overview

The GIFT framework is formally verified in **Lean 4** with Mathlib. The formalization establishes that all claimed algebraic relations between topological inputs and physical predictions follow by pure computation.

| Metric | Value |
|--------|-------|
| **Lean files** | 143 |
| **Build jobs** | 8391 |
| **Axioms** | 15 (4 main-chain + 11 K3 interval-arithmetic) |
| **sorry statements** | 0 |
| **Warnings** | 0 |
| **Certificate conjuncts** | 140 (across the Foundations / Predictions / Spectral pillars) |

## Repository

**Code**: [github.com/gift-framework/core](https://github.com/gift-framework/core)
**Blueprint**: [gift-framework.github.io/core](https://gift-framework.github.io/core/)
**Lean version**: 4.29.0
**Mathlib version**: 4.29.0

## Architecture

```
core/Lean/GIFT/
├── Core.lean              # Constants (dim_E8, b2, b3, H*, ...)
├── Certificate.lean       # Master theorem (127 conjuncts)
├── Foundations/            # E8 roots, G2 cross product
├── Geometry/               # Differential geometry
├── Spectral/               # Spectral theory, mass gap, Yukawa
├── Relations/              # Physical predictions
│   ├── GaugeSector.lean
│   ├── NeutrinoSector.lean
│   ├── LeptonSector.lean
│   ├── YukawaDuality.lean
│   └── ...
├── ExplicitG2Metric.lean   # 400 lines
├── NewtonKantorovich.lean   # 401 lines, 0 axioms
├── K3Harmonic.lean          # 447 lines
├── K7Orthonormality.lean    # 0 axioms, 13 theorems
├── TCSGaugeBreaking.lean    # 0 axioms, 14 theorems
├── GaugeBundleData.lean     # 0 axioms, 12 theorems
├── AssociativeVolumes.lean  # 0 axioms, 19 theorems
├── CompactificationCorrection.lean  # δ_CP = 12214/69
└── ComputedWeylLaw.lean     # 0 axioms, 8 theorems
```

## Key Certificates

### Master Certificate (127 conjuncts)

The `GIFT_framework_certified` theorem verifies all relations in one compile:

- **Foundations** (34): E₈ dimensions, G₂ structure, K₇ topology, Weyl group
- **Predictions** (56): Gauge, lepton, quark, neutrino, CKM, boson, cosmological
- **Spectral** (37): Mass gap, Yukawa ratios, Weyl law, orthonormality

### Zero-Axiom Modules

Several modules prove their results with **zero domain-specific axioms**:

| Module | Axioms | Theorems | Certificate |
|--------|--------|----------|-------------|
| NewtonKantorovich | 0 |, | NK convergence |
| TCSGaugeBreaking | 0 | 14 | 10 conjuncts |
| GaugeBundleData | 0 | 12 | 11 conjuncts |
| AssociativeVolumes | 0 | 19 | 14 conjuncts |
| CompactificationCorrection | 0 | 12 | 6 conjuncts |
| ComputedWeylLaw | 0 | 8 | 7 conjuncts |
| K7Orthonormality | 0 | 13 | Gram matrices |

## Axiom Audit

All 7 axioms are substantive (standard mathematical theorems or GIFT conjectures). Zero are placeholders. Categories:

- **Topological inputs**: b₂ = 21, b₃ = 77, dim(G₂) = 14, etc.
- **E₈ properties**: Root system, Weyl group factorization
- **Joyce/anistropic existence input**: compact torsion-free `G_2` existence on
  `K_7` is an analytic assumption/target, tracked in `docs/analytic_status.md`
- **Spectral bounds**: Numerical results from certified computation

## Verification

To verify all proofs locally:

```bash
git clone https://github.com/gift-framework/core
cd core/Lean
lake build
```

Completes in ~30 seconds (warm cache). CI rebuilds on every commit.

## What is Proven vs Not Proven

**Proven**: Given topological inputs (b₂, b₃, dim(G₂), etc.), all 127 algebraic relations follow by pure computation. Zero domain-specific axioms for the arithmetic.

**Not proven**: (1) Existence of K₇ with these specific Betti numbers is axiomatized, not constructed from scratch. (2) Physical interpretation of the relations. (3) Uniqueness of the construction.

---

## Related

- [For Formalization Experts](For-Formalization-Experts.html): Methodology and context
- [Paper Main Framework](Paper-Main-Framework.html): Physics predictions these proofs certify
- [Paper S2 Derivations](Paper-S2-Derivations.html): The derivations being formalized
