---
title: "Changelog Highlights"
layout: default
---

# Changelog Highlights

Abridged version history. For the full changelog, see [`CHANGELOG.md`](https://github.com/gift-framework/GIFT/blob/main/CHANGELOG.md).

---

## v3.4.13, 2026-04-29

**Triptyque published & axiom reduction milestone**

- **Papers A, B, C published on Zenodo** (DOIs 19892350, 19893371, 19708916): the peer-reviewable companion triptyque (certified G₂ structure, spectral geometry, K3 NK diagnostics)
- **Lean axiom reduction**: 38 → 4 main-chain axioms (15 total incl. interval-arithmetic certificates), 0 sorry, 213 conjuncts certified
- **K3NewtonKantorovich v3.0 hardcore**: η ×2.4 tighter, Joyce margin ×17 below ε₀
- **γ² = 24π²/7 derived** (T² Hodge Laplacian + H¹(K3)=0; was PSLQ artifact 135/4)
- **95 observables**: 35 Type I + 19 Type II + 21 Type III + 22 Type IV; 0.39% mean deviation on Type I (PDG 2024 / NuFIT 6.0)
- **3 integer primitives**: N=3, r₈=8, r₂=2 (no continuous adjustable parameters)

## v3.4.3, 2026-04

**G₂ Mathlib steps 1-5 promoted to theorems**

- φ₀ ordered 3-form on ℝ⁷, Bryant identity ∑φ₀² = 6δ, rank=35 → dim(g₂)=14, B = 144δ, det·gram theorem (Aristotle)
- 8 → 4 axioms by promoting algebraic identities to native_decide proofs
- MollifiedSum archived; G₂ThreeForm axiomized cleanly

## v3.4.0, 2026-04

**Metric-first program complete · K3 CAP**

- Computer-assisted proof of torsion-free G₂ metric existence on TCS neck model: h ≤ 8.95×10⁻⁹, ×56 million margin below Joyce ε₀
- K3 NK certificates: Fermat quartic ×990, CI(2,2,2) ×6.4 (Lean-formalized)
- Off-diagonal PSLQ verdict: L[4,2], L[5,3] formulas were PSLQ artifacts (dropped from claims)

## v3.3.24, 2026-03-02

**NuFIT 6.0 Update & Publication Cleanup**

- Updated to NuFIT 6.0 experimental values (δ_CP: 177°±20°)
- New neutrino formulas: θ₁₂ = arctan(2/3), θ₂₃ = arctan(√(14/11))
- Key insight: tan(θ₁₂) = Q_Koide = 2/3
- Mean deviation: **0.24%** (32 well-measured) / 0.57% (all 33 incl. δ_CP)
- S1 construction claims softened (conditional on Joyce existence theorem)
- Weyl → w rename to avoid collision with Weyl curvature

## v3.3.17, 2026-02-04

**θ₂₃ Formula Correction**

- Fixed atmospheric mixing angle: arcsin(25/33) = 49.25° (was 59.16°)
- θ₂₃ deviation: 20% → **0.10%**
- Mean deviation: 0.84% → **0.21%**

## v3.3.14, 2026-01-28

**Selection Principle & 290+ Relations**

- Selection Principle formalized in Lean 4
- TCS Spectral Bounds added
- 290+ total certified relations
- Lean 4 as sole verification system (Coq archived)

## v3.3.0, 2026-01-12

**33 Observables & PDG 2024**

- Expanded to 33 dimensionless predictions
- Updated to PDG 2024 experimental values
- 192,349-configuration Monte Carlo validation

## v3.1.0, 2025-12-17

**Analytical G₂ Metric**

- Explicit Chebyshev-Cholesky metric (169 parameters)
- Newton-Kantorovich certification (h = 6.65×10⁻⁸)
- 185 certified Lean relations

## v3.0.0, 2025-12-09

**Major Release**

- 165+ certified relations in Lean 4
- Exceptional group structure explorations

## v2.3.x, 2025-12

- Lean 4 verification introduced

## v2.2.0, 2025-11-27

- Zero-parameter paradigm established

## v2.1.0, 2025-11-22

- Torsional dynamics framework
- Scale bridge (Λ_GIFT = 21×e⁸×248/(7×π⁴))

## v2.0.0, 2025-10-24

- Framework reorganization

---

*For complete details, see the [full changelog](https://github.com/gift-framework/GIFT/blob/main/CHANGELOG.md).*
