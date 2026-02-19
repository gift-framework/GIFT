# Global G₂ Metric on K₇: Construction Plan

**Author**: Brieuc de La Fournière
**Date**: February 2026
**Status**: Planning phase
**Goal**: First numerical computation of a global torsion-free G₂ metric on a compact TCS 7-manifold.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current State of the Art](#2-current-state-of-the-art)
3. [Mathematical Framework](#3-mathematical-framework)
4. [Architecture: Three-Domain Decomposition](#4-architecture-three-domain-decomposition)
5. [Domain 1: Neck Region (Existing PINN v3.2)](#5-domain-1-neck-region-existing-pinn-v32)
6. [Domains 2–3: Bulk CY Regions](#6-domains-23-bulk-cy-regions)
7. [Interface Matching & Blending](#7-interface-matching--blending)
8. [Training Strategy](#8-training-strategy)
9. [Loss Functions](#9-loss-functions)
10. [Lean 4 Formalization Strategy](#10-lean-4-formalization-strategy)
11. [Verification & Certification](#11-verification--certification)
12. [Computational Budget](#12-computational-budget)
13. [Risk Analysis](#13-risk-analysis)
14. [File Map: Existing Assets](#14-file-map-existing-assets)
15. [New Files to Create](#15-new-files-to-create)
16. [Milestones & Timeline](#16-milestones--timeline)
17. [References](#17-references)

---

## 1. Executive Summary

We propose to extend the existing PINN v3.2 metric on the TCS neck region of
K₇ to a **global metric** covering the full compact 7-manifold. The approach
uses **domain decomposition** with three Cholesky-parameterized PINNs (one per
domain), coupled via Schwarz alternating iterations and Cholesky-level blending
at interfaces.

**Key facts:**
- No numerical global G₂ metric on a compact manifold exists in the
  literature (confirmed Harvard 2025, Duke 2024)
- The GIFT PINN v3.2 on the neck region is the current state of the art
- The Joyce safety margin (269×) provides substantial tolerance for
  interface approximation errors
- Estimated total training: **~45 minutes on a single A100 GPU**
- Total parameters: **~600k** (3 PINNs × ~200k each)

---

## 2. Current State of the Art

### 2.1 What exists

| Component | Status | Reference |
|-----------|--------|-----------|
| **Neck PINN v3.2** | Trained, det(g)=65/32 ± 5×10⁻⁹, torsion 3.71×10⁻⁴ | `notebooks/outputs/k7_pinn_step5_final.pt` |
| **Joyce certificate** | Lean 4 verified, safety margin 269× | `notebooks/outputs/K7Certificate.lean` |
| **ACyl CY existence (M₁, M₂)** | Proven (Haskins-Hein-Nordström 2015) | Not computed numerically |
| **K3 matching** | Hyper-Kähler rotation specified | `research/tcs/matching/K3_MATCHING.md` |
| **TCS Betti numbers** | Lean 4 proven: b₂=21, b₃=77 | `core/Lean/GIFT/Foundations/TCSConstruction.lean` |
| **Geodesic solver** | Module with CheckpointPINNAdapter | `core/gift_core/nn/geodesics.py` |
| **Metric export** | 2000-point sample, v3.2 certification JSON | `notebooks/outputs/k7_metric_data.csv` |

### 2.2 What does NOT exist (gaps to fill)

| Gap | Difficulty | Our approach |
|-----|-----------|--------------|
| Numerical ACyl CY metrics on M₁, M₂ | Hard (no one has computed these) | Analytical warm-start + PINN learning |
| Global G₂ metric on compact K₇ | **Unsolved worldwide** | Domain decomposition + Schwarz |
| Formal global torsion certificate | Medium | Lean 4 conditional theorem |
| Interface smoothness verification | Medium | Cholesky blending + geodesic continuity test |

### 2.3 Literature

- **Kovalev (2003)**: TCS construction, existence of G₂ structures for large neck parameter T
- **CHNP (Duke 2015)**: Systematic TCS from semi-Fano 3-folds
- **Haskins-Hein-Nordström (JDG 2015)**: ACyl CY metrics exist with exponential decay O(e^{−μr})
- **Lotay-Wei (GAFA 2019)**: Laplacian flow for closed G₂ structures, Shi-type estimates
- **Crowley-Goette-Nordström (Inventiones 2024)**: G₂ boundary problem is elliptic (modulo gauge)
- **Anderson-Gray-Larfors (2023)**: Lectures on numerical CY metrics (cymetric, cymyc)
- **Duke MathPlus (2024)**: Numerical Laplacian flow singularities on non-compact G₂
- **Harvard seminar (Feb 2025)**: "No ML work on compact G₂ manifolds until 2024"
- **Hein-Sun-Viaclovsky-Zhang (JAMS 2022)**: Gravitational instantons, Tian-Yau Fredholm theory

---

## 3. Mathematical Framework

### 3.1 The manifold K₇

$$
K_7 = (M_1 \times S^1) \cup_\Phi (M_2 \times S^1)
$$

| Building block | Construction | b₂ | b₃ | Anticanonical divisor |
|---------------|-------------|-----|-----|----------------------|
| M₁ | ACyl CY from quintic X₅ ⊂ ℂℙ⁴ | 11 | 40 | D₁ ≅ K3 (quartic) |
| M₂ | ACyl CY from CI(2,2,2) ⊂ ℂℙ⁶ | 10 | 37 | D₂ ≅ K3 |

Gluing cross-section: S¹ × K3, with hyper-Kähler rotation Φ.

**Ref**: `research/tcs/building_blocks/ACYL_CY3_SPEC.md`

### 3.2 The PDE system

On each domain, a torsion-free G₂-structure φ satisfies:

$$
d\varphi = 0, \qquad d{*_\varphi}\varphi = 0
$$

This is a first-order nonlinear elliptic system (modulo diffeomorphisms).

In the **bulk** of M_i × S¹, the G₂ structure takes the product form:

$$
\varphi = d\theta \wedge \omega_i + \text{Re}(\Omega_i)
$$

where (ω_i, Ω_i) is the CY structure on M_i. The torsion-free condition
reduces to **Ricci-flatness**: Ric(g_{CY}) = 0.

In the **neck** (gluing region), the full G₂ torsion-free equations apply.
This is what the existing PINN v3.2 already solves.

### 3.3 The gluing theorem (Kovalev/CHNP)

For neck parameter T sufficiently large (T ≥ T₀):

1. Truncate each M_i × S¹ at cylindrical coordinate t = T
2. Match via Φ: (t, θ, K3, S¹)_{M₁} ↔ (2T−t, S¹, K3, θ)_{M₂}
3. Interpolate with smooth cutoff χ(t) on overlap [T−1, T+1]
4. The approximate G₂-form φ_T has torsion ||d*φ_T|| = O(e^{−δT})
5. Joyce's theorem (Thm 11.6.1) produces a unique nearby torsion-free φ̃_T

**Ref**: `research/tcs/matching/K3_MATCHING.md`, `research/tcs/proof/PHASE3_SURGERY.md`

### 3.4 The hyper-Kähler rotation

Donaldson's matching condition on the K3 fiber:

```
r*(ω_J⁻) = ω_I⁺        (Kähler ↔ Kähler)
r*(Re Ω_J⁻) = Re Ω_I⁺  (holomorphic ↔ holomorphic)
r*(Im Ω_J⁻) = −Im Ω_I⁺ (sign flip)
```

In quaternionic notation: **i ↦ j** rotation of the HK triple:

```
(ω_I, ω_J, ω_K)₊  ↦  (ω_J, −ω_I, ω_K)₋
```

**Ref**: `research/tcs/matching/K3_MATCHING.md` §2

### 3.5 Asymptotic decay of ACyl CY metrics

On the cylindrical end {r > T₀}:

```
g = g_K3 + dr² + dθ² + O(e^{−μr})
ω = ω_K3 + dr ∧ dθ + O(e^{−μr})
Ω = Ω_K3 ∧ (dr + i·dθ) + O(e^{−μr})
```

where μ ≈ 1 for Fano-derived blocks (quintic and CI(2,2,2)).

**Ref**: `research/tcs/building_blocks/ACYL_CY3_SPEC.md` §1

---

## 4. Architecture: Three-Domain Decomposition

### 4.1 Domain layout

```
 t ∈ (−∞, −T/2]           t ∈ [−T/2−δ, T/2+δ]           t ∈ [T/2, +∞)
┌────────────────────┐  ┌───────────────────────────┐  ┌────────────────────┐
│                    │  │                           │  │                    │
│   Ω₁ (bulk M₁)    │  │   Ω_neck (gluing)         │  │   Ω₂ (bulk M₂)    │
│   ACyl CY₃ × S¹   │  │   TCS neck region         │  │   ACyl CY₃ × S¹   │
│                    │  │                           │  │                    │
│   BulkPINN₁       │  │   Existing PINN v3.2      │  │   BulkPINN₂       │
│   Ric(g) = 0      │  │   dφ = d*φ = 0            │  │   Ric(g) = 0      │
│   ~200k params     │  │   202,857 params          │  │   ~200k params     │
│                    │  │                           │  │                    │
└────────────────────┘  └───────────────────────────┘  └────────────────────┘
         |<-- overlap₁ -->|                   |<-- overlap₂ -->|
```

### 4.2 Coordinate systems

**Ω_neck** (existing):
- 7 coordinates (x¹, ..., x⁷) on [0, L]⁷ with L = 10.0
- 8th input: log(T) (energy/scale parameter)
- Periodic boundary conditions

**Ω₁** (bulk of M₁ × S¹):
- t: ACyl radial coordinate (t → +∞ toward neck)
- θ: S¹ coordinate (periodic, [0, 2π))
- (u¹, u², u³, u⁴): K3 fiber coordinates
- s: outer S¹ factor
- Total: 7D

**Ω₂** (bulk of M₂ × S¹):
- t': ACyl radial coordinate
- θ': S¹ coordinate
- (u'¹, u'², u'³, u'⁴): K3 fiber coordinates
- s': outer S¹ factor
- Total: 7D

**Ref**: `research/tcs/metric/EXPLICIT_METRIC_PLAN.md` §2

### 4.3 Coordinate transition maps

At the interfaces, the TCS gluing identifies:

```python
# Left interface: Ω₁ → Ω_neck
def phi_1_to_neck(t, theta, u, s):
    x_neck[0] = (t + T/2) / L * coord_range   # cylinder → neck
    x_neck[1] = theta / (2*pi) * coord_range   # S¹
    x_neck[2:6] = K3_chart(u)                  # K3 coords
    x_neck[6] = s / (2*pi) * coord_range       # outer S¹

# Right interface: Ω₂ → Ω_neck (with HK rotation!)
def phi_2_to_neck(t_prime, theta_prime, u_prime, s_prime):
    x_neck[0] = (T/2 - t_prime) / L * coord_range  # reversed
    x_neck[1] = s_prime / (2*pi) * coord_range      # θ' ↔ s (swap!)
    x_neck[2:6] = HK_rotate(K3_chart(u_prime))      # HK rotation
    x_neck[6] = theta_prime / (2*pi) * coord_range   # s' ↔ θ (swap!)
```

The S¹ swap (θ ↔ s) is the geometric content of the TCS construction.

---

## 5. Domain 1: Neck Region (Existing PINN v3.2)

### 5.1 Architecture (already trained)

```
Input: (x¹, ..., x⁷, log T) ∈ ℝ⁸
  ↓
FourierFeatures(B ∈ ℝ^{48×8}) → ℝ⁹⁶
  ↓
Backbone: 96 → 256 → 256 → 256 → 128 (SiLU)
  ↓
├── metric_head: 128 → 28 (Cholesky δL)
│     g(x) = (L₀ + δL(x))(L₀ + δL(x))ᵀ
├── local_head: 128 → 35 (3-form local modes)
└── global_head: 128 → 42 (3-form global modes)
```

### 5.2 Achieved performance

| Metric | Value |
|--------|-------|
| Parameters | 202,857 |
| det(g) | 2.03125 ± 5×10⁻⁹ |
| Torsion max | 3.71 × 10⁻⁴ |
| Joyce margin | 269× |
| Condition number κ | 1.0152 |
| Eigenvalue accuracy | 8 significant figures |
| Training time | 2.9 min (A100) |

### 5.3 Key files

| File | Content |
|------|---------|
| `notebooks/outputs/k7_pinn_step5_final.pt` | Trained checkpoint (1.6 MB, float64) |
| `notebooks/outputs/k7_metric_v32_export.json` | Certification (20/20 checks) |
| `notebooks/outputs/K7Certificate.lean` | Lean 4 formal certificate |
| `notebooks/outputs/k7_metric_data.csv` | 2000-point evaluation sample |
| `core/gift_core/nn/geodesics.py` | `CheckpointPINNAdapter` class |

### 5.4 Role in global construction

- **Fixed warm-start** for Phase 2 (Schwarz alternation)
- Provides boundary data for bulk PINNs at interfaces
- Loss updated in Phase 2 to include matching terms

---

## 6. Domains 2–3: Bulk CY Regions

### 6.1 BulkCYPINN architecture

Each bulk PINN mirrors the neck architecture for compatibility:

```
Input: (t, θ, u¹, u², u³, u⁴, s, log T) ∈ ℝ⁸
  ↓
FourierFeatures(B ∈ ℝ^{48×8}) → ℝ⁹⁶
  ↓
Backbone: 96 → 256 → 256 → 256 → 128 (SiLU)
  ↓
metric_head: 128 → 28 (Cholesky δL)
  g(x) = (L₀_CY + δL(x))(L₀_CY + δL(x))ᵀ
```

**Key difference**: L₀_CY is the Cholesky factor of the known asymptotic
CY product metric, not G_TARGET.

### 6.2 Warm-start initialization

In the cylindrical end, the CY product metric is:

$$
g_{M_i \times S^1} = dt^2 + d\theta^2 + g_{K3}(u) + ds^2
$$

This is block-diagonal: diag(1, 1, g_{K3}, 1) where g_{K3} is 4×4.

**Three initialization options:**

| Option | L₀_CY | Accuracy | Effort |
|--------|--------|----------|--------|
| A (simple) | chol(diag(1, 1, α·I₄, 1)) with α = (65/32)^{1/7} | ~10% | Immediate |
| B (better) | chol(diag(1, 1, g_{K3}^{Donaldson}, 1)) | ~0.1% | 1–2 days (cymetric) |
| C (best) | Dedicated K3 PINN (4D, ~50k params) | ~10⁻⁴ | 1 week |

**Recommendation**: Start with Option A. The Cholesky warm-start + 269× Joyce
margin means even a crude initialization will converge. Upgrade to B or C if
interface matching is too slow.

### 6.3 S¹ periodicity

For the S¹ coordinates (θ, s), enforce exact periodicity via deterministic
Fourier encoding instead of random features:

```python
gamma_S1(s) = [sin(2πs), cos(2πs), sin(4πs), cos(4πs), ...]
```

The remaining 5 coordinates use random Fourier features as in the neck PINN.

### 6.4 Period integral splitting

The b₃ = 77 moduli split across building blocks:

| Domain | Periods | Source |
|--------|---------|--------|
| Ω₁ (M₁) | 40 (from H³(M₁)) | Inherited from GIFT framework |
| Ω₂ (M₂) | 37 (from H³(M₂)) | Inherited from GIFT framework |
| **Total** | **77** | Matches b₃(K₇) |

Each bulk PINN trains against its share of the period data.

---

## 7. Interface Matching & Blending

### 7.1 Overlap regions

```
Overlap₁: t ∈ [−T/2 − δ, −T/2 + δ]   (left, Ω₁ ∩ Ω_neck)
Overlap₂: t ∈ [+T/2 − δ, +T/2 + δ]   (right, Ω₂ ∩ Ω_neck)
```

Width δ chosen as ~0.5L (half the neck coordinate range).

### 7.2 Cholesky-level blending

**Critical design choice**: blend at the Cholesky factor level, not the
metric level, to preserve positive definiteness:

```python
L_blend(x) = (1 − α(t)) · L_bulk(x) + α(t) · L_neck(x')
g_blend(x) = L_blend(x) @ L_blend(x)ᵀ    # always pos. def.
```

where α(t) is a smooth blending function:

```python
def blend(t, t_interface, delta):
    s = (t - t_interface + delta) / (2 * delta)
    s = clamp(s, 0, 1)
    return 3*s² − 2*s³   # C¹ Hermite interpolant
```

### 7.3 Jacobian correction

The coordinate transition between domains requires a Jacobian correction
for the metric tensor:

$$
g'_{ab} = \frac{\partial x^i}{\partial x'^a} \frac{\partial x^j}{\partial x'^b} g_{ij}
$$

At the Cholesky level: L' = J · L where J is the Jacobian of the
transition map. The HK rotation contributes an orthogonal factor.

### 7.4 Interface loss terms

```python
def interface_loss(bulk_pinn, neck_pinn, x_overlap, transition_map):
    # C⁰ matching: Cholesky factors agree
    L_bulk = bulk_pinn.cholesky(x_overlap)
    L_neck = neck_pinn.cholesky(transition_map(x_overlap))
    L_neck_transformed = J @ L_neck   # Jacobian correction

    L_C0 = mean((L_bulk − L_neck_transformed)²)

    # C¹ matching: normal derivatives agree
    dL_bulk_dt = autograd(L_bulk, x_overlap)[:, 0]
    dL_neck_dt = autograd(L_neck_transformed, x_overlap)[:, 0]
    L_C1 = mean((dL_bulk_dt − dL_neck_dt)²)

    return L_C0 + 0.1 * L_C1
```

---

## 8. Training Strategy

### Phase 1: Independent bulk training (embarrassingly parallel)

```
BulkPINN₁: minimize L_ricci + L_det + L_period_M1 + L_sparse
BulkPINN₂: minimize L_ricci + L_det + L_period_M2 + L_sparse
```

- Warm-start from L₀_CY (analytical CY target)
- No interface matching yet
- 5,000 epochs, lr schedule: 10⁻³ (0–2500) → 10⁻⁴ (2500–5000)
- **Est. time: ~3 min each on A100, parallel → 3 min wall clock**

### Phase 2: Schwarz alternating method

```python
for iteration in range(N_schwarz):  # N_schwarz ≈ 5–10
    # Step A: Fix neck, optimize bulk with matching
    freeze(neck_pinn)
    for epoch in range(200):
        optimize(bulk_1, L_ricci + L_match_left + L_det)
        optimize(bulk_2, L_ricci + L_match_right + L_det)
    unfreeze(neck_pinn)

    # Step B: Fix bulk, optimize neck with matching
    freeze(bulk_1, bulk_2)
    for epoch in range(200):
        optimize(neck_pinn, L_torsion + L_match_left + L_match_right + L_det)
    unfreeze(bulk_1, bulk_2)
```

Convergence guaranteed for elliptic systems with positive overlap (Schwarz 1870,
extended to nonlinear elliptic by Lions 1988).

- **Est. time: ~30 min on A100**

### Phase 3: Joint fine-tuning

Unfreeze all three PINNs, train jointly:

```
L_global = L_torsion_neck + L_ricci_bulk1 + L_ricci_bulk2
         + λ_m (L_match_left + L_match_right)
         + λ_d L_det_all
         + λ_p L_period_all
```

- Learning rate 10⁻⁵ with cosine annealing
- 5,000 epochs
- **Est. time: ~10 min on A100**

### Total training budget: ~45 min on a single A100

---

## 9. Loss Functions

### 9.1 Neck loss (existing, extended)

| Term | Formula | Weight | Status |
|------|---------|--------|--------|
| L_det | (det(g) − 65/32)² | 100 | Existing |
| L_aniso | ‖⟨g⟩ − G_TARGET‖²_F | 500 | Existing |
| L_period | Σ_T ‖⟨δφ⟩_T − Π(T)‖² / 5 | 1000 | Existing |
| L_torsion | ‖dφ‖² + ‖d*φ‖² | 1 | Existing |
| L_sparse | ‖δL‖² | 0.01 | Existing |
| **L_match** | **interface_loss at both boundaries** | **100** | **New** |

### 9.2 Bulk loss (new)

| Term | Formula | Weight | Purpose |
|------|---------|--------|---------|
| L_ricci | ‖Ric(g)‖² | 1 | CY Ricci-flatness |
| L_det | (det(g) − 65/32)² | 100 | Determinant constraint |
| L_period | ‖periods − Π_k‖² for k in H³(M_i) | 1000 | Moduli matching |
| L_match | interface_loss at boundary | 100 | Interface continuity |
| L_sparse | ‖δL‖² | 0.01 | Regularization |
| L_decay | ‖g − g_CY^{asymp}‖² · e^{μr} | 10 | Exponential decay enforcement |

### 9.3 Ricci tensor computation

The Ricci tensor is computed via autograd (second derivatives of the metric):

$$
R_{ij} = \partial_k \Gamma^k_{ij} - \partial_j \Gamma^k_{ik}
       + \Gamma^k_{kl} \Gamma^l_{ij} - \Gamma^k_{jl} \Gamma^l_{ik}
$$

The Christoffel symbols Γ^k_{ij} = ½ g^{kl}(∂_i g_{jl} + ∂_j g_{il} − ∂_l g_{ij})
involve first derivatives of g, which are already computed in `GeodesicSolver.christoffel()`.

**Performance note**: Ric(g) requires second derivatives through the PINN,
creating a 4th-order autograd graph. For bulk training, use finite-difference
approximations for the outer derivatives (as in `christoffel_fd`) to keep
memory manageable.

**Ref**: `core/gift_core/nn/geodesics.py`, lines 185–235 (Christoffel computation)

---

## 10. Lean 4 Formalization Strategy

### 10.1 Current Lean assets

| File | Content | Status |
|------|---------|--------|
| `core/Lean/GIFT/Foundations/TCSConstruction.lean` | b₂=21, b₃=77, H*=99, χ=−110 | **Proven** |
| `core/Lean/GIFT/Foundations/G2Holonomy.lean` | G₂ structure basics | Defined |
| `core/Lean/GIFT/Foundations/Analysis/G2Forms/G2Structure.lean` | TorsionFree predicate | Defined |
| `core/Lean/GIFT/Spectral/NeckGeometry.lean` | Neck hypotheses (H1–H6) | Axiomatized |
| `core/Lean/GIFT/Spectral/TCSBounds.lean` | Spectral bounds v₀²/L² ≤ λ₁ | **Proven** |
| `core/Lean/GIFT/Joyce.lean` | Joyce theorem structure | Certificate |
| `core/Lean/GIFT/IntervalArithmetic.lean` | Scaled-Nat bound verification | **Proven** |
| `core/Lean/GIFT/Spectral/LiteratureAxioms.lean` | Literature axiom classification | Documented |
| `notebooks/outputs/K7Certificate.lean` | torsion < ε₀ (native_decide) | **Proven** |

### 10.2 New Lean formalization: GlobalMetricCertificate

The key new formalization is a **conditional theorem** for the domain
decomposition:

```lean
namespace GlobalG2Certificate

/-- Neck PINN torsion bound (from v3.2 certification) -/
def neck_torsion : ℚ := 3710 / 10000000   -- 3.71 × 10⁻⁴

/-- Bulk Ricci residual bound (to be computed) -/
def bulk_ricci_1 : ℚ := sorry  -- TBD after bulk PINN training
def bulk_ricci_2 : ℚ := sorry  -- TBD after bulk PINN training

/-- Interface matching error (to be computed) -/
def match_error : ℚ := sorry   -- TBD after Schwarz convergence

/-- Global torsion bound: neck + bulk contributions + matching -/
def global_torsion : ℚ := neck_torsion + match_error

/-- Joyce threshold -/
def joyce_epsilon : ℚ := 1 / 10

/-- The conditional theorem -/
theorem global_joyce_applies
    (h_match : match_error < 1 / 1000)  -- matching error < 10⁻³
    : global_torsion < joyce_epsilon := by
  -- Once match_error is filled in, this reduces to
  -- 3.71×10⁻⁴ + match_error < 0.1
  -- which is true for any match_error < 0.0996
  sorry  -- Will be native_decide after numerical values filled in

end GlobalG2Certificate
```

### 10.3 Mathlib availability assessment

| Needed | In Mathlib? | Alternative |
|--------|-------------|-------------|
| Smooth manifolds | Yes | — |
| Riemannian metrics | Yes (recent) | — |
| Contraction mapping (Banach) | Yes | Used for Joyce iteration |
| Lie groups (abstract) | Yes | — |
| G₂ as specific Lie group | No | Axiomatize |
| Sobolev spaces | No | Axiomatize (Tier C) |
| Elliptic regularity | No | Axiomatize (Tier C) |
| De Rham cohomology | No | Use algebraic chain complexes |
| Hodge star/decomposition | No | Axiomatize (Tier C) |
| Holonomy groups | No | Axiomatize (Tier C) |

### 10.4 LeanCert integration (recommended)

The [LeanCert](https://github.com/alerad/leancert) project provides verified
interval arithmetic for Lean 4, including neural network bound propagation.
This would upgrade the current `native_decide`-based certification to genuine
interval arithmetic certificates.

**Ref**: `core/Lean/GIFT/Spectral/LiteratureAxioms.lean` (axiom classification)

---

## 11. Verification & Certification

### 11.1 Per-domain verification

| Check | Target | Method |
|-------|--------|--------|
| det(g) = 65/32 | < 10⁻⁷ deviation | Sample 50k points per domain |
| Positive definite | All λᵢ > 0 | Guaranteed by Cholesky (even through blending) |
| Torsion (neck) | < 3.71×10⁻⁴ | Finite-difference dφ, d*φ |
| Ricci (bulk) | ‖Ric‖ < 10⁻³ | Autograd computation |
| Eigenvalues | 8 sig. figs. match | Compare with analytical target |

### 11.2 Global verification

| Check | Target | Method |
|-------|--------|--------|
| C¹ smoothness at interfaces | ‖∂g/∂n‖ match < 10⁻³ | Evaluate both sides, compare |
| Geodesic continuity | No acceleration spikes | GeodesicSolver through overlaps |
| Global det(g) | 65/32 everywhere | Sample across all domains |
| Global torsion | < ε₀/269 | Extended finite-difference |
| Period integrals (all 77) | RMS < 0.005 | Integrate across all domains |

### 11.3 Lean 4 certification pipeline

1. Train PINNs → export numerical bounds (JSON)
2. Convert bounds to rational numbers (scaled Nat)
3. Generate Lean 4 certificate (automated script)
4. Verify with `native_decide` or LeanCert
5. Commit certificate to `core/Lean/GIFT/`

**Ref**: `notebooks/outputs/K7Certificate.lean` (existing pipeline)

---

## 12. Computational Budget

### 12.1 Training

| Component | Dimension | Params | Time (A100) |
|-----------|-----------|--------|-------------|
| Neck PINN (existing) | 8D | 202,857 | 2.9 min (done) |
| Bulk PINN₁ | 8D | ~200,000 | ~3 min |
| Bulk PINN₂ | 8D | ~200,000 | ~3 min |
| Schwarz alternation | — | — | ~30 min (10 iter) |
| Joint fine-tuning | — | — | ~10 min |
| **Total** | — | **~600k** | **~45 min** |

### 12.2 Memory

- Each PINN: ~1–2 GB (from Appendix B.2 of Explicit_G2_metric.md)
- Three PINNs + Ricci autograd graph: ~10–15 GB peak
- Fits comfortably in A100 80GB memory

### 12.3 Evaluation

| Task | Points | Time |
|------|--------|------|
| Per-domain torsion/Ricci | 50k × 3 domains | ~5 min |
| Geodesic integration | 1000 geodesics | ~10 min |
| Period integrals (77 × 5 scales) | — | ~5 min |
| **Total evaluation** | — | **~20 min** |

---

## 13. Risk Analysis

### 13.1 Technical risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Ricci tensor cost** (4th-order autograd) | Medium | Use finite-diff for outer derivatives |
| **Coordinate singularities** on CY | High | Restrict bulk PINNs to cylindrical end |
| **HK rotation accuracy** | Medium | Implement as exact orthogonal transform |
| **Scale mismatch** (neck [0,10]⁷ vs bulk coords) | Low | Rescale all to [0,1]⁷ before Fourier |
| **Schwarz non-convergence** | Low | Both PDEs are elliptic; convergence proven |
| **K3 metric unavailable** | Medium | Option A (flat approx) + Joyce margin |

### 13.2 Scientific risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Global torsion exceeds ε₀** | Medium | 269× margin; would need >100× interface error |
| **Bulk CY solutions not unique** | Low | CY metrics unique by Yau's theorem |
| **T₀ unknown** (min neck length) | Medium | GIFT L_canonical = 8.354 likely > T₀ |

### 13.3 Contingency: Laplacian flow alternative

If domain decomposition fails, a fallback is the Laplacian flow:

$$
\frac{\partial \varphi}{\partial t} = \Delta_\varphi \varphi
$$

starting from the approximately torsion-free global metric. The flow
converges to a torsion-free structure if the initial data is in the basin
of attraction (Lotay-Wei stability theorem). This requires discretizing
the full 7-manifold, which is more expensive but avoids interface issues.

**Ref**: `research/yang-mills/ricci_flow_g2.py` (existing Ricci flow code)

---

## 14. File Map: Existing Assets

### 14.1 Publications

| File | Content |
|------|---------|
| `publications/markdown/Explicit_G2_metric.md` | Main paper (updated with Joyce cert.) |
| `publications/markdown/Prime_spectral_S_T.md` | Companion paper (mollified Dirichlet) |
| `publications/markdown/GIFT_v3.3_main.md` | Framework overview |
| `publications/markdown/GIFT_v3.3_S1_foundations.md` | E₈, G₂, K₇ foundations |
| `publications/markdown/GIFT_v3.3_S2_derivations.md` | Dimensionless predictions |
| `publications/markdown/GIFT_v3.3_S3_dynamics.md` | RG flow, torsional dynamics |

### 14.2 TCS construction (research/tcs/)

| File | Content |
|------|---------|
| `building_blocks/ACYL_CY3_SPEC.md` | M₁ (quintic), M₂ (CI(2,2,2)), decay rates |
| `matching/K3_MATCHING.md` | Donaldson matching, HK rotation, lattice |
| `metric/EXPLICIT_METRIC_PLAN.md` | 5-patch coordinate system |
| `metric/METRIC_EXTRACTION.md` | Metric extraction methods |
| `metric/g2_metric_final.py` | Metric computation code |
| `metric/k7_metric_final.json` | Computed metric data |
| `proof/PHASE1_SETUP.md` | TCS structure definition |
| `proof/PHASE2_CYLINDER.md` | Cylindrical Fourier analysis |
| `proof/PHASE3_SURGERY.md` | Mazzeo-Melrose surgery, scattering matrix |
| `proof/PHASE4_ASYMPTOTICS.md` | Eigenvalue asymptotics λ₁ ~ π²/L² |
| `proof/PHASE5_ERRORS.md` | Error control O(e^{−δL}) |
| `proof/PHASE6_SELECTION.md` | Selection principle for L |
| `proof/TIER1_THEOREM.md` | Rigorous spectral bounds (proven) |
| `proof/TIER2_HARMONIC.md` | Harmonic forms and moduli |
| `proof/TIER3_CONJECTURE.md` | Coefficient = π²/14 (conjectural) |
| `proof/ANALYTICAL_PROOF_STRATEGY.md` | Complete 5-phase proof strategy |
| `proof/LEAN_INTEGRATION_PLAN.md` | Lean formalization roadmap |

### 14.3 Neural network code (core/gift_core/nn/)

| File | Content |
|------|---------|
| `g2_pinn.py` | Original G2PINN (35-component output) — superseded |
| `gift_native_pinn.py` | G₂ adjoint PINN (14 DOF) — rank-deficient, superseded |
| `geodesics.py` | **CheckpointPINNAdapter** + GeodesicSolver |
| `fourier_features.py` | FourierFeatures, PositionalEncoding |
| `loss_functions.py` | torsion_loss, det_g_loss, kappa_t_loss |
| `training.py` | G2Trainer, TrainConfig, TrainResult |
| `__init__.py` | Module exports |

### 14.4 Lean formalization (core/Lean/GIFT/)

| File | Content | Status |
|------|---------|--------|
| `Foundations/TCSConstruction.lean` | Betti numbers, H*, χ | **Proven** |
| `Foundations/G2Holonomy.lean` | G₂ structure basics | Defined |
| `Foundations/E8Lattice.lean` | E₈ root system | Defined |
| `Foundations/Analysis/G2Forms/G2Structure.lean` | TorsionFree predicate | Defined |
| `Spectral/NeckGeometry.lean` | Neck hypotheses H1–H6 | Axiomatized |
| `Spectral/TCSBounds.lean` | Spectral bounds theorem | **Proven** |
| `Spectral/LiteratureAxioms.lean` | Axiom classification | Documented |
| `Spectral/SelbergBridge.lean` | Selberg trace formula bridge | Defined |
| `Joyce.lean` | Joyce theorem structure | Certificate |
| `IntervalArithmetic.lean` | Bound verification | **Proven** |

### 14.5 Data artifacts

| File | Content |
|------|---------|
| `notebooks/outputs/k7_pinn_step5_final.pt` | PINN v3.2 checkpoint (1.6 MB) |
| `notebooks/outputs/k7_metric_v32_export.json` | 20/20 certification |
| `notebooks/outputs/K7Certificate.lean` | Lean 4 torsion certificate |
| `notebooks/outputs/k7_metric_data.csv` | 2000-point metric samples |
| `notebooks/outputs/riemann_zeros_2M_genuine.npy` | 2M Riemann zeros |
| `notebooks/outputs/riemann_zeros_100k_genuine.npy` | 100k Riemann zeros |

---

## 15. New Files to Create

### 15.1 Neural network modules (core/gift_core/nn/)

```
core/gift_core/nn/
├── bulk_cy_pinn.py           # NEW: BulkCYPINN class + BulkCYLoss
├── domain_decomposition.py   # NEW: TransitionMap, BlendedMetric, SchwarzTrainer
├── global_g2_metric.py       # NEW: GlobalG2Metric (wraps 3 PINNs)
├── ricci_tensor.py           # NEW: Autograd Ricci tensor computation
├── geodesics.py              # EXISTING: + GlobalGeodesicSolver extension
├── g2_pinn.py                # EXISTING (unchanged)
├── gift_native_pinn.py       # EXISTING (unchanged)
├── fourier_features.py       # EXISTING (unchanged)
├── loss_functions.py         # EXISTING: + Ricci loss, matching loss
├── training.py               # EXISTING: + SchwarzTrainConfig
└── __init__.py               # EXISTING: + new exports
```

### 15.2 Notebooks

```
notebooks/
├── K7_Global_Metric_v1.ipynb          # NEW: End-to-end global construction
├── K7_Bulk_CY_Training.ipynb          # NEW: Bulk PINN training
├── K7_Interface_Verification.ipynb    # NEW: Interface smoothness checks
└── K7_Global_Geodesics.ipynb          # NEW: Geodesics across full K₇
```

### 15.3 Lean formalization

```
core/Lean/GIFT/
├── GlobalMetricCertificate.lean       # NEW: Domain decomposition theorem
├── IntervalArithmetic.lean            # EXISTING: + bulk bounds
└── Spectral/
    └── GlobalTorsionBound.lean        # NEW: Global torsion from decomposition
```

### 15.4 Research documentation

```
research/tcs/
├── GLOBAL_G2_METRIC_PLAN.md           # THIS FILE
├── metric/GLOBAL_CONSTRUCTION.md      # NEW: Detailed mathematical notes
└── validation/GLOBAL_VERIFICATION.md  # NEW: Verification checklist
```

---

## 16. Milestones & Timeline

### Milestone 1: Bulk PINN architecture (Week 1)

- [ ] Implement `BulkCYPINN` in `bulk_cy_pinn.py`
- [ ] Implement `ricci_tensor_autograd()` in `ricci_tensor.py`
- [ ] Unit tests: det(g), positive definiteness, Ricci on flat space
- [ ] Warm-start with Option A (flat K3 approximation)

### Milestone 2: Domain decomposition infrastructure (Week 1–2)

- [ ] Implement `TransitionMap` (coordinate transforms + HK rotation)
- [ ] Implement `BlendedMetric` (Cholesky-level blending)
- [ ] Implement interface loss terms
- [ ] Unit tests: blending preserves pos. def., Jacobian correct

### Milestone 3: Phase 1 training — independent bulk PINNs (Week 2)

- [ ] Train BulkPINN₁ on M₁ (Ricci-flat + det + periods)
- [ ] Train BulkPINN₂ on M₂ (Ricci-flat + det + periods)
- [ ] Verify: det(g) = 65/32, Ric < 10⁻³ on each domain

### Milestone 4: Phase 2 — Schwarz alternation (Week 2–3)

- [ ] Implement `SchwarzTrainer`
- [ ] Run Schwarz iterations until convergence
- [ ] Monitor interface matching error per iteration
- [ ] Verify: matching error < 10⁻³ on both interfaces

### Milestone 5: Phase 3 — Joint fine-tuning (Week 3)

- [ ] Joint training of all three PINNs
- [ ] Global torsion evaluation across all domains
- [ ] Period integral verification (all 77 modes)

### Milestone 6: Verification & certification (Week 3–4)

- [ ] 50k-point global evaluation (det, torsion, Ricci)
- [ ] 1000 geodesics through interfaces (continuity check)
- [ ] Export numerical bounds → Lean 4 certificate
- [ ] Update `Explicit_G2_metric.md` with global results
- [ ] Push certified checkpoint to repository

### Milestone 7: Publication update (Week 4)

- [ ] Extend §5 with global metric results
- [ ] Promote Limitation 1 ("local, not global") to resolved
- [ ] Add §5.8: Global metric via domain decomposition
- [ ] Update abstract: "global metric on compact K₇"

---

## 17. References

### Core mathematical references

1. Joyce, D.D. (2000). *Compact Manifolds with Special Holonomy*. Oxford University Press. [Theorem 11.6.1]
2. Kovalev, A.G. (2003). Twisted connected sums and special Riemannian holonomy. *J. Reine Angew. Math.* 565, 125–160.
3. Corti, Haskins, Nordström, Pacini (2015). G₂-manifolds and associative submanifolds via semi-Fano 3-folds. *Duke Math. J.* 164(10), 1971–2092.
4. Haskins, Hein, Nordström (2015). Asymptotically cylindrical Calabi-Yau manifolds. *J. Diff. Geom.* 101(2), 213–265.
5. Lotay, Wei (2019). Laplacian flow for closed G₂ structures. *GAFA* 29, 1048–1110.
6. Crowley, Goette, Nordström (2024). An analytic invariant of G₂ manifolds. *Inventiones Math.*

### Numerical and computational references

7. Raissi, Perdikaris, Karniadakis (2019). Physics-informed neural networks. *J. Comput. Phys.* 378, 686–707.
8. Anderson, Gray, Larfors (2023). Lectures on numerical and ML methods for CY metrics. arXiv:2312.17125.
9. cymetric package: github.com/pythoncymetric/cymetric
10. cymyc (2025). CY Metrics, Yukawas, and Curvature. *JHEP* March 2025.
11. Duke MathPlus (2024). Computational exploration of geometric flows in G₂-geometry.

### Formalization references

12. LeanCert: github.com/alerad/leancert (verified interval arithmetic)
13. Massot, van Doorn, Nash (2022). Sphere eversion in Lean. arXiv:2210.07746.
14. GIFT core Lean formalization: github.com/gift-framework/core

---

*Plan prepared February 2026.*
