# Lean Integration Plan for Tier 1 Spectral Bounds

## Current State in gift-core

### Existing Modules (Already Formalized)

| File | Content | Status |
|------|---------|--------|
| `NeckGeometry.lean` | TCSManifold, Hypotheses (H1)-(H6) | ✅ Complete |
| `TCSBounds.lean` | Model theorem: c₁/L² ≤ λ₁ ≤ c₂/L² | ✅ Complete |
| `CheegerInequality.lean` | λ₁ ≥ h²/4 | ✅ Axiom |
| `SpectralTheory.lean` | Abstract Laplacian framework | ✅ Axioms |

### What's Missing for Tier 1

1. **Cross-section spectral gap** γ = λ₁(Y) as explicit hypothesis
2. **π² coefficient** identification (c₁ → π², c₂ → π² as L → ∞)
3. **Exponential error terms** O(e^{-δL})
4. **Localization lemma** (eigenfunction concentration on neck)

---

## Integration Plan

### Phase 1: Add Cross-Section Gap (H7)

**File:** Extend `NeckGeometry.lean`

```lean
/-- (H7) Cross-section spectral gap: λ₁(Y) = γ > 0.
This ensures transverse modes decay exponentially. -/
structure CrossSectionGap (K : TCSManifold) where
  γ : ℝ
  γ_pos : γ > 0
  is_first_eigenvalue : True  -- Placeholder for spectral theory

/-- Extended hypotheses including (H7) -/
structure TCSHypothesesExt (K : TCSManifold) extends TCSHypotheses K where
  crossGap : CrossSectionGap K
```

### Phase 2: Refine Bound Constants

**File:** Create `Tier1Bounds.lean`

```lean
/-- Refined lower bound constant approaching π². -/
noncomputable def c₁_refined (K : TCSManifold) (hyp : TCSHypothesesExt K) : ℝ :=
  Real.pi ^ 2 / (1 + C / (hyp.crossGap.γ * K.neckLength))

/-- Refined upper bound constant approaching π². -/
noncomputable def c₂_refined (K : TCSManifold) (hyp : TCSHypothesesExt K) : ℝ :=
  Real.pi ^ 2 * (1 + C / K.neckLength)

/-- As L → ∞, both bounds converge to π². -/
theorem bounds_converge_to_pi_squared : ∀ ε > 0, ∃ L₀, ∀ L > L₀,
  |c₁_refined - π²| < ε ∧ |c₂_refined - π²| < ε := sorry
```

### Phase 3: Localization Lemma

**File:** Add to `Tier1Bounds.lean`

```lean
/-- Localization: eigenfunctions with λ < γ/2 concentrate on neck. -/
theorem eigenfunction_localization (K : TCSManifold) (hyp : TCSHypothesesExt K)
    (f : EigenFunction K) (hλ : f.eigenvalue < hyp.crossGap.γ / 2) :
    ∫_{M \ N} |f|² ≤ C * exp(-δ * K.neckLength) * ∫_M |f|² := sorry
```

### Phase 4: Main Theorem with Error

**File:** `Tier1Bounds.lean`

```lean
/-- TIER 1 THEOREM: Spectral bounds with exponential error.

For TCS manifold K with hypotheses (H1)-(H7):
  π²/L² - O(e^{-δL}) ≤ λ₁(K) ≤ π²/L² + O(e^{-δL})

This is the rigorous formalization of the analytical proof. -/
theorem tier1_spectral_bounds (K : TCSManifold) (hyp : TCSHypothesesExt K)
    (hL : K.neckLength > L₀ K hyp.toTCSHypotheses) :
    ∃ (C δ : ℝ), C > 0 ∧ δ > 0 ∧
      (Real.pi ^ 2 / K.neckLength ^ 2 - C * Real.exp (-δ * K.neckLength)
         ≤ MassGap K.toCompactManifold) ∧
      (MassGap K.toCompactManifold
         ≤ Real.pi ^ 2 / K.neckLength ^ 2 + C * Real.exp (-δ * K.neckLength)) :=
  sorry  -- Proof via localization + variational bounds
```

---

## File Structure

```
gift-core/Lean/GIFT/Spectral/
├── SpectralTheory.lean       (existing - abstract framework)
├── NeckGeometry.lean         (existing - extend with H7)
├── TCSBounds.lean            (existing - model theorem)
├── CheegerInequality.lean    (existing)
├── Tier1Bounds.lean          ⬅️ NEW: Refined bounds with π²
├── Tier2Harmonic.lean        ⬅️ FUTURE: L ↔ H* connection
├── Tier3Conjecture.lean      ⬅️ FUTURE: κ = π²/14 conjecture
└── Spectral.lean             (existing - re-exports all)
```

---

## Dependencies

```
SpectralTheory.lean
       ↓
NeckGeometry.lean (H1-H7)
       ↓
TCSBounds.lean (model: c₁/L² ≤ λ₁ ≤ c₂/L²)
       ↓
Tier1Bounds.lean (refined: π²/L² ± O(e^{-δL}))
       ↓
UniversalLaw.lean (λ₁ × H* = 14)
```

---

## Axioms Required

### From Differential Geometry (blocking full proof)

| Axiom | Purpose | Mathlib Status |
|-------|---------|----------------|
| `LaplaceBeltrami` | Laplacian on manifolds | In development |
| `ProductNeckMetric` | g = dt² + g_Y | Requires vector bundles |
| `NeckMinimality` | Coarea formula | Requires measure theory |

### New Axioms for Tier 1

| Axiom | Purpose | Justification |
|-------|---------|---------------|
| `CrossSectionFirstEigenvalue` | γ = λ₁(Y) | Standard spectral theory |
| `EigenfunctionDecay` | Exponential decay into caps | Elliptic regularity |
| `PoincaréOnInterval` | π²/L² for Neumann on [0,L] | Classical 1D result |

---

## Proof Outline in Lean

### Upper Bound (constructive)

```lean
-- 1. Define test function
def test_function (K : TCSManifold) (hyp : TCSHypotheses K) : L2Function K :=
  { toFun := fun x =>
      if x ∈ K.neck then cos (π * t(x) / K.neckLength)
      else ±1  -- smoothly extended
    square_integrable := sorry }

-- 2. Compute Rayleigh quotient
theorem rayleigh_of_test_function (K : TCSManifold) (hyp : TCSHypotheses K) :
    rayleigh_quotient (test_function K hyp) ≤ π² / K.neckLength² + O(1/L³) := sorry

-- 3. Apply min-max
theorem upper_bound (K : TCSManifold) (hyp : TCSHypotheses K) :
    MassGap K ≤ π² / K.neckLength² + O(1/L³) := by
  have h := rayleigh_of_test_function K hyp
  exact le_of_rayleigh_quotient h
```

### Lower Bound (via localization)

```lean
-- 1. Decompose eigenfunction
theorem eigenfunction_decomposition (f : EigenFunction K) :
    f = f₀ + f_perp where
    f₀ := project_to_zero_mode f
    f_perp := orthogonal_complement f₀

-- 2. Transverse modes decay
theorem transverse_decay (f_perp : ...) (hλ : λ < γ) :
    ∫_{N} |f_perp|² ≤ (2λ/γ) * ∫_K |f|² := sorry

-- 3. 1D Poincaré on zero mode
theorem poincare_on_neck (f₀ : ...) :
    ∫ |∇f₀|² ≥ (π²/L²) * ∫ |f₀ - mean(f₀)|² := sorry

-- 4. Combine
theorem lower_bound (K : TCSManifold) (hyp : TCSHypothesesExt K) :
    MassGap K ≥ π² / K.neckLength² - O(e^{-δL}) := sorry
```

---

## Timeline

| Task | Effort | Dependencies |
|------|--------|--------------|
| Add H7 to NeckGeometry | 1h | None |
| Create Tier1Bounds.lean skeleton | 2h | H7 |
| Formalize test function construction | 3h | L2 spaces |
| Prove upper bound (modulo axioms) | 4h | Rayleigh quotient |
| Formalize localization lemma | 4h | Decay estimates |
| Prove lower bound (modulo axioms) | 4h | Localization + Poincaré |
| Connect to UniversalLaw | 2h | Full bounds |

**Total:** ~20h of focused Lean work

---

## Success Criteria

1. **Tier1Bounds.lean compiles** with documented axioms (no `sorry`)
2. **Main theorem statement** matches TIER1_THEOREM.md exactly
3. **Constants π², δ, C** are explicit (not existential)
4. **Certificate theorem** verifies algebraic relations
5. **Integration with UniversalLaw** produces λ₁ × H* = 14
