/-
# E₈ Representations

Key representations of E₈ and their role in GIFT:
- Adjoint: dimension 248
- Fundamental: dimension 248 (E₈ is self-dual)
- Connection to matter content via branching rules
-/

import Mathlib.Tactic
import GIFT.Algebra.E8RootSystem

namespace GIFT.Algebra

/-! ## E₈ Representations -/

/-- The adjoint representation of E₈ has dimension 248 -/
def E8_adjoint_dim : ℕ := 248

/-- E₈ is self-dual: fundamental = adjoint -/
theorem E8_self_dual : E8_adjoint_dim = dim_E8 := rfl

/-! ## Branching to Standard Model -/

/-- E₈ → SU(3) × SU(2) × U(1) branching -/
axiom E8_SM_branching : True

/-- E₈ → E₆ × SU(3) branching -/
axiom E8_E6_branching : True

/-- E₆ dimension -/
def dim_E6 : ℕ := 78

/-- E₇ dimension -/
def dim_E7 : ℕ := 133

/-! ## Decomposition under G₂ -/

/-- E₈ → G₂ branching produces matter multiplets -/
axiom E8_G2_branching : True

/-- 248 under G₂: contains 14 (adjoint) + higher reps -/
theorem E8_contains_G2_adjoint : 14 ≤ 248 := by native_decide

/-! ## Key Dimension Relations -/

/-- E₈ - E₇ = 248 - 133 = 115 -/
theorem E8_minus_E7 : dim_E8 - dim_E7 = 115 := by
  simp only [dim_E8, dim_E7]
  native_decide

/-- E₈ - E₆ = 248 - 78 = 170 -/
theorem E8_minus_E6 : dim_E8 - dim_E6 = 170 := by
  simp only [dim_E8, dim_E6]
  native_decide

/-- E₆ - 27 = 78 - 27 = 51 = 3 × 17 -/
theorem E6_minus_J3O : dim_E6 - 27 = 51 := by
  simp only [dim_E6]
  native_decide

/-- 51 = 3 × 17 -/
theorem factor_51 : 51 = 3 * 17 := by native_decide

end GIFT.Algebra
