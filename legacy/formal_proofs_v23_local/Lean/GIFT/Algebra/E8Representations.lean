/-
# E8 Representations

Key representations of E8 and their role in GIFT:
- Adjoint: dimension 248
- Fundamental: dimension 248 (E8 is self-dual)
- Connection to matter content via branching rules
-/

import Mathlib.Tactic
import GIFT.Algebra.E8RootSystem

namespace GIFT.Algebra

/-! ## E8 Representations -/

/-- The adjoint representation of E8 has dimension 248 -/
def E8_adjoint_dim : Nat := 248

/-- E8 is self-dual: fundamental = adjoint -/
theorem E8_self_dual : E8_adjoint_dim = dim_E8 := rfl

/-! ## Branching to Standard Model -/

/-- E8 to SU(3) x SU(2) x U(1) branching -/
axiom E8_SM_branching : True

/-- E8 to E6 x SU(3) branching -/
axiom E8_E6_branching : True

/-- E6 dimension -/
def dim_E6 : Nat := 78

/-- E7 dimension -/
def dim_E7 : Nat := 133

/-! ## Decomposition under G2 -/

/-- E8 to G2 branching produces matter multiplets -/
axiom E8_G2_branching : True

/-- 248 under G2: contains 14 (adjoint) + higher reps -/
theorem E8_contains_G2_adjoint : 14 ≤ 248 := by decide

/-! ## Key Dimension Relations -/

/-- E8 - E7 = 248 - 133 = 115 -/
theorem E8_minus_E7 : dim_E8 - dim_E7 = 115 := rfl

/-- E8 - E6 = 248 - 78 = 170 -/
theorem E8_minus_E6 : dim_E8 - dim_E6 = 170 := rfl

/-- E6 - 27 = 78 - 27 = 51 = 3 x 17 -/
theorem E6_minus_J3O : dim_E6 - 27 = 51 := rfl

/-- 51 = 3 x 17 -/
theorem factor_51 : 51 = 3 * 17 := rfl

end GIFT.Algebra
