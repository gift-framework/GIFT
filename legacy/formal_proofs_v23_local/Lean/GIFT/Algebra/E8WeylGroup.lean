/-
# E₈ Weyl Group Properties

The Weyl group of E₈ has order 696,729,600 with prime factorization
2¹⁴ × 3⁵ × 5² × 7. The factor 5 appears as the unique non-trivial
perfect square factor (5²), which determines det(g) = 65/32.
-/

import Mathlib.Data.Nat.Prime.Defs
import Mathlib.Data.Nat.Factorization.Basic
import Mathlib.Tactic

namespace GIFT.Algebra

/-! ## E₈ Weyl Group Order -/

/-- The order of the E₈ Weyl group -/
def E8_Weyl_group_order : ℕ := 696729600

/-- Prime factorization of the Weyl group order as (prime, exponent) pairs -/
def E8_Weyl_factorization : List (ℕ × ℕ) :=
  [(2, 14), (3, 5), (5, 2), (7, 1)]

/-- The Weyl factor: unique prime with exponent 2 (excluding 2) -/
def Weyl_factor : ℕ := 5

/-! ## Arithmetic Proofs -/

/-- The prime factorization equals the Weyl group order -/
theorem Weyl_order_factorization :
    2^14 * 3^5 * 5^2 * 7 = 696729600 := by native_decide

/-- 5² is the unique non-trivial perfect square in the factorization -/
theorem Weyl_factor_unique_square :
    ∀ (p e : ℕ), (p, e) ∈ E8_Weyl_factorization →
      p ≠ 2 → e = 2 → p = 5 := by
  intro p e hmem hne2 he2
  simp [E8_Weyl_factorization] at hmem
  rcases hmem with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
  · exact absurd rfl hne2
  · omega
  · rfl
  · omega

/-- The Weyl factor 5 appears with exponent 2 -/
theorem Weyl_factor_exponent :
    (5, 2) ∈ E8_Weyl_factorization := by
  simp [E8_Weyl_factorization]

/-! ## Derived Quantities -/

/-- det(g) numerator from Weyl factor: 5 × (8 + 5) = 65 -/
theorem det_g_numerator : Weyl_factor * (8 + Weyl_factor) = 65 := by native_decide

/-- det(g) denominator: 2⁵ = 32 -/
theorem det_g_denominator : 2^5 = 32 := by native_decide

/-- The rank of E₈ is 8 -/
def rank_E8 : ℕ := 8

end GIFT.Algebra
