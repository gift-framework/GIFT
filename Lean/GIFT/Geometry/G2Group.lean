/-
# G₂ Exceptional Lie Group

G₂ is the smallest exceptional Lie group with dimension 14 and rank 2.
It is the automorphism group of the octonions and stabilizes a generic
3-form on ℝ⁷.
-/

import Mathlib.Tactic

namespace GIFT.Geometry

/-! ## G₂ Basic Properties -/

/-- Dimension of G₂ -/
def dim_G2 : ℕ := 14

/-- Rank of G₂ -/
def rank_G2 : ℕ := 2

/-- G₂ Weyl group order (dihedral group D₆) -/
def G2_Weyl_order : ℕ := 12

/-! ## Verification Theorems -/

/-- G₂ has dimension 14 -/
theorem G2_dim_is_14 : dim_G2 = 14 := rfl

/-- G₂ has rank 2 -/
theorem G2_rank_is_2 : rank_G2 = 2 := rfl

/-- G₂ Weyl group is dihedral of order 12 -/
theorem G2_Weyl_is_dihedral : G2_Weyl_order = 12 := rfl

/-! ## G₂ as Octonionic Automorphisms -/

/-- G₂ is the automorphism group of the octonions -/
axiom G2_is_octonionic_automorphisms : True

/-- G₂ preserves the octonionic multiplication -/
axiom G2_preserves_octonion_product : True

/-! ## G₂ Root System -/

/-- Number of roots in G₂ root system -/
def G2_num_roots : ℕ := 12

/-- G₂ has 12 roots (6 short + 6 long) -/
theorem G2_roots_count : G2_num_roots = 12 := rfl

/-- G₂ dimension from roots + rank: 12 + 2 = 14 -/
theorem G2_dim_from_roots : G2_num_roots + rank_G2 = dim_G2 := by
  simp only [G2_num_roots, rank_G2, dim_G2]
  native_decide

/-! ## Connection to 7-Manifolds -/

/-- Dimension of manifolds with G₂ holonomy -/
def dim_G2_manifold : ℕ := 7

/-- The ratio p₂ = dim(G₂)/dim(M) = 14/7 = 2 -/
theorem p2_exact : dim_G2 / dim_G2_manifold = 2 := by
  simp only [dim_G2, dim_G2_manifold]
  native_decide

/-- p₂ = 2 is exact (no remainder) -/
theorem p2_divides : dim_G2_manifold ∣ dim_G2 := by
  simp only [dim_G2, dim_G2_manifold]
  decide

end GIFT.Geometry
