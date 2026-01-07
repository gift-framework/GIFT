-- Auto-generated from K7_Explicit_Metric_v32.ipynb
-- GIFT v3.2 Numerical Certificate

namespace K7Certificate

/-- Target metric determinant -/
def det_g_target : ℚ := 65 / 32

/-- Measured mean determinant -/
def det_g_measured : Float := 2.031250

/-- Torsion bound -/
def torsion_bound : ℚ := 4459 / 10000000

/-- Joyce threshold -/
def joyce_epsilon : ℚ := 1 / 10

/-- Lipschitz constant -/
def lipschitz_L : ℚ := 0 / 10000

/-- Contraction constant -/
def contraction_K : ℚ := 9 / 10

/-- Safety margin -/
def safety_margin : Float := 224.26

/-- Joyce theorem applicability -/
theorem joyce_applies : torsion_bound < joyce_epsilon := by
  native_decide

end K7Certificate
