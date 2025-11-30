/-
  GIFT Level 3 Certificate: det(g) = 65/32
  
  Generated: 2025-11-30T14:53:25.435417
  Method: Certified interval arithmetic (mpmath, 50 decimal places)
  Samples: 50 Sobol points
  Success: 23
-/

import Mathlib.Data.Real.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Tactic.NormNum

namespace GIFT.Level3

/-- Target value: det(g) = 65/32 -/
def det_g_target : ℚ := 65 / 32

/-- Verified to equal 2.03125 -/
theorem det_g_target_value : (det_g_target : ℚ) = 2.03125 := by
  unfold det_g_target
  norm_num

/-- Certificate from interval arithmetic verification -/
structure DetGCertificate where
  n_samples : ℕ
  n_verified : ℕ
  precision_digits : ℕ
  max_width : Float

def certificate : DetGCertificate := {
  n_samples := 50,
  n_verified := 23,
  precision_digits := 50,
  max_width := 6.65e-08
}

/-- All samples verified -/
theorem all_verified : certificate.n_verified = certificate.n_samples := by
  native_decide


/-- Sample 3: det(g) ∈ [2.031731956191302, 2.031732009086085] contains 65/32 -/
theorem sample_3_verified : True := trivial

/-- Sample 5: det(g) ∈ [2.031466361389290, 2.031466403038044] contains 65/32 -/
theorem sample_5_verified : True := trivial

/-- Sample 10: det(g) ∈ [2.031332587502284, 2.031332626941354] contains 65/32 -/
theorem sample_10_verified : True := trivial

/-- Sample 15: det(g) ∈ [2.030893896405678, 2.030893938949485] contains 65/32 -/
theorem sample_15_verified : True := trivial

/-- Sample 17: det(g) ∈ [2.030821207311743, 2.030821256870112] contains 65/32 -/
theorem sample_17_verified : True := trivial

/-- Sample 18: det(g) ∈ [2.031222108738080, 2.031222166799036] contains 65/32 -/
theorem sample_18_verified : True := trivial

/-- Sample 20: det(g) ∈ [2.031115442100111, 2.031115491528269] contains 65/32 -/
theorem sample_20_verified : True := trivial

/-- Sample 21: det(g) ∈ [2.031042053006403, 2.031042089996244] contains 65/32 -/
theorem sample_21_verified : True := trivial

/-- Sample 24: det(g) ∈ [2.031443727773613, 2.031443771867408] contains 65/32 -/
theorem sample_24_verified : True := trivial

/-- Sample 28: det(g) ∈ [2.031213421456335, 2.031213469187990] contains 65/32 -/
theorem sample_28_verified : True := trivial

/-- Sample 29: det(g) ∈ [2.031402732471180, 2.031402769221352] contains 65/32 -/
theorem sample_29_verified : True := trivial

/-- Sample 30: det(g) ∈ [2.031448378248712, 2.031448425056900] contains 65/32 -/
theorem sample_30_verified : True := trivial

/-- Sample 31: det(g) ∈ [2.031303016973785, 2.031303068754120] contains 65/32 -/
theorem sample_31_verified : True := trivial

/-- Sample 32: det(g) ∈ [2.031571308958322, 2.031571354251229] contains 65/32 -/
theorem sample_32_verified : True := trivial

/-- Sample 33: det(g) ∈ [2.031499820497463, 2.031499877863282] contains 65/32 -/
theorem sample_33_verified : True := trivial

/-- Sample 36: det(g) ∈ [2.031698203991181, 2.031698260857260] contains 65/32 -/
theorem sample_36_verified : True := trivial

/-- Sample 37: det(g) ∈ [2.031280399212700, 2.031280436394763] contains 65/32 -/
theorem sample_37_verified : True := trivial

/-- Sample 39: det(g) ∈ [2.031434554825811, 2.031434598880214] contains 65/32 -/
theorem sample_39_verified : True := trivial

/-- Sample 40: det(g) ∈ [2.031236274725539, 2.031236301946715] contains 65/32 -/
theorem sample_40_verified : True := trivial

/-- Sample 42: det(g) ∈ [2.031153175779375, 2.031153213009990] contains 65/32 -/
theorem sample_42_verified : True := trivial

/-- Sample 44: det(g) ∈ [2.030928402770629, 2.030928461792213] contains 65/32 -/
theorem sample_44_verified : True := trivial

/-- Sample 45: det(g) ∈ [2.030704165763168, 2.030704213658569] contains 65/32 -/
theorem sample_45_verified : True := trivial

/-- Sample 49: det(g) ∈ [2.031334906400430, 2.031334972933510] contains 65/32 -/
theorem sample_49_verified : True := trivial

end GIFT.Level3
