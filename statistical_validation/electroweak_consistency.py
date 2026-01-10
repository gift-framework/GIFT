#!/usr/bin/env python3
"""
Electroweak Consistency Analysis

Tests whether GIFT values for sin²θ_W and m_W/m_Z are consistent
with electroweak theory: m_W/m_Z = cos(θ_W) = √(1 - sin²θ_W)
"""

import math
from fractions import Fraction

# GIFT values
sin2_W_GIFT = Fraction(3, 13)  # = 0.23077
mW_mZ_GIFT = Fraction(46, 52)  # = 23/26 = 0.88462

# Experimental values
sin2_W_exp = 0.23122
mW_mZ_exp = 0.8815

print("="*70)
print("ELECTROWEAK CONSISTENCY ANALYSIS")
print("="*70)

# Calculate theoretical consistency
cos2_W_from_sin2 = 1 - float(sin2_W_GIFT)  # = 10/13
cos_W_from_sin2 = math.sqrt(cos2_W_from_sin2)  # = √(10/13)

print(f"""
GIFT sin²θ_W = 3/13 = {float(sin2_W_GIFT):.6f}

Electroweak theory requires: m_W/m_Z = cos(θ_W) = √(1 - sin²θ_W)

Predicted m_W/m_Z from sin²θ_W = 3/13:
  cos²θ_W = 1 - 3/13 = 10/13 = {10/13:.6f}
  cos θ_W = √(10/13) = {cos_W_from_sin2:.6f}

GIFT proposed m_W/m_Z = 46/52 = 23/26 = {float(mW_mZ_GIFT):.6f}

DISCREPANCY: {abs(cos_W_from_sin2 - float(mW_mZ_GIFT))/cos_W_from_sin2 * 100:.3f}%
""")

print("="*70)
print("POSSIBLE INTERPRETATIONS")
print("="*70)

print("""
1. RADIATIVE CORRECTIONS

   In the Standard Model, the relationship m_W = m_Z × cos(θ_W) is
   modified by radiative corrections:

   m_W = m_Z × cos(θ_W) × (1 + Δr)^{-1/2}

   where Δr ≈ 0.03 at the Z scale.

   Check: √(10/13) × (1 + 0.03)^{1/2} ≈ 0.8771 × 1.0149 ≈ 0.8902

   This is LARGER than 23/26 = 0.8846, so doesn't explain the discrepancy.
""")

# Actually calculate the radiative correction explanation
delta_r_needed = (float(mW_mZ_GIFT) / cos_W_from_sin2)**2 - 1
print(f"""
   To reconcile:
   If m_W/m_Z = cos(θ_W) × (1 + Δr')^{{1/2}}, then:
   Δr' needed = (23/26 / √(10/13))² - 1 = {delta_r_needed:.4f}

   This is a ~1.7% correction, plausible for radiative effects.
""")

print("""
2. BARE vs DRESSED VALUES

   sin²θ_W = 3/13 could be the "bare" (topological) value
   m_W/m_Z = 23/26 could be the "dressed" (physical) value

   This would mean GIFT captures BOTH the underlying topology
   AND the effective physical corrections.
""")

print("""
3. DIFFERENT RENORMALIZATION SCHEMES

   sin²θ_W has different values in different schemes:
   - MS-bar scheme at M_Z: 0.23122
   - On-shell scheme: 0.2229
   - Effective scheme: 0.2319

   GIFT 3/13 = 0.2308 is closest to effective scheme.
   Physical m_W/m_Z = 0.8815 uses pole masses.

   Mixing schemes could explain the apparent tension.
""")

print("="*70)
print("ALTERNATIVE: IS √(10/13) EXPRESSIBLE IN GIFT?")
print("="*70)

# √(10/13) ≈ 0.8771
sqrt_10_13 = math.sqrt(10/13)
print(f"""
√(10/13) = {sqrt_10_13:.6f}

This is an IRRATIONAL number and cannot be exactly expressed
as a ratio of GIFT integers.

However, we can find good rational approximations:
""")

# Find best rational approximations
best_approx = []
GIFT = {
    1, 2, 3, 5, 7, 8, 11, 13, 14, 21, 27, 32, 42, 52, 56, 61, 65, 77, 78, 99, 133, 168, 248, 496
}

for num in range(1, 500):
    for den in range(1, 500):
        if num/den > 0.8 and num/den < 0.95:
            error = abs(num/den - sqrt_10_13) / sqrt_10_13 * 100
            if error < 1:
                is_gift = num in GIFT and den in GIFT
                best_approx.append({
                    'frac': f"{num}/{den}",
                    'value': num/den,
                    'error': error,
                    'gift': is_gift
                })

best_approx.sort(key=lambda x: x['error'])

print("Best rational approximations to √(10/13) = 0.87706:")
print("\n| Fraction | Value | Error | GIFT? |")
print("|----------|-------|-------|-------|")
for a in best_approx[:15]:
    gift_str = "✓" if a['gift'] else ""
    print(f"| {a['frac']:8} | {a['value']:.5f} | {a['error']:.3f}% | {gift_str:5} |")

print("""

OBSERVATION: The exact value √(10/13) cannot be a GIFT ratio.

However, 7/8 = 0.875 is very close (0.23% error) and is pure GIFT:
  7 = dim(K₇)
  8 = rank(E₈)

This suggests the PHYSICAL value m_W/m_Z deviates slightly from
the geometric ideal cos(θ_W) = √(10/13).
""")

print("="*70)
print("CONCLUSION")
print("="*70)

print("""
There are THREE possible consistent interpretations:

A. RECOMMENDED: m_W/m_Z = 23/26 is NOT structural

   The proposed m_W/m_Z = 46/52 may be a numerical coincidence.
   The TRUE GIFT prediction should be:

   m_W/m_Z ≈ √(1 - sin²θ_W) = √(10/13) ≈ 0.8771

   which has NO exact GIFT expression (being irrational).
   This is consistent with GIFT predicting sin²θ_W but leaving
   m_W/m_Z as a derived consequence.

B. ALTERNATIVE: Both are structural but at different scales

   sin²θ_W = 3/13 is the UV/topological value
   m_W/m_Z = 23/26 is the IR/physical value

   The ~1.7% difference encodes running effects.

C. PROBLEMATIC: There's tension in GIFT

   If both are claimed as structural, they are inconsistent
   at the 0.86% level with electroweak theory.

RECOMMENDATION: Mark m_W/m_Z = 23/26 as "NUMERICAL COINCIDENCE -
REQUIRES VERIFICATION" in Extended Observables catalog.
""")
