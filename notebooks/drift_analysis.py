#!/usr/bin/env python3
"""
Analytical comparison of two correction forms for the GIFT theta formula.

Base formula:   theta(T) = 7/6 - phi / (log T - 2)

Two candidate corrections:
  Form A (denominator): theta(T) = 7/6 - phi / (log T - 2 - c/log T)
  Form B (log-log):     theta(T) = 7/6 - phi / (log T - 2) + c * log(log T)

This script performs a pure analytical comparison:
  1. Computes theta(T) for both forms across T in [14, 3_000_000]
  2. For various small c values, shows shifts at low/high T and differential effect
  3. Taylor-expands Form A around c=0 to identify its effective correction
  4. Checks whether Mertens constant M or Euler-Mascheroni gamma appear naturally

Run:  python notebooks/drift_analysis.py
"""

import numpy as np
import math

# ============================================================
# Constants
# ============================================================
phi = (1 + math.sqrt(5)) / 2        # Golden ratio ~ 1.6180339887
EULER_GAMMA = 0.5772156649015329     # Euler-Mascheroni
MERTENS = 0.2614972128476428         # Mertens constant

A0 = 7.0 / 6.0                      # ~ 1.16667

print("=" * 80)
print("GIFT Drift Correction Analysis")
print("=" * 80)
print(f"\nConstants:")
print(f"  phi (golden ratio) = {phi:.10f}")
print(f"  A0 = 7/6          = {A0:.10f}")
print(f"  Euler-Mascheroni   = {EULER_GAMMA:.10f}")
print(f"  Mertens constant   = {MERTENS:.10f}")

# ============================================================
# Section 1: Define the formulas
# ============================================================

def theta_base(T):
    """Base formula: 7/6 - phi / (log T - 2)"""
    L = np.log(T)
    return A0 - phi / (L - 2.0)

def theta_formA(T, c):
    """Form A (denominator correction): 7/6 - phi / (log T - 2 - c/log T)"""
    L = np.log(T)
    return A0 - phi / (L - 2.0 - c / L)

def theta_formB(T, c):
    """Form B (log-log correction): 7/6 - phi / (log T - 2) + c * log(log T)"""
    L = np.log(T)
    return A0 - phi / (L - 2.0) + c * np.log(L)

# ============================================================
# Section 2: Compute shifts and differential effects
# ============================================================

T_low = 100.0
T_mid = 10_000.0
T_high = 1_000_000.0

T_points = np.array([T_low, T_mid, T_high])
T_labels = ["T=100", "T=10^4", "T=10^6"]

c_values = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

theta_base_vals = theta_base(T_points)

print("\n" + "=" * 80)
print("SECTION 1: Base formula values")
print("=" * 80)
print(f"\n{'T':>12s}  {'log T':>8s}  {'log T - 2':>10s}  {'theta_base':>12s}")
print("-" * 50)
for i, T in enumerate(T_points):
    L = np.log(T)
    print(f"{T:>12.0f}  {L:>8.4f}  {L-2:>10.4f}  {theta_base_vals[i]:>12.8f}")

# ============================================================
# Table 2: Shifts for Form A
# ============================================================

print("\n" + "=" * 80)
print("SECTION 2: Form A shifts  Delta_theta_A = theta_A(T,c) - theta_base(T)")
print("   Form A: theta = 7/6 - phi / (log T - 2 - c/log T)")
print("=" * 80)

header = f"{'c':>8s}"
for lbl in T_labels:
    header += f"  {'D_' + lbl:>14s}"
header += f"  {'Diff(hi-lo)':>14s}  {'Diff(hi-mid)':>14s}"
print(header)
print("-" * len(header))

diff_A = {}
for c in c_values:
    shifts = theta_formA(T_points, c) - theta_base_vals
    diff_hi_lo = shifts[2] - shifts[0]
    diff_hi_mid = shifts[2] - shifts[1]
    diff_A[c] = {'shifts': shifts, 'diff_hi_lo': diff_hi_lo, 'diff_hi_mid': diff_hi_mid}
    row = f"{c:>8.3f}"
    for s in shifts:
        row += f"  {s:>14.8f}"
    row += f"  {diff_hi_lo:>14.8f}  {diff_hi_mid:>14.8f}"
    print(row)

# ============================================================
# Table 3: Shifts for Form B
# ============================================================

print("\n" + "=" * 80)
print("SECTION 3: Form B shifts  Delta_theta_B = theta_B(T,c) - theta_base(T)")
print("   Form B: theta = 7/6 - phi / (log T - 2) + c * log(log T)")
print("=" * 80)

header = f"{'c':>8s}"
for lbl in T_labels:
    header += f"  {'D_' + lbl:>14s}"
header += f"  {'Diff(hi-lo)':>14s}  {'Diff(hi-mid)':>14s}"
print(header)
print("-" * len(header))

diff_B = {}
for c in c_values:
    shifts = theta_formB(T_points, c) - theta_base_vals
    diff_hi_lo = shifts[2] - shifts[0]
    diff_hi_mid = shifts[2] - shifts[1]
    diff_B[c] = {'shifts': shifts, 'diff_hi_lo': diff_hi_lo, 'diff_hi_mid': diff_hi_mid}
    row = f"{c:>8.3f}"
    for s in shifts:
        row += f"  {s:>14.8f}"
    row += f"  {diff_hi_lo:>14.8f}  {diff_hi_mid:>14.8f}"
    print(row)

# ============================================================
# Section 4: Direct comparison of differential effects
# ============================================================

print("\n" + "=" * 80)
print("SECTION 4: Direct comparison -- Differential effect (high T - low T)")
print("   This is the quantity that corrects the linear alpha drift.")
print("=" * 80)

header = f"{'c':>8s}  {'Form A':>14s}  {'Form B':>14s}  {'Ratio A/B':>12s}"
print(header)
print("-" * len(header))

for c in c_values:
    dA = diff_A[c]['diff_hi_lo']
    dB = diff_B[c]['diff_hi_lo']
    ratio = dA / dB if abs(dB) > 1e-15 else float('inf')
    print(f"{c:>8.3f}  {dA:>14.8f}  {dB:>14.8f}  {ratio:>12.6f}")

# ============================================================
# Section 5: Taylor expansion of Form A around c=0
# ============================================================

print("\n" + "=" * 80)
print("SECTION 5: Taylor expansion of Form A around c=0")
print("=" * 80)

print("""
Let L = log T,  D = L - 2.   Base: theta_0 = A0 - phi/D

Form A: theta_A = A0 - phi / (D - c/L)
                = A0 - phi / [D * (1 - c/(D*L))]
                = A0 - (phi/D) * 1/(1 - c/(D*L))

Taylor expand 1/(1-x) = 1 + x + x^2 + ... where x = c/(D*L):

theta_A = A0 - (phi/D) * [1 + c/(D*L) + c^2/(D*L)^2 + ...]
        = [A0 - phi/D] - phi*c/(D^2 * L) - phi*c^2/(D^3 * L^2) - ...
        = theta_0 - phi*c / [D^2 * L] + O(c^2)

So the first-order correction from Form A is:

  Delta_theta_A ~ -phi * c / [(log T - 2)^2 * log T]

This is a DECREASING function of T (correction shrinks at high T).
The differential effect (high T vs low T) is:

  Delta_theta_A(high) - Delta_theta_A(low)
    ~ -phi*c * [1/((L_hi-2)^2 * L_hi) - 1/((L_lo-2)^2 * L_lo)]

Since L_hi > L_lo, the term 1/(D^2 * L) is smaller at high T,
so the difference is POSITIVE for positive c.
""")

# Verify numerically
print("Numerical verification of Taylor expansion:")
print(f"{'T':>12s}  {'Exact shift':>14s}  {'Taylor O(c)':>14s}  {'Taylor O(c^2)':>14s}")
print("-" * 60)

c_test = 0.01
for T in T_points:
    L = np.log(T)
    D = L - 2.0
    exact = float((theta_formA(np.array([T]), c_test) - theta_base(np.array([T])))[0])
    taylor1 = -phi * c_test / (D**2 * L)
    taylor2 = taylor1 - phi * c_test**2 / (D**3 * L**2)
    print(f"{T:>12.0f}  {exact:>14.10f}  {taylor1:>14.10f}  {taylor2:>14.10f}")

# ============================================================
# Section 6: Effective form comparison
# ============================================================

print("\n" + "=" * 80)
print("SECTION 6: Effective correction profiles across full T range")
print("=" * 80)

T_range = np.logspace(np.log10(14), np.log10(3_000_000), 1000)
L_range = np.log(T_range)

c_demo = 0.01

shift_A = theta_formA(T_range, c_demo) - theta_base(T_range)
shift_B = theta_formB(T_range, c_demo) - theta_base(T_range)

# Compute the profile shape: normalize by value at T=100
idx100 = np.argmin(np.abs(T_range - 100))
idx1k = np.argmin(np.abs(T_range - 1000))
idx10k = np.argmin(np.abs(T_range - 10000))
idx100k = np.argmin(np.abs(T_range - 100000))
idx1M = np.argmin(np.abs(T_range - 1000000))

sample_idx = [0, idx100, idx1k, idx10k, idx100k, idx1M, -1]
sample_labels = ["T=14", "T~100", "T~1000", "T~10^4", "T~10^5", "T~10^6", "T=3M"]

print(f"\nCorrection profile for c = {c_demo}:")
print(f"{'T':>10s}  {'log T':>8s}  {'Shift A':>14s}  {'Shift B':>14s}  {'A/B ratio':>12s}")
print("-" * 60)

for i, lbl in zip(sample_idx, sample_labels):
    sA = shift_A[i]
    sB = shift_B[i]
    ratio = sA / sB if abs(sB) > 1e-15 else float('inf')
    print(f"{lbl:>10s}  {L_range[i]:>8.4f}  {sA:>14.10f}  {sB:>14.10f}  {ratio:>12.6f}")

# ============================================================
# Section 7: Scaling behavior analysis
# ============================================================

print("\n" + "=" * 80)
print("SECTION 7: How each correction scales with log T")
print("=" * 80)

print("""
Form A first-order correction:
  Delta_A(T) ~ -phi * c / [(log T - 2)^2 * log T]

  At large T:  ~ -phi * c / (log T)^3     [decays as 1/(log T)^3]

Form B correction:
  Delta_B(T) = c * log(log T)

  At large T:  ~ c * log(log T)            [grows as log(log T)]

KEY INSIGHT:
  - Form A provides a VANISHING correction at large T (goes as 1/(logT)^3)
  - Form B provides a GROWING correction at large T (goes as log(log T))
  - If the drift requires INCREASING correction with T, Form B is the right shape
  - If the drift requires DECREASING correction with T, Form A is the right shape
""")

# Compute scaling ratios
print("Scaling ratios relative to T=100:")
print(f"{'T':>12s}  {'log T':>8s}  {'A rel':>10s}  {'B rel':>10s}")
print("-" * 45)

ref_A = shift_A[idx100]
ref_B = shift_B[idx100]

for i, lbl in zip(sample_idx, sample_labels):
    rel_A = shift_A[i] / ref_A if abs(ref_A) > 1e-15 else 0
    rel_B = shift_B[i] / ref_B if abs(ref_B) > 1e-15 else 0
    print(f"{lbl:>12s}  {L_range[i]:>8.4f}  {rel_A:>10.6f}  {rel_B:>10.6f}")

# ============================================================
# Section 8: Check for fundamental constants
# ============================================================

print("\n" + "=" * 80)
print("SECTION 8: Do Mertens or Euler-Mascheroni appear naturally?")
print("=" * 80)

print("""
The denominator shift d = -2 in the base formula (log T - 2).

Key number-theoretic relationships:
  - The prime-counting offset: log T - 2 can be related to
    the effective prime sieve via Mertens' theorem:
      sum_{p<=x} 1/p = log(log x) + M + O(1/log x)
    where M = 0.2615... is the Mertens constant.

  - The Euler product for zeta: log(zeta(s)) = sum_p sum_k p^{-ks}/k
    At s=1 this diverges as log(1/(s-1)) + gamma + O(s-1)
    where gamma = 0.5772... is the Euler-Mascheroni constant.

Let's check if these constants appear in natural correction coefficients.
""")

# Check: if we want Form A's c to produce a shift proportional to 1/logT,
# what c value would match certain constants?

print("Form A Taylor expansion: Delta ~ -phi * c / (D^2 * L)")
print("At T = e^e (L = e, D = e-2 = 0.71828...):")
L_ee = math.e
D_ee = L_ee - 2.0
print(f"  L = {L_ee:.6f},  D = {D_ee:.6f}")
print(f"  D^2 = {D_ee**2:.6f}")
print(f"  D^2 * L = {D_ee**2 * L_ee:.6f}")
print(f"  phi / (D^2 * L) = {phi / (D_ee**2 * L_ee):.6f}")
print()

# Check if c = gamma or c = M produces meaningful relationships
print("If c = gamma (Euler-Mascheroni):")
c_gamma = EULER_GAMMA
for T in [100, 10000, 1000000]:
    L = np.log(T)
    D = L - 2.0
    shift = -phi * c_gamma / (D**2 * L)
    print(f"  T={T:>8d}: shift = {shift:>12.8f}")

print(f"\nIf c = M (Mertens constant):")
c_M = MERTENS
for T in [100, 10000, 1000000]:
    L = np.log(T)
    D = L - 2.0
    shift = -phi * c_M / (D**2 * L)
    print(f"  T={T:>8d}: shift = {shift:>12.8f}")

print(f"\nIf c = 1/phi (reciprocal golden ratio):")
c_inv_phi = 1.0 / phi
for T in [100, 10000, 1000000]:
    L = np.log(T)
    D = L - 2.0
    shift = -phi * c_inv_phi / (D**2 * L)  # = -1/(D^2 * L) since phi*(1/phi)=1
    print(f"  T={T:>8d}: shift = {shift:>12.8f}  (note: phi * 1/phi = 1)")

# ============================================================
# Section 9: What c makes the differential match the drift?
# ============================================================

print("\n" + "=" * 80)
print("SECTION 9: Required c to produce specific differential effects")
print("=" * 80)

print("""
If alpha decreases linearly across 12 windows from T~14 to T~3M,
a typical drift might be delta_alpha ~ 0.05 (5% total drift).

The alpha normalization is proportional to theta, so we need:
  Delta_theta(high T) - Delta_theta(low T) ~ target_drift * scale_factor

Let's compute what c is needed for each form to produce a specific
differential theta shift between T=100 and T=10^6.
""")

targets = [0.001, 0.005, 0.01, 0.02, 0.05]

print(f"{'Target diff':>12s}  {'c for Form A':>14s}  {'c for Form B':>14s}")
print("-" * 45)

L_lo = np.log(100.0)
D_lo = L_lo - 2.0
L_hi = np.log(1e6)
D_hi = L_hi - 2.0

# Form A linear approximation: diff ~ -phi*c * [1/(D_hi^2*L_hi) - 1/(D_lo^2*L_lo)]
factor_A = -phi * (1.0/(D_hi**2 * L_hi) - 1.0/(D_lo**2 * L_lo))

# Form B exact: diff = c * [log(L_hi) - log(L_lo)] = c * log(L_hi/L_lo)
factor_B = np.log(L_hi) - np.log(L_lo)

for target in targets:
    c_A = target / factor_A
    c_B = target / factor_B
    print(f"{target:>12.4f}  {c_A:>14.8f}  {c_B:>14.8f}")

print(f"\nForm A sensitivity factor: {factor_A:.8f}")
print(f"Form B sensitivity factor: {factor_B:.8f}")
print(f"Ratio (A sensitivity / B sensitivity): {abs(factor_A / factor_B):.8f}")

# ============================================================
# Section 10: Special constant checks
# ============================================================

print("\n" + "=" * 80)
print("SECTION 10: Natural constant checks in required c values")
print("=" * 80)

print("""
Checking if the required c values for reasonable drift corrections
match known mathematical constants.
""")

# For Form B, if c = M (Mertens), what differential does it produce?
diff_with_M = MERTENS * (np.log(np.log(1e6)) - np.log(np.log(100)))
diff_with_gamma = EULER_GAMMA * (np.log(np.log(1e6)) - np.log(np.log(100)))

print(f"Form B with c = M (Mertens = {MERTENS:.6f}):")
print(f"  Differential (T=100 to T=10^6) = {diff_with_M:.8f}")
print(f"  Shift at T=100:  {MERTENS * np.log(np.log(100)):.8f}")
print(f"  Shift at T=10^6: {MERTENS * np.log(np.log(1e6)):.8f}")

print(f"\nForm B with c = gamma (Euler = {EULER_GAMMA:.6f}):")
print(f"  Differential (T=100 to T=10^6) = {diff_with_gamma:.8f}")
print(f"  Shift at T=100:  {EULER_GAMMA * np.log(np.log(100)):.8f}")
print(f"  Shift at T=10^6: {EULER_GAMMA * np.log(np.log(1e6)):.8f}")

# Check: Mertens' theorem says sum(1/p, p<=x) = log(log(x)) + M
# So log(log(x)) = sum(1/p, p<=x) - M
# The Form B correction c*log(log(T)) could be interpreted as
# c * [sum(1/p, p<=T) - M]

print(f"""
Interpretive connection via Mertens' theorem:
  sum(1/p, p <= T) = log(log T) + M + O(1/log T)

  Form B correction: c * log(log T) = c * [sum(1/p, p<=T) - M - O(1/logT)]

  If c = M:  correction = M * [sum(1/p, p<=T) - M - O(1/logT)]
  If c = 1:  correction = sum(1/p, p<=T) - M - O(1/logT)

  The spectral-prime connection in GIFT suggests that the correction
  should encode prime number information. Form B with c related to
  Mertens' constant would provide exactly this interpretation.
""")

# ============================================================
# Section 11: Combined form check
# ============================================================

print("=" * 80)
print("SECTION 11: What if the denominator shift is not exactly 2?")
print("=" * 80)

print("""
The current formula uses d = -2 giving (log T - 2).
Previous work tested d = -15/8 = -1.875.

If we allow d to float, the Taylor expansion of Form A gives:
  theta(T,c,d) = 7/6 - phi / (log T + d - c/log T)

At first order in c:
  Delta ~ -phi * c / [(log T + d)^2 * log T]

The optimal d absorbs part of the drift, and c handles the residual.
""")

d_values = [-2.0, -15.0/8.0, -1.75, -1.5]
c_test = 0.01

print(f"Differential effect (T=100 to T=10^6) for c = {c_test}:")
print(f"{'d':>8s}  {'Diff_A':>14s}  {'D_lo':>8s}  {'D_hi':>8s}")
print("-" * 45)

for d in d_values:
    D_lo_d = np.log(100) + d
    D_hi_d = np.log(1e6) + d
    diff = -phi * c_test * (1.0/(D_hi_d**2 * np.log(1e6)) - 1.0/(D_lo_d**2 * np.log(100)))
    print(f"{d:>8.4f}  {diff:>14.10f}  {D_lo_d:>8.4f}  {D_hi_d:>8.4f}")

# ============================================================
# Section 12: Summary
# ============================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"""
1. SCALING BEHAVIOR:
   - Form A correction decays as 1/(log T)^3 at large T
   - Form B correction grows as log(log T) at large T

   => Form A tunes the formula NEAR the pole (small T)
   => Form B provides a GLOBAL slowly-growing shift

2. DIFFERENTIAL EFFECT (T=100 to T=10^6):
   For c = 0.01:
     Form A: {diff_A[0.01]['diff_hi_lo']:>14.8f}
     Form B: {diff_B[0.01]['diff_hi_lo']:>14.8f}

   Form B differential is {abs(diff_B[0.01]['diff_hi_lo'] / diff_A[0.01]['diff_hi_lo']):.1f}x larger per unit c.

3. REQUIRED c FOR 0.01 DIFFERENTIAL:
     Form A: c = {0.01 / factor_A:.6f}
     Form B: c = {0.01 / factor_B:.6f}

   Form A requires {'much larger' if abs(0.01/factor_A) > abs(0.01/factor_B) else 'much smaller'} c.

4. NATURAL CONSTANTS:
   - Mertens M = {MERTENS:.4f} in Form B gives differential = {diff_with_M:.6f}
   - Euler gamma = {EULER_GAMMA:.4f} in Form B gives differential = {diff_with_gamma:.6f}
   - Form B with c = M has a direct Mertens theorem interpretation:
     correction ~ M * [sum(1/p, p<=T) - M]

5. TAYLOR STRUCTURE OF FORM A:
   Delta_A ~ -phi*c / [(logT - 2)^2 * logT]
   This is a rational function of logT, NOT a log-log term.
   It modifies the pole structure, not the asymptotic behavior.

6. RECOMMENDATION:
   - If the drift is LINEAR in log T (alpha decreases uniformly with
     increasing log T), Form B is the natural choice because log(log T)
     is approximately linear in log T over the relevant range.
   - If the drift is concentrated at small T, Form A is more appropriate.
   - The Mertens constant M appears naturally in Form B via the prime
     harmonic sum connection, which fits GIFT's spectral philosophy.
""")
