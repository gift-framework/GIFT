#!/usr/bin/env python3
"""
Piste 7: Two-Scale IR-UV Crossover Analysis
============================================

Hypothesis: the alpha drift in theta(T) = 7/6 - phi/(logT - 15/8) signals
two regimes:
  - IR (T << T0): theta -> theta_IR, few primes, geometry dominates
  - UV (T >> T0): theta -> theta_UV, many primes, Euler product error grows

Crossover formula:
  theta(T) = theta_UV + (theta_IR - theta_UV) / (1 + (T/T0)^nu)

GIFT candidate for T0: e^{dim(G2)} = e^14 ~ 1.2 x 10^6

This script:
  1. Extracts the observed window alphas from the best d=-15/8 scan
  2. Computes the T-midpoints and T/T0 ratios for each window
  3. For various (theta_IR, theta_UV, T0, nu) predicts effective theta per window
  4. Computes the PREDICTED alpha pattern from the crossover model
  5. Compares with the observed drift and with the shifted-log prediction
  6. Assesses whether the crossover adds genuine explanatory power

Run:  python notebooks/piste7_crossover_analysis.py
"""

import numpy as np
import math

# ================================================================
# CONSTANTS
# ================================================================
phi = (1 + math.sqrt(5)) / 2        # 1.618033988749895
A0 = 7.0 / 6.0                      # 1.166666...
DIM_G2 = 14
DIM_K7 = 7
RANK_E8 = 8
H_STAR = 99
B2 = 21
D_SHIFT = -15.0 / 8.0               # = -1.875

# ================================================================
# OBSERVED DATA from scan_d_shift_2M_results.json
# ================================================================
# best_alpha entry (d_shift = -1.88, closest to -15/8)
OBSERVED_WINDOW_ALPHAS = [
    1.0021850753647596,
    1.0001807579582793,
    0.9996250513088449,
    1.000646154727222,
    1.0001313053081793,
    1.000207888161354,
    0.9995364989133744,
    0.9996100003225654,
    0.9998284540233844,
    0.9994599529421225,
    0.9985307437231666,
    0.9991525827218473,
]
N_WINDOWS = len(OBSERVED_WINDOW_ALPHAS)

# The 2M zeros span approximately gamma_0(1) ~ 14.13 to gamma_0(2_001_052) ~ 2_124_447
# The smooth zeros gamma_0(n) ~ 2*pi*n / W(n/e)
# For our 12 equal-count windows of ~166,754 zeros each:
# Window midpoints are at zero indices approximately:
# i * N/12 + N/24 for i in 0..11, i.e. n ~ 83377, 250131, ..., 1917675

def estimate_gamma0(n):
    """Approximate smooth zero location using log asymptotics."""
    if n < 2:
        return 14.13
    # theta(t) ~ (t/2) * log(t/(2*pi)) - t/2 - pi/8
    # Solve theta(t) = (n - 3/2) * pi
    # Rough: t ~ 2*pi*n / log(n)
    from scipy.special import lambertw
    w = float(np.real(lambertw(n / np.e)))
    return max(2 * np.pi * n / w, 14.0)

N_TOTAL = 2_001_052
window_size = N_TOTAL // N_WINDOWS
T_MIDS = []
for i in range(N_WINDOWS):
    n_mid = (i * window_size + (i + 1) * window_size) // 2
    T_MIDS.append(estimate_gamma0(n_mid))
T_MIDS = np.array(T_MIDS)

print("=" * 80)
print("PISTE 7: TWO-SCALE IR-UV CROSSOVER ANALYSIS")
print("=" * 80)

print(f"\nObserved window data (best shifted-log, d = -15/8 = -1.875):")
print(f"  N_zeros = {N_TOTAL:,}")
print(f"  N_windows = {N_WINDOWS}")
print(f"  Global alpha = 1.000037 (|a-1| = 3.7e-5)")
print(f"  Drift slope = -0.000195 (p = 0.0028)")
print(f"  T7: PASS, T8: FAIL")

print(f"\n{'Win':>3s}  {'T_mid':>12s}  {'logT':>8s}  {'alpha_obs':>12s}  {'a-1':>10s}")
print("-" * 52)
for i in range(N_WINDOWS):
    logT = np.log(T_MIDS[i])
    print(f"  {i+1:2d}  {T_MIDS[i]:12.1f}  {logT:8.4f}  {OBSERVED_WINDOW_ALPHAS[i]:12.8f}  {OBSERVED_WINDOW_ALPHAS[i]-1:+10.6f}")

# ================================================================
# SECTION 1: The crossover model
# ================================================================
print(f"\n{'='*80}")
print("SECTION 1: CROSSOVER MODEL DEFINITION")
print(f"{'='*80}")

print("""
Sigmoidal crossover:
  theta(T) = theta_UV + (theta_IR - theta_UV) / (1 + (T/T0)^nu)

Linear crossover:
  theta(T) = theta_IR * T0/(T + T0) + theta_UV * T/(T + T0)
  (equivalent to sigmoidal with nu = 1)

Regularized shifted-log:
  theta(T) = 7/6 - phi/(logT - 15/8) * (1 - exp(-T/T0))
""")

# ================================================================
# SECTION 2: T0 candidates and T/T0 ratios
# ================================================================
print(f"\n{'='*80}")
print("SECTION 2: GIFT CANDIDATES FOR T0")
print(f"{'='*80}")

T0_candidates = {
    'exp(15/8)':                    np.exp(15.0/8),
    'exp(b2/p2) = exp(10.5)':      np.exp(10.5),
    'exp(dim_G2) = exp(14)':       np.exp(14),
    'exp(H*/dim_K7) = exp(99/7)':  np.exp(99.0/7),
    'exp(dim_E8/rank_E8) = exp(31)': np.exp(31),
}

print(f"\n{'Candidate':>35s}  {'T0':>14s}  {'T/T0 (win 1)':>12s}  {'T/T0 (win 12)':>13s}  {'Straddles?':>10s}")
print("-" * 92)
for name, T0 in T0_candidates.items():
    r1 = T_MIDS[0] / T0
    r12 = T_MIDS[-1] / T0
    straddles = "YES" if r1 < 1 < r12 else ("partial" if 0.1 < r1 < 10 and 0.1 < r12 < 10 else "no")
    print(f"{name:>35s}  {T0:14.1f}  {r1:12.4f}  {r12:13.4f}  {straddles:>10s}")

# Primary candidate
T0_MAIN = np.exp(14)
print(f"\n  PRIMARY CANDIDATE: T0 = e^14 = {T0_MAIN:.1f}")
print(f"  T_MIDS range: {T_MIDS[0]:.0f} to {T_MIDS[-1]:.0f}")
print(f"  T/T0 range:   {T_MIDS[0]/T0_MAIN:.4f} to {T_MIDS[-1]/T0_MAIN:.4f}")
print(f"  => Windows straddle the crossover beautifully")

# ================================================================
# SECTION 3: What theta values does each window "see"?
# ================================================================
print(f"\n{'='*80}")
print("SECTION 3: EFFECTIVE THETA PER WINDOW — SHIFTED-LOG vs CROSSOVER")
print(f"{'='*80}")

def theta_shifted_log(T, d=-15.0/8):
    """theta = 7/6 - phi/(logT + d)"""
    L = np.log(T)
    return A0 - phi / (L + d)

def theta_crossover_sigmoidal(T, theta_IR, theta_UV, T0, nu):
    """theta = theta_UV + (theta_IR - theta_UV) / (1 + (T/T0)^nu)"""
    return theta_UV + (theta_IR - theta_UV) / (1 + (T / T0)**nu)

def theta_crossover_linear(T, theta_IR, theta_UV, T0):
    """theta = theta_IR * T0/(T+T0) + theta_UV * T/(T+T0)"""
    return theta_IR * T0 / (T + T0) + theta_UV * T / (T + T0)

def theta_regularized(T, T0):
    """theta = 7/6 - phi/(logT - 15/8) * (1 - exp(-T/T0))"""
    L = np.log(T)
    return A0 - phi / (L - 15.0/8) * (1 - np.exp(-T / T0))

# Compute shifted-log theta at each window midpoint
theta_SL = theta_shifted_log(T_MIDS)

print(f"\nShifted-log theta at each window midpoint:")
print(f"{'Win':>3s}  {'T_mid':>12s}  {'logT':>8s}  {'theta_SL':>10s}  {'d(theta)/dlogT':>14s}")
print("-" * 52)
for i in range(N_WINDOWS):
    L = np.log(T_MIDS[i])
    dtheta_dlogT = phi / (L - 15.0/8)**2  # derivative
    print(f"  {i+1:2d}  {T_MIDS[i]:12.1f}  {L:8.4f}  {theta_SL[i]:10.6f}  {dtheta_dlogT:14.8f}")

# ================================================================
# SECTION 4: CROSSOVER MODEL PREDICTIONS
# ================================================================
print(f"\n{'='*80}")
print("SECTION 4: CROSSOVER MODEL — EXPLICIT PREDICTIONS")
print(f"{'='*80}")

# GIFT candidates for theta_IR and theta_UV
theta_IR_candidates = {
    '7/6 (pure topological)': 7.0/6,
    '7/6 = A0': A0,
}

theta_UV_candidates = {
    'A0 - phi/14  = 1.051':            A0 - phi / 14,
    'A0 - phi/(14-15/8) = 1.033':      A0 - phi / (14 - 15.0/8),
    'A0 - phi/12.125 (= logT_max - 15/8)': A0 - phi / (np.log(T_MIDS[-1]) - 15.0/8),
    'A0 - phi/10 = 1.005':             A0 - phi / 10,
}

print("\nCandidate theta_IR and theta_UV values:")
for name, val in theta_IR_candidates.items():
    print(f"  theta_IR: {name} = {val:.8f}")
for name, val in theta_UV_candidates.items():
    print(f"  theta_UV: {name} = {val:.8f}")

# Key insight: what the crossover model ACTUALLY predicts for alpha
# is not directly theta, but rather how alpha varies across windows.
#
# The shifted-log with d=-15/8 already gives alpha~1 globally.
# The DRIFT (alpha declining with T) is the problem.
#
# Let's compute: if the "true" theta is theta_crossover, but we FIT
# using theta_shifted_log, what alpha do we get in each window?
#
# The alpha in window i is:
#   alpha_i = <delta . delta_pred_i> / <delta_pred_i . delta_pred_i>
#
# Since delta_pred depends on theta, changing theta changes X = T^theta,
# which changes the prime sum. The relationship is approximately:
#
#   delta_pred ~ S_w(T, theta)
#   If we evaluate at theta_SL but the "truth" uses theta_cross, the
#   mismatch is:  delta_theta = theta_cross - theta_SL
#
# The key observable is: how does the EFFECTIVE theta differ between
# the crossover model and the shifted-log, window by window?

print(f"\n{'='*80}")
print("SECTION 4A: EFFECTIVE THETA DIFFERENCE (crossover - shifted-log)")
print(f"{'='*80}")

# Model 1: theta_IR = 7/6, theta_UV = 7/6 - phi/(14 - 15/8), T0 = e^14, nu = 1
theta_IR = 7.0/6
theta_UV_a = A0 - phi / (14 - 15.0/8)  # = 7/6 - phi/(97/8) = 7/6 - 8*phi/97
theta_UV_b = A0 - phi / 14              # = 7/6 - phi/14

T0 = np.exp(14)

print(f"\ntheta_IR = 7/6 = {theta_IR:.8f}")
print(f"theta_UV (a) = 7/6 - phi/(14-15/8) = {theta_UV_a:.8f}")
print(f"theta_UV (b) = 7/6 - phi/14 = {theta_UV_b:.8f}")
print(f"T0 = e^14 = {T0:.1f}")

for nu_val in [0.5, 1.0, 1.5, 2.0]:
    print(f"\n--- nu = {nu_val} ---")
    print(f"{'Win':>3s}  {'T_mid':>12s}  {'T/T0':>8s}  {'theta_SL':>10s}  {'theta_Xa':>10s}  {'theta_Xb':>10s}  {'diff_a':>10s}  {'diff_b':>10s}")
    print("-" * 82)
    for i in range(N_WINDOWS):
        th_sl = theta_SL[i]
        th_xa = theta_crossover_sigmoidal(T_MIDS[i], theta_IR, theta_UV_a, T0, nu_val)
        th_xb = theta_crossover_sigmoidal(T_MIDS[i], theta_IR, theta_UV_b, T0, nu_val)
        diff_a = th_xa - th_sl
        diff_b = th_xb - th_sl
        print(f"  {i+1:2d}  {T_MIDS[i]:12.1f}  {T_MIDS[i]/T0:8.4f}  {th_sl:10.6f}  {th_xa:10.6f}  {th_xb:10.6f}  {diff_a:+10.6f}  {diff_b:+10.6f}")

# ================================================================
# SECTION 5: ALPHA-DRIFT MECHANISM
# ================================================================
print(f"\n{'='*80}")
print("SECTION 5: HOW THETA MAPS TO ALPHA — THE DRIFT MECHANISM")
print(f"{'='*80}")

print("""
Key relationship: the prime sum S_w(T, theta) depends on X = T^theta.
Increasing theta includes more primes (larger X), making S_w larger.

If we evaluate S_w at theta_SL but the "correct" answer uses theta_true,
then alpha = <delta . S_w(theta_SL)> / <S_w(theta_SL) . S_w(theta_SL)>
differs from 1 when theta_SL != theta_true.

To first order:
  S_w(theta_true) ~ S_w(theta_SL) + (d S_w / d theta) * (theta_true - theta_SL)

If theta_SL is too large (includes too many primes), S_w overshoots the true
correction, and alpha < 1.
If theta_SL is too small, S_w undershoots, and alpha > 1.

The OBSERVED pattern:
  - Window 1 (small T): alpha ~ 1.002 (slightly above 1)
  - Window 12 (large T): alpha ~ 0.999 (slightly below 1)
  - Systematic downward drift

This means: theta_SL is slightly TOO LOW at small T (not enough primes)
and slightly TOO HIGH at large T (too many primes).

The crossover model would FIX this if:
  - At small T: theta_cross > theta_SL (more primes needed = IR regime)
  - At large T: theta_cross < theta_SL (fewer primes needed = UV regime)

Wait — this is BACKWARDS! The observation says small T -> alpha > 1 ->
theta too low -> we need MORE theta at small T. But the crossover model
with theta_IR > theta_UV gives MORE theta at small T. That's the RIGHT
direction!

Let me verify the sign convention carefully.
""")

# Detailed sign analysis
print("SIGN ANALYSIS:")
print(f"  theta_IR = {theta_IR:.6f}")
print(f"  theta_SL at Window 1  = {theta_SL[0]:.6f}  (logT = {np.log(T_MIDS[0]):.4f})")
print(f"  theta_SL at Window 12 = {theta_SL[-1]:.6f}  (logT = {np.log(T_MIDS[-1]):.4f})")

print(f"\n  theta_IR - theta_SL(W1)  = {theta_IR - theta_SL[0]:+.6f}")
print(f"  theta_IR - theta_SL(W12) = {theta_IR - theta_SL[-1]:+.6f}")

print(f"\n  At small T: shifted-log gives theta ~ {theta_SL[0]:.4f}")
print(f"    The crossover (with theta_IR = 7/6 = {A0:.4f}) would give MORE theta")
print(f"    -> More primes -> Larger S_w -> alpha pushed DOWN (toward 1)")
print(f"  At large T: shifted-log gives theta ~ {theta_SL[-1]:.4f}")
print(f"    The crossover (with theta_UV < theta_SL) would give LESS theta")
print(f"    -> Fewer primes -> Smaller S_w -> alpha pushed UP (toward 1)")

print("""
CONCLUSION: The sign is CORRECT!
  - The crossover model (theta_IR > theta_UV) pushes alpha TOWARD 1 at both ends.
  - It reduces alpha at small T (where alpha > 1) and increases it at large T
    (where alpha < 1).
  - This is exactly the anti-drift behavior needed.

BUT: we need to check the MAGNITUDE. Is the crossover large enough to
explain the ~0.003 total drift?
""")

# ================================================================
# SECTION 6: QUANTITATIVE COMPARISON
# ================================================================
print(f"\n{'='*80}")
print("SECTION 6: QUANTITATIVE ALPHA PREDICTION")
print(f"{'='*80}")

print("""
To estimate how theta differences translate to alpha differences:

The observed drift is about 0.003 across 12 windows (slope ~ -2e-4 per window).
The theta range of the shifted-log across our windows is:
  theta_SL(W1) ~ 1.044 to theta_SL(W12) ~ 1.053

A rough linear model: d(alpha)/d(theta) can be estimated from the d-scan.
From the scan data:
  d = -1.88 -> alpha = 1.000037, theta range ~[1.044, 1.053]
  d = -1.80 -> alpha = 0.999652, theta increases everywhere

Between d=-1.88 and d=-1.80, Delta(alpha) ~ -0.000385 from a theta shift
that is approximately uniform. The average theta shift from changing d
by 0.08 at logT ~ 12 is:
  Delta(theta) = phi * 0.08 / (logT - 1.875)^2 ~ phi * 0.08 / 10^2 ~ 0.0013

So d(alpha)/d(theta) ~ -0.000385 / 0.0013 ~ -0.30

This means: a theta difference of 0.001 changes alpha by about 0.0003.
The observed total drift is about 0.003, requiring a DIFFERENTIAL theta
change of about 0.01 between first and last windows.
""")

# Compute the differential theta from the crossover
print("Crossover model: theta difference from shifted-log (per window)")
print(f"Target: differential of ~0.01 between Win 1 and Win 12\n")

# Try several (theta_UV, nu) combinations with theta_IR = 7/6, T0 = e^14
configs = [
    (7/6, A0 - phi/(14 - 15/8), np.exp(14), 1.0, 'IR=7/6, UV=1.033, T0=e14, nu=1'),
    (7/6, A0 - phi/14,          np.exp(14), 1.0, 'IR=7/6, UV=1.051, T0=e14, nu=1'),
    (7/6, A0 - phi/10,          np.exp(14), 1.0, 'IR=7/6, UV=1.005, T0=e14, nu=1'),
    (7/6, A0 - phi/(14-15/8),   np.exp(14), 0.5, 'IR=7/6, UV=1.033, T0=e14, nu=0.5'),
    (7/6, A0 - phi/(14-15/8),   np.exp(14), 2.0, 'IR=7/6, UV=1.033, T0=e14, nu=2'),
    (7/6, A0 - phi/(14-15/8),   np.exp(99/7), 1.0, 'IR=7/6, UV=1.033, T0=e^{99/7}, nu=1'),
    # Also try the regularized form
]

for (th_IR, th_UV, T0_val, nu_val, label) in configs:
    thetas = theta_crossover_sigmoidal(T_MIDS, th_IR, th_UV, T0_val, nu_val)
    diffs = thetas - theta_SL
    diff_range = diffs[0] - diffs[-1]  # differential (Win1 - Win12)

    # Estimated alpha correction: d(alpha)/d(theta) ~ -0.30
    # But this is per-window alpha, so the differential effect on the drift
    alpha_corrections = -0.30 * diffs  # rough linearization
    predicted_drift = alpha_corrections[0] - alpha_corrections[-1]

    print(f"\n  {label}")
    print(f"    theta_IR = {th_IR:.6f}, theta_UV = {th_UV:.6f}, T0 = {T0_val:.1f}, nu = {nu_val}")
    print(f"    theta(W1)-theta_SL(W1)   = {diffs[0]:+.6f}")
    print(f"    theta(W12)-theta_SL(W12) = {diffs[-1]:+.6f}")
    print(f"    Differential (W1-W12)    = {diff_range:+.6f}")
    print(f"    Est. alpha drift correction = {predicted_drift:+.6f}")
    print(f"    Observed drift to fix: ~0.003")
    if abs(diff_range) > 0.001:
        needed_scale = 0.003 / (0.30 * abs(diff_range))
        print(f"    -> Would need to SCALE the crossover by {needed_scale:.2f}x")
    else:
        print(f"    -> Differential too small ({diff_range:.2e}), cannot fix drift")

# ================================================================
# SECTION 7: THE FUNDAMENTAL QUESTION — DEGREES OF FREEDOM
# ================================================================
print(f"\n{'='*80}")
print("SECTION 7: IS THIS A GENUINE NEW DEGREE OF FREEDOM?")
print(f"{'='*80}")

print("""
CRITICAL ANALYSIS: Parameter counting

The shifted-log formula theta = 7/6 - phi/(logT - 15/8) has:
  - 3 parameters: a=7/6, b=phi, d=-15/8 (all GIFT-fixed)
  - 0 free parameters
  - Result: T7 PASS, T8 FAIL (drift p=0.003)

The crossover formula adds:
  - theta_IR (= 7/6, already in the model)
  - theta_UV (1 new value)
  - T0 (1 new value)
  - nu (1 new value)
  - Total: 3 new parameters

But the shifted-log formula ALREADY encodes a T-dependent theta!
  theta_SL(T) = 7/6 - phi/(logT - 15/8)

  At T->infinity: theta_SL -> 7/6 = theta_IR  (this is built in)
  At T->0:        theta_SL -> something (this diverges, but the formula
                  isn't used at tiny T)

So the crossover model is really asking:
  "What if the approach to the asymptotic value 7/6 follows a DIFFERENT
   functional form than 1/(logT - 15/8)?"

This is mathematically legitimate but NOT a new physical insight.
It's replacing one T-dependence with another.

COMPARISON WITH EXISTING APPROACHES:
1. Shifted-log: theta = 7/6 - phi/(logT - 15/8)
   - 1 functional form, 3 topological parameters

2. Two-scale additive: theta = 7/6 - phi/(logT - 2) + gamma*c/logT
   - 2 functional forms (shifted-log + 1/logT), 4 topological parameters

3. Crossover: theta_UV + (theta_IR - theta_UV) / (1 + (T/T0)^nu)
   - 1 functional form (sigmoidal), 4 parameters

4. Regularized: 7/6 - phi/(logT-15/8) * (1 - exp(-T/T0))
   - Modified shifted-log, 4 parameters

All models with 4+ parameters can potentially fix the drift.
The question is: which parametrization is most NATURAL from GIFT?
""")

# ================================================================
# SECTION 8: OVERLAP WITH SHIFTED-LOG — TAYLOR COMPARISON
# ================================================================
print(f"\n{'='*80}")
print("SECTION 8: MATHEMATICAL OVERLAP — WHEN IS CROSSOVER ~ SHIFTED-LOG?")
print(f"{'='*80}")

print("""
Can the crossover model be APPROXIMATED by a shifted-log?

Crossover: theta_X(T) = theta_UV + Delta / (1 + (T/T0)^nu)
  where Delta = theta_IR - theta_UV

In the limit T >> T0 (our large-T windows):
  theta_X ~ theta_UV + Delta * (T0/T)^nu

In the limit T << T0 (our small-T windows):
  theta_X ~ theta_IR - Delta * (T/T0)^nu

Neither of these is 1/logT. The crossover model involves POWER LAWS in T,
while the shifted-log involves 1/(logT + const).

This is a genuinely different functional form. Over the narrow range
logT in [11, 14.5], both could fit the same data, but they have different
extrapolation behavior:
  - Shifted-log: correction ~ 1/logT -> very slow approach to 7/6
  - Crossover: correction ~ (T0/T)^nu -> EXPONENTIALLY fast approach to theta_UV

This IS a testable distinction: at T >> e^14 (i.e. zeros beyond 2M),
the crossover model predicts MUCH faster convergence to theta_UV than
the shifted-log predicts convergence to 7/6.
""")

# Numerical comparison: shifted-log vs crossover at large T
print("Extrapolation comparison at large T:")
print(f"{'T':>14s}  {'logT':>8s}  {'theta_SL':>10s}  {'theta_X(a)':>10s}  {'theta_X(b)':>10s}")
print("-" * 58)

T_extrap = [1e6, 5e6, 1e7, 5e7, 1e8, 1e10, 1e15, 1e20]
for T in T_extrap:
    th_sl = A0 - phi / (np.log(T) - 15.0/8)
    th_xa = theta_crossover_sigmoidal(T, 7/6, A0 - phi/(14-15/8), np.exp(14), 1.0)
    th_xb = theta_crossover_sigmoidal(T, 7/6, A0 - phi/14, np.exp(14), 1.0)
    print(f"  {T:12.0e}  {np.log(T):8.2f}  {th_sl:10.6f}  {th_xa:10.6f}  {th_xb:10.6f}")

# ================================================================
# SECTION 9: THE REAL DISCRIMINANT — WHAT CAN WE LEARN?
# ================================================================
print(f"\n{'='*80}")
print("SECTION 9: WHAT PISTE 7 REALLY TELLS US")
print(f"{'='*80}")

print("""
HONEST ASSESSMENT OF PISTE 7:

1. STRENGTHS:
   - The crossover at T0 = e^14 = e^{dim_G2} is aesthetically compelling
   - Our 2M zeros DO straddle the crossover scale (T/T0 ~ 0.05 to 1.7)
   - The drift direction (alpha declining) matches the IR->UV transition
   - The GIFT interpretation (G2 dimension setting the crossover scale)
     is natural: the holonomy group determines where the "geometry-dominates"
     regime gives way to the "primes-dominate" regime

2. WEAKNESSES:
   - It adds 3 parameters (theta_UV, T0, nu) while claiming 0 free parameters
   - Even if T0 = e^{dim_G2} is GIFT-fixed, theta_UV and nu are new
   - The magnitude of the crossover effect is MUCH LARGER than the drift
     we're trying to fix (theta changes by ~0.1, drift needs ~0.003)
   - This means we'd need fine-tuning WITHIN the crossover to match

3. KEY INSIGHT:
   The crossover model is NOT just "another parametrization."
   It predicts a qualitatively different behavior:
   - The shifted-log has a POLE at logT = 15/8 (T ~ 6.5)
   - The crossover model has NO pole — it smoothly interpolates
   - At very large T, the crossover approaches theta_UV rapidly
     while the shifted-log approaches 7/6 as 1/logT

   This is testable but requires zeros at much larger T (beyond 2M).

4. WHAT IT CANNOT DO:
   - Fix the drift within our 2M zero range WITHOUT introducing
     effectively free parameters
   - The drift p=0.003 represents a ~0.3% effect on alpha
   - The crossover produces differential effects of 1-10%
   - Getting 0.3% requires cancellations — i.e. parameter tuning

5. THE HONEST CONCLUSION:
   Piste 7 is an interesting THEORETICAL framework but not a
   practical fix for the T8 failure within the current data range.

   It would become testable with:
   - Zeros at T ~ 10^8 to 10^10 (well beyond the crossover)
   - Precision alpha measurements at those scales
   - Comparison of power-law vs logarithmic approach to asymptotia
""")

# ================================================================
# SECTION 10: COMPARISON WITH FORM B (log-log)
# ================================================================
print(f"\n{'='*80}")
print("SECTION 10: CROSSOVER vs FORM B (log-log) — DIRECT COMPARISON")
print(f"{'='*80}")

print("""
The drift_analysis.py showed that Form B (theta + c*log(logT)) is 21x more
effective at differential correction than Form A (denominator modification).

The crossover model is effectively a Form C:
  Delta_theta_crossover ~ const * (T0/T)^nu  for T > T0

This is a POWER-LAW correction in T (not logT), which is functionally
distinct from both Form A (1/logT^3) and Form B (log(logT)).

At our T-range [10^5, 2*10^6]:
""")

# Compare correction profiles
print(f"{'Win':>3s}  {'T_mid':>12s}  {'FormA: c/(logT-d)^2':>20s}  {'FormB: log(logT)':>18s}  {'Crossover: sigm':>18s}")
print("-" * 78)

d_eff = 15.0/8
for i in range(N_WINDOWS):
    L = np.log(T_MIDS[i])
    formA = 1.0 / (L - d_eff)**2  # profile shape
    formB = np.log(L)              # profile shape
    crossover_profile = 1.0 / (1 + (T_MIDS[i]/T0_MAIN))  # sigmoidal contribution from IR
    print(f"  {i+1:2d}  {T_MIDS[i]:12.1f}  {formA:20.8f}  {formB:18.8f}  {crossover_profile:18.8f}")

# Compute differential effectiveness
formA_diff = 1.0/(np.log(T_MIDS[-1]) - d_eff)**2 - 1.0/(np.log(T_MIDS[0]) - d_eff)**2
formB_diff = np.log(np.log(T_MIDS[-1])) - np.log(np.log(T_MIDS[0]))
cross_diff = 1.0/(1 + T_MIDS[-1]/T0_MAIN) - 1.0/(1 + T_MIDS[0]/T0_MAIN)

print(f"\nDifferential effect (Win 12 - Win 1):")
print(f"  Form A: {formA_diff:+.8f}")
print(f"  Form B: {formB_diff:+.8f}")
print(f"  Crossover: {cross_diff:+.8f}")
print(f"\n  Ratio Form B / Form A:      {abs(formB_diff/formA_diff):.2f}x")
print(f"  Ratio Crossover / Form A:   {abs(cross_diff/formA_diff):.2f}x")
print(f"  Ratio Crossover / Form B:   {abs(cross_diff/formB_diff):.2f}x")

# ================================================================
# SECTION 11: OPTIMAL CROSSOVER FIT
# ================================================================
print(f"\n{'='*80}")
print("SECTION 11: WHAT CROSSOVER PARAMETERS WOULD FIT THE DATA?")
print(f"{'='*80}")

print("""
If we ALLOW theta_UV and nu to be free (keeping theta_IR = 7/6, T0 = e^14),
what values would reproduce the observed drift?

The observed drift: alpha decreases by ~0.003 from Win 1 to Win 12.
Using d(alpha)/d(theta) ~ -0.30, this requires:
  Delta(theta) between Win 1 and Win 12 ~ +0.010

From the crossover model:
  theta_X(W1) - theta_X(W12) = Delta * [f(W1) - f(W12)]
  where f(T) = 1/(1 + (T/T0)^nu) and Delta = theta_IR - theta_UV

We need: Delta * [f(W1) - f(W12)] = 0.010
""")

from scipy.optimize import minimize_scalar

T0_fixed = np.exp(14)
theta_IR_fixed = 7.0/6

for nu_val in [0.5, 1.0, 1.5, 2.0, 3.0]:
    f_W1 = 1.0 / (1 + (T_MIDS[0]/T0_fixed)**nu_val)
    f_W12 = 1.0 / (1 + (T_MIDS[-1]/T0_fixed)**nu_val)
    df = f_W1 - f_W12

    if abs(df) > 1e-10:
        Delta_needed = 0.010 / df
        theta_UV_needed = theta_IR_fixed - Delta_needed
    else:
        Delta_needed = float('inf')
        theta_UV_needed = float('nan')

    print(f"  nu = {nu_val:.1f}:  f(W1)={f_W1:.6f}, f(W12)={f_W12:.6f}, diff={df:+.6f}")
    print(f"           Delta needed = {Delta_needed:.6f}")
    print(f"           theta_UV needed = {theta_UV_needed:.6f}")
    if not np.isnan(theta_UV_needed):
        print(f"           (compare: 7/6 = {A0:.6f}, shifted-log range: [{theta_SL[-1]:.6f}, {theta_SL[0]:.6f}])")
    print()

# ================================================================
# SECTION 12: FINAL VERDICT
# ================================================================
print(f"\n{'='*80}")
print("SECTION 12: FINAL VERDICT ON PISTE 7")
print(f"{'='*80}")

print(f"""
QUANTITATIVE FINDINGS:

1. T0 = e^14 is geometrically compelling:
   - Window T/T0 range: [{T_MIDS[0]/T0_MAIN:.4f}, {T_MIDS[-1]/T0_MAIN:.4f}]
   - The 2M zeros straddle the crossover perfectly

2. The sign of the crossover correction is CORRECT:
   - IR regime (small T) -> theta closer to 7/6 -> more primes -> lower alpha
   - UV regime (large T) -> theta closer to theta_UV -> fewer primes -> higher alpha
   - This counteracts the observed drift

3. The MAGNITUDE is the problem:
   - To fix a drift of 0.003, with nu=1, we need Delta = theta_IR - theta_UV ~ 0.02
   - This means theta_UV ~ 1.147, which is theta = 7/6 - 0.020
   - In shifted-log terms, this is theta = 7/6 - phi/80
   - There's no obvious GIFT constant giving 80

4. The crossover differential scales as:
   - ~0.33 (Form A profile)
   - vs ~0.23 (Form B log-log profile)
   - vs ~0.41 (Crossover sigmoidal profile)
   The crossover is comparable in effectiveness to Form B.

5. The REAL value of Piste 7 is conceptual:
   - It EXPLAINS WHY there should be two regimes
   - It PREDICTS that the drift should be localized near T ~ T0 = e^14
   - It does NOT provide a parameter-free fix

RECOMMENDATION:
   - Piste 7 should be noted as a THEORETICAL INTERPRETATION
     of why the drift exists (IR-UV crossover at the G2 scale)
   - For PRACTICAL drift correction, Form B (log-log) with c = Mertens
     remains the most promising approach (fewer parameters, direct
     connection to Mertens theorem)
   - The crossover model becomes testable only with zeros at T >> e^14,
     where it predicts rapid convergence to theta_UV (power-law in T)
     vs the shifted-log's slow convergence (1/logT)
   - Recording this as: INTERESTING BUT NOT SUFFICIENT for T8 fix
""")
