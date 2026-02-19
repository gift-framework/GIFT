#!/usr/bin/env python3
"""
Piste 6: Mertens constant M instead of Euler-Mascheroni gamma
=============================================================

Comprehensive analytical investigation of whether the Mertens constant
M = 0.2614972128... is the "right" constant for the drift correction
in the GIFT mollified Dirichlet polynomial.

Key questions:
  1. Why M might be more natural than gamma for a sum over primes
  2. Quantitative comparison: which magnitude matches observed drift?
  3. Mertens' theorem and Euler product truncation error
  4. GIFT expressions for M (approximations from topological constants)
  5. Effective Mertens constant for cos^2 mollifier
  6. Combined formula predictions vs observed window alphas

Uses observed data from scan_d_shift_2M_results.json and scan_c_order2_2M_results.json.

Run:  python notebooks/piste6_mertens_analysis.py
"""

import numpy as np
import math
import json
import os

# ============================================================
# CONSTANTS
# ============================================================
phi = (1 + math.sqrt(5)) / 2            # 1.6180339887...
EULER_GAMMA = 0.5772156649015329         # Euler-Mascheroni
MERTENS = 0.2614972128476428             # Meissel-Mertens constant

# GIFT topological constants
DIM_E8 = 248
RANK_E8 = 8
DIM_G2 = 14
DIM_K7 = 7
B2 = 21
B3 = 77
H_STAR = 99       # b2 + b3 + 1
P2 = 2             # Pontryagin class
N_GEN = 3          # Generation count
WEYL = 5           # Weyl factor
D_BULK = 11        # M-theory bulk dim
DIM_J3O = 27       # Exceptional Jordan algebra

# Observed data from 2M-zero scans
# From scan_d_shift_2M_results.json: baseline d=-15/8 (=-1.875)
OBSERVED_BASELINE = {
    'd': -1.875,
    'alpha': 1.0000129848014754,
    'drift_slope': -0.00019274618662296693,
    'drift_p': 0.0028448782351159683,
    'window_alphas': [
        1.0021340378417427, 1.0001517222915766, 0.999599947753436,
        1.0006242078787027, 1.0001105754949795, 1.0001881138537425,
        0.9995173625848127, 0.9995914985671138, 0.9998104811874935,
        0.9994423961423151, 0.9985134685907155, 0.9991357060890154
    ],
    'T_mids': [
        63699.78, 170651.68, 270836.22, 367565.12, 461995.31,
        554718.91, 646093.86, 736357.42, 825679.88, 914188.46,
        1001980.11, 1089133.29
    ]
}

# From scan_c_order2_2M_results.json: best drift (c_order2=0.78, d=-15/8)
OBSERVED_BEST_DRIFT = {
    'c_order2': 0.78,
    'alpha': 0.9976715281206268,
    'drift_slope': -1.2406479759218138e-06,
    'drift_p': 0.9780850358701596,
    'window_alphas': [
        0.9971582209522366, 0.9973595329520666, 0.9971870487157921,
        0.9985162347653074, 0.9981191672198015, 0.9982883333661515,
        0.99767876834824,   0.9978137443689788, 0.9980834529998809,
        0.9977552870163072, 0.9968534395306248, 0.9975138654763974
    ]
}


print("=" * 80)
print("PISTE 6: MERTENS CONSTANT M vs EULER-MASCHERONI gamma")
print("=" * 80)
print(f"\nFundamental constants:")
print(f"  gamma (Euler-Mascheroni) = {EULER_GAMMA:.16f}")
print(f"  M     (Mertens)          = {MERTENS:.16f}")
print(f"  M/gamma                  = {MERTENS/EULER_GAMMA:.10f}")
print(f"  gamma - M                = {EULER_GAMMA - MERTENS:.10f}")
print(f"  phi                      = {phi:.16f}")


# ============================================================
# SECTION 1: WHY M IS MORE NATURAL THAN gamma
# ============================================================
print("\n" + "=" * 80)
print("SECTION 1: Why M is more natural than gamma for the mollified sum")
print("=" * 80)

print("""
The mollified Dirichlet polynomial S_w(T) is a sum over PRIMES:

  S_w(T) = -1/theta'(T) * sum_{p<=X} sum_{k>=1} w(p^k/X) * sin(T*log(p^k)) / (k * p^{k/2})

The key number-theoretic series are:

  Harmonic series (all integers):  H_N = sum_{n=1}^{N} 1/n = log(N) + gamma + O(1/N)
  Prime harmonic (primes only):    sum_{p<=N} 1/p = log(log N) + M + O(1/log N)

Since S_w sums over primes, the NATURAL constant is M, not gamma.
The error in truncating the prime sum at p <= X is governed by:

  sum_{p>X} 1/p = -log(log X) - M + sum_{p<=X} 1/p + O(1/log X)

By Mertens' theorem, the REMAINDER is:

  sum_{p>X} 1/p ~ 1/log(X) + O(1/log^2(X))

With X = T^theta, this becomes 1/(theta * log T).
""")

# Numerical illustration
T_values = [100, 1000, 10000, 100000, 1000000]
theta_nominal = 7.0/6.0  # approx theta at moderate T

print("Prime sum truncation remainder ~ 1/(theta*logT):")
print(f"{'T':>12s}  {'logT':>8s}  {'theta*logT':>12s}  {'1/(th*logT)':>12s}  {'1/log^2(T)':>12s}")
print("-" * 60)
for T in T_values:
    L = math.log(T)
    th_L = theta_nominal * L
    print(f"{T:>12d}  {L:>8.4f}  {th_L:>12.4f}  {1.0/th_L:>12.8f}  {1.0/L**2:>12.8f}")


# ============================================================
# SECTION 2: QUANTITATIVE COMPARISON FOR DRIFT CORRECTION
# ============================================================
print("\n" + "=" * 80)
print("SECTION 2: Quantitative comparison of drift correction magnitudes")
print("=" * 80)

# Observed drift: window alphas decrease from ~1.0021 to ~0.9985
# over logT range from log(63700) ~ 11.06 to log(1089133) ~ 13.90
# Total alpha range: ~0.0036 (from window 1 to window 11)
wa = np.array(OBSERVED_BASELINE['window_alphas'])
T_mids = np.array(OBSERVED_BASELINE['T_mids'])
log_T_mids = np.log(T_mids)

alpha_spread = wa[0] - wa[-1]  # first window minus last
alpha_mean = np.mean(wa)

print(f"\nObserved baseline (d = -15/8):")
print(f"  Global alpha  = {OBSERVED_BASELINE['alpha']:.10f}")
print(f"  Alpha spread  = {alpha_spread:.6f} (window 1 - window 12)")
print(f"  Drift slope   = {OBSERVED_BASELINE['drift_slope']:.8f} per window")
print(f"  Drift p-value = {OBSERVED_BASELINE['drift_p']:.6f}")
print(f"  logT range    = [{log_T_mids[0]:.4f}, {log_T_mids[-1]:.4f}]")

# The drift correction needs to produce a DIFFERENTIAL theta shift
# of the right magnitude to counteract this alpha drift.
# The alpha drift over 11 windows is: 11 * drift_slope = 11 * (-0.000193) = -0.00212
total_drift = 11 * OBSERVED_BASELINE['drift_slope']
print(f"\n  Total drift (11 windows) = {total_drift:.6f}")

print(f"\nCandidate correction coefficients c = constant/H*:")
candidates = {
    'gamma/H*':     EULER_GAMMA / H_STAR,
    'M/H*':         MERTENS / H_STAR,
    'gamma*dim_G2/H*': EULER_GAMMA * DIM_G2 / H_STAR,
    'M*dim_G2/H*':  MERTENS * DIM_G2 / H_STAR,
    'gamma/dim_E8': EULER_GAMMA / DIM_E8,
    'M/dim_E8':     MERTENS / DIM_E8,
    '1/H*':         1.0 / H_STAR,
    '1/dim_E8':     1.0 / DIM_E8,
}

print(f"\n{'Name':>25s}  {'c value':>14s}  {'c/logT(lo)':>12s}  {'c/logT(hi)':>12s}  {'Diff':>12s}")
print("-" * 80)

for name, c in sorted(candidates.items(), key=lambda x: x[1]):
    shift_lo = c / log_T_mids[0]
    shift_hi = c / log_T_mids[-1]
    diff = shift_hi - shift_lo
    print(f"{name:>25s}  {c:>14.10f}  {shift_lo:>12.8f}  {shift_hi:>12.8f}  {diff:>12.8f}")


# ============================================================
# SECTION 3: MERTENS' THEOREM AND EULER PRODUCT TRUNCATION
# ============================================================
print("\n" + "=" * 80)
print("SECTION 3: Mertens' theorem and Euler product truncation error")
print("=" * 80)

print("""
The mollified Dirichlet polynomial is approximately a truncated Euler product.
At s = 1/2 + iT, the Euler product is:

  zeta(s) = prod_{p} (1 - p^{-s})^{-1}

Truncating at p <= X = T^theta gives error:

  log[ prod_{p>X} (1 - p^{-s})^{-1} ] = sum_{p>X} sum_{k>=1} p^{-ks}/k
                                        ~ sum_{p>X} 1/p^{1/2}  (at s=1/2+iT, oscillatory)

The NON-OSCILLATORY part of the truncation error is governed by:

  E(X) ~ exp(sum_{p>X} 1/p) = exp(1/log(X) + O(1/log^2(X)))    [by Mertens]
       ~ 1 + 1/log(X) + 1/(2*log^2(X)) + ...

So the truncation error contributes a MULTIPLICATIVE correction:
  alpha(T) ~ 1 - correction/log(X) = 1 - correction/(theta*logT)
""")

# Compute the Mertens remainder for different X = T^theta
print("Mertens remainder 1/(theta*logT) at various T:")
print(f"{'T':>12s}  {'logT':>8s}  {'X=T^(7/6)':>12s}  {'logX':>8s}  {'1/logX':>10s}  {'M/logX':>10s}")
print("-" * 65)

for T in T_values:
    L = math.log(T)
    X = T ** theta_nominal
    logX = theta_nominal * L
    remainder = 1.0 / logX
    M_remainder = MERTENS / logX
    print(f"{T:>12d}  {L:>8.4f}  {X:>12.1f}  {logX:>8.4f}  {remainder:>10.6f}  {M_remainder:>10.6f}")


# ============================================================
# SECTION 4: GIFT EXPRESSIONS FOR M
# ============================================================
print("\n" + "=" * 80)
print("SECTION 4: GIFT topological approximations to M = 0.2614972128...")
print("=" * 80)

# Systematic search over ratios of GIFT constants
gift_constants = {
    'dim_E8': DIM_E8, 'rank_E8': RANK_E8, 'dim_G2': DIM_G2,
    'dim_K7': DIM_K7, 'b2': B2, 'b3': B3, 'H*': H_STAR,
    'p2': P2, 'N_gen': N_GEN, 'Weyl': WEYL, 'D_bulk': D_BULK,
    'dim_J3O': DIM_J3O,
}

# Simple ratios a/b
print("\nSimple ratios closest to M = 0.2614972128:")
print(f"{'Expression':>40s}  {'Value':>14s}  {'|Error|':>12s}  {'Rel %':>8s}")
print("-" * 80)

approx_list = []
for n1, v1 in gift_constants.items():
    for n2, v2 in gift_constants.items():
        if v2 == 0: continue
        ratio = v1 / v2
        if 0.01 < ratio < 10.0:
            err = abs(ratio - MERTENS)
            approx_list.append((f"{n1}/{n2}", ratio, err))

# Also try (a+b)/c, a/(b+c), a*b/c, etc.
extras = [
    ("p2/rank_E8",            P2 / RANK_E8),
    ("1/4",                   0.25),
    ("dim_K7/dim_J3O",        DIM_K7 / DIM_J3O),
    ("(dim_G2+1)/(2*D_bulk*rank_E8/N_gen)", (DIM_G2+1)/(2*D_BULK*RANK_E8/N_GEN)),
    ("N_gen/(D_bulk+p2)",     N_GEN / (D_BULK + P2)),
    ("dim_K7/(b2+p2+Weyl)",   DIM_K7 / (B2 + P2 + WEYL)),
    ("dim_K7/(dim_J3O-1)",    DIM_K7 / (DIM_J3O - 1)),
    ("(b2-dim_G2)/(dim_J3O-1)", (B2 - DIM_G2) / (DIM_J3O - 1)),
    ("rank_E8/(b2+dim_G2-Weyl)", RANK_E8 / (B2 + DIM_G2 - WEYL)),
    ("p2*N_gen/(dim_J3O-Weyl)",  P2*N_GEN / (DIM_J3O - WEYL)),
    ("rank_E8/(2*b2-dim_G2-Weyl)", RANK_E8 / (2*B2 - DIM_G2 - WEYL)),
    ("(Weyl+p2)/(dim_J3O-1)",  (WEYL+P2)/(DIM_J3O-1)),
    ("b3/(2*dim_E8+H*-p2)",  B3 / (2*DIM_E8 + H_STAR - P2)),
    ("dim_G2/(b3-dim_G2-p2+N_gen)", DIM_G2 / (B3 - DIM_G2 - P2 + N_GEN)),
    ("dim_G2/(b3-N_gen*DIM_G2+dim_K7)", DIM_G2 / (B3 - N_GEN*DIM_G2 + DIM_K7)),
    ("N_gen*Weyl/(b3-dim_G2-Weyl-p2)", N_GEN*WEYL / (B3 - DIM_G2 - WEYL - P2)),
    ("(rank_E8-1)/(dim_J3O-1)", (RANK_E8-1)/(DIM_J3O-1)),
    ("dim_K7/(b2+p2+N_gen+1)", DIM_K7/(B2+P2+N_GEN+1)),
    ("dim_K7/(b2+Weyl+1)",   DIM_K7/(B2+WEYL+1)),
    ("Weyl/(2*(dim_G2+Weyl))", WEYL/(2*(DIM_G2+WEYL))),
    ("dim_K7*N_gen/(b3+N_gen)", DIM_K7*N_GEN/(B3+N_GEN)),
    ("(b2-dim_K7)/(b3-p2*N_gen*dim_K7)", (B2-DIM_K7)/(B3-P2*N_GEN*DIM_K7)),
    ("dim_G2/(b3-dim_G2*N_gen+dim_K7)", DIM_G2/(B3-DIM_G2*N_GEN+DIM_K7)),
    ("(N_gen+p2)*Weyl/(H*-p2)", (N_GEN+P2)*WEYL/(H_STAR-P2)),
    ("dim_G2/(H*-b2-dim_G2-Weyl)", DIM_G2/(H_STAR-B2-DIM_G2-WEYL)),
    ("(dim_G2-D_bulk)/(D_bulk+p2)", (DIM_G2-D_BULK)/(D_BULK+P2)),
    ("Weyl*p2/(b3-b2-dim_G2-Weyl)", WEYL*P2/(B3-B2-DIM_G2-WEYL)),
    ("p2/(rank_E8-p2+N_gen/(dim_K7/N_gen))", P2/(RANK_E8-P2+N_GEN/(DIM_K7/N_GEN))),
    ("(p2*dim_K7)/(b3-dim_G2-p2*dim_K7)", (P2*DIM_K7)/(B3-DIM_G2-P2*DIM_K7)),
    ("(N_gen*Weyl-1)/(b3-dim_G2-Weyl)", (N_GEN*WEYL-1)/(B3-DIM_G2-WEYL)),
    ("rank_E8/(b2+dim_G2-Weyl-p2)", RANK_E8/(B2+DIM_G2-WEYL-P2)),
]

for name, val in extras:
    err = abs(val - MERTENS)
    approx_list.append((name, val, err))

# Sort by error
approx_list.sort(key=lambda x: x[2])

for name, val, err in approx_list[:25]:
    rel = 100 * err / MERTENS
    print(f"{name:>40s}  {val:>14.10f}  {err:>12.10f}  {rel:>7.3f}%")

print(f"\n  M exact = {MERTENS:.16f}")
print(f"\n  Best GIFT approximation: {approx_list[0][0]} = {approx_list[0][1]:.10f}")
print(f"  Error: {approx_list[0][2]:.2e} ({100*approx_list[0][2]/MERTENS:.4f}%)")


# ============================================================
# SECTION 5: EFFECTIVE MERTENS CONSTANT FOR cos^2 MOLLIFIER
# ============================================================
print("\n" + "=" * 80)
print("SECTION 5: Effective Mertens constant for cos^2 mollifier")
print("=" * 80)

print("""
The standard Mertens theorem uses a SHARP cutoff at p <= X.
Our mollified sum uses a SMOOTH cos^2 cutoff:

  w(u) = cos^2(pi*u/2)  for u in [0,1],  0 otherwise
  where u = k*log(p) / log(X)

The effective Mertens constant for a general smooth cutoff h(u) is:

  M_eff = M + integral_0^1 [h(u) - 1] * du/u   (leading correction)

For the sharp cutoff h(u) = 1 for u in [0,1]:
  M_sharp = M + 0 = M

For cos^2(pi*u/2):
  h(u) = cos^2(pi*u/2) = (1 + cos(pi*u))/2
  h(u) - 1 = (cos(pi*u) - 1)/2

  integral_0^1 (cos(pi*u)-1)/(2u) du = (1/2) * integral_0^1 (cos(pi*u)-1)/u du

  Using: integral_0^x (cos(t)-1)/t dt = Ci(x) - gamma - log(x)
  where Ci is the cosine integral.

  integral_0^pi (cos(t)-1)/t dt = Ci(pi) - gamma - log(pi)

  Ci(pi) ~ 0.07366...
  So: integral = 0.07366 - 0.57722 - 1.14473 = -1.64829

  M_eff_cos2 = M + (1/2) * (-1.64829) = M - 0.82415
""")

# Numerical computation of the effective Mertens constant
from scipy import integrate

def integrand_mertens_correction(u):
    """(cos^2(pi*u/2) - 1) / u"""
    if u < 1e-15:
        return -math.pi**2 / 8  # limit as u -> 0
    return (math.cos(math.pi * u / 2)**2 - 1.0) / u

result, error = integrate.quad(integrand_mertens_correction, 0, 1)
M_eff_cos2 = MERTENS + result

print(f"Numerical integration of (cos^2(pi*u/2) - 1)/u from 0 to 1:")
print(f"  integral = {result:.10f}")
print(f"  M_eff(cos^2) = M + integral = {M_eff_cos2:.10f}")
print(f"  |M_eff| = {abs(M_eff_cos2):.10f}")

# But this might not be the right normalization. Let's compute the
# WEIGHTED prime sum contribution instead.
# The Mertens-like sum with cos^2 weight is:
# sum_{p<=X} cos^2(pi*log(p)/(2*log(X))) / p
# ~ integral_2^X cos^2(pi*log(t)/(2*log(X))) / (t*log(t)) dt  [by PNT]

# Substituting u = log(t)/log(X):
# = integral_{log2/logX}^1 cos^2(pi*u/2) * (1/(log(X)*u)) du   [approx]
# ~ (1/logX) * integral_0^1 cos^2(pi*u/2)/u du

# So the coefficient of 1/logX in the weighted Mertens sum is:
def weighted_integral_cos2(u):
    """cos^2(pi*u/2) / u"""
    if u < 1e-15:
        return 1.0  # limit as u -> 0
    return math.cos(math.pi * u / 2)**2 / u

# This integral diverges logarithmically as u->0.
# More precisely, the finite part (after subtracting log(1/epsilon)):
# integral_eps^1 cos^2(pi*u/2)/u du ~ log(1/eps) + (finite correction)
# The finite correction is the effective Mertens constant.

# Better approach: compute integral_0^1 [cos^2(pi*u/2) - 1]/u du + integral_0^1 1/u du
# The second integral diverges but cancels the prime sum's log divergence.
# The finite correction IS the integral we already computed.

print(f"\nThe cos^2 mollifier modifies the Mertens constant by:")
print(f"  Delta_M = integral_0^1 [cos^2(pi*u/2) - 1]/u du = {result:.10f}")
print(f"  This REDUCES the effective Mertens constant to M_eff = {M_eff_cos2:.10f}")
print(f"  Sign reversal: the correction is NEGATIVE.")

# Actually, for the truncation ERROR (tail sum), the relevant quantity is different.
# The truncation error with cos^2 mollifier is:
# E_trunc = sum_p [1 - w(p)] / p  (for p <= X)  +  sum_{p > X} 1/p
# The second sum is the standard Mertens remainder ~ 1/log(X)
# The first sum adds a smoothing correction

def truncation_correction_integrand(u):
    """[1 - cos^2(pi*u/2)] / u  for the excess near cutoff"""
    if u < 1e-15:
        return math.pi**2 / 8
    return (1.0 - math.cos(math.pi * u / 2)**2) / u

result_trunc, _ = integrate.quad(truncation_correction_integrand, 0, 1)
print(f"\nTruncation correction integral_0^1 [1-cos^2(pi*u/2)]/u du = {result_trunc:.10f}")
print(f"  = -Delta_M = {-result:.10f}  (as expected, opposite sign)")

# The EFFECTIVE truncation coefficient:
# alpha ~ 1 - C_eff / (theta * log T)
# where C_eff incorporates the mollifier shape
C_sharp = 1.0  # for sharp cutoff, coefficient of 1/logX in remainder
C_cos2 = 1.0 + result_trunc  # adds the smoothing excess near boundary
# Actually, more carefully:
# With sharp cutoff: E ~ 1/logX
# With cos^2: E ~ C_cos2 / logX where C_cos2 accounts for losing weight near cutoff

print(f"\n  For sharp cutoff: truncation error ~ 1/logX")
print(f"  For cos^2 cutoff: truncation error ~ {C_cos2:.6f} / logX")
print(f"  Ratio cos^2/sharp: {C_cos2:.6f}")


# ============================================================
# SECTION 6: PREDICTED DRIFT vs OBSERVED DRIFT
# ============================================================
print("\n" + "=" * 80)
print("SECTION 6: Predicted drift from Mertens vs observed window alphas")
print("=" * 80)

# Model: alpha(T) = 1 - c_eff / (theta * logT)
# The drift is the CHANGE in alpha across windows.
# alpha(T_1) - alpha(T_12) = c_eff * [1/(theta*logT_12) - 1/(theta*logT_1)]

theta_val = 7.0 / 6.0  # approximate theta
log_T = np.log(T_mids)

# Several candidate c_eff values
c_eff_candidates = {
    # Pure constants
    'c = 1 (unit)':              1.0,
    'c = M':                     MERTENS,
    'c = gamma':                 EULER_GAMMA,
    'c = gamma - M':             EULER_GAMMA - MERTENS,
    'c = M*gamma':               MERTENS * EULER_GAMMA,
    # GIFT-scaled
    'c = M * dim_G2':            MERTENS * DIM_G2,
    'c = gamma * dim_G2':        EULER_GAMMA * DIM_G2,
    'c = M * H*':                MERTENS * H_STAR,
    'c = M * b2':                MERTENS * B2,
    'c = M * dim_K7':            MERTENS * DIM_K7,
    'c = M * 2pi':               MERTENS * 2 * math.pi,
    'c = M * phi':               MERTENS * phi,
    'c = M * D_bulk':            MERTENS * D_BULK,
    'c = gamma * dim_K7':        EULER_GAMMA * DIM_K7,
    # Small coefficients
    'c = M / dim_G2':            MERTENS / DIM_G2,
    'c = M / dim_K7':            MERTENS / DIM_K7,
    'c = gamma / dim_G2':        EULER_GAMMA / DIM_G2,
    'c = gamma / dim_K7':        EULER_GAMMA / DIM_K7,
    'c = 1/dim_G2':              1.0 / DIM_G2,
}

# Observed drift: total alpha change from window 1 to window 12
observed_drift_total = wa[0] - wa[-1]  # positive = decreasing trend
observed_drift_slope = -OBSERVED_BASELINE['drift_slope']  # make positive for downward drift

print(f"\nObserved drift:")
print(f"  alpha[win1] - alpha[win12] = {observed_drift_total:.6f}")
print(f"  drift slope (per window)    = {OBSERVED_BASELINE['drift_slope']:.8f}")
print(f"  Mean alpha                  = {np.mean(wa):.8f}")

print(f"\nPredicted drift: alpha(T) = 1 - c_eff/(theta*logT)")
print(f"  theta = {theta_val:.6f}")
print(f"  logT range: [{log_T[0]:.4f}, {log_T[-1]:.4f}]")

# For each c_eff, compute predicted alpha_1 - alpha_12
inv_th_logT = 1.0 / (theta_val * log_T)
predicted_diff_factor = inv_th_logT[0] - inv_th_logT[-1]  # negative since logT_1 < logT_12

print(f"  1/(th*logT_1) - 1/(th*logT_12) = {predicted_diff_factor:.10f}")
print(f"\n{'Candidate':>30s}  {'c_eff':>12s}  {'Pred drift':>12s}  {'Obs drift':>12s}  {'Match':>8s}")
print("-" * 80)

best_match = None
best_match_err = float('inf')

for name, c_eff in sorted(c_eff_candidates.items(), key=lambda x: x[1]):
    # Predicted: alpha_window = 1 - c_eff/(theta*logT_window)
    # Predicted drift (win1 - win12):
    pred_drift = c_eff * predicted_diff_factor  # positive means alpha decreases
    err = abs(pred_drift - observed_drift_total)
    match = "***" if err < 0.001 else ("**" if err < 0.003 else ("*" if err < 0.005 else ""))
    print(f"{name:>30s}  {c_eff:>12.8f}  {pred_drift:>12.8f}  {observed_drift_total:>12.8f}  {match:>8s}")

    if err < best_match_err:
        best_match_err = err
        best_match = (name, c_eff, pred_drift)

print(f"\n  Best match: {best_match[0]}")
print(f"    c_eff = {best_match[1]:.8f}")
print(f"    Predicted drift = {best_match[2]:.8f}")
print(f"    Observed drift  = {observed_drift_total:.8f}")
print(f"    Error = {best_match_err:.8f}")


# ============================================================
# SECTION 7: WINDOW-BY-WINDOW PREDICTION
# ============================================================
print("\n" + "=" * 80)
print("SECTION 7: Window-by-window alpha prediction")
print("=" * 80)

# Find optimal c_eff by fitting to observed window alphas
# Model: alpha_w = 1 - c_eff / (theta * logT_w)
# Minimize sum (alpha_obs - alpha_pred)^2

# Linear regression: alpha_obs = A + B * (1/(theta*logT))
# where A = 1 and B = -c_eff
# But let's also allow free intercept to check
from numpy.polynomial import polynomial as P

x_var = 1.0 / (theta_val * log_T)

# Constrained fit: alpha = 1 - c * x
# => c = (1 - alpha_obs) . x / (x . x)
residuals = 1.0 - wa
c_fit_constrained = np.dot(residuals, x_var) / np.dot(x_var, x_var)

print(f"\nConstrained fit: alpha = 1 - c_eff/(theta*logT)")
print(f"  Optimal c_eff = {c_fit_constrained:.8f}")

# Unconstrained fit: alpha = a + b * x
from scipy import stats as sp_stats
slope_unc, intercept_unc, r_unc, p_unc, se_unc = sp_stats.linregress(x_var, wa)
c_fit_unconstrained = -slope_unc

print(f"\nUnconstrained fit: alpha = a + b/(theta*logT)")
print(f"  a (intercept) = {intercept_unc:.8f} (should be ~1)")
print(f"  b             = {slope_unc:.8f}")
print(f"  c_eff = -b    = {c_fit_unconstrained:.8f}")
print(f"  R^2           = {r_unc**2:.6f}")

# Check which GIFT expression matches c_fit
print(f"\nMatching c_eff = {c_fit_constrained:.8f} to GIFT expressions:")
gift_c_candidates = {
    'M':            MERTENS,
    'gamma':        EULER_GAMMA,
    'M*phi':        MERTENS * phi,
    'M*N_gen':      MERTENS * N_GEN,
    'gamma*1/phi':  EULER_GAMMA / phi,
    'M*dim_K7':     MERTENS * DIM_K7,
    'M*2pi':        MERTENS * 2 * math.pi,
    'gamma*dim_K7': EULER_GAMMA * DIM_K7,
    'M*rank_E8':    MERTENS * RANK_E8,
    'gamma*Weyl':   EULER_GAMMA * WEYL,
    'M*D_bulk':     MERTENS * D_BULK,
    'M*dim_G2':     MERTENS * DIM_G2,
    'gamma*rank_E8':EULER_GAMMA * RANK_E8,
    'gamma*N_gen':  EULER_GAMMA * N_GEN,
    'M*p2*dim_K7':  MERTENS * P2 * DIM_K7,
    'gamma*p2*N_gen': EULER_GAMMA * P2 * N_GEN,
    'M*b2':         MERTENS * B2,
    'M*dim_J3O':    MERTENS * DIM_J3O,
    'phi':          phi,
    'phi^2':        phi**2,
    '1':            1.0,
    'pi':           math.pi,
    'e':            math.e,
    'pi/2':         math.pi/2,
    'phi+1':        phi+1,
}

print(f"{'Expression':>25s}  {'Value':>12s}  {'|c-value|':>12s}  {'Rel %':>8s}")
print("-" * 62)

matches = [(name, val, abs(val - c_fit_constrained)) for name, val in gift_c_candidates.items()]
matches.sort(key=lambda x: x[2])

for name, val, err in matches[:15]:
    rel = 100 * err / abs(c_fit_constrained) if abs(c_fit_constrained) > 1e-15 else float('inf')
    print(f"{name:>25s}  {val:>12.8f}  {err:>12.8f}  {rel:>7.2f}%")

# Show window-by-window comparison for the best few
print(f"\nWindow-by-window comparison for top matches:")
print(f"{'Window':>8s}  {'logT':>8s}  {'Observed':>10s}  ", end="")

top3 = matches[:3]
for name, _, _ in top3:
    print(f"  {name:>12s}", end="")
print()
print("-" * (8 + 8 + 10 + 14 * len(top3) + 6))

for i in range(len(wa)):
    print(f"{i+1:>8d}  {log_T[i]:>8.4f}  {wa[i]:>10.6f}  ", end="")
    for name, val, _ in top3:
        pred = 1.0 - val / (theta_val * log_T[i])
        print(f"  {pred:>12.6f}", end="")
    print()


# ============================================================
# SECTION 8: THE FULL TWO-SCALE FORMULA WITH MERTENS
# ============================================================
print("\n" + "=" * 80)
print("SECTION 8: Two-scale formula with Mertens constant")
print("=" * 80)

print("""
Current two-scale formula:
  theta(T) = 7/6 - phi/(logT - d) + c_corr * f(T)

where d = -15/8 = -(dim_G2+1)/rank_E8

The drift correction needs to ADD a T-dependent shift proportional to
the Mertens truncation error. Candidate forms:

  Form M1:  c * M / (theta * logT)           [Mertens remainder]
  Form M2:  c * M * log(logT)                 [Mertens sum]
  Form M3:  c * M / logT                      [simplified]
  Form M4:  c * M / (logT - 15/8)            [matched denominator]

Each form should be evaluated with c being a GIFT topological ratio.
""")

# Compute the shift each form produces at the window T-values
d_shift = -15.0 / 8.0
L = log_T

forms = {
    'M1: M/(theta*logT)': MERTENS / (theta_val * L),
    'M2: M*log(logT)':    MERTENS * np.log(L),
    'M3: M/logT':         MERTENS / L,
    'M4: M/(logT-15/8)':  MERTENS / (L + d_shift),
}

print(f"Correction values at window midpoints:")
print(f"{'T_mid':>12s}  {'logT':>8s}", end="")
for name in forms:
    print(f"  {name:>18s}", end="")
print()
print("-" * (12 + 8 + 20 * len(forms)))

for i in range(len(T_mids)):
    print(f"{T_mids[i]:>12.0f}  {L[i]:>8.4f}", end="")
    for name, vals in forms.items():
        print(f"  {vals[i]:>18.10f}", end="")
    print()

# The DIFFERENTIAL effect (last window minus first)
print(f"\nDifferential (win12 - win1):")
for name, vals in forms.items():
    diff = vals[-1] - vals[0]
    print(f"  {name:>22s}: {diff:>+14.10f}")

# Compare with needed correction
# The drift in alpha is ~ -0.0036 from win1 to win12
# The correction delta_theta should modify alpha approximately as:
# delta_alpha ~ (d alpha/d theta) * delta_theta
# From the chain rule, alpha depends on theta through the prime sum.
# Roughly: d(alpha)/d(theta) ~ -alpha * d(log_X)/d(theta) * (something)
# But more directly, the drift slope is -0.000193 per window.
# Over 11 windows that's ~ -0.00212.

print(f"\n  Needed alpha correction (total) = {-total_drift:.6f}")
print(f"  = {observed_drift_total:.6f} (win1 - win12)")


# ============================================================
# SECTION 9: REQUIRED COEFFICIENT FOR EACH FORM
# ============================================================
print("\n" + "=" * 80)
print("SECTION 9: Required c to match drift for each form")
print("=" * 80)

print("""
We need: c * [f(T_12) - f(T_1)] = needed_alpha_correction

Since the drift correction modifies theta, and alpha depends on theta,
we need to account for the sensitivity d(alpha)/d(theta).

From the observed data:
  - When d changes from -2 to -15/8 (delta_d = 0.125), alpha changes from
    1.0006 to 1.00001, so d(alpha)/d(d) ~ -0.005/0.125 = -0.04
  - When c_order2 changes by 0.1, alpha changes by ~0.0003

For the c_order2 form, the sensitivity is:
  d(alpha)/d(c_order2) ~ -0.003 per unit c_order2

The needed alpha shift to fix drift = total_drift = -0.00212
But this is the DIFFERENTIAL shift across windows, not absolute.
""")

# The observed c_order2 that eliminates drift is 0.78
# At that point alpha = 0.9977 (shifted from 1.00001 by ~0.0023)
# The alpha-drift tradeoff shows that pushing drift to zero costs ~0.0023 in alpha

# From the c_order2 scan: per unit c, alpha shifts by about -0.003
# and drift_slope shifts by about +0.00023 per unit c
# So to zero out drift_slope = -0.000193, need c ~ 0.193/0.23 ~ 0.84
# Observed best: c_order2 = 0.78, consistent.

alpha_per_c = (OBSERVED_BEST_DRIFT['alpha'] - OBSERVED_BASELINE['alpha']) / OBSERVED_BEST_DRIFT['c_order2']
drift_per_c = (OBSERVED_BEST_DRIFT['drift_slope'] - OBSERVED_BASELINE['drift_slope']) / OBSERVED_BEST_DRIFT['c_order2'] if OBSERVED_BEST_DRIFT['c_order2'] != 0 else 0

print(f"Sensitivity from c_order2 scan (d=-15/8):")
print(f"  d(alpha)/d(c_order2)      = {alpha_per_c:.8f} per unit c")
print(f"  d(drift_slope)/d(c_order2)= {drift_per_c:.8f} per unit c")
print(f"  c_order2 to zero drift    = {-OBSERVED_BASELINE['drift_slope']/drift_per_c:.6f}")
print(f"  (observed: c_order2 = {OBSERVED_BEST_DRIFT['c_order2']})")

# Now check: is c_order2 = 0.78 expressible in GIFT terms?
print(f"\n  c_order2_optimal = 0.78 -- GIFT matches:")
c_opt = 0.78
gift_c2 = {
    'phi^2/N_gen':         phi**2 / N_GEN,
    'phi^2/e':             phi**2 / math.e,
    'b2/(dim_J3O-1)':      B2 / (DIM_J3O - 1),
    '(Weyl+N_gen)/rank_E8':   (WEYL+N_GEN)/RANK_E8,
    'D_bulk/dim_G2':       D_BULK / DIM_G2,
    '(dim_G2+1)/(2*D_bulk)':  (DIM_G2+1)/(2*D_BULK),
    'dim_K7/(rank_E8+1)':  DIM_K7 / (RANK_E8 + 1),
    'phi^2/p2^2':          phi**2 / P2**2,  # = phi^2/4
    '3/4':                 0.75,
    '4/Weyl':              4.0 / WEYL,
    'p2/e':                P2 / math.e,
    'pi/4':                math.pi / 4,
    'M*N_gen':             MERTENS * N_GEN,
    'M*pi':                MERTENS * math.pi,
    'M*e':                 MERTENS * math.e,
    'gamma*phi':           EULER_GAMMA * phi,
    'M*phi^2':             MERTENS * phi**2,
    'gamma/phi^(1/2)':     EULER_GAMMA / phi**0.5,
    '(dim_G2+1)/(2*rank_E8+N_gen)': (DIM_G2+1)/(2*RANK_E8+N_GEN),
    'b2/(dim_J3O)':        B2/DIM_J3O,
    'M*N_gen':             MERTENS * N_GEN,
    'M*e':                 MERTENS * math.e,
    'M*phi*e/N_gen':       MERTENS * phi * math.e / N_GEN,
}

c2_matches = [(name, val, abs(val - c_opt)) for name, val in gift_c2.items()]
c2_matches.sort(key=lambda x: x[2])

print(f"{'Expression':>30s}  {'Value':>12s}  {'|Error|':>12s}  {'Rel %':>8s}")
print("-" * 66)
for name, val, err in c2_matches[:12]:
    rel = 100 * err / c_opt
    print(f"{name:>30s}  {val:>12.8f}  {err:>12.8f}  {rel:>7.2f}%")


# ============================================================
# SECTION 10: THE MERTENS-CONNES CONNECTION
# ============================================================
print("\n" + "=" * 80)
print("SECTION 10: Mertens-Connes connection and test function analysis")
print("=" * 80)

print("""
In Connes' approach to the explicit formula:

  sum_rho h(rho) = integral h(t) * d[N(t)] = hat_h(0)*log(pi)/2 + ...
                   - sum_p sum_k log(p)/p^{k/2} * [hat_h(k*log p) + conj]

The test function h determines which primes contribute.
For our mollified Dirichlet polynomial, the effective test function has
compact support in [0, log X] = [0, theta * logT].

The Mertens constant appears when:
  sum_p h(log p) / p = integral_0^infty h(u) * du/u + M * hat_h(0) + ...

For h = indicator function of [0, logX]:
  sum_{p<=X} 1/p = log(logX) + M + O(1/logX)     [standard Mertens]

For h = cos^2(pi*u/(2*logX)) * indicator:
  sum_p cos^2(pi*log(p)/(2*logX)) / p
    = log(logX) + M + Delta_cos2 + O(1/logX)

where Delta_cos2 is the integral correction from Section 5.

The key insight: the drift correction coefficient should be
  c_drift ~ M (or M_eff) * (sensitivity of alpha to sum perturbation)

From the Weil explicit formula, each prime p contributes
  ~ sin(T*logp) / (sqrt(p) * theta'(T))
to the approximation S_w(T).

The TRUNCATION ERROR (primes we miss) is:
  E_trunc(T) ~ sum_{p>T^theta} sin(T*logp) / sqrt(p)
            ~ (random walk) * sqrt(sum_{p>T^theta} 1/p)
            ~ sqrt(1/(theta*logT)) * (Gaussian)

But the SYSTEMATIC part (mean, not variance) involves M directly.
""")

# Compute the systematic truncation error coefficient
print("Systematic truncation error: M * hat_h(0) / (theta * logT)")
print(f"where hat_h(0) = integral of cos^2 mollifier")

# hat_h(0) for our cos^2 mollifier
# integral_0^1 cos^2(pi*u/2) du = 1/2
hat_h0 = 0.5
print(f"  hat_h(0) = integral_0^1 cos^2(pi*u/2) du = {hat_h0}")

systematic_coeff = MERTENS * hat_h0
print(f"  Systematic coefficient: M * hat_h(0) = {systematic_coeff:.10f}")

print(f"\n  At T = 10^6 (logT = 13.816):")
L_1M = math.log(1e6)
print(f"    M * hat_h(0) / (theta*logT) = {systematic_coeff / (theta_val * L_1M):.10f}")


# ============================================================
# SECTION 11: COMBINED FORMULA PREDICTIONS
# ============================================================
print("\n" + "=" * 80)
print("SECTION 11: Combined formula predictions and comparison")
print("=" * 80)

# The full proposed formula:
# theta(T) = 7/6 - phi/(logT - 15/8) + C_M / (logT - 15/8)^2
#
# where C_M is the Mertens-derived coefficient.
# From Section 9, the optimal c_order2 ~ 0.78.
# The closest GIFT/Mertens expressions:

best_proposals = {
    'c = D_bulk/dim_G2 = 11/14':      D_BULK / DIM_G2,          # 0.7857
    'c = pi/4':                        math.pi / 4,              # 0.7854
    'c = b21/(dim_J3O-1) = 21/26':    21.0 / 26.0,              # 0.8077
    'c = (Weyl+N_gen)/rank_E8 = 1':   (WEYL+N_GEN)/RANK_E8,    # 1.0
    'c = dim_K7/(rank_E8+1) = 7/9':   DIM_K7/(RANK_E8+1),      # 0.7778
    'c = M*N_gen = 3M':               MERTENS * N_GEN,          # 0.7845
    'c = M*e':                         MERTENS * math.e,         # 0.7109
    'c = M*pi':                        MERTENS * math.pi,        # 0.8213
    'c = gamma*phi':                   EULER_GAMMA * phi,        # 0.9340
    'c = 3/4':                         0.75,
    'c = 4/5 = 4/Weyl':               4.0/5.0,
}

print(f"\nTop formula proposals for c_order2:")
print(f"  Observed optimal: c ~ 0.78")
print(f"\n{'Name':>40s}  {'c':>10s}  {'|c-0.78|':>10s}")
print("-" * 65)

for name, val in sorted(best_proposals.items(), key=lambda x: abs(x[1]-c_opt)):
    print(f"{name:>40s}  {val:>10.6f}  {abs(val-c_opt):>10.6f}")

# Check the specific Mertens connection: c = M * N_gen
print(f"\n  ** NOTABLE: M * N_gen = {MERTENS:.10f} * {N_GEN} = {MERTENS*N_GEN:.10f} **")
print(f"     D_bulk/dim_G2 = {D_BULK}/{DIM_G2} = {D_BULK/DIM_G2:.10f}")
print(f"     pi/4 = {math.pi/4:.10f}")
print(f"     Optimal c_order2 = ~0.78")
print(f"\n     These three are remarkably close:")
print(f"       M * N_gen  = {MERTENS*N_GEN:.10f}  (error from 0.78: {abs(MERTENS*N_GEN-0.78):.6f})")
print(f"       D_bulk/dim_G2 = {D_BULK/DIM_G2:.10f}  (error from 0.78: {abs(D_BULK/DIM_G2-0.78):.6f})")
print(f"       pi/4       = {math.pi/4:.10f}  (error from 0.78: {abs(math.pi/4-0.78):.6f})")

# The Mertens interpretation: 3M means three copies of the Mertens constant,
# one per generation. This is VERY natural in GIFT:
# The three generations contribute independently to the prime harmonic sum,
# and each contributes a Mertens correction M.

print(f"""
INTERPRETATION:

  c_order2 = 3M = N_gen * M

  This means: each of the N_gen = 3 fermion generations contributes
  independently a Mertens correction M to the Euler product truncation.

  The total truncation error in the mollified Dirichlet polynomial is:
    E_trunc ~ N_gen * M / (logT - 15/8)^2

  This gives the full GIFT formula:

    theta(T) = 7/6 - phi/(logT - 15/8) + 3M/(logT - 15/8)^2

  where:
    7 = dim(K7)
    6 = 3 * 2 = N_gen * p2
    phi = golden ratio (G2 metric eigenvalue)
    15 = dim(G2) + 1
    8 = rank(E8)
    3 = N_gen
    M = Mertens constant (prime harmonic truncation)
""")


# ============================================================
# SECTION 12: PREDICTIONS TO TEST ON COLAB
# ============================================================
print("\n" + "=" * 80)
print("SECTION 12: Specific predictions for Colab validation")
print("=" * 80)

# Compute predicted alpha for the 3M formula
c_3M = 3 * MERTENS  # 0.7844916...

def theta_3M(logT):
    """theta(T) = 7/6 - phi/(logT - 15/8) + 3M/(logT - 15/8)^2"""
    D = logT - 15.0/8.0
    return 7.0/6.0 - phi/D + c_3M/(D*D)

# Show theta values at window midpoints
print(f"\nFormula: theta(T) = 7/6 - phi/(logT-15/8) + 3M/(logT-15/8)^2")
print(f"  3M = {c_3M:.10f}")
print(f"\n{'T':>12s}  {'logT':>8s}  {'theta_base':>12s}  {'theta_3M':>12s}  {'Diff':>12s}")
print("-" * 60)

for T in [100, 1000, 10000, 100000, 1000000, 2000000]:
    L = math.log(T)
    D = L - 15.0/8.0
    th_base = 7.0/6.0 - phi/D
    th_3M = th_base + c_3M/(D*D)
    print(f"{T:>12d}  {L:>8.4f}  {th_base:>12.8f}  {th_3M:>12.8f}  {th_3M-th_base:>12.8f}")

# Alternative: 11/14 (D_bulk/dim_G2) -- also very close to 3M
c_11_14 = 11.0 / 14.0  # 0.7857142...

print(f"\nAlternative: c = D_bulk/dim_G2 = 11/14 = {c_11_14:.10f}")
print(f"  3M = {c_3M:.10f}")
print(f"  Difference: {abs(c_3M - c_11_14):.10f} ({100*abs(c_3M - c_11_14)/c_3M:.4f}%)")

# Alternative: pi/4
c_pi4 = math.pi / 4

print(f"\nAlternative: c = pi/4 = {c_pi4:.10f}")
print(f"  3M = {c_3M:.10f}")
print(f"  Difference: {abs(c_3M - c_pi4):.10f} ({100*abs(c_3M - c_pi4)/c_3M:.4f}%)")


# ============================================================
# SECTION 13: RELATION M = gamma + Sigma_p [log(1-1/p) + 1/p]
# ============================================================
print("\n" + "=" * 80)
print("SECTION 13: M = gamma + sum_p [log(1-1/p) + 1/p] and GIFT interpretation")
print("=" * 80)

# M = gamma + sum_p [log(1-1/p) + 1/p]
# The sum converges: each term ~ -1/(2p^2)
# sum_p 1/(2p^2) = (1/2) * P(2) where P is the prime zeta function
# P(2) ~ 0.4522474... so the correction ~ -0.2261

delta_M_gamma = MERTENS - EULER_GAMMA  # should be negative
sum_correction = delta_M_gamma  # = sum_p [log(1-1/p) + 1/p]

print(f"\n  M = gamma + sum_p [log(1-1/p) + 1/p]")
print(f"  gamma = {EULER_GAMMA:.16f}")
print(f"  M     = {MERTENS:.16f}")
print(f"  M - gamma = sum_p [log(1-1/p) + 1/p] = {delta_M_gamma:.16f}")
print(f"  |M - gamma| = {abs(delta_M_gamma):.16f}")

# Is gamma - M expressible in GIFT terms?
gamma_minus_M = EULER_GAMMA - MERTENS
print(f"\n  gamma - M = {gamma_minus_M:.16f}")
print(f"  Checking GIFT approximations:")

gm_candidates = {
    'N_gen/rank_E8 - 1/4': N_GEN/RANK_E8 - 0.25,
    '1/pi': 1.0/math.pi,
    '1/e': 1.0/math.e,
    'M': MERTENS,
    'dim_K7/b2': DIM_K7/float(B2),
    'N_gen/D_bulk': N_GEN/float(D_BULK),
    'p2/dim_K7': P2/float(DIM_K7),
    '1/pi^2': 1.0/math.pi**2,
    '(e-phi)/e': (math.e - phi)/math.e,
    'phi-pi/2': phi - math.pi/2,
    'Weyl/(2*b2)': WEYL/(2.0*B2),
    '1/(2*phi^2)': 1.0/(2*phi**2),
    'dim_K7/(2*b2)': DIM_K7/(2.0*B2),
    'rank_E8/(3*D_bulk)': RANK_E8/(3.0*D_BULK),
    'N_gen/(3*p2*Weyl)': N_GEN/(3*P2*WEYL),
}

gm_list = [(n, v, abs(v - gamma_minus_M)) for n, v in gm_candidates.items()]
gm_list.sort(key=lambda x: x[2])

print(f"{'Expression':>25s}  {'Value':>14s}  {'|Error|':>14s}  {'Rel %':>8s}")
print("-" * 65)
for name, val, err in gm_list[:10]:
    rel = 100 * err / gamma_minus_M if gamma_minus_M > 0 else float('inf')
    print(f"{name:>25s}  {val:>14.10f}  {err:>14.10f}  {rel:>7.2f}%")


# ============================================================
# SECTION 14: SUMMARY AND COLAB SCRIPT RECOMMENDATIONS
# ============================================================
print("\n" + "=" * 80)
print("GRAND SUMMARY")
print("=" * 80)

print(f"""
Key findings from Piste 6 analysis:

1. WHY MERTENS OVER EULER-MASCHERONI:
   The mollified Dirichlet polynomial sums over PRIMES, not all integers.
   Mertens' theorem (sum 1/p = log(log N) + M) directly governs the
   truncation error, while gamma governs the harmonic series (all N).
   M is the CORRECT constant for this prime-only context.

2. QUANTITATIVE MATCH:
   The optimal c_order2 for drift elimination is ~0.78.

   The BEST matches are:
     - N_gen * M = 3M = {3*MERTENS:.6f}  (0.6% from 0.78)
     - D_bulk/dim_G2 = 11/14 = {11/14:.6f}  (0.7% from 0.78)
     - pi/4 = {math.pi/4:.6f}  (0.7% from 0.78)

   The 3M interpretation is physically most compelling: each generation
   contributes one Mertens correction to the Euler product truncation.

3. EFFECTIVE MERTENS FOR cos^2 MOLLIFIER:
   The cos^2 mollifier reduces the effective Mertens constant by
   Delta_M = {result:.6f}, but this is a correction to the LEADING
   behavior, not to the c_order2 coefficient.

4. FULL GIFT FORMULA:
   theta(T) = 7/6 - phi/(logT - 15/8) + 3M/(logT - 15/8)^2

   Every piece has a topological interpretation:
     7/6 = dim(K7) / (N_gen * p2)     [asymptotic cutoff]
     phi = golden ratio                [G2 metric eigenvalue]
     15/8 = (dim(G2)+1)/rank(E8)      [resonance shift]
     3M = N_gen * Mertens             [generation-weighted truncation]

5. TO TEST ON COLAB:
   Run the following c_order2 values on 2M zeros with P_MAX=500k:
     c = 3M          = {3*MERTENS:.10f}
     c = 11/14       = {11/14:.10f}
     c = pi/4        = {math.pi/4:.10f}
     c = 0.78        = 0.7800000000  (empirical optimum)

   If 3M passes T7+T8 (or comes closer than 11/14 or pi/4),
   the Mertens interpretation is strongly supported.

6. DEEPER PREDICTION:
   If c = 3M is correct, then the formula is:
     theta(T) = dim(K7)/(N_gen*p2) - phi/(logT - (dim_G2+1)/rank_E8)
                + N_gen*M/(logT - (dim_G2+1)/rank_E8)^2

   This predicts that:
   - Changing N_gen (different universe) changes BOTH 7/6 and the
     Mertens coefficient, with correlated shifts
   - The Mertens constant M is NOT a free parameter; it is the
     universal constant governing prime harmonic sums
""")

print("=" * 80)
print("Analysis complete.")
print("=" * 80)
