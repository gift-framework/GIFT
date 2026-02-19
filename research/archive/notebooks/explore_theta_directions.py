#!/usr/bin/env python3
"""
Multi-direction exploration of adaptive cutoff θ(T).

Starting point: 7/6 − (e/φ)/logT − 2φ/log²T  (α=0.999792, T7 PASS, T8 FAIL)

Directions explored:
  A. Drift characterization — what function of logT fits the residual?
  B. Shifted-log resummation: θ(T) = a − b/(logT + d)
  C. 4-parameter polynomial: θ(T) = a − b/logT + c/log²T + d/log³T
  D. Log-log corrections: θ(T) = a − b/logT + c·log(logT)/log²T
  E. Continuous optimization (scipy.optimize)
  F. Mixed functional forms

Uses subsampled zeros for speed (50k), with full 2M validation for the best.
"""

import numpy as np
from scipy.special import loggamma, lambertw
from scipy import stats, optimize
import json, time, math, os, sys

# ==============================================================
# SETUP
# ==============================================================
phi = (1 + math.sqrt(5)) / 2
ZEROS_FILE = 'outputs/riemann_zeros_2M_genuine.npy'
OUTPUT_FILE = 'outputs/theta_directions_results.json'

P_MAX = 100_000       # Primes up to 100k for screening
K_MAX = 3
N_WINDOWS = 12        # Fine windows for drift

# Load zeros
print("Loading zeros...")
all_zeros = np.load(ZEROS_FILE)
N_TOTAL = len(all_zeros)
gamma_n_full = all_zeros
print(f"  {N_TOTAL:,} zeros loaded")

# Subsample for fast iteration
STRIDE = 40
idx_sub = np.arange(0, N_TOTAL, STRIDE)
gamma_n = gamma_n_full[idx_sub]
N = len(gamma_n)
print(f"  Subsampled to {N:,} zeros (stride {STRIDE})")

# ---------------------------------------------------------------
# Infrastructure (same as notebooks)
# ---------------------------------------------------------------
def theta_rs(t):
    t = np.asarray(t, dtype=np.float64)
    return np.imag(loggamma(0.25 + 0.5j * t)) - 0.5 * t * np.log(np.pi)

def theta_deriv(t):
    return 0.5 * np.log(np.maximum(np.asarray(t, dtype=np.float64), 1.0) / (2 * np.pi))

def smooth_zeros(gn):
    N = len(gn)
    ns = np.arange(1, N + 1, dtype=np.float64)
    # Use actual indices from full dataset
    return None  # Will compute differently

def smooth_zeros_from_indices(indices):
    """Compute gamma0 for specific zero indices."""
    ns = indices.astype(np.float64) + 1  # 1-based
    targets = (ns - 1.5) * np.pi
    w = np.real(lambertw(ns / np.e))
    t = np.maximum(2 * np.pi * ns / w, 2.0)
    for it in range(40):
        dt = (theta_rs(t) - targets) / np.maximum(np.abs(theta_deriv(t)), 1e-15)
        t -= dt
        if np.max(np.abs(dt)) < 1e-12:
            break
    return t

print("Computing smooth zeros...")
t0 = time.time()
gamma0 = smooth_zeros_from_indices(idx_sub)
delta = gamma_n - gamma0
tp = theta_deriv(gamma0)
log_g0 = np.log(np.maximum(gamma0, 2.0))
print(f"  Done in {time.time()-t0:.1f}s")

# Prime sieve
def sieve(N):
    is_p = np.ones(N + 1, dtype=bool); is_p[:2] = False
    for i in range(2, int(N**0.5) + 1):
        if is_p[i]: is_p[i*i::i] = False
    return np.where(is_p)[0]

print("Sieving primes...")
primes = sieve(P_MAX)
log_primes = np.log(primes.astype(np.float64))
print(f"  {len(primes):,} primes up to {P_MAX:,}")

# ---------------------------------------------------------------
# Core: mollified prime sum (CPU, scalar for flexibility)
# ---------------------------------------------------------------
def prime_sum(g0, tp_v, primes, log_primes, k_max, theta_func):
    """
    General prime sum where theta_func(log_g0) returns theta per zero.
    theta_func takes log(gamma0) array, returns theta array.
    """
    N = len(g0)
    S = np.zeros(N, dtype=np.float64)
    log_g0 = np.log(np.maximum(g0, 2.0))

    theta_vals = theta_func(log_g0)
    theta_vals = np.clip(theta_vals, 0.5, 2.0)
    log_X = theta_vals * log_g0

    for m in range(1, k_max + 1):
        for j, p in enumerate(primes):
            lp = log_primes[j]
            x = m * lp / log_X  # array
            mask = x < 1.0
            if not np.any(mask):
                break
            w = np.where(mask, np.cos(np.pi/2 * x)**2, 0.0)
            phase = g0 * m * lp
            coeff = 1.0 / (m * p**(m/2.0))
            S -= w * np.sin(phase) * coeff

    return -S / tp_v

def prime_sum_batched(g0, tp_v, primes_arr, log_primes_arr, k_max, theta_func,
                      batch_size=500):
    """Batched version for speed."""
    N = len(g0)
    S = np.zeros(N, dtype=np.float64)
    log_g0_v = np.log(np.maximum(g0, 2.0))

    theta_vals = theta_func(log_g0_v)
    theta_vals = np.clip(theta_vals, 0.5, 2.0)
    log_X = theta_vals * log_g0_v
    log_X_max = np.max(log_X)

    for m in range(1, k_max + 1):
        cutoff = log_X_max / m
        j_max = int(np.searchsorted(log_primes_arr, cutoff + 0.1))
        if j_max == 0:
            continue

        for b_start in range(0, j_max, batch_size):
            b_end = min(b_start + batch_size, j_max)
            logp_b = log_primes_arr[b_start:b_end]
            p_b = primes_arr[b_start:b_end].astype(np.float64)

            x = (m * logp_b[:, None]) / log_X[None, :]  # (B, N)
            w = np.where(x < 1.0, np.cos(np.pi/2 * x)**2, 0.0)
            phase = g0[None, :] * (m * logp_b[:, None])
            coeff = (1.0 / m) / p_b ** (m / 2.0)
            S -= np.sum(w * np.sin(phase) * coeff[:, None], axis=0)

    return -S / tp_v

# ---------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------
def compute_metrics(delta, delta_pred, gamma0, n_windows=12):
    denom = np.dot(delta_pred, delta_pred)
    alpha = float(np.dot(delta, delta_pred) / denom) if denom > 0 else 0.0
    residual = delta - delta_pred
    R2 = float(1.0 - np.var(residual) / np.var(delta))

    # Window alphas
    N = len(delta)
    bounds = [int(i * N / n_windows) for i in range(n_windows + 1)]
    alphas = []
    T_mids = []
    for i in range(n_windows):
        lo, hi = bounds[i], bounds[i + 1]
        d_w = delta[lo:hi]
        dp_w = delta_pred[lo:hi]
        den = np.dot(dp_w, dp_w)
        a_w = float(np.dot(d_w, dp_w) / den) if den > 0 else 0.0
        alphas.append(a_w)
        mid = (lo + hi) // 2
        T_mids.append(float(gamma0[mid]))

    # Drift
    x = np.arange(n_windows)
    slope, intercept, r, p, se = stats.linregress(x, alphas)

    # Bootstrap CI (fast, 500 resamples on subsample)
    rng = np.random.default_rng(42)
    boot = np.empty(500)
    for i in range(500):
        idx = rng.integers(0, N, size=N)
        d = delta[idx]; dp = delta_pred[idx]
        den = np.dot(dp, dp)
        boot[i] = np.dot(d, dp) / den if den > 0 else 0.0
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    T7 = bool(ci_lo <= 1.0 <= ci_hi)

    return {
        'alpha': alpha,
        'abs_alpha_m1': abs(alpha - 1),
        'R2': R2,
        'drift_slope': float(slope),
        'drift_p': float(p),
        'T7_pass': T7,
        'T7_ci': (float(ci_lo), float(ci_hi)),
        'T8_pass': bool(p > 0.05),
        'window_alphas': alphas,
        'T_mids': T_mids,
    }

def eval_theta(name, theta_func, verbose=True):
    """Evaluate a theta function and return metrics."""
    t0 = time.time()
    dp = prime_sum_batched(gamma0, tp, primes, log_primes, K_MAX, theta_func)
    elapsed = time.time() - t0
    m = compute_metrics(delta, dp, gamma0)
    m['name'] = name
    m['elapsed_s'] = elapsed
    if verbose:
        t7 = "PASS" if m['T7_pass'] else "FAIL"
        t8 = "PASS" if m['T8_pass'] else "FAIL"
        print(f"  {name:<55} α={m['alpha']:+.6f}  |α-1|={m['abs_alpha_m1']:.6f}  "
              f"drift={m['drift_slope']:+.8f} (p={m['drift_p']:.4f})  "
              f"T7={t7}  T8={t8}  [{elapsed:.1f}s]")
    return m

# ==============================================================
# DIRECTION A: DRIFT CHARACTERIZATION
# ==============================================================
print("\n" + "="*80)
print("DIRECTION A: Characterize the drift pattern")
print("="*80)

# Evaluate the winner to get window alphas on our subsample
winner_func = lambda lg: 7/6 - (math.e/phi)/lg - 2*phi/lg**2
m_winner = eval_theta("WINNER: 7/6 - e/phi/logT - 2phi/log²T", winner_func)

alphas_w = np.array(m_winner['window_alphas'])
T_mids = np.array(m_winner['T_mids'])
logT = np.log(T_mids)

print(f"\n  Window data ({N_WINDOWS} windows):")
print(f"  T range: [{T_mids[0]:.0f}, {T_mids[-1]:.0f}]")
print(f"  logT range: [{logT[0]:.2f}, {logT[-1]:.2f}]")
print(f"  alpha range: [{alphas_w.min():.6f}, {alphas_w.max():.6f}]")

# Fit different drift models to the window alphas
# α(T) ≈ 1 + f(T) where f(T) → 0
# Try: f(T) = a/logT, a/log²T, a/log³T, a*loglogT/log²T

residual_alpha = alphas_w - 1.0

# Model 1: d/log³T
x1 = 1.0 / logT**3
fit1 = np.polyfit(x1, residual_alpha, 1)
r1 = np.corrcoef(x1, residual_alpha)[0,1]**2
print(f"\n  Drift model fits (to window alpha - 1):")
print(f"    d/log³T:             coeff={fit1[0]:+.4f}  R²={r1:.4f}")

# Model 2: d*loglogT/log²T
x2 = np.log(logT) / logT**2
fit2 = np.polyfit(x2, residual_alpha, 1)
r2 = np.corrcoef(x2, residual_alpha)[0,1]**2
print(f"    d*loglogT/log²T:     coeff={fit2[0]:+.4f}  R²={r2:.4f}")

# Model 3: d/log²T  (already included, so residual shouldn't correlate)
x3 = 1.0 / logT**2
fit3 = np.polyfit(x3, residual_alpha, 1)
r3 = np.corrcoef(x3, residual_alpha)[0,1]**2
print(f"    d/log²T (check):     coeff={fit3[0]:+.4f}  R²={r3:.4f}")

# Model 4: d/(logT*(loglogT))
x4 = 1.0 / (logT * np.log(logT))
fit4 = np.polyfit(x4, residual_alpha, 1)
r4 = np.corrcoef(x4, residual_alpha)[0,1]**2
print(f"    d/(logT·loglogT):    coeff={fit4[0]:+.4f}  R²={r4:.4f}")

# Model 5: linear in 1/logT (should be mostly absorbed by b)
x5 = 1.0 / logT
fit5 = np.polyfit(x5, residual_alpha, 1)
r5 = np.corrcoef(x5, residual_alpha)[0,1]**2
print(f"    d/logT (check):      coeff={fit5[0]:+.4f}  R²={r5:.4f}")

best_drift_model = max(
    [("1/log³T", r1), ("loglogT/log²T", r2), ("1/log²T", r3),
     ("1/(logT·loglogT)", r4), ("1/logT", r5)],
    key=lambda x: x[1]
)
print(f"\n  Best drift model: {best_drift_model[0]} (R²={best_drift_model[1]:.4f})")

# ==============================================================
# DIRECTION B: SHIFTED-LOG RESUMMATION
# ==============================================================
print("\n" + "="*80)
print("DIRECTION B: Shifted-log resummation θ(T) = a − b/(logT + d)")
print("="*80)

# From the winner: a=7/6, b=e/φ, c=-2φ
# Expansion: θ = a - b/(logT+d) = a - b/logT + bd/log²T - bd²/log³T + ...
# Matching c = bd: d = c/b = (-2φ)/(e/φ) = -2φ²/e
d_predicted = -2 * phi**2 / math.e
print(f"  Predicted d from matching c/log²T: d = -2φ²/e = {d_predicted:.6f}")
print(f"  Next-order prediction: -bd²/log³T = {-(math.e/phi)*d_predicted**2:.4f}/log³T")

# Test shifted-log with predicted d
def make_shifted_log(a, b, d):
    return lambda lg: a - b / (lg + d)

results_B = []
print(f"\n  Scanning d around prediction ({d_predicted:.3f}):")

for d_val in np.linspace(d_predicted - 2, d_predicted + 2, 41):
    f = make_shifted_log(7/6, math.e/phi, d_val)
    m = eval_theta(f"shifted-log d={d_val:.3f}", f, verbose=False)
    m['d'] = d_val
    results_B.append(m)

# Find best
best_B = min(results_B, key=lambda r: r['abs_alpha_m1'])
print(f"\n  Best shifted-log:")
t7 = "PASS" if best_B['T7_pass'] else "FAIL"
t8 = "PASS" if best_B['T8_pass'] else "FAIL"
print(f"    d = {best_B['d']:.6f}  (predicted: {d_predicted:.6f})")
print(f"    α = {best_B['alpha']:+.8f}  |α-1| = {best_B['abs_alpha_m1']:.8f}")
print(f"    drift = {best_B['drift_slope']:+.8f}  (p={best_B['drift_p']:.4f})")
print(f"    T7: {t7}  T8: {t8}")

# Also optimize a,b,d jointly
print(f"\n  Joint optimization of a, b, d:")
def shifted_log_objective(params):
    a, b, d = params
    f = lambda lg: a - b / (lg + d)
    dp = prime_sum_batched(gamma0, tp, primes, log_primes, K_MAX, f)
    den = np.dot(dp, dp)
    alpha = np.dot(delta, dp) / den if den > 0 else 0.0
    return abs(alpha - 1.0)

res_B_opt = optimize.minimize(
    shifted_log_objective,
    x0=[7/6, math.e/phi, d_predicted],
    method='Nelder-Mead',
    options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
)
a_opt, b_opt, d_opt = res_B_opt.x
print(f"    Optimal: a={a_opt:.8f}  b={b_opt:.8f}  d={d_opt:.8f}")
m_Bopt = eval_theta(f"shifted-log OPT a={a_opt:.4f} b={b_opt:.4f} d={d_opt:.4f}",
                     make_shifted_log(a_opt, b_opt, d_opt))

# Check if optimal a ≈ 7/6, b ≈ e/φ
print(f"\n  Closeness to symbolic values:")
print(f"    a: {a_opt:.8f} vs 7/6 = {7/6:.8f}  (diff = {abs(a_opt-7/6):.2e})")
print(f"    b: {b_opt:.8f} vs e/φ = {math.e/phi:.8f}  (diff = {abs(b_opt-math.e/phi):.2e})")
print(f"    d: {d_opt:.8f} vs -2φ²/e = {d_predicted:.8f}  (diff = {abs(d_opt-d_predicted):.2e})")

# ==============================================================
# DIRECTION C: 4-PARAMETER POLYNOMIAL
# ==============================================================
print("\n" + "="*80)
print("DIRECTION C: 4-parameter polynomial θ = a − b/logT + c/log²T + d/log³T")
print("="*80)

# Shifted-log predicts d = -bd² = -(e/φ)(-2φ²/e)² = -(e/φ)(4φ⁴/e²) = -4φ⁴/(eφ) = -4φ³/e
d_from_shifted = -4 * phi**3 / math.e
print(f"  Shifted-log prediction for d: -4φ³/e = {d_from_shifted:.6f}")

results_C = []
print(f"\n  Scanning d/log³T around prediction ({d_from_shifted:.3f}):")

for d_val in np.linspace(-15, 15, 61):
    f = lambda lg, d=d_val: 7/6 - (math.e/phi)/lg - 2*phi/lg**2 + d/lg**3
    m = eval_theta(f"poly4 d={d_val:.2f}", f, verbose=False)
    m['d'] = d_val
    results_C.append(m)

best_C = min(results_C, key=lambda r: r['abs_alpha_m1'])
print(f"\n  Best 4-parameter polynomial:")
t7 = "PASS" if best_C['T7_pass'] else "FAIL"
t8 = "PASS" if best_C['T8_pass'] else "FAIL"
print(f"    d = {best_C['d']:.6f}  (shifted-log prediction: {d_from_shifted:.6f})")
print(f"    α = {best_C['alpha']:+.8f}  |α-1| = {best_C['abs_alpha_m1']:.8f}")
print(f"    drift = {best_C['drift_slope']:+.8f}  (p={best_C['drift_p']:.4f})")
print(f"    T7: {t7}  T8: {t8}")

# Fine scan around best d
d_center = best_C['d']
results_C2 = []
print(f"\n  Fine scan around d={d_center:.2f}:")
for d_val in np.linspace(d_center - 2, d_center + 2, 81):
    f = lambda lg, d=d_val: 7/6 - (math.e/phi)/lg - 2*phi/lg**2 + d/lg**3
    m = eval_theta(f"poly4-fine d={d_val:.3f}", f, verbose=False)
    m['d'] = d_val
    results_C2.append(m)

best_C2 = min(results_C2, key=lambda r: r['abs_alpha_m1'])
t7 = "PASS" if best_C2['T7_pass'] else "FAIL"
t8 = "PASS" if best_C2['T8_pass'] else "FAIL"
print(f"    Best d = {best_C2['d']:.6f}")
print(f"    α = {best_C2['alpha']:+.8f}  |α-1| = {best_C2['abs_alpha_m1']:.8f}")
print(f"    drift = {best_C2['drift_slope']:+.8f}  (p={best_C2['drift_p']:.4f})")
print(f"    T7: {t7}  T8: {t8}")

# ==============================================================
# DIRECTION D: LOG-LOG CORRECTIONS
# ==============================================================
print("\n" + "="*80)
print("DIRECTION D: Log-log corrections")
print("="*80)

# Form 1: θ(T) = a - b/logT + c*log(logT)/log²T
print("\n  Form D1: θ = a − b/logT + c·log(logT)/log²T")
results_D1 = []
for c_val in np.linspace(-5, 5, 41):
    f = lambda lg, c=c_val: 7/6 - (math.e/phi)/lg + c * np.log(lg) / lg**2
    m = eval_theta(f"loglog c={c_val:.2f}", f, verbose=False)
    m['c_loglog'] = c_val
    results_D1.append(m)

best_D1 = min(results_D1, key=lambda r: r['abs_alpha_m1'])
t7 = "PASS" if best_D1['T7_pass'] else "FAIL"
t8 = "PASS" if best_D1['T8_pass'] else "FAIL"
print(f"    Best c = {best_D1['c_loglog']:.6f}")
print(f"    α = {best_D1['alpha']:+.8f}  |α-1| = {best_D1['abs_alpha_m1']:.8f}")
print(f"    drift = {best_D1['drift_slope']:+.8f}  (p={best_D1['drift_p']:.4f})")
print(f"    T7: {t7}  T8: {t8}")

# Form 2: θ(T) = a - b/logT + c/log²T + d·log(logT)/log³T
print("\n  Form D2: θ = a − b/logT + c/log²T + d·log(logT)/log³T  (4-param with loglog)")
results_D2 = []
for d_val in np.linspace(-10, 10, 41):
    f = lambda lg, d=d_val: (7/6 - (math.e/phi)/lg - 2*phi/lg**2
                              + d * np.log(lg) / lg**3)
    m = eval_theta(f"poly3+loglog d={d_val:.2f}", f, verbose=False)
    m['d_loglog'] = d_val
    results_D2.append(m)

best_D2 = min(results_D2, key=lambda r: r['abs_alpha_m1'])
t7 = "PASS" if best_D2['T7_pass'] else "FAIL"
t8 = "PASS" if best_D2['T8_pass'] else "FAIL"
print(f"    Best d = {best_D2['d_loglog']:.6f}")
print(f"    α = {best_D2['alpha']:+.8f}  |α-1| = {best_D2['abs_alpha_m1']:.8f}")
print(f"    drift = {best_D2['drift_slope']:+.8f}  (p={best_D2['drift_p']:.4f})")
print(f"    T7: {t7}  T8: {t8}")

# ==============================================================
# DIRECTION E: CONTINUOUS OPTIMIZATION
# ==============================================================
print("\n" + "="*80)
print("DIRECTION E: Continuous optimization (scipy.optimize)")
print("="*80)

# E1: Optimize 3-param polynomial (a, b, c) continuously
print("\n  E1: Optimize 3-param polynomial θ = a − b/logT + c/log²T")
def poly3_objective(params):
    a, b, c = params
    f = lambda lg: a - b/lg + c/lg**2
    dp = prime_sum_batched(gamma0, tp, primes, log_primes, K_MAX, f)
    den = np.dot(dp, dp)
    alpha = np.dot(delta, dp) / den if den > 0 else 0.0
    # Composite: |α-1| + weight*|drift|
    alphas_w = []
    N = len(delta)
    for i in range(6):
        lo = int(i * N / 6)
        hi = int((i+1) * N / 6)
        d_w = delta[lo:hi]; dp_w = dp[lo:hi]
        den_w = np.dot(dp_w, dp_w)
        alphas_w.append(np.dot(d_w, dp_w) / den_w if den_w > 0 else 0.0)
    x = np.arange(6)
    slope = stats.linregress(x, alphas_w).slope
    return abs(alpha - 1.0) + 10 * abs(slope)

res_E1 = optimize.minimize(
    poly3_objective,
    x0=[7/6, math.e/phi, -2*phi],
    method='Nelder-Mead',
    options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-9}
)
a_e1, b_e1, c_e1 = res_E1.x
f_E1 = lambda lg: a_e1 - b_e1/lg + c_e1/lg**2
m_E1 = eval_theta(f"poly3-opt a={a_e1:.6f} b={b_e1:.6f} c={c_e1:.6f}", f_E1)
print(f"    Symbolic proximity:")
print(f"      a={a_e1:.8f}  (7/6={7/6:.8f}, diff={abs(a_e1-7/6):.2e})")
print(f"      b={b_e1:.8f}  (e/φ={math.e/phi:.8f}, diff={abs(b_e1-math.e/phi):.2e})")
print(f"      c={c_e1:.8f}  (-2φ={-2*phi:.8f}, diff={abs(c_e1-(-2*phi)):.2e})")

# E2: Optimize 4-param polynomial
print("\n  E2: Optimize 4-param polynomial θ = a − b/logT + c/log²T + d/log³T")
def poly4_objective(params):
    a, b, c, d = params
    f = lambda lg: a - b/lg + c/lg**2 + d/lg**3
    dp = prime_sum_batched(gamma0, tp, primes, log_primes, K_MAX, f)
    den = np.dot(dp, dp)
    alpha = np.dot(delta, dp) / den if den > 0 else 0.0
    alphas_w = []
    N = len(delta)
    for i in range(6):
        lo = int(i * N / 6)
        hi = int((i+1) * N / 6)
        d_w = delta[lo:hi]; dp_w = dp[lo:hi]
        den_w = np.dot(dp_w, dp_w)
        alphas_w.append(np.dot(d_w, dp_w) / den_w if den_w > 0 else 0.0)
    x = np.arange(6)
    slope = stats.linregress(x, alphas_w).slope
    return abs(alpha - 1.0) + 10 * abs(slope)

res_E2 = optimize.minimize(
    poly4_objective,
    x0=[7/6, math.e/phi, -2*phi, d_from_shifted],
    method='Nelder-Mead',
    options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-9}
)
a_e2, b_e2, c_e2, d_e2 = res_E2.x
f_E2 = lambda lg: a_e2 - b_e2/lg + c_e2/lg**2 + d_e2/lg**3
m_E2 = eval_theta(f"poly4-opt a={a_e2:.4f} b={b_e2:.4f} c={c_e2:.4f} d={d_e2:.4f}", f_E2)
print(f"    Optimal d = {d_e2:.8f}")
print(f"    Shifted-log prediction was: {d_from_shifted:.8f}  (diff={abs(d_e2-d_from_shifted):.4f})")

# E3: Optimize shifted-log with |α-1| + drift composite
print("\n  E3: Optimize shifted-log θ = a − b/(logT + d) with composite objective")
def shifted_composite(params):
    a, b, d = params
    f = lambda lg: a - b / (lg + d)
    dp = prime_sum_batched(gamma0, tp, primes, log_primes, K_MAX, f)
    den = np.dot(dp, dp)
    alpha = np.dot(delta, dp) / den if den > 0 else 0.0
    alphas_w = []
    N = len(delta)
    for i in range(6):
        lo = int(i * N / 6)
        hi = int((i+1) * N / 6)
        d_w = delta[lo:hi]; dp_w = dp[lo:hi]
        den_w = np.dot(dp_w, dp_w)
        alphas_w.append(np.dot(d_w, dp_w) / den_w if den_w > 0 else 0.0)
    x = np.arange(6)
    slope = stats.linregress(x, alphas_w).slope
    return abs(alpha - 1.0) + 10 * abs(slope)

res_E3 = optimize.minimize(
    shifted_composite,
    x0=[7/6, math.e/phi, d_predicted],
    method='Nelder-Mead',
    options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-9}
)
a_e3, b_e3, d_e3 = res_E3.x
f_E3 = lambda lg: a_e3 - b_e3 / (lg + d_e3)
m_E3 = eval_theta(f"shifted-opt a={a_e3:.6f} b={b_e3:.6f} d={d_e3:.6f}", f_E3)

# ==============================================================
# DIRECTION F: EXOTIC FORMS
# ==============================================================
print("\n" + "="*80)
print("DIRECTION F: Exotic functional forms")
print("="*80)

# F1: Power-law: θ(T) = a*(1 - (logT0/logT)^s)
print("\n  F1: Power-law θ = a·(1 − (L₀/logT)^s)")
def power_law_objective(params):
    a, L0, s = params
    if s <= 0 or L0 <= 0 or a <= 0:
        return 1e6
    f = lambda lg: a * (1.0 - (L0/lg)**s)
    f_safe = lambda lg: np.clip(a * (1.0 - np.where(lg > 0.1, (L0/lg)**s, 1.0)), 0.5, 2.0)
    dp = prime_sum_batched(gamma0, tp, primes, log_primes, K_MAX, f_safe)
    den = np.dot(dp, dp)
    alpha = np.dot(delta, dp) / den if den > 0 else 0.0
    return abs(alpha - 1.0)

res_F1 = optimize.minimize(
    power_law_objective,
    x0=[7/6, 2.0, 1.0],
    method='Nelder-Mead',
    options={'maxiter': 500}
)
a_f1, L0_f1, s_f1 = res_F1.x
f_F1 = lambda lg: np.clip(a_f1 * (1.0 - np.where(lg > 0.1, (L0_f1/lg)**s_f1, 1.0)), 0.5, 2.0)
m_F1 = eval_theta(f"power-law a={a_f1:.4f} L0={L0_f1:.4f} s={s_f1:.4f}", f_F1)

# F2: Double-shifted: θ(T) = a - b/(logT + d₁ + d₂/logT)
print("\n  F2: Double-shifted θ = a − b/(logT + d₁ + d₂/logT)")
def double_shifted_objective(params):
    a, b, d1, d2 = params
    f = lambda lg: a - b / (lg + d1 + d2/lg)
    dp = prime_sum_batched(gamma0, tp, primes, log_primes, K_MAX, f)
    den = np.dot(dp, dp)
    alpha = np.dot(delta, dp) / den if den > 0 else 0.0
    alphas_w = []
    N = len(delta)
    for i in range(6):
        lo = int(i * N / 6)
        hi = int((i+1) * N / 6)
        d_w = delta[lo:hi]; dp_w = dp[lo:hi]
        den_w = np.dot(dp_w, dp_w)
        alphas_w.append(np.dot(d_w, dp_w) / den_w if den_w > 0 else 0.0)
    slope = stats.linregress(np.arange(6), alphas_w).slope
    return abs(alpha - 1.0) + 10 * abs(slope)

res_F2 = optimize.minimize(
    double_shifted_objective,
    x0=[7/6, math.e/phi, d_predicted, 0.0],
    method='Nelder-Mead',
    options={'maxiter': 1000, 'xatol': 1e-7}
)
a_f2, b_f2, d1_f2, d2_f2 = res_F2.x
f_F2 = lambda lg: a_f2 - b_f2 / (lg + d1_f2 + d2_f2/lg)
m_F2 = eval_theta(f"double-shifted a={a_f2:.4f} b={b_f2:.4f} d1={d1_f2:.4f} d2={d2_f2:.4f}", f_F2)

# F3: Mertens-inspired: θ(T) = a - b/(logT + M) where M = Meissel-Mertens constant
print("\n  F3: Mertens-inspired θ = a − b/(logT + M)")
M_MERTENS = 0.2614972128  # Meissel-Mertens constant
f_F3 = make_shifted_log(7/6, math.e/phi, -M_MERTENS)
m_F3 = eval_theta(f"Mertens-shifted d=-M={-M_MERTENS:.4f}", f_F3)

# Also with optimized b
def mertens_objective(params):
    a, b = params
    f = lambda lg: a - b / (lg + (-M_MERTENS))
    dp = prime_sum_batched(gamma0, tp, primes, log_primes, K_MAX, f)
    den = np.dot(dp, dp)
    alpha = np.dot(delta, dp) / den if den > 0 else 0.0
    return abs(alpha - 1.0)

res_F3b = optimize.minimize(mertens_objective, x0=[7/6, math.e/phi],
                             method='Nelder-Mead', options={'maxiter': 200})
a_f3b, b_f3b = res_F3b.x
f_F3b = make_shifted_log(a_f3b, b_f3b, -M_MERTENS)
m_F3b = eval_theta(f"Mertens-opt a={a_f3b:.6f} b={b_f3b:.6f}", f_F3b)

# ==============================================================
# GRAND COMPARISON
# ==============================================================
print("\n" + "="*80)
print("GRAND COMPARISON — ALL DIRECTIONS")
print("="*80)

all_results = [
    m_winner,
    best_B, m_Bopt,
    best_C2,
    best_D1, best_D2,
    m_E1, m_E2, m_E3,
    m_F1, m_F2, m_F3, m_F3b,
]

# Sort by |α-1|
all_sorted = sorted(all_results, key=lambda r: r['abs_alpha_m1'])

print(f"\n{'Rk':<4} {'Name':<60} {'α':>10} {'|α-1|':>10} {'drift':>12} {'p':>8} {'T7':>5} {'T8':>5}")
print("-"*115)
for rk, r in enumerate(all_sorted, 1):
    t7 = "PASS" if r['T7_pass'] else "FAIL"
    t8 = "PASS" if r['T8_pass'] else "FAIL"
    print(f"{rk:<4} {r['name'][:58]:<60} {r['alpha']:>+10.7f} {r['abs_alpha_m1']:>10.7f} "
          f"{r['drift_slope']:>+12.8f} {r['drift_p']:>8.4f} {t7:>5} {t8:>5}")

# T8 passers
t8_passers = [r for r in all_sorted if r['T8_pass']]
print(f"\nT8 passers: {len(t8_passers)}")
for r in t8_passers:
    print(f"  {r['name']}")

# Both T7+T8 passers
both = [r for r in all_sorted if r['T7_pass'] and r['T8_pass']]
print(f"\nT7+T8 passers: {len(both)}")
for r in both:
    print(f"  {r['name']}  α={r['alpha']:+.7f}  drift_p={r['drift_p']:.4f}")

# Save
print(f"\nSaving results to {OUTPUT_FILE}...")
save = []
for r in all_sorted:
    sr = {k: v for k, v in r.items() if k != 'window_alphas' and k != 'T_mids'}
    sr['T7_ci'] = list(r.get('T7_ci', (0,0)))
    save.append(sr)

with open(OUTPUT_FILE, 'w') as f:
    json.dump(save, f, indent=2)

print("\nDone!")
