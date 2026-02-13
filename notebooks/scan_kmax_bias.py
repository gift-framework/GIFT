#!/usr/bin/env python3
"""
TOP-DOWN QUESTION: Can GIFT predict the residual bias |α-1| ≈ 0.0003?
=====================================================================

Hypothesis: The bias depends on K_MAX (number of mollifier harmonics).
We use K_MAX = 3 = N_gen. Scan K_MAX = 1..10 to find the functional form.

Also test:
  - Does the bias scale with a GIFT topological expression?
  - Does the oscillation period (Piste 4) depend on K_MAX?

Uses genuine Odlyzko zeros (2M), P_MAX=50k for speed.
"""
import numpy as np
from scipy.special import loggamma
from scipy import stats
import time, json, os, math, sys

phi = (1 + math.sqrt(5)) / 2

# GIFT constants
DIM_G2 = 14; DIM_K7 = 7; H_STAR = 99; B2 = 21; B3 = 77
RANK_E8 = 8; P2 = 2; N_GEN = 3; DIM_E8 = 248; WEYL = 5
D_BULK = 11; DIM_J3O = 27; DIM_E8xE8 = 496
TAU_GIFT = 3472 / 891

os.environ.setdefault('CUDA_PATH', '/usr')
try:
    import cupy as cp
    _ = cp.array([1.0])
    GPU = True
    mem = cp.cuda.Device(0).mem_info[0] // 1024**2
    print(f"[GPU] free={mem} MB")
    PRIME_BATCH = 500
    ZERO_CHUNK = 80_000
except Exception as e:
    GPU = False
    PRIME_BATCH = 200
    ZERO_CHUNK = 200_000
    print(f"[CPU] {e}")


def theta_rs(t):
    t = np.asarray(t, dtype=np.float64)
    return np.imag(loggamma(0.25 + 0.5j * t)) - 0.5 * t * np.log(np.pi)

def theta_deriv(t):
    return 0.5 * np.log(np.maximum(np.asarray(t, dtype=np.float64), 1.0) / (2 * np.pi))

def sieve(N):
    is_p = np.ones(N + 1, dtype=bool); is_p[:2] = False
    for i in range(2, int(N**0.5) + 1):
        if is_p[i]: is_p[i*i::i] = False
    return np.where(is_p)[0]


def prime_sum(g0, tp_v, primes, k_max, a, b, d_shift):
    xp = cp if GPU else np
    N = len(g0)
    result = np.zeros(N, dtype=np.float64)
    log_primes_np = np.log(primes.astype(np.float64))

    for ic in range((N + ZERO_CHUNK - 1) // ZERO_CHUNK):
        lo = ic * ZERO_CHUNK
        hi = min(lo + ZERO_CHUNK, N)
        g0_c = xp.asarray(g0[lo:hi], dtype=xp.float64)
        tp_c = xp.asarray(tp_v[lo:hi], dtype=xp.float64)
        log_g0 = xp.log(xp.maximum(g0_c, 2.0))

        if abs(b) < 1e-15:
            theta_per = xp.full(hi - lo, xp.float64(a))
        else:
            denom = log_g0 + xp.float64(d_shift)
            denom = xp.maximum(denom, 0.1)
            theta_per = xp.float64(a) - xp.float64(b) / denom
        theta_per = xp.clip(theta_per, 0.5, 2.0)
        log_X = theta_per * log_g0

        S = xp.zeros(hi - lo, dtype=xp.float64)
        log_X_max = float(xp.max(log_X))

        for m in range(1, k_max + 1):
            cutoff = log_X_max / m
            j_max = int(np.searchsorted(log_primes_np, cutoff + 0.1))
            if j_max == 0:
                continue
            for b_start in range(0, j_max, PRIME_BATCH):
                b_end = min(b_start + PRIME_BATCH, j_max)
                logp_b = xp.asarray(log_primes_np[b_start:b_end], dtype=xp.float64)
                p_b = xp.asarray(primes[b_start:b_end].astype(np.float64))
                x = (xp.float64(m) * logp_b[:, None]) / log_X[None, :]
                w = xp.where(x < 1.0,
                             xp.cos(xp.float64(math.pi / 2) * x)**2,
                             xp.float64(0))
                phase = g0_c[None, :] * (xp.float64(m) * logp_b[:, None])
                coeff = xp.float64(1.0 / m) / p_b ** (m / 2.0)
                S -= xp.sum(w * xp.sin(phase) * coeff[:, None], axis=0)
                del x, w, phase, coeff

        chunk_result = -S / tp_c
        if GPU:
            result[lo:hi] = cp.asnumpy(chunk_result)
            cp.get_default_memory_pool().free_all_blocks()
        else:
            result[lo:hi] = chunk_result
    return result


def eval_kmax(gamma0, tp_v, delta_pred, primes, k_max, a, b, d_shift, n_win=50):
    """Evaluate alpha and window data for a given K_MAX."""
    S_w = prime_sum(gamma0, tp_v, primes, k_max, a, b, d_shift)
    delta = delta_pred + S_w
    N = len(gamma0)

    # Global alpha (OLS)
    denom = np.dot(delta_pred, delta_pred)
    alpha_g = float(np.dot(delta, delta_pred) / denom)

    # Window alphas
    bounds = [int(i * N / n_win) for i in range(n_win + 1)]
    alphas = np.empty(n_win)
    T_mids = np.empty(n_win)
    for i in range(n_win):
        lo, hi = bounds[i], bounds[i+1]
        dp = delta_pred[lo:hi]
        d = delta[lo:hi]
        den = np.dot(dp, dp)
        alphas[i] = float(np.dot(d, dp) / den) if den > 0 else 0.0
        T_mids[i] = float(gamma0[(lo + hi) // 2])

    # T7: Bootstrap CI
    rng = np.random.default_rng(42)
    boot_alpha = np.array([np.mean(alphas[rng.integers(0, n_win, size=n_win)])
                           for _ in range(500)])
    ci = (float(np.percentile(boot_alpha, 2.5)), float(np.percentile(boot_alpha, 97.5)))
    t7 = bool(ci[0] <= 1.0 <= ci[1])

    # T8: Drift (linear regression on 12 windows)
    n_drift = 12
    bounds12 = [int(i * N / n_drift) for i in range(n_drift + 1)]
    alphas12 = np.empty(n_drift)
    for i in range(n_drift):
        lo2, hi2 = bounds12[i], bounds12[i+1]
        dp = delta_pred[lo2:hi2]
        d2 = delta[lo2:hi2]
        den = np.dot(dp, dp)
        alphas12[i] = float(np.dot(d2, dp) / den) if den > 0 else 0.0
    x_12 = np.arange(n_drift, dtype=np.float64)
    slope, _, _, p_drift, _ = stats.linregress(x_12, alphas12)
    t8 = bool(p_drift > 0.05)

    # Fourier analysis on 50-window alphas (for oscillation period)
    alphas_centered = alphas - np.mean(alphas)
    fft = np.fft.rfft(alphas_centered)
    power = np.abs(fft)**2
    freqs = np.fft.rfftfreq(n_win)
    # Skip DC (idx 0)
    if len(power) > 1:
        peak_idx = int(np.argmax(power[1:])) + 1
        peak_period = float(1.0 / freqs[peak_idx]) if freqs[peak_idx] > 0 else float('inf')
        peak_power_frac = float(power[peak_idx] / np.sum(power[1:]))
    else:
        peak_period = float('inf')
        peak_power_frac = 0.0

    return {
        'k_max': k_max,
        'alpha': alpha_g,
        'abs_alpha_m1': abs(alpha_g - 1.0),
        't7': t7,
        't7_ci': list(ci),
        't8': t8,
        'drift_p': float(p_drift),
        'drift_slope': float(slope),
        'alpha_std': float(np.std(alphas)),
        'alpha_range': [float(np.min(alphas)), float(np.max(alphas))],
        'osc_period': peak_period,
        'osc_power_frac': peak_power_frac,
    }


# ================================================================
# MAIN
# ================================================================
print()
print("=" * 72)
print("  Loading genuine Odlyzko zeros")
print("=" * 72)
zeros_path = "outputs/odlyzko_zeros_2M.npy"
gamma0 = np.load(zeros_path)
print(f"  {len(gamma0):,} genuine zeros, range [{gamma0[0]:.2f}, {gamma0[-1]:.2f}]")

# Precompute (MUST match explore_genuine_zeros.py line 197)
tp_v = theta_deriv(gamma0)
delta_pred = np.sign(np.sin(np.pi * (theta_rs(gamma0) / np.pi - 0.5)))

# Primes
P_MAX = 50_000
primes = sieve(P_MAX)
print(f"  {len(primes):,} primes up to {P_MAX:,}")

# ================================================================
# Test formulas
# ================================================================
formulas = [
    ("const_1.0", 1.0, 0.0, 0.0),                  # Constant theta = 1
    ("best: 1.0 - phi/(logT - 15/8)", 1.0, phi, -15/8),  # Best from exploration
    ("GIFT: 7/6 - phi/(logT - 15/8)", 7/6, phi, -15/8),  # GIFT formula
]

K_MAX_VALUES = [1, 2, 3, 4, 5, 6, 7, 8]

print()
print("=" * 72)
print(f"  K_MAX SCAN: Testing {len(K_MAX_VALUES)} values × {len(formulas)} formulas")
print("=" * 72)
print(f"  K_MAX values: {K_MAX_VALUES}")
print(f"  P_MAX = {P_MAX:,}")
print()

all_results = []
t0_global = time.time()

for fi, (fname, a, b, d) in enumerate(formulas):
    print(f"\n  Formula {fi+1}/{len(formulas)}: {fname}")
    print(f"  {'K':>4}  {'|α-1|':>12}  {'α':>12}  {'T7':>3} {'T8':>3}  {'drift_p':>8}  {'osc_T':>7}  {'time':>5}")
    print(f"  {'─'*4}  {'─'*12}  {'─'*12}  {'─'*3} {'─'*3}  {'─'*8}  {'─'*7}  {'─'*5}")

    for K in K_MAX_VALUES:
        t0 = time.time()
        res = eval_kmax(gamma0, tp_v, delta_pred, primes, K, a, b, d)
        res['formula'] = fname
        res['a'] = a
        res['b'] = b
        res['d'] = d
        elapsed = time.time() - t0

        t7s = "Y" if res['t7'] else "N"
        t8s = "Y" if res['t8'] else "N"
        print(f"  {K:>4}  {res['abs_alpha_m1']:>12.7f}  {res['alpha']:>12.7f}  {t7s:>3} {t8s:>3}  "
              f"{res['drift_p']:>8.4f}  {res['osc_period']:>7.1f}  {elapsed:>4.0f}s")
        sys.stdout.flush()
        all_results.append(res)

elapsed_total = time.time() - t0_global

# ================================================================
# ANALYSIS: Fit |α-1| vs K_MAX
# ================================================================
print()
print("=" * 72)
print("  ANALYSIS")
print("=" * 72)

for fi, (fname, a, b, d) in enumerate(formulas):
    print(f"\n  Formula: {fname}")
    subset = [r for r in all_results if r['formula'] == fname]
    Ks = np.array([r['k_max'] for r in subset], dtype=np.float64)
    biases = np.array([r['abs_alpha_m1'] for r in subset])

    # Power law fit: |α-1| = c * K^(-p)
    log_K = np.log(Ks)
    log_bias = np.log(biases)
    slope_pw, intercept_pw, r_pw, _, _ = stats.linregress(log_K, log_bias)
    c_pw = np.exp(intercept_pw)
    print(f"  Power law: |α-1| ≈ {c_pw:.6f} × K^({slope_pw:.3f})   R²={r_pw**2:.4f}")

    # Exponential fit: |α-1| = c * exp(-λ*K)
    slope_exp, intercept_exp, r_exp, _, _ = stats.linregress(Ks, log_bias)
    c_exp = np.exp(intercept_exp)
    print(f"  Exponential: |α-1| ≈ {c_exp:.6f} × exp({slope_exp:.4f}·K)   R²={r_exp**2:.4f}")

    # 1/K fit
    slope_inv, intercept_inv, r_inv, _, _ = stats.linregress(1.0/Ks, biases)
    print(f"  Linear 1/K: |α-1| ≈ {intercept_inv:.6f} + {slope_inv:.6f}/K   R²={r_inv**2:.4f}")

    # Check GIFT expressions
    print()
    print(f"  At K=3 (=N_gen): |α-1| = {biases[2]:.7f}")
    print(f"  GIFT candidates for |α-1|:")
    candidates_gift = [
        ("1/(dim_G2 × dim_E8)",   1/(DIM_G2 * DIM_E8)),       # = 1/3472
        ("1/(dim_E8xE8 × dim_K7)", 1/(DIM_E8xE8 * DIM_K7)),   # = 1/3472 (same!)
        ("1/(b3 × dim_G2 × N_gen)", 1/(B3 * DIM_G2 * N_GEN)),  # = 1/3234
        ("N_gen/(dim_E8xE8 × b2)", N_GEN/(DIM_E8xE8 * B2)),    # = 3/10416
        ("1/(H_star × dim_K7 × Weyl)", 1/(H_STAR * DIM_K7 * WEYL)),  # = 1/3465
        ("phi/(dim_E8xE8 × dim_K7)", phi/(DIM_E8xE8 * DIM_K7)),     # = φ/3472
        ("rank_E8/(dim_E8 × D_bulk)", RANK_E8/(DIM_E8 * D_BULK)),    # = 8/2728
        ("1/(tau_GIFT × H_star × rank_E8)", 1/(TAU_GIFT * H_STAR * RANK_E8)),
        ("Weyl/(dim_E8 × RANK_E8)", WEYL/(DIM_E8 * RANK_E8)),
    ]
    for name, val in sorted(candidates_gift, key=lambda x: abs(x[1] - biases[2])):
        ratio = biases[2] / val if val > 0 else float('inf')
        pct = abs(ratio - 1) * 100
        print(f"    {name:<40s} = {val:.7f}  ratio={ratio:.4f} ({pct:.1f}% off)")

    # Also check if |α-1| at K=N_gen has a K-dependent GIFT formula
    if abs(slope_pw) > 0.1:  # If power law is decent
        print()
        print(f"  Power law exponent: {slope_pw:.3f}")
        print(f"  If exponent = -1: |α-1|(K=3) predicted = {c_pw / 3:.7f} (actual: {biases[2]:.7f})")
        print(f"  If exponent = -2: |α-1|(K=3) predicted = {c_pw / 9:.7f} (actual: {biases[2]:.7f})")

# Oscillation period analysis
print()
print("=" * 72)
print("  OSCILLATION PERIOD vs K_MAX")
print("=" * 72)
for fi, (fname, a, b, d) in enumerate(formulas):
    subset = [r for r in all_results if r['formula'] == fname]
    print(f"\n  {fname}:")
    print(f"  {'K':>4}  {'period':>8}  {'power%':>8}")
    for r in subset:
        print(f"  {r['k_max']:>4}  {r['osc_period']:>8.1f}  {r['osc_power_frac']*100:>7.1f}%")

# Save
print()
print("=" * 72)
print(f"  Total runtime: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
print("=" * 72)

outpath = "outputs/kmax_scan_results.json"
with open(outpath, 'w') as f:
    json.dump({
        'p_max': P_MAX,
        'n_zeros': len(gamma0),
        'k_max_values': K_MAX_VALUES,
        'formulas': [fname for fname, _, _, _ in formulas],
        'elapsed_s': elapsed_total,
        'results': all_results,
    }, f, indent=2)
print(f"  Saved to {outpath}")
print()
print("  DONE")
