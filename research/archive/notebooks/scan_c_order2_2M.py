#!/usr/bin/env python3
"""
Scan second-order correction c in:
  θ(T) = 7/6 − φ/(logT − 15/8) + c/(logT − 15/8)²

on full 2M zeros.  Fixed a = 7/6, b = φ, d = −15/8.
Scans c to find the value that eliminates the drift (T8).

Uses CuPy GPU if available, falls back to NumPy CPU.
"""
import numpy as np
from scipy.special import loggamma, lambertw
from scipy import stats
import time, json, os, math, sys

# Constants
phi = (1 + math.sqrt(5)) / 2
EULER_GAMMA = 0.5772156649015329
MERTENS = 0.2614972128476428

# GPU detection
os.environ.setdefault('CUDA_PATH', '/usr')
try:
    import cupy as cp
    _ = cp.maximum(cp.array([1.0]), 0.0)
    GPU = True
    mem = cp.cuda.Device(0).mem_info[0] // 1024**2
    print(f"[GPU] CuPy, free={mem} MB")
    PRIME_BATCH = 200 if mem < 6000 else 500
    ZERO_CHUNK = 100_000 if mem < 6000 else 200_000
except Exception as e:
    GPU = False
    PRIME_BATCH = 200
    ZERO_CHUNK = 200_000
    print(f"[CPU] {e}")

# Config
P_MAX = 100_000
K_MAX = 3
N_WINDOWS = 12

# Fixed parameters
A_FIXED = 7/6           # dim_K7 / (2 * N_gen)
B_FIXED = phi           # Golden ratio
D_FIXED = -15/8         # -(dim_G2 + 1) / rank_E8

# Scan range for c
C_COARSE = np.arange(-1.0, 1.05, 0.1)  # 21 points

# ================================================================
# INFRASTRUCTURE
# ================================================================
def theta_rs(t):
    t = np.asarray(t, dtype=np.float64)
    return np.imag(loggamma(0.25 + 0.5j * t)) - 0.5 * t * np.log(np.pi)

def theta_deriv(t):
    return 0.5 * np.log(np.maximum(np.asarray(t, dtype=np.float64), 1.0) / (2 * np.pi))

def smooth_zeros(N):
    ns = np.arange(1, N + 1, dtype=np.float64)
    targets = (ns - 1.5) * np.pi
    w = np.real(lambertw(ns / np.e))
    t = np.maximum(2 * np.pi * ns / w, 2.0)
    for it in range(40):
        dt = (theta_rs(t) - targets) / np.maximum(np.abs(theta_deriv(t)), 1e-15)
        t -= dt
        if np.max(np.abs(dt)) < 1e-12:
            break
    return t

def sieve(N):
    is_p = np.ones(N + 1, dtype=bool); is_p[:2] = False
    for i in range(2, int(N**0.5) + 1):
        if is_p[i]: is_p[i*i::i] = False
    return np.where(is_p)[0]

def prime_sum_order2(g0, tp_v, primes, k_max, a, b, d_shift, c_order2):
    """Order-2 shifted-log: theta = a - b/(logT + d) + c/(logT + d)^2"""
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

        denom = log_g0 + xp.float64(d_shift)
        denom = xp.maximum(denom, 0.1)
        # Order-2: theta = a - b/denom + c/denom^2
        theta_per = xp.float64(a) - xp.float64(b) / denom + xp.float64(c_order2) / (denom * denom)
        theta_per = xp.clip(theta_per, 0.5, 2.0)
        log_X = theta_per * log_g0

        S = xp.zeros(hi - lo, dtype=xp.float64)
        log_X_max = float(xp.max(log_X))

        for m in range(1, k_max + 1):
            cutoff = log_X_max / m
            j_max = int(np.searchsorted(log_primes_np, cutoff + 0.1))
            if j_max == 0: continue

            for b_start in range(0, j_max, PRIME_BATCH):
                b_end = min(b_start + PRIME_BATCH, j_max)
                logp_b = xp.asarray(log_primes_np[b_start:b_end], dtype=xp.float64)
                p_b = xp.asarray(primes[b_start:b_end].astype(np.float64))

                x = (xp.float64(m) * logp_b[:, None]) / log_X[None, :]
                w = xp.where(x < 1.0, xp.cos(xp.float64(math.pi / 2) * x)**2, xp.float64(0))
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

def compute_metrics(delta, delta_pred, gamma0, n_windows):
    denom = np.dot(delta_pred, delta_pred)
    alpha = float(np.dot(delta, delta_pred) / denom) if denom > 0 else 0.0

    N = len(delta)
    bounds = [int(i * N / n_windows) for i in range(n_windows + 1)]
    alphas = []
    T_mids = []
    for i in range(n_windows):
        lo, hi = bounds[i], bounds[i + 1]
        d_w = delta[lo:hi]; dp_w = delta_pred[lo:hi]
        den = np.dot(dp_w, dp_w)
        a_w = float(np.dot(d_w, dp_w) / den) if den > 0 else 0.0
        alphas.append(a_w)
        T_mids.append(float(gamma0[(lo + hi) // 2]))

    x = np.arange(len(alphas))
    slope, _, _, p, _ = stats.linregress(x, alphas)

    rng = np.random.default_rng(42)
    boot = np.empty(500)
    for i in range(500):
        idx = rng.integers(0, N, size=N)
        d = delta[idx]; dp = delta_pred[idx]
        den = np.dot(dp, dp)
        boot[i] = np.dot(d, dp) / den if den > 0 else 0.0
    ci_lo = float(np.percentile(boot, 2.5))
    ci_hi = float(np.percentile(boot, 97.5))
    t7 = bool(ci_lo <= 1.0 <= ci_hi)

    return {
        'alpha': alpha,
        'abs_alpha_minus_1': abs(alpha - 1),
        'drift_slope': float(slope),
        'drift_p': float(p),
        'T7_pass': t7,
        'T8_pass': bool(p > 0.05),
        'ci_lo': ci_lo,
        'ci_hi': ci_hi,
        'window_alphas': alphas,
        'T_mids': T_mids,
    }

# ================================================================
# MAIN
# ================================================================
print("Loading zeros...")
ZEROS_FILE = 'outputs/riemann_zeros_2M_genuine.npy'
gamma_n = np.load(ZEROS_FILE)
N_TOTAL = len(gamma_n)
print(f"  {N_TOTAL:,} zeros loaded")

print("Computing smooth zeros...")
t0 = time.time()
gamma0 = smooth_zeros(N_TOTAL)
delta = gamma_n - gamma0
tp = theta_deriv(gamma0)
print(f"  Done in {time.time()-t0:.1f}s")

print("Sieving primes...")
primes = sieve(P_MAX)
print(f"  {len(primes):,} primes up to {P_MAX:,}")

# ================================================================
# BASELINE: d = -15/8, c = 0
# ================================================================
print(f"\n{'='*70}")
print(f"BASELINE: theta = 7/6 - phi/(logT - 15/8)")
print(f"  a = {A_FIXED:.6f} (7/6)")
print(f"  b = {B_FIXED:.6f} (phi)")
print(f"  d = {D_FIXED:.6f} (-15/8)")
print(f"{'='*70}")

t0 = time.time()
dp_base = prime_sum_order2(gamma0, tp, primes, K_MAX, A_FIXED, B_FIXED, D_FIXED, 0.0)
t_one = time.time() - t0
m_base = compute_metrics(delta, dp_base, gamma0, N_WINDOWS)
m_base['c_order2'] = 0.0
m_base['d_shift'] = D_FIXED

t7 = "PASS" if m_base['T7_pass'] else "FAIL"
t8 = "PASS" if m_base['T8_pass'] else "FAIL"
print(f"  alpha = {m_base['alpha']:+.8f}")
print(f"  |alpha-1| = {m_base['abs_alpha_minus_1']:.8f}")
print(f"  drift = {m_base['drift_slope']:+.8f} (p={m_base['drift_p']:.4f})")
print(f"  T7: {t7}  T8: {t8}")
print(f"  CI: [{m_base['ci_lo']:.6f}, {m_base['ci_hi']:.6f}]")
print(f"  Window alphas: {['%.4f' % a for a in m_base['window_alphas']]}")
print(f"  [{t_one:.1f}s per eval]")
print(f"  Estimated coarse scan ({len(C_COARSE)} pts): {t_one * len(C_COARSE) / 60:.1f} min")
sys.stdout.flush()

# ================================================================
# COARSE SCAN of c
# ================================================================
print(f"\n{'='*70}")
print(f"COARSE SCAN: c in [{C_COARSE[0]:.1f}, {C_COARSE[-1]:.1f}], {len(C_COARSE)} points")
print(f"  theta = 7/6 - phi/(logT - 15/8) + c/(logT - 15/8)^2")
print(f"{'='*70}")

results = [m_base]  # include baseline
t_scan = time.time()

for i, c in enumerate(C_COARSE):
    t0 = time.time()
    dp = prime_sum_order2(gamma0, tp, primes, K_MAX, A_FIXED, B_FIXED, D_FIXED, c)
    m = compute_metrics(delta, dp, gamma0, N_WINDOWS)
    m['c_order2'] = float(c)
    m['d_shift'] = D_FIXED
    m['a'] = A_FIXED
    m['b'] = B_FIXED
    elapsed = time.time() - t0

    t7 = "PASS" if m['T7_pass'] else "FAIL"
    t8 = "PASS" if m['T8_pass'] else "FAIL"
    print(f"  [{i+1:2d}/{len(C_COARSE)}] c={c:+.2f}  alpha={m['alpha']:+.6f}  |a-1|={m['abs_alpha_minus_1']:.6f}  "
          f"drift={m['drift_slope']:+.8f} (p={m['drift_p']:.4f})  T7={t7}  T8={t8}  [{elapsed:.1f}s]", flush=True)
    results.append(m)

total = time.time() - t_scan
print(f"\nCoarse scan done in {total/60:.1f} min")

# Find best by drift_p (highest = most stationary)
best_drift = max(results, key=lambda r: r['drift_p'])
# Find best T7+T8 passer
t7t8_passers = [r for r in results if r['T7_pass'] and r['T8_pass']]
# Find best |alpha-1|
best_alpha = min(results, key=lambda r: r['abs_alpha_minus_1'])

print(f"\nBest |alpha-1|: c={best_alpha['c_order2']:.2f}  alpha={best_alpha['alpha']:+.6f}  T7={'PASS' if best_alpha['T7_pass'] else 'FAIL'}  T8={'PASS' if best_alpha['T8_pass'] else 'FAIL'}")
print(f"Best drift_p: c={best_drift['c_order2']:.2f}  drift_p={best_drift['drift_p']:.4f}  alpha={best_drift['alpha']:+.6f}  T7={'PASS' if best_drift['T7_pass'] else 'FAIL'}  T8={'PASS' if best_drift['T8_pass'] else 'FAIL'}")
if t7t8_passers:
    best_both = min(t7t8_passers, key=lambda r: r['abs_alpha_minus_1'])
    print(f"Best T7+T8: c={best_both['c_order2']:.2f}  alpha={best_both['alpha']:+.6f}  drift_p={best_both['drift_p']:.4f}")

# ================================================================
# FINE SCAN around best drift_p (where drift vanishes)
# ================================================================
c_center = best_drift['c_order2']
C_FINE = np.arange(c_center - 0.3, c_center + 0.32, 0.02)  # step 0.02

print(f"\n{'='*70}")
print(f"FINE SCAN: c in [{C_FINE[0]:.2f}, {C_FINE[-1]:.2f}], {len(C_FINE)} points")
print(f"{'='*70}")

fine_results = []
t_fine = time.time()

for i, c in enumerate(C_FINE):
    t0 = time.time()
    dp = prime_sum_order2(gamma0, tp, primes, K_MAX, A_FIXED, B_FIXED, D_FIXED, c)
    m = compute_metrics(delta, dp, gamma0, N_WINDOWS)
    m['c_order2'] = float(c)
    m['d_shift'] = D_FIXED
    m['a'] = A_FIXED
    m['b'] = B_FIXED
    elapsed = time.time() - t0

    t7 = "PASS" if m['T7_pass'] else "FAIL"
    t8 = "PASS" if m['T8_pass'] else "FAIL"
    print(f"  [{i+1:2d}/{len(C_FINE)}] c={c:+.4f}  alpha={m['alpha']:+.6f}  |a-1|={m['abs_alpha_minus_1']:.6f}  "
          f"drift={m['drift_slope']:+.8f} (p={m['drift_p']:.4f})  T7={t7}  T8={t8}  [{elapsed:.1f}s]", flush=True)
    fine_results.append(m)

total_fine = time.time() - t_fine
print(f"\nFine scan done in {total_fine/60:.1f} min")

all_results = results + fine_results

# ================================================================
# FINAL ANALYSIS
# ================================================================
best = min(all_results, key=lambda r: r['abs_alpha_minus_1'])
best_drift_all = max(all_results, key=lambda r: r['drift_p'])
t7t8_all = [r for r in all_results if r['T7_pass'] and r['T8_pass']]

print(f"\n{'='*70}")
print("FINAL RESULTS")
print(f"{'='*70}")

print(f"\nFormula: theta(T) = 7/6 - phi/(logT - 15/8) + c/(logT - 15/8)^2")
print(f"  a = 7/6 = {A_FIXED:.6f}")
print(f"  b = phi = {B_FIXED:.6f}")
print(f"  d = -15/8 = {D_FIXED:.6f}")

print(f"\nBest |alpha-1|:")
print(f"  c = {best['c_order2']:.4f}")
print(f"  alpha = {best['alpha']:+.8f}")
print(f"  |alpha-1| = {best['abs_alpha_minus_1']:.8f}")
print(f"  drift = {best['drift_slope']:+.8f} (p={best['drift_p']:.4f})")
print(f"  T7: {'PASS' if best['T7_pass'] else 'FAIL'}  T8: {'PASS' if best['T8_pass'] else 'FAIL'}")
print(f"  CI: [{best['ci_lo']:.6f}, {best['ci_hi']:.6f}]")

print(f"\nBest drift elimination (max drift_p):")
print(f"  c = {best_drift_all['c_order2']:.4f}")
print(f"  alpha = {best_drift_all['alpha']:+.8f}")
print(f"  |alpha-1| = {best_drift_all['abs_alpha_minus_1']:.8f}")
print(f"  drift = {best_drift_all['drift_slope']:+.8f} (p={best_drift_all['drift_p']:.4f})")
print(f"  T7: {'PASS' if best_drift_all['T7_pass'] else 'FAIL'}  T8: {'PASS' if best_drift_all['T8_pass'] else 'FAIL'}")

if t7t8_all:
    best_both = min(t7t8_all, key=lambda r: r['abs_alpha_minus_1'])
    print(f"\nBest T7+T8 PASS:")
    print(f"  c = {best_both['c_order2']:.4f}")
    print(f"  alpha = {best_both['alpha']:+.8f}")
    print(f"  |alpha-1| = {best_both['abs_alpha_minus_1']:.8f}")
    print(f"  drift_p = {best_both['drift_p']:.4f}")
    print(f"  CI: [{best_both['ci_lo']:.6f}, {best_both['ci_hi']:.6f}]")
    print(f"  Window alphas: {['%.4f' % a for a in best_both['window_alphas']]}")
else:
    print(f"\n  ** NO c value passes both T7 and T8 **")
    best_both = None

# Symbolic c candidates
print(f"\nSymbolic c candidates:")
print(f"  phi^2/14 = {phi**2/14:.6f}")
print(f"  1/rank_E8 = {1/8:.6f}")
print(f"  phi/8 = {phi/8:.6f}")
print(f"  1/7 = {1/7:.6f}")
print(f"  phi^2/8 = {phi**2/8:.6f}")
print(f"  3/8 = {3/8:.6f}")
print(f"  phi/3 = {phi/3:.6f}")
print(f"  phi^2/3 = {phi**2/3:.6f}")

# Save
out = {
    'config': {
        'a': A_FIXED, 'b': B_FIXED, 'd': D_FIXED,
        'P_MAX': P_MAX, 'K_MAX': K_MAX,
        'N_zeros': N_TOTAL, 'N_primes': len(primes),
        'formula': 'theta = 7/6 - phi/(logT - 15/8) + c/(logT - 15/8)^2',
    },
    'baseline': {k: v for k, v in m_base.items()},
    'coarse': [{k: v for k, v in r.items() if k not in ('window_alphas', 'T_mids')} for r in results],
    'fine': [{k: v for k, v in r.items() if k not in ('window_alphas', 'T_mids')} for r in fine_results],
    'best_alpha': {k: v for k, v in best.items()},
    'best_drift': {k: v for k, v in best_drift_all.items()},
    'best_t7t8': {k: v for k, v in best_both.items()} if best_both else None,
}
out_path = 'outputs/scan_c_order2_2M_results.json'
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2, default=float)
print(f"\nSaved to {out_path}")
