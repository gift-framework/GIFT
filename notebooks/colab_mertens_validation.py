#!/usr/bin/env python3
"""
Colab validation of Piste 6 Mertens predictions.

Tests the formula:
  theta(T) = 7/6 - phi/(logT - 15/8) + c/(logT - 15/8)^2

for specific Mertens-derived c values against 2M zeros.

Candidates:
  c = 3M          = 0.7844916385  (N_gen * Mertens)
  c = 7/9         = 0.7777777778  (dim_K7 / (rank_E8 + 1))
  c = 11/14       = 0.7857142857  (D_bulk / dim_G2)
  c = pi/4        = 0.7853981634  (transcendental)
  c = 21/27       = 0.7777777778  (b2 / dim_J3O)
  c = 0.78        = empirical baseline

Upload to Colab, run:
  !python colab_mertens_validation.py

Expects: outputs/riemann_zeros_2M_genuine.npy
"""
import numpy as np
from scipy.special import loggamma, lambertw
from scipy import stats
import time, json, os, math, sys

# ================================================================
# CONSTANTS
# ================================================================
phi = (1 + math.sqrt(5)) / 2
EULER_GAMMA = 0.5772156649015329
MERTENS = 0.2614972128476428

# GIFT constants
DIM_G2 = 14; DIM_K7 = 7; H_STAR = 99; B2 = 21; B3 = 77
RANK_E8 = 8; P2 = 2; N_GEN = 3; DIM_E8 = 248; WEYL = 5
D_BULK = 11; DIM_J3O = 27

# GPU detection
os.environ.setdefault('CUDA_PATH', '/usr')
try:
    import cupy as cp
    _ = cp.maximum(cp.array([1.0]), 0.0)
    GPU = True
    mem = cp.cuda.Device(0).mem_info[0] // 1024**2
    print(f"[GPU] CuPy, free={mem} MB")
    PRIME_BATCH = 500 if mem > 6000 else 200
    ZERO_CHUNK = 500_000 if mem > 20000 else 200_000
except Exception as e:
    GPU = False
    PRIME_BATCH = 200
    ZERO_CHUNK = 200_000
    print(f"[CPU] {e}")

# ================================================================
# CONFIG
# ================================================================
A_FIXED = 7.0 / 6.0
B_FIXED = phi
D_FIXED = -15.0 / 8.0
P_MAX = 500_000
K_MAX = 3
N_WINDOWS = 12

# Candidate c values (order-2 coefficient)
CANDIDATES = {
    'c=3M (N_gen*Mertens)':      N_GEN * MERTENS,
    'c=7/9 (dim_K7/(rank_E8+1))': DIM_K7 / (RANK_E8 + 1),
    'c=21/27 (b2/dim_J3O)':     B2 / float(DIM_J3O),
    'c=11/14 (D_bulk/dim_G2)':   D_BULK / float(DIM_G2),
    'c=pi/4':                    math.pi / 4,
    'c=0.78 (empirical)':        0.78,
    'c=15/19 (GIFT_test)':       15.0 / 19.0,
    'c=4/5 (4/Weyl)':           4.0 / WEYL,
    'c=0 (baseline)':            0.0,
}

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
    slope, _, r_value, p, _ = stats.linregress(x, alphas)

    rng = np.random.default_rng(42)
    boot = np.empty(1000)
    for i in range(1000):
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
print("=" * 70)
print("PISTE 6 VALIDATION: Mertens-based c_order2 candidates")
print("  theta(T) = 7/6 - phi/(logT - 15/8) + c/(logT - 15/8)^2")
print("=" * 70)
print(f"  a     = 7/6 = {A_FIXED:.6f}")
print(f"  b     = phi = {B_FIXED:.6f}")
print(f"  d     = -15/8 = {D_FIXED:.6f}")
print(f"  P_MAX = {P_MAX:,}")
print(f"  k_max = {K_MAX}")
print(f"  Mertens M = {MERTENS:.10f}")
print(f"  3M        = {3*MERTENS:.10f}")

# Load zeros
ZEROS_FILE = 'outputs/riemann_zeros_2M_genuine.npy'
print(f"\nLoading zeros from {ZEROS_FILE}...")
gamma_n = np.load(ZEROS_FILE)
N_TOTAL = len(gamma_n)
print(f"  {N_TOTAL:,} zeros loaded")

print("Computing smooth zeros...")
t0 = time.time()
gamma0 = smooth_zeros(N_TOTAL)
delta = gamma_n - gamma0
tp = theta_deriv(gamma0)
print(f"  Done in {time.time()-t0:.1f}s")

primes = sieve(P_MAX)
n_primes = len(primes)
print(f"  {n_primes:,} primes (P_MAX={P_MAX:,})")

# ================================================================
# RUN ALL CANDIDATES
# ================================================================
print(f"\n{'='*70}")
print(f"TESTING {len(CANDIDATES)} CANDIDATES")
print(f"{'='*70}\n")

all_results = {}
total_start = time.time()

for name, c_val in CANDIDATES.items():
    t0 = time.time()
    dp = prime_sum_order2(gamma0, tp, primes, K_MAX, A_FIXED, B_FIXED, D_FIXED, c_val)
    elapsed = time.time() - t0
    m = compute_metrics(delta, dp, gamma0, N_WINDOWS)

    t7 = "PASS" if m['T7_pass'] else "FAIL"
    t8 = "PASS" if m['T8_pass'] else "FAIL"
    marker = " <<<" if m['T7_pass'] and m['T8_pass'] else ""

    print(f"  {name}")
    print(f"    c = {c_val:.10f}")
    print(f"    alpha = {m['alpha']:+.10f}  |a-1| = {m['abs_alpha_minus_1']:.8f}")
    print(f"    drift = {m['drift_slope']:+.10f}  p = {m['drift_p']:.6f}")
    print(f"    T7: {t7}  T8: {t8}  CI: [{m['ci_lo']:.6f}, {m['ci_hi']:.6f}]  [{elapsed:.1f}s]{marker}")
    print(f"    Window alphas: {['%.6f' % a for a in m['window_alphas']]}")
    print()

    all_results[name] = {
        'c': float(c_val),
        **{k: v for k, v in m.items()},
    }
    sys.stdout.flush()

total_time = time.time() - total_start
print(f"\nTotal time: {total_time/60:.1f} min")

# ================================================================
# RANKING
# ================================================================
print(f"\n{'='*70}")
print("RANKING BY |alpha-1|")
print(f"{'='*70}")

ranked = sorted(all_results.items(), key=lambda x: x[1]['abs_alpha_minus_1'])
for name, r in ranked:
    t7 = "P" if r['T7_pass'] else "F"
    t8 = "P" if r['T8_pass'] else "F"
    marker = " <-- BOTH" if r['T7_pass'] and r['T8_pass'] else ""
    print(f"  {name:35s}  c={r['c']:.10f}  |a-1|={r['abs_alpha_minus_1']:.8f}  "
          f"drift_p={r['drift_p']:.4f}  T7={t7} T8={t8}{marker}")

print(f"\n{'='*70}")
print("RANKING BY drift_p (higher = less drift)")
print(f"{'='*70}")

ranked_drift = sorted(all_results.items(), key=lambda x: x[1]['drift_p'], reverse=True)
for name, r in ranked_drift:
    t7 = "P" if r['T7_pass'] else "F"
    t8 = "P" if r['T8_pass'] else "F"
    marker = " <-- BOTH" if r['T7_pass'] and r['T8_pass'] else ""
    print(f"  {name:35s}  c={r['c']:.10f}  drift_p={r['drift_p']:.6f}  "
          f"|a-1|={r['abs_alpha_minus_1']:.8f}  T7={t7} T8={t8}{marker}")

# Winners
both = {n: r for n, r in all_results.items() if r['T7_pass'] and r['T8_pass']}
if both:
    print(f"\n*** CANDIDATES PASSING BOTH T7+T8: ***")
    for name, r in sorted(both.items(), key=lambda x: x[1]['abs_alpha_minus_1']):
        print(f"  {name}: c={r['c']:.10f}  alpha={r['alpha']:+.10f}  drift_p={r['drift_p']:.6f}")
else:
    print(f"\n  No candidate passes both T7 and T8.")
    # Show closest to both
    score = [(n, r, r['abs_alpha_minus_1'] + max(0, 0.05 - r['drift_p']))
             for n, r in all_results.items()]
    score.sort(key=lambda x: x[2])
    print(f"  Closest to passing both (composite score = |a-1| + max(0, 0.05-drift_p)):")
    for name, r, sc in score[:3]:
        print(f"    {name}: score={sc:.6f}  |a-1|={r['abs_alpha_minus_1']:.6f}  drift_p={r['drift_p']:.4f}")

# ================================================================
# KEY COMPARISON: 3M vs 7/9 vs 11/14 vs pi/4
# ================================================================
print(f"\n{'='*70}")
print("HEAD-TO-HEAD: Mertens (3M) vs Pure Topology (7/9, 11/14) vs Transcendental (pi/4)")
print(f"{'='*70}")

key_names = ['c=3M (N_gen*Mertens)', 'c=7/9 (dim_K7/(rank_E8+1))',
             'c=11/14 (D_bulk/dim_G2)', 'c=pi/4']
for name in key_names:
    if name in all_results:
        r = all_results[name]
        print(f"\n  {name}:")
        print(f"    c = {r['c']:.10f}")
        print(f"    alpha = {r['alpha']:+.10f}")
        print(f"    |alpha-1| = {r['abs_alpha_minus_1']:.10f}")
        print(f"    drift_p = {r['drift_p']:.6f}")
        print(f"    T7: {'PASS' if r['T7_pass'] else 'FAIL'}  T8: {'PASS' if r['T8_pass'] else 'FAIL'}")

# ================================================================
# SAVE
# ================================================================
out = {
    'config': {
        'formula': 'theta = 7/6 - phi/(logT - 15/8) + c/(logT - 15/8)^2',
        'a': A_FIXED, 'b': B_FIXED, 'd': D_FIXED,
        'N_zeros': N_TOTAL, 'P_MAX': P_MAX,
        'K_MAX': K_MAX, 'N_WINDOWS': N_WINDOWS,
        'Mertens': MERTENS,
    },
    'candidates': all_results,
}
out_path = 'outputs/mertens_validation_results.json'
os.makedirs('outputs', exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2, default=float)
print(f"\nSaved to {out_path}")
print("\nDone!")
