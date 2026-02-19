#!/usr/bin/env python3
"""
Colab A100 validation: θ(T) = 7/6 − φ/(logT − 15/8) on 2M zeros.
P_MAX = 500k primes (vs 100k local). Tests if drift is a truncation artifact.

Upload to Colab, mount Drive, run:
  !python colab_validate_gift_formula_500k.py

Expects: outputs/riemann_zeros_2M_genuine.npy (or generates from scratch).
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
# CONFIG — the key parameters to test
# ================================================================
# GIFT formula: θ(T) = 7/6 − φ/(logT − 15/8)
A = 7/6                # dim_K7 / (2 * N_gen)
B = phi                # Golden ratio
D = -15/8              # -(dim_G2 + 1) / rank_E8

# Prime limits to compare
P_MAX_LIST = [100_000, 500_000]
K_MAX = 3
N_WINDOWS = 12

# Scan d around -15/8 with P_MAX=500k
D_SCAN = np.arange(-2.10, -1.60, 0.02)  # 25 points around -1.875

# Scan k_max at d=-15/8, P_MAX=500k
K_SCAN = [3, 4, 5]

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

def prime_sum_shifted(g0, tp_v, primes, k_max, a, b, d_shift):
    """Shifted-log: theta = a - b/(logT + d_shift)"""
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
        theta_per = xp.float64(a) - xp.float64(b) / denom
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

def print_result(label, m, elapsed=None):
    t7 = "PASS" if m['T7_pass'] else "FAIL"
    t8 = "PASS" if m['T8_pass'] else "FAIL"
    timing = f"  [{elapsed:.1f}s]" if elapsed else ""
    print(f"  {label}")
    print(f"    alpha = {m['alpha']:+.8f}  |a-1| = {m['abs_alpha_minus_1']:.8f}")
    print(f"    drift = {m['drift_slope']:+.8f} (p={m['drift_p']:.4f})")
    print(f"    T7: {t7}  T8: {t8}  CI: [{m['ci_lo']:.6f}, {m['ci_hi']:.6f}]{timing}")

# ================================================================
# MAIN
# ================================================================
print("=" * 70)
print("GIFT FORMULA VALIDATION — Colab A100")
print("theta(T) = 7/6 - phi/(logT - 15/8)")
print("=" * 70)

# Load / generate zeros
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

all_results = {}

# ================================================================
# PART 1: P_MAX comparison at d = -15/8
# ================================================================
print(f"\n{'='*70}")
print("PART 1: P_MAX COMPARISON at d = -15/8")
print(f"  a = 7/6 = {A:.6f}")
print(f"  b = phi = {B:.6f}")
print(f"  d = -15/8 = {D:.6f}")
print(f"{'='*70}")

for p_max in P_MAX_LIST:
    primes = sieve(p_max)
    n_primes = len(primes)
    print(f"\n  P_MAX = {p_max:,} ({n_primes:,} primes)")

    t0 = time.time()
    dp = prime_sum_shifted(gamma0, tp, primes, K_MAX, A, B, D)
    elapsed = time.time() - t0
    m = compute_metrics(delta, dp, gamma0, N_WINDOWS)

    print_result(f"P_MAX={p_max:,}", m, elapsed)
    print(f"    Window alphas: {['%.4f' % a for a in m['window_alphas']]}")

    all_results[f'pmax_{p_max}'] = {
        'P_MAX': p_max, 'N_primes': n_primes, 'k_max': K_MAX,
        'd_shift': D, 'elapsed': elapsed,
        **{k: v for k, v in m.items() if k != 'T_mids'},
    }
    sys.stdout.flush()

# ================================================================
# PART 2: d scan with P_MAX=500k
# ================================================================
P_MAX_MAIN = 500_000
primes_500k = sieve(P_MAX_MAIN)
n_primes_500k = len(primes_500k)
print(f"\n{'='*70}")
print(f"PART 2: d SCAN with P_MAX={P_MAX_MAIN:,} ({n_primes_500k:,} primes)")
print(f"  d in [{D_SCAN[0]:.2f}, {D_SCAN[-1]:.2f}], {len(D_SCAN)} points")
print(f"{'='*70}")

d_results = []
t_scan = time.time()

for i, d_val in enumerate(D_SCAN):
    t0 = time.time()
    dp = prime_sum_shifted(gamma0, tp, primes_500k, K_MAX, A, B, d_val)
    elapsed = time.time() - t0
    m = compute_metrics(delta, dp, gamma0, N_WINDOWS)
    m['d_shift'] = float(d_val)
    m['elapsed'] = elapsed

    t7 = "PASS" if m['T7_pass'] else "FAIL"
    t8 = "PASS" if m['T8_pass'] else "FAIL"
    print(f"  [{i+1:2d}/{len(D_SCAN)}] d={d_val:+.4f}  alpha={m['alpha']:+.6f}  |a-1|={m['abs_alpha_minus_1']:.6f}  "
          f"drift={m['drift_slope']:+.8f} (p={m['drift_p']:.4f})  T7={t7}  T8={t8}  [{elapsed:.1f}s]", flush=True)
    d_results.append({k: v for k, v in m.items() if k != 'T_mids'})

total_scan = time.time() - t_scan
print(f"\nd scan done in {total_scan/60:.1f} min")

best_alpha = min(d_results, key=lambda r: r['abs_alpha_minus_1'])
t7t8 = [r for r in d_results if r['T7_pass'] and r['T8_pass']]

print(f"\nBest |alpha-1|: d={best_alpha['d_shift']:.4f}  alpha={best_alpha['alpha']:+.8f}")
print(f"  T7={'PASS' if best_alpha['T7_pass'] else 'FAIL'}  T8={'PASS' if best_alpha['T8_pass'] else 'FAIL'}")

if t7t8:
    best_both = min(t7t8, key=lambda r: r['abs_alpha_minus_1'])
    print(f"\nBest T7+T8: d={best_both['d_shift']:.4f}  alpha={best_both['alpha']:+.8f}  drift_p={best_both['drift_p']:.4f}")
else:
    best_both = None
    print(f"\n  ** NO d passes both T7 and T8 at P_MAX={P_MAX_MAIN:,} **")

# ================================================================
# PART 3: k_max comparison at d = -15/8, P_MAX=500k
# ================================================================
print(f"\n{'='*70}")
print(f"PART 3: k_max COMPARISON at d = -15/8, P_MAX={P_MAX_MAIN:,}")
print(f"{'='*70}")

k_results = []
for k in K_SCAN:
    t0 = time.time()
    dp = prime_sum_shifted(gamma0, tp, primes_500k, k, A, B, D)
    elapsed = time.time() - t0
    m = compute_metrics(delta, dp, gamma0, N_WINDOWS)

    print_result(f"k_max={k}", m, elapsed)
    k_results.append({
        'k_max': k, 'elapsed': elapsed,
        **{kk: v for kk, v in m.items() if kk != 'T_mids'},
    })
    sys.stdout.flush()

# ================================================================
# PART 4: Symbolic d comparison at P_MAX=500k
# ================================================================
print(f"\n{'='*70}")
print(f"PART 4: SYMBOLIC d VALUES at P_MAX={P_MAX_MAIN:,}")
print(f"{'='*70}")

symbolic_d = {
    '-p2 = -2': -2.0,
    '-2phi^2/e': -2*phi**2/math.e,
    '-15/8': -15/8,
    '-11/6': -11/6,
    '-phi': -phi,
    '-gamma': -EULER_GAMMA,
}
sym_results = {}
for name, d_val in symbolic_d.items():
    t0 = time.time()
    dp = prime_sum_shifted(gamma0, tp, primes_500k, K_MAX, A, B, d_val)
    elapsed = time.time() - t0
    m = compute_metrics(delta, dp, gamma0, N_WINDOWS)
    print_result(f"d = {name} = {d_val:.6f}", m, elapsed)
    sym_results[name] = {
        'd_shift': float(d_val), 'elapsed': elapsed,
        **{k: v for k, v in m.items() if k != 'T_mids'},
    }
    sys.stdout.flush()

# ================================================================
# SUMMARY
# ================================================================
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")

print(f"\nP_MAX comparison at d=-15/8, k_max=3:")
for key in ['pmax_100000', 'pmax_500000']:
    r = all_results[key]
    t7 = "PASS" if r['T7_pass'] else "FAIL"
    t8 = "PASS" if r['T8_pass'] else "FAIL"
    print(f"  P_MAX={r['P_MAX']:>7,}: alpha={r['alpha']:+.8f}  drift_p={r['drift_p']:.4f}  T7={t7}  T8={t8}")

print(f"\nk_max comparison at d=-15/8, P_MAX={P_MAX_MAIN:,}:")
for r in k_results:
    t7 = "PASS" if r['T7_pass'] else "FAIL"
    t8 = "PASS" if r['T8_pass'] else "FAIL"
    print(f"  k_max={r['k_max']}: alpha={r['alpha']:+.8f}  drift_p={r['drift_p']:.4f}  T7={t7}  T8={t8}")

print(f"\nSymbolic d at P_MAX={P_MAX_MAIN:,}:")
for name, r in sym_results.items():
    t7 = "PASS" if r['T7_pass'] else "FAIL"
    t8 = "PASS" if r['T8_pass'] else "FAIL"
    print(f"  d={name:>15}: alpha={r['alpha']:+.8f}  |a-1|={r['abs_alpha_minus_1']:.6f}  drift_p={r['drift_p']:.4f}  T7={t7}  T8={t8}")

key_question = "RESOLVED" if all_results.get('pmax_500000', {}).get('T8_pass', False) else "PERSISTS"
print(f"\nKEY QUESTION: Does drift disappear with P_MAX=500k?")
print(f"  → Drift {key_question} at P_MAX=500k")

# ================================================================
# SAVE
# ================================================================
out = {
    'config': {
        'formula': 'theta = 7/6 - phi/(logT - 15/8)',
        'a': A, 'b': B, 'd': D,
        'N_zeros': N_TOTAL,
        'P_MAX_list': P_MAX_LIST,
        'K_MAX': K_MAX,
    },
    'pmax_comparison': all_results,
    'd_scan': d_results,
    'd_scan_best': {k: v for k, v in best_alpha.items()},
    'd_scan_best_t7t8': {k: v for k, v in best_both.items()} if best_both else None,
    'k_comparison': k_results,
    'symbolic_d': sym_results,
}
out_path = 'outputs/colab_gift_formula_500k_results.json'
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2, default=float)
print(f"\nSaved to {out_path}")
print("\nDone!")
