#!/usr/bin/env python3
"""
Two-scale correction: θ(T) = 7/6 − φ/(logT − p₂) + γ·c/logT

The bulk topological term φ/(logT − 2) fixes α ≈ 1 (T7).
The archimedean correction γ·c/logT absorbs the drift (T8).

Goal: find the value of c that passes BOTH T7 and T8 simultaneously,
then check if c matches a GIFT topological constant.

Upload to Colab, mount Drive, run:
  !python colab_two_scale_correction.py

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

# GIFT topological constants
DIM_G2 = 14
DIM_K7 = 7
H_STAR = 99
B2 = 21
B3 = 77
RANK_E8 = 8
P2 = 2
N_GEN = 3
DIM_E8 = 248
WEYL = 5
SPECTRAL_GAP = 13  # dim(G2) - 1

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
A = 7 / 6          # θ_∞ = dim(K7) / (2 * N_gen)
B_COEFF = phi       # numerator of main correction
D = -P2             # d = -p₂ = -2 (Pontryagin, best at 500k)
P_MAX = 500_000
K_MAX = 3
N_WINDOWS = 12

# c scan: fine grid around 0, plus symbolic candidates
C_SCAN_FINE = np.arange(-0.20, 0.205, 0.005)  # 81 points

# Symbolic GIFT candidates for c
SYMBOLIC_C = {
    'c=0 (baseline)':         0.0,
    'c=1/H*':                 1 / H_STAR,
    'c=1/dim(G2)':            1 / DIM_G2,
    'c=1/dim(K7)':            1 / DIM_K7,
    'c=p2/H*':                P2 / H_STAR,
    'c=dim(G2)/H*':           DIM_G2 / H_STAR,
    'c=1/b2':                 1 / B2,
    'c=1/b3':                 1 / B3,
    'c=N_gen/H*':             N_GEN / H_STAR,
    'c=1/13':                 1 / SPECTRAL_GAP,
    'c=1/(2*dim(K7))':        1 / (2 * DIM_K7),
    'c=rank(E8)/H*':          RANK_E8 / H_STAR,
    'c=1/dim(E8)':            1 / DIM_E8,
    'c=phi/H*':               phi / H_STAR,
    'c=1':                    1.0,
    'c=gamma':                EULER_GAMMA,
    'c=-1/H*':                -1 / H_STAR,
    'c=-1/dim(G2)':           -1 / DIM_G2,
    'c=-1/dim(K7)':           -1 / DIM_K7,
    'c=-p2/H*':               -P2 / H_STAR,
    'c=-dim(G2)/H*':          -DIM_G2 / H_STAR,
    'c=-1/b2':                -1 / B2,
    'c=-1/b3':                -1 / B3,
    'c=-1/13':                -1 / SPECTRAL_GAP,
    'c=-phi/H*':              -phi / H_STAR,
    'c=-1':                   -1.0,
    'c=-gamma':               -EULER_GAMMA,
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

def prime_sum_two_scale(g0, tp_v, primes, k_max, a, b, d_shift, c_corr):
    """Two-scale formula: theta = a - b/(logT + d_shift) + gamma*c/logT"""
    xp = cp if GPU else np
    N = len(g0)
    result = np.zeros(N, dtype=np.float64)
    log_primes_np = np.log(primes.astype(np.float64))
    gamma_c = EULER_GAMMA * c_corr

    for ic in range((N + ZERO_CHUNK - 1) // ZERO_CHUNK):
        lo = ic * ZERO_CHUNK
        hi = min(lo + ZERO_CHUNK, N)

        g0_c = xp.asarray(g0[lo:hi], dtype=xp.float64)
        tp_c = xp.asarray(tp_v[lo:hi], dtype=xp.float64)
        log_g0 = xp.log(xp.maximum(g0_c, 2.0))

        # Main term: a - b/(logT + d)
        denom = log_g0 + xp.float64(d_shift)
        denom = xp.maximum(denom, 0.1)
        theta_main = xp.float64(a) - xp.float64(b) / denom

        # Correction term: gamma * c / logT
        theta_corr = xp.float64(gamma_c) / log_g0

        # Combined
        theta_per = theta_main + theta_corr
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

def fmt_result(label, m, elapsed=None):
    t7 = "PASS" if m['T7_pass'] else "FAIL"
    t8 = "PASS" if m['T8_pass'] else "FAIL"
    timing = f"  [{elapsed:.1f}s]" if elapsed else ""
    return (f"  {label}\n"
            f"    alpha={m['alpha']:+.8f}  |a-1|={m['abs_alpha_minus_1']:.8f}\n"
            f"    drift={m['drift_slope']:+.8f} (p={m['drift_p']:.4f})\n"
            f"    T7: {t7}  T8: {t8}  CI: [{m['ci_lo']:.6f}, {m['ci_hi']:.6f}]{timing}")

# ================================================================
# MAIN
# ================================================================
print("=" * 70)
print("TWO-SCALE CORRECTION SCAN")
print("θ(T) = 7/6 − φ/(logT − 2) + γ·c/logT")
print("=" * 70)
print(f"  a       = 7/6 = {A:.6f}")
print(f"  b       = φ   = {B_COEFF:.6f}")
print(f"  d       = -p₂ = {D}")
print(f"  γ       = {EULER_GAMMA:.10f}")
print(f"  P_MAX   = {P_MAX:,}")
print(f"  k_max   = {K_MAX}")
print(f"  windows = {N_WINDOWS}")

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

# Sieve primes
primes = sieve(P_MAX)
n_primes = len(primes)
print(f"  {n_primes:,} primes (P_MAX={P_MAX:,})")

all_results = {}

# ================================================================
# PART 1: BASELINE (c=0, i.e. no correction)
# ================================================================
print(f"\n{'='*70}")
print("PART 1: BASELINE — θ(T) = 7/6 − φ/(logT − 2)  [c=0]")
print(f"{'='*70}")

t0 = time.time()
dp_baseline = prime_sum_two_scale(gamma0, tp, primes, K_MAX, A, B_COEFF, D, 0.0)
elapsed_base = time.time() - t0
m_base = compute_metrics(delta, dp_baseline, gamma0, N_WINDOWS)
print(fmt_result("BASELINE (c=0)", m_base, elapsed_base))
print(f"    Window alphas: {['%.4f' % a for a in m_base['window_alphas']]}")
all_results['baseline'] = {
    'c': 0.0, 'label': 'baseline',
    **{k: v for k, v in m_base.items() if k != 'T_mids'},
}
sys.stdout.flush()

# ================================================================
# PART 2: FINE c SCAN
# ================================================================
print(f"\n{'='*70}")
print(f"PART 2: FINE c SCAN — {len(C_SCAN_FINE)} values in [{C_SCAN_FINE[0]:.3f}, {C_SCAN_FINE[-1]:.3f}]")
print(f"{'='*70}")

c_results = []
t_scan = time.time()

for i, c_val in enumerate(C_SCAN_FINE):
    t0 = time.time()
    dp = prime_sum_two_scale(gamma0, tp, primes, K_MAX, A, B_COEFF, D, c_val)
    elapsed = time.time() - t0
    m = compute_metrics(delta, dp, gamma0, N_WINDOWS)

    t7 = "PASS" if m['T7_pass'] else "FAIL"
    t8 = "PASS" if m['T8_pass'] else "FAIL"
    marker = " ***" if m['T7_pass'] and m['T8_pass'] else ""
    print(f"  [{i+1:2d}/{len(C_SCAN_FINE)}] c={c_val:+.4f}  gamma*c={EULER_GAMMA*c_val:+.6f}  "
          f"alpha={m['alpha']:+.8f}  |a-1|={m['abs_alpha_minus_1']:.6f}  "
          f"drift_p={m['drift_p']:.4f}  T7={t7}  T8={t8}  [{elapsed:.1f}s]{marker}",
          flush=True)

    c_results.append({
        'c': float(c_val),
        'gamma_c': float(EULER_GAMMA * c_val),
        **{k: v for k, v in m.items() if k != 'T_mids'},
    })

total_scan = time.time() - t_scan
print(f"\nFine scan done in {total_scan/60:.1f} min")

# Identify best
best_alpha = min(c_results, key=lambda r: r['abs_alpha_minus_1'])
best_drift = max(c_results, key=lambda r: r['drift_p'])
t7t8 = [r for r in c_results if r['T7_pass'] and r['T8_pass']]

print(f"\n  Best |alpha-1|: c={best_alpha['c']:+.4f}  alpha={best_alpha['alpha']:+.8f}")
print(f"    T7={'PASS' if best_alpha['T7_pass'] else 'FAIL'}  T8={'PASS' if best_alpha['T8_pass'] else 'FAIL'}")
print(f"  Best drift_p:   c={best_drift['c']:+.4f}  drift_p={best_drift['drift_p']:.6f}")
print(f"    T7={'PASS' if best_drift['T7_pass'] else 'FAIL'}  T8={'PASS' if best_drift['T8_pass'] else 'FAIL'}")

if t7t8:
    best_both = min(t7t8, key=lambda r: r['abs_alpha_minus_1'])
    print(f"\n  *** BOTH T7+T8: c={best_both['c']:+.4f}  alpha={best_both['alpha']:+.8f}  "
          f"drift_p={best_both['drift_p']:.4f} ***")
else:
    best_both = None
    print(f"\n  ** NO c passes both T7 and T8 in fine scan **")

# ================================================================
# PART 3: SYMBOLIC GIFT CANDIDATES
# ================================================================
print(f"\n{'='*70}")
print(f"PART 3: SYMBOLIC GIFT CANDIDATES for c")
print(f"  θ(T) = 7/6 − φ/(logT − 2) + γ·c/logT")
print(f"{'='*70}")

sym_results = {}
for name, c_val in SYMBOLIC_C.items():
    t0 = time.time()
    dp = prime_sum_two_scale(gamma0, tp, primes, K_MAX, A, B_COEFF, D, c_val)
    elapsed = time.time() - t0
    m = compute_metrics(delta, dp, gamma0, N_WINDOWS)

    t7 = "PASS" if m['T7_pass'] else "FAIL"
    t8 = "PASS" if m['T8_pass'] else "FAIL"
    marker = " ***" if m['T7_pass'] and m['T8_pass'] else ""
    print(f"  {name:25s}  c={c_val:+.8f}  gamma*c={EULER_GAMMA*c_val:+.8f}  "
          f"alpha={m['alpha']:+.8f}  |a-1|={m['abs_alpha_minus_1']:.6f}  "
          f"drift_p={m['drift_p']:.4f}  T7={t7}  T8={t8}  [{elapsed:.1f}s]{marker}",
          flush=True)

    sym_results[name] = {
        'c': float(c_val),
        'gamma_c': float(EULER_GAMMA * c_val),
        **{k: v for k, v in m.items() if k != 'T_mids'},
    }

# ================================================================
# PART 4: DRIFT FUNCTIONAL FORM ANALYSIS
# ================================================================
print(f"\n{'='*70}")
print("PART 4: DRIFT ANALYSIS — is it 1/logT, 1/logT², or other?")
print(f"{'='*70}")

# Use baseline (c=0) window data
alphas_base = np.array(m_base['window_alphas'])
T_mids_base = np.array(m_base['T_mids'])
log_T = np.log(T_mids_base)
inv_log_T = 1.0 / log_T
inv_log_T_sq = 1.0 / log_T**2

# Fit alpha(T) = A + B * f(T) for various f(T)
models = {
    'linear in window index': (np.arange(len(alphas_base)), 'alpha = a + b*i'),
    '1/logT':                 (inv_log_T, 'alpha = a + b/logT'),
    '1/logT^2':               (inv_log_T_sq, 'alpha = a + b/logT²'),
    'logT':                   (log_T, 'alpha = a + b*logT'),
}

print(f"\n  {'Model':<30s}  {'R²':>8s}  {'slope':>12s}  {'intercept':>10s}  {'p-value':>10s}")
print(f"  {'-'*30}  {'-'*8}  {'-'*12}  {'-'*10}  {'-'*10}")

for name, (x_var, formula) in models.items():
    slope, intercept, r_value, p, stderr = stats.linregress(x_var, alphas_base)
    print(f"  {name:<30s}  {r_value**2:8.4f}  {slope:+12.6f}  {intercept:10.6f}  {p:10.6f}")

# ================================================================
# SUMMARY
# ================================================================
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")

print(f"\nFormula: θ(T) = 7/6 − φ/(logT − 2) + γ·c/logT")
print(f"  P_MAX = {P_MAX:,}  ({n_primes:,} primes)")
print(f"  N_zeros = {N_TOTAL:,}")
print(f"  k_max = {K_MAX}")

print(f"\nBaseline (c=0):")
t7 = "PASS" if m_base['T7_pass'] else "FAIL"
t8 = "PASS" if m_base['T8_pass'] else "FAIL"
print(f"  alpha={m_base['alpha']:+.8f}  drift_p={m_base['drift_p']:.4f}  T7={t7}  T8={t8}")

print(f"\nFine scan best:")
print(f"  Best |a-1|:  c={best_alpha['c']:+.4f}  alpha={best_alpha['alpha']:+.8f}  "
      f"drift_p={best_alpha['drift_p']:.4f}")
print(f"  Best drift:  c={best_drift['c']:+.4f}  alpha={best_drift['alpha']:+.8f}  "
      f"drift_p={best_drift['drift_p']:.6f}")
if best_both:
    print(f"  *** T7+T8:   c={best_both['c']:+.4f}  alpha={best_both['alpha']:+.8f}  "
          f"drift_p={best_both['drift_p']:.4f} ***")

print(f"\nSymbolic candidates ranking (by |alpha-1|):")
ranked = sorted(sym_results.items(), key=lambda x: x[1]['abs_alpha_minus_1'])
for name, r in ranked[:10]:
    t7 = "PASS" if r['T7_pass'] else "FAIL"
    t8 = "PASS" if r['T8_pass'] else "FAIL"
    marker = " <-- BOTH" if r['T7_pass'] and r['T8_pass'] else ""
    print(f"  {name:25s}  c={r['c']:+.8f}  |a-1|={r['abs_alpha_minus_1']:.6f}  "
          f"drift_p={r['drift_p']:.4f}  T7={t7}  T8={t8}{marker}")

# Check if any symbolic passes both
sym_both = {n: r for n, r in sym_results.items() if r['T7_pass'] and r['T8_pass']}
if sym_both:
    print(f"\n  *** SYMBOLIC CANDIDATES PASSING BOTH T7+T8: ***")
    for name, r in sym_both.items():
        print(f"    {name}: c={r['c']:+.8f}  alpha={r['alpha']:+.8f}  drift_p={r['drift_p']:.4f}")
else:
    print(f"\n  No symbolic candidate passes both T7 and T8")

# ================================================================
# SAVE
# ================================================================
out = {
    'config': {
        'formula': 'theta = 7/6 - phi/(logT - 2) + gamma*c/logT',
        'a': A, 'b': B_COEFF, 'd': D,
        'gamma': EULER_GAMMA,
        'N_zeros': N_TOTAL,
        'P_MAX': P_MAX, 'N_primes': n_primes,
        'K_MAX': K_MAX, 'N_WINDOWS': N_WINDOWS,
    },
    'baseline': all_results.get('baseline'),
    'c_scan': c_results,
    'c_scan_best_alpha': {k: v for k, v in best_alpha.items()},
    'c_scan_best_drift': {k: v for k, v in best_drift.items()},
    'c_scan_best_both': {k: v for k, v in best_both.items()} if best_both else None,
    'symbolic_candidates': sym_results,
    'symbolic_both_t7t8': {n: {k: v for k, v in r.items()} for n, r in sym_both.items()},
}
out_path = 'outputs/two_scale_correction_results.json'
os.makedirs('outputs', exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2, default=float)
print(f"\nSaved to {out_path}")
print("\nDone!")
