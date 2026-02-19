#!/usr/bin/env python3
"""
Drift correction v2: test two functional forms to kill the drift.

Form A (denominator):  θ(T) = 7/6 − φ/(logT − 2 − c/logT)
Form B (log-log):      θ(T) = 7/6 − φ/(logT − 2) + c·log(logT)

Both fix d = -p₂ = -2 (topological). The correction targets the T8 drift.

Key question: is c a GIFT topological ratio or Mertens-like?

Upload to Colab, run:
  !python colab_drift_correction_v2.py

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
MERTENS = 0.2614972128476428  # Meissel-Mertens constant

# GIFT constants
DIM_G2 = 14; DIM_K7 = 7; H_STAR = 99; B2 = 21; B3 = 77
RANK_E8 = 8; P2 = 2; N_GEN = 3; DIM_E8 = 248; WEYL = 5

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
A = 7 / 6
B_COEFF = phi
P_MAX = 500_000
K_MAX = 3
N_WINDOWS = 12

# c values for Form A (denominator: c/logT in denominator)
C_SCAN_A = np.concatenate([
    np.arange(-1.0, -0.05, 0.05),   # coarse negative
    np.arange(-0.05, 0.055, 0.005),  # fine around 0
    np.arange(0.10, 1.05, 0.05),     # coarse positive
])

# c values for Form B (additive: c*log(logT))
C_SCAN_B = np.concatenate([
    np.arange(-0.010, -0.001, 0.001),  # negative
    np.arange(-0.001, 0.0011, 0.0001), # very fine around 0
    np.arange(0.002, 0.011, 0.001),    # positive
])

# Symbolic candidates (shared for both forms, but different ranges)
SYMBOLIC_C_A = {
    'c=0':                0.0,
    'c=1/H*':             1/H_STAR,
    'c=1/dim(G2)':        1/DIM_G2,
    'c=1/dim(K7)':        1/DIM_K7,
    'c=p2/dim(G2)':       P2/DIM_G2,
    'c=1/b2':             1/B2,
    'c=N_gen/dim(G2)':    N_GEN/DIM_G2,
    'c=dim(G2)/H*':       DIM_G2/H_STAR,
    'c=gamma':            EULER_GAMMA,
    'c=Mertens':          MERTENS,
    'c=phi-1':            phi - 1,
    'c=1/phi':            1/phi,
    'c=1':                1.0,
    'c=-1/H*':            -1/H_STAR,
    'c=-1/dim(G2)':       -1/DIM_G2,
    'c=-1/dim(K7)':       -1/DIM_K7,
    'c=-gamma':           -EULER_GAMMA,
    'c=-Mertens':         -MERTENS,
}

SYMBOLIC_C_B = {
    'c=0':                0.0,
    'c=1/(H*^2)':         1/H_STAR**2,
    'c=1/(dim(G2)*H*)':   1/(DIM_G2*H_STAR),
    'c=1/H*':             1/H_STAR,
    'c=1/dim(E8)':        1/DIM_E8,
    'c=1/(2*H*)':         1/(2*H_STAR),
    'c=gamma/H*':         EULER_GAMMA/H_STAR,
    'c=Mertens/H*':       MERTENS/H_STAR,
    'c=1/dim(G2)^2':      1/DIM_G2**2,
    'c=1/(b2*b3)':        1/(B2*B3),
    'c=Mertens/dim(G2)':  MERTENS/DIM_G2,
    'c=gamma/dim(G2)':    EULER_GAMMA/DIM_G2,
    'c=-1/(H*^2)':        -1/H_STAR**2,
    'c=-1/H*':            -1/H_STAR,
    'c=-1/dim(E8)':       -1/DIM_E8,
    'c=-gamma/H*':        -EULER_GAMMA/H_STAR,
    'c=-Mertens/H*':      -MERTENS/H_STAR,
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

def prime_sum_generic(g0, tp_v, primes, k_max, theta_func):
    """Generic prime sum with arbitrary theta(T) function."""
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

        # Compute theta using provided function
        theta_per = theta_func(log_g0, xp)
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

def run_test(label, theta_func, delta, gamma0, tp, primes, k_max, n_windows):
    t0 = time.time()
    dp = prime_sum_generic(gamma0, tp, primes, k_max, theta_func)
    elapsed = time.time() - t0
    m = compute_metrics(delta, dp, gamma0, n_windows)

    t7 = "PASS" if m['T7_pass'] else "FAIL"
    t8 = "PASS" if m['T8_pass'] else "FAIL"
    marker = " ***" if m['T7_pass'] and m['T8_pass'] else ""
    print(f"  {label:40s}  alpha={m['alpha']:+.8f}  |a-1|={m['abs_alpha_minus_1']:.6f}  "
          f"drift_p={m['drift_p']:.4f}  T7={t7}  T8={t8}  [{elapsed:.1f}s]{marker}",
          flush=True)
    return m, elapsed

# ================================================================
# THETA FUNCTIONS
# ================================================================
def make_theta_baseline():
    """θ(T) = 7/6 - φ/(logT - 2)"""
    def f(log_g0, xp):
        denom = xp.maximum(log_g0 - xp.float64(2.0), xp.float64(0.1))
        return xp.float64(A) - xp.float64(B_COEFF) / denom
    return f

def make_theta_form_A(c):
    """Form A: θ(T) = 7/6 - φ/(logT - 2 - c/logT)"""
    def f(log_g0, xp):
        denom = log_g0 - xp.float64(2.0) - xp.float64(c) / log_g0
        denom = xp.maximum(denom, xp.float64(0.1))
        return xp.float64(A) - xp.float64(B_COEFF) / denom
    return f

def make_theta_form_B(c):
    """Form B: θ(T) = 7/6 - φ/(logT - 2) + c*log(logT)"""
    def f(log_g0, xp):
        denom = xp.maximum(log_g0 - xp.float64(2.0), xp.float64(0.1))
        main = xp.float64(A) - xp.float64(B_COEFF) / denom
        corr = xp.float64(c) * xp.log(log_g0)
        return main + corr
    return f

def make_theta_form_C(c):
    """Form C (hybrid): θ(T) = 7/6 - φ/(logT - 2) + c/log²T
    Intermediate: steeper than 1/logT but not as wild as log(logT)"""
    def f(log_g0, xp):
        denom = xp.maximum(log_g0 - xp.float64(2.0), xp.float64(0.1))
        main = xp.float64(A) - xp.float64(B_COEFF) / denom
        corr = xp.float64(c) / (log_g0 ** 2)
        return main + corr
    return f

# ================================================================
# MAIN
# ================================================================
print("=" * 70)
print("DRIFT CORRECTION v2 — Two Functional Forms")
print("Form A: θ = 7/6 − φ/(logT − 2 − c/logT)        [denominator]")
print("Form B: θ = 7/6 − φ/(logT − 2) + c·log(logT)   [Mertens-like]")
print("Form C: θ = 7/6 − φ/(logT − 2) + c/log²T        [1/log² decay]")
print("=" * 70)

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

all_results = {}

# ================================================================
# PART 0: BASELINE
# ================================================================
print(f"\n{'='*70}")
print("PART 0: BASELINE — θ = 7/6 − φ/(logT − 2)")
print(f"{'='*70}")
m_base, t_base = run_test("baseline (c=0)", make_theta_baseline(),
                           delta, gamma0, tp, primes, K_MAX, N_WINDOWS)
print(f"    Window alphas: {['%.4f' % a for a in m_base['window_alphas']]}")
all_results['baseline'] = {'c': 0.0, **{k: v for k, v in m_base.items() if k != 'T_mids'}}

# ================================================================
# PART 1: FORM A SCAN (denominator correction)
# ================================================================
print(f"\n{'='*70}")
print(f"PART 1: FORM A SCAN — θ = 7/6 − φ/(logT − 2 − c/logT)")
print(f"  {len(C_SCAN_A)} c values")
print(f"{'='*70}")

results_A = []
for i, c_val in enumerate(C_SCAN_A):
    m, elapsed = run_test(f"[{i+1:2d}/{len(C_SCAN_A)}] c={c_val:+.4f}",
                          make_theta_form_A(c_val),
                          delta, gamma0, tp, primes, K_MAX, N_WINDOWS)
    results_A.append({'c': float(c_val), 'form': 'A',
                      **{k: v for k, v in m.items() if k != 'T_mids'}})

# ================================================================
# PART 2: FORM B SCAN (log-log correction)
# ================================================================
print(f"\n{'='*70}")
print(f"PART 2: FORM B SCAN — θ = 7/6 − φ/(logT − 2) + c·log(logT)")
print(f"  {len(C_SCAN_B)} c values")
print(f"{'='*70}")

results_B = []
for i, c_val in enumerate(C_SCAN_B):
    m, elapsed = run_test(f"[{i+1:2d}/{len(C_SCAN_B)}] c={c_val:+.6f}",
                          make_theta_form_B(c_val),
                          delta, gamma0, tp, primes, K_MAX, N_WINDOWS)
    results_B.append({'c': float(c_val), 'form': 'B',
                      **{k: v for k, v in m.items() if k != 'T_mids'}})

# ================================================================
# PART 3: FORM C SCAN (1/log²T correction)
# ================================================================
C_SCAN_C = np.arange(-1.0, 1.05, 0.05)
print(f"\n{'='*70}")
print(f"PART 3: FORM C SCAN — θ = 7/6 − φ/(logT − 2) + c/log²T")
print(f"  {len(C_SCAN_C)} c values")
print(f"{'='*70}")

results_C = []
for i, c_val in enumerate(C_SCAN_C):
    m, elapsed = run_test(f"[{i+1:2d}/{len(C_SCAN_C)}] c={c_val:+.4f}",
                          make_theta_form_C(c_val),
                          delta, gamma0, tp, primes, K_MAX, N_WINDOWS)
    results_C.append({'c': float(c_val), 'form': 'C',
                      **{k: v for k, v in m.items() if k != 'T_mids'}})

# ================================================================
# PART 4: SYMBOLIC CANDIDATES — ALL FORMS
# ================================================================
print(f"\n{'='*70}")
print("PART 4: SYMBOLIC GIFT CANDIDATES")
print(f"{'='*70}")

sym_results = {}

print(f"\n  --- Form A: θ = 7/6 − φ/(logT − 2 − c/logT) ---")
for name, c_val in SYMBOLIC_C_A.items():
    m, elapsed = run_test(f"A: {name}", make_theta_form_A(c_val),
                          delta, gamma0, tp, primes, K_MAX, N_WINDOWS)
    sym_results[f'A:{name}'] = {'c': float(c_val), 'form': 'A',
                                 **{k: v for k, v in m.items() if k != 'T_mids'}}

print(f"\n  --- Form B: θ = 7/6 − φ/(logT − 2) + c·log(logT) ---")
for name, c_val in SYMBOLIC_C_B.items():
    m, elapsed = run_test(f"B: {name}", make_theta_form_B(c_val),
                          delta, gamma0, tp, primes, K_MAX, N_WINDOWS)
    sym_results[f'B:{name}'] = {'c': float(c_val), 'form': 'B',
                                 **{k: v for k, v in m.items() if k != 'T_mids'}}

# ================================================================
# SUMMARY
# ================================================================
print(f"\n{'='*70}")
print("GRAND SUMMARY")
print(f"{'='*70}")

for form_name, results in [('A (denom)', results_A), ('B (loglog)', results_B), ('C (1/log²)', results_C)]:
    best_a = min(results, key=lambda r: r['abs_alpha_minus_1'])
    best_d = max(results, key=lambda r: r['drift_p'])
    both = [r for r in results if r['T7_pass'] and r['T8_pass']]

    print(f"\nForm {form_name}:")
    print(f"  Best |α-1|:  c={best_a['c']:+.6f}  α={best_a['alpha']:+.8f}  drift_p={best_a['drift_p']:.4f}  "
          f"T7={'P' if best_a['T7_pass'] else 'F'}  T8={'P' if best_a['T8_pass'] else 'F'}")
    print(f"  Best drift:  c={best_d['c']:+.6f}  α={best_d['alpha']:+.8f}  drift_p={best_d['drift_p']:.6f}  "
          f"T7={'P' if best_d['T7_pass'] else 'F'}  T8={'P' if best_d['T8_pass'] else 'F'}")
    if both:
        best_b = min(both, key=lambda r: r['abs_alpha_minus_1'])
        print(f"  *** T7+T8:   c={best_b['c']:+.6f}  α={best_b['alpha']:+.8f}  drift_p={best_b['drift_p']:.4f} ***")
    else:
        print(f"  No c passes both T7 and T8")

# Symbolic winners
print(f"\nSymbolic candidates passing T7+T8:")
sym_both = {n: r for n, r in sym_results.items() if r['T7_pass'] and r['T8_pass']}
if sym_both:
    for name, r in sorted(sym_both.items(), key=lambda x: x[1]['abs_alpha_minus_1']):
        print(f"  {name:40s}  c={r['c']:+.8f}  α={r['alpha']:+.8f}  drift_p={r['drift_p']:.4f}")
else:
    print("  None")

print(f"\nTop 5 symbolic by |α-1| (any form):")
ranked = sorted(sym_results.items(), key=lambda x: x[1]['abs_alpha_minus_1'])
for name, r in ranked[:5]:
    t7 = "P" if r['T7_pass'] else "F"
    t8 = "P" if r['T8_pass'] else "F"
    print(f"  {name:40s}  c={r['c']:+.8f}  |α-1|={r['abs_alpha_minus_1']:.6f}  "
          f"drift_p={r['drift_p']:.4f}  T7={t7}  T8={t8}")

# ================================================================
# SAVE
# ================================================================
out = {
    'config': {
        'forms': ['A: denom', 'B: loglog', 'C: 1/log^2'],
        'a': A, 'b': B_COEFF, 'd': -2,
        'N_zeros': N_TOTAL, 'P_MAX': P_MAX,
        'K_MAX': K_MAX, 'N_WINDOWS': N_WINDOWS,
    },
    'baseline': all_results.get('baseline'),
    'form_A_scan': results_A,
    'form_B_scan': results_B,
    'form_C_scan': results_C,
    'symbolic': sym_results,
}
out_path = 'outputs/drift_correction_v2_results.json'
os.makedirs('outputs', exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(out, f, indent=2, default=float)
print(f"\nSaved to {out_path}")
print("\nDone!")
