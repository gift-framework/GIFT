#!/usr/bin/env python3
"""
3D Topological Exploration of θ(T) = a − b/log(T) + c/log²(T)

Methodical grid search over GIFT topological constants + mathematical constants.
All (a, b, c) are constructed from topology — zero free parameters.

Uses CuPy/CUDA when available, falls back to NumPy.

Strategy:
  Stage 1: Fast screen on 50k zeros (small primes) — full 3D grid
  Stage 2: Refine top 500 with medium primes on 50k zeros
  Stage 3: Full 2M validation of top 10 with all primes + drift/bootstrap
"""

import numpy as np
from scipy.special import loggamma, lambertw
from scipy import stats
import json, time, math, os, sys
from math import gcd
from itertools import product as cartesian

# ============================================================
# GPU detection
# ============================================================
os.environ.setdefault('CUDA_PATH', '/usr')  # Ubuntu nvrtc path

try:
    import cupy as cp
    # Test kernel compilation
    _ = cp.maximum(cp.array([1.0]), 0.0)
    GPU = True
    print(f"[GPU] CuPy {cp.__version__}, "
          f"free={cp.cuda.Device(0).mem_info[0]//1024//1024} MB")
except (ImportError, Exception) as e:
    GPU = False
    print(f"[CPU] GPU not available ({e}), using NumPy")

# ============================================================
# Configuration
# ============================================================
ZEROS_FILE = os.path.join(os.path.dirname(__file__),
                          "outputs/riemann_zeros_2M_genuine.npy")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "explore_3d_results.json")

K_MAX = 3           # prime power cutoff
P_FAST = 5_000      # primes for Stage 1 (fast) — ~669 primes
P_MED = 50_000      # primes for Stage 2 (medium) — ~5,133 primes
P_FULL = 500_000    # primes for Stage 3 (full) — ~41,538 primes
N_SCREEN = 5_000    # zeros for screening Stage 1 — 214 evals/s on GPU
N_REFINE = 50_000   # zeros for Stage 2 refinement
N_FULL = 0          # set after loading (all 2M)
BATCH_GPU = 500     # primes per GPU batch (tuned for RTX 2050 4GB)

# ============================================================
# GIFT topological constants
# ============================================================
DIM_E8 = 248
RANK_E8 = 8
DIM_G2 = 14
DIM_K7 = 7
B2 = 21
B3 = 77
HSTAR = 99       # b2 + b3 + 1
P2 = 2           # Pontryagin
N_GEN = 3
WEYL = 5
D_BULK = 11
DIM_J3O = 27
TWO_B2 = 42

# Mathematical constants
PHI = (1 + math.sqrt(5)) / 2
EULER_GAMMA = 0.5772156649015329
SQRT2 = math.sqrt(2)
LOG2 = math.log(2)
LOG2PI = math.log(2 * math.pi)

# ============================================================
# Infrastructure (exact copy from theta_candidates_test.py)
# ============================================================

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
    for _ in range(40):
        dt = (theta_rs(t) - targets) / np.maximum(np.abs(theta_deriv(t)), 1e-15)
        t -= dt
        if np.max(np.abs(dt)) < 1e-12:
            break
    return t

def w_cosine(x):
    return np.where(x < 1.0, np.cos(np.pi * x / 2)**2, 0.0)

def sieve(N):
    is_p = np.ones(N + 1, dtype=bool); is_p[:2] = False
    for i in range(2, int(N**0.5) + 1):
        if is_p[i]: is_p[i*i::i] = False
    return np.where(is_p)[0]

def prime_sum_var(g0, tp_v, primes, k_max, theta_inf, theta_coeff, c_coeff=0.0):
    """Scalar version (for Stage 3 chunks)."""
    S = np.zeros_like(g0)
    log_g0 = np.log(np.maximum(g0, 2.0))
    if theta_coeff == 0.0 and c_coeff == 0.0:
        log_X = theta_inf * log_g0
    else:
        theta_per = theta_inf + theta_coeff / log_g0
        if c_coeff != 0.0:
            theta_per += c_coeff / log_g0**2
        theta_per = np.clip(theta_per, 0.5, 2.0)
        log_X = theta_per * log_g0
    log_primes = np.log(primes.astype(float))
    for j in range(len(primes)):
        p = float(primes[j])
        logp = log_primes[j]
        if logp / np.max(log_X) > 3.0:
            break
        for m in range(1, k_max + 1):
            x = m * logp / log_X
            weight = w_cosine(x)
            if np.max(weight) < 1e-15:
                continue
            S -= weight * np.sin(g0 * m * logp) / (m * p ** (m / 2.0))
    return -S / tp_v


def prime_sum_batched(g0, tp_v, primes, k_max, theta_inf, theta_coeff,
                       c_coeff=0.0, batch_size=200):
    """Vectorized over primes in batches — 10-50x faster than scalar loop.

    For each batch of B primes and each power m, compute (B, N) matrices
    and reduce. Removes Python loop over individual primes.
    """
    xp = cp if GPU else np
    N = len(g0)
    g0_ = xp.asarray(g0) if not isinstance(g0, xp.ndarray) else g0
    tp_ = xp.asarray(tp_v) if not isinstance(tp_v, xp.ndarray) else tp_v

    log_g0 = xp.log(xp.maximum(g0_, 2.0))

    if theta_coeff == 0.0 and c_coeff == 0.0:
        log_X = theta_inf * log_g0
    else:
        theta_per = theta_inf + theta_coeff / log_g0
        if c_coeff != 0.0:
            theta_per += c_coeff / log_g0**2
        theta_per = xp.clip(theta_per, 0.5, 2.0)
        log_X = theta_per * log_g0

    S = xp.zeros(N, dtype=xp.float64)
    log_X_max = float(xp.max(log_X))

    log_primes = np.log(primes.astype(np.float64))

    for m in range(1, k_max + 1):
        # Find which primes contribute for this m
        cutoff = log_X_max * 1.0 / m  # x < 1 means m*logp < logX
        j_max = np.searchsorted(log_primes, cutoff + 0.1)
        if j_max == 0:
            continue

        active_primes = primes[:j_max]
        active_logp = log_primes[:j_max]

        # Process in batches
        for b_start in range(0, j_max, batch_size):
            b_end = min(b_start + batch_size, j_max)
            B = b_end - b_start

            logp_b = xp.asarray(active_logp[b_start:b_end])  # (B,)
            p_b = xp.asarray(active_primes[b_start:b_end].astype(np.float64))

            # x[b, n] = m * logp[b] / log_X[n]
            x = m * logp_b[:, None] / log_X[None, :]  # (B, N)

            # Cosine-squared weight, zero outside [0, 1)
            w = xp.where(x < 1.0, xp.cos(xp.pi * x / 2)**2, 0.0)  # (B, N)

            # sin(g0 * m * logp) / (m * p^{m/2})
            phase = g0_[None, :] * (m * logp_b[:, None])  # (B, N)
            coeff = 1.0 / (m * p_b ** (m / 2.0))         # (B,)
            terms = w * xp.sin(phase) * coeff[:, None]    # (B, N)

            S -= xp.sum(terms, axis=0)  # reduce over batch → (N,)

    result = -S / tp_
    if GPU:
        return cp.asnumpy(result)
    return result

def compute_alpha_R2(delta, delta_pred):
    denom = np.dot(delta_pred, delta_pred)
    alpha = float(np.dot(delta, delta_pred) / denom) if denom > 0 else 0.0
    R2 = float(1.0 - np.var(delta - delta_pred) / np.var(delta))
    return alpha, R2

def compute_drift(alphas):
    if len(alphas) < 3:
        return 0.0, 1.0
    x = np.arange(len(alphas))
    slope, intercept, r, p, se = stats.linregress(x, alphas)
    return slope, p

# ============================================================
# BUILD CANDIDATE POOLS — topology + math, methodical
# ============================================================

def build_pools():
    """Build (a, b, c) candidate pools from GIFT topology + math constants.

    Design choices:
    - a ∈ [1.00, 1.55]: best 2D winners cluster at 1.1–1.3
    - b ∈ [0.5, 6.0]: best 2D winners cluster at 2–4
    - c ∈ [-10, 10]: Phase 4 showed c=10 already overcorrects
    - Denominators ≤ 11 for a,b rationals (GIFT-scale complexity)
    - Target: ~50 × ~80 × ~100 ≈ 400k triplets
    """

    # === a-pool: θ_∞ ∈ [1.00, 1.55] ===
    a_set = set()

    # Rationals p/q with denominators ≤ 11
    for q in range(1, 12):
        for p in range(1, 20):
            if gcd(p, q) == 1:
                v = p / q
                if 1.00 <= v <= 1.55:
                    a_set.add(round(v, 12))

    # Named topological ratios
    for v in [
        D_BULK / (DIM_K7 + P2),          # 11/9 = 1.222
        DIM_K7 / (WEYL + 1),             # 7/6 = 1.167
        (DIM_K7 + N_GEN) / DIM_K7,       # 10/7 = 1.429 (GIFT original)
        RANK_E8 / DIM_K7,                # 8/7 = 1.143
        DIM_G2 / D_BULK,                 # 14/11 = 1.273
        B2 / DIM_G2,                     # 3/2 = 1.500
        (DIM_K7 + P2) / DIM_K7,          # 9/7 = 1.286
        DIM_G2 / (D_BULK + P2),          # 14/13 = 1.077
        (DIM_K7 + WEYL) / D_BULK,        # 12/11 = 1.091
        D_BULK / RANK_E8,                # 11/8 = 1.375
        N_GEN / P2,                      # 3/2
        1.0,                             # exact 1
    ]:
        if 1.00 <= v <= 1.55:
            a_set.add(round(v, 12))

    # Math constant expressions
    for v in [
        PHI - 0.5,              # ≈ 1.118
        math.pi / math.e,       # ≈ 1.156
        (1 + SQRT2) / 2,        # ≈ 1.207
        PHI / SQRT2,            # ≈ 1.144
        SQRT2 - 0.2,            # ≈ 1.214
        EULER_GAMMA + 0.5,      # ≈ 1.077
        math.pi / 3,            # ≈ 1.047
        math.e / 2,             # ≈ 1.359
        3 * math.pi / 7,        # ≈ 1.347
        math.log(math.pi),      # ≈ 1.145
        (math.e + 1)/(math.pi + 1),  # ≈ 0.897 (out)
        PHI**2 - 1,             # ≈ 1.618 (out)
    ]:
        if 1.00 <= v <= 1.55:
            a_set.add(round(v, 12))

    a_vals = sorted(a_set)

    # === b-pool: correction coefficient ∈ [0.5, 6.0] ===
    b_set = set()

    # Rationals with denominators ≤ 9 (keeps pool manageable ~100 values)
    for q in range(1, 10):
        for p in range(1, 55):
            if gcd(p, q) == 1:
                v = p / q
                if 0.5 <= v <= 6.0:
                    b_set.add(round(v, 12))

    # Named topological
    for v in [
        WEYL / P2,                  # 5/2
        DIM_G2 / N_GEN,             # 14/3 (GIFT original)
        DIM_K7 / N_GEN,             # 7/3
        DIM_K7 / P2,                # 7/2
        D_BULK / N_GEN,             # 11/3
        D_BULK / P2,                # 11/2
        RANK_E8 / N_GEN,            # 8/3
        RANK_E8 / WEYL,             # 8/5
        DIM_G2 / WEYL,              # 14/5
        B2 / DIM_K7,                # 3
        B2 / D_BULK,                # 21/11
        (DIM_K7 + N_GEN) / N_GEN,   # 10/3
    ]:
        if 0.5 <= v <= 6.0:
            b_set.add(round(v, 12))

    # Math constants
    for v in [
        PHI, math.pi, math.e, SQRT2, 2*PHI, PHI + 1,
        math.pi/2, math.e/2, 2*SQRT2, LOG2PI,
        math.pi/PHI, math.e/PHI, PHI**2,
    ]:
        if 0.5 <= v <= 6.0:
            b_set.add(round(v, 12))

    b_vals = sorted(b_set)

    # === c-pool: subleading ∈ [-10, 10] ===
    c_set = {0.0}  # always include c=0 baseline

    # GIFT integers (those ≤ 10)
    for v in [2, 3, 5, 7, 8]:
        c_set.add(float(v))
        c_set.add(float(-v))

    # Rationals from GIFT atoms with denominators ≤ 7
    for p in [1, 2, 3, 5, 7, 8, 11, 14]:
        for q in [2, 3, 5, 7]:
            if gcd(p, q) == 1:
                v = p / q
                if 0 < v <= 10:
                    c_set.add(round(v, 12))
                    c_set.add(round(-v, 12))

    # Fine grid near zero (small corrections, most likely to help)
    for v in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5,
              3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
        c_set.add(v)
        c_set.add(-v)

    # Math constants
    for v in [PHI, math.pi, math.e, EULER_GAMMA, SQRT2, LOG2, LOG2PI,
              PHI**2, math.pi/2, math.e/2, 2*PHI, 2*math.pi, 2*math.e,
              math.pi*EULER_GAMMA, 7*EULER_GAMMA, math.pi**2/6]:
        for s in [1, -1]:
            if abs(s * v) <= 10:
                c_set.add(round(s * v, 10))

    c_vals = sorted(c_set)

    return a_vals, b_vals, c_vals


def describe_val(v):
    """Identify a value as a topological/math expression."""
    if abs(v) < 1e-12:
        return "0"
    # Check simple rationals
    for p in range(0, 100):
        for q in range(1, 30):
            if gcd(p, q) == 1 and abs(abs(v) - p/q) < 1e-8:
                sign = "" if v > 0 else "−"
                return f"{sign}{p}/{q}" if q > 1 else f"{sign}{p}"
    # Check math constants
    checks = [
        ('φ', PHI), ('π', math.pi), ('e', math.e), ('γ', EULER_GAMMA),
        ('√2', SQRT2), ('log2', LOG2), ('log2π', LOG2PI),
        ('φ²', PHI**2), ('π/2', math.pi/2), ('e/2', math.e/2),
        ('2φ', 2*PHI), ('2π', 2*math.pi), ('2e', 2*math.e),
        ('π/φ', math.pi/PHI), ('e/φ', math.e/PHI),
        ('πγ', math.pi*EULER_GAMMA), ('7γ', 7*EULER_GAMMA),
        ('π²/6', math.pi**2/6), ('φπ', PHI*math.pi),
        ('e√2', math.e*SQRT2),
    ]
    for label, ref in checks:
        if abs(v - ref) < 1e-7:
            return label
        if abs(v + ref) < 1e-7:
            return f"−{label}"
    return f"{v:.6f}"


# ============================================================
# MAIN
# ============================================================

def prime_sum_gpu_f32(g0, tp, log_g0, primes_np, log_primes_np, k_max,
                      theta_inf, theta_coeff, c_coeff=0.0):
    """GPU float32 prime sum — optimized for RTX 2050.

    All (J, N) matrix operations in float32, one batch per m-value.
    """
    xp = cp if GPU else np
    N = len(g0)

    theta_per = xp.float32(theta_inf) + xp.float32(theta_coeff) / log_g0
    if c_coeff != 0.0:
        theta_per = theta_per + xp.float32(c_coeff) / log_g0**2
    theta_per = xp.clip(theta_per, 0.5, 2.0)
    log_X = theta_per * log_g0

    S = xp.zeros(N, dtype=xp.float32)
    log_X_max = float(xp.max(log_X))

    for m in range(1, k_max + 1):
        cutoff = log_X_max / m
        j_max = np.searchsorted(log_primes_np, cutoff + 0.1)
        if j_max == 0:
            continue

        logp = xp.asarray(log_primes_np[:j_max], dtype=xp.float32)
        p = xp.asarray(primes_np[:j_max].astype(np.float32))
        coeff = xp.float32(1.0 / m) / p ** (m / 2.0)  # (J,)

        # (J, N) operations — one shot per m
        x = (xp.float32(m) * logp[:, None]) / log_X[None, :]
        w = xp.where(x < 1.0, xp.cos(xp.float32(math.pi / 2) * x)**2,
                     xp.float32(0))

        sin_w = xp.sin(g0[None, :] * (xp.float32(m) * logp[:, None]))
        sin_w *= coeff[:, None]
        sin_w *= w
        S -= xp.sum(sin_w, axis=0)

        del x, w, sin_w, logp, p, coeff

    result = -S / tp
    if GPU:
        xp.get_default_memory_pool().free_all_blocks()
        return cp.asnumpy(result.astype(cp.float64))
    return result.astype(np.float64)


def main():
    t_global = time.time()

    # --- Load zeros ---
    print(f"Loading zeros from {ZEROS_FILE}...")
    gamma_n_all = np.load(ZEROS_FILE)
    N_FULL = len(gamma_n_all)
    print(f"  {N_FULL:,} zeros, range [{gamma_n_all[0]:.2f}, {gamma_n_all[-1]:.2f}]")

    # --- Smooth zeros ---
    print("Computing smooth zeros (Newton iteration)...")
    t0 = time.time()
    gamma0_all = smooth_zeros(N_FULL)
    tp_all = theta_deriv(gamma0_all)
    delta_all = gamma_n_all - gamma0_all
    print(f"  Done in {time.time()-t0:.1f}s")

    # --- Primes ---
    print(f"Sieving primes up to {P_FULL:,}...")
    primes_all = sieve(P_FULL)
    primes_fast = primes_all[primes_all <= P_FAST]
    primes_med = primes_all[primes_all <= P_MED]
    log_primes_fast = np.log(primes_fast.astype(np.float64))
    log_primes_med = np.log(primes_med.astype(np.float64))
    print(f"  {len(primes_all):,} primes total, "
          f"{len(primes_fast):,} fast, {len(primes_med):,} medium")

    # --- Build pools ---
    a_vals, b_vals, c_vals = build_pools()
    total = len(a_vals) * len(b_vals) * len(c_vals)
    print(f"\n{'='*70}")
    print(f"CANDIDATE POOLS")
    print(f"  a: {len(a_vals)} values in [{min(a_vals):.4f}, {max(a_vals):.4f}]")
    print(f"  b: {len(b_vals)} values in [{min(b_vals):.4f}, {max(b_vals):.4f}]")
    print(f"  c: {len(c_vals)} values in [{min(c_vals):.4f}, {max(c_vals):.4f}]")
    print(f"  Total: {len(a_vals)} × {len(b_vals)} × {len(c_vals)} = {total:,}")
    print(f"{'='*70}")

    # --- Screening data ---
    xp = cp if GPU else np
    dtype_screen = xp.float32 if GPU else np.float64

    # Stage 1: 5k zeros, fast primes
    g0_s1 = xp.asarray(gamma0_all[:N_SCREEN], dtype=dtype_screen)
    tp_s1 = xp.asarray(tp_all[:N_SCREEN], dtype=dtype_screen)
    log_g0_s1 = xp.log(xp.maximum(g0_s1, xp.float32(2.0) if GPU else 2.0))
    delta_s1 = delta_all[:N_SCREEN]  # keep on CPU for metrics
    ws = N_SCREEN // 4
    WINDOWS = [(i*ws, (i+1)*ws) for i in range(4)]

    # Stage 2: 50k zeros
    g0_s2 = gamma0_all[:N_REFINE]
    tp_s2 = tp_all[:N_REFINE]
    delta_s2 = delta_all[:N_REFINE]
    ws_ref = N_REFINE // 4
    WINDOWS_REF = [(i*ws_ref, (i+1)*ws_ref) for i in range(4)]

    # --- Benchmark ---
    print("\nBenchmarking GPU f32 eval speed...")
    dp_test = prime_sum_gpu_f32(g0_s1, tp_s1, log_g0_s1, primes_fast,
                                 log_primes_fast, K_MAX, 11/9, -5/2, 1.0)
    if GPU:
        cp.cuda.Stream.null.synchronize()
    t_bench = time.time()
    for _ in range(10):
        _ = prime_sum_gpu_f32(g0_s1, tp_s1, log_g0_s1, primes_fast,
                               log_primes_fast, K_MAX, 11/9, -5/2, 1.0)
    if GPU:
        cp.cuda.Stream.null.synchronize()
    bench_ms = (time.time() - t_bench) / 10 * 1000
    rate_est = 1000 / bench_ms
    print(f"  {bench_ms:.1f} ms/eval → {rate_est:.0f} evals/s")
    print(f"  Estimated Stage 1 ({total:,} evals): {total/rate_est/60:.0f} min")

    # ============================================================
    # STAGE 1: FAST SCREENING (5k zeros, small primes, GPU f32)
    # ============================================================
    print(f"\n{'='*70}")
    print(f"STAGE 1 — FAST SCREENING ({N_SCREEN:,} zeros, {len(primes_fast):,} primes, "
          f"{'GPU f32' if GPU else 'CPU'})")
    print(f"{'='*70}\n")

    TOP_KEEP = 1000
    top_results = []
    best_score = 999.0
    t_s1 = time.time()
    tested = 0

    for a_val in a_vals:
        for b_val in b_vals:
            for c_val in c_vals:
                tested += 1

                if tested % 5000 == 0:
                    el = time.time() - t_s1
                    rate = tested / el if el > 0 else 0
                    eta = (total - tested) / rate / 60 if rate > 0 else 0
                    pct = 100 * tested / total
                    print(f"  [{pct:5.1f}%] {tested:>10,}/{total:,}  "
                          f"{rate:.0f}/s  ETA {eta:.0f}m  best={best_score:.6f}",
                          flush=True)

                dp = prime_sum_gpu_f32(g0_s1, tp_s1, log_g0_s1, primes_fast,
                                       log_primes_fast, K_MAX,
                                       a_val, -b_val, c_val)

                denom = np.dot(dp, dp)
                if denom <= 0:
                    continue
                alpha_val = float(np.dot(delta_s1, dp) / denom)

                # Quick 4-window drift
                alphas_w = []
                for lo, hi in WINDOWS:
                    d_w = delta_s1[lo:hi]
                    dp_w = dp[lo:hi]
                    dot_pp = np.dot(dp_w, dp_w)
                    alphas_w.append(float(np.dot(d_w, dp_w) / dot_pp)
                                    if dot_pp > 0 else 0.0)

                drift_val = float(stats.linregress(
                    np.arange(4, dtype=float), alphas_w).slope)

                score = abs(alpha_val - 1) + abs(drift_val) * 50

                if score < best_score:
                    best_score = score
                    c_s = describe_val(c_val)
                    print(f"  ** NEW BEST: a={a_val:.6f} b={b_val:.6f} c={c_val:+.4f} ({c_s}) "
                          f"→ α={alpha_val:.6f} drift={drift_val:+.6f} score={score:.6f}",
                          flush=True)

                if len(top_results) < TOP_KEEP or score < top_results[-1]['score']:
                    entry = {
                        'a': float(a_val), 'b': float(b_val), 'c': float(c_val),
                        'alpha': float(alpha_val), 'drift': float(drift_val),
                        'score': float(score),
                        'window_alphas': [float(x) for x in alphas_w],
                    }
                    top_results.append(entry)
                    top_results.sort(key=lambda x: x['score'])
                    if len(top_results) > TOP_KEEP:
                        top_results = top_results[:TOP_KEEP]

    s1_time = time.time() - t_s1
    print(f"\nStage 1 complete: {tested:,} triplets in {s1_time/60:.1f} min "
          f"({tested/s1_time:.0f}/s)")

    # Print top 30
    print(f"\n{'='*130}")
    print(f"STAGE 1 — TOP 30")
    print(f"{'='*130}")
    print(f"{'Rk':>3} {'a':>10} {'b':>10} {'c':>10} "
          f"{'alpha':>10} {'drift':>10} {'score':>10}  Formula")
    print("-" * 130)
    for i, r in enumerate(top_results[:30], 1):
        a_s = describe_val(r['a'])
        b_s = describe_val(r['b'])
        c_s = describe_val(r['c'])
        formula = f"{a_s} − {b_s}/logT"
        if abs(r['c']) > 1e-10:
            formula += f" + {c_s}/log²T" if r['c'] > 0 else f" − {describe_val(-r['c'])}/log²T"
        else:
            formula += "  [2D]"
        print(f"{i:>3} {r['a']:>10.6f} {r['b']:>10.6f} {r['c']:>+10.4f} "
              f"{r['alpha']:>+10.6f} {r['drift']:>+10.6f} "
              f"{r['score']:>10.6f}  {formula}")

    n_3d = sum(1 for r in top_results[:30] if abs(r['c']) > 1e-10)
    print(f"\n  → {n_3d}/30 use c≠0  |  {30-n_3d}/30 are 2D (c=0)")

    # ============================================================
    # STAGE 2: REFINE TOP 500 WITH MEDIUM PRIMES (50k)
    # ============================================================
    print(f"\n{'='*70}")
    print(f"STAGE 2 — REFINEMENT ({len(primes_med):,} primes, top 500)")
    print(f"{'='*70}")

    refined = []
    t_s2 = time.time()
    for i, r in enumerate(top_results[:500]):
        if (i+1) % 50 == 0:
            el = time.time() - t_s2
            print(f"  {i+1}/500 [{el:.0f}s, ~{el/(i+1)*500/60:.0f}m total]",
                  flush=True)

        # Stage 2 uses CPU f64 for precision (only 500 evals)
        dp = prime_sum_var(g0_s2, tp_s2, primes_med, K_MAX,
                           r['a'], -r['b'], r['c'])
        alpha_val, R2_val = compute_alpha_R2(delta_s2, dp)

        alphas_w = []
        for lo, hi in WINDOWS_REF:
            d_w = delta_s2[lo:hi]
            dp_w = dp[lo:hi]
            dot_pp = np.dot(dp_w, dp_w)
            alphas_w.append(float(np.dot(d_w, dp_w) / dot_pp)
                            if dot_pp > 0 else 0.0)
        drift_val, drift_p = compute_drift(alphas_w)
        score = abs(alpha_val - 1) + abs(drift_val) * 50

        refined.append({
            **r,
            'alpha_ref': float(alpha_val),
            'R2_ref': float(R2_val),
            'drift_ref': float(drift_val),
            'drift_p_ref': float(drift_p),
            'score_ref': float(score),
            'window_alphas_ref': [float(x) for x in alphas_w],
        })

    refined.sort(key=lambda x: x['score_ref'])
    s2_time = time.time() - t_s2
    print(f"  Done in {s2_time:.0f}s")

    print(f"\n{'='*130}")
    print(f"STAGE 2 — TOP 30 REFINED")
    print(f"{'='*130}")
    print(f"{'Rk':>3} {'a':>10} {'b':>10} {'c':>10} "
          f"{'alpha':>10} {'R2':>8} {'drift':>10} {'score':>10}  Formula")
    print("-" * 130)
    for i, r in enumerate(refined[:30], 1):
        a_s = describe_val(r['a'])
        b_s = describe_val(r['b'])
        c_s = describe_val(r['c'])
        formula = f"{a_s} − {b_s}/logT"
        if abs(r['c']) > 1e-10:
            formula += f" + {c_s}/log²T" if r['c'] > 0 else f" − {describe_val(-r['c'])}/log²T"
        else:
            formula += "  [2D]"
        print(f"{i:>3} {r['a']:>10.6f} {r['b']:>10.6f} {r['c']:>+10.4f} "
              f"{r['alpha_ref']:>+10.6f} {r['R2_ref']:>8.4f} "
              f"{r['drift_ref']:>+10.6f} {r['score_ref']:>10.6f}  {formula}")

    # ============================================================
    # STAGE 3: FULL 2M VALIDATION (top 10)
    # ============================================================
    CHUNK = 100_000
    N_VAL = 10
    print(f"\n{'='*70}")
    print(f"STAGE 3 — FULL {N_FULL:,} VALIDATION (top {N_VAL})")
    print(f"  {len(primes_all):,} primes, chunks of {CHUNK:,}")
    print(f"{'='*70}")

    # Window bounds for 2M
    n_windows = 6
    w_size = N_FULL // n_windows
    WINDOWS_2M = [(i * w_size, min((i+1) * w_size, N_FULL))
                   for i in range(n_windows)]

    validated = []
    for v_idx, cand in enumerate(refined[:N_VAL]):
        a_val, b_val, c_val = cand['a'], cand['b'], cand['c']
        name = f"{describe_val(a_val)} − {describe_val(b_val)}/logT"
        if abs(c_val) > 1e-10:
            name += f" + {describe_val(c_val)}/log²T" if c_val > 0 \
                else f" − {describe_val(-c_val)}/log²T"

        print(f"\n  [{v_idx+1}/{N_VAL}] {name}")
        print(f"    a={a_val:.8f}  b={b_val:.8f}  c={c_val:.8f}")

        dp_full = np.zeros(N_FULL)
        t1 = time.time()
        for i in range(0, N_FULL, CHUNK):
            j = min(i + CHUNK, N_FULL)
            dp_full[i:j] = prime_sum_var(
                gamma0_all[i:j], tp_all[i:j], primes_all, K_MAX,
                a_val, -b_val, c_val)
            if j % (CHUNK * 4) == 0 or j >= N_FULL:
                pct = 100 * j / N_FULL
                el = time.time() - t1
                eta = el / j * (N_FULL - j) if j > 0 else 0
                print(f"      [{j:>9,}/{N_FULL:,}] {pct:5.1f}%  "
                      f"[{el/60:.1f}m, ETA {eta/60:.1f}m]", flush=True)
        compute_time = time.time() - t1

        # Metrics
        alpha_full, R2_full = compute_alpha_R2(delta_all, dp_full)

        # 6-window drift
        alphas_w = []
        for lo, hi in WINDOWS_2M:
            d_w = delta_all[lo:hi]
            dp_w = dp_full[lo:hi]
            dot_pp = np.dot(dp_w, dp_w)
            alphas_w.append(float(np.dot(d_w, dp_w) / dot_pp)
                            if dot_pp > 0 else 0.0)
        drift_slope, drift_p = compute_drift(alphas_w)

        # Monotonicity check
        diffs = [alphas_w[i+1] - alphas_w[i] for i in range(len(alphas_w)-1)]
        monotone_down = all(d < 0 for d in diffs)
        monotone_up = all(d > 0 for d in diffs)

        # Bootstrap CI for alpha
        B_BOOT = 2000
        np.random.seed(42)
        alpha_boots = np.empty(B_BOOT)
        for b_idx in range(B_BOOT):
            idx = np.random.randint(0, N_FULL, N_FULL)
            d_b = delta_all[idx]; dp_b = dp_full[idx]
            dot_pp = np.dot(dp_b, dp_b)
            alpha_boots[b_idx] = np.dot(d_b, dp_b) / dot_pp if dot_pp > 0 else 0.0
        ci_lo = float(np.percentile(alpha_boots, 2.5))
        ci_hi = float(np.percentile(alpha_boots, 97.5))
        T7 = ci_lo <= 1.0 <= ci_hi
        T8 = drift_p > 0.05

        print(f"    α(2M)  = {alpha_full:+.6f}   |α−1| = {abs(alpha_full-1):.6f}")
        print(f"    R²(2M) = {R2_full:.6f}")
        print(f"    drift  = {drift_slope:+.6f}  (p={drift_p:.4f})")
        print(f"    windows = {[f'{a:.4f}' for a in alphas_w]}")
        print(f"    {'↘ MONOTONE DOWN' if monotone_down else '↗ MONOTONE UP' if monotone_up else '~ non-monotone'}")
        print(f"    T7: {'PASS' if T7 else 'FAIL'}  CI=[{ci_lo:.6f}, {ci_hi:.6f}]")
        print(f"    T8: {'PASS' if T8 else 'FAIL'}  (p={drift_p:.4f})")
        print(f"    time: {compute_time:.0f}s")

        validated.append({
            'name': name,
            'a': float(a_val), 'b': float(b_val), 'c': float(c_val),
            'a_desc': describe_val(a_val),
            'b_desc': describe_val(b_val),
            'c_desc': describe_val(c_val),
            'alpha_50k': float(cand['alpha_ref']),
            'score_50k': float(cand['score_ref']),
            'alpha_2M': float(alpha_full),
            'abs_alpha_minus_1': float(abs(alpha_full - 1)),
            'R2_2M': float(R2_full),
            'drift_slope': float(drift_slope),
            'drift_p': float(drift_p),
            'window_alphas': [float(x) for x in alphas_w],
            'monotone_down': bool(monotone_down),
            'T7_pass': bool(T7),
            'T7_ci_lo': float(ci_lo),
            'T7_ci_hi': float(ci_hi),
            'T8_pass': bool(T8),
            'compute_time_s': float(compute_time),
        })

    # ============================================================
    # SAVE
    # ============================================================
    total_time = time.time() - t_global

    output = {
        'metadata': {
            'date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'strategy': '3-stage: fast screen → medium refine → 2M validate',
            'pools': {
                'a': len(a_vals), 'b': len(b_vals), 'c': len(c_vals),
                'total': total,
            },
            'primes': {'fast': len(primes_fast), 'med': len(primes_med), 'full': len(primes_all)},
            'N_screen': N_SCREEN,
            'N_full': N_FULL,
            'stage1_time_min': round(s1_time / 60, 1),
            'stage2_time_min': round(s2_time / 60, 1),
            'total_time_min': round(total_time / 60, 1),
        },
        'stage1_top50': top_results[:50],
        'stage2_refined_top50': [{k: v for k, v in r.items()
                                   if k not in ('window_alphas',)}
                                  for r in refined[:50]],
        'stage3_validated': validated,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUTPUT_FILE}")

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print(f"\n{'='*80}")
    print(f"EXPLORATION COMPLETE — {total_time/60:.1f} min total")
    print(f"  Stage 1: {total:,} triplets screened in {s1_time/60:.1f} min")
    print(f"  Stage 2: 500 refined in {s2_time/60:.1f} min")
    print(f"  Stage 3: {N_VAL} validated on {N_FULL:,} zeros")
    print(f"{'='*80}")

    for i, r in enumerate(validated, 1):
        tag = '3D' if abs(r['c']) > 1e-10 else '2D'
        print(f"\n  #{i} [{tag}] {r['name']}")
        print(f"    α(50k)={r['alpha_50k']:+.6f}  →  α(2M)={r['alpha_2M']:+.6f}")
        print(f"    drift={r['drift_slope']:+.6f} (p={r['drift_p']:.4f})  "
              f"{'↘' if r['monotone_down'] else '~'}")
        print(f"    T7={'PASS' if r['T7_pass'] else 'FAIL'}  "
              f"T8={'PASS' if r['T8_pass'] else 'FAIL'}")

    # Key question
    has_3d = [r for r in validated if abs(r['c']) > 1e-10]
    has_2d = [r for r in validated if abs(r['c']) < 1e-10]
    if has_3d and has_2d:
        best_3d = min(r['abs_alpha_minus_1'] for r in has_3d)
        best_2d = min(r['abs_alpha_minus_1'] for r in has_2d)
        print(f"\n  KEY: best |α−1| 3D = {best_3d:.6f}  vs  2D = {best_2d:.6f}")
        if best_3d < best_2d:
            print(f"  → 3D IMPROVES over 2D on 2M zeros!")
        else:
            print(f"  → 2D still better. c/log²T does not help.")

    print(f"\n{'='*80}")


if __name__ == '__main__':
    main()
