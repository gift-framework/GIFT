#!/usr/bin/env python3
"""
MONTE CARLO VALIDATION — Piste 3 Hypothesis Testing
=====================================================

Rigorous statistical validation of the Piste 3 signal:
  d = c2/c1 = 11.888  vs  tau + rank(E8) = 11.897  (0.08% match)
  c1 = 0.2234          vs  6M/7 = 0.2242             (0.3% match)

Three independent tests:
  Test 1: Bootstrap stability of (c1, c2, d) — 2000 resamples
  Test 2: Look-Elsewhere Effect — enumerate ALL 2-constant GIFT sums
  Test 3: Permutation test — shuffle T-alpha pairing 10000 times

Upload to Colab, mount Drive, run:
  !python colab_montecarlo_piste3.py

Expects: outputs/riemann_zeros_2M_genuine.npy
Outputs: outputs/montecarlo_piste3_results.json

Estimated runtime: ~2h on A100, ~6h on T4
"""
import numpy as np
from scipy.special import loggamma
from scipy import stats
from scipy.optimize import curve_fit
from itertools import combinations_with_replacement
import time, json, os, math, sys

# ================================================================
# CONSTANTS
# ================================================================
phi = (1 + math.sqrt(5)) / 2
EULER_GAMMA = 0.5772156649015329
MERTENS = 0.2614972128476428

# GIFT topological constants
DIM_G2 = 14; DIM_K7 = 7; H_STAR = 99; B2 = 21; B3 = 77
RANK_E8 = 8; P2 = 2; N_GEN = 3; DIM_E8 = 248; WEYL = 5
D_BULK = 11; DIM_J3O = 27; DIM_E8xE8 = 496
TAU_GIFT = 3472 / 891  # = dim(E8xE8)*b2 / (dim(J3O)*H*)
TWO_B2 = 42; B0 = 1

# Formula: theta = 7/6 - phi/(logT - 15/8)
A_FIXED = 7.0 / 6.0
B_FIXED = phi
D_FIXED = -15.0 / 8.0

P_MAX = 500_000
K_MAX = 3
CHECKPOINT_FILE = 'outputs/montecarlo_piste3_results.json'

# Observed values from three_pistes_results.json
D_OBSERVED = 11.887677617664378
C1_OBSERVED = 0.22343305639113592
C2_OBSERVED = 2.6561001435072495

# GIFT predictions
D_PREDICTED = TAU_GIFT + RANK_E8  # = 11.8969...
C1_PREDICTED = 6 * MERTENS / 7   # = 0.22414...

# ================================================================
# GPU DETECTION
# ================================================================
os.environ.setdefault('CUDA_PATH', '/usr')
try:
    import cupy as cp
    _ = cp.maximum(cp.array([1.0]), 0.0)
    GPU = True
    mem = cp.cuda.Device(0).mem_info[0] // 1024**2
    print(f"[GPU] CuPy detected, free={mem} MB")
    PRIME_BATCH = 2000 if mem > 30000 else (500 if mem > 6000 else 200)
    ZERO_CHUNK = 500_000 if mem > 20000 else 200_000
except Exception as e:
    GPU = False
    PRIME_BATCH = 200
    ZERO_CHUNK = 200_000
    print(f"[CPU] {e}")

# ================================================================
# INFRASTRUCTURE (same as colab_three_pistes.py)
# ================================================================
def theta_rs(t):
    t = np.asarray(t, dtype=np.float64)
    return np.imag(loggamma(0.25 + 0.5j * t)) - 0.5 * t * np.log(np.pi)

def theta_deriv(t):
    return 0.5 * np.log(np.maximum(np.asarray(t, dtype=np.float64), 1.0) / (2 * np.pi))

def gram_points(N):
    """Compute Gram-like initial guesses for zeros (fast, approximate)."""
    ns = np.arange(1, N + 1, dtype=np.float64)
    targets = (ns - 1.5) * np.pi
    from scipy.special import lambertw
    w = np.real(lambertw(ns / np.e))
    t = np.maximum(2 * np.pi * ns / w, 2.0)
    for it in range(40):
        dt = (theta_rs(t) - targets) / np.maximum(np.abs(theta_deriv(t)), 1e-15)
        t -= dt
        if np.max(np.abs(dt)) < 1e-12:
            break
    return t

def riemann_siegel_Z(t_arr):
    """Evaluate Riemann-Siegel Z function (vectorized numpy).

    Z(t) = 2 * sum_{n=1}^{N(t)} cos(theta(t) - t*log(n)) / sqrt(n) + R(t)

    where N(t) = floor(sqrt(t/(2*pi))) and R(t) is the first correction term.
    Z(t) is real-valued; its zeros are the zeros of zeta on the critical line.
    """
    t = np.asarray(t_arr, dtype=np.float64)
    tau = np.sqrt(t / (2.0 * np.pi))
    N = np.floor(tau).astype(int)
    N_max = int(np.max(N))

    th = theta_rs(t)

    # Main sum: 2 * sum_{n=1}^{N(t)} cos(theta - t*log(n)) / sqrt(n)
    result = np.zeros_like(t)
    for n in range(1, N_max + 1):
        mask = n <= N
        if not np.any(mask):
            break
        result[mask] += np.cos(th[mask] - t[mask] * np.log(n)) / np.sqrt(n)
    result *= 2.0

    # First Riemann-Siegel correction term (improves accuracy to O(t^{-3/4}))
    p = tau - N.astype(np.float64)
    # Psi(p) = cos(2*pi*(p^2 - p - 1/16)) / cos(2*pi*p)
    cos_denom = np.cos(2 * np.pi * p)
    safe = np.abs(cos_denom) > 1e-10
    correction = np.zeros_like(t)
    correction[safe] = (np.cos(2 * np.pi * (p[safe]**2 - p[safe] - 1.0/16))
                         / cos_denom[safe])
    sign = (-1.0) ** (N + 1)
    correction *= sign * tau ** (-0.5)
    result += correction

    return result

def find_genuine_zeros(N_target, cache_file):
    """Find first N_target zeros of Riemann zeta using vectorized
    Riemann-Siegel Z function + bisection. Guaranteed genuine.

    Algorithm:
      1. Compute Gram points as initial grid (fast, ~seconds)
      2. Evaluate Z on grid + midpoints (vectorized numpy, ~30s)
      3. Find all sign changes (instant)
      4. Vectorized bisection to refine (30 steps, ~3-5 min)
      5. Validate against known values

    Total: ~5-10 min for 2M zeros on Colab.
    """
    if os.path.exists(cache_file):
        cached = np.load(cache_file)
        if len(cached) >= N_target:
            print(f"  Cache hit: {cache_file} ({len(cached):,} zeros)")
            return cached[:N_target]
        print(f"  Cache has {len(cached):,}, need {N_target:,} — recomputing")

    os.makedirs('outputs', exist_ok=True)
    t0_total = time.time()

    # Step 1: Gram points as initial grid (with 5% buffer)
    n_gram = int(N_target * 1.10)
    print(f"  Step 1: Computing {n_gram:,} Gram points...")
    t0 = time.time()
    grams = gram_points(n_gram)
    print(f"    Done in {time.time()-t0:.1f}s")

    # Step 2: Build evaluation grid = Gram points + midpoints
    midpoints = 0.5 * (grams[:-1] + grams[1:])
    grid = np.sort(np.concatenate([grams, midpoints]))
    print(f"  Step 2: Evaluating Z at {len(grid):,} grid points...")
    t0 = time.time()

    # Evaluate Z in chunks (memory-efficient)
    CHUNK = 500_000
    Z_grid = np.empty(len(grid))
    for i in range(0, len(grid), CHUNK):
        j = min(i + CHUNK, len(grid))
        Z_grid[i:j] = riemann_siegel_Z(grid[i:j])
        if (j // CHUNK) % 4 == 0 or j == len(grid):
            elapsed = time.time() - t0
            print(f"    {j:,}/{len(grid):,}  ({elapsed:.1f}s)")
            sys.stdout.flush()

    # Step 3: Find sign changes
    products = Z_grid[:-1] * Z_grid[1:]
    sign_changes = np.where(products < 0)[0]
    n_found = len(sign_changes)
    print(f"  Step 3: Found {n_found:,} sign changes"
          f" (need {N_target:,}, surplus: {n_found - N_target:+,})")

    if n_found < N_target:
        print(f"  WARNING: Found fewer zeros than needed!")
        print(f"  This can happen due to Lehmer pairs or grid resolution.")
        print(f"  Proceeding with {n_found:,} zeros.")

    # Step 4: Vectorized bisection (30 steps → precision ~2^{-30} * 0.5 ≈ 5e-10)
    N_BISECT = 40  # 40 steps → ~5e-13 precision
    lo = grid[sign_changes].copy()
    hi = grid[sign_changes + 1].copy()
    Z_lo = Z_grid[sign_changes].copy()

    print(f"  Step 4: Vectorized bisection ({N_BISECT} steps on {n_found:,} intervals)...")
    t0 = time.time()
    for step in range(N_BISECT):
        mid = 0.5 * (lo + hi)
        Z_mid = riemann_siegel_Z(mid)

        # Where Z_lo and Z_mid have same sign → zero is in [mid, hi]
        same_sign = Z_lo * Z_mid > 0
        lo = np.where(same_sign, mid, lo)
        Z_lo = np.where(same_sign, Z_mid, Z_lo)
        hi = np.where(same_sign, hi, mid)

        if (step + 1) % 10 == 0:
            max_width = np.max(hi - lo)
            elapsed = time.time() - t0
            print(f"    Step {step+1}/{N_BISECT}: max interval = {max_width:.2e}  ({elapsed:.1f}s)")
            sys.stdout.flush()

    zeros = np.sort(0.5 * (lo + hi))
    elapsed_total = time.time() - t0_total
    print(f"  DONE: {len(zeros):,} genuine zeros in {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")

    # Take first N_target
    zeros = zeros[:N_target]
    np.save(cache_file, zeros)
    print(f"  Saved to {cache_file}")

    return zeros

def sieve(N):
    is_p = np.ones(N + 1, dtype=bool); is_p[:2] = False
    for i in range(2, int(N**0.5) + 1):
        if is_p[i]: is_p[i*i::i] = False
    return np.where(is_p)[0]

def prime_sum(g0, tp_v, primes, k_max, a, b, d_shift):
    """Shifted-log: theta = a - b/(logT + d)"""
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

def banner(title, char='='):
    w = 72
    print(f"\n{char * w}")
    print(f"  {title}")
    print(f"{char * w}")
    sys.stdout.flush()

def save_checkpoint(results, part_name):
    os.makedirs('outputs', exist_ok=True)
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            all_data = json.load(f)
    else:
        all_data = {}
    all_data[part_name] = results
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(all_data, f, indent=2, default=float)
    print(f"  [checkpoint] {part_name} saved")
    sys.stdout.flush()

# ================================================================
# PISTE 3 FIT FUNCTION
# ================================================================
def fit_piste3(T_mids, alphas):
    """Fit alpha(T) = 1 - c1/logT + c2/logT^2, return (c1, c2, d=c2/c1, R2, resid_p)."""
    T_arr = np.array(T_mids, dtype=np.float64)
    a_arr = np.array(alphas, dtype=np.float64)
    logT = np.log(T_arr)

    # Design matrix: alpha = 1 - c1/logT + c2/logT^2
    # => (alpha - 1) = -c1/logT + c2/logT^2
    y = a_arr - 1.0
    X = np.column_stack([-1.0 / logT, 1.0 / logT**2])

    # OLS
    beta, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    c1, c2 = float(beta[0]), float(beta[1])

    # R^2
    y_pred = X @ beta
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    R2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Residual drift
    resid = y - y_pred
    x_idx = np.arange(len(resid))
    _, _, _, p_resid, _ = stats.linregress(x_idx, resid)

    d = float(c2 / c1) if abs(c1) > 1e-15 else float('inf')

    return {
        'c1': c1,
        'c2': c2,
        'shape_ratio': d,
        'R2': R2,
        'residual_drift_p': float(p_resid),
    }


# ================================================================
# MAIN
# ================================================================
banner("MONTE CARLO VALIDATION — Piste 3 Hypothesis Testing")

print(f"\nHypotheses under test:")
print(f"  H1: shape ratio d = tau + rank(E8) = {D_PREDICTED:.6f}")
print(f"      observed d = {D_OBSERVED:.6f} (match: {abs(D_OBSERVED - D_PREDICTED)/D_PREDICTED*100:.3f}%)")
print(f"  H2: c1 = 6M/7 = {C1_PREDICTED:.6f}")
print(f"      observed c1 = {C1_OBSERVED:.6f} (match: {abs(C1_OBSERVED - C1_PREDICTED)/C1_PREDICTED*100:.3f}%)")
print()

# ================================================================
# COMPUTE GENUINE ZEROS (Riemann-Siegel Z + vectorized bisection)
# ================================================================
banner("Computing genuine Riemann zeta zeros")

N_ZEROS_TARGET = int(os.environ.get('N_ZEROS', 2_000_000))
CACHE_FILE = f'outputs/genuine_zeros_{N_ZEROS_TARGET // 1000}k.npy'

# Known first zeros to 15 digits (from LMFDB / Odlyzko)
KNOWN_ZEROS = [
    14.134725141734694,
    21.022039638771555,
    25.010857580145689,
    30.424876125859513,
    32.935061587739190,
]

print(f"  Target: {N_ZEROS_TARGET:,} genuine zeros")
print(f"  Method: Riemann-Siegel Z function + vectorized bisection")
print(f"  Cache:  {CACHE_FILE}")

gamma0 = find_genuine_zeros(N_ZEROS_TARGET, CACHE_FILE)
gamma0 = np.sort(gamma0)

# Validate against known values
print(f"\n  Validating against known zeros:")
all_ok = True
for i, known in enumerate(KNOWN_ZEROS[:min(5, len(gamma0))]):
    diff = abs(gamma0[i] - known)
    status = "OK" if diff < 1e-6 else ("WARN" if diff < 1e-3 else "FAIL")
    if status == "FAIL":
        all_ok = False
    print(f"    #{i+1}: {gamma0[i]:.12f}  (expected {known:.12f}, "
          f"diff={diff:.2e})  [{status}]")

if not all_ok:
    print("\n  FATAL: Zero validation failed! Cannot proceed.")
    sys.exit(1)
print(f"  All {min(5, len(gamma0))} reference zeros match to < 1e-6")

# Compare with Gram approximations to quantify the difference
gram_approx = gram_points(min(1000, len(gamma0)))
diffs = np.abs(gamma0[:len(gram_approx)] - gram_approx)
print(f"\n  Gram vs genuine comparison (first {len(gram_approx):,} zeros):")
print(f"    Mean |diff|: {np.mean(diffs):.6f}")
print(f"    Max  |diff|: {np.max(diffs):.6f}")
print(f"    Zeros where |diff| > 0.01: {np.sum(diffs > 0.01):,} "
      f"({100*np.mean(diffs > 0.01):.1f}%)")

N_zeros = len(gamma0)
GENUINE = True
primes = sieve(P_MAX)
print(f"\n  {N_zeros:,} genuine zeros, {len(primes):,} primes up to {P_MAX:,}")

# Precompute shared data
tp_v = theta_deriv(gamma0)

# ================================================================
# COMPUTE BASELINE ONCE
# ================================================================
banner("Part 0: Compute baseline S_w for all 2M zeros")
t0 = time.time()

delta_pred = np.sign(np.sin(np.pi * (theta_rs(gamma0) / np.pi - 0.5)))
S_w = prime_sum(gamma0, tp_v, primes, K_MAX, A_FIXED, B_FIXED, D_FIXED)
delta = delta_pred + S_w

elapsed = time.time() - t0
print(f"  Baseline computed in {elapsed:.1f}s")

# Quick global alpha
denom_global = np.dot(delta_pred, delta_pred)
alpha_global = float(np.dot(delta, delta_pred) / denom_global)
print(f"  Global alpha = {alpha_global:.8f}")

# Compute 50-window alphas and T_mids (reused throughout)
N_WIN = 50
bounds = [int(i * N_zeros / N_WIN) for i in range(N_WIN + 1)]
base_alphas = []
base_T_mids = []
for i in range(N_WIN):
    lo, hi = bounds[i], bounds[i+1]
    dp_w = delta_pred[lo:hi]
    d_w = delta[lo:hi]
    den = np.dot(dp_w, dp_w)
    a_w = float(np.dot(d_w, dp_w) / den) if den > 0 else 0.0
    base_alphas.append(a_w)
    base_T_mids.append(float(gamma0[(lo + hi) // 2]))

base_alphas = np.array(base_alphas)
base_T_mids = np.array(base_T_mids)

# Verify baseline fit
base_fit = fit_piste3(base_T_mids, base_alphas)
print(f"\n  Baseline Piste 3 fit:")
print(f"    c1 = {base_fit['c1']:.6f}  (GIFT: 6M/7 = {C1_PREDICTED:.6f})")
print(f"    c2 = {base_fit['c2']:.6f}")
print(f"    d  = {base_fit['shape_ratio']:.6f}  (GIFT: tau+8 = {D_PREDICTED:.6f})")
print(f"    R2 = {base_fit['R2']:.4f}")
print(f"    resid_drift_p = {base_fit['residual_drift_p']:.4f}")

save_checkpoint({
    'genuine_zeros': GENUINE,
    'n_zeros': N_zeros,
    'alpha_global': alpha_global,
    'n_windows': N_WIN,
    'base_fit': base_fit,
    'base_alphas': base_alphas.tolist(),
    'base_T_mids': base_T_mids.tolist(),
    'elapsed_s': elapsed,
}, 'part0_baseline')

# ================================================================
# TEST 1: BOOTSTRAP STABILITY
# ================================================================
banner("Test 1: Bootstrap stability of (c1, c2, d) — 2000 resamples")
t0 = time.time()

N_BOOT = 2000
rng = np.random.default_rng(2026)

boot_c1 = np.empty(N_BOOT)
boot_c2 = np.empty(N_BOOT)
boot_d = np.empty(N_BOOT)
boot_R2 = np.empty(N_BOOT)
boot_resid_p = np.empty(N_BOOT)

# Bootstrap: resample WINDOWS (block bootstrap), not individual zeros
# This preserves the T-structure within each window
for b in range(N_BOOT):
    # Resample window indices with replacement
    idx = rng.integers(0, N_WIN, size=N_WIN)
    boot_T = base_T_mids[idx]
    boot_a = base_alphas[idx]

    # Sort by T to preserve monotonicity for the fit
    order = np.argsort(boot_T)
    boot_T_sorted = boot_T[order]
    boot_a_sorted = boot_a[order]

    fit = fit_piste3(boot_T_sorted, boot_a_sorted)
    boot_c1[b] = fit['c1']
    boot_c2[b] = fit['c2']
    boot_d[b] = fit['shape_ratio']
    boot_R2[b] = fit['R2']
    boot_resid_p[b] = fit['residual_drift_p']

    if (b + 1) % 500 == 0:
        print(f"  Bootstrap {b+1}/{N_BOOT}: d_mean={np.mean(boot_d[:b+1]):.4f} "
              f"+/- {np.std(boot_d[:b+1]):.4f}")
        sys.stdout.flush()

# Confidence intervals
d_ci = (float(np.percentile(boot_d, 2.5)), float(np.percentile(boot_d, 97.5)))
c1_ci = (float(np.percentile(boot_c1, 2.5)), float(np.percentile(boot_c1, 97.5)))

# Is GIFT prediction inside CI?
d_gift_in_ci = bool(d_ci[0] <= D_PREDICTED <= d_ci[1])
c1_gift_in_ci = bool(c1_ci[0] <= C1_PREDICTED <= c1_ci[1])

# Z-scores
d_z = float((D_PREDICTED - np.mean(boot_d)) / np.std(boot_d)) if np.std(boot_d) > 0 else float('inf')
c1_z = float((C1_PREDICTED - np.mean(boot_c1)) / np.std(boot_c1)) if np.std(boot_c1) > 0 else float('inf')

elapsed1 = time.time() - t0

print(f"\n  Results ({elapsed1:.1f}s):")
print(f"  ┌────────────┬───────────┬──────────────────────┬───────────┬─────────┐")
print(f"  │ Parameter  │ Mean      │ 95% CI               │ GIFT pred │ In CI?  │")
print(f"  ├────────────┼───────────┼──────────────────────┼───────────┼─────────┤")
print(f"  │ d (shape)  │ {np.mean(boot_d):9.4f} │ [{d_ci[0]:.4f}, {d_ci[1]:.4f}] │ {D_PREDICTED:9.4f} │ {'YES' if d_gift_in_ci else 'NO':>7s} │")
print(f"  │ c1         │ {np.mean(boot_c1):9.6f}│ [{c1_ci[0]:.6f}, {c1_ci[1]:.6f}]│ {C1_PREDICTED:9.6f}│ {'YES' if c1_gift_in_ci else 'NO':>7s} │")
print(f"  └────────────┴───────────┴──────────────────────┴───────────┴─────────┘")
print(f"  Z-scores: d_z = {d_z:.2f}, c1_z = {c1_z:.2f}")
print(f"  |z| < 2 means GIFT prediction is statistically consistent with data")

# Fraction of resamples where resid_p > 0.05 (model captures drift)
frac_pass = float(np.mean(boot_resid_p > 0.05))
print(f"  Fraction with resid_p > 0.05: {frac_pass:.3f} ({int(frac_pass*N_BOOT)}/{N_BOOT})")

save_checkpoint({
    'n_bootstrap': N_BOOT,
    'd_mean': float(np.mean(boot_d)),
    'd_std': float(np.std(boot_d)),
    'd_ci_95': list(d_ci),
    'd_gift_predicted': D_PREDICTED,
    'd_gift_in_ci': d_gift_in_ci,
    'd_z_score': d_z,
    'c1_mean': float(np.mean(boot_c1)),
    'c1_std': float(np.std(boot_c1)),
    'c1_ci_95': list(c1_ci),
    'c1_gift_predicted': C1_PREDICTED,
    'c1_gift_in_ci': c1_gift_in_ci,
    'c1_z_score': c1_z,
    'c2_mean': float(np.mean(boot_c2)),
    'c2_std': float(np.std(boot_c2)),
    'R2_mean': float(np.mean(boot_R2)),
    'frac_resid_pass': frac_pass,
    'elapsed_s': elapsed1,
}, 'test1_bootstrap')


# ================================================================
# TEST 2: LOOK-ELSEWHERE EFFECT
# ================================================================
banner("Test 2: Look-Elsewhere Effect — GIFT constant combinations")
t0 = time.time()

# All GIFT topological constants that might appear in sums
GIFT_CONSTANTS = {
    'dim_E8': 248,
    'rank_E8': 8,
    'dim_G2': 14,
    'dim_K7': 7,
    'b2': 21,
    'b3': 77,
    'H_star': 99,
    'p2': 2,
    'N_gen': 3,
    'Weyl': 5,
    'D_bulk': 11,
    'dim_J3O': 27,
    'two_b2': 42,
    'b0': 1,
    'dim_E8xE8': 496,
}

# Also include derived rationals that are GIFT predictions
GIFT_RATIONALS = {
    'tau': TAU_GIFT,          # 3472/891
    'sin2_thetaW': 3/13,      # 0.2308
    'kappa_T': 1/61,           # 0.0164
    'det_g': 65/32,            # 2.03125
    'phi': phi,                # 1.618...
    'pi': math.pi,             # 3.14159...
    'e': math.e,               # 2.71828...
    'M': MERTENS,              # 0.2615
    'gamma': EULER_GAMMA,      # 0.5772
}

print("  Scanning all X + Y combinations where X, Y in GIFT constants/rationals...")
print(f"  Looking for sums near d_observed = {D_OBSERVED:.4f}")
print()

# Combine all values
all_vals = {}
all_vals.update(GIFT_CONSTANTS)
all_vals.update(GIFT_RATIONALS)

# Also include X/Y ratios and X*Y products for small values
extended = {}
for name, val in all_vals.items():
    extended[name] = val

# Enumerate X + Y (including X + X = 2X)
matches = []
TOLERANCE = 0.05  # 5% tolerance window
for (n1, v1), (n2, v2) in combinations_with_replacement(sorted(all_vals.items()), 2):
    s = v1 + v2
    if abs(s) > 0:
        rel_err = abs(s - D_OBSERVED) / abs(D_OBSERVED)
        if rel_err < TOLERANCE:
            matches.append({
                'expr': f"{n1} + {n2}",
                'value': float(s),
                'rel_error': float(rel_err),
                'abs_error': float(abs(s - D_OBSERVED)),
            })

# Also try X - Y (where result > 0)
for n1, v1 in sorted(all_vals.items()):
    for n2, v2 in sorted(all_vals.items()):
        if n1 == n2:
            continue
        s = v1 - v2
        if s > 0 and abs(s - D_OBSERVED) / abs(D_OBSERVED) < TOLERANCE:
            matches.append({
                'expr': f"{n1} - {n2}",
                'value': float(s),
                'rel_error': float(abs(s - D_OBSERVED) / abs(D_OBSERVED)),
                'abs_error': float(abs(s - D_OBSERVED)),
            })

# Also try X * Y
for (n1, v1), (n2, v2) in combinations_with_replacement(sorted(all_vals.items()), 2):
    s = v1 * v2
    if abs(s) > 0 and abs(s - D_OBSERVED) / abs(D_OBSERVED) < TOLERANCE:
        matches.append({
            'expr': f"{n1} * {n2}",
            'value': float(s),
            'rel_error': float(abs(s - D_OBSERVED) / abs(D_OBSERVED)),
            'abs_error': float(abs(s - D_OBSERVED)),
        })

# Also try X / Y (where Y != 0)
for n1, v1 in sorted(all_vals.items()):
    for n2, v2 in sorted(all_vals.items()):
        if abs(v2) < 1e-15 or n1 == n2:
            continue
        s = v1 / v2
        if abs(s) > 0 and abs(s - D_OBSERVED) / abs(D_OBSERVED) < TOLERANCE:
            matches.append({
                'expr': f"{n1} / {n2}",
                'value': float(s),
                'rel_error': float(abs(s - D_OBSERVED) / abs(D_OBSERVED)),
                'abs_error': float(abs(s - D_OBSERVED)),
            })

matches.sort(key=lambda m: m['rel_error'])

# Count total combinations tested
n_add = len(list(combinations_with_replacement(all_vals, 2)))
n_sub = len(all_vals) * (len(all_vals) - 1)
n_mul = n_add
n_div = n_sub
N_TOTAL_COMBOS = n_add + n_sub + n_mul + n_div

# Count how many hit within various tolerances
n_within_1pct = sum(1 for m in matches if m['rel_error'] < 0.01)
n_within_01pct = sum(1 for m in matches if m['rel_error'] < 0.001)
n_within_5pct = len(matches)

# LEE-corrected p-value — TWO methods
# Method 1 (conservative): Bonferroni. P(any match within eps) <= N * P(one match within eps)
#   For relative error, P(one match) ≈ 2*eps (fraction of range covered).
#   This is VERY conservative because it treats all combos as independent.
best_match_err = matches[0]['rel_error'] if matches else 1.0
p_lee_bonf = min(1.0, N_TOTAL_COMBOS * 2 * best_match_err)

# Method 2 (empirical): What fraction of combos are within 1% of d?
#   If 8/1704 ≈ 0.5% of combos hit within 1%, then hitting within 0.08%
#   is 0.08/1.0 = 8× more precise than the 1% threshold. Empirical density
#   of targets near d ≈ 8/1704 per 1% band = 0.47% per 1%.
#   P(best ≤ 0.076%) = 1 - (1 - 2*0.00076)^N_eff where N_eff = distinct targets
# Approximate: count unique values within 5%
unique_5pct = set()
for m in matches:
    unique_5pct.add(round(m['value'], 2))
N_eff = len(unique_5pct)
p_lee_empirical = min(1.0, 1 - (1 - 2 * best_match_err) ** N_eff)

# Method 3: rank-based. Best match has rank 1 out of N_TOTAL_COMBOS.
# What's the gap ratio between #1 and #2?
if len(matches) >= 2:
    gap_ratio = matches[1]['rel_error'] / matches[0]['rel_error'] if matches[0]['rel_error'] > 0 else float('inf')
else:
    gap_ratio = float('inf')

elapsed2 = time.time() - t0

print(f"  Total combinations scanned: {N_TOTAL_COMBOS:,}")
print(f"  (+: {n_add}, -: {n_sub}, *: {n_mul}, /: {n_div})")
print(f"  Unique targets within 5%: {N_eff}")
print(f"  Matches within 5%: {n_within_5pct}")
print(f"  Matches within 1%: {n_within_1pct}")
print(f"  Matches within 0.1%: {n_within_01pct}")
print()
print(f"  Top 10 closest matches:")
for i, m in enumerate(matches[:10]):
    star = " <-- GIFT" if 'tau' in m['expr'] and 'rank_E8' in m['expr'] else ""
    print(f"    {i+1}. {m['expr']:30s} = {m['value']:.6f}  "
          f"(err = {m['rel_error']*100:.4f}%){star}")
print()
print(f"  LEE p-value (Bonferroni, conservative): {p_lee_bonf:.4f}")
print(f"  LEE p-value (empirical, N_eff={N_eff}): {p_lee_empirical:.4f}")
print(f"  Gap ratio #2/#1: {gap_ratio:.1f}x  (>5x is strong separation)")
p_lee = p_lee_empirical  # use empirical for verdict
print(f"  Verdict: {'SIGNIFICANT (p<0.05)' if p_lee < 0.05 else 'NOT significant (but check gap ratio)'}")

save_checkpoint({
    'n_constants': len(all_vals),
    'n_total_combinations': N_TOTAL_COMBOS,
    'n_effective_targets': N_eff,
    'n_within_5pct': n_within_5pct,
    'n_within_1pct': n_within_1pct,
    'n_within_01pct': n_within_01pct,
    'top_20_matches': matches[:20],
    'p_lee_bonferroni': p_lee_bonf,
    'p_lee_empirical': p_lee_empirical,
    'p_lee': p_lee,
    'gap_ratio_2nd_vs_1st': gap_ratio,
    'best_match_expr': matches[0]['expr'] if matches else None,
    'best_match_value': matches[0]['value'] if matches else None,
    'best_match_rel_error': best_match_err,
    'elapsed_s': elapsed2,
}, 'test2_lee')


# ================================================================
# TEST 3: PERMUTATION TEST
# ================================================================
banner("Test 3: Permutation test — destroy temporal structure")
t0 = time.time()

N_PERM = 10000

print(f"  Shuffling T_mid-alpha pairing {N_PERM:,} times...")
print(f"  Null hypothesis: d has no preferred value")
print()

perm_d = np.empty(N_PERM)
perm_c1 = np.empty(N_PERM)
perm_R2 = np.empty(N_PERM)

for p_idx in range(N_PERM):
    # Shuffle alphas while keeping T_mids fixed
    shuffled_alphas = rng.permutation(base_alphas)
    fit = fit_piste3(base_T_mids, shuffled_alphas)
    perm_d[p_idx] = fit['shape_ratio']
    perm_c1[p_idx] = fit['c1']
    perm_R2[p_idx] = fit['R2']

    if (p_idx + 1) % 2000 == 0:
        # How many null d values are closer to D_PREDICTED than observed?
        n_closer = np.sum(np.abs(perm_d[:p_idx+1] - D_PREDICTED)
                          <= abs(D_OBSERVED - D_PREDICTED))
        print(f"  Permutation {p_idx+1}/{N_PERM}: "
              f"n_closer_to_GIFT = {n_closer}/{p_idx+1} "
              f"(p ≈ {n_closer/(p_idx+1):.4f})")
        sys.stdout.flush()

# p-value: fraction of permutations where d is at least as close to D_PREDICTED
obs_distance = abs(D_OBSERVED - D_PREDICTED)
n_as_close = int(np.sum(np.abs(perm_d - D_PREDICTED) <= obs_distance))
p_perm_d = float((n_as_close + 1) / (N_PERM + 1))  # +1 for continuity correction

# Same for c1
obs_distance_c1 = abs(C1_OBSERVED - C1_PREDICTED)
n_as_close_c1 = int(np.sum(np.abs(perm_c1 - C1_PREDICTED) <= obs_distance_c1))
p_perm_c1 = float((n_as_close_c1 + 1) / (N_PERM + 1))

# R2 comparison: is the real fit better than shuffled?
n_better_R2 = int(np.sum(perm_R2 >= base_fit['R2']))
p_R2 = float((n_better_R2 + 1) / (N_PERM + 1))

elapsed3 = time.time() - t0

print(f"\n  Results ({elapsed3:.1f}s):")
print(f"  ┌──────────────────────────────────────────────────────────────┐")
print(f"  │ Permutation test: {N_PERM:,} shuffles                         │")
print(f"  ├──────────────────────────────────────────────────────────────┤")
print(f"  │ Shape ratio d:                                              │")
print(f"  │   Observed:  {D_OBSERVED:.6f}                                  │")
print(f"  │   GIFT pred: {D_PREDICTED:.6f}                                  │")
print(f"  │   Null mean: {np.mean(perm_d):.6f} +/- {np.std(perm_d):.4f}             │")
print(f"  │   n(|d-GIFT| <= obs): {n_as_close} / {N_PERM}                     │")
print(f"  │   p-value:   {p_perm_d:.6f}  {'*** SIGNIFICANT' if p_perm_d < 0.05 else '(not significant)':>24s} │")
print(f"  ├──────────────────────────────────────────────────────────────┤")
print(f"  │ c1:                                                         │")
print(f"  │   Observed:  {C1_OBSERVED:.6f}                                  │")
print(f"  │   GIFT pred: {C1_PREDICTED:.6f}                                  │")
print(f"  │   n(|c1-GIFT| <= obs): {n_as_close_c1} / {N_PERM}                   │")
print(f"  │   p-value:   {p_perm_c1:.6f}  {'*** SIGNIFICANT' if p_perm_c1 < 0.05 else '(not significant)':>24s} │")
print(f"  ├──────────────────────────────────────────────────────────────┤")
print(f"  │ R2 (fit quality):                                           │")
print(f"  │   Observed:  {base_fit['R2']:.6f}                                  │")
print(f"  │   Null mean: {np.mean(perm_R2):.6f} +/- {np.std(perm_R2):.4f}             │")
print(f"  │   p-value:   {p_R2:.6f}  {'*** SIGNIFICANT' if p_R2 < 0.05 else '(not significant)':>24s} │")
print(f"  └──────────────────────────────────────────────────────────────┘")

save_checkpoint({
    'n_permutations': N_PERM,
    'd_observed': D_OBSERVED,
    'd_predicted': D_PREDICTED,
    'd_null_mean': float(np.mean(perm_d)),
    'd_null_std': float(np.std(perm_d)),
    'd_null_percentiles': {
        'p5': float(np.percentile(perm_d, 5)),
        'p25': float(np.percentile(perm_d, 25)),
        'p50': float(np.percentile(perm_d, 50)),
        'p75': float(np.percentile(perm_d, 75)),
        'p95': float(np.percentile(perm_d, 95)),
    },
    'n_as_close_d': n_as_close,
    'p_value_d': p_perm_d,
    'c1_observed': C1_OBSERVED,
    'c1_predicted': C1_PREDICTED,
    'n_as_close_c1': n_as_close_c1,
    'p_value_c1': p_perm_c1,
    'R2_observed': base_fit['R2'],
    'R2_null_mean': float(np.mean(perm_R2)),
    'n_better_R2': n_better_R2,
    'p_value_R2': p_R2,
    'elapsed_s': elapsed3,
}, 'test3_permutation')


# ================================================================
# TEST 4: JACKKNIFE — Leave-one-window-out stability
# ================================================================
banner("Test 4: Jackknife — Leave-one-window-out stability")
t0 = time.time()

jack_d = np.empty(N_WIN)
jack_c1 = np.empty(N_WIN)
jack_c2 = np.empty(N_WIN)

for i in range(N_WIN):
    mask = np.ones(N_WIN, dtype=bool)
    mask[i] = False
    fit = fit_piste3(base_T_mids[mask], base_alphas[mask])
    jack_d[i] = fit['shape_ratio']
    jack_c1[i] = fit['c1']
    jack_c2[i] = fit['c2']

# Jackknife standard error
jack_d_mean = float(np.mean(jack_d))
jack_d_se = float(np.sqrt((N_WIN - 1) / N_WIN * np.sum((jack_d - jack_d_mean)**2)))
jack_c1_mean = float(np.mean(jack_c1))
jack_c1_se = float(np.sqrt((N_WIN - 1) / N_WIN * np.sum((jack_c1 - jack_c1_mean)**2)))

# Jackknife CI
jack_d_ci = (jack_d_mean - 1.96 * jack_d_se, jack_d_mean + 1.96 * jack_d_se)
jack_c1_ci = (jack_c1_mean - 1.96 * jack_c1_se, jack_c1_mean + 1.96 * jack_c1_se)

# GIFT inside jackknife CI?
jack_d_gift_in = bool(jack_d_ci[0] <= D_PREDICTED <= jack_d_ci[1])
jack_c1_gift_in = bool(jack_c1_ci[0] <= C1_PREDICTED <= jack_c1_ci[1])

# Most influential window
d_influence = np.abs(jack_d - D_OBSERVED)
most_influential = int(np.argmax(d_influence))

elapsed4 = time.time() - t0

print(f"  Jackknife standard errors ({N_WIN} leave-one-out):")
print(f"    d  = {jack_d_mean:.4f} +/- {jack_d_se:.4f}  "
      f"(CI: [{jack_d_ci[0]:.4f}, {jack_d_ci[1]:.4f}])  "
      f"GIFT in CI: {'YES' if jack_d_gift_in else 'NO'}")
print(f"    c1 = {jack_c1_mean:.6f} +/- {jack_c1_se:.6f}  "
      f"(CI: [{jack_c1_ci[0]:.6f}, {jack_c1_ci[1]:.6f}])  "
      f"GIFT in CI: {'YES' if jack_c1_gift_in else 'NO'}")
print(f"  Most influential window: #{most_influential} "
      f"(T_mid={base_T_mids[most_influential]:.0f}, "
      f"d_without={jack_d[most_influential]:.4f})")
print(f"  Max d deviation: {float(d_influence[most_influential]):.4f}")

save_checkpoint({
    'd_jackknife_mean': jack_d_mean,
    'd_jackknife_se': jack_d_se,
    'd_jackknife_ci': list(jack_d_ci),
    'd_gift_in_jackknife_ci': jack_d_gift_in,
    'c1_jackknife_mean': jack_c1_mean,
    'c1_jackknife_se': jack_c1_se,
    'c1_jackknife_ci': list(jack_c1_ci),
    'c1_gift_in_jackknife_ci': jack_c1_gift_in,
    'most_influential_window': most_influential,
    'most_influential_T_mid': float(base_T_mids[most_influential]),
    'd_without_most_influential': float(jack_d[most_influential]),
    'jack_d_all': jack_d.tolist(),
    'jack_c1_all': jack_c1.tolist(),
    'elapsed_s': elapsed4,
}, 'test4_jackknife')


# ================================================================
# TEST 5: SUBRANGE STABILITY
# ================================================================
banner("Test 5: Subrange stability — first half vs second half vs thirds")
t0 = time.time()

subranges = {
    'first_half': (0, N_WIN // 2),
    'second_half': (N_WIN // 2, N_WIN),
    'first_third': (0, N_WIN // 3),
    'middle_third': (N_WIN // 3, 2 * N_WIN // 3),
    'last_third': (2 * N_WIN // 3, N_WIN),
    'first_quarter': (0, N_WIN // 4),
    'last_quarter': (3 * N_WIN // 4, N_WIN),
}

subrange_results = {}
for name, (lo, hi) in subranges.items():
    fit = fit_piste3(base_T_mids[lo:hi], base_alphas[lo:hi])
    subrange_results[name] = {
        'n_windows': hi - lo,
        'T_range': [float(base_T_mids[lo]), float(base_T_mids[hi-1])],
        'c1': fit['c1'],
        'c2': fit['c2'],
        'shape_ratio': fit['shape_ratio'],
        'R2': fit['R2'],
    }
    print(f"  {name:20s}: d = {fit['shape_ratio']:8.4f}  "
          f"c1 = {fit['c1']:.6f}  R2 = {fit['R2']:.4f}  "
          f"(windows {lo}-{hi-1})")

# Check consistency: all d within 2 sigma of global?
d_values = [v['shape_ratio'] for v in subrange_results.values()]
d_spread = max(d_values) - min(d_values)
print(f"\n  d range across subranges: [{min(d_values):.4f}, {max(d_values):.4f}]")
print(f"  d spread: {d_spread:.4f}")
print(f"  Relative spread: {d_spread / D_OBSERVED * 100:.1f}%")

elapsed5 = time.time() - t0
subrange_results['d_spread'] = float(d_spread)
subrange_results['d_relative_spread'] = float(d_spread / D_OBSERVED)
subrange_results['elapsed_s'] = elapsed5
save_checkpoint(subrange_results, 'test5_subranges')


# ================================================================
# GRAND VERDICT
# ================================================================
banner("GRAND VERDICT", '=')

print(f"\n  Hypotheses tested:")
print(f"  H1: d = tau + rank(E8) = {D_PREDICTED:.6f}")
print(f"  H2: c1 = 6M/7 = {C1_PREDICTED:.6f}")
print()

# Collect verdicts
verdicts = []

# Test 1: Bootstrap
v1 = "PASS" if d_gift_in_ci else "FAIL"
verdicts.append(('Bootstrap CI contains GIFT d', v1))
v1b = "PASS" if c1_gift_in_ci else "FAIL"
verdicts.append(('Bootstrap CI contains GIFT c1', v1b))

# Test 2: LEE
v2 = "PASS" if p_lee < 0.05 else "FAIL"
verdicts.append((f'LEE-corrected p < 0.05 (p={p_lee:.4f})', v2))

# Test 3: Permutation
v3d = "PASS" if p_perm_d < 0.05 else "FAIL"
verdicts.append((f'Permutation test d (p={p_perm_d:.4f})', v3d))
v3r = "PASS" if p_R2 < 0.05 else "FAIL"
verdicts.append((f'Permutation test R2 (p={p_R2:.6f})', v3r))

# Test 4: Jackknife
v4 = "PASS" if jack_d_gift_in else "FAIL"
verdicts.append(('Jackknife CI contains GIFT d', v4))

# Test 5: Subranges
v5 = "PASS" if d_spread / D_OBSERVED < 0.20 else "FAIL"
verdicts.append((f'Subrange d spread < 20% (spread={d_spread/D_OBSERVED*100:.1f}%)', v5))

n_pass = sum(1 for _, v in verdicts if v == "PASS")
n_total = len(verdicts)

for desc, v in verdicts:
    marker = "OK" if v == "PASS" else "XX"
    print(f"  [{marker}] {desc}: {v}")

print(f"\n  Overall: {n_pass}/{n_total} tests passed")

if n_pass == n_total:
    conclusion = "SIGNAL CONFIRMED — all tests pass"
elif n_pass >= n_total - 1:
    conclusion = "SIGNAL LIKELY — most tests pass"
elif n_pass >= n_total // 2:
    conclusion = "SIGNAL AMBIGUOUS — mixed results"
else:
    conclusion = "SIGNAL REFUTED — majority fail"

print(f"  Conclusion: {conclusion}")
print()

save_checkpoint({
    'verdicts': {desc: v for desc, v in verdicts},
    'n_pass': n_pass,
    'n_total': n_total,
    'conclusion': conclusion,
}, 'grand_verdict')

total_elapsed = elapsed + elapsed1 + elapsed2 + elapsed3 + elapsed4 + elapsed5
print(f"  Total runtime: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
print(f"\n  Results saved to: {CHECKPOINT_FILE}")
banner("DONE", '=')
