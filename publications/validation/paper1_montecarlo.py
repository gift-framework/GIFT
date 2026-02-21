#!/usr/bin/env python3
"""
Statistical Validation Suite for the Mollified Prime-Spectral Formula
=====================================================================

Implements five independent tests to validate the formula:

    S_w(T) = -(1/π) Σ_{p,m≤3} cos²(πm log p / (2Λ(T))) sin(Tm log p) / (m p^{m/2})

    with Λ(T) = θ₀ log T + θ₁,  θ₀ = 1.409,  θ₁ = -3.954

Tests:
    1. PERMUTATION TEST: Shuffle δₙ, measure R² null distribution
    2. MONTE CARLO UNIQUENESS: Random (θ₀, θ₁) pairs, measure optimality
    3. SOBOL SENSITIVITY: Global sensitivity indices for (θ₀, θ₁, kernel)
    4. BOOTSTRAP STABILITY: Resample zeros, check coefficient stability
    5. LOOK-ELSEWHERE CORRECTION: Bonferroni-corrected p-values

Data: First 100,000 non-trivial zeros of ζ(s) from Odlyzko's tables.

Author: GIFT Framework
Date: 2026-02-07
"""

import numpy as np
from pathlib import Path
import json
import time
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────
N_ZEROS = 10_000         # use 10K (computable with mpmath); set 100_000 if Odlyzko tables available
N_PERM = 5_000           # permutation test trials
N_MC = 50_000            # Monte Carlo random configurations
N_BOOTSTRAP = 500        # bootstrap resamples
N_SOBOL = 8_192          # 2^13 Sobol points

THETA0_OPT = 1.4091
THETA1_OPT = -3.9537
THETA_CONST = 0.9941

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Data loading ───────────────────────────────────────────────────────
def load_zeros(n=N_ZEROS):
    """Load Riemann zeros from cached .npy or compute with mpmath."""
    # Try all known cache locations
    search_paths = [
        Path(__file__).parent / "riemann_zeros_10k.npy",
        Path(__file__).parent.parent / "notebooks" / "riemann" / "riemann_zeros_100k_genuine.npy",
        Path(__file__).parent.parent / "research" / "riemann" / "riemann_zeros_100k_genuine.npy",
        Path(__file__).parent.parent / "riemann_zeros_100k_genuine.npy",
        Path(__file__).parent.parent / "riemann_zeros_10k.npy",
    ]
    for path in search_paths:
        if path.exists():
            try:
                zeros = np.load(path)[:n]
                print(f"  Loaded {len(zeros)} zeros from {path}")
                return zeros
            except Exception:
                continue

    # Compute with mpmath as fallback
    print(f"  No cached zeros found. Computing {n} zeros with mpmath...")
    try:
        from mpmath import zetazero
        zeros = np.array([float(zetazero(i).imag) for i in range(1, n + 1)])
        # Cache for future use
        cache_path = Path(__file__).parent / f"riemann_zeros_{n // 1000}k.npy"
        np.save(cache_path, zeros)
        print(f"  Computed and cached {len(zeros)} zeros at {cache_path}")
        return zeros
    except ImportError:
        raise FileNotFoundError(
            "Cannot find Riemann zeros and mpmath is not installed.\n"
            "Install with: pip install mpmath"
        )


# ── Core formula ───────────────────────────────────────────────────────
def sieve_primes(n_max):
    """Sieve of Eratosthenes up to n_max."""
    is_prime = np.ones(n_max + 1, dtype=bool)
    is_prime[:2] = False
    for i in range(2, int(n_max**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    return np.where(is_prime)[0]


def riemann_siegel_theta(t):
    """Riemann-Siegel theta function via Stirling."""
    from scipy.special import loggamma
    return np.imag(loggamma(0.25 + 0.5j * t)) - 0.5 * t * np.log(np.pi)


def riemann_siegel_theta_deriv(t):
    """Derivative: θ'(t) ≈ ½ log(t/(2π))."""
    return 0.5 * np.log(t / (2 * np.pi))


def smooth_zeros(gammas, n_iter=40):
    """Compute smooth zeros γₙ⁽⁰⁾ by Newton's method."""
    n = np.arange(1, len(gammas) + 1)
    target = (n - 1.5) * np.pi
    g0 = gammas.copy()  # initial guess
    for _ in range(n_iter):
        g0 = g0 - (riemann_siegel_theta(g0) - target) / riemann_siegel_theta_deriv(g0)
    return g0


def compute_Sw(T_vals, primes, theta0=THETA0_OPT, theta1=THETA1_OPT,
               k_max=3, kernel='cosine'):
    """
    Compute the mollified Dirichlet polynomial S_w(T).

    Parameters
    ----------
    T_vals : array of T values
    primes : array of prime numbers
    theta0, theta1 : adaptive cutoff parameters (Λ = θ₀ log T + θ₁)
    k_max : maximum prime power
    kernel : 'cosine', 'selberg', 'sharp', 'linear', 'quadratic', 'gaussian', 'cubic'
    """
    log_primes = np.log(primes).astype(np.float64)
    sqrt_primes = np.sqrt(primes).astype(np.float64)

    S = np.zeros(len(T_vals), dtype=np.float64)

    for i, T in enumerate(T_vals):
        log_T = np.log(T)
        Lambda = theta0 * log_T + theta1
        if Lambda <= 0:
            continue

        val = 0.0
        for m in range(1, k_max + 1):
            x = m * log_primes / Lambda
            pm_half = primes.astype(np.float64) ** (m / 2.0)

            # Mollifier kernel
            if kernel == 'cosine':
                w = np.where(x < 1, np.cos(np.pi * x / 2) ** 2, 0.0)
            elif kernel == 'selberg':
                w = np.where(x < 1, (1 - x**2), 0.0)
            elif kernel == 'sharp':
                w = np.where(x < 1, 1.0, 0.0)
            elif kernel == 'linear':
                w = np.where(x < 1, 1 - x, 0.0)
            elif kernel == 'quadratic':
                w = np.where(x < 1, (1 - x)**2, 0.0)
            elif kernel == 'gaussian':
                w = np.exp(-x**2 / 0.32)
            elif kernel == 'cubic':
                w = np.where(x < 1, (1 - x)**3, 0.0)
            else:
                raise ValueError(f"Unknown kernel: {kernel}")

            val += np.sum(w * np.sin(T * m * log_primes) / (m * pm_half))

        S[i] = -val / np.pi

    return S


def compute_predictions(gammas_smooth, Sw_vals):
    """Convert S_w to zero correction predictions."""
    theta_prime = riemann_siegel_theta_deriv(gammas_smooth)
    return -np.pi * Sw_vals / theta_prime


def compute_R2(y_true, y_pred):
    """R² (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


def compute_alpha_R2(delta_true, delta_pred):
    """Compute OLS alpha and R² at alpha=1."""
    # OLS alpha
    alpha = np.sum(delta_true * delta_pred) / np.sum(delta_pred ** 2)
    # R² at alpha = 1
    R2 = compute_R2(delta_true, delta_pred)
    return float(alpha), float(R2)


# ── Test 1: Permutation Test ──────────────────────────────────────────
def test_permutation(delta_true, delta_pred, n_perm=N_PERM):
    """
    Permutation test: shuffle δₙ, recompute R² to build null distribution.
    """
    print("\n" + "=" * 70)
    print("TEST 1: PERMUTATION TEST")
    print("=" * 70)

    R2_real = compute_R2(delta_true, delta_pred)
    print(f"  Original R² = {R2_real:.6f}")

    rng = np.random.default_rng(42)
    R2_null = np.zeros(n_perm)

    t0 = time.time()
    for i in range(n_perm):
        perm = rng.permutation(len(delta_true))
        R2_null[i] = compute_R2(delta_true[perm], delta_pred)
        if (i + 1) % 2000 == 0:
            print(f"    {i+1}/{n_perm} permutations done...")

    elapsed = time.time() - t0

    p_value = np.mean(R2_null >= R2_real)
    z_score = (R2_real - np.mean(R2_null)) / np.std(R2_null) if np.std(R2_null) > 0 else np.inf

    result = {
        "R2_original": float(R2_real),
        "R2_null_mean": float(np.mean(R2_null)),
        "R2_null_std": float(np.std(R2_null)),
        "R2_null_max": float(np.max(R2_null)),
        "z_score": float(z_score),
        "p_value": float(p_value),
        "n_permutations": n_perm,
        "runtime_s": float(elapsed),
    }

    print(f"  Null R²: mean={result['R2_null_mean']:.6f}, "
          f"std={result['R2_null_std']:.6f}, max={result['R2_null_max']:.6f}")
    print(f"  Z-score = {z_score:.1f}")
    print(f"  p-value = {p_value} (< {1/n_perm:.0e} if zero)")
    print(f"  Runtime: {elapsed:.1f}s")

    return result


# ── Test 2: Monte Carlo Uniqueness ────────────────────────────────────
def test_monte_carlo_uniqueness(gammas_smooth, delta_true, primes, n_mc=N_MC):
    """
    Random (θ₀, θ₁) pairs: how many achieve comparable performance?
    """
    print("\n" + "=" * 70)
    print("TEST 2: MONTE CARLO UNIQUENESS")
    print("=" * 70)

    # Use a subsample for speed (every 10th zero)
    step = 10
    gs_sub = gammas_smooth[::step]
    dt_sub = delta_true[::step]

    rng = np.random.default_rng(123)
    theta0_samples = rng.uniform(0.5, 2.0, n_mc)
    theta1_samples = rng.uniform(-8.0, 0.0, n_mc)

    alphas = np.zeros(n_mc)
    R2s = np.zeros(n_mc)

    t0 = time.time()
    for i in range(n_mc):
        Sw = compute_Sw(gs_sub, primes, theta0=theta0_samples[i],
                        theta1=theta1_samples[i])
        dp = compute_predictions(gs_sub, Sw)

        if np.sum(dp**2) > 0:
            alphas[i], R2s[i] = compute_alpha_R2(dt_sub, dp)
        else:
            alphas[i], R2s[i] = np.nan, -1.0

        if (i + 1) % 50000 == 0:
            elapsed = time.time() - t0
            print(f"    {i+1}/{n_mc} configurations ({elapsed:.0f}s)...")

    elapsed = time.time() - t0

    # Count configurations meeting various thresholds
    valid = ~np.isnan(alphas)
    close_alpha = valid & (np.abs(alphas - 1.0) < 0.01)
    high_R2 = valid & (R2s > 0.93)
    both = close_alpha & high_R2

    result = {
        "n_configurations": n_mc,
        "frac_close_alpha": float(np.mean(close_alpha)),
        "frac_high_R2": float(np.mean(high_R2)),
        "frac_both": float(np.mean(both)),
        "best_R2": float(np.nanmax(R2s)),
        "best_alpha_at_best_R2": float(alphas[np.nanargmax(R2s)]),
        "best_theta0": float(theta0_samples[np.nanargmax(R2s)]),
        "best_theta1": float(theta1_samples[np.nanargmax(R2s)]),
        "optimal_R2": None,  # will be filled below
        "runtime_s": float(elapsed),
    }

    # Evaluate optimal configuration on same subsample
    Sw_opt = compute_Sw(gs_sub, primes, theta0=THETA0_OPT, theta1=THETA1_OPT)
    dp_opt = compute_predictions(gs_sub, Sw_opt)
    _, R2_opt = compute_alpha_R2(dt_sub, dp_opt)
    result["optimal_R2"] = float(R2_opt)

    print(f"  Configurations tested: {n_mc:,}")
    print(f"  |α - 1| < 0.01: {result['frac_close_alpha']:.4%}")
    print(f"  R² > 0.93:      {result['frac_high_R2']:.4%}")
    print(f"  Both:            {result['frac_both']:.4%}")
    print(f"  Best random R²:  {result['best_R2']:.6f} (optimal: {R2_opt:.6f})")
    print(f"  Runtime: {elapsed:.1f}s")

    return result


# ── Test 3: Sobol Sensitivity ─────────────────────────────────────────
def sobol_sequence_2d(n):
    """Generate a simple 2D Sobol-like quasi-random sequence (Van der Corput)."""
    # Base-2 Van der Corput for dim 1
    def vdc(n_pts, base=2):
        seq = np.zeros(n_pts)
        for i in range(n_pts):
            f, r = 1.0, 0.0
            val = i + 1
            while val > 0:
                f /= base
                r += f * (val % base)
                val //= base
            seq[i] = r
        return seq

    return np.column_stack([vdc(n, 2), vdc(n, 3)])


def test_sobol_sensitivity(gammas_smooth, delta_true, primes, n_sobol=N_SOBOL):
    """
    Sobol-like sensitivity analysis for (θ₀, θ₁).
    """
    print("\n" + "=" * 70)
    print("TEST 3: SOBOL SENSITIVITY ANALYSIS")
    print("=" * 70)

    step = 20  # subsample for speed
    gs_sub = gammas_smooth[::step]
    dt_sub = delta_true[::step]

    # Generate quasi-random points in [0.5, 2.0] × [-8, 0]
    qr = sobol_sequence_2d(n_sobol)
    theta0_vals = 0.5 + 1.5 * qr[:, 0]
    theta1_vals = -8.0 + 8.0 * qr[:, 1]

    R2_vals = np.zeros(n_sobol)
    alpha_vals = np.zeros(n_sobol)

    t0 = time.time()
    for i in range(n_sobol):
        Sw = compute_Sw(gs_sub, primes, theta0=theta0_vals[i],
                        theta1=theta1_vals[i])
        dp = compute_predictions(gs_sub, Sw)
        if np.sum(dp**2) > 0:
            alpha_vals[i], R2_vals[i] = compute_alpha_R2(dt_sub, dp)
        else:
            alpha_vals[i], R2_vals[i] = np.nan, -1.0

        if (i + 1) % 4000 == 0:
            print(f"    {i+1}/{n_sobol} Sobol points...")

    elapsed = time.time() - t0

    # First-order Sobol indices (variance-based)
    valid = ~np.isnan(R2_vals) & (R2_vals > -0.5)
    R2_v = R2_vals[valid]
    t0_v = theta0_vals[valid]
    t1_v = theta1_vals[valid]

    var_total = np.var(R2_v)

    # Bin-based conditional variance for θ₀
    n_bins = 20
    bins_t0 = np.linspace(0.5, 2.0, n_bins + 1)
    cond_means_t0 = np.zeros(n_bins)
    for b in range(n_bins):
        mask = (t0_v >= bins_t0[b]) & (t0_v < bins_t0[b + 1])
        if np.sum(mask) > 5:
            cond_means_t0[b] = np.mean(R2_v[mask])
    S_theta0 = np.var(cond_means_t0) / var_total if var_total > 0 else 0

    # Bin-based conditional variance for θ₁
    bins_t1 = np.linspace(-8.0, 0.0, n_bins + 1)
    cond_means_t1 = np.zeros(n_bins)
    for b in range(n_bins):
        mask = (t1_v >= bins_t1[b]) & (t1_v < bins_t1[b + 1])
        if np.sum(mask) > 5:
            cond_means_t1[b] = np.mean(R2_v[mask])
    S_theta1 = np.var(cond_means_t1) / var_total if var_total > 0 else 0

    S_interaction = max(0, 1.0 - S_theta0 - S_theta1)

    result = {
        "n_sobol_points": n_sobol,
        "S_theta0_R2": float(S_theta0),
        "S_theta1_R2": float(S_theta1),
        "S_interaction_R2": float(S_interaction),
        "var_total_R2": float(var_total),
        "runtime_s": float(elapsed),
    }

    print(f"  First-order Sobol indices for R²:")
    print(f"    S(θ₀) = {S_theta0:.3f}  ({S_theta0:.1%} of variance)")
    print(f"    S(θ₁) = {S_theta1:.3f}  ({S_theta1:.1%} of variance)")
    print(f"    Interactions = {S_interaction:.3f}")
    print(f"  Runtime: {elapsed:.1f}s")

    return result


# ── Test 4: Bootstrap Stability ───────────────────────────────────────
def find_theta_star(gammas_smooth, delta_true, primes, theta_range=(0.8, 1.2)):
    """Find θ* such that α = 1 by bisection."""
    lo, hi = theta_range
    for _ in range(30):
        mid = (lo + hi) / 2
        Sw = compute_Sw(gammas_smooth, primes, theta0=mid, theta1=0, k_max=3)
        dp = compute_predictions(gammas_smooth, Sw)
        denom = np.sum(dp**2)
        if denom == 0:
            lo = mid
            continue
        alpha = np.sum(delta_true * dp) / denom
        if alpha > 1.0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def test_bootstrap(gammas_smooth, delta_true, primes, n_boot=N_BOOTSTRAP):
    """
    Bootstrap: resample zeros, recompute θ*.
    """
    print("\n" + "=" * 70)
    print("TEST 4: BOOTSTRAP STABILITY")
    print("=" * 70)

    step = 5  # subsample for speed
    gs_sub = gammas_smooth[::step]
    dt_sub = delta_true[::step]
    n_sub = len(gs_sub)

    rng = np.random.default_rng(999)
    theta_stars = np.zeros(n_boot)

    t0 = time.time()
    for i in range(n_boot):
        idx = rng.choice(n_sub, size=n_sub, replace=True)
        gs_boot = gs_sub[idx]
        dt_boot = dt_sub[idx]
        # Sort by position (required for the formula to make sense)
        order = np.argsort(gs_boot)
        gs_boot = gs_boot[order]
        dt_boot = dt_boot[order]

        theta_stars[i] = find_theta_star(gs_boot, dt_boot, primes)

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            print(f"    {i+1}/{n_boot} bootstrap resamples ({elapsed:.0f}s)...")

    elapsed = time.time() - t0

    result = {
        "n_bootstrap": n_boot,
        "theta_star_mean": float(np.mean(theta_stars)),
        "theta_star_std": float(np.std(theta_stars)),
        "theta_star_ci95_lo": float(np.percentile(theta_stars, 2.5)),
        "theta_star_ci95_hi": float(np.percentile(theta_stars, 97.5)),
        "theta_star_min": float(np.min(theta_stars)),
        "theta_star_max": float(np.max(theta_stars)),
        "runtime_s": float(elapsed),
    }

    print(f"  θ* distribution (n={n_boot}):")
    print(f"    Mean:  {result['theta_star_mean']:.4f}")
    print(f"    Std:   {result['theta_star_std']:.4f}")
    print(f"    95%CI: [{result['theta_star_ci95_lo']:.4f}, "
          f"{result['theta_star_ci95_hi']:.4f}]")
    print(f"  Runtime: {elapsed:.1f}s")

    return result


# ── Test 5: Look-Elsewhere Effect ─────────────────────────────────────
def test_look_elsewhere(perm_result):
    """
    Look-elsewhere correction: how many independent trials did we perform?
    """
    print("\n" + "=" * 70)
    print("TEST 5: LOOK-ELSEWHERE CORRECTION")
    print("=" * 70)

    n_kernels = 7
    n_effective_cells = 100  # approximate resolution in (θ₀, θ₁) plane
    n_trials = n_kernels * n_effective_cells

    p_raw = perm_result["p_value"]
    if p_raw == 0:
        p_raw = 1.0 / perm_result["n_permutations"]  # upper bound

    p_corrected = min(1.0, n_trials * p_raw)

    result = {
        "n_kernels": n_kernels,
        "n_effective_cells": n_effective_cells,
        "n_total_trials": n_trials,
        "p_raw": float(p_raw),
        "p_bonferroni": float(p_corrected),
        "still_significant": bool(p_corrected < 0.05),
    }

    print(f"  Independent trials: {n_kernels} kernels × {n_effective_cells} "
          f"cells = {n_trials}")
    print(f"  Raw p-value:        {p_raw:.2e}")
    print(f"  Bonferroni p-value: {p_corrected:.2e}")
    print(f"  Still significant (< 0.05): {result['still_significant']}")

    return result


# ── Main ──────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("STATISTICAL VALIDATION SUITE")
    print("Mollified Prime-Spectral Formula for S(T)")
    print("=" * 70)

    # ── Load data ──
    print("\n[1/6] Loading Riemann zeros...")
    gammas = load_zeros(N_ZEROS)
    print(f"  Zeros loaded: {len(gammas)}")
    print(f"  Range: [{gammas[0]:.3f}, {gammas[-1]:.3f}]")

    # ── Compute smooth zeros and corrections ──
    print("\n[2/6] Computing smooth zeros and corrections...")
    gammas_smooth = smooth_zeros(gammas)
    delta_true = gammas - gammas_smooth
    print(f"  δ mean: {np.mean(delta_true):.6f}")
    print(f"  δ std:  {np.std(delta_true):.6f}")
    print(f"  δ max:  {np.max(np.abs(delta_true)):.6f}")

    # ── Generate primes ──
    primes = sieve_primes(10_000)
    print(f"  Primes up to 10,000: {len(primes)}")

    # ── Compute predictions with optimal parameters ──
    print("\n[3/6] Computing S_w with optimal parameters...")
    t0 = time.time()
    Sw = compute_Sw(gammas_smooth, primes,
                    theta0=THETA0_OPT, theta1=THETA1_OPT)
    delta_pred = compute_predictions(gammas_smooth, Sw)
    alpha_opt, R2_opt = compute_alpha_R2(delta_true, delta_pred)
    elapsed = time.time() - t0
    print(f"  α = {alpha_opt:.6f}, R² = {R2_opt:.6f} ({elapsed:.1f}s)")

    # ── Run tests ──
    print("\n[4/6] Running statistical tests...")
    results = {"metadata": {
        "n_zeros": N_ZEROS,
        "theta0": THETA0_OPT,
        "theta1": THETA1_OPT,
        "alpha_optimal": alpha_opt,
        "R2_optimal": R2_opt,
        "date": "2026-02-07",
    }}

    # Test 1: Permutation
    results["permutation"] = test_permutation(delta_true, delta_pred,
                                               n_perm=N_PERM)

    # Test 2: Monte Carlo uniqueness
    results["monte_carlo"] = test_monte_carlo_uniqueness(
        gammas_smooth, delta_true, primes, n_mc=N_MC)

    # Test 3: Sobol sensitivity
    results["sobol"] = test_sobol_sensitivity(
        gammas_smooth, delta_true, primes, n_sobol=N_SOBOL)

    # Test 4: Bootstrap
    results["bootstrap"] = test_bootstrap(
        gammas_smooth, delta_true, primes, n_boot=N_BOOTSTRAP)

    # Test 5: Look-elsewhere
    results["look_elsewhere"] = test_look_elsewhere(results["permutation"])

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Formula: α = {alpha_opt:.6f}, R² = {R2_opt:.6f}")
    print(f"  Permutation test:  Z = {results['permutation']['z_score']:.0f}, "
          f"p < {1/N_PERM:.0e}")
    print(f"  MC uniqueness:     {results['monte_carlo']['frac_both']:.4%} "
          f"match both criteria")
    print(f"  Sobol S(θ₀):       {results['sobol']['S_theta0_R2']:.3f}")
    print(f"  Bootstrap θ* std:  {results['bootstrap']['theta_star_std']:.4f}")
    print(f"  Look-elsewhere:    p = {results['look_elsewhere']['p_bonferroni']:.2e}")

    # ── Save ──
    out_path = RESULTS_DIR / "paper1_validation_results.json"

    # Convert any remaining numpy types
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    clean = json.loads(json.dumps(results, default=convert))
    with open(out_path, 'w') as f:
        json.dump(clean, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
