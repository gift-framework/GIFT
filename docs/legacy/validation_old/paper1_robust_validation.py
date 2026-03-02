#!/usr/bin/env python3
"""
Robust Statistical Validation — Paper 1: Mollified Prime-Spectral S(T)
======================================================================

Enhanced validation addressing known weaknesses of paper1_montecarlo.py:

  A) Circular optimization  → Honest train/test split + K-fold CV
  B) No baseline comparison → Null, linear, polynomial, random-frequency
  C) Limited permutations   → 50K trials + Gaussian tail bound
  D) No Bayesian analysis   → BIC-based Bayes factor
  E) Limited bootstrap      → BCa confidence intervals (2K resamples)
  F) No stability analysis  → Per-window α/R² drift test
  G) Narrow MC search       → 100K configurations × 7 kernels

8 independent tests, comprehensive JSON report with pass/fail verdict.

Author: GIFT Framework
Date: 2026-02-08
"""

import numpy as np
from pathlib import Path
import json
import time
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

N_ZEROS = 10_000
N_PERM = 50_000
N_MC = 100_000
N_BOOTSTRAP = 2_000
N_KFOLD = 10
N_BASELINE_RANDOM = 200

THETA0_OPT = 1.4091
THETA1_OPT = -3.9537
THETA_CONST = 0.9941

KERNELS = ['cosine', 'selberg', 'sharp', 'linear', 'quadratic', 'gaussian', 'cubic']

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
#  CORE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def load_zeros(n=N_ZEROS):
    """Load Riemann zeros from cache or compute with mpmath."""
    search_paths = [
        Path(__file__).parent / f"riemann_zeros_{n // 1000}k.npy",
        Path(__file__).parent / "riemann_zeros_10k.npy",
        Path(__file__).parent.parent / "notebooks" / "riemann" / "riemann_zeros_100k_genuine.npy",
        Path(__file__).parent.parent / "research" / "riemann" / "riemann_zeros_100k_genuine.npy",
        Path(__file__).parent.parent / "riemann_zeros_100k_genuine.npy",
    ]
    for path in search_paths:
        if path.exists():
            try:
                zeros = np.load(path)[:n]
                print(f"  Loaded {len(zeros)} zeros from {path.name}")
                return zeros
            except Exception:
                continue
    print(f"  Computing {n} zeros with mpmath...")
    from mpmath import zetazero
    zeros = np.array([float(zetazero(i).imag) for i in range(1, n + 1)])
    cache_path = Path(__file__).parent / f"riemann_zeros_{n // 1000}k.npy"
    np.save(cache_path, zeros)
    return zeros


def sieve_primes(n_max):
    """Sieve of Eratosthenes."""
    is_prime = np.ones(n_max + 1, dtype=bool)
    is_prime[:2] = False
    for i in range(2, int(n_max**0.5) + 1):
        if is_prime[i]:
            is_prime[i * i::i] = False
    return np.where(is_prime)[0]


def riemann_siegel_theta(t):
    """Riemann-Siegel theta function."""
    from scipy.special import loggamma
    return np.imag(loggamma(0.25 + 0.5j * t)) - 0.5 * t * np.log(np.pi)


def riemann_siegel_theta_deriv(t):
    """θ'(t) ≈ ½ log(t/(2π))."""
    return 0.5 * np.log(t / (2 * np.pi))


def smooth_zeros(gammas, n_iter=40):
    """Smooth zero positions via Newton on θ(γ) = (n-3/2)π."""
    n = np.arange(1, len(gammas) + 1)
    target = (n - 1.5) * np.pi
    g0 = gammas.copy()
    for _ in range(n_iter):
        g0 -= (riemann_siegel_theta(g0) - target) / riemann_siegel_theta_deriv(g0)
    return g0


def compute_Sw(T_vals, primes, theta0=THETA0_OPT, theta1=THETA1_OPT,
               k_max=3, kernel='cosine'):
    """Vectorized computation of the mollified Dirichlet polynomial S_w(T)."""
    log_primes = np.log(primes).astype(np.float64)
    log_T = np.log(T_vals)
    Lambda = theta0 * log_T + theta1
    valid = Lambda > 0
    if not np.any(valid):
        return np.zeros(len(T_vals))

    S = np.zeros(len(T_vals))
    T_v = T_vals[valid]
    Lambda_v = Lambda[valid]

    for m in range(1, k_max + 1):
        pm_half = primes.astype(np.float64) ** (m / 2.0)
        x = m * log_primes[None, :] / Lambda_v[:, None]

        if kernel == 'cosine':
            w = np.where(x < 1, np.cos(np.pi * x / 2) ** 2, 0.0)
        elif kernel == 'selberg':
            w = np.where(x < 1, 1 - x ** 2, 0.0)
        elif kernel == 'sharp':
            w = np.where(x < 1, 1.0, 0.0)
        elif kernel == 'linear':
            w = np.where(x < 1, 1 - x, 0.0)
        elif kernel == 'quadratic':
            w = np.where(x < 1, (1 - x) ** 2, 0.0)
        elif kernel == 'gaussian':
            w = np.exp(-x ** 2 / 0.32)
        elif kernel == 'cubic':
            w = np.where(x < 1, (1 - x) ** 3, 0.0)
        else:
            raise ValueError(f"Unknown kernel: {kernel}")

        sin_vals = np.sin(T_v[:, None] * m * log_primes[None, :])
        S[valid] += np.sum(w * sin_vals / (m * pm_half[None, :]), axis=1)

    return -S / np.pi


def compute_predictions(gammas_smooth, Sw_vals):
    """Convert S_w to zero correction predictions."""
    return -np.pi * Sw_vals / riemann_siegel_theta_deriv(gammas_smooth)


def compute_R2(y_true, y_pred):
    """Coefficient of determination."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def compute_alpha_R2(delta_true, delta_pred):
    """OLS regression coefficient α and R²."""
    denom = np.sum(delta_pred ** 2)
    if denom == 0:
        return 0.0, -1.0
    alpha = float(np.sum(delta_true * delta_pred) / denom)
    R2 = compute_R2(delta_true, delta_pred)
    return alpha, R2


def find_theta_star(gs, dt, primes, theta_range=(0.8, 1.2), n_iter=40):
    """Find constant θ* such that α = 1 by bisection."""
    lo, hi = theta_range
    for _ in range(n_iter):
        mid = (lo + hi) / 2
        Sw = compute_Sw(gs, primes, theta0=mid, theta1=0, k_max=3)
        dp = compute_predictions(gs, Sw)
        denom = np.sum(dp ** 2)
        if denom == 0:
            lo = mid
            continue
        alpha = np.sum(dt * dp) / denom
        if alpha > 1.0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ══════════════════════════════════════════════════════════════════════
#  TEST 1: HONEST TRAIN/TEST SPLIT
# ══════════════════════════════════════════════════════════════════════

def test_train_test_split(gammas, gammas_smooth, delta_true, primes):
    """
    Split zeros 50/50. Find θ* on train set via α=1 constraint,
    evaluate on held-out test set. Addresses circular optimization.
    """
    print("\n" + "=" * 70)
    print("TEST 1: HONEST TRAIN/TEST SPLIT")
    print("=" * 70)

    n = len(gammas)
    mid = n // 2

    gs_train, gs_test = gammas_smooth[:mid], gammas_smooth[mid:]
    dt_train, dt_test = delta_true[:mid], delta_true[mid:]

    t0 = time.time()

    # Train: find θ* such that α = 1 on train set
    theta_star_train = find_theta_star(gs_train, dt_train, primes)

    # Evaluate constant-θ model on train
    Sw_tr = compute_Sw(gs_train, primes, theta0=theta_star_train, theta1=0)
    dp_tr = compute_predictions(gs_train, Sw_tr)
    alpha_train, R2_train = compute_alpha_R2(dt_train, dp_tr)

    # Evaluate constant-θ model on TEST with train's θ*
    Sw_te = compute_Sw(gs_test, primes, theta0=theta_star_train, theta1=0)
    dp_te = compute_predictions(gs_test, Sw_te)
    alpha_test, R2_test = compute_alpha_R2(dt_test, dp_te)

    # Also evaluate with paper's adaptive (θ₀, θ₁) on test set
    Sw_te_adap = compute_Sw(gs_test, primes,
                            theta0=THETA0_OPT, theta1=THETA1_OPT)
    dp_te_adap = compute_predictions(gs_test, Sw_te_adap)
    alpha_test_adap, R2_test_adap = compute_alpha_R2(dt_test, dp_te_adap)

    # Also find θ* independently on test set to check consistency
    theta_star_test = find_theta_star(gs_test, dt_test, primes)

    elapsed = time.time() - t0
    gen_gap = R2_train - R2_test

    passed = abs(gen_gap) < 0.05 and R2_test > 0.85

    result = {
        "n_train": mid,
        "n_test": n - mid,
        "theta_star_train": float(theta_star_train),
        "theta_star_test": float(theta_star_test),
        "theta_star_paper": THETA_CONST,
        "theta_consistency": float(abs(theta_star_train - theta_star_test)),
        "train": {"alpha": alpha_train, "R2": R2_train},
        "test_constant_theta": {"alpha": alpha_test, "R2": R2_test},
        "test_adaptive_paper": {"alpha": alpha_test_adap, "R2": R2_test_adap},
        "generalization_gap": float(gen_gap),
        "passed": passed,
        "runtime_s": float(elapsed),
    }

    print(f"  θ*_train = {theta_star_train:.4f}, "
          f"θ*_test = {theta_star_test:.4f} (paper: {THETA_CONST})")
    print(f"  Train:  α = {alpha_train:.4f}, R² = {R2_train:.4f}")
    print(f"  Test (train θ*):  α = {alpha_test:.4f}, R² = {R2_test:.4f}")
    print(f"  Test (paper θ₀,θ₁): α = {alpha_test_adap:.4f}, "
          f"R² = {R2_test_adap:.4f}")
    print(f"  Generalization gap: {gen_gap:+.4f}")
    print(f"  PASSED: {passed}  ({elapsed:.1f}s)")

    return result


# ══════════════════════════════════════════════════════════════════════
#  TEST 2: PERMUTATION TEST + EFFECT SIZE + LEE
# ══════════════════════════════════════════════════════════════════════

def test_permutation(delta_true, delta_pred, n_perm=N_PERM):
    """
    50K permutations + Gaussian tail bound + Cohen's d + look-elsewhere.
    """
    print("\n" + "=" * 70)
    print("TEST 2: PERMUTATION TEST (50K) + EFFECT SIZE + LEE")
    print("=" * 70)

    R2_real = compute_R2(delta_true, delta_pred)
    rng = np.random.default_rng(42)
    R2_null = np.zeros(n_perm)

    t0 = time.time()
    for i in range(n_perm):
        R2_null[i] = compute_R2(delta_true[rng.permutation(len(delta_true))],
                                delta_pred)
        if (i + 1) % 25000 == 0:
            print(f"    {i + 1}/{n_perm} permutations...")

    elapsed = time.time() - t0

    p_empirical = float(np.mean(R2_null >= R2_real))
    null_mean = float(np.mean(R2_null))
    null_std = float(np.std(R2_null))
    z_score = (R2_real - null_mean) / null_std if null_std > 0 else float('inf')

    # Gaussian tail bound
    try:
        from scipy.stats import norm
        p_gaussian = float(norm.sf(z_score)) if np.isfinite(z_score) else 0.0
    except ImportError:
        p_gaussian = None

    # Rule of 3 upper bound when p = 0
    p_upper = 3.0 / n_perm if p_empirical == 0 else p_empirical

    # Effect size: Cohen's d
    cohens_d = z_score  # same as (observed - null_mean) / null_std

    # CLES: probability that formula R² > random R²
    cles = float(np.mean(R2_null < R2_real))

    # Look-Elsewhere Effect correction
    n_kernels = len(KERNELS)
    n_effective_cells = 100  # approximate resolution in (θ₀, θ₁) plane
    n_trials = n_kernels * n_effective_cells
    # Sidak correction: p_global = 1 - (1 - p_local)^n_trials
    p_sidak = float(1 - (1 - p_upper) ** n_trials)
    p_bonferroni = float(min(1.0, n_trials * p_upper))

    passed = p_upper < 0.001 and z_score > 10

    result = {
        "R2_original": float(R2_real),
        "R2_null_mean": null_mean,
        "R2_null_std": null_std,
        "R2_null_max": float(np.max(R2_null)),
        "R2_null_p99": float(np.percentile(R2_null, 99)),
        "z_score": float(z_score),
        "p_empirical": p_empirical,
        "p_upper_bound": float(p_upper),
        "p_gaussian_tail": p_gaussian,
        "n_permutations": n_perm,
        "effect_size": {
            "cohens_d": float(cohens_d),
            "cles": cles,
            "interpretation": "enormous" if cohens_d > 2 else (
                "large" if cohens_d > 0.8 else "medium"),
        },
        "look_elsewhere": {
            "n_kernels": n_kernels,
            "n_effective_cells": n_effective_cells,
            "n_total_trials": n_trials,
            "p_sidak": p_sidak,
            "p_bonferroni": p_bonferroni,
            "still_significant": bool(p_bonferroni < 0.05),
        },
        "passed": passed,
        "runtime_s": float(elapsed),
    }

    print(f"  R² = {R2_real:.6f} vs null: {null_mean:.6f} ± {null_std:.6f}")
    print(f"  Z-score = {z_score:.1f}")
    print(f"  p-value ≤ {p_upper:.2e} (empirical: {p_empirical})")
    if p_gaussian is not None:
        print(f"  p-value (Gaussian tail) = {p_gaussian:.2e}")
    print(f"  Cohen's d = {cohens_d:.1f} ({result['effect_size']['interpretation']})")
    print(f"  CLES = {cles:.6f}")
    print(f"  LEE Bonferroni p = {p_bonferroni:.2e} (Sidak: {p_sidak:.2e})")
    print(f"  PASSED: {passed}  ({elapsed:.1f}s)")

    return result


# ══════════════════════════════════════════════════════════════════════
#  TEST 3: BASELINE MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════

def test_baselines(gammas_smooth, delta_true, delta_pred, primes):
    """
    Compare formula against null, linear, polynomial, and random-frequency
    models. Shows the formula outperforms fundamentally different approaches.
    """
    print("\n" + "=" * 70)
    print("TEST 3: BASELINE MODEL COMPARISON")
    print("=" * 70)

    t0 = time.time()
    n = len(delta_true)
    idx = np.arange(n, dtype=np.float64)
    T_vals = gammas_smooth

    R2_formula = compute_R2(delta_true, delta_pred)
    alpha_formula, _ = compute_alpha_R2(delta_true, delta_pred)

    # 1. Null model: constant = mean(δ) → R² = 0 by definition
    R2_null = 0.0

    # 2. Linear model: δ = a + b·n
    A_lin = np.column_stack([np.ones(n), idx])
    c_lin = np.linalg.lstsq(A_lin, delta_true, rcond=None)[0]
    R2_linear = compute_R2(delta_true, A_lin @ c_lin)

    # 3. Polynomial degree 5: δ = Σ aₖ nᵏ
    idx_norm = idx / n  # normalize to avoid numerical issues
    A_poly = np.column_stack([idx_norm ** k for k in range(6)])
    c_poly = np.linalg.lstsq(A_poly, delta_true, rcond=None)[0]
    R2_poly5 = compute_R2(delta_true, A_poly @ c_poly)

    # 4. Polynomial degree 10
    A_poly10 = np.column_stack([idx_norm ** k for k in range(11)])
    c_poly10 = np.linalg.lstsq(A_poly10, delta_true, rcond=None)[0]
    R2_poly10 = compute_R2(delta_true, A_poly10 @ c_poly10)

    # 5. Random-frequency model: same structure, random "log primes"
    rng = np.random.default_rng(777)
    R2_randoms = np.zeros(N_BASELINE_RANDOM)
    n_freqs = 50  # similar to number of primes contributing significantly
    step = max(1, n // 2000)  # subsample for speed
    T_sub = T_vals[::step]
    dt_sub = delta_true[::step]

    for trial in range(N_BASELINE_RANDOM):
        freqs = rng.uniform(0.5, 10.0, n_freqs)
        amps = 1.0 / np.sqrt(np.arange(1, n_freqs + 1))
        S_rand = np.sum(
            amps[:, None] * np.sin(freqs[:, None] * T_sub[None, :]),
            axis=0
        )
        if np.std(S_rand) > 0:
            S_rand = S_rand / np.std(S_rand) * np.std(dt_sub)
        R2_randoms[trial] = max(0, compute_R2(dt_sub, S_rand))

    R2_random_mean = float(np.mean(R2_randoms))
    R2_random_max = float(np.max(R2_randoms))
    R2_random_p99 = float(np.percentile(R2_randoms, 99))

    elapsed = time.time() - t0

    # Improvement factors
    baselines = {
        "null": R2_null,
        "linear": R2_linear,
        "polynomial_5": R2_poly5,
        "polynomial_10": R2_poly10,
        "random_frequency_mean": R2_random_mean,
        "random_frequency_max": R2_random_max,
    }

    passed = (R2_formula > R2_poly10 and R2_formula > R2_random_max)

    result = {
        "R2_formula": float(R2_formula),
        "R2_null": R2_null,
        "R2_linear": float(R2_linear),
        "R2_polynomial_5": float(R2_poly5),
        "R2_polynomial_10": float(R2_poly10),
        "R2_random_frequency": {
            "n_trials": N_BASELINE_RANDOM,
            "n_freqs": n_freqs,
            "mean": R2_random_mean,
            "max": R2_random_max,
            "p99": R2_random_p99,
            "std": float(np.std(R2_randoms)),
        },
        "formula_beats_all": passed,
        "improvement_over_best_baseline": float(
            R2_formula - max(R2_linear, R2_poly10, R2_random_max)),
        "passed": passed,
        "runtime_s": float(elapsed),
    }

    print(f"  R² comparison:")
    print(f"    Formula (2 params):         {R2_formula:.6f}")
    print(f"    Null (0 params):            {R2_null:.6f}")
    print(f"    Linear (2 params):          {R2_linear:.6f}")
    print(f"    Polynomial-5 (6 params):    {R2_poly5:.6f}")
    print(f"    Polynomial-10 (11 params):  {R2_poly10:.6f}")
    print(f"    Random freq (mean/max):     {R2_random_mean:.6f} / "
          f"{R2_random_max:.6f}")
    print(f"  Formula beats all baselines: {passed}")
    print(f"  PASSED: {passed}  ({elapsed:.1f}s)")

    return result


# ══════════════════════════════════════════════════════════════════════
#  TEST 4: K-FOLD CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════

def test_kfold_cv(gammas_smooth, delta_true, primes, n_folds=N_KFOLD):
    """
    Contiguous K-fold CV: split into K blocks, leave one out each time,
    find θ* on train, evaluate on test block.
    """
    print("\n" + "=" * 70)
    print(f"TEST 4: {n_folds}-FOLD CROSS-VALIDATION")
    print("=" * 70)

    t0 = time.time()
    n = len(gammas_smooth)
    fold_size = n // n_folds

    R2_folds = []
    alpha_folds = []
    theta_folds = []

    for k in range(n_folds):
        test_start = k * fold_size
        test_end = (k + 1) * fold_size if k < n_folds - 1 else n

        train_mask = np.ones(n, dtype=bool)
        train_mask[test_start:test_end] = False

        gs_tr = gammas_smooth[train_mask]
        dt_tr = delta_true[train_mask]
        gs_te = gammas_smooth[~train_mask]
        dt_te = delta_true[~train_mask]

        # Find θ* on train set (subsample for speed)
        step = max(1, len(gs_tr) // 1000)
        theta_k = find_theta_star(gs_tr[::step], dt_tr[::step], primes)

        # Evaluate on test fold
        Sw_te = compute_Sw(gs_te, primes, theta0=theta_k, theta1=0)
        dp_te = compute_predictions(gs_te, Sw_te)
        alpha_k, R2_k = compute_alpha_R2(dt_te, dp_te)

        R2_folds.append(R2_k)
        alpha_folds.append(alpha_k)
        theta_folds.append(float(theta_k))

        print(f"    Fold {k + 1}/{n_folds}: θ* = {theta_k:.4f}, "
              f"α = {alpha_k:.4f}, R² = {R2_k:.4f}")

    elapsed = time.time() - t0

    R2_arr = np.array(R2_folds)
    alpha_arr = np.array(alpha_folds)
    theta_arr = np.array(theta_folds)

    passed = (np.mean(R2_arr) > 0.85 and np.std(R2_arr) < 0.05
              and np.mean(np.abs(alpha_arr - 1.0)) < 0.05)

    result = {
        "n_folds": n_folds,
        "per_fold": {
            "R2": [float(x) for x in R2_folds],
            "alpha": [float(x) for x in alpha_folds],
            "theta_star": theta_folds,
        },
        "R2_mean": float(np.mean(R2_arr)),
        "R2_std": float(np.std(R2_arr)),
        "R2_min": float(np.min(R2_arr)),
        "R2_max": float(np.max(R2_arr)),
        "alpha_mean": float(np.mean(alpha_arr)),
        "alpha_std": float(np.std(alpha_arr)),
        "theta_star_mean": float(np.mean(theta_arr)),
        "theta_star_std": float(np.std(theta_arr)),
        "passed": passed,
        "runtime_s": float(elapsed),
    }

    print(f"  Summary (n={n_folds} folds):")
    print(f"    R²:  {result['R2_mean']:.4f} ± {result['R2_std']:.4f} "
          f"[{result['R2_min']:.4f}, {result['R2_max']:.4f}]")
    print(f"    α:   {result['alpha_mean']:.4f} ± {result['alpha_std']:.4f}")
    print(f"    θ*:  {result['theta_star_mean']:.4f} ± "
          f"{result['theta_star_std']:.4f}")
    print(f"  PASSED: {passed}  ({elapsed:.1f}s)")

    return result


# ══════════════════════════════════════════════════════════════════════
#  TEST 5: MONTE CARLO UNIQUENESS (ENHANCED)
# ══════════════════════════════════════════════════════════════════════

def test_monte_carlo(gammas_smooth, delta_true, primes, n_mc=N_MC):
    """
    100K random (θ₀, θ₁, kernel) configs. Measures how special the
    optimal parameters are in the full configuration space.
    """
    print("\n" + "=" * 70)
    print("TEST 5: MONTE CARLO UNIQUENESS (100K)")
    print("=" * 70)

    # Subsample zeros for speed
    step = 10
    gs_sub = gammas_smooth[::step]
    dt_sub = delta_true[::step]

    rng = np.random.default_rng(123)

    # Wider parameter ranges than original
    theta0_range = (0.3, 2.5)
    theta1_range = (-10.0, 2.0)

    theta0_samples = rng.uniform(*theta0_range, n_mc)
    theta1_samples = rng.uniform(*theta1_range, n_mc)
    # Assign random kernel to each (uniform over 7 kernels)
    kernel_indices = rng.integers(0, len(KERNELS), n_mc)

    alphas = np.full(n_mc, np.nan)
    R2s = np.full(n_mc, -1.0)

    t0 = time.time()
    for i in range(n_mc):
        kernel = KERNELS[kernel_indices[i]]
        Sw = compute_Sw(gs_sub, primes,
                        theta0=theta0_samples[i],
                        theta1=theta1_samples[i],
                        kernel=kernel)
        dp = compute_predictions(gs_sub, Sw)
        if np.sum(dp ** 2) > 0:
            alphas[i], R2s[i] = compute_alpha_R2(dt_sub, dp)

        if (i + 1) % 50000 == 0:
            elapsed = time.time() - t0
            print(f"    {i + 1}/{n_mc} configurations ({elapsed:.0f}s)...")

    elapsed = time.time() - t0

    valid = ~np.isnan(alphas)
    close_alpha = valid & (np.abs(alphas - 1.0) < 0.01)
    high_R2 = valid & (R2s > 0.90)
    very_high_R2 = valid & (R2s > 0.93)
    both = close_alpha & very_high_R2

    # Evaluate optimal on same subsample for fair comparison
    Sw_opt = compute_Sw(gs_sub, primes, theta0=THETA0_OPT, theta1=THETA1_OPT)
    dp_opt = compute_predictions(gs_sub, Sw_opt)
    _, R2_opt = compute_alpha_R2(dt_sub, dp_opt)

    # Percentile rank of optimal
    percentile_rank = float(np.mean(R2s[valid] < R2_opt) * 100)

    passed = (np.mean(both) < 0.001 and percentile_rank > 99.0)

    result = {
        "n_configurations": n_mc,
        "theta0_range": list(theta0_range),
        "theta1_range": list(theta1_range),
        "n_kernels": len(KERNELS),
        "frac_close_alpha": float(np.mean(close_alpha)),
        "frac_R2_above_90": float(np.mean(high_R2)),
        "frac_R2_above_93": float(np.mean(very_high_R2)),
        "frac_both_criteria": float(np.mean(both)),
        "best_random_R2": float(np.nanmax(R2s)),
        "R2_optimal_subsample": float(R2_opt),
        "percentile_rank": percentile_rank,
        "R2_distribution": {
            "mean": float(np.nanmean(R2s[valid])),
            "std": float(np.nanstd(R2s[valid])),
            "p50": float(np.nanpercentile(R2s[valid], 50)),
            "p90": float(np.nanpercentile(R2s[valid], 90)),
            "p99": float(np.nanpercentile(R2s[valid], 99)),
        },
        "passed": passed,
        "runtime_s": float(elapsed),
    }

    print(f"  Configurations: {n_mc:,}")
    print(f"  |α - 1| < 0.01:         {result['frac_close_alpha']:.4%}")
    print(f"  R² > 0.93:              {result['frac_R2_above_93']:.4%}")
    print(f"  Both criteria:          {result['frac_both_criteria']:.4%}")
    print(f"  Best random R²:         {result['best_random_R2']:.6f}")
    print(f"  Optimal R² (subsample): {R2_opt:.6f}")
    print(f"  Percentile rank:        {percentile_rank:.2f}%")
    print(f"  PASSED: {passed}  ({elapsed:.1f}s)")

    return result


# ══════════════════════════════════════════════════════════════════════
#  TEST 6: BAYESIAN MODEL COMPARISON (BIC)
# ══════════════════════════════════════════════════════════════════════

def test_bayesian_bic(gammas_smooth, delta_true, delta_pred):
    """
    BIC-based Bayes factor: formula (2 params) vs null (1 param)
    vs linear (2 params) vs polynomial-5 (6 params).
    """
    print("\n" + "=" * 70)
    print("TEST 6: BAYESIAN MODEL COMPARISON (BIC)")
    print("=" * 70)

    t0 = time.time()
    n = len(delta_true)
    idx = np.arange(n, dtype=np.float64)
    idx_norm = idx / n

    models = {}

    # Formula model: k = 2 (θ₀, θ₁)
    RSS_f = np.sum((delta_true - delta_pred) ** 2)
    k_f = 2
    BIC_f = n * np.log(RSS_f / n) + k_f * np.log(n)
    models["formula"] = {"k": k_f, "RSS": float(RSS_f), "BIC": float(BIC_f)}

    # Null model: k = 1 (mean)
    RSS_null = np.sum((delta_true - np.mean(delta_true)) ** 2)
    k_null = 1
    BIC_null = n * np.log(RSS_null / n) + k_null * np.log(n)
    models["null"] = {"k": k_null, "RSS": float(RSS_null), "BIC": float(BIC_null)}

    # Linear model: k = 2
    A_lin = np.column_stack([np.ones(n), idx])
    c_lin = np.linalg.lstsq(A_lin, delta_true, rcond=None)[0]
    RSS_lin = np.sum((delta_true - A_lin @ c_lin) ** 2)
    k_lin = 2
    BIC_lin = n * np.log(RSS_lin / n) + k_lin * np.log(n)
    models["linear"] = {"k": k_lin, "RSS": float(RSS_lin), "BIC": float(BIC_lin)}

    # Polynomial-5: k = 6
    A_p5 = np.column_stack([idx_norm ** k for k in range(6)])
    c_p5 = np.linalg.lstsq(A_p5, delta_true, rcond=None)[0]
    RSS_p5 = np.sum((delta_true - A_p5 @ c_p5) ** 2)
    k_p5 = 6
    BIC_p5 = n * np.log(RSS_p5 / n) + k_p5 * np.log(n)
    models["polynomial_5"] = {"k": k_p5, "RSS": float(RSS_p5), "BIC": float(BIC_p5)}

    # Polynomial-10: k = 11
    A_p10 = np.column_stack([idx_norm ** k for k in range(11)])
    c_p10 = np.linalg.lstsq(A_p10, delta_true, rcond=None)[0]
    RSS_p10 = np.sum((delta_true - A_p10 @ c_p10) ** 2)
    k_p10 = 11
    BIC_p10 = n * np.log(RSS_p10 / n) + k_p10 * np.log(n)
    models["polynomial_10"] = {"k": k_p10, "RSS": float(RSS_p10),
                                "BIC": float(BIC_p10)}

    # Bayes factors (approximate via BIC)
    bayes_factors = {}
    for name, m in models.items():
        if name != "formula":
            delta_bic = m["BIC"] - BIC_f
            log10_bf = delta_bic / (2 * np.log(10))
            bayes_factors[f"formula_vs_{name}"] = {
                "delta_BIC": float(delta_bic),
                "log10_BF": float(log10_bf),
                "interpretation": (
                    "decisive" if log10_bf > 2 else
                    "very_strong" if log10_bf > 1.5 else
                    "strong" if log10_bf > 1 else
                    "substantial" if log10_bf > 0.5 else
                    "inconclusive"
                ),
            }

    elapsed = time.time() - t0

    best_non_formula = min(m["BIC"] for name, m in models.items()
                          if name != "formula")
    formula_is_best = BIC_f < best_non_formula
    passed = formula_is_best

    result = {
        "models": models,
        "bayes_factors": bayes_factors,
        "formula_has_lowest_BIC": formula_is_best,
        "passed": passed,
        "runtime_s": float(elapsed),
    }

    print(f"  BIC comparison:")
    for name, m in models.items():
        marker = " ← BEST" if m["BIC"] == BIC_f and formula_is_best else ""
        print(f"    {name:>15s} (k={m['k']:>2d}): BIC = {m['BIC']:>12.1f}{marker}")
    print(f"  Bayes factors (formula vs alternatives):")
    for name, bf in bayes_factors.items():
        print(f"    {name:>25s}: log₁₀(BF) = {bf['log10_BF']:>8.1f} "
              f"({bf['interpretation']})")
    print(f"  PASSED: {passed}  ({elapsed:.1f}s)")

    return result


# ══════════════════════════════════════════════════════════════════════
#  TEST 7: BOOTSTRAP BCa
# ══════════════════════════════════════════════════════════════════════

def test_bootstrap_bca(gammas_smooth, delta_true, primes, n_boot=N_BOOTSTRAP):
    """
    BCa bootstrap: bias-corrected and accelerated confidence intervals
    for θ*, α, and R². More robust than standard percentile bootstrap.
    """
    print("\n" + "=" * 70)
    print(f"TEST 7: BOOTSTRAP BCa ({n_boot} resamples)")
    print("=" * 70)

    step = 10
    gs_sub = gammas_smooth[::step]
    dt_sub = delta_true[::step]
    n_sub = len(gs_sub)

    # Original statistics
    theta_orig = find_theta_star(gs_sub, dt_sub, primes)
    Sw_orig = compute_Sw(gs_sub, primes, theta0=theta_orig, theta1=0)
    dp_orig = compute_predictions(gs_sub, Sw_orig)
    alpha_orig, R2_orig = compute_alpha_R2(dt_sub, dp_orig)

    rng = np.random.default_rng(999)
    theta_boots = np.zeros(n_boot)
    alpha_boots = np.zeros(n_boot)
    R2_boots = np.zeros(n_boot)

    t0 = time.time()
    for i in range(n_boot):
        idx = np.sort(rng.choice(n_sub, size=n_sub, replace=True))
        gs_b = gs_sub[idx]
        dt_b = dt_sub[idx]

        theta_boots[i] = find_theta_star(gs_b, dt_b, primes)
        Sw_b = compute_Sw(gs_b, primes, theta0=theta_boots[i], theta1=0)
        dp_b = compute_predictions(gs_b, Sw_b)
        alpha_boots[i], R2_boots[i] = compute_alpha_R2(dt_b, dp_b)

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"    {i + 1}/{n_boot} resamples ({elapsed:.0f}s)...")

    elapsed = time.time() - t0

    # BCa correction for θ*
    def bca_ci(boot_samples, original, alpha_level=0.05):
        """Compute BCa confidence interval."""
        try:
            from scipy.stats import norm

            # Bias correction
            z0 = norm.ppf(np.mean(boot_samples < original))
            if not np.isfinite(z0):
                z0 = 0.0

            # Acceleration (jackknife)
            n_jack = min(100, n_sub)
            jack_vals = np.zeros(n_jack)
            jack_step = max(1, n_sub // n_jack)
            for j in range(n_jack):
                mask = np.ones(n_sub, dtype=bool)
                start = j * jack_step
                end = min(start + jack_step, n_sub)
                mask[start:end] = False
                gs_j = gs_sub[mask]
                dt_j = dt_sub[mask]
                jack_vals[j] = find_theta_star(gs_j, dt_j, primes)

            jack_mean = np.mean(jack_vals)
            num = np.sum((jack_mean - jack_vals) ** 3)
            denom = 6 * np.sum((jack_mean - jack_vals) ** 2) ** 1.5
            a = num / denom if denom > 0 else 0.0

            # Adjusted percentiles
            z_lo = norm.ppf(alpha_level / 2)
            z_hi = norm.ppf(1 - alpha_level / 2)
            a1 = norm.cdf(z0 + (z0 + z_lo) / (1 - a * (z0 + z_lo)))
            a2 = norm.cdf(z0 + (z0 + z_hi) / (1 - a * (z0 + z_hi)))

            ci_lo = float(np.percentile(boot_samples, 100 * max(0, a1)))
            ci_hi = float(np.percentile(boot_samples, 100 * min(1, a2)))
            return ci_lo, ci_hi, float(z0), float(a)

        except (ImportError, ValueError):
            # Fallback to standard percentile
            ci_lo = float(np.percentile(boot_samples, 100 * alpha_level / 2))
            ci_hi = float(np.percentile(boot_samples, 100 * (1 - alpha_level / 2)))
            return ci_lo, ci_hi, 0.0, 0.0

    theta_ci_lo, theta_ci_hi, z0, accel = bca_ci(theta_boots, theta_orig)

    # Standard percentile CIs for comparison
    theta_pct_ci = [float(np.percentile(theta_boots, 2.5)),
                    float(np.percentile(theta_boots, 97.5))]
    alpha_pct_ci = [float(np.percentile(alpha_boots, 2.5)),
                    float(np.percentile(alpha_boots, 97.5))]
    R2_pct_ci = [float(np.percentile(R2_boots, 2.5)),
                 float(np.percentile(R2_boots, 97.5))]

    passed = (theta_ci_lo < THETA_CONST < theta_ci_hi
              and np.std(theta_boots) < 0.01)

    result = {
        "n_bootstrap": n_boot,
        "theta_star": {
            "original": float(theta_orig),
            "mean": float(np.mean(theta_boots)),
            "std": float(np.std(theta_boots)),
            "ci95_percentile": theta_pct_ci,
            "ci95_bca": [theta_ci_lo, theta_ci_hi],
            "bca_z0": z0,
            "bca_acceleration": accel,
            "contains_paper_value": bool(
                theta_ci_lo < THETA_CONST < theta_ci_hi),
        },
        "alpha": {
            "mean": float(np.mean(alpha_boots)),
            "std": float(np.std(alpha_boots)),
            "ci95": alpha_pct_ci,
        },
        "R2": {
            "mean": float(np.mean(R2_boots)),
            "std": float(np.std(R2_boots)),
            "ci95": R2_pct_ci,
        },
        "passed": passed,
        "runtime_s": float(elapsed),
    }

    print(f"  θ* = {np.mean(theta_boots):.4f} ± {np.std(theta_boots):.4f}")
    print(f"    95% CI (percentile): [{theta_pct_ci[0]:.4f}, "
          f"{theta_pct_ci[1]:.4f}]")
    print(f"    95% CI (BCa):        [{theta_ci_lo:.4f}, {theta_ci_hi:.4f}]")
    print(f"    Contains paper θ* = {THETA_CONST}: "
          f"{result['theta_star']['contains_paper_value']}")
    print(f"  α = {np.mean(alpha_boots):.4f} ± {np.std(alpha_boots):.4f}")
    print(f"  R² = {np.mean(R2_boots):.4f} ± {np.std(R2_boots):.4f}")
    print(f"  PASSED: {passed}  ({elapsed:.1f}s)")

    return result


# ══════════════════════════════════════════════════════════════════════
#  TEST 8: WINDOW STABILITY
# ══════════════════════════════════════════════════════════════════════

def test_window_stability(gammas, gammas_smooth, delta_true, delta_pred,
                          n_windows=10):
    """
    Per-window α and R² analysis: split zeros into contiguous windows,
    evaluate performance in each. Test for systematic drift.
    """
    print("\n" + "=" * 70)
    print(f"TEST 8: WINDOW STABILITY ({n_windows} windows)")
    print("=" * 70)

    t0 = time.time()
    n = len(gammas)
    win_size = n // n_windows

    win_alpha = []
    win_R2 = []
    win_T_mid = []

    for w in range(n_windows):
        start = w * win_size
        end = (w + 1) * win_size if w < n_windows - 1 else n

        dt_w = delta_true[start:end]
        dp_w = delta_pred[start:end]
        gs_w = gammas_smooth[start:end]

        alpha_w, R2_w = compute_alpha_R2(dt_w, dp_w)
        T_mid = float(gammas[start + (end - start) // 2])

        win_alpha.append(alpha_w)
        win_R2.append(R2_w)
        win_T_mid.append(T_mid)

        print(f"    Window {w + 1}: T ∈ [{gammas[start]:.0f}, "
              f"{gammas[end - 1]:.0f}], α = {alpha_w:.4f}, "
              f"R² = {R2_w:.4f}")

    alpha_arr = np.array(win_alpha)
    R2_arr = np.array(win_R2)
    idx_arr = np.arange(n_windows, dtype=np.float64)

    # Linear trend test for α drift
    slope_alpha, intercept_alpha = np.polyfit(idx_arr, alpha_arr, 1)
    residuals = alpha_arr - (slope_alpha * idx_arr + intercept_alpha)
    se_slope = np.sqrt(np.sum(residuals ** 2) / (n_windows - 2) /
                       np.sum((idx_arr - idx_arr.mean()) ** 2))
    t_stat = slope_alpha / se_slope if se_slope > 0 else 0.0

    # p-value for trend (two-sided, t-distribution)
    try:
        from scipy.stats import t as t_dist
        p_trend = float(2 * t_dist.sf(abs(t_stat), df=n_windows - 2))
    except ImportError:
        p_trend = None

    elapsed = time.time() - t0

    no_significant_drift = (p_trend is None or p_trend > 0.05)
    alpha_stable = np.std(alpha_arr) < 0.03
    R2_stable = np.std(R2_arr) < 0.05
    passed = no_significant_drift and alpha_stable and R2_stable

    result = {
        "n_windows": n_windows,
        "per_window": {
            "T_mid": [float(x) for x in win_T_mid],
            "alpha": [float(x) for x in win_alpha],
            "R2": [float(x) for x in win_R2],
        },
        "alpha_summary": {
            "mean": float(np.mean(alpha_arr)),
            "std": float(np.std(alpha_arr)),
            "range": float(np.ptp(alpha_arr)),
            "drift_slope": float(slope_alpha),
            "drift_t_stat": float(t_stat),
            "drift_p_value": p_trend,
            "significant_drift": not no_significant_drift,
        },
        "R2_summary": {
            "mean": float(np.mean(R2_arr)),
            "std": float(np.std(R2_arr)),
            "range": float(np.ptp(R2_arr)),
            "min": float(np.min(R2_arr)),
        },
        "passed": passed,
        "runtime_s": float(elapsed),
    }

    print(f"  α: {np.mean(alpha_arr):.4f} ± {np.std(alpha_arr):.4f}, "
          f"range = {np.ptp(alpha_arr):.4f}")
    print(f"  R²: {np.mean(R2_arr):.4f} ± {np.std(R2_arr):.4f}, "
          f"min = {np.min(R2_arr):.4f}")
    print(f"  α drift: slope = {slope_alpha:.6f}, "
          f"t = {t_stat:.2f}, p = {p_trend}")
    print(f"  PASSED: {passed}  ({elapsed:.1f}s)")

    return result


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print("╔" + "═" * 68 + "╗")
    print("║  ROBUST STATISTICAL VALIDATION — Paper 1: Prime-Spectral S(T)    ║")
    print("║  8 independent tests with comprehensive diagnostics              ║")
    print("╚" + "═" * 68 + "╝")

    # ── Load data ──
    print("\n[SETUP] Loading Riemann zeros...")
    gammas = load_zeros(N_ZEROS)
    print(f"  {len(gammas)} zeros loaded, range [{gammas[0]:.1f}, {gammas[-1]:.1f}]")

    print("[SETUP] Computing smooth zeros and corrections...")
    gammas_smooth = smooth_zeros(gammas)
    delta_true = gammas - gammas_smooth
    print(f"  δ: mean={np.mean(delta_true):.6f}, "
          f"std={np.std(delta_true):.6f}, max|δ|={np.max(np.abs(delta_true)):.6f}")

    primes = sieve_primes(10_000)
    print(f"  {len(primes)} primes up to 10,000")

    print("[SETUP] Computing S_w with optimal parameters...")
    t0 = time.time()
    Sw_opt = compute_Sw(gammas_smooth, primes,
                        theta0=THETA0_OPT, theta1=THETA1_OPT)
    delta_pred = compute_predictions(gammas_smooth, Sw_opt)
    alpha_global, R2_global = compute_alpha_R2(delta_true, delta_pred)
    print(f"  α = {alpha_global:.6f}, R² = {R2_global:.6f} "
          f"({time.time() - t0:.1f}s)")

    # ── Run all tests ──
    results = {
        "metadata": {
            "n_zeros": N_ZEROS,
            "theta0_optimal": THETA0_OPT,
            "theta1_optimal": THETA1_OPT,
            "theta_constant": THETA_CONST,
            "alpha_global": alpha_global,
            "R2_global": R2_global,
            "n_primes": len(primes),
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "script": "paper1_robust_validation.py",
        }
    }

    results["test_1_train_test_split"] = test_train_test_split(
        gammas, gammas_smooth, delta_true, primes)

    results["test_2_permutation"] = test_permutation(
        delta_true, delta_pred, n_perm=N_PERM)

    results["test_3_baselines"] = test_baselines(
        gammas_smooth, delta_true, delta_pred, primes)

    results["test_4_kfold_cv"] = test_kfold_cv(
        gammas_smooth, delta_true, primes, n_folds=N_KFOLD)

    results["test_5_monte_carlo"] = test_monte_carlo(
        gammas_smooth, delta_true, primes, n_mc=N_MC)

    results["test_6_bayesian_bic"] = test_bayesian_bic(
        gammas_smooth, delta_true, delta_pred)

    results["test_7_bootstrap_bca"] = test_bootstrap_bca(
        gammas_smooth, delta_true, primes, n_boot=N_BOOTSTRAP)

    results["test_8_window_stability"] = test_window_stability(
        gammas, gammas_smooth, delta_true, delta_pred)

    # ── Verdict ──
    test_keys = [k for k in results if k.startswith("test_")]
    n_passed = sum(1 for k in test_keys if results[k].get("passed", False))
    n_total = len(test_keys)

    results["summary"] = {
        "tests_passed": n_passed,
        "tests_total": n_total,
        "overall_verdict": "VALIDATED" if n_passed == n_total else (
            "PARTIALLY_VALIDATED" if n_passed >= n_total - 1 else "FAILED"),
        "per_test": {k: results[k]["passed"] for k in test_keys},
    }

    print("\n" + "╔" + "═" * 68 + "╗")
    print("║                        VALIDATION VERDICT                        ║")
    print("╠" + "═" * 68 + "╣")
    labels = {
        "test_1_train_test_split": "T1 Train/Test Split",
        "test_2_permutation": "T2 Permutation + LEE",
        "test_3_baselines": "T3 Baseline Comparison",
        "test_4_kfold_cv": "T4 K-Fold CV",
        "test_5_monte_carlo": "T5 Monte Carlo (100K)",
        "test_6_bayesian_bic": "T6 Bayesian BIC",
        "test_7_bootstrap_bca": "T7 Bootstrap BCa",
        "test_8_window_stability": "T8 Window Stability",
    }
    for key in test_keys:
        status = "PASS" if results[key]["passed"] else "FAIL"
        label = labels.get(key, key)
        print(f"║  {label:<30s} : {status:<6s}                          ║")
    print("╠" + "═" * 68 + "╣")
    verdict = results["summary"]["overall_verdict"]
    print(f"║  OVERALL: {n_passed}/{n_total} PASSED → "
          f"{verdict:<44s}║")
    print("╚" + "═" * 68 + "╝")

    # ── Save ──
    out_path = RESULTS_DIR / "paper1_robust_results.json"

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
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    main()
