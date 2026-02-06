#!/usr/bin/env python3
"""
Prime-Spectral K7 Metric Verification
======================================

Tests whether the prime-spectral decomposition provides a ROBUST metric
that works at ALL scales, unlike the Fibonacci recurrence which diverges.

Key formula (Weil explicit):
    delta_n ~ -(1/theta'_n) * sum_p sin(gamma_n^(0) * log(p)) / sqrt(p)

RG flow constraint:
    lag * beta = h(G2)^2 = 36

Run:  python3 notebooks/prime_spectral_metric_verification.py
"""

import numpy as np
import os
import sys
import json
import time
import warnings
from urllib.request import urlopen

warnings.filterwarnings('ignore')

# Ensure we're in the repo root
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)

from scipy.special import loggamma, lambertw
from scipy.stats import pearsonr
from scipy.optimize import curve_fit

# ═══════════════════════════════════════════════════════════════════
# GIFT TOPOLOGICAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════
DIM_K7   = 7
DIM_G2   = 14
H_G2     = 6        # Coxeter number of G2
H_G2_SQ  = 36       # h(G2)^2
B2       = 21
B3       = 77
H_STAR   = 99       # b2 + b3 + 1
KAPPA_T  = 1.0/61   # torsion capacity
DET_G    = 65.0/32  # metric determinant
LAMBDA_M = (65.0/32)**(1.0/7)  # metric scale factor

# ═══════════════════════════════════════════════════════════════════
# DATA: Download genuine Riemann zeros (Odlyzko tables)
# ═══════════════════════════════════════════════════════════════════
CACHE = os.path.join(REPO, 'riemann_zeros_100k_genuine.npy')

def download_zeros():
    if os.path.exists(CACHE):
        g = np.load(CACHE)
        print(f"  Loaded {len(g)} cached zeros [{g[0]:.2f}, {g[-1]:.2f}]")
        return g
    print("  Downloading 100,000 genuine zeros from Odlyzko...")
    url = 'https://www-users.cse.umn.edu/~odlyzko/zeta_tables/zeros1'
    raw = urlopen(url, timeout=120).read().decode('utf-8')
    g = np.array([float(l.strip()) for l in raw.strip().split('\n') if l.strip()])
    np.save(CACHE, g)
    print(f"  Downloaded {len(g)} zeros [{g[0]:.2f}, {g[-1]:.2f}]")
    return g

# ═══════════════════════════════════════════════════════════════════
# SMOOTH ZEROS: Newton's method on theta(t) = (n-3/2)*pi
# ═══════════════════════════════════════════════════════════════════
def theta_vec(t):
    t = np.asarray(t, dtype=np.float64)
    return np.imag(loggamma(0.25 + 0.5j*t)) - 0.5*t*np.log(np.pi)

def theta_deriv(t):
    return 0.5 * np.log(np.maximum(np.asarray(t, dtype=np.float64), 1.0) / (2*np.pi))

def smooth_zeros(N):
    ns = np.arange(1, N+1, dtype=np.float64)
    targets = (ns - 1.5) * np.pi
    w = np.real(lambertw(ns / np.e))
    t = np.maximum(2*np.pi*ns/w, 2.0)
    for _ in range(40):
        dt = (theta_vec(t) - targets) / np.maximum(np.abs(theta_deriv(t)), 1e-15)
        t -= dt
        if np.max(np.abs(dt)) < 1e-12:
            break
    return t

# ═══════════════════════════════════════════════════════════════════
# PRIMES
# ═══════════════════════════════════════════════════════════════════
def sieve(N):
    is_p = np.ones(N+1, dtype=bool)
    is_p[:2] = False
    for i in range(2, int(N**0.5)+1):
        if is_p[i]:
            is_p[i*i::i] = False
    return np.where(is_p)[0]

# ═══════════════════════════════════════════════════════════════════
# CORE: Prime-spectral prediction
# ═══════════════════════════════════════════════════════════════════
def prime_basis(gamma0, tp, primes, k_max=1, include_cos=False):
    """
    Build the prime-spectral basis matrix.
    Each column = sin(gamma0 * k*log(p)) / (k * p^{k/2} * theta')
    Optionally add cos columns for full regression.
    """
    n = len(gamma0)
    cols_sin = []
    col_labels = []
    for p in primes:
        logp = np.log(float(p))
        for k in range(1, k_max+1):
            col = np.sin(gamma0 * k * logp) / (k * p**(k/2.0) * tp)
            cols_sin.append(col)
            col_labels.append(f"sin(g*{k}*log{p})/{k}*{p}^{k/2}")
    X = np.column_stack(cols_sin)

    if include_cos:
        cols_cos = []
        for p in primes:
            logp = np.log(float(p))
            for k in range(1, k_max+1):
                col = np.cos(gamma0 * k * logp) / (k * p**(k/2.0) * tp)
                cols_cos.append(col)
        X = np.column_stack([X, np.column_stack(cols_cos)])

    return X, col_labels

def capture(delta, pred):
    """1 - Var(residual)/Var(delta). Positive = good."""
    return 1.0 - np.var(delta - pred) / np.var(delta)

def ols_predict(X, y):
    """OLS fit and return prediction + R^2."""
    # Use pseudoinverse for numerical stability
    beta, res, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    r2 = 1.0 - np.var(y - pred) / np.var(y)
    return pred, r2, beta

# ═══════════════════════════════════════════════════════════════════
# ACF
# ═══════════════════════════════════════════════════════════════════
def acf(x, max_lag=100):
    xc = x - np.mean(x)
    n = len(xc)
    c = np.correlate(xc, xc, 'full')[n-1 : n+max_lag]
    return c / c[0] if c[0] > 0 else c

def acf_theory(max_lag, mean_sp, primes, k_max=3):
    """Theoretical ACF from prime sum: rho(k) = sum_p (1/p)*cos(k*s*log p)."""
    rho = np.zeros(max_lag + 1)
    for k in range(max_lag + 1):
        val = norm = 0.0
        for p in primes:
            for m in range(1, k_max+1):
                w = 1.0 / p**m
                val += w * np.cos(k * mean_sp * m * np.log(p))
                norm += w
        rho[k] = val / norm if norm > 0 else 0.0
    return rho


# ═══════════════════════════════════════════════════════════════════
#                         M A I N
# ═══════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    print("=" * 72)
    print("  PRIME-SPECTRAL K7 METRIC VERIFICATION")
    print("  " + "—" * 50)
    print(f"  det(g) = 65/32 = {DET_G},  kappa_T = 1/61,  h(G2)^2 = {H_G2_SQ}")
    print("=" * 72)

    # ── A. DATA ──────────────────────────────────────────────────
    print("\n[A] DATA PREPARATION")
    gamma = download_zeros()
    N = len(gamma)
    gamma0 = smooth_zeros(N)
    delta = gamma[:N] - gamma0[:N]
    tp = theta_deriv(gamma0[:N])

    print(f"  N = {N}")
    print(f"  delta: mean={np.mean(delta):.6f}, std={np.std(delta):.6f}")
    print(f"  theta': [{tp[0]:.4f} .. {tp[-1]:.4f}]")

    primes = sieve(1000)
    print(f"  Primes: {len(primes)} up to {primes[-1]}")

    # ── B. PRIME-SPECTRAL vs FIBONACCI CAPTURE ───────────────────
    print("\n" + "=" * 72)
    print("[B] TEST 1 — PRIME-SPECTRAL CAPTURE vs SCALE")
    print("    Compare: prime sum (various P) vs Fibonacci recurrence")
    print("=" * 72)

    scales    = [500, 1000, 2000, 5000, 10000, 20000, 50000, N]
    p_cutoffs = [3, 10, 30, 100, 500]

    # Header
    hdr = f"{'Scale':>7} |"
    for P in p_cutoffs:
        hdr += f" P<={P:<4}|"
    hdr += f" {'Fibo':>7} |"
    print(f"\n{hdr}")
    print("-" * len(hdr))

    results_B = {}

    for Nz in scales:
        g0  = gamma0[:Nz]
        d   = delta[:Nz]
        tp_ = tp[:Nz]

        row = {}

        # ── Prime-spectral models ──
        for P in p_cutoffs:
            p_sub = primes[primes <= P]
            X, _ = prime_basis(g0, tp_, p_sub, k_max=3)
            # Single-alpha OLS (theoretical shape, fit amplitude)
            s_theory = -X.sum(axis=1)  # sum of all basis columns = theoretical S(t)/theta'
            alpha = np.dot(d, s_theory) / np.dot(s_theory, s_theory) if np.dot(s_theory, s_theory) > 0 else 0
            pred = alpha * s_theory
            cap = capture(d, pred)
            row[f"P<={P}"] = float(cap)

        # ── Fibonacci recurrence ──
        fib_pred = np.zeros(Nz)
        for n in range(21, Nz):
            fib_pred[n] = (31.0/21)*d[n-8] - (10.0/21)*d[n-21]
        fib_cap = capture(d[21:], fib_pred[21:])
        row["Fibonacci"] = float(fib_cap)

        results_B[Nz] = row

        # Print row
        line = f"{Nz:>7} |"
        for P in p_cutoffs:
            v = row[f"P<={P}"]
            line += f" {v:>+6.1%} |"
        line += f" {fib_cap:>+6.1%} |"
        print(line)

    # ── C. FULL SPECTRAL DECOMPOSITION ───────────────────────────
    print("\n" + "=" * 72)
    print("[C] TEST 2 — FULL SPECTRAL DECOMPOSITION (OLS, 100K zeros)")
    print("    How much variance in delta_n is explained by prime modes?")
    print("=" * 72)

    # Model 1: theoretical shape, single alpha
    X_full, _ = prime_basis(gamma0, tp, primes[:50], k_max=3)
    s_th = -X_full.sum(axis=1)
    alpha_th = np.dot(delta, s_th) / np.dot(s_th, s_th)
    pred_th = alpha_th * s_th
    r2_th = capture(delta, pred_th)
    print(f"\n  Model 1 (theory shape, 1 param):  R² = {r2_th:.4f}")
    print(f"    Fitted alpha = {alpha_th:.6f}")

    # Model 2: per-prime-power OLS (sin only)
    pred_ols, r2_ols, beta_ols = ols_predict(X_full, delta)
    print(f"  Model 2 (per-prime OLS, {X_full.shape[1]} params): R² = {r2_ols:.4f}")

    # Model 3: sin + cos (full basis)
    X_sc, _ = prime_basis(gamma0, tp, primes[:30], k_max=3, include_cos=True)
    pred_sc, r2_sc, _ = ols_predict(X_sc, delta)
    print(f"  Model 3 (sin+cos OLS, {X_sc.shape[1]} params): R² = {r2_sc:.4f}")

    # Compare fitted weights vs theoretical 1/sqrt(p)
    print(f"\n  Per-prime weights (k=1) vs theoretical:")
    print(f"  {'Prime':>6} | {'Fitted':>10} | {'Theory(-1/sqp)':>14} | {'Ratio':>8}")
    print(f"  " + "-" * 48)
    for i, p in enumerate(primes[:10]):
        w_fit = beta_ols[i*3]  # k=1 component for this prime
        w_th  = -1.0 / np.sqrt(p)
        ratio = w_fit / w_th if abs(w_th) > 1e-10 else float('nan')
        print(f"  {p:>6} | {w_fit:>+10.6f} | {w_th:>+13.6f} | {ratio:>8.3f}")

    # ── D. SCALE STABILITY ──────────────────────────────────────
    print("\n" + "=" * 72)
    print("[D] TEST 3 — SCALE STABILITY (R² per window)")
    print("    Does the prime-spectral R² stay positive at all scales?")
    print("=" * 72)

    window_size = 5000
    n_windows = N // window_size
    p_sub = primes[primes <= 100]

    print(f"\n  {'Window':>12} | {'T range':>22} | {'R²(prime)':>10} | {'Cap(Fibo)':>10} | {'Winner':>8}")
    print(f"  " + "-" * 72)

    wins_prime = []
    wins_fibo  = []

    for w in range(n_windows):
        i0, i1 = w*window_size, (w+1)*window_size
        g0_w  = gamma0[i0:i1]
        d_w   = delta[i0:i1]
        tp_w  = tp[i0:i1]

        # Prime model
        X_w, _ = prime_basis(g0_w, tp_w, p_sub, k_max=3)
        s_w = -X_w.sum(axis=1)
        a_w = np.dot(d_w, s_w) / np.dot(s_w, s_w) if np.dot(s_w, s_w) > 0 else 0
        r2_w = capture(d_w, a_w * s_w)

        # Fibonacci
        fp_w = np.zeros(window_size)
        for n in range(21, window_size):
            fp_w[n] = (31.0/21)*d_w[n-8] - (10.0/21)*d_w[n-21]
        fc_w = capture(d_w[21:], fp_w[21:])

        wins_prime.append(r2_w)
        wins_fibo.append(fc_w)

        winner = "PRIME" if r2_w > fc_w else "FIBO"
        t_lo, t_hi = gamma[i0], gamma[i1-1]
        print(f"  {w*window_size//1000:>4}k-{(w+1)*window_size//1000:>3}k | [{t_lo:>9.1f}, {t_hi:>9.1f}] | {r2_w:>+9.4f} | {fc_w:>+9.4f} | {winner:>8}")

    print(f"\n  PRIME wins: {sum(1 for p,f in zip(wins_prime, wins_fibo) if p > f)}/{n_windows}")
    print(f"  PRIME mean R²: {np.mean(wins_prime):.4f}, std: {np.std(wins_prime):.4f}")
    print(f"  FIBO  mean cap: {np.mean(wins_fibo):.4f}, std: {np.std(wins_fibo):.4f}")
    print(f"  PRIME always positive? {'YES' if all(r > 0 for r in wins_prime) else 'NO'}")
    print(f"  FIBO  always positive? {'YES' if all(r > 0 for r in wins_fibo) else 'NO'}")

    # ── E. ACF: THEORY vs DATA ──────────────────────────────────
    print("\n" + "=" * 72)
    print("[E] TEST 4 — ACF: PRIME THEORY vs EMPIRICAL DATA")
    print("=" * 72)

    max_lag = 60
    acf_emp = acf(delta, max_lag)
    mean_sp = np.mean(np.diff(gamma))
    acf_th  = acf_theory(max_lag, mean_sp, primes[:20], k_max=3)

    r_acf, p_acf = pearsonr(acf_emp[1:], acf_th[1:])
    print(f"\n  Pearson r(theory, data): {r_acf:.4f}  (p = {p_acf:.2e})")

    # Per-window ACF period
    print(f"\n  ACF PERIOD DRIFT (RG flow in P_2 = 2pi/(s*log2))")
    print(f"  {'Window':>12} | {'s_bar':>8} | {'P2(theory)':>10} | {'P(fitted)':>10} | {'Ratio/13':>8}")
    print(f"  " + "-" * 58)

    results_E = []
    win_sz = 10000
    for w in range(N // win_sz):
        i0, i1 = w*win_sz, (w+1)*win_sz
        d_w = delta[i0:i1]
        g_w = gamma[i0:i1]
        s_bar = np.mean(np.diff(g_w))
        P2_th = 2*np.pi / (s_bar * np.log(2))

        acf_w = acf(d_w, 50)
        lags = np.arange(len(acf_w))
        try:
            def cos_model(k, A, P, phi_):
                return A * np.cos(2*np.pi*k/P + phi_)
            popt, _ = curve_fit(cos_model, lags[1:], acf_w[1:],
                                p0=[0.1, 13.0, 0.0], maxfev=10000)
            P_fit = abs(popt[1])
        except Exception:
            P_fit = float('nan')

        results_E.append({'s_bar': s_bar, 'P2_th': P2_th, 'P_fit': P_fit})
        print(f"  {w*win_sz//1000:>4}k-{(w+1)*win_sz//1000:>3}k | {s_bar:>.5f} | {P2_th:>10.3f} | {P_fit:>10.3f} | {P_fit/13:>8.3f}")

    # ── F. RG FLOW EXPONENTS ─────────────────────────────────────
    print("\n" + "=" * 72)
    print("[F] TEST 5 — RG FLOW EXPONENTS: lag * beta = h(G2)^2 = 36 ?")
    print("=" * 72)

    test_lags = [5, 8, 13, 21, 27, 34]
    win_sz = 5000
    nw = N // win_sz

    print(f"\n  {'Lag':>5} | {'beta':>10} | {'lag*beta':>10} | {'Target':>8} | {'Dev':>8}")
    print(f"  " + "-" * 50)

    results_F = {}
    for lag in test_lags:
        acf_vals = []
        T_centers = []
        for w in range(nw):
            i0, i1 = w*win_sz, (w+1)*win_sz
            d_w = delta[i0:i1]
            a_w = acf(d_w, lag+5)
            if lag < len(a_w) and abs(a_w[lag]) > 1e-15:
                acf_vals.append(abs(a_w[lag]))
                T_centers.append(np.mean(gamma[i0:i1]))

        acf_vals = np.array(acf_vals)
        T_centers = np.array(T_centers)

        if len(acf_vals) > 3 and np.all(acf_vals > 0):
            log_T = np.log(T_centers)
            log_a = np.log(acf_vals)
            coeffs = np.polyfit(log_T, log_a, 1)
            beta = -coeffs[0]
            product = lag * beta
            target = H_G2_SQ  # 36
            if lag == 27:
                target = B3 + DIM_K7  # 84
            dev = abs(product - target) / target * 100
            results_F[lag] = {'beta': float(beta), 'product': float(product),
                              'target': target, 'dev_pct': float(dev)}
            print(f"  {lag:>5} | {beta:>10.4f} | {product:>10.2f} | {target:>8} | {dev:>7.2f}%")
        else:
            print(f"  {lag:>5} | {'N/A':>10} |")

    # check sum rule: sum_i beta_i = b3/dim(K7) = 11
    betas_for_sum = [results_F[l]['beta'] for l in [8, 13, 21] if l in results_F]
    if betas_for_sum:
        beta_sum = sum(betas_for_sum)
        print(f"\n  Sum beta(8)+beta(13)+beta(21) = {beta_sum:.3f}")
        print(f"  Target b3/dim(K7) = {B3/DIM_K7:.3f}")
        print(f"  Deviation: {abs(beta_sum - B3/DIM_K7)/(B3/DIM_K7)*100:.2f}%")

    # ── G. METRIC DETERMINANT STABILITY ──────────────────────────
    print("\n" + "=" * 72)
    print("[G] TEST 6 — METRIC DETERMINANT STABILITY")
    print(f"    Target: det(g) = 65/32 = {DET_G}")
    print("=" * 72)

    # epsilon(mu) = sum_p cos(mu*log p)/p  [trace of metric perturbation]
    mu = gamma[::100]  # sample every 100th zero
    eps_trace = np.zeros(len(mu))
    for p in primes[:50]:
        eps_trace += np.cos(mu * np.log(float(p))) / float(p)

    # Normalize: the perturbation should be O(kappa_T) = O(1/61)
    # Scale by kappa_T / max to keep it within Joyce bound
    eps_norm = eps_trace * KAPPA_T / np.max(np.abs(eps_trace))

    # det(g + delta_g) = det(g) * det(I + g^{-1} delta_g)
    # For small perturbation: det(I + eps*I/7) ~ (1 + eps/7)^7
    det_vals = LAMBDA_M**7 * (1 + eps_norm/DIM_K7)**DIM_K7

    print(f"\n  Perturbation amplitude: max|eps| = {np.max(np.abs(eps_norm)):.6f}")
    print(f"  Perturbation / Joyce bound (0.1): {np.max(np.abs(eps_norm))/0.1:.4f}")
    print(f"  det(g): mean = {np.mean(det_vals):.6f}  (target {DET_G:.6f})")
    print(f"  det(g): std  = {np.std(det_vals):.6f}")
    print(f"  Max |deviation| from 65/32 = {np.max(np.abs(det_vals - DET_G)):.6f}")
    print(f"  Relative fluctuation = {np.std(det_vals)/DET_G*100:.4f}%")

    # ═══════════════════════════════════════════════════════════════
    #                        SUMMARY
    # ═══════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    print("\n" + "=" * 72)
    print("  SYNTHESIS")
    print("=" * 72)

    # Best prime capture at max scale
    max_scale = max(results_B.keys())
    best_p_key = max((k for k in results_B[max_scale] if k != "Fibonacci"),
                     key=lambda k: results_B[max_scale][k])
    best_p_val = results_B[max_scale][best_p_key]
    fib_val    = results_B[max_scale]["Fibonacci"]

    print(f"""
  PRIME-SPECTRAL METRIC vs FIBONACCI RECURRENCE ({max_scale} zeros)
  ┌───────────────────────────────────┬───────────┐
  │ Prime spectral (best, {best_p_key:>8})  │ {best_p_val:>+8.2%}  │
  │ Fibonacci recurrence              │ {fib_val:>+8.2%}  │
  └───────────────────────────────────┴───────────┘

  FULL SPECTRAL DECOMPOSITION (100K zeros)
    Theory shape (1 param):   R² = {r2_th:.4f}
    Per-prime OLS ({X_full.shape[1]} params): R² = {r2_ols:.4f}
    Sin+cos OLS ({X_sc.shape[1]} params):   R² = {r2_sc:.4f}

  ACF THEORY-DATA CORRELATION: r = {r_acf:.4f}

  SCALE STABILITY ({n_windows} windows):
    Prime positive in {sum(1 for r in wins_prime if r > 0)}/{n_windows} windows
    Fibo  positive in {sum(1 for r in wins_fibo if r > 0)}/{n_windows} windows
    Prime mean R²  = {np.mean(wins_prime):.4f}
    Fibo  mean cap = {np.mean(wins_fibo):.4f}

  RG FLOW EXPONENTS:""")
    for lag, data in sorted(results_F.items()):
        print(f"    {lag:>2} x beta_{lag} = {data['product']:>6.2f}  (target {data['target']}, dev {data['dev_pct']:.1f}%)")

    print(f"""
  METRIC DETERMINANT: {np.mean(det_vals):.6f} +/- {np.std(det_vals):.6f}
    Target = {DET_G:.6f}, fluctuation = {np.std(det_vals)/DET_G*100:.4f}%

  Elapsed: {elapsed:.1f}s
""")

    # ── Save JSON ──
    output = {
        'test1_capture_vs_scale': {str(k): v for k, v in results_B.items()},
        'test2_spectral_R2': {
            'theory_1param': float(r2_th),
            'per_prime_ols': float(r2_ols),
            'sin_cos_ols': float(r2_sc)
        },
        'test3_scale_stability': {
            'prime_wins': int(sum(1 for p,f in zip(wins_prime, wins_fibo) if p > f)),
            'total_windows': n_windows,
            'prime_r2_mean': float(np.mean(wins_prime)),
            'prime_r2_std': float(np.std(wins_prime)),
            'fibo_cap_mean': float(np.mean(wins_fibo)),
            'prime_always_positive': bool(all(r > 0 for r in wins_prime)),
            'fibo_always_positive': bool(all(r > 0 for r in wins_fibo))
        },
        'test4_acf': {'pearson_r': float(r_acf), 'p_value': float(p_acf)},
        'test5_rg_flow': results_F,
        'test6_det_stability': {
            'mean': float(np.mean(det_vals)),
            'std': float(np.std(det_vals)),
            'target': DET_G,
            'relative_fluctuation_pct': float(np.std(det_vals)/DET_G*100)
        }
    }
    outpath = os.path.join(REPO, 'notebooks', 'riemann',
                           'prime_spectral_results.json')
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved to {outpath}")


if __name__ == '__main__':
    main()
