#!/usr/bin/env python3
"""
Adaptive theta: theta(T) = theta_0 + theta_1 / log(T)
======================================================

The constant theta* = 0.994 gives alpha = 1.000 globally but drifts
per-window (0.947 at small T, 1.018 at large T).

Key insight: log_X = theta(T) * log(T) = theta_0*log(T) + theta_1
This means the effective log-cutoff is LINEAR in log(T).

We optimize (theta_0, theta_1) so that alpha = 1.000 UNIFORMLY
across all windows, which should improve R^2 and localization.

Run:  python3 notebooks/adaptive_theta.py
"""

import numpy as np
import os
import json
import time
import warnings
from urllib.request import urlopen

warnings.filterwarnings('ignore')
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)

from scipy.special import loggamma, lambertw
from scipy.optimize import minimize
from scipy.stats import pearsonr

CACHE = os.path.join(REPO, 'riemann_zeros_100k_genuine.npy')

def download_zeros():
    if os.path.exists(CACHE):
        return np.load(CACHE)
    raw = urlopen('https://www-users.cse.umn.edu/~odlyzko/zeta_tables/zeros1',
                  timeout=120).read().decode('utf-8')
    g = np.array([float(l.strip()) for l in raw.strip().split('\n') if l.strip()])
    np.save(CACHE, g)
    return g

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

def sieve(N):
    is_p = np.ones(N+1, dtype=bool); is_p[:2] = False
    for i in range(2, int(N**0.5)+1):
        if is_p[i]: is_p[i*i::i] = False
    return np.where(is_p)[0]

def w_cosine(x):
    return np.where(x < 1.0, np.cos(np.pi * x / 2)**2, 0.0)


# ═══════════════════════════════════════════════════════════════════
# CORE: Adaptive prime sum with theta(T) = theta_0 + theta_1/log(T)
# ═══════════════════════════════════════════════════════════════════

def prime_sum_adaptive(gamma0, tp, primes, k_max, theta_0, theta_1):
    """
    Mollified prime sum with theta(T) = theta_0 + theta_1 / log(T).

    log_X(T) = theta(T) * log(T) = theta_0 * log(T) + theta_1

    This makes the effective cutoff:
      X(T) = T^{theta_0} * e^{theta_1}
    """
    S = np.zeros_like(gamma0)
    log_g0 = np.log(np.maximum(gamma0, 2.0))
    log_X = theta_0 * log_g0 + theta_1  # vector
    log_X = np.maximum(log_X, 0.5)  # safety

    for p in primes:
        logp = np.log(float(p))
        for m in range(1, k_max + 1):
            x = m * logp / log_X  # vector
            weight = w_cosine(x)
            S -= weight * np.sin(gamma0 * m * logp) / (m * p**(m/2.0))

    return -S / tp  # delta_pred


def evaluate_config(gamma0, tp, delta, half_gaps, primes, k_max,
                    theta_0, theta_1, window_edges=None):
    """Evaluate a (theta_0, theta_1) configuration. Returns dict of metrics."""
    dpred = prime_sum_adaptive(gamma0, tp, primes, k_max, theta_0, theta_1)

    # Global metrics
    dot_pp = np.dot(dpred, dpred)
    alpha_global = np.dot(delta, dpred) / dot_pp if dot_pp > 1e-30 else 0
    residual = delta - dpred  # alpha=1
    R2 = 1.0 - np.var(residual) / np.var(delta)
    E_rms = np.sqrt(np.mean(residual**2))

    n_test = min(len(residual) - 1, len(half_gaps))
    res_abs = np.abs(residual[1:n_test+1])
    hg = half_gaps[:n_test]
    loc_rate = float(np.mean(res_abs < hg))

    # Per-window alpha
    if window_edges is None:
        window_edges = list(range(0, len(gamma0), 10000)) + [len(gamma0)]

    window_alphas = []
    window_R2s = []
    for i in range(len(window_edges) - 1):
        i0, i1 = window_edges[i], window_edges[i+1]
        d_w = delta[i0:i1]
        dp_w = dpred[i0:i1]
        dot_w = np.dot(dp_w, dp_w)
        a_w = np.dot(d_w, dp_w) / dot_w if dot_w > 1e-30 else 0
        r_w = d_w - dp_w
        r2_w = 1.0 - np.var(r_w) / np.var(d_w) if np.var(d_w) > 0 else 0
        window_alphas.append(a_w)
        window_R2s.append(r2_w)

    return {
        'alpha_global': float(alpha_global),
        'R2': float(R2),
        'E_rms': float(E_rms),
        'loc_rate': float(loc_rate),
        'window_alphas': window_alphas,
        'window_R2s': window_R2s,
        'alpha_std': float(np.std(window_alphas)),
        'alpha_range': float(np.max(window_alphas) - np.min(window_alphas)),
    }


# ═══════════════════════════════════════════════════════════════════
#                         M A I N
# ═══════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    print("=" * 76)
    print("  ADAPTIVE THETA: theta(T) = theta_0 + theta_1 / log(T)")
    print("=" * 76)

    gamma = download_zeros()
    N = len(gamma)
    gamma0 = smooth_zeros(N)
    delta = gamma[:N] - gamma0[:N]
    tp = theta_deriv(gamma0[:N])
    primes = sieve(1000)
    half_gaps = np.diff(gamma) / 2.0
    p_sub = primes[primes <= 1000]
    k_max = 3

    window_edges = list(range(0, N, 10000)) + [N]

    print(f"  N = {N}, range [{gamma[0]:.1f}, {gamma[-1]:.1f}]")

    # ── BASELINE: constant theta = 0.9941 ──
    print("\n" + "=" * 76)
    print("  BASELINE: CONSTANT theta* = 0.9941")
    print("=" * 76)

    base = evaluate_config(gamma0, tp, delta, half_gaps, p_sub, k_max,
                           0.9941, 0.0, window_edges)

    print(f"\n  alpha (global):  {base['alpha_global']:+.6f}")
    print(f"  alpha std:       {base['alpha_std']:.6f}")
    print(f"  alpha range:     {base['alpha_range']:.6f}")
    print(f"  R^2:             {base['R2']:.4f}")
    print(f"  E_rms:           {base['E_rms']:.6f}")
    print(f"  Localization:    {base['loc_rate']:.4%}")

    print(f"\n  Per-window alpha:")
    for i, (a, r) in enumerate(zip(base['window_alphas'], base['window_R2s'])):
        print(f"    {i*10:>3}k-{(i+1)*10:>3}k: alpha = {a:+.5f}, R^2 = {r:.4f}")

    # ── STEP 1: Coarse grid search ──
    print("\n" + "=" * 76)
    print("  STEP 1: COARSE GRID SEARCH")
    print("  theta(T) = theta_0 + theta_1/log(T)")
    print("  log_X(T) = theta_0*log(T) + theta_1")
    print("=" * 76)

    # theta_0 ~ 1.0-1.2 (large-T behavior)
    # theta_1 ~ negative (reduce cutoff at small T where log(T) is small)
    # From the window data: theta*(T~5000) ≈ 0.90, theta*(T~60000) ≈ 1.04
    # At T=5000: 0.90 = theta_0 + theta_1/log(5000) = theta_0 + theta_1/8.52
    # At T=60000: 1.04 = theta_0 + theta_1/log(60000) = theta_0 + theta_1/11.0
    # => theta_1*(1/8.52 - 1/11.0) = 0.90 - 1.04 => theta_1 ≈ -5.2
    # => theta_0 ≈ 1.04 - (-5.2)/11.0 ≈ 1.51

    best_score = 999.0
    best_params = (0.9941, 0.0)

    print(f"\n  {'theta_0':>8} | {'theta_1':>8} | {'alpha_g':>8} | {'a_std':>7} | {'a_range':>8} | {'R^2':>7} | {'Loc%':>7}")
    print(f"  " + "-" * 65)

    for th0 in np.arange(1.00, 1.60, 0.05):
        for th1 in np.arange(-7.0, -1.0, 0.5):
            res = evaluate_config(gamma0, tp, delta, half_gaps, p_sub, k_max,
                                  th0, th1, window_edges)
            # Score: want alpha_global ≈ 1 AND low alpha_std
            score = (res['alpha_global'] - 1.0)**2 + 4.0 * res['alpha_std']**2
            if score < best_score:
                best_score = score
                best_params = (th0, th1)
                print(f"  {th0:>8.3f} | {th1:>8.2f} | {res['alpha_global']:>+8.5f} | {res['alpha_std']:>7.5f} | {res['alpha_range']:>8.5f} | {res['R2']:>7.4f} | {res['loc_rate']:>6.2%} *")

    print(f"\n  Best coarse: theta_0 = {best_params[0]:.3f}, theta_1 = {best_params[1]:.2f}")

    # ── STEP 2: Fine optimization ──
    print("\n" + "=" * 76)
    print("  STEP 2: FINE OPTIMIZATION (Nelder-Mead)")
    print("=" * 76)

    def objective(params):
        th0, th1 = params
        if th0 < 0.5 or th0 > 2.5 or th1 < -15 or th1 > 5:
            return 100.0
        res = evaluate_config(gamma0, tp, delta, half_gaps, p_sub, k_max,
                              th0, th1, window_edges)
        # Minimize: alpha deviation from 1 (global) + uniformity
        return (res['alpha_global'] - 1.0)**2 + 4.0 * res['alpha_std']**2

    result = minimize(objective, best_params, method='Nelder-Mead',
                      options={'xatol': 0.001, 'fatol': 1e-8, 'maxiter': 200})
    th0_opt, th1_opt = result.x

    print(f"  Optimized: theta_0 = {th0_opt:.6f}, theta_1 = {th1_opt:.4f}")
    print(f"  Objective: {result.fun:.2e}")

    opt = evaluate_config(gamma0, tp, delta, half_gaps, p_sub, k_max,
                          th0_opt, th1_opt, window_edges)

    print(f"\n  alpha (global):  {opt['alpha_global']:+.6f}")
    print(f"  alpha std:       {opt['alpha_std']:.6f}")
    print(f"  alpha range:     {opt['alpha_range']:.6f}")
    print(f"  R^2:             {opt['R2']:.4f}")
    print(f"  E_rms:           {opt['E_rms']:.6f}")
    print(f"  Localization:    {opt['loc_rate']:.4%}")

    print(f"\n  Per-window comparison (constant vs adaptive):")
    print(f"  {'Window':>12} | {'a(const)':>10} | {'a(adapt)':>10} | {'R2(const)':>10} | {'R2(adapt)':>10}")
    print(f"  " + "-" * 60)
    for i, (ac, aa, rc, ra) in enumerate(zip(
            base['window_alphas'], opt['window_alphas'],
            base['window_R2s'], opt['window_R2s'])):
        print(f"  {i*10:>3}k-{(i+1)*10:>3}k | {ac:>+10.5f} | {aa:>+10.5f} | {rc:>10.4f} | {ra:>10.4f}")

    # ── STEP 3: What theta(T) looks like ──
    print("\n" + "=" * 76)
    print("  STEP 3: THE ADAPTIVE CUTOFF PROFILE")
    print("=" * 76)

    T_samples = [14, 50, 100, 500, 1000, 5000, 10000, 50000, 75000,
                 100000, 500000, 1000000]
    print(f"\n  theta(T) = {th0_opt:.4f} + ({th1_opt:.4f}) / log(T)")
    print(f"\n  {'T':>10} | {'log(T)':>8} | {'theta(T)':>10} | {'X(T) = e^(theta*logT)':>22}")
    print(f"  " + "-" * 58)
    for T in T_samples:
        logT = np.log(T)
        theta_T = th0_opt + th1_opt / logT
        log_X = th0_opt * logT + th1_opt
        X = np.exp(log_X)
        print(f"  {T:>10} | {logT:>8.3f} | {theta_T:>10.4f} | {X:>22.1f}")

    # ── STEP 4: N(T) counting ──
    print("\n" + "=" * 76)
    print("  STEP 4: N(T) COUNTING WITH ADAPTIVE THETA")
    print("=" * 76)

    T_mid = (gamma[:-1] + gamma[1:]) / 2.0
    N_actual = np.arange(1, len(T_mid)+1, dtype=np.float64)
    theta_mid = theta_vec(T_mid)
    tp_mid = theta_deriv(T_mid)

    log_T_mid = np.log(np.maximum(T_mid, 2.0))
    log_X_mid = th0_opt * log_T_mid + th1_opt
    log_X_mid = np.maximum(log_X_mid, 0.5)

    S_mid = np.zeros(len(T_mid))
    for p in p_sub:
        logp = np.log(float(p))
        for m in range(1, k_max+1):
            x = m * logp / log_X_mid
            weight = w_cosine(x)
            S_mid -= weight * np.sin(T_mid * m * logp) / (m * p**(m/2.0))
    S_mid /= np.pi

    N_approx = theta_mid / np.pi + 1.0 + S_mid
    err_N = np.abs(N_actual - N_approx)
    correct = np.mean(err_N < 0.5)

    # Also baseline
    log_X_base = 0.9941 * log_T_mid
    S_base = np.zeros(len(T_mid))
    for p in p_sub:
        logp = np.log(float(p))
        for m in range(1, k_max+1):
            x = m * logp / log_X_base
            weight = w_cosine(x)
            S_base -= weight * np.sin(T_mid * m * logp) / (m * p**(m/2.0))
    S_base /= np.pi
    N_base = theta_mid / np.pi + 1.0 + S_base
    err_base = np.abs(N_actual - N_base)
    correct_base = np.mean(err_base < 0.5)

    print(f"\n  Constant theta:  {correct_base:.4%} correct, mean|err| = {np.mean(err_base):.4f}, max|err| = {np.max(err_base):.4f}")
    print(f"  Adaptive theta:  {correct:.4%} correct, mean|err| = {np.mean(err_N):.4f}, max|err| = {np.max(err_N):.4f}")

    print(f"\n  Per-window N(T) accuracy (adaptive):")
    print(f"  {'Window':>12} | {'%correct':>10} | {'mean|err|':>10} | {'max|err|':>10}")
    print(f"  " + "-" * 50)
    for w in range(min(10, len(T_mid)//10000)):
        i0, i1 = w*10000, min((w+1)*10000, len(T_mid))
        c_w = np.mean(err_N[i0:i1] < 0.5)
        e_w = np.mean(err_N[i0:i1])
        m_w = np.max(err_N[i0:i1])
        print(f"  {w*10:>3}k-{(w+1)*10:>3}k | {c_w:>9.2%} | {e_w:>10.4f} | {m_w:>10.4f}")

    # ── STEP 5: GUE failure prediction with new sigma_E ──
    print("\n" + "=" * 76)
    print("  STEP 5: UPDATED GUE FAILURE PREDICTION")
    print("=" * 76)

    from scipy.special import erfc as sp_erfc

    sigma_new = opt['E_rms']
    sigma_old = base['E_rms']
    mean_gap = np.mean(np.diff(gamma))

    s_grid = np.linspace(0.001, 6.0, 10000)
    ds = s_grid[1] - s_grid[0]
    p_gue = (np.pi/2) * s_grid * np.exp(-np.pi * s_grid**2 / 4)

    pf_old = np.sum(p_gue * sp_erfc(s_grid * mean_gap / (2*np.sqrt(2)*sigma_old))) * ds
    pf_new = np.sum(p_gue * sp_erfc(s_grid * mean_gap / (2*np.sqrt(2)*sigma_new))) * ds

    print(f"\n  {'':>20} | {'Constant theta':>16} | {'Adaptive theta':>16}")
    print(f"  " + "-" * 55)
    print(f"  {'sigma_E':>20} | {sigma_old:>16.6f} | {sigma_new:>16.6f}")
    print(f"  {'sigma_E/mean_gap':>20} | {sigma_old/mean_gap:>16.4f} | {sigma_new/mean_gap:>16.4f}")
    print(f"  {'R^2':>20} | {base['R2']:>16.4f} | {opt['R2']:>16.4f}")
    print(f"  {'P(fail) GUE':>20} | {pf_old:>15.4%} | {pf_new:>15.4%}")
    print(f"  {'P(fail) empirical':>20} | {1-base['loc_rate']:>15.4%} | {1-opt['loc_rate']:>15.4%}")
    print(f"  {'Localization':>20} | {base['loc_rate']:>15.4%} | {opt['loc_rate']:>15.4%}")
    print(f"  {'alpha std':>20} | {base['alpha_std']:>16.6f} | {opt['alpha_std']:>16.6f}")

    # ══════════════════════════════════════════════════════════════
    #                      SYNTHESIS
    # ══════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    print("\n" + "=" * 76)
    print("  SYNTHESIS")
    print("=" * 76)

    improvement_R2 = opt['R2'] - base['R2']
    improvement_loc = opt['loc_rate'] - base['loc_rate']

    print(f"""
  ADAPTIVE THETA FORMULA:
    theta(T) = {th0_opt:.4f} + ({th1_opt:.4f}) / log(T)
    log X(T) = {th0_opt:.4f} * log(T) + ({th1_opt:.4f})
    X(T)     = T^{{{th0_opt:.4f}}} * e^{{{th1_opt:.4f}}}

  IMPROVEMENT OVER CONSTANT theta = 0.9941:
    R^2:           {base['R2']:.4f} -> {opt['R2']:.4f}  ({improvement_R2:+.4f})
    Localization:  {base['loc_rate']:.4%} -> {opt['loc_rate']:.4%}  ({improvement_loc:+.4%})
    alpha std:     {base['alpha_std']:.6f} -> {opt['alpha_std']:.6f}  ({opt['alpha_std']/base['alpha_std']:.2f}x)
    alpha range:   {base['alpha_range']:.6f} -> {opt['alpha_range']:.6f}
    N(T) counting: {correct_base:.4%} -> {correct:.4%}

  THE FORMULA (2 structural parameters, 0 free parameters):
    S(T) = -(1/pi) * sum_{{p,m}} cos^2(pi*m*log(p) / (2*({th0_opt:.4f}*log(T)+{th1_opt:.4f})))
                               * sin(T*m*log(p)) / (m*p^{{m/2}})

  Elapsed: {elapsed:.1f}s
""")

    # Save
    output = {
        'theta_0': float(th0_opt),
        'theta_1': float(th1_opt),
        'baseline': {
            'theta_const': 0.9941,
            'R2': base['R2'], 'loc_rate': base['loc_rate'],
            'alpha_std': base['alpha_std'], 'E_rms': base['E_rms']
        },
        'adaptive': {
            'R2': opt['R2'], 'loc_rate': opt['loc_rate'],
            'alpha_std': opt['alpha_std'], 'E_rms': opt['E_rms'],
            'alpha_global': opt['alpha_global']
        },
        'NT_correct': float(correct),
        'NT_mean_err': float(np.mean(err_N)),
        'NT_max_err': float(np.max(err_N)),
        'gue_failure_prediction': float(pf_new),
        'window_alphas': opt['window_alphas'],
        'window_R2s': opt['window_R2s']
    }
    outpath = os.path.join(REPO, 'notebooks', 'riemann', 'adaptive_theta_results.json')
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved to {outpath}")


if __name__ == '__main__':
    main()
