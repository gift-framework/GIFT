#!/usr/bin/env python3
"""
Quick test of theta(T) candidates on Odlyzko 2M zeros.

Replicates the exact methodology from GIFT_Correction_2M_Zeros.ipynb:
- Truth: delta = gamma_n - gamma0 (positional deviation from smooth zeros)
- Prediction: delta_pred = S_w_raw / tp (mollified sum divided by local density)
- Evaluate S_w at gamma0 (smooth positions), not gamma_n
- tp = 0.5 * log(T/(2*pi))
"""

import numpy as np
from scipy.special import loggamma, lambertw
from scipy import stats
import json, time

# ============================================================
# CONFIGURATION
# ============================================================
N_TEST = 200_000        # First 200k zeros
P_MAX = 500_000         # Primes up to 500k for coverage
K_MAX = 3               # Prime power cutoff
ZEROS_FILE = "/home/brieuc/gift-framework/GIFT-research/notebooks/outputs/riemann_zeros_2M_genuine.npy"

# Window boundaries for drift analysis
WINDOW_BOUNDS = [0, 50_000, 100_000, 150_000, 200_000]

# ============================================================
# CANDIDATE MODELS
# ============================================================
CANDIDATES = [
    # Constant (baseline) - should give alpha ~ 1.006
    ("Constant theta=0.9941",            0.9941, 0.0,     0.0),
    # Original GIFT
    ("GIFT 10/7 - (14/3)/logT",         10/7,   14/3,    0.0),
    # Spinor correction
    ("Spinor 10/7 - (13/3)/logT",       10/7,   13/3,    0.0),
    # Rank candidate
    ("Rank 8/7 - (11/7)/logT",          8/7,    11/7,    0.0),
    # h(E8)/dim(K7) correction
    ("GIFT 10/7 - (30/7)/logT",         10/7,   30/7,    0.0),
    # Connes candidate 17/4
    ("GIFT 10/7 - (17/4)/logT",         10/7,   17/4,    0.0),
    # 8/7 with 8/5
    ("8/7 - (8/5)/logT",                8/7,    8/5,     0.0),
    # User candidate: H*/70
    ("H*/70 = 99/70 - b_fit/logT",      99/70,  None,    0.0),
    # User candidate: (b2+b3)/70
    ("98/70 = 7/5 - b_fit/logT",        98/70,  None,    0.0),
    # 8/7 with free b
    ("8/7 - b_fit/logT",                8/7,    None,    0.0),
    # Free 2-param
    ("Free 2-param",                     None,   None,    0.0),
    # Free 3-param with log^2 term
    ("Free 3-param",                     None,   None,    None),
]

# ============================================================
# INFRASTRUCTURE (exact copy from notebook)
# ============================================================

def theta_rs(t):
    """Riemann-Siegel theta function."""
    t = np.asarray(t, dtype=np.float64)
    return np.imag(loggamma(0.25 + 0.5j * t)) - 0.5 * t * np.log(np.pi)


def theta_deriv(t):
    """Derivative of Riemann-Siegel theta: theta'(t) ~ 0.5*log(t/(2pi))."""
    return 0.5 * np.log(np.maximum(np.asarray(t, dtype=np.float64), 1.0) / (2 * np.pi))


def smooth_zeros(N):
    """Compute smooth zero positions gamma0_n where theta_RS(gamma0_n) = (n-1.5)*pi."""
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
    """Cosine-squared kernel."""
    return np.where(x < 1.0, np.cos(np.pi * x / 2)**2, 0.0)


def sieve(N):
    """Fast prime sieve up to N."""
    is_p = np.ones(N + 1, dtype=bool); is_p[:2] = False
    for i in range(2, int(N**0.5) + 1):
        if is_p[i]: is_p[i*i::i] = False
    return np.where(is_p)[0]


# ============================================================
# MOLLIFIED SUM (exact notebook formula, vectorized over zeros)
# ============================================================

def prime_sum_var(g0, tp_v, primes, k_max, theta_inf, theta_coeff, c_coeff=0.0):
    """Mollified prime sum: delta_pred = S_w_raw / tp.

    theta(T) = theta_inf + theta_coeff/log(T) + c_coeff/log^2(T)
    S_w = -sum_pm w(m*logp/logX) * sin(g0*m*logp) / (m * p^{m/2})
    return -S_w / tp  (positive convention matching delta)
    """
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


# ============================================================
# METRICS
# ============================================================

def compute_alpha_R2(delta, delta_pred):
    """OLS alpha and R^2 at alpha=1."""
    denom = np.dot(delta_pred, delta_pred)
    alpha = float(np.dot(delta, delta_pred) / denom) if denom > 0 else 0.0
    residual = delta - delta_pred
    R2 = float(1.0 - np.var(residual) / np.var(delta))
    return alpha, R2


def compute_window_metrics(delta, delta_pred, bounds):
    """Alpha per window for drift analysis."""
    alphas = []
    for i in range(len(bounds) - 1):
        lo, hi = bounds[i], min(bounds[i + 1], len(delta))
        d_w = delta[lo:hi]
        dp_w = delta_pred[lo:hi]
        denom = np.dot(dp_w, dp_w)
        alpha = float(np.dot(d_w, dp_w) / denom) if denom > 0 else 0.0
        alphas.append(alpha)
    return alphas


def compute_drift(alphas):
    """Linear regression of alpha vs window index."""
    if len(alphas) < 3:
        return 0.0, 1.0
    x = np.arange(len(alphas))
    slope, intercept, r, p, se = stats.linregress(x, alphas)
    return slope, p


def compute_localization(delta, delta_pred, gamma_n):
    """Fraction of zeros where prediction localizes actual zero."""
    half_gaps = np.diff(gamma_n) / 2.0
    residual = delta - delta_pred
    n = min(len(residual) - 1, len(half_gaps))
    localized = np.abs(residual[1:n+1]) < half_gaps[:n]
    return float(np.mean(localized))


# ============================================================
# OPTIMIZATION
# ============================================================

def optimize_b(g0, tp, delta, primes, theta_inf, k_max=3):
    """Find optimal b by minimizing |alpha - 1|."""
    from scipy.optimize import minimize_scalar

    def objective(b):
        dp = prime_sum_var(g0, tp, primes, k_max, theta_inf, -abs(b))
        alpha, _ = compute_alpha_R2(delta, dp)
        return (alpha - 1.0) ** 2

    result = minimize_scalar(objective, bounds=(0.0, 12.0), method='bounded',
                             options={'xatol': 0.05, 'maxiter': 15})
    return result.x


def optimize_2param(g0, tp, delta, primes, k_max=3):
    """Find optimal (theta_inf, b)."""
    from scipy.optimize import minimize

    def objective(params):
        a, b = params
        dp = prime_sum_var(g0, tp, primes, k_max, a, -abs(b))
        alpha, _ = compute_alpha_R2(delta, dp)
        alphas = compute_window_metrics(delta, dp, WINDOW_BOUNDS)
        slope, _ = compute_drift(alphas)
        return (alpha - 1.0) ** 2 + slope ** 2

    result = minimize(objective, x0=[1.4, 4.5], method='Nelder-Mead',
                      options={'maxiter': 40, 'xatol': 0.02, 'fatol': 1e-5})
    return result.x[0], abs(result.x[1])


def optimize_3param(g0, tp, delta, primes, k_max=3):
    """Find optimal (theta_inf, b, c)."""
    from scipy.optimize import minimize

    def objective(params):
        a, b, c = params
        dp = prime_sum_var(g0, tp, primes, k_max, a, -abs(b), c)
        alpha, _ = compute_alpha_R2(delta, dp)
        alphas = compute_window_metrics(delta, dp, WINDOW_BOUNDS)
        slope, _ = compute_drift(alphas)
        return (alpha - 1.0) ** 2 + slope ** 2

    result = minimize(objective, x0=[1.4, 4.5, -10.0], method='Nelder-Mead',
                      options={'maxiter': 60, 'xatol': 0.05, 'fatol': 1e-5})
    return result.x[0], abs(result.x[1]), result.x[2]


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("THETA CANDIDATE QUICK TEST (exact notebook methodology)")
    print(f"  N_zeros: {N_TEST}, P_max: {P_MAX}, K_max: {K_MAX}")
    print("=" * 70)

    # Load zeros
    all_zeros = np.load(ZEROS_FILE)
    gamma_n = all_zeros[:N_TEST]
    print(f"  Loaded {len(gamma_n)} zeros, T range: [{gamma_n[0]:.1f}, {gamma_n[-1]:.1f}]")

    # Compute smooth zeros and delta
    print("  Computing smooth zeros...", flush=True)
    t0 = time.time()
    gamma0 = smooth_zeros(N_TEST)
    delta = gamma_n - gamma0
    tp = theta_deriv(gamma0)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  delta: mean={delta.mean():.6f}, std={delta.std():.6f}")

    # Generate primes (fast sieve)
    t0 = time.time()
    primes = sieve(P_MAX)
    print(f"  Sieved {len(primes)} primes up to {P_MAX} [{time.time()-t0:.1f}s]")
    print()

    results = []
    t_global = time.time()

    for i, (name, theta_inf, b, c) in enumerate(CANDIDATES):
        t0 = time.time()
        print(f"[{i+1}/{len(CANDIDATES)}] {name}...")

        actual_a, actual_b, actual_c = theta_inf, b, c

        # Handle optimization
        if name == "Free 2-param":
            print("  Optimizing 2 params...", flush=True)
            actual_a, actual_b = optimize_2param(gamma0, tp, delta, primes, K_MAX)
            actual_c = 0.0
            print(f"  Found: a={actual_a:.6f}, b={actual_b:.6f}")
        elif name == "Free 3-param":
            print("  Optimizing 3 params...", flush=True)
            actual_a, actual_b, actual_c = optimize_3param(gamma0, tp, delta, primes, K_MAX)
            print(f"  Found: a={actual_a:.6f}, b={actual_b:.6f}, c={actual_c:.6f}")
        elif b is None and c == 0.0:
            print(f"  Optimizing b for theta_inf={theta_inf:.6f}...", flush=True)
            actual_b = optimize_b(gamma0, tp, delta, primes, theta_inf, K_MAX)
            actual_c = 0.0
            print(f"  Found: b={actual_b:.6f}")

        # theta_coeff is NEGATIVE of b (convention: theta = a + coeff/logT = a - b/logT)
        theta_coeff = -actual_b if actual_b else 0.0
        c_coeff = actual_c if actual_c else 0.0

        # Compute mollified sum
        delta_pred = prime_sum_var(gamma0, tp, primes, K_MAX,
                                    actual_a, theta_coeff, c_coeff)

        # Metrics
        alpha, R2 = compute_alpha_R2(delta, delta_pred)
        loc = compute_localization(delta, delta_pred, gamma_n)
        alphas = compute_window_metrics(delta, delta_pred, WINDOW_BOUNDS)
        drift_slope, drift_p = compute_drift(alphas)

        elapsed = time.time() - t0
        print(f"  alpha={alpha:.6f}, R2={R2:.6f}, loc={loc:.4f}")
        print(f"  drift={drift_slope:+.6f} (p={drift_p:.4f}), |alpha-1|={abs(alpha-1):.6f}")
        print(f"  window_alphas={[f'{a:.4f}' for a in alphas]}")
        print(f"  [{elapsed:.1f}s]")

        results.append({
            'name': name,
            'theta_inf': float(actual_a),
            'b': float(actual_b) if actual_b is not None else 0.0,
            'c': float(actual_c) if actual_c else 0.0,
            'alpha': float(alpha),
            'R2': float(R2),
            'localization': float(loc),
            'drift_slope': float(drift_slope),
            'drift_p': float(drift_p),
            'abs_alpha_minus_1': float(abs(alpha - 1)),
            'window_alphas': [float(a) for a in alphas],
            'elapsed_s': float(elapsed),
        })
        print()

    # ============================================================
    # RANKING
    # ============================================================
    total_time = time.time() - t_global
    print("=" * 70)
    print(f"RANKING (by |alpha-1|)   [total: {total_time:.0f}s]")
    print("=" * 70)
    print(f"{'Rank':<5} {'Name':<35} {'alpha':>10} {'|a-1|':>8} {'drift':>10} {'R2':>8} {'loc':>6}")
    print("-" * 82)

    ranked = sorted(results, key=lambda r: r['abs_alpha_minus_1'])
    for rank, r in enumerate(ranked, 1):
        print(f"{rank:<5} {r['name']:<35} {r['alpha']:>10.6f} {r['abs_alpha_minus_1']:>8.6f} "
              f"{r['drift_slope']:>+10.6f} {r['R2']:>8.4f} {r['localization']:>6.4f}")

    out_path = "/home/brieuc/gift-framework/GIFT-research/notebooks/outputs/theta_candidates_results.json"
    with open(out_path, 'w') as f:
        json.dump(ranked, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # ============================================================
    # RATIONAL SCAN
    # ============================================================
    print("\n" + "=" * 70)
    print("RATIONAL SCAN: best a=p/q, b=r/s with small denominators")
    print("  Using 10k zeros subset for speed")
    print("=" * 70)

    N_scan = 10_000
    g0_s = gamma0[:N_scan]
    tp_s = tp[:N_scan]
    delta_s = delta[:N_scan]
    gn_s = gamma_n[:N_scan]

    best_score = 999
    best_rationals = []

    a_candidates = [
        (8, 7), (9, 7), (10, 7), (11, 7),
        (7, 5), (6, 5),
        (99, 70), (98, 70),
        (7, 6), (8, 6),
        (11, 10), (12, 10), (14, 10), (15, 10),
        (13, 9), (14, 9),
        (9, 8), (11, 8),
    ]

    from math import gcd
    b_values = []
    b_seen = set()
    for bd in range(1, 11):
        for bn in range(0, bd * 8 + 1):
            b_val = bn / bd
            if b_val > 8.0:
                continue
            g = gcd(bn, bd)
            key = (bn // g, bd // g)
            if key not in b_seen:
                b_seen.add(key)
                b_values.append((bn, bd))

    total_combos = len(a_candidates) * len(b_values)
    print(f"  Testing {len(a_candidates)} a-values x {len(b_values)} b-values = {total_combos} combos")

    t_scan = time.time()
    tested = 0
    for a_num, a_den in a_candidates:
        a = a_num / a_den
        for b_num, b_den in b_values:
            b = b_num / b_den
            tested += 1
            if tested % 200 == 0:
                elapsed = time.time() - t_scan
                eta = elapsed / tested * (total_combos - tested) if tested > 0 else 0
                print(f"  {tested}/{total_combos} [{elapsed:.0f}s, ETA {eta:.0f}s]")

            dp = prime_sum_var(g0_s, tp_s, primes, K_MAX, a, -b)
            alpha_val, R2_val = compute_alpha_R2(delta_s, dp)
            alphas_w = compute_window_metrics(delta_s, dp, [0, 2500, 5000, 7500, 10000])
            drift_s_val, _ = compute_drift(alphas_w)

            score = abs(alpha_val - 1) + abs(drift_s_val) * 50
            if score < best_score:
                best_score = score
                entry = {
                    'a_num': int(a_num), 'a_den': int(a_den),
                    'b_num': int(b_num), 'b_den': int(b_den),
                    'a': float(a), 'b': float(b),
                    'alpha': float(alpha_val), 'R2': float(R2_val),
                    'drift': float(drift_s_val), 'score': float(score),
                }
                best_rationals.append(entry)
                print(f"  NEW BEST: theta={a_num}/{a_den} - ({b_num}/{b_den})/logT "
                      f"-> alpha={alpha_val:.6f}, drift={drift_s_val:+.6f}, score={score:.6f}")

    scan_time = time.time() - t_scan
    print(f"\n  Scan complete in {scan_time:.0f}s ({total_combos} combos)")
    if best_rationals:
        w = best_rationals[-1]
        print(f"\n  WINNER: theta(T) = {w['a_num']}/{w['a_den']} - ({w['b_num']}/{w['b_den']})/log(T)")
        print(f"  = {w['a']:.6f} - {w['b']:.6f}/log(T)")
        print(f"  alpha={w['alpha']:.6f}, R2={w['R2']:.6f}, drift={w['drift']:+.8f}")

    top10 = sorted(best_rationals, key=lambda x: x['score'])[:10] if best_rationals else []
    scan_out = "/home/brieuc/gift-framework/GIFT-research/notebooks/outputs/rational_scan_results.json"
    with open(scan_out, 'w') as f:
        json.dump(top10, f, indent=2)
    print(f"  Top 10 saved to {scan_out}")


if __name__ == '__main__':
    main()
