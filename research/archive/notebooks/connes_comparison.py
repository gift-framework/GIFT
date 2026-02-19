#!/usr/bin/env python3
"""
Compare GIFT mollifier with Connes' 6-prime approach.
Connes (arXiv:2602.04022): primes {2,3,5,7,11,13} recover 50 zeros
with precision 10^{-55} to 10^{-3}.

Test: what does our mollified Dirichlet polynomial give with the same 6 primes?
Also compare constant theta vs GIFT correction theta(T) = 10/7 - (14/3)/log(T).
"""
import numpy as np
import json
import time
from scipy import stats
from scipy.special import loggamma, lambertw
import sys

OUTDIR = '/home/brieuc/gift-framework/GIFT-research/notebooks/outputs'
print("=" * 70, flush=True)
print("CONNES COMPARISON: GIFT mollifier with 6 primes", flush=True)
print("=" * 70, flush=True)
t0 = time.time()

# ── Load data ──
gamma_n = np.load(f'{OUTDIR}/riemann_zeros_2M_genuine.npy')
N_ZEROS = len(gamma_n)
print(f"Loaded {N_ZEROS:,} zeros, T_max={gamma_n[-1]:.0f}", flush=True)

# ── Smooth zeros (Newton) ──
print("Computing smooth zeros...", flush=True)
ns = np.arange(1, N_ZEROS + 1, dtype=np.float64)
targets = (ns - 1.5) * np.pi
w_lw = np.real(lambertw(ns / np.e))
gamma0 = np.maximum(2 * np.pi * ns / w_lw, 2.0)
for _ in range(40):
    th = np.imag(loggamma(0.25 + 0.5j * gamma0)) - 0.5 * gamma0 * np.log(np.pi)
    tp = 0.5 * np.log(np.maximum(gamma0, 1.0) / (2 * np.pi))
    dt = (th - targets) / np.maximum(np.abs(tp), 1e-15)
    gamma0 -= dt
    if np.max(np.abs(dt)) < 1e-10:
        break

delta = gamma_n - gamma0
tp = 0.5 * np.log(np.maximum(gamma0, 1.0) / (2 * np.pi))
print(f"  Done [{time.time()-t0:.1f}s]", flush=True)

# ── Prime sets to test ──
CONNES_PRIMES = np.array([2, 3, 5, 7, 11, 13])
K_MAX = 3

def sieve(N):
    is_p = np.ones(N + 1, dtype=bool); is_p[:2] = False
    for i in range(2, int(N**0.5) + 1):
        if is_p[i]: is_p[i*i::i] = False
    return np.where(is_p)[0]

PRIME_SETS = {
    'Connes_6': CONNES_PRIMES,
    'primes_30': sieve(30),
    'primes_100': sieve(100),
    'primes_1000': sieve(1000),
    'primes_5000': sieve(5000),
    'primes_10000': sieve(10000),
}

# ── Mollifier function ──
def w_cosine(x):
    return np.where(x < 1.0, np.cos(np.pi * x / 2)**2, 0.0)

def prime_sum(g0, tp_v, primes, k_max, theta_inf, theta_coeff):
    """Mollified prime sum. theta_coeff=0 for constant theta."""
    S = np.zeros_like(g0)
    log_g0 = np.log(np.maximum(g0, 2.0))
    if theta_coeff == 0.0:
        log_X = theta_inf * log_g0
    else:
        theta_per = np.clip(theta_inf + theta_coeff / log_g0, 0.5, 2.0)
        log_X = theta_per * log_g0

    for p in primes:
        logp = np.log(float(p))
        if logp / np.max(log_X) > 3.0:
            break
        for m in range(1, k_max + 1):
            x = m * logp / log_X
            weight = w_cosine(x)
            if np.max(weight) < 1e-15:
                continue
            S -= weight * np.sin(g0 * m * logp) / (m * p ** (m / 2.0))
    return -S / tp_v

# ── Models to test ──
MODELS = {
    'constant_0.9941': (0.9941, 0.0),
    'GIFT_10/7': (10/7, -14/3),
    'theta_1.0': (1.0, 0.0),
    'theta_1/log2': (1/np.log(2), 0.0),
}

# ── Run all combinations ──
print("\n" + "=" * 70, flush=True)
print("TESTING ALL (prime_set, model) COMBINATIONS", flush=True)
print("=" * 70, flush=True)

# First test on 50 zeros (Connes comparison) and 100k zeros
TEST_RANGES = {
    'first_50': (0, 50),
    'first_1000': (0, 1000),
    'first_100k': (0, 100_000),
    'full_2M': (0, N_ZEROS),
}

results = {}

for pname, primes in PRIME_SETS.items():
    for mname, (theta_inf, theta_coeff) in MODELS.items():
        key = f"{pname}__{mname}"
        print(f"\n--- {pname} ({len(primes)} primes) x {mname} ---", flush=True)

        res = {}
        for rname, (a, b) in TEST_RANGES.items():
            t1 = time.time()
            d = delta[a:b]
            dp = prime_sum(gamma0[a:b], tp[a:b], primes, K_MAX, theta_inf, theta_coeff)

            # Metrics
            dot_pp = np.dot(dp, dp)
            alpha = float(np.dot(d, dp) / dot_pp) if dot_pp > 0 else 0.0
            r = d - dp
            R2 = float(1.0 - np.var(r) / np.var(d)) if np.var(d) > 0 else 0.0
            R2_scaled = float(1.0 - np.var(d - alpha * dp) / np.var(d)) if np.var(d) > 0 else 0.0

            el = time.time() - t1
            res[rname] = {
                'alpha': alpha, 'R2': R2, 'R2_scaled': R2_scaled,
                'E_rms': float(np.sqrt(np.mean(r**2))),
                'time': el,
            }

            if rname in ('first_50', 'first_100k'):
                print(f"  {rname:>12}: alpha={alpha:+.4f} R2={R2:.4f} "
                      f"R2_sc={R2_scaled:.4f} [{el:.2f}s]", flush=True)

        results[key] = res

# ── Summary table ──
print("\n\n" + "=" * 70, flush=True)
print("SUMMARY: R2 (alpha=1) on first 100k zeros", flush=True)
print("=" * 70, flush=True)
print(f"{'':>15}", end="", flush=True)
for mname in MODELS:
    print(f" | {mname:>16}", end="", flush=True)
print(flush=True)
print("-" * (16 + 19 * len(MODELS)), flush=True)

for pname in PRIME_SETS:
    print(f"{pname:>15}", end="", flush=True)
    for mname in MODELS:
        key = f"{pname}__{mname}"
        r2 = results[key]['first_100k']['R2']
        print(f" | {r2:>16.4f}", end="", flush=True)
    print(flush=True)

print("\n" + "=" * 70, flush=True)
print("SUMMARY: alpha on first 100k zeros", flush=True)
print("=" * 70, flush=True)
print(f"{'':>15}", end="", flush=True)
for mname in MODELS:
    print(f" | {mname:>16}", end="", flush=True)
print(flush=True)
print("-" * (16 + 19 * len(MODELS)), flush=True)

for pname in PRIME_SETS:
    print(f"{pname:>15}", end="", flush=True)
    for mname in MODELS:
        key = f"{pname}__{mname}"
        a = results[key]['first_100k']['alpha']
        print(f" | {a:>+16.4f}", end="", flush=True)
    print(flush=True)

# ── Connes 6-prime detail: per-zero accuracy ──
print("\n\n" + "=" * 70, flush=True)
print("CONNES 6-PRIME DETAIL: per-zero prediction quality", flush=True)
print("=" * 70, flush=True)

for mname, (theta_inf, theta_coeff) in MODELS.items():
    dp = prime_sum(gamma0[:50], tp[:50], CONNES_PRIMES, K_MAX, theta_inf, theta_coeff)
    alpha = float(np.dot(delta[:50], dp) / np.dot(dp, dp))

    print(f"\n  Model: {mname} (alpha_OLS={alpha:.4f})", flush=True)
    print(f"  {'n':>5} {'gamma_n':>14} {'delta':>10} {'pred':>10} {'error':>10} {'|err/delta|':>12}", flush=True)
    for i in range(min(20, 50)):
        d = delta[i]
        p = dp[i]
        err = d - p
        rel = abs(err / d) if abs(d) > 1e-15 else 0
        print(f"  {i+1:>5} {gamma_n[i]:>14.6f} {d:>+10.6f} {p:>+10.6f} "
              f"{err:>+10.6f} {rel:>12.4f}", flush=True)

# ── Window analysis for GIFT model with Connes primes ──
print("\n\n" + "=" * 70, flush=True)
print("WINDOW ANALYSIS: GIFT 10/7 model with different prime sets", flush=True)
print("=" * 70, flush=True)

WINDOWS = [
    (0, 100_000), (100_000, 200_000), (200_000, 500_000),
    (500_000, 1_000_000), (1_000_000, 1_500_000), (1_500_000, N_ZEROS),
]

for pname in ['Connes_6', 'primes_1000', 'primes_5000']:
    primes = PRIME_SETS[pname]
    print(f"\n  --- {pname} ({len(primes)} primes), GIFT correction ---", flush=True)
    print(f"  {'Window':>20} | {'alpha':>8} | {'R2':>8} | {'theta_mid':>10}", flush=True)
    print(f"  " + "-" * 55, flush=True)

    for (a, b) in WINDOWS:
        d_w = delta[a:b]
        dp_w = prime_sum(gamma0[a:b], tp[a:b], primes, K_MAX, 10/7, -14/3)
        dot_pp = np.dot(dp_w, dp_w)
        alpha_w = float(np.dot(d_w, dp_w) / dot_pp) if dot_pp > 0 else 0
        R2_w = float(1.0 - np.var(d_w - dp_w) / np.var(d_w))
        T_mid = (gamma_n[a] + gamma_n[min(b-1, N_ZEROS-1)]) / 2
        log_T = np.log(T_mid)
        th_mid = 10/7 - 14/(3*log_T)
        label = f"[{a//1000}k, {b//1000}k)"
        print(f"  {label:>20} | {alpha_w:>+8.4f} | {R2_w:>8.4f} | {th_mid:>10.4f}", flush=True)

# ── Convergence: R2 vs number of primes ──
print("\n\n" + "=" * 70, flush=True)
print("CONVERGENCE: R2 vs prime cutoff (first 100k zeros)", flush=True)
print("=" * 70, flush=True)

cutoffs = [13, 30, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
d_100k = delta[:100_000]
g0_100k = gamma0[:100_000]
tp_100k = tp[:100_000]

print(f"  {'P_max':>8} | {'n_primes':>8} | {'R2_const':>10} | {'R2_GIFT':>10} | "
      f"{'alpha_c':>8} | {'alpha_G':>8}", flush=True)
print(f"  " + "-" * 70, flush=True)

convergence = []
for pmax in cutoffs:
    p = sieve(pmax)
    t1 = time.time()

    dp_c = prime_sum(g0_100k, tp_100k, p, K_MAX, 0.9941, 0.0)
    dp_g = prime_sum(g0_100k, tp_100k, p, K_MAX, 10/7, -14/3)

    alpha_c = float(np.dot(d_100k, dp_c) / np.dot(dp_c, dp_c))
    alpha_g = float(np.dot(d_100k, dp_g) / np.dot(dp_g, dp_g))
    R2_c = float(1.0 - np.var(d_100k - dp_c) / np.var(d_100k))
    R2_g = float(1.0 - np.var(d_100k - dp_g) / np.var(d_100k))

    el = time.time() - t1
    print(f"  {pmax:>8} | {len(p):>8} | {R2_c:>10.6f} | {R2_g:>10.6f} | "
          f"{alpha_c:>+8.4f} | {alpha_g:>+8.4f}  [{el:.1f}s]", flush=True)
    convergence.append({
        'P_max': int(pmax), 'n_primes': int(len(p)),
        'R2_const': R2_c, 'R2_GIFT': R2_g,
        'alpha_const': alpha_c, 'alpha_GIFT': alpha_g,
    })

# ── Figure ──
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (0) R2 vs P_max
    ax = axes[0]
    pms = [c['P_max'] for c in convergence]
    ax.semilogx(pms, [c['R2_const'] for c in convergence], 'rs-', ms=6, label='Constant 0.9941')
    ax.semilogx(pms, [c['R2_GIFT'] for c in convergence], 'bo-', ms=6, label='GIFT 10/7')
    ax.axvline(13, color='purple', ls=':', alpha=0.5, label='Connes (p<=13)')
    ax.set_xlabel('P_max')
    ax.set_ylabel('R2')
    ax.set_title('R2 vs Prime Cutoff (100k zeros)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (1) Alpha vs P_max
    ax = axes[1]
    ax.semilogx(pms, [c['alpha_const'] for c in convergence], 'rs-', ms=6, label='Constant 0.9941')
    ax.semilogx(pms, [c['alpha_GIFT'] for c in convergence], 'bo-', ms=6, label='GIFT 10/7')
    ax.axhline(1.0, color='gray', ls='--', alpha=0.5)
    ax.axvline(13, color='purple', ls=':', alpha=0.5, label='Connes (p<=13)')
    ax.set_xlabel('P_max')
    ax.set_ylabel('alpha (OLS)')
    ax.set_title('Alpha vs Prime Cutoff')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (2) R2 improvement (GIFT - const) vs P_max
    ax = axes[2]
    delta_R2 = [c['R2_GIFT'] - c['R2_const'] for c in convergence]
    colors = ['green' if d > 0 else 'red' for d in delta_R2]
    ax.bar(range(len(pms)), delta_R2, color=colors, alpha=0.7)
    ax.set_xticks(range(len(pms)))
    ax.set_xticklabels([str(p) for p in pms], rotation=45, fontsize=8)
    ax.axhline(0, color='gray', ls='-', alpha=0.3)
    ax.set_xlabel('P_max')
    ax.set_ylabel('R2(GIFT) - R2(constant)')
    ax.set_title('GIFT Advantage')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    figpath = f'{OUTDIR}/fig8_connes_comparison.png'
    plt.savefig(figpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved {figpath}", flush=True)
except Exception as e:
    print(f"\nPlot error: {e}", flush=True)

# ── Save results ──
total = time.time() - t0
print(f"\nTotal: {total:.0f}s", flush=True)

out = {
    'convergence': convergence,
    'summary_100k': {
        key: results[key]['first_100k']
        for key in results
    },
    'total_time': total,
}
jpath = f'{OUTDIR}/connes_comparison_results.json'
with open(jpath, 'w') as f:
    json.dump(out, f, indent=2)
print(f"Saved {jpath}", flush=True)
