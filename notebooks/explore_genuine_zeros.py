#!/usr/bin/env python3
"""
FRESH START: Explore GIFT parameter space on GENUINE Odlyzko zeros
===================================================================

All previous explorations may have used smooth (Gram) zeros.
This script re-explores EVERYTHING from scratch with genuine data.

Phase 1: Landscape scan (P_MAX=50k, fast ~30s/eval)
  - Constant theta scan
  - Named GIFT formulas
  - d-shift scan for shifted-log family
  - 2D (a,b) scan around promising regions

Phase 2: Fine-tune (P_MAX=100k, ~90s/eval)
  - Zoom into best candidates

Phase 3: Validate (P_MAX=500k, ~5min/eval)
  - Top 3 candidates, full T7/T8 analysis
"""
import numpy as np
from scipy.special import loggamma
from scipy import stats
import time, json, os, math, sys

phi = (1 + math.sqrt(5)) / 2
MERTENS = 0.2614972128476428
EULER_GAMMA = 0.5772156649015329

# GIFT constants
DIM_G2 = 14; DIM_K7 = 7; H_STAR = 99; B2 = 21; B3 = 77
RANK_E8 = 8; P2 = 2; N_GEN = 3; DIM_E8 = 248; WEYL = 5
D_BULK = 11; DIM_J3O = 27; DIM_E8xE8 = 496
TAU_GIFT = 3472 / 891

os.environ.setdefault('CUDA_PATH', '/usr')
try:
    import cupy as cp
    _ = cp.array([1.0])
    GPU = True
    mem = cp.cuda.Device(0).mem_info[0] // 1024**2
    print(f"[GPU] free={mem} MB")
    PRIME_BATCH = 500
    ZERO_CHUNK = 80_000
except Exception as e:
    GPU = False
    PRIME_BATCH = 200
    ZERO_CHUNK = 200_000
    print(f"[CPU] {e}")

# ================================================================
# INFRASTRUCTURE
# ================================================================
def theta_rs(t):
    t = np.asarray(t, dtype=np.float64)
    return np.imag(loggamma(0.25 + 0.5j * t)) - 0.5 * t * np.log(np.pi)

def theta_deriv(t):
    return 0.5 * np.log(np.maximum(np.asarray(t, dtype=np.float64), 1.0) / (2 * np.pi))

def sieve(N):
    is_p = np.ones(N + 1, dtype=bool); is_p[:2] = False
    for i in range(2, int(N**0.5) + 1):
        if is_p[i]: is_p[i*i::i] = False
    return np.where(is_p)[0]

def prime_sum(g0, tp_v, primes, k_max, a, b, d_shift):
    """Shifted-log: theta(T) = a - b/(logT + d_shift)
    If b=0: constant theta = a.
    """
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

        if abs(b) < 1e-15:
            theta_per = xp.full(hi - lo, xp.float64(a))
        else:
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

def eval_formula(gamma0, tp_v, delta_pred, primes, k_max, a, b, d_shift, n_win=50):
    """Evaluate a formula. Returns dict with alpha, T7, T8, window data."""
    S_w = prime_sum(gamma0, tp_v, primes, k_max, a, b, d_shift)
    delta = delta_pred + S_w
    N = len(gamma0)

    # Global alpha
    denom = np.dot(delta_pred, delta_pred)
    alpha_g = float(np.dot(delta, delta_pred) / denom)

    # Window alphas
    bounds = [int(i * N / n_win) for i in range(n_win + 1)]
    alphas = np.empty(n_win)
    T_mids = np.empty(n_win)
    for i in range(n_win):
        lo, hi = bounds[i], bounds[i+1]
        dp = delta_pred[lo:hi]
        d = delta[lo:hi]
        den = np.dot(dp, dp)
        alphas[i] = float(np.dot(d, dp) / den) if den > 0 else 0.0
        T_mids[i] = float(gamma0[(lo + hi) // 2])

    # T7: Bootstrap CI contains 1
    rng = np.random.default_rng(42)
    boot_alpha = np.empty(500)
    for b_idx in range(500):
        idx = rng.integers(0, n_win, size=n_win)
        # Global alpha from resampled windows
        boot_alpha[b_idx] = np.mean(alphas[idx])
    ci = (float(np.percentile(boot_alpha, 2.5)), float(np.percentile(boot_alpha, 97.5)))
    t7 = bool(ci[0] <= 1.0 <= ci[1])

    # T8: No drift (12 windows for comparison)
    n_drift = 12
    bounds12 = [int(i * N / n_drift) for i in range(n_drift + 1)]
    alphas12 = np.empty(n_drift)
    for i in range(n_drift):
        lo, hi = bounds12[i], bounds12[i+1]
        dp = delta_pred[lo:hi]
        d = delta[lo:hi]
        den = np.dot(dp, dp)
        alphas12[i] = float(np.dot(d, dp) / den) if den > 0 else 0.0
    _, _, _, p_drift, _ = stats.linregress(np.arange(n_drift), alphas12)
    t8 = bool(p_drift > 0.05)

    return {
        'alpha': alpha_g,
        'abs_alpha_m1': abs(alpha_g - 1.0),
        't7': t7, 't7_ci': ci,
        't8': t8, 'drift_p': float(p_drift),
        'alpha_range': [float(np.min(alphas)), float(np.max(alphas))],
        'alpha_std': float(np.std(alphas)),
    }

def banner(title, char='='):
    w = 72
    print(f"\n{char * w}\n  {title}\n{char * w}", flush=True)

# ================================================================
# LOAD GENUINE ZEROS
# ================================================================
banner("Loading genuine Odlyzko zeros")

npy_file = 'outputs/odlyzko_zeros_2M.npy'
if not os.path.exists(npy_file):
    print("ERROR: Run download first")
    sys.exit(1)

gamma0 = np.load(npy_file)
N_zeros = len(gamma0)
print(f"  {N_zeros:,} genuine zeros, range [{gamma0[0]:.2f}, {gamma0[-1]:.2f}]")

tp_v = theta_deriv(gamma0)
delta_pred = np.sign(np.sin(np.pi * (theta_rs(gamma0) / np.pi - 0.5)))

K_MAX = 3

# ================================================================
# PHASE 1: LANDSCAPE SCAN (P_MAX = 50k)
# ================================================================
banner("PHASE 1: Landscape scan (P_MAX = 50k)")

P_MAX_1 = 50_000
primes_1 = sieve(P_MAX_1)
print(f"  {len(primes_1):,} primes up to {P_MAX_1:,}")
print(f"  Estimated: ~25s per candidate on RTX 2050\n")

candidates = []

# --- Group A: Constant theta ---
for theta_const in [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40]:
    candidates.append({
        'name': f'const_{theta_const:.2f}',
        'group': 'constant',
        'a': theta_const, 'b': 0.0, 'd': 0.0,
        'formula': f'θ = {theta_const:.2f}',
    })

# --- Group B: Linear theta = a - b/logT (shifted-log with d=0) ---
linear_formulas = [
    ('GIFT_v1',       10/7,  14/3,    '10/7 - (14/3)/logT'),
    ('empirical_fit', 1.146, 2.378,   '1.146 - 2.378/logT'),
    ('7_6__e_phi',    7/6,   math.e/phi, '7/6 - (e/φ)/logT'),
    ('7_6__phi',      7/6,   phi,     '7/6 - φ/logT'),
    ('6_5__12_5',     6/5,   12/5,    '6/5 - (12/5)/logT'),
    ('11_9__5_2',     11/9,  5/2,     '11/9 - (5/2)/logT'),
]
for name, a, b, formula in linear_formulas:
    candidates.append({
        'name': name, 'group': 'linear',
        'a': a, 'b': b, 'd': 0.0,
        'formula': formula,
    })

# --- Group C: Shifted-log theta = a - b/(logT + d) ---
shifted_log_formulas = [
    ('SL_7_6_phi_m2',       7/6, phi,       -2.0,    '7/6 - φ/(logT - 2)'),
    ('SL_7_6_phi_m15_8',    7/6, phi,       -15/8,   '7/6 - φ/(logT - 15/8)'),
    ('SL_7_6_phi_m1',       7/6, phi,       -1.0,    '7/6 - φ/(logT - 1)'),
    ('SL_7_6_phi_m1_5',     7/6, phi,       -1.5,    '7/6 - φ/(logT - 1.5)'),
    ('SL_7_6_phi_m2_5',     7/6, phi,       -2.5,    '7/6 - φ/(logT - 2.5)'),
    ('SL_7_6_phi_m3',       7/6, phi,       -3.0,    '7/6 - φ/(logT - 3)'),
    ('SL_7_6_ephi_m2',      7/6, math.e/phi, -2.0,   '7/6 - (e/φ)/(logT - 2)'),
    ('SL_7_6_ephi_m15_8',   7/6, math.e/phi, -15/8,  '7/6 - (e/φ)/(logT - 15/8)'),
    ('SL_10_7_14_3_m15_8',  10/7, 14/3,     -15/8,   '10/7 - (14/3)/(logT - 15/8)'),
    ('SL_10_7_phi_m2',      10/7, phi,       -2.0,    '10/7 - φ/(logT - 2)'),
]
for name, a, b, d, formula in shifted_log_formulas:
    candidates.append({
        'name': name, 'group': 'shifted-log',
        'a': a, 'b': b, 'd': d,
        'formula': formula,
    })

# --- Group D: d-scan for 7/6 - phi/(logT + d) ---
for d_val in np.arange(-3.0, 0.1, 0.25):
    d_val = round(d_val, 2)
    candidates.append({
        'name': f'd_scan_{d_val:.2f}',
        'group': 'd-scan',
        'a': 7/6, 'b': phi, 'd': d_val,
        'formula': f'7/6 - φ/(logT + {d_val})',
    })

# --- Group E: a-scan for a - phi/(logT - 15/8) ---
for a_val in [1.0, 1.05, 1.10, 7/6, 1.20, 1.25, 10/7, 1.45, 1.50]:
    candidates.append({
        'name': f'a_scan_{a_val:.4f}',
        'group': 'a-scan',
        'a': a_val, 'b': phi, 'd': -15/8,
        'formula': f'{a_val:.4f} - φ/(logT - 15/8)',
    })

print(f"  Total candidates: {len(candidates)}")
print(f"  Estimated time: {len(candidates) * 25 / 60:.0f} min\n")

# Run Phase 1
results_1 = []
t0_phase1 = time.time()

for idx, cand in enumerate(candidates):
    t0 = time.time()
    res = eval_formula(gamma0, tp_v, delta_pred, primes_1, K_MAX,
                       cand['a'], cand['b'], cand['d'])
    elapsed = time.time() - t0

    res.update({
        'name': cand['name'], 'group': cand['group'], 'formula': cand['formula'],
        'a': cand['a'], 'b': cand['b'], 'd': cand['d'],
        'elapsed_s': elapsed,
    })
    results_1.append(res)

    # Status indicator
    t7_s = 'T7' if res['t7'] else '  '
    t8_s = 'T8' if res['t8'] else '  '
    print(f"  {idx+1:3d}/{len(candidates)} [{t7_s}|{t8_s}] "
          f"|α-1|={res['abs_alpha_m1']:.6f}  drift_p={res['drift_p']:.3f}  "
          f"α={res['alpha']:.7f}  {cand['formula'][:40]:40s} ({elapsed:.0f}s)",
          flush=True)

phase1_elapsed = time.time() - t0_phase1
print(f"\n  Phase 1 done in {phase1_elapsed:.0f}s ({phase1_elapsed/60:.1f} min)")

# ================================================================
# PHASE 1 ANALYSIS
# ================================================================
banner("PHASE 1 RESULTS")

# Sort by |alpha - 1|
ranked = sorted(results_1, key=lambda r: r['abs_alpha_m1'])

print(f"\n  TOP 20 by |α - 1| (P_MAX={P_MAX_1:,}):")
print(f"  {'Rank':>4s}  {'|α-1|':>10s}  {'α':>10s}  {'T7':>3s} {'T8':>3s}  {'drift_p':>8s}  Formula")
print(f"  {'─'*4}  {'─'*10}  {'─'*10}  {'─'*3} {'─'*3}  {'─'*8}  {'─'*40}")
for i, r in enumerate(ranked[:20]):
    print(f"  {i+1:4d}  {r['abs_alpha_m1']:10.7f}  {r['alpha']:10.7f}  "
          f"{'YES' if r['t7'] else ' no':>3s} {'YES' if r['t8'] else ' no':>3s}  "
          f"{r['drift_p']:8.4f}  {r['formula'][:40]}")

# T7+T8 passing
both = [r for r in results_1 if r['t7'] and r['t8']]
print(f"\n  Candidates passing BOTH T7 and T8: {len(both)}")
for r in both:
    print(f"    {r['formula']:40s}  α={r['alpha']:.7f}  drift_p={r['drift_p']:.4f}")

# Best per group
print(f"\n  BEST per group:")
for group in ['constant', 'linear', 'shifted-log', 'd-scan', 'a-scan']:
    grp = [r for r in results_1 if r['group'] == group]
    if grp:
        best = min(grp, key=lambda r: r['abs_alpha_m1'])
        print(f"    {group:15s}: {best['formula']:40s}  |α-1|={best['abs_alpha_m1']:.7f}  "
              f"T7={'Y' if best['t7'] else 'N'} T8={'Y' if best['t8'] else 'N'}")

# d-scan analysis (find optimal d for 7/6 - phi/(logT+d))
d_scan_results = [r for r in results_1 if r['group'] == 'd-scan']
if d_scan_results:
    print(f"\n  D-SCAN: 7/6 - φ/(logT + d)")
    print(f"  {'d':>8s}  {'|α-1|':>10s}  {'α':>10s}  {'T7':>3s} {'T8':>3s}  {'drift_p':>8s}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*10}  {'─'*3} {'─'*3}  {'─'*8}")
    for r in sorted(d_scan_results, key=lambda r: r['d']):
        print(f"  {r['d']:8.2f}  {r['abs_alpha_m1']:10.7f}  {r['alpha']:10.7f}  "
              f"{'YES' if r['t7'] else ' no':>3s} {'YES' if r['t8'] else ' no':>3s}  "
              f"{r['drift_p']:8.4f}")

# Save Phase 1
os.makedirs('outputs', exist_ok=True)
with open('outputs/explore_genuine_phase1.json', 'w') as f:
    json.dump({
        'p_max': P_MAX_1,
        'n_zeros': N_zeros,
        'k_max': K_MAX,
        'n_candidates': len(candidates),
        'elapsed_s': phase1_elapsed,
        'results': results_1,
        'top_20': [r['name'] for r in ranked[:20]],
    }, f, indent=2, default=float)
print(f"\n  Phase 1 saved to outputs/explore_genuine_phase1.json")

# ================================================================
# PHASE 2: VALIDATE TOP CANDIDATES (P_MAX = 500k)
# ================================================================
banner("PHASE 2: Validate top candidates (P_MAX = 500k)")

P_MAX_2 = 500_000
primes_2 = sieve(P_MAX_2)
print(f"  {len(primes_2):,} primes up to {P_MAX_2:,}")

# Pick top candidates: best per group + T7+T8 passers + overall top 5
top_names = set()
# Top 5 overall
for r in ranked[:5]:
    top_names.add(r['name'])
# Best per group
for group in ['constant', 'linear', 'shifted-log', 'd-scan', 'a-scan']:
    grp = [r for r in results_1 if r['group'] == group]
    if grp:
        top_names.add(min(grp, key=lambda r: r['abs_alpha_m1'])['name'])
# T7+T8 passers
for r in both:
    top_names.add(r['name'])
# Key named formulas
for name in ['GIFT_v1', 'SL_7_6_phi_m2', 'SL_7_6_phi_m15_8', 'SL_7_6_ephi_m15_8']:
    if any(r['name'] == name for r in results_1):
        top_names.add(name)

top_candidates = [c for c in candidates if c['name'] in top_names]
print(f"  Validating {len(top_candidates)} candidates")
print(f"  Estimated time: {len(top_candidates) * 5:.0f} min\n")

results_2 = []
t0_phase2 = time.time()

for idx, cand in enumerate(top_candidates):
    t0 = time.time()
    res = eval_formula(gamma0, tp_v, delta_pred, primes_2, K_MAX,
                       cand['a'], cand['b'], cand['d'])
    elapsed = time.time() - t0

    res.update({
        'name': cand['name'], 'group': cand['group'], 'formula': cand['formula'],
        'a': cand['a'], 'b': cand['b'], 'd': cand['d'],
        'elapsed_s': elapsed,
    })
    results_2.append(res)

    t7_s = 'T7' if res['t7'] else '  '
    t8_s = 'T8' if res['t8'] else '  '
    print(f"  {idx+1:3d}/{len(top_candidates)} [{t7_s}|{t8_s}] "
          f"|α-1|={res['abs_alpha_m1']:.6f}  drift_p={res['drift_p']:.3f}  "
          f"α={res['alpha']:.7f}  {cand['formula'][:40]:40s} ({elapsed:.0f}s)",
          flush=True)

phase2_elapsed = time.time() - t0_phase2

# Phase 2 analysis
ranked_2 = sorted(results_2, key=lambda r: r['abs_alpha_m1'])
both_2 = [r for r in results_2 if r['t7'] and r['t8']]

banner("PHASE 2 RESULTS (P_MAX=500k)")
print(f"\n  ALL candidates ranked by |α-1|:")
print(f"  {'Rank':>4s}  {'|α-1|':>10s}  {'α':>10s}  {'T7':>3s} {'T8':>3s}  {'drift_p':>8s}  Formula")
print(f"  {'─'*4}  {'─'*10}  {'─'*10}  {'─'*3} {'─'*3}  {'─'*8}  {'─'*40}")
for i, r in enumerate(ranked_2):
    print(f"  {i+1:4d}  {r['abs_alpha_m1']:10.7f}  {r['alpha']:10.7f}  "
          f"{'YES' if r['t7'] else ' no':>3s} {'YES' if r['t8'] else ' no':>3s}  "
          f"{r['drift_p']:8.4f}  {r['formula'][:40]}")

print(f"\n  T7+T8 passers at P_MAX=500k: {len(both_2)}")
for r in both_2:
    print(f"    {r['formula']:40s}  α={r['alpha']:.7f}  drift_p={r['drift_p']:.4f}")

# Comparison: P_MAX=50k vs 500k
print(f"\n  Shift 50k → 500k:")
for r2 in ranked_2[:10]:
    r1 = next((r for r in results_1 if r['name'] == r2['name']), None)
    if r1:
        shift = r2['alpha'] - r1['alpha']
        print(f"    {r2['formula'][:35]:35s}  α(50k)={r1['alpha']:.6f}  α(500k)={r2['alpha']:.6f}  Δ={shift:+.6f}")

# Save Phase 2
with open('outputs/explore_genuine_phase2.json', 'w') as f:
    json.dump({
        'p_max': P_MAX_2,
        'n_zeros': N_zeros,
        'k_max': K_MAX,
        'n_candidates': len(top_candidates),
        'elapsed_s': phase2_elapsed,
        'results': results_2,
    }, f, indent=2, default=float)

# ================================================================
# GRAND SUMMARY
# ================================================================
banner("GRAND SUMMARY", '=')

total_time = phase1_elapsed + phase2_elapsed

print(f"\n  Data: {N_zeros:,} genuine Odlyzko zeros")
print(f"  Source: www-users.cse.umn.edu/~odlyzko/zeta_tables/zeros6")
print(f"  K_MAX: {K_MAX}")
print(f"  Total runtime: {total_time:.0f}s ({total_time/60:.1f} min)")

print(f"\n  KEY FINDINGS:")
print(f"  ─────────────")

# Overall best at 500k
if ranked_2:
    best = ranked_2[0]
    print(f"  Best formula (P_MAX=500k): {best['formula']}")
    print(f"    α = {best['alpha']:.8f}, |α-1| = {best['abs_alpha_m1']:.8f}")
    print(f"    T7: {'PASS' if best['t7'] else 'FAIL'}, T8: {'PASS' if best['t8'] else 'FAIL'}")

# Compare key GIFT formulas
key_names = ['GIFT_v1', 'SL_7_6_phi_m2', 'SL_7_6_phi_m15_8']
print(f"\n  Key GIFT formulas comparison:")
for name in key_names:
    r = next((r for r in results_2 if r['name'] == name), None)
    if r:
        print(f"    {r['formula']:35s}  α={r['alpha']:.7f}  "
              f"T7={'Y' if r['t7'] else 'N'} T8={'Y' if r['t8'] else 'N'}  "
              f"drift_p={r['drift_p']:.4f}")

print(f"\n  Results saved to outputs/explore_genuine_phase*.json")
banner("DONE", '=')
