#!/usr/bin/env python3
"""
LOCAL Monte Carlo Validation — Piste 3 Hypothesis Testing
==========================================================

Uses Odlyzko's genuine zeros (downloaded from dtc.umn.edu/~odlyzko/zeta_tables/).
Runs on local RTX 2050 GPU via CuPy.

Usage: python3 run_montecarlo_local.py
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
MERTENS = 0.2614972128476428
EULER_GAMMA = 0.5772156649015329

# GIFT topological constants
DIM_G2 = 14; DIM_K7 = 7; H_STAR = 99; B2 = 21; B3 = 77
RANK_E8 = 8; P2 = 2; N_GEN = 3; DIM_E8 = 248; WEYL = 5
D_BULK = 11; DIM_J3O = 27; DIM_E8xE8 = 496
TAU_GIFT = 3472 / 891

# Formula: theta = 7/6 - phi/(logT - 15/8)
A_FIXED = 7.0 / 6.0
B_FIXED = phi
D_FIXED = -15.0 / 8.0

P_MAX = 500_000
K_MAX = 3
CHECKPOINT_FILE = 'outputs/montecarlo_genuine_results.json'

# GIFT predictions for shape ratio and leading correction
D_PREDICTED = TAU_GIFT + RANK_E8  # 11.8969...
C1_PREDICTED = 6 * MERTENS / 7   # 0.22414...

# ================================================================
# GPU DETECTION
# ================================================================
os.environ.setdefault('CUDA_PATH', '/usr')
try:
    import cupy as cp
    _ = cp.maximum(cp.array([1.0]), 0.0)
    GPU = True
    mem = cp.cuda.Device(0).mem_info[0] // 1024**2
    print(f"[GPU] CuPy — RTX 2050, free={mem} MB")
    # Conservative for 4GB VRAM
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

        # Progress
        pct = 100 * hi / N
        print(f"    chunk {ic+1}: zeros [{lo:,}..{hi:,}]  ({pct:.0f}%)", flush=True)

    return result

def banner(title, char='='):
    w = 72
    print(f"\n{char * w}\n  {title}\n{char * w}", flush=True)

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
    print(f"  [checkpoint] {part_name} saved", flush=True)

def fit_piste3(T_mids, alphas):
    T_arr = np.array(T_mids, dtype=np.float64)
    a_arr = np.array(alphas, dtype=np.float64)
    logT = np.log(T_arr)
    y = a_arr - 1.0
    X = np.column_stack([-1.0 / logT, 1.0 / logT**2])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    c1, c2 = float(beta[0]), float(beta[1])
    y_pred = X @ beta
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    R2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    resid = y - y_pred
    x_idx = np.arange(len(resid))
    _, _, _, p_resid, _ = stats.linregress(x_idx, resid)
    d = float(c2 / c1) if abs(c1) > 1e-15 else float('inf')
    return {'c1': c1, 'c2': c2, 'shape_ratio': d, 'R2': R2, 'residual_drift_p': float(p_resid)}

# ================================================================
# MAIN
# ================================================================
banner("MONTE CARLO VALIDATION — Genuine Odlyzko Zeros")

print(f"\nData source: Andrew Odlyzko's tables (www-users.cse.umn.edu/~odlyzko/)")
print(f"Accuracy: within 4×10⁻⁹")
print(f"Predictions under test:")
print(f"  H1: shape ratio d = tau + rank(E8) = {D_PREDICTED:.6f}")
print(f"  H2: c1 = 6M/7 = {C1_PREDICTED:.6f}")
print()

# ================================================================
# LOAD GENUINE ZEROS
# ================================================================
banner("Loading genuine Odlyzko zeros")

npy_file = 'outputs/odlyzko_zeros_2M.npy'
if not os.path.exists(npy_file):
    print("  ERROR: Run the download first")
    sys.exit(1)

gamma0 = np.load(npy_file)
N_zeros = len(gamma0)
print(f"  Loaded {N_zeros:,} zeros")
print(f"  Range: [{gamma0[0]:.6f}, {gamma0[-1]:.6f}]")

# Cross-validate: these must NOT be Gram approximations
# Gram points for first 100 zeros
ns = np.arange(1, 101, dtype=np.float64)
targets = (ns - 1.5) * np.pi
from scipy.special import lambertw
w_lw = np.real(lambertw(ns / np.e))
t_gram = np.maximum(2 * np.pi * ns / w_lw, 2.0)
for _ in range(40):
    dt = (theta_rs(t_gram) - targets) / np.maximum(np.abs(0.5 * np.log(t_gram / (2 * np.pi))), 1e-15)
    t_gram -= dt

diffs_gram = np.abs(gamma0[:100] - t_gram)
mean_diff = np.mean(diffs_gram)
n_different = np.sum(diffs_gram > 0.01)
print(f"  Cross-validation (first 100 zeros vs Gram):")
print(f"    Mean |genuine - gram|: {mean_diff:.6f}")
print(f"    Zeros differing by >0.01: {n_different}/100")
if mean_diff < 0.001:
    print("  FATAL: These look like Gram/smooth zeros, not genuine!")
    sys.exit(1)
print(f"  CONFIRMED: genuine Odlyzko zeros")

primes = sieve(P_MAX)
tp_v = theta_deriv(gamma0)
print(f"  {len(primes):,} primes up to {P_MAX:,}")

# ================================================================
# BASELINE
# ================================================================
banner("Part 0: Compute baseline S_w for all 2M zeros")
t0 = time.time()

delta_pred = np.sign(np.sin(np.pi * (theta_rs(gamma0) / np.pi - 0.5)))
S_w = prime_sum(gamma0, tp_v, primes, K_MAX, A_FIXED, B_FIXED, D_FIXED)
delta = delta_pred + S_w

elapsed = time.time() - t0
print(f"  Baseline computed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

# Global alpha
denom_global = np.dot(delta_pred, delta_pred)
alpha_global = float(np.dot(delta, delta_pred) / denom_global)
print(f"  Global alpha = {alpha_global:.8f}")

# 50-window alphas
N_WIN = 50
bounds = [int(i * N_zeros / N_WIN) for i in range(N_WIN + 1)]
base_alphas = np.empty(N_WIN)
base_T_mids = np.empty(N_WIN)
for i in range(N_WIN):
    lo, hi = bounds[i], bounds[i+1]
    dp_w = delta_pred[lo:hi]
    d_w = delta[lo:hi]
    den = np.dot(dp_w, dp_w)
    base_alphas[i] = float(np.dot(d_w, dp_w) / den) if den > 0 else 0.0
    base_T_mids[i] = float(gamma0[(lo + hi) // 2])

base_fit = fit_piste3(base_T_mids, base_alphas)
print(f"\n  Baseline Piste 3 fit:")
print(f"    c1 = {base_fit['c1']:.6f}  (GIFT: 6M/7 = {C1_PREDICTED:.6f}, "
      f"match: {abs(base_fit['c1'] - C1_PREDICTED)/C1_PREDICTED*100:.2f}%)")
print(f"    c2 = {base_fit['c2']:.6f}")
print(f"    d  = {base_fit['shape_ratio']:.6f}  (GIFT: tau+8 = {D_PREDICTED:.6f}, "
      f"match: {abs(base_fit['shape_ratio'] - D_PREDICTED)/D_PREDICTED*100:.3f}%)")
print(f"    R2 = {base_fit['R2']:.4f}")
print(f"    resid_drift_p = {base_fit['residual_drift_p']:.4f}")

# Store observed values from THIS run (genuine zeros)
D_OBSERVED = base_fit['shape_ratio']
C1_OBSERVED = base_fit['c1']

save_checkpoint({
    'data_source': 'Odlyzko zeros6 (www-users.cse.umn.edu/~odlyzko/zeta_tables/)',
    'accuracy': '4e-9',
    'n_zeros': N_zeros,
    'alpha_global': alpha_global,
    'n_windows': N_WIN,
    'base_fit': base_fit,
    'base_alphas': base_alphas.tolist(),
    'base_T_mids': base_T_mids.tolist(),
    'elapsed_s': elapsed,
}, 'part0_baseline')

# ================================================================
# TEST 1: BOOTSTRAP
# ================================================================
banner("Test 1: Bootstrap stability — 2000 resamples")
t0 = time.time()

N_BOOT = 2000
rng = np.random.default_rng(2026)

boot_c1 = np.empty(N_BOOT)
boot_c2 = np.empty(N_BOOT)
boot_d = np.empty(N_BOOT)
boot_R2 = np.empty(N_BOOT)
boot_resid_p = np.empty(N_BOOT)

for b in range(N_BOOT):
    idx = rng.integers(0, N_WIN, size=N_WIN)
    boot_T = base_T_mids[idx]
    boot_a = base_alphas[idx]
    order = np.argsort(boot_T)
    fit = fit_piste3(boot_T[order], boot_a[order])
    boot_c1[b] = fit['c1']
    boot_c2[b] = fit['c2']
    boot_d[b] = fit['shape_ratio']
    boot_R2[b] = fit['R2']
    boot_resid_p[b] = fit['residual_drift_p']

    if (b + 1) % 500 == 0:
        print(f"  Boot {b+1}/{N_BOOT}: d={np.mean(boot_d[:b+1]):.4f} "
              f"+/- {np.std(boot_d[:b+1]):.4f}", flush=True)

d_ci = (float(np.percentile(boot_d, 2.5)), float(np.percentile(boot_d, 97.5)))
c1_ci = (float(np.percentile(boot_c1, 2.5)), float(np.percentile(boot_c1, 97.5)))

d_gift_in_ci = bool(d_ci[0] <= D_PREDICTED <= d_ci[1])
c1_gift_in_ci = bool(c1_ci[0] <= C1_PREDICTED <= c1_ci[1])

d_z = float((D_PREDICTED - np.mean(boot_d)) / np.std(boot_d)) if np.std(boot_d) > 0 else float('inf')
c1_z = float((C1_PREDICTED - np.mean(boot_c1)) / np.std(boot_c1)) if np.std(boot_c1) > 0 else float('inf')

frac_pass = float(np.mean(boot_resid_p > 0.05))
elapsed1 = time.time() - t0

print(f"\n  Results ({elapsed1:.1f}s):")
print(f"  d:  {np.mean(boot_d):.4f} +/- {np.std(boot_d):.4f}  "
      f"CI=[{d_ci[0]:.4f}, {d_ci[1]:.4f}]  GIFT {D_PREDICTED:.4f} in CI: "
      f"{'YES' if d_gift_in_ci else 'NO'}  z={d_z:.2f}")
print(f"  c1: {np.mean(boot_c1):.6f} +/- {np.std(boot_c1):.6f}  "
      f"CI=[{c1_ci[0]:.6f}, {c1_ci[1]:.6f}]  GIFT {C1_PREDICTED:.6f} in CI: "
      f"{'YES' if c1_gift_in_ci else 'NO'}  z={c1_z:.2f}")
print(f"  Frac resid_p>0.05: {frac_pass:.3f}")

save_checkpoint({
    'n_bootstrap': N_BOOT,
    'd_mean': float(np.mean(boot_d)), 'd_std': float(np.std(boot_d)),
    'd_ci_95': list(d_ci), 'd_gift_in_ci': d_gift_in_ci, 'd_z_score': d_z,
    'c1_mean': float(np.mean(boot_c1)), 'c1_std': float(np.std(boot_c1)),
    'c1_ci_95': list(c1_ci), 'c1_gift_in_ci': c1_gift_in_ci, 'c1_z_score': c1_z,
    'c2_mean': float(np.mean(boot_c2)), 'c2_std': float(np.std(boot_c2)),
    'R2_mean': float(np.mean(boot_R2)),
    'frac_resid_pass': frac_pass,
    'elapsed_s': elapsed1,
}, 'test1_bootstrap')

# ================================================================
# TEST 2: LOOK-ELSEWHERE EFFECT
# ================================================================
banner("Test 2: Look-Elsewhere Effect — GIFT constant combinations")
t0 = time.time()

GIFT_CONSTANTS = {
    'dim_E8': 248, 'rank_E8': 8, 'dim_G2': 14, 'dim_K7': 7,
    'b2': 21, 'b3': 77, 'H_star': 99, 'p2': 2, 'N_gen': 3,
    'Weyl': 5, 'D_bulk': 11, 'dim_J3O': 27, 'two_b2': 42, 'b0': 1,
    'dim_E8xE8': 496,
}
GIFT_RATIONALS = {
    'tau': TAU_GIFT, 'sin2_thetaW': 3/13, 'kappa_T': 1/61,
    'det_g': 65/32, 'phi': phi, 'pi': math.pi, 'e': math.e,
    'M': MERTENS, 'gamma': EULER_GAMMA,
}

all_vals = {}
all_vals.update(GIFT_CONSTANTS)
all_vals.update(GIFT_RATIONALS)

matches = []
TOLERANCE = 0.05
items = sorted(all_vals.items())

# X + Y
for (n1, v1), (n2, v2) in combinations_with_replacement(items, 2):
    s = v1 + v2
    if abs(s) > 0:
        rel_err = abs(s - D_OBSERVED) / abs(D_OBSERVED)
        if rel_err < TOLERANCE:
            matches.append({'expr': f"{n1} + {n2}", 'value': float(s),
                            'rel_error': float(rel_err), 'abs_error': float(abs(s - D_OBSERVED))})
# X - Y
for n1, v1 in items:
    for n2, v2 in items:
        if n1 == n2: continue
        s = v1 - v2
        if s > 0 and abs(s - D_OBSERVED) / abs(D_OBSERVED) < TOLERANCE:
            matches.append({'expr': f"{n1} - {n2}", 'value': float(s),
                            'rel_error': float(abs(s - D_OBSERVED) / abs(D_OBSERVED)),
                            'abs_error': float(abs(s - D_OBSERVED))})
# X * Y
for (n1, v1), (n2, v2) in combinations_with_replacement(items, 2):
    s = v1 * v2
    if abs(s) > 0 and abs(s - D_OBSERVED) / abs(D_OBSERVED) < TOLERANCE:
        matches.append({'expr': f"{n1} * {n2}", 'value': float(s),
                        'rel_error': float(abs(s - D_OBSERVED) / abs(D_OBSERVED)),
                        'abs_error': float(abs(s - D_OBSERVED))})
# X / Y
for n1, v1 in items:
    for n2, v2 in items:
        if abs(v2) < 1e-15 or n1 == n2: continue
        s = v1 / v2
        if abs(s) > 0 and abs(s - D_OBSERVED) / abs(D_OBSERVED) < TOLERANCE:
            matches.append({'expr': f"{n1} / {n2}", 'value': float(s),
                            'rel_error': float(abs(s - D_OBSERVED) / abs(D_OBSERVED)),
                            'abs_error': float(abs(s - D_OBSERVED))})

matches.sort(key=lambda m: m['rel_error'])

n_add = len(list(combinations_with_replacement(all_vals, 2)))
n_sub = len(all_vals) * (len(all_vals) - 1)
N_TOTAL_COMBOS = 2 * n_add + 2 * n_sub

n_within_1pct = sum(1 for m in matches if m['rel_error'] < 0.01)
n_within_01pct = sum(1 for m in matches if m['rel_error'] < 0.001)

best_err = matches[0]['rel_error'] if matches else 1.0
p_lee_bonf = min(1.0, N_TOTAL_COMBOS * 2 * best_err)

unique_5pct = set(round(m['value'], 2) for m in matches)
N_eff = len(unique_5pct)
p_lee_empirical = min(1.0, 1 - (1 - 2 * best_err) ** N_eff)

gap_ratio = matches[1]['rel_error'] / matches[0]['rel_error'] if len(matches) >= 2 and matches[0]['rel_error'] > 0 else float('inf')

elapsed2 = time.time() - t0

print(f"  Scanned {N_TOTAL_COMBOS:,} combinations ({len(all_vals)} constants)")
print(f"  Matches within 5%: {len(matches)}, within 1%: {n_within_1pct}, within 0.1%: {n_within_01pct}")
print(f"\n  Top 10:")
for i, m in enumerate(matches[:10]):
    star = " <-- GIFT" if 'tau' in m['expr'] and 'rank_E8' in m['expr'] else ""
    print(f"    {i+1}. {m['expr']:30s} = {m['value']:.6f}  (err={m['rel_error']*100:.4f}%){star}")
print(f"\n  LEE p-value (Bonferroni): {p_lee_bonf:.4f}")
print(f"  LEE p-value (empirical, N_eff={N_eff}): {p_lee_empirical:.4f}")
print(f"  Gap ratio #2/#1: {gap_ratio:.1f}x")

p_lee = p_lee_empirical

save_checkpoint({
    'n_total_combinations': N_TOTAL_COMBOS, 'n_effective_targets': N_eff,
    'n_within_1pct': n_within_1pct, 'n_within_01pct': n_within_01pct,
    'top_20_matches': matches[:20],
    'p_lee_bonferroni': p_lee_bonf, 'p_lee_empirical': p_lee_empirical,
    'p_lee': p_lee, 'gap_ratio': gap_ratio,
    'best_match': matches[0] if matches else None,
    'elapsed_s': elapsed2,
}, 'test2_lee')

# ================================================================
# TEST 3: PERMUTATION TEST
# ================================================================
banner("Test 3: Permutation test — 10000 shuffles")
t0 = time.time()

N_PERM = 10000
perm_d = np.empty(N_PERM)
perm_c1 = np.empty(N_PERM)
perm_R2 = np.empty(N_PERM)

for p_idx in range(N_PERM):
    shuffled = rng.permutation(base_alphas)
    fit = fit_piste3(base_T_mids, shuffled)
    perm_d[p_idx] = fit['shape_ratio']
    perm_c1[p_idx] = fit['c1']
    perm_R2[p_idx] = fit['R2']

    if (p_idx + 1) % 2500 == 0:
        n_cl = np.sum(np.abs(perm_d[:p_idx+1] - D_PREDICTED) <= abs(D_OBSERVED - D_PREDICTED))
        print(f"  Perm {p_idx+1}/{N_PERM}: n_closer={n_cl}", flush=True)

obs_dist_d = abs(D_OBSERVED - D_PREDICTED)
n_close_d = int(np.sum(np.abs(perm_d - D_PREDICTED) <= obs_dist_d))
p_perm_d = float((n_close_d + 1) / (N_PERM + 1))

obs_dist_c1 = abs(C1_OBSERVED - C1_PREDICTED)
n_close_c1 = int(np.sum(np.abs(perm_c1 - C1_PREDICTED) <= obs_dist_c1))
p_perm_c1 = float((n_close_c1 + 1) / (N_PERM + 1))

n_better_R2 = int(np.sum(perm_R2 >= base_fit['R2']))
p_R2 = float((n_better_R2 + 1) / (N_PERM + 1))

elapsed3 = time.time() - t0

print(f"\n  Results ({elapsed3:.1f}s):")
print(f"  d:  obs={D_OBSERVED:.4f}, pred={D_PREDICTED:.4f}, "
      f"null={np.mean(perm_d):.4f}+/-{np.std(perm_d):.2f}, p={p_perm_d:.4f} "
      f"{'***' if p_perm_d < 0.05 else ''}")
print(f"  c1: obs={C1_OBSERVED:.6f}, pred={C1_PREDICTED:.6f}, p={p_perm_c1:.4f} "
      f"{'***' if p_perm_c1 < 0.05 else ''}")
print(f"  R2: obs={base_fit['R2']:.4f}, null={np.mean(perm_R2):.4f}, p={p_R2:.4f} "
      f"{'***' if p_R2 < 0.05 else ''}")

save_checkpoint({
    'n_permutations': N_PERM,
    'd_observed': D_OBSERVED, 'd_predicted': D_PREDICTED,
    'd_null_mean': float(np.mean(perm_d)), 'd_null_std': float(np.std(perm_d)),
    'n_close_d': n_close_d, 'p_value_d': p_perm_d,
    'c1_observed': C1_OBSERVED, 'c1_predicted': C1_PREDICTED,
    'n_close_c1': n_close_c1, 'p_value_c1': p_perm_c1,
    'R2_observed': base_fit['R2'], 'R2_null_mean': float(np.mean(perm_R2)),
    'n_better_R2': n_better_R2, 'p_value_R2': p_R2,
    'elapsed_s': elapsed3,
}, 'test3_permutation')

# ================================================================
# TEST 4: JACKKNIFE
# ================================================================
banner("Test 4: Jackknife — leave-one-window-out")
t0 = time.time()

jack_d = np.empty(N_WIN)
jack_c1 = np.empty(N_WIN)

for i in range(N_WIN):
    mask = np.ones(N_WIN, dtype=bool); mask[i] = False
    fit = fit_piste3(base_T_mids[mask], base_alphas[mask])
    jack_d[i] = fit['shape_ratio']
    jack_c1[i] = fit['c1']

jack_d_mean = float(np.mean(jack_d))
jack_d_se = float(np.sqrt((N_WIN - 1) / N_WIN * np.sum((jack_d - jack_d_mean)**2)))
jack_c1_mean = float(np.mean(jack_c1))
jack_c1_se = float(np.sqrt((N_WIN - 1) / N_WIN * np.sum((jack_c1 - jack_c1_mean)**2)))

jack_d_ci = (jack_d_mean - 1.96 * jack_d_se, jack_d_mean + 1.96 * jack_d_se)
jack_c1_ci = (jack_c1_mean - 1.96 * jack_c1_se, jack_c1_mean + 1.96 * jack_c1_se)

jack_d_gift_in = bool(jack_d_ci[0] <= D_PREDICTED <= jack_d_ci[1])
jack_c1_gift_in = bool(jack_c1_ci[0] <= C1_PREDICTED <= jack_c1_ci[1])

most_inf = int(np.argmax(np.abs(jack_d - D_OBSERVED)))
elapsed4 = time.time() - t0

print(f"  d:  {jack_d_mean:.4f} +/- {jack_d_se:.4f}  CI=[{jack_d_ci[0]:.4f}, {jack_d_ci[1]:.4f}]  "
      f"GIFT in CI: {'YES' if jack_d_gift_in else 'NO'}")
print(f"  c1: {jack_c1_mean:.6f} +/- {jack_c1_se:.6f}  CI=[{jack_c1_ci[0]:.6f}, {jack_c1_ci[1]:.6f}]  "
      f"GIFT in CI: {'YES' if jack_c1_gift_in else 'NO'}")
print(f"  Most influential window: #{most_inf} (d_without={jack_d[most_inf]:.4f})")

save_checkpoint({
    'd_jack_mean': jack_d_mean, 'd_jack_se': jack_d_se,
    'd_jack_ci': list(jack_d_ci), 'd_gift_in_ci': jack_d_gift_in,
    'c1_jack_mean': jack_c1_mean, 'c1_jack_se': jack_c1_se,
    'c1_jack_ci': list(jack_c1_ci), 'c1_gift_in_ci': jack_c1_gift_in,
    'most_influential': most_inf, 'jack_d_all': jack_d.tolist(),
    'elapsed_s': elapsed4,
}, 'test4_jackknife')

# ================================================================
# TEST 5: SUBRANGE STABILITY
# ================================================================
banner("Test 5: Subrange stability")
t0 = time.time()

subranges = {
    'first_half': (0, N_WIN // 2), 'second_half': (N_WIN // 2, N_WIN),
    'first_third': (0, N_WIN // 3), 'middle_third': (N_WIN // 3, 2 * N_WIN // 3),
    'last_third': (2 * N_WIN // 3, N_WIN),
    'first_quarter': (0, N_WIN // 4), 'last_quarter': (3 * N_WIN // 4, N_WIN),
}

sub_results = {}
for name, (lo, hi) in subranges.items():
    fit = fit_piste3(base_T_mids[lo:hi], base_alphas[lo:hi])
    sub_results[name] = {'d': fit['shape_ratio'], 'c1': fit['c1'], 'R2': fit['R2']}
    print(f"  {name:20s}: d={fit['shape_ratio']:8.4f}  c1={fit['c1']:.6f}  R2={fit['R2']:.4f}")

d_vals = [v['d'] for v in sub_results.values()]
d_spread = max(d_vals) - min(d_vals)
elapsed5 = time.time() - t0

print(f"\n  d spread: {d_spread:.4f} ({d_spread/D_OBSERVED*100:.1f}%)")

sub_results['d_spread'] = float(d_spread)
sub_results['elapsed_s'] = elapsed5
save_checkpoint(sub_results, 'test5_subranges')

# ================================================================
# GRAND VERDICT
# ================================================================
banner("GRAND VERDICT", '=')

verdicts = [
    ('Bootstrap CI contains GIFT d', 'PASS' if d_gift_in_ci else 'FAIL'),
    ('Bootstrap CI contains GIFT c1', 'PASS' if c1_gift_in_ci else 'FAIL'),
    (f'LEE p < 0.05 (p={p_lee:.4f})', 'PASS' if p_lee < 0.05 else 'FAIL'),
    (f'Permutation d (p={p_perm_d:.4f})', 'PASS' if p_perm_d < 0.05 else 'FAIL'),
    (f'Permutation R2 (p={p_R2:.4f})', 'PASS' if p_R2 < 0.05 else 'FAIL'),
    ('Jackknife CI contains GIFT d', 'PASS' if jack_d_gift_in else 'FAIL'),
    (f'Subrange spread < 20% ({d_spread/D_OBSERVED*100:.1f}%)', 'PASS' if d_spread / D_OBSERVED < 0.20 else 'FAIL'),
]

n_pass = sum(1 for _, v in verdicts if v == "PASS")
for desc, v in verdicts:
    print(f"  [{'OK' if v == 'PASS' else 'XX'}] {desc}: {v}")

conclusion = ("SIGNAL CONFIRMED" if n_pass == len(verdicts) else
              "SIGNAL LIKELY" if n_pass >= len(verdicts) - 1 else
              "SIGNAL AMBIGUOUS" if n_pass >= len(verdicts) // 2 else
              "SIGNAL REFUTED")

print(f"\n  Score: {n_pass}/{len(verdicts)} tests passed")
print(f"  Conclusion: {conclusion}")

save_checkpoint({
    'verdicts': {d: v for d, v in verdicts},
    'n_pass': n_pass, 'n_total': len(verdicts),
    'conclusion': conclusion,
}, 'grand_verdict')

total = elapsed + elapsed1 + elapsed2 + elapsed3 + elapsed4 + elapsed5
print(f"\n  Total runtime: {total:.0f}s ({total/60:.1f} min)")
print(f"  Results: {CHECKPOINT_FILE}")
banner("DONE", '=')
