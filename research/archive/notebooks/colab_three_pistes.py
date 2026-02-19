#!/usr/bin/env python3
"""
THREE WINNING PISTES — Unified Colab Validation
=================================================

Explores the three most promising approaches to the T7-T8 anti-correlation:

  Piste 4: The "drift" may be an oscillation — test with 50 windows
  Piste 3: alpha(T) = 1 - c1/logT + c2/logT^2 — drift as GIFT signature
  Piste 6: theta = 7/6 - phi/(logT - 15/8) + 3M/(logT - 15/8)^2

Upload to Colab, mount Drive, run:
  !python colab_three_pistes.py

Expects: outputs/riemann_zeros_2M_genuine.npy
Outputs: outputs/three_pistes_results.json (with per-part checkpoints)

Estimated runtime: ~45 min on A100, ~2h on T4
"""
import numpy as np
from scipy.special import loggamma, lambertw
from scipy import stats
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
D_BULK = 11; DIM_J3O = 27

# Formula: theta = 7/6 - phi/(logT - 15/8)
A_FIXED = 7.0 / 6.0
B_FIXED = phi
D_FIXED = -15.0 / 8.0

P_MAX = 500_000
K_MAX = 3
ZEROS_FILE = 'outputs/riemann_zeros_2M_genuine.npy'
CHECKPOINT_FILE = 'outputs/three_pistes_results.json'

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
# INFRASTRUCTURE
# ================================================================
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
    for it in range(40):
        dt = (theta_rs(t) - targets) / np.maximum(np.abs(theta_deriv(t)), 1e-15)
        t -= dt
        if np.max(np.abs(dt)) < 1e-12:
            break
    return t

def sieve(N):
    is_p = np.ones(N + 1, dtype=bool); is_p[:2] = False
    for i in range(2, int(N**0.5) + 1):
        if is_p[i]: is_p[i*i::i] = False
    return np.where(is_p)[0]

def prime_sum_order2(g0, tp_v, primes, k_max, a, b, d_shift, c_order2):
    """Shifted-log with order-2: theta = a - b/(logT+d) + c/(logT+d)^2"""
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
        theta_per = (xp.float64(a)
                     - xp.float64(b) / denom
                     + xp.float64(c_order2) / (denom * denom))
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

def compute_metrics(delta, delta_pred, gamma0, n_windows, n_boot=1000):
    """Compute alpha, window alphas, T7, T8 for arbitrary window count."""
    denom = np.dot(delta_pred, delta_pred)
    alpha = float(np.dot(delta, delta_pred) / denom) if denom > 0 else 0.0

    N = len(delta)
    bounds = [int(i * N / n_windows) for i in range(n_windows + 1)]
    alphas = []
    T_mids = []
    for i in range(n_windows):
        lo, hi = bounds[i], bounds[i + 1]
        d_w = delta[lo:hi]; dp_w = delta_pred[lo:hi]
        den = np.dot(dp_w, dp_w)
        a_w = float(np.dot(d_w, dp_w) / den) if den > 0 else 0.0
        alphas.append(a_w)
        T_mids.append(float(gamma0[(lo + hi) // 2]))

    # T8: linear regression
    x = np.arange(len(alphas))
    slope, intercept, r_value, p, stderr = stats.linregress(x, alphas)

    # T7: bootstrap CI
    rng = np.random.default_rng(42)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, N, size=N)
        d = delta[idx]; dp = delta_pred[idx]
        den = np.dot(dp, dp)
        boot[i] = np.dot(d, dp) / den if den > 0 else 0.0
    ci_lo = float(np.percentile(boot, 2.5))
    ci_hi = float(np.percentile(boot, 97.5))
    t7 = bool(ci_lo <= 1.0 <= ci_hi)

    return {
        'alpha': alpha,
        'abs_alpha_minus_1': abs(alpha - 1),
        'drift_slope': float(slope),
        'drift_p': float(p),
        'T7_pass': t7,
        'T8_pass': bool(p > 0.05),
        'ci_lo': ci_lo,
        'ci_hi': ci_hi,
        'window_alphas': alphas,
        'T_mids': T_mids,
    }

def save_checkpoint(results, part_name):
    """Incremental save after each part."""
    os.makedirs('outputs', exist_ok=True)
    # Load existing or create new
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            all_data = json.load(f)
    else:
        all_data = {}
    all_data[part_name] = results
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(all_data, f, indent=2, default=float)
    print(f"  [checkpoint] {part_name} saved to {CHECKPOINT_FILE}")
    sys.stdout.flush()

def banner(title, char='='):
    w = 72
    print(f"\n{char * w}")
    print(f"  {title}")
    print(f"{char * w}")
    sys.stdout.flush()

def fmt_short(m):
    t7 = "PASS" if m['T7_pass'] else "FAIL"
    t8 = "PASS" if m['T8_pass'] else "FAIL"
    both = " ***" if m['T7_pass'] and m['T8_pass'] else ""
    return (f"alpha={m['alpha']:+.8f}  |a-1|={m['abs_alpha_minus_1']:.6f}  "
            f"drift_p={m['drift_p']:.4f}  T7={t7}  T8={t8}{both}")

# ================================================================
# MAIN
# ================================================================
banner("THREE WINNING PISTES — Unified Colab Validation", '=')
print("""
  Piste 4: Oscillation (50 windows, quadratic fit, Fourier)
  Piste 3: alpha(T) = 1 - c1/logT + c2/logT^2 (signature)
  Piste 6: theta = 7/6 - phi/(logT - 15/8) + 3M/(logT - 15/8)^2
""")
print(f"  GPU:    {'Yes (CuPy)' if GPU else 'No (CPU)'}")
print(f"  P_MAX:  {P_MAX:,}")
print(f"  k_max:  {K_MAX}")
print(f"  phi:    {phi:.10f}")
print(f"  d:      -15/8 = {D_FIXED:.10f}")
print(f"  M:      {MERTENS:.10f}")
print(f"  3M:     {3*MERTENS:.10f}")

# ----------------------------------------------------------------
# PART 0: SETUP
# ----------------------------------------------------------------
banner("PART 0: SETUP — Loading zeros and computing primes")

print(f"  Loading {ZEROS_FILE}...")
gamma_n = np.load(ZEROS_FILE)
N_TOTAL = len(gamma_n)
print(f"  {N_TOTAL:,} zeros loaded, T range: [{gamma_n[0]:.1f}, {gamma_n[-1]:.1f}]")

t0 = time.time()
print(f"  Computing smooth zeros (Newton iteration)...")
gamma0 = smooth_zeros(N_TOTAL)
delta = gamma_n - gamma0
tp = theta_deriv(gamma0)
print(f"  Done in {time.time()-t0:.1f}s")

primes = sieve(P_MAX)
n_primes = len(primes)
print(f"  {n_primes:,} primes (P_MAX={P_MAX:,})")
sys.stdout.flush()

# ----------------------------------------------------------------
# PART 1: BASELINE — theta = 7/6 - phi/(logT - 15/8) [c=0]
# ----------------------------------------------------------------
banner("PART 1: BASELINE — theta = 7/6 - phi/(logT - 15/8)")
print(f"  Computing with 12 AND 50 windows...")

t0 = time.time()
dp_base = prime_sum_order2(gamma0, tp, primes, K_MAX, A_FIXED, B_FIXED, D_FIXED, 0.0)
elapsed = time.time() - t0
print(f"  Prime sum computed in {elapsed:.1f}s")

m12 = compute_metrics(delta, dp_base, gamma0, 12)
m50 = compute_metrics(delta, dp_base, gamma0, 50)

print(f"\n  12 windows: {fmt_short(m12)}")
print(f"    alphas: {['%.4f' % a for a in m12['window_alphas']]}")
print(f"\n  50 windows: {fmt_short(m50)}")
print(f"    alphas (first 20): {['%.4f' % a for a in m50['window_alphas'][:20]]}")
print(f"    alphas (last 20):  {['%.4f' % a for a in m50['window_alphas'][30:]]}")

save_checkpoint({
    'config': {'a': A_FIXED, 'b': B_FIXED, 'd': D_FIXED, 'c': 0.0,
               'P_MAX': P_MAX, 'K_MAX': K_MAX, 'N_zeros': N_TOTAL,
               'N_primes': n_primes},
    '12_windows': {k: v for k, v in m12.items()},
    '50_windows': {k: v for k, v in m50.items()},
    'elapsed_s': elapsed,
}, 'part1_baseline')

# ----------------------------------------------------------------
# PART 2: PISTE 4 — Oscillation Analysis (50 windows)
# ----------------------------------------------------------------
banner("PART 2: PISTE 4 — Oscillation Analysis")
print("  Is the 'drift' actually a long-period oscillation?")
print("  Analyzing 50-window alphas from the baseline...\n")

alphas_50 = np.array(m50['window_alphas'])
T_mids_50 = np.array(m50['T_mids'])
n_win = len(alphas_50)
x_idx = np.arange(n_win)

# 2a: Linear vs Quadratic vs Sinusoidal fit
print("  --- Model Comparison ---")

# Linear
sl_lin, int_lin, r_lin, p_lin, se_lin = stats.linregress(x_idx, alphas_50)
r2_lin = r_lin**2
aic_lin = n_win * np.log(np.mean((alphas_50 - (int_lin + sl_lin * x_idx))**2)) + 4

# Quadratic
coeffs_quad = np.polyfit(x_idx, alphas_50, 2)
pred_quad = np.polyval(coeffs_quad, x_idx)
ss_res_quad = np.sum((alphas_50 - pred_quad)**2)
ss_tot = np.sum((alphas_50 - np.mean(alphas_50))**2)
r2_quad = 1 - ss_res_quad / ss_tot
aic_quad = n_win * np.log(ss_res_quad / n_win) + 6

# F-test: quadratic term significant?
ss_res_lin = np.sum((alphas_50 - (int_lin + sl_lin * x_idx))**2)
f_stat = ((ss_res_lin - ss_res_quad) / 1) / (ss_res_quad / (n_win - 3))
p_ftest = 1 - stats.f.cdf(f_stat, 1, n_win - 3)

# Cubic
coeffs_cub = np.polyfit(x_idx, alphas_50, 3)
pred_cub = np.polyval(coeffs_cub, x_idx)
ss_res_cub = np.sum((alphas_50 - pred_cub)**2)
r2_cub = 1 - ss_res_cub / ss_tot
aic_cub = n_win * np.log(ss_res_cub / n_win) + 8

print(f"  {'Model':<18s}  {'R^2':>8s}  {'AIC':>10s}  {'Notes':>30s}")
print(f"  {'-'*18}  {'-'*8}  {'-'*10}  {'-'*30}")
print(f"  {'Linear':<18s}  {r2_lin:8.4f}  {aic_lin:10.1f}  {'p(slope)=%.4f' % p_lin}")
print(f"  {'Quadratic':<18s}  {r2_quad:8.4f}  {aic_quad:10.1f}  "
      f"{'F=%.2f, p=%.4f' % (f_stat, p_ftest)}")
print(f"  {'Cubic':<18s}  {r2_cub:8.4f}  {aic_cub:10.1f}  ")

quad_vertex = -coeffs_quad[1] / (2 * coeffs_quad[0])
print(f"\n  Quadratic vertex (minimum): window {quad_vertex:.1f} / {n_win}")
if 0 < quad_vertex < n_win:
    print(f"    => Alpha reaches minimum INSIDE the window range!")
    print(f"    => Curve reverses direction — consistent with oscillation")
else:
    print(f"    => Vertex outside range — consistent with monotone drift")

# 2b: Fourier analysis of linear residuals
print(f"\n  --- Fourier Analysis (linear residuals) ---")
resid_lin = alphas_50 - (int_lin + sl_lin * x_idx)
fft_vals = np.fft.rfft(resid_lin)
power = np.abs(fft_vals)**2
freqs = np.fft.rfftfreq(n_win)

# Top 5 frequencies
idx_sorted = np.argsort(power[1:])[::-1] + 1  # skip DC
print(f"  {'Rank':<6s}  {'Freq':>10s}  {'Period':>12s}  {'Power':>12s}")
print(f"  {'-'*6}  {'-'*10}  {'-'*12}  {'-'*12}")
for rank, idx in enumerate(idx_sorted[:5]):
    per = 1.0 / freqs[idx] if freqs[idx] > 0 else float('inf')
    print(f"  {rank+1:<6d}  {freqs[idx]:10.4f}  {per:12.1f}  {power[idx]:12.2e}")

# 2c: Prime-induced oscillation periods
print(f"\n  --- Theoretical Prime Oscillation Periods ---")
log_T_span = np.log(T_mids_50[-1]) - np.log(T_mids_50[0])
print(f"  Window T range: [{T_mids_50[0]:.0f}, {T_mids_50[-1]:.0f}]")
print(f"  log(T) span: {log_T_span:.3f}")
print(f"  {'Prime':>6s}  {'Period (logT)':>14s}  {'Period (windows)':>18s}")
print(f"  {'-'*6}  {'-'*14}  {'-'*18}")
for p_val in [2, 3, 5, 7, 11, 13]:
    period_logT = 2 * np.pi / np.log(p_val)
    period_win = period_logT / log_T_span * n_win
    print(f"  {p_val:6d}  {period_logT:14.2f}  {period_win:18.1f}")

# 2d: Non-monotonicity stats
diffs = np.diff(alphas_50)
n_increases = np.sum(diffs > 0)
n_decreases = np.sum(diffs < 0)
sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
print(f"\n  --- Non-Monotonicity (50 windows) ---")
print(f"  Increases: {n_increases}  Decreases: {n_decreases}  Sign changes: {sign_changes}")
print(f"  (Pure monotone: 0 sign changes; Pure oscillation: ~{n_win//2} sign changes)")

# 2e: Second-half stationarity test
half = n_win // 2
alphas_2nd = alphas_50[half:]
x_2nd = np.arange(len(alphas_2nd))
sl2, _, _, p2, _ = stats.linregress(x_2nd, alphas_2nd)
print(f"\n  Second-half stationarity (windows {half}-{n_win}):")
print(f"    slope = {sl2:+.6f}, p = {p2:.4f}  "
      f"{'=> NOT significant (stalled!)' if p2 > 0.05 else '=> still significant'}")

piste4_results = {
    'n_windows': n_win,
    'linear_R2': float(r2_lin), 'linear_p': float(p_lin),
    'quadratic_R2': float(r2_quad), 'quadratic_Ftest_p': float(p_ftest),
    'cubic_R2': float(r2_cub),
    'quadratic_vertex_window': float(quad_vertex),
    'n_increases': int(n_increases), 'n_decreases': int(n_decreases),
    'sign_changes': int(sign_changes),
    'second_half_slope': float(sl2), 'second_half_p': float(p2),
    'fourier_top5_periods': [float(1/freqs[i]) if freqs[i] > 0 else 0
                              for i in idx_sorted[:5]],
    'fourier_top5_powers': [float(power[i]) for i in idx_sorted[:5]],
    'quadratic_coeffs': [float(c) for c in coeffs_quad],
    'verdict': ('oscillation' if (p_ftest < 0.05 and 0 < quad_vertex < n_win)
                else 'ambiguous' if p_ftest < 0.10
                else 'linear_drift'),
}
save_checkpoint(piste4_results, 'part2_piste4_oscillation')

# ----------------------------------------------------------------
# PART 3: PISTE 3 — alpha(T) = 1 - c1/logT + c2/logT^2
# ----------------------------------------------------------------
banner("PART 3: PISTE 3 — Drift as GIFT Signature")
print("  Fitting alpha(T) = 1 - c1/logT + c2/logT^2 to window alphas")
print("  Testing if shape ratio d = c2/c1 is a topological constant\n")

# Use 50-window data for better fit
log_T = np.log(T_mids_50)

# 3a: Free 2-parameter fit: alpha = A + B/logT
print("  --- Free fit: alpha = A_inf + B/logT ---")
inv_logT = 1.0 / log_T
sl_free, A_free, r_free, p_free, _ = stats.linregress(inv_logT, alphas_50)
print(f"  A_inf = {A_free:.6f}  (should be ~1 if drift vanishes at T->inf)")
print(f"  B     = {sl_free:.6f}")
print(f"  R^2   = {r_free**2:.4f}")

# 3b: Forced 2-term fit: alpha = 1 - c1/logT + c2/logT^2
print(f"\n  --- Forced fit: alpha = 1 - c1/logT + c2/logT^2 ---")
y = alphas_50 - 1.0  # alpha - 1
X_mat = np.column_stack([-1.0 / log_T, 1.0 / log_T**2])
# OLS: y = X @ [c1, c2]
c_ols, residuals, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
c1, c2 = c_ols
pred_forced = 1.0 + X_mat @ c_ols
ss_res_forced = np.sum((alphas_50 - pred_forced)**2)
r2_forced = 1 - ss_res_forced / ss_tot

# Residual drift test
resid_forced = alphas_50 - pred_forced
sl_resid, _, _, p_resid, _ = stats.linregress(x_idx, resid_forced)

print(f"  c1 = {c1:.6f}")
print(f"  c2 = {c2:.6f}")
print(f"  R^2 = {r2_forced:.4f}")
print(f"  Residual drift: slope={sl_resid:+.8f}, p={p_resid:.4f}  "
      f"{'PASS (no drift!)' if p_resid > 0.05 else 'FAIL'}")

shape_ratio = c2 / c1 if abs(c1) > 1e-10 else float('inf')
print(f"\n  Shape ratio d = c2/c1 = {shape_ratio:.4f}")

# 3c: GIFT candidates for c1
print(f"\n  --- GIFT matches for c1 = {c1:.6f} ---")
c1_candidates = {
    '2*dim_G2/H* = 28/99':     28.0 / 99,
    'phi/dim_K7 = phi/7':       phi / 7,
    '2/dim_K7 = 2/7':           2.0 / 7,
    'dim_G2/H* = 14/99':        14.0 / 99,
    'gamma/2':                   EULER_GAMMA / 2,
    'M':                         MERTENS,
    '3*M':                       3 * MERTENS,
    'b2/H* = 21/99':            21.0 / 99,
    'phi/dim_G2':               phi / DIM_G2,
    '1/(2*pi)':                 1 / (2 * np.pi),
}
print(f"  {'Expression':<28s}  {'Value':>10s}  {'|error|':>10s}  {'%err':>8s}")
print(f"  {'-'*28}  {'-'*10}  {'-'*10}  {'-'*8}")
for name, val in sorted(c1_candidates.items(), key=lambda x: abs(x[1] - c1)):
    err = abs(val - c1)
    pct = 100 * err / abs(c1) if abs(c1) > 1e-10 else float('inf')
    print(f"  {name:<28s}  {val:10.6f}  {err:10.6f}  {pct:7.2f}%")

# 3d: GIFT candidates for shape ratio d
print(f"\n  --- GIFT matches for d = c2/c1 = {shape_ratio:.4f} ---")
d_candidates = {
    'b3/dim_K7 + phi = 11+phi':  B3 / DIM_K7 + phi,
    '4*pi':                       4 * np.pi,
    'H*/rank_E8 = 99/8':         H_STAR / RANK_E8,
    'dim_G2 - 1 = 13':           13.0,
    'dim_G2':                     14.0,
    'D_bulk + phi':               D_BULK + phi,
    'b2/phi':                     B2 / phi,
    'b3/dim_K7 = 11':            B3 / DIM_K7,
    '2*dim_K7 - phi':            2 * DIM_K7 - phi,
    'sqrt(dim_E8) - 3':          np.sqrt(DIM_E8) - 3,
}
print(f"  {'Expression':<28s}  {'Value':>10s}  {'|error|':>10s}  {'%err':>8s}")
print(f"  {'-'*28}  {'-'*10}  {'-'*10}  {'-'*8}")
for name, val in sorted(d_candidates.items(), key=lambda x: abs(x[1] - shape_ratio)):
    err = abs(val - shape_ratio)
    pct = 100 * err / abs(shape_ratio) if abs(shape_ratio) > 1e-10 else float('inf')
    print(f"  {name:<28s}  {val:10.4f}  {err:10.4f}  {pct:7.2f}%")

piste3_results = {
    'free_fit': {'A_inf': float(A_free), 'B': float(sl_free), 'R2': float(r_free**2)},
    'forced_fit': {
        'c1': float(c1), 'c2': float(c2), 'R2': float(r2_forced),
        'residual_drift_slope': float(sl_resid),
        'residual_drift_p': float(p_resid),
        'residual_drift_pass': bool(p_resid > 0.05),
    },
    'shape_ratio': float(shape_ratio),
    'best_c1_match': '28/99' if abs(28/99 - c1) == min(abs(v - c1) for v in c1_candidates.values()) else 'other',
    'best_d_match': 'b3/dim_K7+phi' if abs(B3/DIM_K7 + phi - shape_ratio) < 0.1 else 'other',
}
save_checkpoint(piste3_results, 'part3_piste3_signature')

# ----------------------------------------------------------------
# PART 4: PISTE 6 — Mertens candidates for c_order2
# ----------------------------------------------------------------
banner("PART 4: PISTE 6 — Mertens-Based Correction")
print("  theta = 7/6 - phi/(logT - 15/8) + c/(logT - 15/8)^2")
print("  Testing specific c candidates including c = 3M\n")

CANDIDATES = {
    'c=0 (baseline)':             0.0,
    'c=3M (N_gen*Mertens)':       N_GEN * MERTENS,
    'c=7/9 (dim_K7/(rank_E8+1))': DIM_K7 / (RANK_E8 + 1),
    'c=21/27 (b2/dim_J3O)':       B2 / float(DIM_J3O),
    'c=11/14 (D_bulk/dim_G2)':    D_BULK / float(DIM_G2),
    'c=pi/4':                      math.pi / 4,
    'c=0.78 (empirical)':          0.78,
    'c=4/5 (4/Weyl)':             4.0 / WEYL,
    'c=15/19':                     15.0 / 19.0,
    'c=phi/2':                     phi / 2,
    'c=gamma':                     EULER_GAMMA,
    'c=1':                         1.0,
}

cand_results = {}
for name, c_val in CANDIDATES.items():
    t0 = time.time()
    dp = prime_sum_order2(gamma0, tp, primes, K_MAX, A_FIXED, B_FIXED, D_FIXED, c_val)
    elapsed = time.time() - t0

    m12 = compute_metrics(delta, dp, gamma0, 12)
    m50 = compute_metrics(delta, dp, gamma0, 50)

    print(f"  {name:<30s}  c={c_val:.8f}  [{elapsed:.1f}s]")
    print(f"    12w: {fmt_short(m12)}")
    print(f"    50w: {fmt_short(m50)}")

    # Piste 3 analysis on this candidate's 50-window alphas
    alphas_c = np.array(m50['window_alphas'])
    T_mids_c = np.array(m50['T_mids'])
    log_T_c = np.log(T_mids_c)
    y_c = alphas_c - 1.0
    X_c = np.column_stack([-1.0 / log_T_c, 1.0 / log_T_c**2])
    c_fit, _, _, _ = np.linalg.lstsq(X_c, y_c, rcond=None)
    pred_c = 1.0 + X_c @ c_fit
    resid_c = alphas_c - pred_c
    _, _, _, p_resid_c, _ = stats.linregress(np.arange(len(resid_c)), resid_c)
    shape_c = c_fit[1] / c_fit[0] if abs(c_fit[0]) > 1e-10 else float('inf')
    print(f"    P3 fit: c1={c_fit[0]:.6f}, c2={c_fit[1]:.6f}, d=c2/c1={shape_c:.2f}, "
          f"resid_p={p_resid_c:.4f}")
    print()

    cand_results[name] = {
        'c': float(c_val),
        '12w': {k: v for k, v in m12.items()},
        '50w': {k: v for k, v in m50.items()},
        'piste3_fit': {
            'c1': float(c_fit[0]), 'c2': float(c_fit[1]),
            'shape_ratio': float(shape_c),
            'residual_drift_p': float(p_resid_c),
        },
        'elapsed_s': elapsed,
    }
    sys.stdout.flush()

save_checkpoint(cand_results, 'part4_piste6_mertens')

# ----------------------------------------------------------------
# PART 5: GRAND COMPARISON & RANKINGS
# ----------------------------------------------------------------
banner("PART 5: GRAND COMPARISON")

# 5a: 12-window rankings
print("\n  --- 12-Window Rankings (by |alpha-1|) ---")
print(f"  {'Name':<30s}  {'|a-1|':>10s}  {'drift_p':>8s}  {'T7':>4s}  {'T8':>4s}  {'Both':>5s}")
print(f"  {'-'*30}  {'-'*10}  {'-'*8}  {'-'*4}  {'-'*4}  {'-'*5}")
for name, r in sorted(cand_results.items(), key=lambda x: x[1]['12w']['abs_alpha_minus_1']):
    m = r['12w']
    t7 = 'P' if m['T7_pass'] else 'F'
    t8 = 'P' if m['T8_pass'] else 'F'
    both = 'YES' if m['T7_pass'] and m['T8_pass'] else ''
    print(f"  {name:<30s}  {m['abs_alpha_minus_1']:10.6f}  {m['drift_p']:8.4f}  {t7:>4s}  {t8:>4s}  {both:>5s}")

# 5b: 50-window rankings
print("\n  --- 50-Window Rankings (by |alpha-1|) ---")
print(f"  {'Name':<30s}  {'|a-1|':>10s}  {'drift_p':>8s}  {'T7':>4s}  {'T8':>4s}  {'Both':>5s}")
print(f"  {'-'*30}  {'-'*10}  {'-'*8}  {'-'*4}  {'-'*4}  {'-'*5}")
for name, r in sorted(cand_results.items(), key=lambda x: x[1]['50w']['abs_alpha_minus_1']):
    m = r['50w']
    t7 = 'P' if m['T7_pass'] else 'F'
    t8 = 'P' if m['T8_pass'] else 'F'
    both = 'YES' if m['T7_pass'] and m['T8_pass'] else ''
    print(f"  {name:<30s}  {m['abs_alpha_minus_1']:10.6f}  {m['drift_p']:8.4f}  {t7:>4s}  {t8:>4s}  {both:>5s}")

# 5c: Shape ratio stability
print("\n  --- Shape Ratio d = c2/c1 Stability (Piste 3 on each candidate) ---")
print(f"  {'Name':<30s}  {'c1':>10s}  {'c2':>10s}  {'d=c2/c1':>10s}  {'resid_p':>8s}")
print(f"  {'-'*30}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")
shape_ratios = []
for name, r in cand_results.items():
    pf = r['piste3_fit']
    print(f"  {name:<30s}  {pf['c1']:10.6f}  {pf['c2']:10.6f}  "
          f"{pf['shape_ratio']:10.4f}  {pf['residual_drift_p']:8.4f}")
    if abs(pf['c1']) > 0.01:  # skip near-zero c1
        shape_ratios.append(pf['shape_ratio'])

if shape_ratios:
    d_mean = np.mean(shape_ratios)
    d_std = np.std(shape_ratios)
    print(f"\n  Shape ratio statistics: mean = {d_mean:.4f}, std = {d_std:.4f}, "
          f"CV = {100*d_std/abs(d_mean):.1f}%")
    print(f"  b3/dim_K7 + phi = {B3/DIM_K7 + phi:.4f}  "
          f"(diff = {abs(d_mean - (B3/DIM_K7 + phi)):.4f})")

# 5d: Winners
print(f"\n  --- WINNERS ---")
both_12 = {n: r for n, r in cand_results.items() if r['12w']['T7_pass'] and r['12w']['T8_pass']}
both_50 = {n: r for n, r in cand_results.items() if r['50w']['T7_pass'] and r['50w']['T8_pass']}

if both_12:
    print(f"\n  Passing BOTH T7+T8 at 12 windows:")
    for name, r in sorted(both_12.items(), key=lambda x: x[1]['12w']['abs_alpha_minus_1']):
        m = r['12w']
        print(f"    {name}: alpha={m['alpha']:+.8f}  drift_p={m['drift_p']:.4f}")
else:
    print(f"\n  No candidate passes both T7+T8 at 12 windows.")
    # Closest
    score_12 = [(n, r, r['12w']['abs_alpha_minus_1'] + max(0, 0.05 - r['12w']['drift_p']))
                for n, r in cand_results.items()]
    score_12.sort(key=lambda x: x[2])
    print(f"  Closest (composite score):")
    for name, r, sc in score_12[:3]:
        m = r['12w']
        print(f"    {name}: score={sc:.6f}  |a-1|={m['abs_alpha_minus_1']:.6f}  "
              f"drift_p={m['drift_p']:.4f}")

if both_50:
    print(f"\n  Passing BOTH T7+T8 at 50 windows:")
    for name, r in sorted(both_50.items(), key=lambda x: x[1]['50w']['abs_alpha_minus_1']):
        m = r['50w']
        print(f"    {name}: alpha={m['alpha']:+.8f}  drift_p={m['drift_p']:.4f}")
else:
    print(f"\n  No candidate passes both T7+T8 at 50 windows.")
    score_50 = [(n, r, r['50w']['abs_alpha_minus_1'] + max(0, 0.05 - r['50w']['drift_p']))
                for n, r in cand_results.items()]
    score_50.sort(key=lambda x: x[2])
    print(f"  Closest (composite score):")
    for name, r, sc in score_50[:3]:
        m = r['50w']
        print(f"    {name}: score={sc:.6f}  |a-1|={m['abs_alpha_minus_1']:.6f}  "
              f"drift_p={m['drift_p']:.4f}")

# ----------------------------------------------------------------
# FINAL SAVE
# ----------------------------------------------------------------
banner("DONE")

save_checkpoint({
    'n_candidates': len(CANDIDATES),
    'both_t7t8_12w': list(both_12.keys()),
    'both_t7t8_50w': list(both_50.keys()),
    'shape_ratio_mean': float(d_mean) if shape_ratios else None,
    'shape_ratio_std': float(d_std) if shape_ratios else None,
    'piste4_verdict': piste4_results['verdict'],
    'piste3_residual_drift_pass': piste3_results['forced_fit']['residual_drift_pass'],
    'piste6_best_12w': min(cand_results.items(),
                            key=lambda x: x[1]['12w']['abs_alpha_minus_1'])[0],
}, 'summary')

print(f"\n  All results saved to {CHECKPOINT_FILE}")
print(f"  Total runtime: see individual part timings above")
print(f"\n  Key questions answered:")
print(f"    P4: Is the drift an oscillation?  => {piste4_results['verdict'].upper()}")
print(f"    P3: Is shape ratio d stable?      => d = {shape_ratio:.2f}")
print(f"    P6: Does 3M pass T7+T8?           => {'YES' if 'c=3M (N_gen*Mertens)' in both_12 else 'CHECK RESULTS'}")
print(f"\nDone!")
