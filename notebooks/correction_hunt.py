#!/usr/bin/env python3
"""
Reverse-engineering the theta correction from 2M zeros.
Goal: find theta(T) = theta_inf + c/log(T) where theta_inf, c
are expressible in terms of GIFT structural constants.
"""
import numpy as np
import json
import time
from scipy import stats
from scipy.special import loggamma, lambertw
from scipy.optimize import minimize_scalar, minimize
from itertools import product as iterproduct
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

OUTDIR = '/home/brieuc/gift-framework/GIFT-research/notebooks/outputs'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}", flush=True)
t0 = time.time()

# =====================================================================
# LOAD DATA
# =====================================================================
print("Loading data...", flush=True)
gamma_n = np.load(f'{OUTDIR}/riemann_zeros_2M_genuine.npy')
delta_pred = np.load(f'{OUTDIR}/dp_const_2M.npy')

N_ZEROS = len(gamma_n)
THETA_STAR = 0.9941
K_MAX = 3

# Smooth zeros
ns = np.arange(1, N_ZEROS + 1, dtype=np.float64)
targets = (ns - 1.5) * np.pi
w_lw = np.real(lambertw(ns / np.e))
gamma0 = np.maximum(2 * np.pi * ns / w_lw, 2.0)
for _ in range(40):
    th = np.imag(loggamma(0.25 + 0.5j * gamma0)) - 0.5 * gamma0 * np.log(np.pi)
    tp = 0.5 * np.log(np.maximum(gamma0, 1.0) / (2 * np.pi))
    gamma0 -= (th - targets) / np.maximum(np.abs(tp), 1e-15)
    if np.max(np.abs((th - targets) / np.maximum(np.abs(tp), 1e-15))) < 1e-10:
        break

delta = gamma_n - gamma0
tp = 0.5 * np.log(np.maximum(gamma0, 1.0) / (2 * np.pi))
residuals = delta - delta_pred
print(f"  Ready [{time.time()-t0:.1f}s]", flush=True)


# =====================================================================
# PART 1: SLIDING-WINDOW ALPHA(T) -- FAST
# =====================================================================
print("\n" + "=" * 70, flush=True)
print("PART 1: ALPHA(T) IN SLIDING WINDOWS", flush=True)
print("=" * 70, flush=True)

WIN_SIZE = 50_000
STRIDE = 25_000
alpha_data = []

for start in range(0, N_ZEROS - WIN_SIZE + 1, STRIDE):
    end = start + WIN_SIZE
    d_w = delta[start:end]
    dp_w = delta_pred[start:end]
    dot_pp = np.dot(dp_w, dp_w)
    alpha_w = float(np.dot(d_w, dp_w) / dot_pp) if dot_pp > 0 else 1.0
    R2_w = float(1.0 - np.var(d_w - dp_w) / np.var(d_w))
    T_mid = float((gamma_n[start] + gamma_n[min(end - 1, N_ZEROS - 1)]) / 2)
    alpha_data.append({'start': start, 'end': end, 'T_mid': T_mid,
                       'alpha': alpha_w, 'R2': R2_w})

n_win = len(alpha_data)
T_mids = np.array([d['T_mid'] for d in alpha_data])
alphas = np.array([d['alpha'] for d in alpha_data])
R2s = np.array([d['R2'] for d in alpha_data])
log_T = np.log(T_mids)
inv_log_T = 1.0 / log_T

print(f"  {n_win} windows of {WIN_SIZE//1000}k, stride {STRIDE//1000}k", flush=True)
print(f"  alpha range: [{alphas.min():.4f}, {alphas.max():.4f}]", flush=True)
print(f"  R2 range:    [{R2s.min():.4f}, {R2s.max():.4f}]", flush=True)

# Fit alpha(T) = a0 + a1/log(T)
sl1, ic1, rv1, pv1, se1 = stats.linregress(inv_log_T, alphas)
print(f"\n  Fit 1: alpha = a0 + a1/log(T)", flush=True)
print(f"    a0 = {ic1:.6f}  (alpha at T->inf)", flush=True)
print(f"    a1 = {sl1:+.4f}", flush=True)
print(f"    R2 = {rv1**2:.6f}", flush=True)

# Fit alpha(T) = a0 + a1/log(T) + a2/log^2(T)
X_2 = np.column_stack([np.ones(n_win), inv_log_T, inv_log_T**2])
beta2, res2, _, _ = np.linalg.lstsq(X_2, alphas, rcond=None)
alphas_pred2 = X_2 @ beta2
R2_2 = 1.0 - np.var(alphas - alphas_pred2) / np.var(alphas)
print(f"\n  Fit 2: alpha = a0 + a1/log(T) + a2/log^2(T)", flush=True)
print(f"    a0 = {beta2[0]:.6f}", flush=True)
print(f"    a1 = {beta2[1]:+.4f}", flush=True)
print(f"    a2 = {beta2[2]:+.4f}", flush=True)
print(f"    R2 = {R2_2:.6f}", flush=True)

# Fit alpha(T) = 1 + a1/log(T) + a2/log^2(T)  [constrained: alpha_inf = 1]
X_c = np.column_stack([inv_log_T, inv_log_T**2])
beta_c, _, _, _ = np.linalg.lstsq(X_c, alphas - 1.0, rcond=None)
alphas_pred_c = 1.0 + X_c @ beta_c
R2_c = 1.0 - np.var(alphas - alphas_pred_c) / np.var(alphas)
print(f"\n  Fit 3: alpha = 1 + a1/log(T) + a2/log^2(T)  [constrained]", flush=True)
print(f"    a1 = {beta_c[0]:+.6f}", flush=True)
print(f"    a2 = {beta_c[1]:+.6f}", flush=True)
print(f"    R2 = {R2_c:.6f}", flush=True)


# =====================================================================
# PART 2: THETA_EFF(T) VIA GPU BISECTION
# =====================================================================
print("\n" + "=" * 70, flush=True)
print("PART 2: THETA_EFF(T) VIA GPU BISECTION", flush=True)
print("=" * 70, flush=True)

def sieve(N):
    is_p = np.ones(N + 1, dtype=bool)
    is_p[:2] = False
    for i in range(2, int(N**0.5) + 1):
        if is_p[i]:
            is_p[i*i::i] = False
    return np.where(is_p)[0]

primes = sieve(5_000)

def prime_sum_gpu(g0_np, tp_np, primes, k_max, theta):
    N = len(g0_np)
    batch_size = max(10, min(669, 500_000_000 // (N * 8 * 5)))
    g0 = torch.tensor(g0_np, dtype=torch.float64, device=DEVICE)
    tp_t = torch.tensor(tp_np, dtype=torch.float64, device=DEVICE)
    log_X = theta * torch.log(torch.clamp(g0, min=2.0))
    S = torch.zeros(N, dtype=torch.float64, device=DEVICE)
    logp = np.log(primes.astype(np.float64))
    mask = logp < 3.0 * float(log_X[-1])
    p_use = primes[mask].astype(np.float64)
    lp_use = logp[mask]
    n_p = len(p_use)
    for m in range(1, k_max + 1):
        pm = p_use ** (m / 2.0)
        mlp = m * lp_use
        for bs in range(0, n_p, batch_size):
            be = min(bs + batch_size, n_p)
            mlp_t = torch.tensor(mlp[bs:be], dtype=torch.float64, device=DEVICE)
            pm_t = torch.tensor(pm[bs:be], dtype=torch.float64, device=DEVICE)
            x = mlp_t.unsqueeze(0) / log_X.unsqueeze(1)
            w = torch.where(x < 1.0, torch.cos(torch.pi * x / 2)**2, torch.zeros_like(x))
            angles = g0.unsqueeze(1) * mlp_t.unsqueeze(0)
            S -= (w * torch.sin(angles) / (m * pm_t.unsqueeze(0))).sum(dim=1)
            del x, w, angles, mlp_t, pm_t
    result = (-S / tp_t).cpu().numpy()
    del g0, tp_t, log_X, S
    torch.cuda.empty_cache()
    return result

# Warmup
_ = prime_sum_gpu(gamma0[:100], tp[:100], primes[:10], 3, 1.0)
torch.cuda.synchronize()

# Bisection in 40 windows of 50k
THETA_WIN = 50_000
N_THETA_WIN = N_ZEROS // THETA_WIN
theta_eff_data = []

print(f"  {N_THETA_WIN} windows of {THETA_WIN//1000}k zeros", flush=True)
t1 = time.time()

for wi in range(N_THETA_WIN):
    a = wi * THETA_WIN
    b = min(a + THETA_WIN, N_ZEROS)
    d_w = delta[a:b]
    g0_w = gamma0[a:b]
    tp_w = tp[a:b]

    lo, hi = 0.7, 1.4
    for _ in range(20):
        mid = (lo + hi) / 2
        dp_w = prime_sum_gpu(g0_w, tp_w, primes, K_MAX, mid)
        dot_pp = np.dot(dp_w, dp_w)
        a_w = np.dot(d_w, dp_w) / dot_pp if dot_pp > 0 else 2.0
        if a_w > 1.0:
            lo = mid
        else:
            hi = mid

    theta_eff = (lo + hi) / 2
    T_mid = float((gamma_n[a] + gamma_n[min(b-1, N_ZEROS-1)]) / 2)
    theta_eff_data.append({'wi': wi, 'T_mid': T_mid, 'theta_eff': theta_eff})

    if (wi + 1) % 10 == 0:
        el = time.time() - t1
        eta = el / (wi + 1) * (N_THETA_WIN - wi - 1)
        print(f"    Window {wi+1}/{N_THETA_WIN}... th_eff={theta_eff:.4f} [{el:.0f}s, ETA {eta:.0f}s]", flush=True)

T_th = np.array([d['T_mid'] for d in theta_eff_data])
theta_effs = np.array([d['theta_eff'] for d in theta_eff_data])
log_T_th = np.log(T_th)
inv_log_T_th = 1.0 / log_T_th

print(f"\n  theta_eff range: [{theta_effs.min():.4f}, {theta_effs.max():.4f}]", flush=True)

# Fit theta_eff = t0 + t1/log(T)
sl_t1, ic_t1, rv_t1, _, _ = stats.linregress(inv_log_T_th, theta_effs)
print(f"\n  Fit 1: theta = t0 + t1/log(T)", flush=True)
print(f"    t0 (theta_inf) = {ic_t1:.6f}", flush=True)
print(f"    t1             = {sl_t1:+.4f}", flush=True)
print(f"    R2             = {rv_t1**2:.6f}", flush=True)

# Fit theta_eff = t0 + t1/log(T) + t2/log^2(T)
X_th2 = np.column_stack([np.ones(len(theta_effs)), inv_log_T_th, inv_log_T_th**2])
beta_th2, _, _, _ = np.linalg.lstsq(X_th2, theta_effs, rcond=None)
theta_pred2 = X_th2 @ beta_th2
R2_th2 = 1.0 - np.var(theta_effs - theta_pred2) / np.var(theta_effs)
print(f"\n  Fit 2: theta = t0 + t1/log(T) + t2/log^2(T)", flush=True)
print(f"    t0 = {beta_th2[0]:.6f}", flush=True)
print(f"    t1 = {beta_th2[1]:+.6f}", flush=True)
print(f"    t2 = {beta_th2[2]:+.6f}", flush=True)
print(f"    R2 = {R2_th2:.6f}", flush=True)


# =====================================================================
# PART 3: GIFT CONSTANT SEARCH
# =====================================================================
print("\n" + "=" * 70, flush=True)
print("PART 3: SYMBOLIC SEARCH IN GIFT CONSTANTS", flush=True)
print("=" * 70, flush=True)

# GIFT structural constants
GIFT = {
    'dim_K7': 7, 'dim_G2': 14, 'rank_E8': 8, 'dim_E8': 248,
    'b2': 21, 'b3': 77, 'H_star': 99, 'p2': 2,
    'N_gen': 3, 'dim_J3O': 27, 'parallel_spinors': 1,
    'physical_gap_num': 13, 'K_MAX': 3,
    'Weyl_factor': 5, 'alpha_sum': 13,
}

# Build candidate rationals p/q from GIFT numbers
gift_nums = sorted(set(GIFT.values()))  # [1, 2, 3, 5, 7, 8, 13, 14, 21, 27, 77, 99, 248]
# Also include derived numbers
derived = [70, 50, 198, 14*7, 99-77, 77-21, 21-14, 99+1, 14-1, 7-1, 7+1]
all_nums = sorted(set(gift_nums + derived + [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 70]))

# Generate candidate values for theta_inf and coefficients
candidates_theta = {}
for p in all_nums:
    for q in all_nums:
        if q == 0:
            continue
        val = p / q
        if 0.5 < val < 3.0:
            label = f"{p}/{q}"
            # Simplify
            from math import gcd
            g = gcd(p, q)
            label = f"{p//g}/{q//g}" if q//g > 1 else f"{p//g}"
            candidates_theta[label] = val

# Add special values
import math
candidates_theta['1'] = 1.0
candidates_theta['sqrt(2)'] = math.sqrt(2)
candidates_theta['99/70'] = 99/70
candidates_theta['pi/2'] = math.pi/2
candidates_theta['7/5'] = 7/5
candidates_theta['3/2'] = 3/2
candidates_theta['e/2'] = math.e/2
candidates_theta['log(4)'] = math.log(4)
candidates_theta['1/log(2)'] = 1/math.log(2)

# Also generate candidate coefficient values
candidates_coeff = {}
for p in all_nums:
    for q in all_nums:
        if q == 0:
            continue
        for sign in [1, -1]:
            val = sign * p / q
            if -30 < val < 30:
                g = gcd(p, q)
                label = f"{'-' if sign < 0 else ''}{p//g}/{q//g}" if q//g > 1 else f"{sign*p//g}"
                candidates_coeff[label] = val

# Search for best (theta_inf, c1) match to alpha data
# Model: alpha(T) = 1 + c1/log(T) where theta drives this
# But more directly, search for theta(T) = theta_inf + c1/log(T)
print("\n  Searching theta_inf + c1/log(T) fits...", flush=True)

# Target: fitted values from data
target_t0 = ic_t1  # asymptote from linear fit
target_t1 = sl_t1  # coefficient of 1/log(T)

# Also get the 2-param fit values
target_t0_2 = beta_th2[0]
target_t1_2 = beta_th2[1]
target_t2_2 = beta_th2[2]

print(f"\n  Fitted targets (1-param): t0={target_t0:.4f}, t1={target_t1:+.4f}", flush=True)
print(f"  Fitted targets (2-param): t0={target_t0_2:.4f}, t1={target_t1_2:+.4f}, t2={target_t2_2:+.4f}", flush=True)

# For alpha, directly:
target_a0 = ic1  # alpha at T->inf
target_a1 = sl1  # coeff of 1/log(T) in alpha
target_a0_2 = beta2[0]
target_a1_2 = beta2[1]
target_a2_2 = beta2[2]
print(f"\n  Alpha fit (1-param): a0={target_a0:.6f}, a1={target_a1:+.4f}", flush=True)
print(f"  Alpha fit (2-param): a0={target_a0_2:.6f}, a1={target_a1_2:+.4f}, a2={target_a2_2:+.4f}", flush=True)

# Approach 1: find rational matches for fitted parameters
print("\n  --- MATCHES FOR theta_inf ---", flush=True)
matches_t0 = []
for label, val in candidates_theta.items():
    err = abs(val - target_t0)
    err2 = abs(val - target_t0_2)
    best_err = min(err, err2)
    if best_err < 0.15:
        matches_t0.append((best_err, label, val))

matches_t0.sort()
for err, label, val in matches_t0[:15]:
    print(f"    {label:>12} = {val:.6f}  (err = {err:+.4f})", flush=True)

print("\n  --- MATCHES FOR alpha_inf ---", flush=True)
matches_a0 = []
for label, val in candidates_theta.items():
    err = abs(val - target_a0)
    err2 = abs(val - target_a0_2)
    best_err = min(err, err2)
    if best_err < 0.05:
        matches_a0.append((best_err, label, val))
matches_a0.sort()
for err, label, val in matches_a0[:15]:
    print(f"    {label:>12} = {val:.6f}  (err = {err:+.4f})", flush=True)

# Approach 2: GRID SEARCH over (theta_inf, c1) from GIFT rationals
# Evaluate fit quality on the theta_eff data
print("\n  --- GRID SEARCH: theta(T) = theta_inf + c/log(T) ---", flush=True)

best_fits = []
for label_t0, val_t0 in candidates_theta.items():
    for label_c1, val_c1 in candidates_coeff.items():
        theta_model = val_t0 + val_c1 * inv_log_T_th
        ss_res = np.sum((theta_effs - theta_model)**2)
        ss_tot = np.sum((theta_effs - np.mean(theta_effs))**2)
        r2 = 1.0 - ss_res / ss_tot
        if r2 > 0.95:
            # Complexity score: prefer simpler expressions
            complexity = len(label_t0) + len(label_c1)
            best_fits.append((r2, complexity, label_t0, val_t0, label_c1, val_c1))

best_fits.sort(key=lambda x: (-x[0], x[1]))
print(f"  Found {len(best_fits)} fits with R2 > 0.95", flush=True)
print(f"\n  {'R2':>8} | {'theta_inf':>15} | {'c':>15} | formula", flush=True)
print(f"  " + "-" * 70, flush=True)
for r2, cx, lt0, vt0, lc1, vc1 in best_fits[:20]:
    print(f"  {r2:.6f} | {lt0:>15} = {vt0:.4f} | {lc1:>15} = {vc1:.4f} | "
          f"th = {lt0} + ({lc1})/log(T)", flush=True)

# Approach 3: 2-param GIFT search: theta(T) = theta_inf + c1/log(T) + c2/log^2(T)
print("\n  --- 2-PARAM GRID: theta = t0 + c1/log + c2/log^2 ---", flush=True)

# For each theta_inf candidate, optimize c1 and c2
best_2param = []
for label_t0, val_t0 in candidates_theta.items():
    resid = theta_effs - val_t0
    # Fit c1, c2 by least squares
    X_fit = np.column_stack([inv_log_T_th, inv_log_T_th**2])
    beta_fit, _, _, _ = np.linalg.lstsq(X_fit, resid, rcond=None)
    pred = val_t0 + X_fit @ beta_fit
    ss_res = np.sum((theta_effs - pred)**2)
    ss_tot = np.sum((theta_effs - np.mean(theta_effs))**2)
    r2 = 1.0 - ss_res / ss_tot
    if r2 > 0.98:
        best_2param.append((r2, label_t0, val_t0, beta_fit[0], beta_fit[1]))

best_2param.sort(key=lambda x: -x[0])
print(f"  {'R2':>8} | {'theta_inf':>12} | {'c1':>10} | {'c2':>10}", flush=True)
print(f"  " + "-" * 55, flush=True)
for r2, lt0, vt0, c1, c2 in best_2param[:15]:
    print(f"  {r2:.6f} | {lt0:>12} = {vt0:.4f} | {c1:>+10.4f} | {c2:>+10.4f}", flush=True)

# Also check for ALPHA fit: alpha(T) = alpha_inf + a1/log(T)
print("\n  --- GRID SEARCH: alpha(T) = a0 + a1/log(T) ---", flush=True)

best_alpha_fits = []
for label_a0, val_a0 in candidates_theta.items():
    for label_a1, val_a1 in candidates_coeff.items():
        alpha_model = val_a0 + val_a1 * inv_log_T
        ss_res = np.sum((alphas - alpha_model)**2)
        ss_tot = np.sum((alphas - np.mean(alphas))**2)
        r2 = 1.0 - ss_res / ss_tot
        if r2 > 0.95:
            complexity = len(label_a0) + len(label_a1)
            best_alpha_fits.append((r2, complexity, label_a0, val_a0, label_a1, val_a1))

best_alpha_fits.sort(key=lambda x: (-x[0], x[1]))
print(f"  Found {len(best_alpha_fits)} fits with R2 > 0.95", flush=True)
for r2, cx, la0, va0, la1, va1 in best_alpha_fits[:15]:
    print(f"  {r2:.6f} | a0={la0:>8}={va0:.4f} | a1={la1:>8}={va1:.4f} | "
          f"alpha = {la0} + ({la1})/log(T)", flush=True)


# =====================================================================
# PART 4: CONTINUED FRACTION ANALYSIS
# =====================================================================
print("\n" + "=" * 70, flush=True)
print("PART 4: CONTINUED FRACTION DECOMPOSITION", flush=True)
print("=" * 70, flush=True)

def continued_fraction(x, n_terms=8):
    """Compute continued fraction representation."""
    cf = []
    for _ in range(n_terms):
        a = int(np.floor(x))
        cf.append(a)
        x = x - a
        if abs(x) < 1e-10:
            break
        x = 1.0 / x
    return cf

def convergents(cf):
    """Compute convergents from continued fraction."""
    convs = []
    h_prev, k_prev = 1, 0
    h_curr, k_curr = cf[0], 1
    convs.append((h_curr, k_curr))
    for i in range(1, len(cf)):
        h_next = cf[i] * h_curr + h_prev
        k_next = cf[i] * k_curr + k_prev
        convs.append((h_next, k_next))
        h_prev, k_prev = h_curr, k_curr
        h_curr, k_curr = h_next, k_next
    return convs

for name, val in [
    ("alpha_inf (1-param)", target_a0),
    ("alpha_inf (2-param)", target_a0_2),
    ("theta_inf (1-param)", target_t0),
    ("theta_inf (2-param)", target_t0_2),
    ("a1 (alpha coeff)", target_a1),
    ("a1 (alpha 2-param)", target_a1_2),
    ("a2 (alpha 2-param)", target_a2_2),
    ("t1 (theta coeff)", target_t1),
    ("t1 (theta 2-param)", target_t1_2),
    ("t2 (theta 2-param)", target_t2_2),
]:
    cf = continued_fraction(abs(val))
    convs = convergents(cf)
    sign = "-" if val < 0 else "+"
    print(f"\n  {name} = {val:.6f}", flush=True)
    print(f"    CF: [{', '.join(str(c) for c in cf)}]", flush=True)
    print(f"    Convergents: ", end="", flush=True)
    for h, k in convs[:6]:
        approx = h / k if k > 0 else 0
        print(f" {sign}{h}/{k}={sign}{approx:.4f}", end="", flush=True)
    print(flush=True)


# =====================================================================
# PART 5: PHYSICALLY MOTIVATED MODELS
# =====================================================================
print("\n" + "=" * 70, flush=True)
print("PART 5: PHYSICALLY MOTIVATED MODELS", flush=True)
print("=" * 70, flush=True)

# Model A: theta(T) = 1 + correction
# Physical: the "natural" cutoff is X = T (theta=1)
# Correction from prime density: 1/log(T)
print("\n  Model A: theta_inf = 1 (natural cutoff)", flush=True)
resid_A = theta_effs - 1.0
X_A = np.column_stack([inv_log_T_th, inv_log_T_th**2])
beta_A, _, _, _ = np.linalg.lstsq(X_A, resid_A, rcond=None)
pred_A = 1.0 + X_A @ beta_A
R2_A = 1.0 - np.var(theta_effs - pred_A) / np.var(theta_effs)
print(f"    theta(T) = 1 + ({beta_A[0]:+.4f})/log(T) + ({beta_A[1]:+.4f})/log^2(T)", flush=True)
print(f"    R2 = {R2_A:.6f}", flush=True)

# Model B: theta(T) = 99/70 + correction
# Physical: Pell equation, 99/70 ~ sqrt(2)
print(f"\n  Model B: theta_inf = 99/70 = {99/70:.6f} (Pell)", flush=True)
resid_B = theta_effs - 99/70
X_B = np.column_stack([inv_log_T_th, inv_log_T_th**2])
beta_B, _, _, _ = np.linalg.lstsq(X_B, resid_B, rcond=None)
pred_B = 99/70 + X_B @ beta_B
R2_B = 1.0 - np.var(theta_effs - pred_B) / np.var(theta_effs)
print(f"    theta(T) = 99/70 + ({beta_B[0]:+.4f})/log(T) + ({beta_B[1]:+.4f})/log^2(T)", flush=True)
print(f"    R2 = {R2_B:.6f}", flush=True)

# Model C: theta(T) = 3/2 + correction
# Physical: Weyl exponent
print(f"\n  Model C: theta_inf = 3/2 (Weyl)", flush=True)
resid_C = theta_effs - 1.5
X_C = np.column_stack([inv_log_T_th, inv_log_T_th**2])
beta_C, _, _, _ = np.linalg.lstsq(X_C, resid_C, rcond=None)
pred_C = 1.5 + X_C @ beta_C
R2_C = 1.0 - np.var(theta_effs - pred_C) / np.var(theta_effs)
print(f"    theta(T) = 3/2 + ({beta_C[0]:+.4f})/log(T) + ({beta_C[1]:+.4f})/log^2(T)", flush=True)
print(f"    R2 = {R2_C:.6f}", flush=True)

# Model D: theta(T) = 14/13 + correction
# Physical: dim(G2)/physical_gap_num
print(f"\n  Model D: theta_inf = 14/13 = {14/13:.6f}", flush=True)
resid_D = theta_effs - 14/13
X_D = np.column_stack([inv_log_T_th, inv_log_T_th**2])
beta_D, _, _, _ = np.linalg.lstsq(X_D, resid_D, rcond=None)
pred_D = 14/13 + X_D @ beta_D
R2_D = 1.0 - np.var(theta_effs - pred_D) / np.var(theta_effs)
print(f"    theta(T) = 14/13 + ({beta_D[0]:+.4f})/log(T) + ({beta_D[1]:+.4f})/log^2(T)", flush=True)
print(f"    R2 = {R2_D:.6f}", flush=True)

# Model E: alpha_inf = 1 exactly (constrained)
# Then find what theta correction gives alpha = 1 at all T
print(f"\n  Model E: alpha_inf = 1 exactly", flush=True)
# Fit alpha(T) = 1 + c1/log(T) + c2/log^2(T)
X_E = np.column_stack([inv_log_T, inv_log_T**2])
beta_E, _, _, _ = np.linalg.lstsq(X_E, alphas - 1.0, rcond=None)
pred_E = 1.0 + X_E @ beta_E
R2_E = 1.0 - np.var(alphas - pred_E) / np.var(alphas)
print(f"    alpha(T) = 1 + ({beta_E[0]:+.6f})/log(T) + ({beta_E[1]:+.6f})/log^2(T)", flush=True)
print(f"    R2 = {R2_E:.6f}", flush=True)
print(f"    c1 = {beta_E[0]:+.6f}", flush=True)
print(f"    c2 = {beta_E[1]:+.6f}", flush=True)

# Search for c1, c2 in GIFT constants
print(f"\n  Searching c1={beta_E[0]:.4f} in GIFT rationals:", flush=True)
for label, val in sorted(candidates_coeff.items(), key=lambda x: abs(x[1] - beta_E[0])):
    if abs(val - beta_E[0]) < 0.15:
        print(f"    {label:>12} = {val:.6f}  (err={val - beta_E[0]:+.4f})", flush=True)

print(f"\n  Searching c2={beta_E[1]:.4f} in GIFT rationals:", flush=True)
for label, val in sorted(candidates_coeff.items(), key=lambda x: abs(x[1] - beta_E[1])):
    if abs(val - beta_E[1]) < 2.0:
        print(f"    {label:>12} = {val:.6f}  (err={val - beta_E[1]:+.4f})", flush=True)


# =====================================================================
# FIGURE
# =====================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (0,0) Alpha vs T
ax = axes[0, 0]
ax.plot(np.log10(T_mids), alphas, 'b.', ms=3, alpha=0.5, label='Data')
log10_ext = np.linspace(1, 7, 100)
inv_log_ext = 1.0 / (log10_ext * np.log(10))
ax.plot(log10_ext, ic1 + sl1 * inv_log_ext, 'r-', lw=2,
        label=f'Fit: {ic1:.4f} + {sl1:.1f}/log(T)')
ax.axhline(1.0, color='gray', ls=':', alpha=0.5)
ax.set_xlabel('log10(T)'); ax.set_ylabel('alpha')
ax.set_title('alpha(T) sliding windows'); ax.legend()

# (0,1) Theta_eff vs T
ax = axes[0, 1]
ax.plot(np.log10(T_th), theta_effs, 'bo', ms=4, label='theta_eff (bisection)')
pred_line = ic_t1 + sl_t1 * inv_log_T_th
ax.plot(np.log10(T_th), pred_line, 'r-', lw=2,
        label=f'Fit: {ic_t1:.3f} + {sl_t1:.1f}/log(T)')
ax.axhline(THETA_STAR, color='green', ls='--', alpha=0.7, label=f'theta*={THETA_STAR}')
ax.axhline(99/70, color='purple', ls=':', alpha=0.7, label=f'99/70={99/70:.4f}')
ax.axhline(1.0, color='gray', ls=':', alpha=0.3)
ax.set_xlabel('log10(T)'); ax.set_ylabel('theta_eff')
ax.set_title('theta_eff per window'); ax.legend(fontsize=8)

# (1,0) Alpha residuals
ax = axes[1, 0]
alpha_resid = alphas - (ic1 + sl1 * inv_log_T)
ax.plot(np.log10(T_mids), alpha_resid, 'b.', ms=3)
ax.axhline(0, color='r', ls='-', alpha=0.3)
ax.set_xlabel('log10(T)'); ax.set_ylabel('alpha - fit')
ax.set_title('Alpha fit residuals')

# (1,1) R2 vs T
ax = axes[1, 1]
ax.plot(np.log10(T_mids), R2s, 'g.', ms=3)
ax.set_xlabel('log10(T)'); ax.set_ylabel('R2')
ax.set_title('R2 per window')

plt.tight_layout()
figpath = f'{OUTDIR}/fig7_correction_hunt.png'
plt.savefig(figpath, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved {figpath}", flush=True)

# =====================================================================
# SUMMARY
# =====================================================================
total = time.time() - t0
print("\n" + "=" * 70, flush=True)
print("CORRECTION HUNT SUMMARY", flush=True)
print("=" * 70, flush=True)

print(f"\n  ALPHA curve (80 windows, full prime set via precomputed):", flush=True)
print(f"    alpha(T) = {ic1:.6f} + ({sl1:+.4f})/log(T)   R2={rv1**2:.4f}", flush=True)
print(f"    alpha(T) = {beta2[0]:.6f} + ({beta2[1]:+.4f})/log + ({beta2[2]:+.4f})/log^2   R2={R2_2:.4f}", flush=True)
print(f"\n  THETA curve (40 windows, primes<=5k):", flush=True)
print(f"    theta(T) = {ic_t1:.4f} + ({sl_t1:+.4f})/log(T)   R2={rv_t1**2:.4f}", flush=True)
print(f"    theta(T) = {beta_th2[0]:.4f} + ({beta_th2[1]:+.4f})/log + ({beta_th2[2]:+.4f})/log^2   R2={R2_th2:.4f}", flush=True)
print(f"\n  Best physical models:", flush=True)
print(f"    Model A (theta_inf=1):     R2={R2_A:.4f}  c1={beta_A[0]:+.4f} c2={beta_A[1]:+.4f}", flush=True)
print(f"    Model B (theta_inf=99/70): R2={R2_B:.4f}  c1={beta_B[0]:+.4f} c2={beta_B[1]:+.4f}", flush=True)
print(f"    Model C (theta_inf=3/2):   R2={R2_C:.4f}  c1={beta_C[0]:+.4f} c2={beta_C[1]:+.4f}", flush=True)
print(f"    Model D (theta_inf=14/13): R2={R2_D:.4f}  c1={beta_D[0]:+.4f} c2={beta_D[1]:+.4f}", flush=True)
print(f"\n  Total: {total:.0f}s", flush=True)

# Save results
out = {
    'alpha_fit_1param': {'a0': float(ic1), 'a1': float(sl1), 'R2': float(rv1**2)},
    'alpha_fit_2param': {'a0': float(beta2[0]), 'a1': float(beta2[1]),
                         'a2': float(beta2[2]), 'R2': float(R2_2)},
    'theta_fit_1param': {'t0': float(ic_t1), 't1': float(sl_t1), 'R2': float(rv_t1**2)},
    'theta_fit_2param': {'t0': float(beta_th2[0]), 't1': float(beta_th2[1]),
                         't2': float(beta_th2[2]), 'R2': float(R2_th2)},
    'models': {
        'A_theta1': {'R2': float(R2_A), 'c1': float(beta_A[0]), 'c2': float(beta_A[1])},
        'B_99_70': {'R2': float(R2_B), 'c1': float(beta_B[0]), 'c2': float(beta_B[1])},
        'C_3_2': {'R2': float(R2_C), 'c1': float(beta_C[0]), 'c2': float(beta_C[1])},
        'D_14_13': {'R2': float(R2_D), 'c1': float(beta_D[0]), 'c2': float(beta_D[1])},
    },
    'alpha_constrained': {'c1': float(beta_E[0]), 'c2': float(beta_E[1]), 'R2': float(R2_E)},
    'theta_eff_data': [{'T': float(d['T_mid']), 'theta': float(d['theta_eff'])}
                       for d in theta_eff_data],
    'alpha_data': [{'T': float(d['T_mid']), 'alpha': float(d['alpha']), 'R2': float(d['R2'])}
                   for d in alpha_data]
}
jpath = f'{OUTDIR}/correction_hunt_results.json'
with open(jpath, 'w') as f:
    json.dump(out, f, indent=2)
print(f"  Saved {jpath}", flush=True)
