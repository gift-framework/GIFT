#!/usr/bin/env python3
"""
Standalone analysis of 2M zeros results.
GPU-accelerated prime sum via PyTorch CUDA (RTX 2050).
"""
import sys
import numpy as np
import json
import time
from scipy import stats
from scipy.special import loggamma, lambertw
from scipy.stats import kstest, anderson, norm
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

with open(f'{OUTDIR}/prime_spectral_2M_results.json') as f:
    results_json = json.load(f)

N_ZEROS = len(gamma_n)
THETA_STAR = 0.9941
K_MAX = 3
window_results = results_json['windows']
print(f"  N = {N_ZEROS:,}, T_max = {gamma_n[-1]:.0f}", flush=True)

# =====================================================================
# SMOOTH ZEROS
# =====================================================================
print("Computing smooth zeros...", flush=True)
ns = np.arange(1, N_ZEROS + 1, dtype=np.float64)
targets = (ns - 1.5) * np.pi
w_lw = np.real(lambertw(ns / np.e))
gamma0 = np.maximum(2 * np.pi * ns / w_lw, 2.0)
for it in range(40):
    theta_val = np.imag(loggamma(0.25 + 0.5j * gamma0)) - 0.5 * gamma0 * np.log(np.pi)
    tp_val = 0.5 * np.log(np.maximum(gamma0, 1.0) / (2 * np.pi))
    dt = (theta_val - targets) / np.maximum(np.abs(tp_val), 1e-15)
    gamma0 -= dt
    if np.max(np.abs(dt)) < 1e-10:
        break

delta = gamma_n - gamma0
tp = 0.5 * np.log(np.maximum(gamma0, 1.0) / (2 * np.pi))
residuals = delta - delta_pred
alpha_OLS = float(np.dot(delta, delta_pred) / np.dot(delta_pred, delta_pred))
R2_global = float(1.0 - np.var(residuals) / np.var(delta))
print(f"  alpha_OLS={alpha_OLS:.6f}, R2={R2_global:.6f} [{time.time()-t0:.1f}s]", flush=True)

# =====================================================================
# PRIME SUM (GPU with batching)
# =====================================================================
def sieve(N):
    is_p = np.ones(N + 1, dtype=bool)
    is_p[:2] = False
    for i in range(2, int(N**0.5) + 1):
        if is_p[i]:
            is_p[i*i::i] = False
    return np.where(is_p)[0]

primes_5k = sieve(5_000)
print(f"  {len(primes_5k)} primes <= 5k", flush=True)

def prime_sum_gpu(g0_np, tp_np, primes, k_max, theta, batch_size=None):
    N = len(g0_np)
    # Auto batch_size: keep (N, B) tensors under ~500 MB
    if batch_size is None:
        batch_size = max(10, min(669, 500_000_000 // (N * 8 * 5)))

    g0 = torch.tensor(g0_np, dtype=torch.float64, device=DEVICE)
    tp_t = torch.tensor(tp_np, dtype=torch.float64, device=DEVICE)
    log_X = theta * torch.log(torch.clamp(g0, min=2.0))
    S = torch.zeros(N, dtype=torch.float64, device=DEVICE)

    logp = np.log(primes.astype(np.float64))
    max_lx = float(log_X[-1])
    mask = logp < 3.0 * max_lx
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

def prime_sum_cpu(g0, tp_v, primes, k_max, theta):
    S = np.zeros_like(g0)
    log_X = theta * np.log(np.maximum(g0, 2.0))
    for p in primes:
        logp = np.log(float(p))
        if logp / log_X[-1] > 3.0:
            break
        for m in range(1, k_max + 1):
            x = m * logp / log_X
            w = np.where(x < 1.0, np.cos(np.pi * x / 2)**2, 0.0)
            if np.max(w) < 1e-15:
                continue
            S -= w * np.sin(g0 * m * logp) / (m * p**(m / 2.0))
    return -S / tp_v

prime_sum = prime_sum_gpu if DEVICE.type == 'cuda' else prime_sum_cpu

# GPU warmup
if DEVICE.type == 'cuda':
    _ = prime_sum(gamma0[:100], tp[:100], primes_5k[:10], 3, 1.0)
    torch.cuda.synchronize()
    print("  GPU warm", flush=True)


# =====================================================================
# T7: Bootstrap CI for alpha (FAST - precomputed data)
# =====================================================================
print("\n" + "=" * 70, flush=True)
print("VALIDATION TESTS -- CONSTANT MODEL (theta* = 0.9941)", flush=True)
print("=" * 70, flush=True)

print("\n[T7] Bootstrap CI for alpha", flush=True)
print("-" * 50, flush=True)
B = 5000
np.random.seed(123)
alpha_boots = np.empty(B)
for b in range(B):
    idx = np.random.randint(0, N_ZEROS, N_ZEROS)
    d_b, dp_b = delta[idx], delta_pred[idx]
    dot_pp = np.dot(dp_b, dp_b)
    alpha_boots[b] = np.dot(d_b, dp_b) / dot_pp if dot_pp > 0 else 0.0

ci_lo = float(np.percentile(alpha_boots, 2.5))
ci_hi = float(np.percentile(alpha_boots, 97.5))
T7_pass = ci_lo <= 1.0 <= ci_hi
print(f"  alpha(OLS):    {alpha_OLS:.6f}", flush=True)
print(f"  95% CI:        [{ci_lo:.6f}, {ci_hi:.6f}]", flush=True)
print(f"  Contains 1.0?  {'YES' if T7_pass else 'NO'}", flush=True)
print(f"  >> {'PASS' if T7_pass else 'FAIL'}", flush=True)

# =====================================================================
# T8: Drift test (FAST - from JSON)
# =====================================================================
print(f"\n[T8] Drift Test (alpha across windows)", flush=True)
print("-" * 50, flush=True)
alphas_w = np.array([w['alpha'] for w in window_results])
slope, intercept, r_val, p_val_drift, se = stats.linregress(
    np.arange(len(alphas_w), dtype=float), alphas_w)
T8_pass = p_val_drift > 0.05
print(f"  alphas:        {[round(a, 4) for a in alphas_w]}", flush=True)
print(f"  Slope:         {slope:+.6f}/window", flush=True)
print(f"  p-value:       {p_val_drift:.4f}", flush=True)
print(f"  >> {'PASS' if T8_pass else 'FAIL'}", flush=True)

# =====================================================================
# T5: Monte Carlo (GPU accelerated)
# =====================================================================
print(f"\n[T5] Monte Carlo Permutation Test (GPU)", flush=True)
print("-" * 50, flush=True)
N_TRIALS = 50
N_MC = 100_000
np.random.seed(42)

d_mc = delta[:N_MC]
g0_mc, tp_mc = gamma0[:N_MC], tp[:N_MC]

# Fair comparison: compute R2_opt with SAME prime set
t1 = time.time()
dp_opt_mc = prime_sum(g0_mc, tp_mc, primes_5k, K_MAX, THETA_STAR)
R2_opt = float(1.0 - np.var(d_mc - dp_opt_mc) / np.var(d_mc))
print(f"  R2(theta*):    {R2_opt:.6f}  [{time.time()-t1:.1f}s]", flush=True)

theta_random = np.random.uniform(0.3, 2.0, N_TRIALS)
R2_random = []
t1 = time.time()
for i, th in enumerate(theta_random):
    dp_r = prime_sum(g0_mc, tp_mc, primes_5k, K_MAX, float(th))
    R2_random.append(float(1.0 - np.var(d_mc - dp_r) / np.var(d_mc)))
    if (i + 1) % 10 == 0:
        el = time.time() - t1
        eta = el / (i + 1) * (N_TRIALS - i - 1)
        print(f"    {i+1}/{N_TRIALS}... [{el:.0f}s, ETA {eta:.0f}s]", flush=True)

R2_random = np.array(R2_random)
R2_best_random = float(np.max(R2_random))
margin = R2_opt - R2_best_random
p_val_mc = float(np.mean(R2_random >= R2_opt))
T5_pass = margin > 0

print(f"\n  R2(theta*):    {R2_opt:.6f}", flush=True)
print(f"  R2(best rnd):  {R2_best_random:.6f}", flush=True)
print(f"  R2(mean rnd):  {np.mean(R2_random):.6f} +/- {np.std(R2_random):.6f}", flush=True)
print(f"  Margin:        {margin:+.6f}", flush=True)
print(f"  p-value:       {p_val_mc:.4f}", flush=True)
print(f"  >> {'PASS' if T5_pass else 'FAIL'}", flush=True)

# Fine scan around theta*
print("\n  [T5b] Fine scan around theta*:", flush=True)
theta_fine = np.linspace(0.90, 1.10, 21)
R2_fine = []
for th in theta_fine:
    dp_f = prime_sum(g0_mc, tp_mc, primes_5k, K_MAX, float(th))
    R2_fine.append(float(1.0 - np.var(d_mc - dp_f) / np.var(d_mc)))
R2_fine = np.array(R2_fine)
best_idx = np.argmax(R2_fine)
print(f"    Best in scan:  theta={theta_fine[best_idx]:.4f}, R2={R2_fine[best_idx]:.6f}", flush=True)
print(f"    theta*=0.9941: R2={R2_opt:.6f}", flush=True)

n_pass = sum([T5_pass, T7_pass, T8_pass])
print(f"\n{'=' * 70}", flush=True)
print(f"CONSTANT MODEL: {n_pass}/3 passed  (vs 0/3 for adaptive)", flush=True)
print(f"{'=' * 70}", flush=True)


# =====================================================================
# Cell B: EFFECTIVE THETA PER WINDOW
# =====================================================================
print("\n" + "=" * 70, flush=True)
print("CELL B: EFFECTIVE theta PER WINDOW (bisection to alpha=1)", flush=True)
print("=" * 70, flush=True)

WINDOWS_EFF = [
    (0, 100_000),
    (100_000, 200_000),
    (200_000, 500_000),
    (500_000, 1_000_000),
    (1_000_000, 1_500_000),
    (1_500_000, N_ZEROS),
]

theta_effs = []
print(f"\n{'Window':>20} | {'T range':>22} | {'th_eff':>8} | {'diff':>8}", flush=True)
print("-" * 70, flush=True)

t1 = time.time()
for wi, (a, b) in enumerate(WINDOWS_EFF):
    d_w, g0_w, tp_w = delta[a:b], gamma0[a:b], tp[a:b]
    # Use batch_size adapted to window size
    bsz = min(300, max(50, 200_000 // max(b - a, 1) * 300))

    lo, hi = 0.5, 1.5
    for _ in range(20):
        mid = (lo + hi) / 2
        dp_w = prime_sum(g0_w, tp_w, primes_5k, K_MAX, mid)
        dot_pp = np.dot(dp_w, dp_w)
        a_w = np.dot(d_w, dp_w) / dot_pp if dot_pp > 0 else 2.0
        if a_w > 1.0:
            lo = mid
        else:
            hi = mid
    theta_eff = (lo + hi) / 2
    theta_effs.append(theta_eff)

    label = f"[{a//1000}k, {b//1000}k)"
    T_lo = gamma_n[a]
    T_hi = gamma_n[min(b - 1, N_ZEROS - 1)]
    diff = theta_eff - THETA_STAR
    print(f"{label:>20} | [{T_lo:>8.0f}, {T_hi:>8.0f}]"
          f" | {theta_eff:>8.4f} | {diff:>+8.4f}  [{time.time()-t1:.0f}s]", flush=True)

theta_effs = np.array(theta_effs)
print(f"\n  theta_eff mean:  {np.mean(theta_effs):.4f}", flush=True)
print(f"  theta_eff std:   {np.std(theta_effs):.4f}", flush=True)
print(f"  theta* global:   {THETA_STAR}", flush=True)
print(f"  Max |diff|:      {np.max(np.abs(theta_effs - THETA_STAR)):.4f}", flush=True)

T_mids_eff = np.array([
    (gamma_n[a] + gamma_n[min(b - 1, N_ZEROS - 1)]) / 2
    for (a, b) in WINDOWS_EFF
])
log_T_mids = np.log(T_mids_eff)
sl_th, ic_th, rv_th, pv_th, se_th = stats.linregress(1.0 / log_T_mids, theta_effs)
print(f"\n  Fit theta_eff ~ a + b/log(T):", flush=True)
print(f"    a (asymptote):  {ic_th:.4f}", flush=True)
print(f"    b (correction): {sl_th:+.4f}", flush=True)
print(f"    R2 of fit:      {rv_th**2:.4f}", flush=True)
print(f"    theta(T->inf):  {ic_th:.4f}", flush=True)
print(f"    99/70 = {99/70:.6f}", flush=True)


# =====================================================================
# Cell C: R2 DECAY ANALYSIS (precomputed)
# =====================================================================
print("\n" + "=" * 70, flush=True)
print("CELL C: R2 DECAY ANALYSIS", flush=True)
print("=" * 70, flush=True)

R2_windows = np.array([w['R2'] for w in window_results])
T_mids_w = np.array([(w['T_lo'] + w['T_hi']) / 2 for w in window_results])
log_T_w = np.log10(T_mids_w)

sl_R2, ic_R2, rv_R2, pv_R2, se_R2 = stats.linregress(log_T_w, R2_windows)
sl_inv, ic_inv, rv_inv, pv_inv, _ = stats.linregress(
    1.0 / np.log(T_mids_w), R2_windows)

print(f"\n  Model 1: R2 = a + b*log10(T)", flush=True)
print(f"    a={ic_R2:.4f}, b={sl_R2:+.6f}/decade, R2_fit={rv_R2**2:.4f}", flush=True)
print(f"    R2(T=10^7)={ic_R2 + sl_R2*7:.4f}, R2(T=10^8)={ic_R2 + sl_R2*8:.4f}, R2(T=10^10)={ic_R2 + sl_R2*10:.4f}", flush=True)

print(f"\n  Model 2: R2 = a + b/log(T)", flush=True)
print(f"    a(plateau)={ic_inv:.4f}, b={sl_inv:+.4f}, R2_fit={rv_inv**2:.4f}", flush=True)
print(f"    R2(T=10^7)={ic_inv + sl_inv/np.log(1e7):.4f}, R2(T=10^8)={ic_inv + sl_inv/np.log(1e8):.4f}, R2(T->inf)={ic_inv:.4f}", flush=True)

# Figure 5
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
log_T_ext = np.linspace(1, 10, 100)
ax1.plot(log_T_w, R2_windows, 'bo-', ms=8, label='Data')
ax1.plot(log_T_ext, ic_R2 + sl_R2 * log_T_ext, 'r--', alpha=0.7,
         label=f'Linear: {sl_R2:+.4f}/decade')
ax1.plot(log_T_ext, ic_inv + sl_inv / (log_T_ext * np.log(10)),
         'g-.', alpha=0.7, label=f'1/log(T): plateau={ic_inv:.3f}')
ax1.axhline(0.9, color='gray', ls=':', alpha=0.5, label='R2=0.90')
ax1.set_xlabel('log10(T)'); ax1.set_ylabel('R2')
ax1.set_title('R2 Decay & Extrapolation'); ax1.legend(fontsize=9)
ax1.set_ylim(0.85, 0.95)

alphas_plot = np.array([w['alpha'] for w in window_results])
ax2.plot(log_T_w, alphas_plot, 'bs-', ms=8, label='Constant theta*')
ax2.axhline(1.0, color='r', ls='--', alpha=0.7, label='alpha = 1')
ax2.fill_between([log_T_w[0]-0.2, log_T_w[-1]+0.2], 0.995, 1.005,
    color='green', alpha=0.15, label='+/-0.5%')
ax2.set_xlabel('log10(T)'); ax2.set_ylabel('alpha (OLS)')
ax2.set_title('Scaling Exponent Stability'); ax2.legend(fontsize=9)
ax2.set_ylim(0.98, 1.02)
plt.tight_layout()
fig5 = f'{OUTDIR}/fig5_R2_extrapolation.png'
plt.savefig(fig5, dpi=150, bbox_inches='tight'); plt.close()
print(f"\nSaved {fig5}", flush=True)


# =====================================================================
# Cell D: GUE COMPARISON
# =====================================================================
print("\n" + "=" * 70, flush=True)
print("CELL D: GUE COMPARISON", flush=True)
print("=" * 70, flush=True)

d_c = delta - np.mean(delta); var_d = np.var(d_c)
r_c = residuals - np.mean(residuals); var_r = np.var(r_c)
dp_c = delta_pred - np.mean(delta_pred); var_dp = np.var(dp_c)

acf1_d = float(np.mean(d_c[1:] * d_c[:-1]) / var_d)
acf1_r = float(np.mean(r_c[1:] * r_c[:-1]) / var_r)
acf1_p = float(np.mean(dp_c[1:] * dp_c[:-1]) / var_dp)

print(f"\n  Lag-1 autocorrelation:", flush=True)
print(f"    delta:      {acf1_d:+.4f}", flush=True)
print(f"    delta_pred: {acf1_p:+.4f}", flush=True)
print(f"    residual:   {acf1_r:+.4f}", flush=True)
print(f"    GUE pred:   ~-0.47", flush=True)

vd, vp, vr = float(np.var(delta)), float(np.var(delta_pred)), float(np.var(residuals))
print(f"\n  Variance decomposition:", flush=True)
print(f"    Var(delta):   {vd:.6f}", flush=True)
print(f"    Var(pred):    {vp:.6f}  ({100*vp/vd:.1f}%)", flush=True)
print(f"    Var(resid):   {vr:.6f}  ({100*vr/vd:.1f}%)", flush=True)
print(f"    R2 check:     {1.0-vr/vd:.4f}", flush=True)

# Spacing distribution
spacings = np.diff(gamma_n)
s = spacings / np.mean(spacings)
s_grid = np.linspace(0, 4, 1000)
P_wig = (32/np.pi**2) * s_grid**2 * np.exp(-4*s_grid**2/np.pi)
hist_v, hist_e = np.histogram(s, bins=200, range=(0,4), density=True)
hist_c = (hist_e[:-1] + hist_e[1:]) / 2

# Multi-lag ACF
lags = [1, 2, 3, 5, 8, 13, 21]
acf_d = [float(np.mean(d_c[l:]*d_c[:-l])/var_d) for l in lags]
acf_r = [float(np.mean(r_c[l:]*r_c[:-l])/var_r) for l in lags]
acf_p = [float(np.mean(dp_c[l:]*dp_c[:-l])/var_dp) for l in lags]

print(f"\n  {'Lag':>5} | {'delta':>8} | {'pred':>8} | {'resid':>8}", flush=True)
print(f"  " + "-" * 40, flush=True)
for i, l in enumerate(lags):
    print(f"  {l:>5} | {acf_d[i]:>+8.4f} | {acf_p[i]:>+8.4f} | {acf_r[i]:>+8.4f}", flush=True)

# KS test
def wigner_cdf(sv):
    return 1.0 - np.exp(-4*sv**2/np.pi)
ks_stat, ks_pval = kstest(s, wigner_cdf)
print(f"\n  KS test vs GUE Wigner:", flush=True)
print(f"    KS={ks_stat:.6f}, p={ks_pval:.4e}", flush=True)
print(f"    >> {'Consistent' if ks_pval > 0.01 else 'Deviates'}", flush=True)

# Anderson-Darling on residuals
ad = anderson(residuals[::100])
print(f"\n  Anderson-Darling (residuals, 1/100 subsample):", flush=True)
print(f"    stat={ad.statistic:.4f}, crit(5%)={ad.critical_values[2]:.4f}", flush=True)
print(f"    >> {'Normal' if ad.statistic < ad.critical_values[2] else 'Non-normal'}", flush=True)

# Figure 6: 2x2 panel
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0,0]
ax.bar(hist_c, hist_v, width=hist_c[1]-hist_c[0], alpha=0.5, color='steelblue',
       label='Empirical (2M)')
ax.plot(s_grid, P_wig, 'r-', lw=2, label='GUE Wigner')
ax.set_xlabel('Normalized spacing s'); ax.set_ylabel('Density')
ax.set_title('Spacing Distribution'); ax.legend(); ax.set_xlim(0, 3.5)

ax = axes[0,1]
x_pos = np.arange(len(lags)); wb = 0.25
ax.bar(x_pos-wb, acf_d, wb, label='delta', color='steelblue', alpha=0.8)
ax.bar(x_pos, acf_p, wb, label='pred', color='coral', alpha=0.8)
ax.bar(x_pos+wb, acf_r, wb, label='resid', color='seagreen', alpha=0.8)
ax.set_xticks(x_pos); ax.set_xticklabels([str(l) for l in lags])
ax.set_xlabel('Lag'); ax.set_ylabel('ACF')
ax.set_title('ACF Decomposition'); ax.legend(); ax.axhline(0, color='gray', alpha=0.3)

ax = axes[1,0]
ax.hist(residuals, bins=200, density=True, alpha=0.6, color='steelblue', label='Residuals')
xn = np.linspace(-0.5, 0.5, 500)
ax.plot(xn, norm.pdf(xn, np.mean(residuals), np.std(residuals)),
        'r-', lw=2, label=f'Normal (std={np.std(residuals):.4f})')
ax.set_xlabel('Residual'); ax.set_ylabel('Density')
ax.set_title('Residual Distribution'); ax.legend(); ax.set_xlim(-0.5, 0.5)

ax = axes[1,1]
log_T_plot = np.log10(T_mids_eff)
ax.plot(log_T_plot, theta_effs, 'bo-', ms=8, label='theta_eff')
ax.axhline(THETA_STAR, color='r', ls='--', lw=2, label=f'theta*={THETA_STAR}')
ax.axhline(99/70, color='purple', ls=':', alpha=0.7, label=f'99/70={99/70:.4f}')
if rv_th**2 > 0.3:
    xext = np.linspace(1, 7, 100)
    ax.plot(xext, ic_th + sl_th/(xext*np.log(10)), 'g-.', alpha=0.7,
            label=f'Fit: plateau={ic_th:.4f}')
ax.set_xlabel('log10(T)'); ax.set_ylabel('theta_eff')
ax.set_title('Effective theta per window'); ax.legend(fontsize=9)

plt.tight_layout()
fig6 = f'{OUTDIR}/fig6_GUE_theta.png'
plt.savefig(fig6, dpi=150, bbox_inches='tight'); plt.close()
print(f"\nSaved {fig6}", flush=True)


# =====================================================================
# SUMMARY
# =====================================================================
total = time.time() - t0
print("\n" + "=" * 70, flush=True)
print("SUMMARY", flush=True)
print("=" * 70, flush=True)
print(f"\n  Constant model (theta*={THETA_STAR}):", flush=True)
print(f"    alpha = {alpha_OLS:.6f}, R2 = {R2_global:.6f}", flush=True)
print(f"    T5 MC:    {'PASS' if T5_pass else 'FAIL'}  margin={margin:+.6f}", flush=True)
print(f"    T7 Boot:  {'PASS' if T7_pass else 'FAIL'}  CI=[{ci_lo:.4f},{ci_hi:.4f}]", flush=True)
print(f"    T8 Drift: {'PASS' if T8_pass else 'FAIL'}  slope={slope:+.6f}, p={p_val_drift:.4f}", flush=True)
print(f"    SCORE: {n_pass}/3 (adaptive: 0/3)", flush=True)
print(f"\n  R2 extrapolation:", flush=True)
print(f"    decay = {sl_R2:+.4f}/decade", flush=True)
print(f"    plateau (1/logT model) = {ic_inv:.4f}", flush=True)
print(f"\n  theta_eff: mean={np.mean(theta_effs):.4f} std={np.std(theta_effs):.4f} asymptote={ic_th:.4f}", flush=True)
print(f"\n  GUE: KS={ks_stat:.6f} lag1_delta={acf1_d:+.4f} lag1_pred={acf1_p:+.4f}", flush=True)
print(f"\n  Total: {total:.0f}s", flush=True)

# Save JSON
out = {
    'validation': {
        'T5_MC': {'pass': bool(T5_pass), 'R2_opt': R2_opt,
                  'R2_best_random': R2_best_random, 'margin': margin,
                  'p_value': p_val_mc, 'n_trials': N_TRIALS},
        'T7_bootstrap': {'pass': bool(T7_pass), 'alpha_hat': alpha_OLS,
                         'CI_lo': ci_lo, 'CI_hi': ci_hi, 'B': B},
        'T8_drift': {'pass': bool(T8_pass), 'slope': float(slope),
                     'p_value': float(p_val_drift)},
        'score': f'{n_pass}/3'
    },
    'theta_eff': {'values': theta_effs.tolist(),
                  'mean': float(np.mean(theta_effs)),
                  'std': float(np.std(theta_effs)),
                  'asymptote': float(ic_th), 'fit_R2': float(rv_th**2)},
    'R2_decay': {'linear_slope': float(sl_R2), 'inv_log_plateau': float(ic_inv),
                 'inv_log_R2_fit': float(rv_inv**2)},
    'GUE': {'KS_stat': float(ks_stat), 'KS_pval': float(ks_pval),
            'lag1_delta': acf1_d, 'lag1_pred': acf1_p, 'lag1_resid': acf1_r}
}
jpath = f'{OUTDIR}/analysis_2M_extended_results.json'
with open(jpath, 'w') as f:
    json.dump(out, f, indent=2)
print(f"  Saved {jpath}", flush=True)
