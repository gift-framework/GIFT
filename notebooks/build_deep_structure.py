#!/usr/bin/env python3
"""
Build the Deep Structure notebook -- why do Fibonacci lags resonate with primes?

Run:    python3 notebooks/build_deep_structure.py
Output: notebooks/Deep_Structure_Fibonacci_Primes.ipynb
"""
import json
import textwrap

cells = []


def md(text):
    text = textwrap.dedent(text).strip()
    lines = text.split('\n')
    source = [l + '\n' for l in lines[:-1]]
    if lines:
        source.append(lines[-1])
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source})


def code(text):
    text = textwrap.dedent(text).strip()
    lines = text.split('\n')
    source = [l + '\n' for l in lines[:-1]]
    if lines:
        source.append(lines[-1])
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source
    })


# ================================================================
md(r"""
# Deep Structure: Why Do Fibonacci Lags Resonate with Primes?

## The Signal Hunt finding

The autocorrelation of $\varepsilon_n = S(\gamma_n)/\pi$ at Fibonacci lags has
**Z-score = 6.93** (p=0.000). The mean $|\rho(k)|$ at Fibonacci lags is
**4.4x larger** than at non-Fibonacci lags.

## The hypothesis

The explicit formula gives:
$$\rho(k) \propto \sum_p \frac{1}{p} \cos(k \cdot \bar{s} \cdot \log p)$$

The dominant term ($p=2$) oscillates with period $P_2 = \frac{2\pi}{\bar{s} \cdot \log 2} \approx 12.1$.

Since $13$ is a Fibonacci number, the Fibonacci sequence may **resonate** with this prime-induced oscillation.

## This notebook answers 4 questions

| # | Question | Test |
|---|----------|------|
| D1 | What is the exact ACF period? | Fit $\rho(k)$ to damped cosines |
| D2 | Can we decompose the ACF into prime contributions? | Theoretical $\rho(k)$ from explicit formula |
| D3 | Is Fibonacci special, or do other sequences do equally well? | Exhaustive comparison |
| D4 | Is there a theoretical reason for Fibonacci-prime resonance? | Analytical investigation |
""")

# ================================================================
code("""
# ============================================================
# SETUP
# ============================================================
!pip install -q tqdm matplotlib scipy mpmath 2>/dev/null || true

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    'font.size': 12, 'figure.figsize': (14, 6),
    'figure.dpi': 100, 'savefig.dpi': 150,
    'axes.grid': True, 'grid.alpha': 0.3
})
import time, json, os, warnings
from scipy.special import loggamma, lambertw
from scipy.optimize import curve_fit, minimize
from tqdm.auto import tqdm
warnings.filterwarnings('ignore')

try:
    import cupy as cp
    GPU = True
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
except Exception:
    GPU = False
    print("CPU mode")

PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)
FIBONACCI = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]
results = {}
print("Ready.")
""")

# ================================================================
md(r"""
## Data: Load zeros + compute $\varepsilon_n$ and ACF
""")

code("""
# ============================================================
# LOAD DATA + COMPUTE ACF
# ============================================================
import urllib.request

CACHE = 'riemann_zeros_100k_genuine.npy'
if os.path.exists(CACHE):
    gamma_n = np.load(CACHE)
    print(f"Loaded {len(gamma_n):,} cached zeros")
else:
    print("Downloading 100k zeros...")
    resp = urllib.request.urlopen('https://www-users.cse.umn.edu/~odlyzko/zeta_tables/zeros1', timeout=120)
    raw = resp.read().decode('utf-8')
    gamma_n = np.array([float(l.strip()) for l in raw.strip().split('\\n') if l.strip()])
    np.save(CACHE, gamma_n)
    print(f"Downloaded {len(gamma_n):,} zeros")

N_ZEROS = len(gamma_n)

# Theta function + smooth zeros
def theta_vec(t):
    s = 0.25 + 0.5j * np.asarray(t, dtype=np.float64)
    return np.imag(loggamma(s)) - 0.5 * np.asarray(t) * np.log(np.pi)

def compute_smooth_zeros(N):
    ns = np.arange(1, N + 1, dtype=np.float64)
    targets = (ns - 1.5) * np.pi
    w = np.real(lambertw(ns / np.e))
    t = np.maximum(2 * np.pi * ns / w, 2.0)
    for it in range(40):
        val = theta_vec(t)
        deriv = np.where(np.abs(d := 0.5 * np.log(np.maximum(t, 1.0) / (2*np.pi))) < 1e-15, 1e-15, d)
        dt = (val - targets) / deriv
        t -= dt
        if np.max(np.abs(dt)) < 1e-12:
            break
    return t

print("Computing smooth zeros...")
gamma_smooth = compute_smooth_zeros(N_ZEROS)
delta_n = gamma_n - gamma_smooth

# Unfolded zeros
ns = np.arange(1, N_ZEROS + 1, dtype=np.float64)
epsilon_n = theta_vec(gamma_n) / np.pi + 1.0 - ns

# ACF up to lag 1000
eps_c = epsilon_n - np.mean(epsilon_n)
eps_var = np.var(eps_c)
MAX_LAG = 1000

print(f"Computing ACF (lags 1..{MAX_LAG})...")
acf = np.zeros(MAX_LAG + 1)
acf[0] = 1.0
if GPU:
    eg = cp.asarray(eps_c)
    for k in tqdm(range(1, MAX_LAG+1), desc="ACF"):
        acf[k] = float(cp.mean(eg[:-k] * eg[k:]) / eps_var)
    cp.get_default_memory_pool().free_all_blocks()
else:
    for k in tqdm(range(1, MAX_LAG+1), desc="ACF"):
        acf[k] = np.mean(eps_c[:-k] * eps_c[k:]) / eps_var

mean_spacing = np.mean(np.diff(gamma_n))
print(f"\\nDone. Mean spacing = {mean_spacing:.6f}")
print(f"ACF at lag 1: {acf[1]:.6f}")
print(f"ACF at lag 13: {acf[13]:.6f}")
results['mean_spacing'] = float(mean_spacing)
""")

# ================================================================
# D1: Precise ACF period
# ================================================================
md(r"""
## D1: Precise Period of the ACF Oscillation

Fit the ACF to a sum of damped cosines:
$$\rho(k) = \sum_{j=1}^{M} A_j \, e^{-\alpha_j k} \cos(\omega_j k + \phi_j)$$

We start with $M=1$ (single prime $p=2$ dominance) and increase.
""")

code("""
# ============================================================
# D1: FIT THE ACF PERIOD
# ============================================================
print("=" * 70)
print("  D1: PRECISE ACF PERIOD")
print("=" * 70)

ks = np.arange(1, MAX_LAG + 1)
acf_data = acf[1:]

# --- Single damped cosine fit ---
def model_1(k, A, alpha, omega, phi):
    return A * np.exp(-alpha * k) * np.cos(omega * k + phi)

# Initial guess from p=2 theory
omega_2_theory = mean_spacing * np.log(2)
p0_1 = [0.2, 0.005, omega_2_theory, 0.0]
try:
    popt1, pcov1 = curve_fit(model_1, ks, acf_data, p0=p0_1, maxfev=50000)
    period_1 = 2 * np.pi / abs(popt1[2])
    print(f"\\n1-cosine fit:")
    print(f"  A={popt1[0]:.4f}, alpha={popt1[1]:.6f}, omega={popt1[2]:.6f}, phi={popt1[3]:.4f}")
    print(f"  Period = {period_1:.4f}")
    print(f"  Theory (p=2): omega = s_bar * log(2) = {omega_2_theory:.6f}, period = {2*np.pi/omega_2_theory:.4f}")
    resid_1 = np.sum((acf_data - model_1(ks, *popt1))**2)
    print(f"  Residual SS = {resid_1:.6f}")
except Exception as e:
    print(f"  1-cosine fit failed: {e}")
    popt1 = None
    period_1 = 2 * np.pi / omega_2_theory

# --- Two damped cosines (p=2 and p=3) ---
def model_2(k, A1, a1, w1, p1, A2, a2, w2, p2):
    return (A1 * np.exp(-a1*k) * np.cos(w1*k + p1) +
            A2 * np.exp(-a2*k) * np.cos(w2*k + p2))

omega_3_theory = mean_spacing * np.log(3)
p0_2 = [0.2, 0.005, omega_2_theory, 0.0, 0.1, 0.005, omega_3_theory, 0.0]
try:
    popt2, pcov2 = curve_fit(model_2, ks, acf_data, p0=p0_2, maxfev=100000)
    print(f"\\n2-cosine fit (p=2 + p=3):")
    print(f"  Component 1: A={popt2[0]:.4f}, omega={popt2[2]:.6f}, period={2*np.pi/abs(popt2[2]):.4f}")
    print(f"  Component 2: A={popt2[4]:.4f}, omega={popt2[6]:.6f}, period={2*np.pi/abs(popt2[6]):.4f}")
    resid_2 = np.sum((acf_data - model_2(ks, *popt2))**2)
    print(f"  Residual SS = {resid_2:.6f} (improvement: {(1-resid_2/resid_1)*100:.1f}%)")
except Exception as e:
    print(f"  2-cosine fit failed: {e}")
    popt2 = None

# --- Three cosines (p=2, p=3, p=5) ---
def model_3(k, A1,a1,w1,p1, A2,a2,w2,p2, A3,a3,w3,p3):
    return (A1*np.exp(-a1*k)*np.cos(w1*k+p1) +
            A2*np.exp(-a2*k)*np.cos(w2*k+p2) +
            A3*np.exp(-a3*k)*np.cos(w3*k+p3))

omega_5_theory = mean_spacing * np.log(5)
p0_3 = list(p0_2) + [0.05, 0.005, omega_5_theory, 0.0]
try:
    popt3, pcov3 = curve_fit(model_3, ks, acf_data, p0=p0_3, maxfev=200000)
    print(f"\\n3-cosine fit (p=2 + p=3 + p=5):")
    for i, label in enumerate(['p=2', 'p=3', 'p=5']):
        j = 4*i
        print(f"  {label}: A={popt3[j]:.4f}, omega={popt3[j+2]:.6f}, period={2*np.pi/abs(popt3[j+2]):.4f}")
    resid_3 = np.sum((acf_data - model_3(ks, *popt3))**2)
    print(f"  Residual SS = {resid_3:.6f} (improvement over 1-cos: {(1-resid_3/resid_1)*100:.1f}%)")
except Exception as e:
    print(f"  3-cosine fit failed: {e}")
    popt3 = None

# --- FFT of the ACF itself to find the period directly ---
print(f"\\nFFT of ACF (direct period detection):")
acf_fft = np.abs(np.fft.rfft(acf[1:501]))  # first 500 lags
acf_freqs = np.fft.rfftfreq(500, d=1.0)
top_acf_peaks = np.argsort(acf_fft[1:])[::-1][:10] + 1
print(f"  Top 5 ACF spectral peaks:")
for i in range(5):
    idx = top_acf_peaks[i]
    period = 1.0 / acf_freqs[idx] if acf_freqs[idx] > 0 else float('inf')
    print(f"    #{i+1}: freq_idx={idx}, freq={acf_freqs[idx]:.6f}, period={period:.2f}")

# KEY NUMBER: the ratio of ACF period to Fibonacci
fitted_period = period_1 if popt1 is not None else 2*np.pi/omega_2_theory
print(f"\\n{'='*60}")
print(f"  KEY NUMBERS:")
print(f"  Fitted ACF period:        {fitted_period:.4f}")
print(f"  Nearest Fibonacci:        13")
print(f"  Ratio period/13:          {fitted_period/13:.6f}")
print(f"  Theory (p=2):             {2*np.pi/(mean_spacing*np.log(2)):.4f}")
print(f"  log(T_mid/(2pi))/log(2):  {np.log(np.median(gamma_n)/(2*np.pi))/np.log(2):.4f}")
print(f"{'='*60}")

# Plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. ACF with fit
axes[0,0].bar(ks[:100], acf_data[:100], color='steelblue', alpha=0.6, width=1)
if popt1 is not None:
    axes[0,0].plot(ks[:100], model_1(ks[:100], *popt1), 'r-', lw=2, label='1-cosine fit')
if popt2 is not None:
    axes[0,0].plot(ks[:100], model_2(ks[:100], *popt2), 'g--', lw=2, label='2-cosine fit')
axes[0,0].set_xlabel('Lag k')
axes[0,0].set_ylabel(r'$\\rho(k)$')
axes[0,0].set_title('ACF with damped cosine fits')
axes[0,0].legend()

# 2. FFT of ACF
axes[0,1].plot(acf_freqs[1:], acf_fft[1:], 'b-', lw=1)
# Mark expected prime frequencies
for p, col in [(2,'red'), (3,'green'), (5,'orange')]:
    f_p = mean_spacing * np.log(p) / (2*np.pi)
    axes[0,1].axvline(f_p, color=col, ls='--', lw=1.5, label=f'p={p}: f={f_p:.4f}')
axes[0,1].set_xlabel('Frequency (cycles per lag)')
axes[0,1].set_title('FFT of ACF (peaks = prime frequencies)')
axes[0,1].legend(fontsize=9)

# 3. Residual after removing p=2 contribution
if popt1 is not None:
    resid_acf = acf_data - model_1(ks, *popt1)
    axes[1,0].bar(ks[:100], resid_acf[:100], color='steelblue', alpha=0.6, width=1)
    axes[1,0].set_title('ACF residual after removing p=2 oscillation')
    axes[1,0].set_xlabel('Lag k')

# 4. Period convergence: compute period for different N ranges
print("\\nPeriod stability across zero ranges:")
period_vs_N = []
N_ranges = [(0, 20000), (20000, 40000), (40000, 60000), (60000, 80000), (80000, 100000)]
for lo, hi in N_ranges:
    eps_slice = epsilon_n[lo:hi]
    ec = eps_slice - np.mean(eps_slice)
    ev = np.var(ec)
    local_acf = np.zeros(101)
    for k in range(1, 101):
        local_acf[k] = np.mean(ec[:-k] * ec[k:]) / ev
    # Fit period
    try:
        p_local, _ = curve_fit(model_1, np.arange(1, 101), local_acf[1:],
                                p0=[0.2, 0.01, omega_2_theory, 0.0], maxfev=20000)
        local_period = 2*np.pi/abs(p_local[2])
    except:
        local_period = np.nan
    # Local mean spacing
    local_spacing = np.mean(np.diff(gamma_n[lo:hi]))
    theory_period = 2*np.pi / (local_spacing * np.log(2))
    period_vs_N.append((lo, hi, local_period, theory_period, local_spacing))
    print(f"  [{lo:>6},{hi:>6}]: period={local_period:.3f}, theory(p=2)={theory_period:.3f}, spacing={local_spacing:.4f}")

axes[1,1].plot([f"{lo//1000}k-{hi//1000}k" for lo,hi,_,_,_ in period_vs_N],
               [p for _,_,p,_,_ in period_vs_N], 'bo-', markersize=8, label='Fitted period')
axes[1,1].plot([f"{lo//1000}k-{hi//1000}k" for lo,hi,_,_,_ in period_vs_N],
               [t for _,_,_,t,_ in period_vs_N], 'rs--', markersize=8, label='Theory (p=2)')
axes[1,1].axhline(13, color='green', ls=':', lw=2, label='Fibonacci 13')
axes[1,1].set_ylabel('ACF period')
axes[1,1].set_title('Period stability across zero ranges')
axes[1,1].legend()

plt.tight_layout()
plt.savefig('deep_D1_period.png', dpi=150, bbox_inches='tight')
plt.show()

results['D1_period'] = {
    'fitted_period': float(fitted_period),
    'theory_p2': float(2*np.pi/(mean_spacing*np.log(2))),
    'ratio_to_13': float(fitted_period/13),
}
""")

# ================================================================
# D2: Prime decomposition of ACF
# ================================================================
md(r"""
## D2: Theoretical ACF from the Explicit Formula

If $\varepsilon_n = S(\gamma_n)/\pi$ and $S(T) \sim -\frac{1}{\pi}\sum_p \frac{\sin(T\log p)}{\sqrt{p}}$,
then the autocorrelation should be:
$$\rho_{\text{theory}}(k) = \frac{\sum_p \frac{1}{p} \cos(k \bar{s} \log p)}{\sum_p \frac{1}{p}}$$

We compute this and compare with the empirical ACF.
""")

code("""
# ============================================================
# D2: THEORETICAL ACF FROM PRIMES
# ============================================================
print("=" * 70)
print("  D2: THEORETICAL ACF FROM EXPLICIT FORMULA")
print("=" * 70)

# Primes up to 1000
def sieve(n):
    is_p = np.ones(n+1, dtype=bool)
    is_p[:2] = False
    for i in range(2, int(n**0.5)+1):
        if is_p[i]:
            is_p[i*i::i] = False
    return np.where(is_p)[0]

primes = sieve(1000)
print(f"Using {len(primes)} primes up to {primes[-1]}")

# Theoretical ACF: rho(k) = sum_p (1/p) cos(k * s_bar * log(p)) / sum_p (1/p)
# Actually include prime powers: sum_p sum_m (1/p^m) cos(k * s_bar * m*log(p))
s_bar = mean_spacing

def rho_theory(k, n_primes=None, include_powers=True):
    if n_primes is None:
        ps = primes
    else:
        ps = primes[:n_primes]
    val = 0.0
    norm = 0.0
    for p in ps:
        m_max = 5 if include_powers else 1
        for m in range(1, m_max + 1):
            w = 1.0 / (p ** m)
            val += w * np.cos(k * s_bar * m * np.log(p))
            norm += w
    return val / norm

# Compute theoretical ACF
ks = np.arange(0, MAX_LAG + 1)
rho_th_full = np.array([rho_theory(k) for k in ks])
rho_th_p2_only = np.array([rho_theory(k, n_primes=1) for k in ks])
rho_th_p23 = np.array([rho_theory(k, n_primes=2) for k in ks])
rho_th_p235 = np.array([rho_theory(k, n_primes=3) for k in ks])

# Correlation between theory and data
from scipy.stats import pearsonr
corr_full, _ = pearsonr(acf[1:201], rho_th_full[1:201])
corr_p2, _ = pearsonr(acf[1:201], rho_th_p2_only[1:201])
corr_p23, _ = pearsonr(acf[1:201], rho_th_p23[1:201])

print(f"\\nCorrelation (theory vs empirical ACF, lags 1-200):")
print(f"  p=2 only:          r = {corr_p2:.4f}")
print(f"  p=2,3:             r = {corr_p23:.4f}")
print(f"  p=2,3,5:           r = {pearsonr(acf[1:201], rho_th_p235[1:201])[0]:.4f}")
print(f"  All primes to 1000: r = {corr_full:.4f}")

# Scale theoretical to match empirical amplitude
scale = np.std(acf[1:201]) / np.std(rho_th_full[1:201])
print(f"\\nAmplitude scale factor: {scale:.4f}")
print(f"  (theory needs to be scaled by {scale:.2f}x to match empirical amplitude)")

# Fibonacci lags in theory vs data
print(f"\\nFibonacci lags: theory vs data")
print(f"  {'Lag':>6} {'Empirical':>12} {'Theory':>12} {'Theory(p=2)':>14}")
for f in FIBONACCI:
    if f <= 55:
        print(f"  {f:>6} {acf[f]:>12.6f} {rho_th_full[f]*scale:>12.6f} {rho_th_p2_only[f]*scale:>12.6f}")

# KEY: compute |rho| at Fibonacci lags for the THEORETICAL ACF
fib_in = [f for f in FIBONACCI if 2 <= f <= MAX_LAG]
non_fib = [k for k in range(2, MAX_LAG+1) if k not in FIBONACCI]

th_mean_fib = np.mean([abs(rho_th_full[f]) for f in fib_in])
th_mean_nonfib = np.mean([abs(rho_th_full[k]) for k in non_fib])
th_ratio = th_mean_fib / th_mean_nonfib

print(f"\\nTheoretical Fibonacci advantage:")
print(f"  Mean |rho_theory| at Fibonacci lags:     {th_mean_fib:.6f}")
print(f"  Mean |rho_theory| at non-Fibonacci lags: {th_mean_nonfib:.6f}")
print(f"  Ratio: {th_ratio:.3f}x")
print(f"  (Empirical was 4.44x)")

# Plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

axes[0,0].plot(ks[1:100], acf[1:100], 'b.-', alpha=0.7, label='Empirical')
axes[0,0].plot(ks[1:100], rho_th_full[1:100]*scale, 'r-', lw=2, label='Theory (all primes)')
axes[0,0].set_xlabel('Lag k')
axes[0,0].set_title('Empirical vs theoretical ACF')
axes[0,0].legend()

# Decomposition by prime
axes[0,1].plot(ks[1:80], rho_th_p2_only[1:80]*scale, 'r-', lw=2, label='p=2 only')
axes[0,1].plot(ks[1:80], (rho_th_p23[1:80]-rho_th_p2_only[1:80])*scale, 'g-', lw=1.5, label='p=3 contribution')
axes[0,1].plot(ks[1:80], (rho_th_p235[1:80]-rho_th_p23[1:80])*scale, 'orange', lw=1.5, label='p=5 contribution')
for f in FIBONACCI:
    if f < 80:
        axes[0,1].axvline(f, color='purple', alpha=0.15, lw=2)
axes[0,1].set_xlabel('Lag k')
axes[0,1].set_title('Prime decomposition of ACF (purple = Fibonacci)')
axes[0,1].legend(fontsize=9)

# Theory predicts Fibonacci advantage
axes[1,0].bar(['Fibonacci', 'Non-Fib'], [th_mean_fib, th_mean_nonfib], color=['red', 'steelblue'])
axes[1,0].set_ylabel(r'Mean $|\\rho_{theory}|$')
axes[1,0].set_title(f'Theoretical Fibonacci advantage: {th_ratio:.2f}x')

# Scatter: empirical vs theory at each lag
axes[1,1].scatter(rho_th_full[1:201]*scale, acf[1:201], s=5, alpha=0.5, color='steelblue')
fib_mask = [f for f in FIBONACCI if f <= 200]
axes[1,1].scatter([rho_th_full[f]*scale for f in fib_mask],
                   [acf[f] for f in fib_mask],
                   s=50, color='red', zorder=5, label='Fibonacci')
axes[1,1].plot([-0.3, 0.3], [-0.3, 0.3], 'k--', lw=1)
axes[1,1].set_xlabel('Theoretical ACF')
axes[1,1].set_ylabel('Empirical ACF')
axes[1,1].set_title(f'Theory vs data (r={corr_full:.3f})')
axes[1,1].legend()

plt.tight_layout()
plt.savefig('deep_D2_primes.png', dpi=150, bbox_inches='tight')
plt.show()

results['D2_theory'] = {
    'corr_p2': float(corr_p2),
    'corr_all': float(corr_full),
    'theory_fib_ratio': float(th_ratio),
    'amplitude_scale': float(scale),
}
""")

# ================================================================
# D3: Is Fibonacci special?
# ================================================================
md(r"""
## D3: Is Fibonacci Special? Exhaustive Comparison

The critical test: we compare the Fibonacci sequence against **every other
subsequence** of the same density.

Specifically:
1. **All geometric sequences** $\lfloor c \cdot r^k \rfloor$ for various $c, r$
2. **All "Fibonacci-like" sequences** $a_{n+2} = a_{n+1} + a_n$ with different seeds
3. **Random sequences** of the same cardinality
4. **Optimal sequences**: what sequence maximizes the mean |ACF| sampling?
""")

code("""
# ============================================================
# D3: IS FIBONACCI SPECIAL?
# ============================================================
print("=" * 70)
print("  D3: IS FIBONACCI SPECIAL AMONG ALL SEQUENCES?")
print("=" * 70)

def mean_abs_acf(lags, acf_data, max_lag=MAX_LAG):
    valid = [l for l in lags if 2 <= l <= max_lag]
    if len(valid) == 0:
        return 0.0
    return np.mean([abs(acf_data[l]) for l in valid])

# Reference: Fibonacci
fib_in_range = [f for f in FIBONACCI if 2 <= f <= MAX_LAG]
fib_score = mean_abs_acf(fib_in_range, acf)
n_fib = len(fib_in_range)
print(f"\\nFibonacci ({n_fib} lags in [2,{MAX_LAG}]): mean|ACF| = {fib_score:.6f}")

# --- 1. Geometric sequences a_k = ceil(r^k) ---
print(f"\\n1. Geometric sequences (r^k):")
geom_scores = []
rs = np.linspace(1.1, 3.0, 200)
for r in rs:
    lags = list(set(int(np.ceil(r**k)) for k in range(1, 50) if r**k <= MAX_LAG))
    lags = [l for l in lags if l >= 2]
    if len(lags) >= n_fib - 2:  # similar cardinality
        score = mean_abs_acf(lags, acf)
        geom_scores.append((r, score, len(lags)))

best_geom = max(geom_scores, key=lambda x: x[1])
print(f"  Best geometric: r={best_geom[0]:.3f}, score={best_geom[1]:.6f}, n_lags={best_geom[2]}")
print(f"  Fibonacci score: {fib_score:.6f}")
print(f"  Fibonacci is {'BETTER' if fib_score >= best_geom[1] else 'WORSE'} by {abs(fib_score-best_geom[1])/fib_score*100:.1f}%")

# Special check: golden ratio
phi_lags = list(set(int(np.ceil(PHI**k)) for k in range(2, 50) if PHI**k <= MAX_LAG))
phi_lags = [l for l in phi_lags if l >= 2]
phi_score = mean_abs_acf(phi_lags, acf)
print(f"  Golden ratio (phi^k): score={phi_score:.6f} ({len(phi_lags)} lags)")

# --- 2. Fibonacci-like sequences (generalized) ---
print(f"\\n2. Fibonacci-like sequences (a_n = a_{'{n-1}'} + a_{'{n-2}'}, various seeds):")
fiblike_scores = []
for a, b in [(1,1), (1,2), (1,3), (2,1), (2,3), (1,4), (3,1), (2,5), (1,5), (3,2)]:
    seq = [a, b]
    while seq[-1] + seq[-2] <= MAX_LAG:
        seq.append(seq[-1] + seq[-2])
    lags = [s for s in seq if s >= 2]
    score = mean_abs_acf(lags, acf)
    fiblike_scores.append((a, b, score, len(lags)))
    print(f"  ({a},{b}): score={score:.6f}, n_lags={len(lags)}")

# --- 3. Lucas numbers ---
lucas = [2, 1]
while lucas[-1] + lucas[-2] <= MAX_LAG:
    lucas.append(lucas[-1] + lucas[-2])
lucas_lags = [l for l in lucas if l >= 2]
lucas_score = mean_abs_acf(lucas_lags, acf)
print(f"\\n  Lucas numbers: score={lucas_score:.6f} ({len(lucas_lags)} lags)")

# --- 4. Random sequences ---
print(f"\\n3. Random sequences (same cardinality as Fibonacci):")
N_RAND = 100000
rand_scores = np.zeros(N_RAND)
for i in range(N_RAND):
    rand_lags = sorted(np.random.choice(range(2, MAX_LAG+1), size=n_fib, replace=False))
    rand_scores[i] = mean_abs_acf(rand_lags, acf)

z_vs_random = (fib_score - np.mean(rand_scores)) / np.std(rand_scores)
pct_beaten = np.mean(rand_scores >= fib_score) * 100
print(f"  Fibonacci Z-score vs random: {z_vs_random:.2f}")
print(f"  Fibonacci percentile: {100 - pct_beaten:.4f}%")
print(f"  Random sets that beat Fibonacci: {pct_beaten:.4f}%")

# --- 5. What is the OPTIMAL sequence? ---
print(f"\\n4. Optimal sequence (greedy construction):")
# Greedily pick lags that maximize cumulative |ACF|
available = list(range(2, MAX_LAG + 1))
abs_acf_sorted = sorted(available, key=lambda k: -abs(acf[k]))
optimal_lags = abs_acf_sorted[:n_fib]
optimal_score = mean_abs_acf(optimal_lags, acf)
print(f"  Optimal (top-{n_fib} |ACF| lags): score={optimal_score:.6f}")
print(f"  Fibonacci: {fib_score:.6f}")
print(f"  Ratio Fibonacci/Optimal: {fib_score/optimal_score:.4f}")
print(f"  Optimal lags: {sorted(optimal_lags)[:20]}...")

# --- 6. Among geometric sequences, sweep r finely near phi ---
print(f"\\n5. Fine sweep near golden ratio:")
rs_fine = np.linspace(1.5, 1.7, 500)
fine_scores = []
for r in rs_fine:
    lags = sorted(set(int(np.ceil(r**k)) for k in range(2, 50) if 2 <= r**k <= MAX_LAG))
    if len(lags) >= max(3, n_fib - 4):
        fine_scores.append((r, mean_abs_acf(lags, acf), len(lags)))

if fine_scores:
    best_fine = max(fine_scores, key=lambda x: x[1])
    print(f"  Best r in [1.5, 1.7]: r={best_fine[0]:.6f}, score={best_fine[1]:.6f}")
    print(f"  phi = {PHI:.6f}")

# --- 7. THEORY prediction: which geometric base maximizes Fibonacci advantage? ---
print(f"\\n6. Theoretical prediction:")
print(f"  ACF period P = {fitted_period:.4f}")
print(f"  For geometric seq with base r, the sampling pattern hits")
print(f"  phases 2*pi*r^k/P. Resonance when r ~ P^(1/k) for some k.")
print(f"  P^(1/7) = {fitted_period**(1/7):.4f}")
print(f"  phi     = {PHI:.4f}")
print(f"  P^(1/7) ~ phi? Ratio: {fitted_period**(1/7)/PHI:.6f}")

# Plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Geometric sweep
axes[0,0].plot([x[0] for x in geom_scores], [x[1] for x in geom_scores], 'b.', markersize=3, alpha=0.5)
axes[0,0].axhline(fib_score, color='red', lw=2, label=f'Fibonacci = {fib_score:.5f}')
axes[0,0].axvline(PHI, color='green', lw=2, ls=':', label=f'phi = {PHI:.3f}')
axes[0,0].set_xlabel('Geometric ratio r')
axes[0,0].set_ylabel('Mean |ACF| score')
axes[0,0].set_title('Geometric sequences vs Fibonacci')
axes[0,0].legend()

# 2. Random distribution
axes[0,1].hist(rand_scores, bins=200, density=True, alpha=0.7, color='gray')
axes[0,1].axvline(fib_score, color='red', lw=2, label=f'Fibonacci (z={z_vs_random:.1f})')
axes[0,1].axvline(best_geom[1], color='green', lw=1.5, ls='--', label='Best geometric')
axes[0,1].set_xlabel('Mean |ACF| score')
axes[0,1].set_title(f'Fibonacci vs {N_RAND:,} random sequences')
axes[0,1].legend()

# 3. Fine sweep near phi
if fine_scores:
    axes[1,0].plot([x[0] for x in fine_scores], [x[1] for x in fine_scores], 'b.-', markersize=3)
    axes[1,0].axvline(PHI, color='green', lw=2, label=f'phi = {PHI:.4f}')
    axes[1,0].axhline(fib_score, color='red', lw=1.5, ls='--', label='Fibonacci')
    axes[1,0].set_xlabel('Geometric ratio r')
    axes[1,0].set_title('Fine sweep near golden ratio')
    axes[1,0].legend()

# 4. Comparison bar chart
labels = ['Fibonacci', 'Lucas', f'Best geom\\n(r={best_geom[0]:.2f})',
          f'phi^k', 'Random\\n(mean)', 'Optimal']
scores_bar = [fib_score, lucas_score, best_geom[1], phi_score,
              np.mean(rand_scores), optimal_score]
colors = ['red', 'orange', 'green', 'purple', 'gray', 'gold']
axes[1,1].bar(labels, scores_bar, color=colors)
axes[1,1].set_ylabel('Mean |ACF| score')
axes[1,1].set_title('Sequence comparison')
axes[1,1].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('deep_D3_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

results['D3_comparison'] = {
    'fib_score': float(fib_score),
    'best_geom_r': float(best_geom[0]),
    'best_geom_score': float(best_geom[1]),
    'phi_k_score': float(phi_score),
    'lucas_score': float(lucas_score),
    'optimal_score': float(optimal_score),
    'z_vs_random': float(z_vs_random),
    'pct_random_beaten': float(pct_beaten),
}
""")

# ================================================================
# D4: Analytical investigation
# ================================================================
md(r"""
## D4: Why Does Fibonacci Resonate?

### The analytical argument

The ACF has period $P_2 = 2\pi / (\bar{s} \cdot \log 2)$. The Fibonacci
numbers grow as $F_n \sim \varphi^n / \sqrt{5}$.

The key question: **when does a geometric sequence $\lfloor r^k \rfloor$
optimally sample the extrema of $\cos(2\pi k' / P)$?**

This happens when the sequence visits all phases $2\pi r^k / P \pmod{2\pi}$
and preferentially hits near $0$ and $\pi$ (the extrema).

For the golden ratio $\varphi$, the sequence $\varphi^k \bmod P$ has special
equidistribution properties related to the **three-distance theorem**.
""")

code("""
# ============================================================
# D4: ANALYTICAL INVESTIGATION
# ============================================================
print("=" * 70)
print("  D4: WHY DOES FIBONACCI RESONATE?")
print("=" * 70)

P2 = 2 * np.pi / (mean_spacing * np.log(2))
print(f"ACF period from p=2: P2 = {P2:.6f}")

# Phase analysis: where do Fibonacci numbers fall in the cycle?
print(f"\\nPhase analysis (mod P2):")
print(f"  {'Lag':>6} {'Phase/2pi':>12} {'cos(phase)':>12} {'Near extremum':>15}")
for f in FIBONACCI:
    if f <= 55:
        phase = (f % P2) / P2  # fraction of cycle
        cos_val = np.cos(2 * np.pi * f / P2)
        near = "PEAK" if abs(cos_val) > 0.8 else ("node" if abs(cos_val) < 0.2 else "")
        print(f"  {f:>6} {phase:>12.4f} {cos_val:>12.4f} {near:>15}")

# Phase distribution for different geometric sequences
print(f"\\nPhase distribution analysis:")
print(f"  For each base r, compute mean|cos(2*pi*r^k/P2)| over valid lags")
rs_analysis = np.linspace(1.1, 3.0, 1000)
phase_scores = []
for r in rs_analysis:
    lags = [int(np.ceil(r**k)) for k in range(2, 50) if 2 <= r**k <= MAX_LAG]
    if len(lags) >= 5:
        mean_cos = np.mean([abs(np.cos(2*np.pi*l/P2)) for l in lags])
        phase_scores.append((r, mean_cos, len(lags)))

# Fibonacci phase score
fib_phase = np.mean([abs(np.cos(2*np.pi*f/P2)) for f in fib_in_range])

# Three-distance theorem connection
print(f"\\nThree-distance theorem:")
print(f"  The fractional parts of n*alpha for irrational alpha")
print(f"  have exactly 2 or 3 distinct gap sizes.")
print(f"  For alpha = log(phi)/log(2):")
alpha = np.log(PHI) / np.log(2)
print(f"  alpha = log(phi)/log(2) = {alpha:.10f}")
print(f"  Continued fraction: ", end="")
# Compute CF of alpha
a = alpha
cf = []
for _ in range(12):
    cf.append(int(a))
    a = 1.0 / (a - int(a)) if abs(a - int(a)) > 1e-10 else 0
    if a == 0:
        break
print(cf)

# P2 relationship to Fibonacci
print(f"\\nKey relationships:")
print(f"  P2 = {P2:.6f}")
print(f"  F(7) = 13")
print(f"  P2 / 13 = {P2/13:.6f}")
print(f"  phi^7 / sqrt(5) = {PHI**7 / np.sqrt(5):.4f}")
print(f"  Round to nearest: {round(PHI**7 / np.sqrt(5))}")
print(f"  (F(7) = 13 exactly)")

# The resonance condition: when does P2 * k = Fibonacci for some k?
# Equivalently: Fibonacci / P2 should be near integers
print(f"\\nFibonacci / P2 (nearness to integers):")
for f in FIBONACCI:
    if f <= 987:
        ratio = f / P2
        nearest_int = round(ratio)
        frac_part = abs(ratio - nearest_int)
        print(f"  F={f:>4}: F/P2 = {ratio:>8.4f}, nearest int = {nearest_int:>3}, frac = {frac_part:.4f}")

# Multi-prime resonance: does Fibonacci also resonate with p=3?
P3 = 2 * np.pi / (mean_spacing * np.log(3))
P5 = 2 * np.pi / (mean_spacing * np.log(5))
print(f"\\nMulti-prime periods:")
print(f"  P2 = {P2:.4f} (from p=2)")
print(f"  P3 = {P3:.4f} (from p=3)")
print(f"  P5 = {P5:.4f} (from p=5)")
print(f"  P2/P3 = {P2/P3:.6f} (= log(3)/log(2) = {np.log(3)/np.log(2):.6f})")

print(f"\\nFibonacci phase scores per prime:")
for p_val, P_val in [(2, P2), (3, P3), (5, P5)]:
    fib_ph = np.mean([abs(np.cos(2*np.pi*f/P_val)) for f in fib_in_range])
    all_ph = np.mean([abs(np.cos(2*np.pi*k/P_val)) for k in range(2, MAX_LAG+1)])
    print(f"  p={p_val}: Fibonacci phase score = {fib_ph:.4f}, average = {all_ph:.4f}, ratio = {fib_ph/all_ph:.3f}")

# Plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Phase sweep
if phase_scores:
    axes[0,0].plot([x[0] for x in phase_scores], [x[1] for x in phase_scores], 'b.', markersize=2, alpha=0.5)
    axes[0,0].axhline(fib_phase, color='red', lw=2, label=f'Fibonacci = {fib_phase:.4f}')
    axes[0,0].axvline(PHI, color='green', lw=2, ls=':', label=f'phi')
    axes[0,0].set_xlabel('Geometric base r')
    axes[0,0].set_ylabel('Mean |cos(2pi*lag/P2)|')
    axes[0,0].set_title('Phase sampling efficiency')
    axes[0,0].legend()

# 2. Fibonacci on the unit circle (phases mod P2)
thetas = [2*np.pi*f/P2 for f in FIBONACCI if f <= 55]
axes[0,1].plot(np.cos(thetas), np.sin(thetas), 'ro', markersize=10)
for f, th in zip([f for f in FIBONACCI if f <= 55], thetas):
    axes[0,1].annotate(str(f), (np.cos(th)*1.12, np.sin(th)*1.12), ha='center', fontsize=9)
circle = np.linspace(0, 2*np.pi, 100)
axes[0,1].plot(np.cos(circle), np.sin(circle), 'k-', lw=0.5)
axes[0,1].set_xlim(-1.4, 1.4)
axes[0,1].set_ylim(-1.4, 1.4)
axes[0,1].set_aspect('equal')
axes[0,1].set_title(f'Fibonacci phases mod P2={P2:.2f}')
axes[0,1].axhline(0, color='gray', lw=0.3)
axes[0,1].axvline(0, color='gray', lw=0.3)

# 3. ACF with Fibonacci markers + extrema
axes[1,0].plot(ks[1:80], acf[1:80], 'b-', lw=1.5)
# Mark theoretical extrema of cos(2*pi*k/P2)
for m in range(10):
    peak_k = m * P2
    if peak_k < 80:
        axes[1,0].axvline(peak_k, color='orange', alpha=0.3, lw=1)
    trough_k = (m + 0.5) * P2
    if trough_k < 80:
        axes[1,0].axvline(trough_k, color='orange', alpha=0.3, lw=1, ls='--')
for f in FIBONACCI:
    if f < 80:
        axes[1,0].axvline(f, color='red', alpha=0.3, lw=2)
        axes[1,0].plot(f, acf[f], 'ro', markersize=8)
axes[1,0].set_xlabel('Lag k')
axes[1,0].set_title('ACF: Fibonacci (red) vs P2 extrema (orange)')

# 4. Multi-prime Fibonacci resonance
primes_test = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
fib_phases_by_prime = []
avg_phases_by_prime = []
for p_val in primes_test:
    P_val = 2*np.pi / (mean_spacing * np.log(p_val))
    fib_ph = np.mean([abs(np.cos(2*np.pi*f/P_val)) for f in fib_in_range])
    avg_ph = np.mean([abs(np.cos(2*np.pi*k/P_val)) for k in range(2, MAX_LAG+1)])
    fib_phases_by_prime.append(fib_ph)
    avg_phases_by_prime.append(avg_ph)

x_pos = range(len(primes_test))
axes[1,1].bar([x-0.15 for x in x_pos], fib_phases_by_prime, width=0.3, color='red', label='Fibonacci')
axes[1,1].bar([x+0.15 for x in x_pos], avg_phases_by_prime, width=0.3, color='steelblue', label='Average')
axes[1,1].set_xticks(x_pos)
axes[1,1].set_xticklabels([str(p) for p in primes_test])
axes[1,1].set_xlabel('Prime p')
axes[1,1].set_ylabel('Mean |cos(2pi*lag/Pp)|')
axes[1,1].set_title('Fibonacci resonance by prime')
axes[1,1].legend()

plt.tight_layout()
plt.savefig('deep_D4_resonance.png', dpi=150, bbox_inches='tight')
plt.show()

results['D4_resonance'] = {
    'P2': float(P2),
    'P3': float(P3),
    'P5': float(P5),
    'fib_phase_score_p2': float(fib_phase),
    'cf_log_phi_log_2': cf[:8],
}
""")

# ================================================================
# Summary
# ================================================================
md(r"""
## Summary: The Fibonacci-Prime Resonance
""")

code("""
# ============================================================
# FINAL SUMMARY
# ============================================================
print("=" * 70)
print("  DEEP STRUCTURE: SUMMARY")
print("=" * 70)

print(f"\\n1. ACF PERIOD")
print(f"   Fitted period: {results['D1_period']['fitted_period']:.4f}")
print(f"   Theory (p=2):  {results['D1_period']['theory_p2']:.4f}")
print(f"   Nearest Fib:   13")
print(f"   Ratio:         {results['D1_period']['ratio_to_13']:.4f}")

print(f"\\n2. PRIME DECOMPOSITION")
print(f"   Theory-data correlation (all primes): {results['D2_theory']['corr_all']:.4f}")
print(f"   Theory predicts Fibonacci advantage:  {results['D2_theory']['theory_fib_ratio']:.2f}x")
print(f"   Empirical Fibonacci advantage:        4.44x")

print(f"\\n3. FIBONACCI vs ALTERNATIVES")
print(f"   Fibonacci score:                {results['D3_comparison']['fib_score']:.6f}")
print(f"   Best geometric (r={results['D3_comparison']['best_geom_r']:.2f}):  {results['D3_comparison']['best_geom_score']:.6f}")
print(f"   Lucas numbers:                  {results['D3_comparison']['lucas_score']:.6f}")
print(f"   phi^k:                          {results['D3_comparison']['phi_k_score']:.6f}")
print(f"   Optimal (cherry-picked):        {results['D3_comparison']['optimal_score']:.6f}")
print(f"   Z-score vs random:              {results['D3_comparison']['z_vs_random']:.1f}")

print(f"\\n4. RESONANCE MECHANISM")
print(f"   P2 = 2pi/(s_bar*log(2)) = {results['D4_resonance']['P2']:.4f}")
print(f"   P2 ~ 13 (Fibonacci) means Fibonacci lags revisit the same")
print(f"   phase of the prime-2 oscillation every ~13 steps.")
print(f"   The golden ratio growth (each Fib ~ phi * previous)")
print(f"   means successive Fibonacci numbers sample different phases,")
print(f"   but the RATIO phi ensures extrema are preferentially hit.")

with open('deep_structure_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\\nResults saved to deep_structure_results.json")

# Summary plot
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
labels = ['Fibonacci', 'Lucas', f'Best geom', 'phi^k', 'Random (mean)', 'Optimal']
scores = [results['D3_comparison']['fib_score'],
          results['D3_comparison']['lucas_score'],
          results['D3_comparison']['best_geom_score'],
          results['D3_comparison']['phi_k_score'],
          results['D3_comparison']['z_vs_random'] * np.std(rand_scores) * 0 + np.mean(rand_scores),
          results['D3_comparison']['optimal_score']]
colors = ['red', 'orange', 'green', 'purple', 'gray', 'gold']
bars = ax.bar(labels, scores, color=colors)
ax.set_ylabel('Mean |ACF| at sequence lags')
ax.set_title('Deep Structure: Which sequence best captures the arithmetic signal?')
ax.tick_params(axis='x', rotation=25)
plt.tight_layout()
plt.savefig('deep_summary.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\\n{'='*70}")
print(f"  DEEP STRUCTURE ANALYSIS COMPLETE")
print(f"{'='*70}")
""")


# ================================================================
# WRITE NOTEBOOK
# ================================================================
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "colab": {"provenance": [], "gpuType": "A100"},
        "accelerator": "GPU"
    },
    "cells": cells
}

path = "/home/user/GIFT/notebooks/Deep_Structure_Fibonacci_Primes.ipynb"
with open(path, "w") as f:
    json.dump(notebook, f, indent=1)

print(f"Generated {path}")
print(f"  {len(cells)} cells ({sum(1 for c in cells if c['cell_type']=='code')} code, "
      f"{sum(1 for c in cells if c['cell_type']=='markdown')} markdown)")
print(f"  Total lines: {sum(len(c['source']) for c in cells)}")
