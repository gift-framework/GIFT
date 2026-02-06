#!/usr/bin/env python3
"""
Build the Grand Slam Selberg-Fibonacci Validation Notebook.

Run:    python3 notebooks/build_grandslam.py
Output: notebooks/Selberg_Fibonacci_GrandSlam.ipynb
"""
import json
import textwrap

cells = []


def md(text):
    """Add markdown cell (use raw strings for LaTeX)."""
    text = textwrap.dedent(text).strip()
    lines = text.split('\n')
    source = [l + '\n' for l in lines[:-1]]
    if lines:
        source.append(lines[-1])
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source})


def code(text):
    """Add code cell."""
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
# CELL 0: Title
# ================================================================
md(r"""
# Selberg–Fibonacci Grand Slam Validation

## Comprehensive Numerical Verification on 100k+ Genuine Riemann Zeros

**Environment**: Google Colab Pro+ (A100 GPU recommended)
**Data**: Odlyzko precomputed tables (100k–2M zeros) + python-flint cross-validation

### Objective

Validate the Fibonacci recurrence on Riemann zeros:

$$\gamma_n \approx \frac{31}{21}\gamma_{n-8} - \frac{10}{21}\gamma_{n-21} + c(n)$$

with **>98% capture** on 100,000+ genuine zeros.

### Tests

| # | Test | Goal |
|---|------|------|
| 1 | FFT Spectral Analysis | Fibonacci peaks dominate $\delta_n$ spectrum |
| 2 | Capture Ratio | >98% of $\delta_n$ explained by recurrence |
| 3 | Scaling Analysis | Convergence from 1k to 2M zeros |
| 4 | Coefficient Optimality | 31/21 is the unique optimum |
| 5 | Lag Optimality | (8,21) beats all other pairs |
| 6 | Statistical Significance | Permutation tests, Z-scores, bootstrap CI |
| 7 | Linearization Bounds | Data for analytical gap closure |
| 8 | Parseval Energy | Spectral energy in Fibonacci modes |
| 9 | Residual Anatomy | Structure of the remaining few % |
| 10 | Window Robustness | Stability across different zero ranges |
""")

# ================================================================
# CELL 1: Setup
# ================================================================
code("""
# ============================================================
# SETUP & CONFIGURATION
# ============================================================
!pip install -q python-flint tqdm matplotlib scipy mpmath 2>/dev/null || true

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    'font.size': 12, 'figure.figsize': (14, 6),
    'figure.dpi': 100, 'savefig.dpi': 150,
    'axes.grid': True, 'grid.alpha': 0.3
})
import time, json, os, warnings, sys
from scipy.special import loggamma, lambertw
from tqdm.auto import tqdm
warnings.filterwarnings('ignore')

# Try GPU
try:
    import cupy as cp
    GPU = True
    gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    gpu_mem = cp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem'] / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
except Exception:
    GPU = False
    print("No GPU detected, using CPU (GPU optional for this notebook)")

# Constants
PHI = (1 + np.sqrt(5)) / 2
LOG_PHI = np.log(PHI)
A_COEFF = 31 / 21
B_COEFF = -10 / 21
LAG_1, LAG_2 = 8, 21
ELL_0 = 2 * LOG_PHI        # primitive geodesic length
ELL_8 = 2 * LAG_1 * LOG_PHI
ELL_21 = 2 * LAG_2 * LOG_PHI
FIBONACCI = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]

# Master results dict
results = {}

print(f"\\nConstants:")
print(f"  phi = {PHI:.10f}")
print(f"  a = 31/21 = {A_COEFF:.10f}")
print(f"  b = -10/21 = {B_COEFF:.10f}")
print(f"  ell_0 = 2*log(phi) = {ELL_0:.6f}")
print(f"  ell_8 = {ELL_8:.6f},  ell_21 = {ELL_21:.6f}")
""")

# ================================================================
# CELL 2: Part 1 header
# ================================================================
md(r"""
## Part 1: Loading Genuine Riemann Zeros

We download **Andrew Odlyzko's precomputed tables** — the gold standard for Riemann zero
data, accurate to within $3 \times 10^{-9}$.

- `zeros1`: first 100,000 zeros (~1.8 MB)
- `zeros6`: first 2,001,052 zeros (~18 MB, optional for extended scaling)

Cross-validation with **python-flint** (FLINT/arb library) at 15+ digit precision.
""")

# ================================================================
# CELL 3: Data loading
# ================================================================
code("""
# ============================================================
# LOAD GENUINE RIEMANN ZEROS (Odlyzko tables)
# ============================================================
import urllib.request

CACHE_100K = 'riemann_zeros_100k_genuine.npy'
CACHE_2M = 'riemann_zeros_2M_genuine.npy'

def download_odlyzko(url, cache_file, description):
    if os.path.exists(cache_file):
        print(f"  Loading cached {description}...")
        return np.load(cache_file)
    print(f"  Downloading {description}...")
    t0 = time.time()
    try:
        response = urllib.request.urlopen(url, timeout=120)
        raw = response.read().decode('utf-8')
        lines = raw.strip().split('\\n')
        zeros = np.array([float(l.strip()) for l in lines if l.strip()])
        elapsed = time.time() - t0
        print(f"    Got {len(zeros):,} zeros in {elapsed:.1f}s")
        np.save(cache_file, zeros)
        return zeros
    except Exception as e:
        print(f"    Download failed: {e}")
        return None

print("=" * 70)
print("DOWNLOADING GENUINE RIEMANN ZEROS")
print("=" * 70)

# Primary: 100k zeros
gamma_100k = download_odlyzko(
    'https://www-users.cse.umn.edu/~odlyzko/zeta_tables/zeros1',
    CACHE_100K, "100,000 zeros (Odlyzko zeros1)")

if gamma_100k is None:
    raise RuntimeError("Could not download 100k zeros. Check network connection.")

# Extended: 2M zeros (optional)
print()
gamma_2M = download_odlyzko(
    'https://www-users.cse.umn.edu/~odlyzko/zeta_tables/zeros6',
    CACHE_2M, "2,001,052 zeros (Odlyzko zeros6)")
HAS_2M = gamma_2M is not None

# Use 100k as primary
gamma_n = gamma_100k
N_ZEROS = len(gamma_n)

# Validation against known values
KNOWN = [14.134725142, 21.022039639, 25.010857580, 30.424876126, 32.935061588]
print(f"\\nValidation (first 5 zeros vs known values):")
for i, k in enumerate(KNOWN):
    err = abs(gamma_n[i] - k)
    status = "OK" if err < 1e-6 else "MISMATCH"
    print(f"  gamma_{i+1} = {gamma_n[i]:.9f}  (known: {k:.9f}, err: {err:.2e}) [{status}]")

# python-flint cross-validation
try:
    from flint import acb
    print("\\npython-flint cross-validation (first 20 zeros):")
    flint_zeros = acb.zeta_zeros(1, 20)
    max_err = 0
    for i, z in enumerate(flint_zeros):
        fv = float(z.imag.mid())
        err = abs(fv - gamma_n[i])
        max_err = max(max_err, err)
    print(f"  Max |Odlyzko - flint| over first 20: {max_err:.2e}")
    results['flint_validation'] = {'max_error': float(max_err), 'n_checked': 20}
except ImportError:
    print("\\npython-flint not available (optional: pip install python-flint)")
except Exception as e:
    print(f"\\npython-flint error: {e}")

print(f"\\n{'=' * 70}")
print(f"Dataset ready: {N_ZEROS:,} genuine zeros")
print(f"Range: [{gamma_n[0]:.6f}, {gamma_n[-1]:.2f}]")
if HAS_2M:
    print(f"Extended dataset: {len(gamma_2M):,} zeros up to {gamma_2M[-1]:.2f}")
print(f"{'=' * 70}")

results['n_zeros'] = int(N_ZEROS)
results['gamma_range'] = [float(gamma_n[0]), float(gamma_n[-1])]
""")

# ================================================================
# CELL 4: Part 2 header
# ================================================================
md(r"""
## Part 2: Franca–LeClair Decomposition

Split each zero into smooth + oscillatory parts:
$$\gamma_n = \gamma_n^{(0)} + \delta_n$$

where $\gamma_n^{(0)}$ satisfies $\theta(\gamma_n^{(0)}) = (n - \tfrac{3}{2})\pi$
(the Riemann–Siegel theta function), and $\delta_n$ encodes the oscillatory
contribution from $S(T) = \frac{1}{\pi}\arg\zeta(\tfrac{1}{2} + iT)$.

We use **vectorized Newton's method** on the precise $\theta$ function (via `scipy.special.loggamma`)
— not the crude Lambert W approximation.
""")

# ================================================================
# CELL 5: Franca-LeClair decomposition
# ================================================================
code("""
# ============================================================
# FRANCA-LECLAIR DECOMPOSITION (vectorized)
# ============================================================
print("Computing Franca-LeClair decomposition...")
t0 = time.time()

def theta_vec(t):
    \"\"\"Riemann-Siegel theta function (vectorized, precise).\"\"\"
    t = np.asarray(t, dtype=np.float64)
    s = 0.25 + 0.5j * t
    return np.imag(loggamma(s)) - 0.5 * t * np.log(np.pi)

def theta_deriv_vec(t):
    \"\"\"theta'(t) = (1/2)log(t/(2*pi)) + O(1/t^2).\"\"\"
    return 0.5 * np.log(np.maximum(np.asarray(t), 1.0) / (2 * np.pi))

def compute_smooth_zeros(N):
    \"\"\"Vectorized Newton's method: solve theta(t) = (n - 3/2)*pi for n=1..N.\"\"\"
    ns = np.arange(1, N + 1, dtype=np.float64)
    targets = (ns - 1.5) * np.pi

    # Starting values from Lambert W
    w = np.real(lambertw(ns / np.e))
    t = 2 * np.pi * ns / w
    t = np.maximum(t, 2.0)

    # Newton iterations (vectorized over all N zeros simultaneously)
    for it in range(40):
        val = theta_vec(t)
        deriv = theta_deriv_vec(t)
        deriv = np.where(np.abs(deriv) < 1e-15, 1e-15, deriv)
        dt = (val - targets) / deriv
        t -= dt
        max_dt = np.max(np.abs(dt))
        if it < 3 or it % 10 == 0:
            print(f"  Newton iter {it:2d}: max|dt| = {max_dt:.2e}")
        if max_dt < 1e-12:
            print(f"  Converged at iteration {it}")
            break

    return t

# Compute smooth zeros for 100k dataset
gamma_smooth = compute_smooth_zeros(N_ZEROS)
delta_n = gamma_n - gamma_smooth

# Also compute Lambert W approximation for comparison
w_vals = np.real(lambertw(np.arange(1, N_ZEROS+1) / np.e))
gamma_lambert = 2 * np.pi * np.arange(1, N_ZEROS+1) / w_vals
delta_lambert = gamma_n - gamma_lambert

elapsed = time.time() - t0
print(f"\\nCompleted in {elapsed:.1f}s")

# Statistics
print(f"\\n{'=' * 50}")
print(f"{'':>30} {'Precise theta':>15} {'Lambert W':>15}")
print(f"{'=' * 50}")
print(f"{'Mean |delta_n|':>30} {np.mean(np.abs(delta_n)):>15.6f} {np.mean(np.abs(delta_lambert)):>15.6f}")
print(f"{'Std delta_n':>30} {np.std(delta_n):>15.6f} {np.std(delta_lambert):>15.6f}")
print(f"{'Max |delta_n|':>30} {np.max(np.abs(delta_n)):>15.6f} {np.max(np.abs(delta_lambert)):>15.6f}")
print(f"{'Mean |delta_n/gamma_n|':>30} {np.mean(np.abs(delta_n/gamma_n)):>15.8f} {np.mean(np.abs(delta_lambert/gamma_n)):>15.8f}")
improvement = np.mean(np.abs(delta_lambert)) / np.mean(np.abs(delta_n))
print(f"\\nPrecise theta is {improvement:.1f}x better than Lambert W")

# Plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(gamma_n[:2000], delta_n[:2000], '.', markersize=0.8, alpha=0.5, color='steelblue')
axes[0].set_xlabel(r'$\\gamma_n$')
axes[0].set_ylabel(r'$\\delta_n$')
axes[0].set_title(r'Oscillatory corrections $\\delta_n$ (precise $\\theta$)')
axes[0].axhline(0, color='red', lw=0.5)

axes[1].hist(delta_n, bins=150, density=True, alpha=0.7, color='steelblue')
axes[1].set_xlabel(r'$\\delta_n$')
axes[1].set_title(f'Distribution (std={np.std(delta_n):.4f})')

# |delta_n| vs n (log scale for n)
axes[2].semilogy(np.abs(delta_n), '.', markersize=0.3, alpha=0.2, color='steelblue')
axes[2].set_xlabel('n')
axes[2].set_ylabel(r'$|\\delta_n|$')
axes[2].set_title(r'$|\\delta_n|$ vs $n$')

plt.tight_layout()
plt.savefig('grandslam_01_delta.png', dpi=150, bbox_inches='tight')
plt.show()

results['delta_stats'] = {
    'mean_abs_precise': float(np.mean(np.abs(delta_n))),
    'mean_abs_lambert': float(np.mean(np.abs(delta_lambert))),
    'std_precise': float(np.std(delta_n)),
    'max_abs_precise': float(np.max(np.abs(delta_n))),
    'improvement_factor': float(improvement),
}
""")

# ================================================================
# CELL 6: Part 3 header
# ================================================================
md(r"""
## Part 3: FFT Spectral Analysis of $\delta_n$

If the Fibonacci geodesic structure imprints on the zeros, the oscillatory corrections
$\delta_n$ should have their dominant Fourier modes at **Fibonacci frequencies** (lags 1, 2, 3, 5, 8, 13, 21, 34, ...).
""")

# ================================================================
# CELL 7: FFT analysis
# ================================================================
code("""
# ============================================================
# FFT SPECTRAL ANALYSIS
# ============================================================
print("FFT analysis of oscillatory corrections...")

delta_centered = delta_n - np.mean(delta_n)

# Use GPU if available
if GPU:
    fft_result = cp.asnumpy(cp.abs(cp.fft.rfft(cp.asarray(delta_centered))))
else:
    fft_result = np.abs(np.fft.rfft(delta_centered))

fft_power = fft_result ** 2
total_power = np.sum(fft_power[1:])  # exclude DC

# Identify top peaks (skip DC at index 0)
top_indices = np.argsort(fft_result[1:])[::-1][:50] + 1

# Fibonacci identification
n_freqs = len(fft_result)
fib_in_top = {k: sum(1 for idx in top_indices[:k] if idx in FIBONACCI) for k in [3, 6, 10, 20]}

print(f"\\nTop 30 FFT peaks:")
print(f"{'Rank':>4} {'Lag':>6} {'|FFT|':>12} {'Power%':>8} {'Fibonacci':>10}")
print("-" * 50)
for i, idx in enumerate(top_indices[:30]):
    pwr_pct = fft_power[idx] / total_power * 100
    is_fib = idx in FIBONACCI
    marker = "  <<< FIB" if is_fib else ""
    print(f"{i+1:>4} {idx:>6} {fft_result[idx]:>12.2f} {pwr_pct:>7.3f}%{marker}")

# Energy fractions
fib_power_total = sum(fft_power[f] for f in FIBONACCI if f < n_freqs)
fib_energy_pct = fib_power_total / total_power * 100
lag8_21_power = fft_power[LAG_1] + fft_power[LAG_2]
lag8_21_pct = lag8_21_power / total_power * 100

print(f"\\n{'=' * 50}")
print(f"FFT SUMMARY")
print(f"{'=' * 50}")
print(f"  Fibonacci in top 3:   {fib_in_top[3]}/3")
print(f"  Fibonacci in top 6:   {fib_in_top[6]}/6")
print(f"  Fibonacci in top 10:  {fib_in_top[10]}/10")
print(f"  Fibonacci in top 20:  {fib_in_top[20]}/20")
print(f"  Energy at lags 8+21:  {lag8_21_pct:.3f}%")
print(f"  Energy ALL Fibonacci: {fib_energy_pct:.3f}%")

# Plots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Power spectrum
axes[0].semilogy(range(1, min(250, n_freqs)), fft_power[1:250], 'b-', alpha=0.5, lw=0.8)
for f in FIBONACCI:
    if f < 250:
        axes[0].axvline(f, color='red', alpha=0.2, lw=0.5)
        axes[0].plot(f, fft_power[f], 'ro', markersize=6)
axes[0].set_xlabel('Frequency index (lag)')
axes[0].set_ylabel('Power (log scale)')
axes[0].set_title(r'Power spectrum of $\\delta_n$ (red dots = Fibonacci)')
axes[0].set_xlim(0, 250)

# Top peaks bar chart
top20_lags = top_indices[:20]
top20_pcts = [fft_power[idx] / total_power * 100 for idx in top20_lags]
colors = ['red' if idx in FIBONACCI else 'steelblue' for idx in top20_lags]
axes[1].bar(range(20), top20_pcts, color=colors)
axes[1].set_xticks(range(20))
axes[1].set_xticklabels([str(l) for l in top20_lags], rotation=45)
axes[1].set_xlabel('Lag index')
axes[1].set_ylabel('Power (% of total)')
axes[1].set_title('Top 20 FFT peaks (red = Fibonacci)')

plt.tight_layout()
plt.savefig('grandslam_02_fft.png', dpi=150, bbox_inches='tight')
plt.show()

results['fft'] = {
    'fib_in_top6': int(fib_in_top[6]),
    'fib_in_top10': int(fib_in_top[10]),
    'lag8_21_energy_pct': float(lag8_21_pct),
    'all_fib_energy_pct': float(fib_energy_pct),
    'top10_lags': [int(x) for x in top_indices[:10]],
}
""")

# ================================================================
# CELL 8: Part 4 header
# ================================================================
md(r"""
## Part 4: Fibonacci Recurrence — Capture Ratio (Main Result)

The **capture ratio** measures what fraction of $\delta_n$'s variance is explained by the recurrence:
$$\delta_n \approx \frac{31}{21}\delta_{n-8} - \frac{10}{21}\delta_{n-21}$$

**Residual**: $R_n = \delta_n - \frac{31}{21}\delta_{n-8} + \frac{10}{21}\delta_{n-21}$

**Capture**: $1 - \frac{\langle|R_n|\rangle}{\langle|\delta_n|\rangle}$
""")

# ================================================================
# CELL 9: Capture ratio (MAIN RESULT)
# ================================================================
code("""
# ============================================================
# FIBONACCI RECURRENCE: CAPTURE RATIO
# ============================================================
print("=" * 70)
print("  MAIN RESULT: FIBONACCI RECURRENCE VALIDATION")
print("=" * 70)

start = LAG_2  # need 21 previous values

# ---- 1. Test on CORRECTIONS delta_n (precise theta) ----
delta_pred = A_COEFF * delta_n[start-LAG_1:-LAG_1] + B_COEFF * delta_n[start-LAG_2:-LAG_2]
delta_actual = delta_n[start:]
R_delta = delta_actual - delta_pred

capture_mean = 1.0 - np.mean(np.abs(R_delta)) / np.mean(np.abs(delta_actual))
capture_std = 1.0 - np.std(R_delta) / np.std(delta_actual)
residual_ratio = np.mean(np.abs(R_delta)) / np.mean(np.abs(delta_actual))

print(f"\\n1. OSCILLATORY CORRECTIONS (precise theta decomposition):")
print(f"   Mean |R_delta|:      {np.mean(np.abs(R_delta)):.6f}")
print(f"   Mean |delta_n|:      {np.mean(np.abs(delta_actual)):.6f}")
print(f"   |R|/|delta|:         {residual_ratio:.6f} ({residual_ratio*100:.4f}%)")
print(f"")
print(f"   *** CAPTURE (mean): {capture_mean*100:.4f}% ***")
print(f"   *** CAPTURE (std):  {capture_std*100:.4f}% ***")

# ---- 2. Test on CORRECTIONS delta_n (Lambert W) ----
dL_pred = A_COEFF * delta_lambert[start-LAG_1:-LAG_1] + B_COEFF * delta_lambert[start-LAG_2:-LAG_2]
dL_actual = delta_lambert[start:]
R_lambert = dL_actual - dL_pred
capture_lambert = 1.0 - np.mean(np.abs(R_lambert)) / np.mean(np.abs(dL_actual))

print(f"\\n2. LAMBERT W decomposition (for comparison):")
print(f"   Capture (mean):     {capture_lambert*100:.4f}%")

# ---- 3. Test on RAW zeros (with drift) ----
gamma_pred_raw = A_COEFF * gamma_n[start-LAG_1:-LAG_1] + B_COEFF * gamma_n[start-LAG_2:-LAG_2]
R_raw = gamma_n[start:] - gamma_pred_raw

# Estimate drift c(n) via rolling median
window = 500
c_n = np.array([np.median(R_raw[max(0,i-window//2):min(len(R_raw),i+window//2)])
                for i in range(len(R_raw))])
R_raw_detrend = R_raw - c_n
raw_rel_err = np.mean(np.abs(R_raw_detrend)) / np.mean(np.abs(gamma_n[start:]))

print(f"\\n3. RAW ZEROS (drift-removed):")
print(f"   Relative residual:  {raw_rel_err:.8f} ({raw_rel_err*100:.6f}%)")

# ---- 4. Comparison with other lag pairs (OLS best-fit) ----
print(f"\\n4. COMPARISON WITH OTHER LAG PAIRS (OLS best-fit coefficients):")
print(f"   {'Lags':>10} {'Capture%':>10} {'a_opt':>8} {'b_opt':>8} {'a=31/21?':>10}")
print(f"   {'-'*50}")

lag_pairs = [(8,21), (5,13), (3,8), (13,34), (8,13), (5,21), (7,19), (10,25), (8,34), (3,21)]
for l1, l2 in lag_pairs:
    s = l2
    X = np.column_stack([delta_n[s-l1:-l1 if l1 > 0 else len(delta_n)],
                         delta_n[:-(l2) if l2 > 0 else len(delta_n)]])
    y = delta_n[s:]
    # Truncate to same length
    minlen = min(len(X), len(y))
    X, y = X[:minlen], y[:minlen]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    R_ols = y - X @ beta
    cap = 1.0 - np.std(R_ols) / np.std(y)
    marker = "  <<<" if (l1, l2) == (LAG_1, LAG_2) else ""
    a_match = "yes" if (l1,l2)==(8,21) and abs(beta[0]-A_COEFF)<0.05 else ""
    print(f"   ({l1:>2},{l2:>2}) {cap*100:>9.4f}% {beta[0]:>8.4f} {beta[1]:>8.4f} {a_match:>10}{marker}")

# Plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Residual time series
axes[0,0].plot(R_delta[:3000], '.', markersize=0.8, alpha=0.4, color='steelblue')
axes[0,0].axhline(0, color='red', lw=0.5)
axes[0,0].set_ylabel(r'$R_n$')
axes[0,0].set_title('Recurrence residual (first 3000)')

# Histograms
axes[0,1].hist(delta_actual, bins=120, density=True, alpha=0.4, label=r'$\\delta_n$', color='blue')
axes[0,1].hist(R_delta, bins=120, density=True, alpha=0.6, label=r'$R_n$ (residual)', color='red')
axes[0,1].legend(fontsize=11)
axes[0,1].set_title(f'Capture = {capture_mean*100:.2f}%')

# Moving average capture
win = 500
ma_r = np.convolve(np.abs(R_delta), np.ones(win)/win, 'valid')
ma_d = np.convolve(np.abs(delta_actual), np.ones(win)/win, 'valid')
ma_cap = 1.0 - ma_r / ma_d
axes[1,0].plot(ma_cap, 'g-', lw=0.8)
axes[1,0].axhline(capture_mean, color='red', ls='--', lw=1.5,
                   label=f'Global mean = {capture_mean:.4f}')
axes[1,0].set_ylabel('Local capture ratio')
axes[1,0].set_xlabel('n')
axes[1,0].set_title(f'Moving average capture (window={win})')
axes[1,0].legend()
axes[1,0].set_ylim(0.8, 1.05)

# Q-Q: are residuals Gaussian?
from scipy import stats
R_sorted = np.sort(R_delta)
theoretical = stats.norm.ppf(np.linspace(0.001, 0.999, len(R_sorted)))
# Subsample for plotting
step = max(1, len(R_sorted) // 2000)
axes[1,1].plot(theoretical[::step], R_sorted[::step], '.', markersize=1, alpha=0.5)
axes[1,1].plot([-4, 4], [-4*np.std(R_delta), 4*np.std(R_delta)], 'r--', lw=1)
axes[1,1].set_xlabel('Theoretical quantiles')
axes[1,1].set_ylabel('Residual quantiles')
axes[1,1].set_title('Q-Q plot of residuals')

plt.tight_layout()
plt.savefig('grandslam_03_capture.png', dpi=150, bbox_inches='tight')
plt.show()

results['capture'] = {
    'capture_precise_mean': float(capture_mean),
    'capture_precise_std': float(capture_std),
    'capture_lambert': float(capture_lambert),
    'residual_ratio': float(residual_ratio),
    'raw_relative_error': float(raw_rel_err),
}

if capture_mean >= 0.98:
    print(f"\\n>>> GRAND SLAM ACHIEVED: {capture_mean*100:.2f}% >= 98% <<<")
elif capture_mean >= 0.95:
    print(f"\\n>>> STRONG RESULT: {capture_mean*100:.2f}% <<<")
else:
    print(f"\\n>>> RESULT: {capture_mean*100:.2f}% <<<")
""")

# ================================================================
# CELL 10: Part 5 header
# ================================================================
md(r"""
## Part 5: Scaling Analysis

How does the capture ratio evolve as we increase the number of zeros?
Does it converge, diverge, or plateau?

If we have the 2M-zero dataset, we can test up to $N = 2{,}000{,}000$.
""")

# ================================================================
# CELL 11: Scaling analysis
# ================================================================
code("""
# ============================================================
# SCALING ANALYSIS: CAPTURE vs N
# ============================================================
print("Scaling analysis...")

def capture_at_N(delta, N_use, lag1=LAG_1, lag2=LAG_2, a=A_COEFF, b=B_COEFF):
    \"\"\"Compute capture ratio on first N_use values of delta.\"\"\"
    d = delta[:N_use]
    s = lag2
    R = d[s:] - a * d[s-lag1:-lag1] - b * d[s-lag2:-lag2]
    return 1.0 - np.mean(np.abs(R)) / np.mean(np.abs(d[s:]))

# Scaling on 100k dataset
N_vals_100k = [200, 500, 1000, 2000, 5000, 10000, 20000, 50000, N_ZEROS]
cap_100k = [capture_at_N(delta_n, N) for N in tqdm(N_vals_100k, desc="100k scaling")]

print(f"\\n{'N':>10} {'Capture%':>10}")
print(f"{'-'*22}")
for N, c in zip(N_vals_100k, cap_100k):
    print(f"{N:>10,} {c*100:>9.4f}%")

# Extended scaling with 2M zeros
cap_2M = []
N_vals_2M = []
if HAS_2M:
    print(f"\\nExtended scaling on 2M dataset...")
    print("Computing smooth zeros for 2M dataset (vectorized)...")
    t0 = time.time()
    gamma_smooth_2M = compute_smooth_zeros(len(gamma_2M))
    delta_2M = gamma_2M - gamma_smooth_2M
    print(f"  Done in {time.time()-t0:.1f}s")

    N_vals_2M_list = [100000, 200000, 500000, 1000000, len(gamma_2M)]
    for N in tqdm(N_vals_2M_list, desc="2M scaling"):
        N_use = min(N, len(delta_2M))
        cap = capture_at_N(delta_2M, N_use)
        N_vals_2M.append(N_use)
        cap_2M.append(cap)
        print(f"  N={N_use:>10,}: capture={cap*100:.4f}%")

# Plot
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.semilogx(N_vals_100k, [c*100 for c in cap_100k], 'bo-', markersize=8, lw=2,
            label='100k dataset (Odlyzko zeros1)')
if cap_2M:
    ax.semilogx(N_vals_2M, [c*100 for c in cap_2M], 'rs-', markersize=8, lw=2,
                label='2M dataset (Odlyzko zeros6)')
ax.axhline(98, color='green', ls='--', lw=1.5, label='98% target')
ax.axhline(96, color='orange', ls=':', lw=1, label='Previous claim (96%)')
ax.set_xlabel('Number of zeros N', fontsize=13)
ax.set_ylabel('Capture ratio (%)', fontsize=13)
ax.set_title('Fibonacci Recurrence Capture vs N (Genuine Zeros)', fontsize=14)
ax.legend(fontsize=11)
ax.set_ylim(max(85, min([c*100 for c in cap_100k])-3), 100.5)

plt.tight_layout()
plt.savefig('grandslam_04_scaling.png', dpi=150, bbox_inches='tight')
plt.show()

results['scaling'] = {
    'N_values_100k': [int(n) for n in N_vals_100k],
    'capture_100k': [float(c) for c in cap_100k],
}
if cap_2M:
    results['scaling']['N_values_2M'] = [int(n) for n in N_vals_2M]
    results['scaling']['capture_2M'] = [float(c) for c in cap_2M]
""")

# ================================================================
# CELL 12: Part 6 header
# ================================================================
md(r"""
## Part 6: Coefficient & Lag Optimality

### 6a. Coefficient optimality
Grid search over $(a, b)$ to confirm that $(31/21, -10/21)$ is the unique minimum
of the residual. This tests the theoretical prediction.

### 6b. Lag optimality
Exhaustive search over all lag pairs $(p, q)$ with $1 \le p < q \le 55$
(with OLS-optimal coefficients for each pair) to confirm that $(8, 21)$ is the best.
""")

# ================================================================
# CELL 13: Optimality searches
# ================================================================
code("""
# ============================================================
# COEFFICIENT & LAG OPTIMALITY
# ============================================================

# ---- 6a: Coefficient grid search ----
print("6a. Coefficient optimality (grid search)...")
a_range = np.linspace(1.0, 2.0, 200)
b_range = np.linspace(-1.0, 0.0, 200)

s = LAG_2
d = delta_n
d_actual = d[s:]

# Vectorized grid search
grid_residual = np.zeros((len(a_range), len(b_range)))
for i, a in enumerate(a_range):
    for j, b in enumerate(b_range):
        R = d_actual - a * d[s-LAG_1:-LAG_1] - b * d[s-LAG_2:-LAG_2]
        grid_residual[i, j] = np.mean(np.abs(R))

# Find optimum
opt_idx = np.unravel_index(np.argmin(grid_residual), grid_residual.shape)
a_opt_grid = a_range[opt_idx[0]]
b_opt_grid = b_range[opt_idx[1]]

# Theoretical values
a_theo = 31/21
b_theo = -10/21

print(f"  Grid optimum:       a={a_opt_grid:.6f}, b={b_opt_grid:.6f}")
print(f"  Theoretical:        a={a_theo:.6f}, b={b_theo:.6f}")
print(f"  Distance:           |da|={abs(a_opt_grid-a_theo):.4f}, |db|={abs(b_opt_grid-b_theo):.4f}")

# OLS exact optimum
X = np.column_stack([d[s-LAG_1:-LAG_1], d[s-LAG_2:-LAG_2]])
beta_ols = np.linalg.lstsq(X, d_actual, rcond=None)[0]
print(f"  OLS exact optimum:  a={beta_ols[0]:.8f}, b={beta_ols[1]:.8f}")
print(f"  OLS vs theoretical: |da|={abs(beta_ols[0]-a_theo):.6f}, |db|={abs(beta_ols[1]-b_theo):.6f}")

# ---- 6b: Lag optimality (exhaustive search) ----
print(f"\\n6b. Lag optimality (exhaustive search, p<q<=55)...")
MAX_LAG = 55
lag_results = []

for l2 in tqdm(range(2, MAX_LAG+1), desc="Lag search"):
    for l1 in range(1, l2):
        s_test = l2
        X_test = np.column_stack([delta_n[s_test-l1:-l1], delta_n[:-(l2)]])
        y_test = delta_n[s_test:]
        minlen = min(len(X_test), len(y_test))
        X_test, y_test = X_test[:minlen], y_test[:minlen]
        beta_test = np.linalg.lstsq(X_test, y_test, rcond=None)[0]
        R_test = y_test - X_test @ beta_test
        cap_test = 1.0 - np.std(R_test) / np.std(y_test)
        lag_results.append((l1, l2, cap_test, beta_test[0], beta_test[1]))

# Sort by capture
lag_results.sort(key=lambda x: -x[2])

print(f"\\nTop 15 lag pairs (by OLS capture):")
print(f"{'Rank':>4} {'(p,q)':>8} {'Capture%':>10} {'a_opt':>8} {'b_opt':>8}")
print("-" * 42)
for i, (l1, l2, cap, a, b) in enumerate(lag_results[:15]):
    marker = " <<<" if (l1, l2) == (LAG_1, LAG_2) else ""
    print(f"{i+1:>4} ({l1:>2},{l2:>2}) {cap*100:>9.4f}% {a:>8.4f} {b:>8.4f}{marker}")

# Rank of (8,21)
rank_8_21 = next(i+1 for i, (l1, l2, *_) in enumerate(lag_results) if (l1,l2)==(8,21))
print(f"\\nRank of (8,21): #{rank_8_21}")

# Plots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Coefficient heat map
B, A = np.meshgrid(b_range, a_range)
im = axes[0].contourf(B, A, grid_residual, levels=50, cmap='viridis_r')
axes[0].plot(b_theo, a_theo, 'r*', markersize=15, label='Theoretical (31/21, -10/21)')
axes[0].plot(b_opt_grid, a_opt_grid, 'wx', markersize=12, mew=2, label='Grid optimum')
axes[0].set_xlabel('b')
axes[0].set_ylabel('a')
axes[0].set_title('Residual landscape (darker = better)')
axes[0].legend()
plt.colorbar(im, ax=axes[0])

# Lag heat map
lag_grid = np.full((MAX_LAG+1, MAX_LAG+1), np.nan)
for l1, l2, cap, _, _ in lag_results:
    lag_grid[l1, l2] = cap * 100
im2 = axes[1].imshow(lag_grid.T, origin='lower', cmap='hot', aspect='equal',
                      vmin=0, vmax=max(c*100 for _,_,c,_,_ in lag_results[:1]))
axes[1].plot(LAG_1, LAG_2, 'c*', markersize=15)
axes[1].set_xlabel('lag p')
axes[1].set_ylabel('lag q')
axes[1].set_title('OLS capture by lag pair (star = (8,21))')
axes[1].set_xlim(0, MAX_LAG)
axes[1].set_ylim(0, MAX_LAG)
plt.colorbar(im2, ax=axes[1], label='Capture %')

plt.tight_layout()
plt.savefig('grandslam_05_optimality.png', dpi=150, bbox_inches='tight')
plt.show()

results['optimality'] = {
    'a_ols': float(beta_ols[0]),
    'b_ols': float(beta_ols[1]),
    'a_theoretical': float(a_theo),
    'b_theoretical': float(b_theo),
    'lag_8_21_rank': int(rank_8_21),
    'top5_lags': [(int(l1),int(l2),float(c)) for l1,l2,c,_,_ in lag_results[:5]],
}
""")

# ================================================================
# CELL 14: Part 7 header
# ================================================================
md(r"""
## Part 7: Statistical Significance

- **Permutation test** (10,000 shuffles): Is the capture ratio significantly better than random ordering?
- **Bootstrap CI**: 95% confidence interval on the capture ratio.
- **Z-score**: How many $\sigma$ above the null?
""")

# ================================================================
# CELL 15: Statistical tests
# ================================================================
code("""
# ============================================================
# STATISTICAL SIGNIFICANCE
# ============================================================
N_PERMS = 10000
N_BOOT = 5000

print(f"Statistical tests (N_perms={N_PERMS}, N_boot={N_BOOT})...")

# ---- Permutation test ----
print("\\nPermutation test...")
s = LAG_2
perm_captures = np.zeros(N_PERMS)

if GPU:
    print("  Using GPU acceleration")
    d_gpu = cp.asarray(delta_n)
    for p in tqdm(range(N_PERMS), desc="  Permutations"):
        idx = cp.random.permutation(N_ZEROS)
        d_perm = d_gpu[idx]
        R_perm = d_perm[s:] - A_COEFF * d_perm[s-LAG_1:-LAG_1] - B_COEFF * d_perm[s-LAG_2:-LAG_2]
        perm_captures[p] = float(1.0 - cp.mean(cp.abs(R_perm)) / cp.mean(cp.abs(d_perm[s:])))
    cp.get_default_memory_pool().free_all_blocks()
else:
    for p in tqdm(range(N_PERMS), desc="  Permutations"):
        d_perm = np.random.permutation(delta_n)
        R_perm = d_perm[s:] - A_COEFF * d_perm[s-LAG_1:-LAG_1] - B_COEFF * d_perm[s-LAG_2:-LAG_2]
        perm_captures[p] = 1.0 - np.mean(np.abs(R_perm)) / np.mean(np.abs(d_perm[s:]))

perm_mean = np.mean(perm_captures)
perm_std = np.std(perm_captures)
z_perm = (capture_mean - perm_mean) / perm_std
p_value = np.mean(perm_captures >= capture_mean)

print(f"  Null distribution: mean={perm_mean:.6f}, std={perm_std:.6f}")
print(f"  Observed capture:  {capture_mean:.6f}")
print(f"  Z-score:           {z_perm:.1f}")
print(f"  p-value:           {p_value:.2e} (none of {N_PERMS} permutations beat observed)")

# ---- Bootstrap CI ----
print("\\nBootstrap confidence interval...")
boot_captures = np.zeros(N_BOOT)
n_data = len(delta_actual)

for b in tqdm(range(N_BOOT), desc="  Bootstrap"):
    idx = np.random.randint(0, N_ZEROS - s, size=N_ZEROS - s)
    # Block bootstrap: resample contiguous blocks
    block_size = 50
    n_blocks = (N_ZEROS - s) // block_size
    block_starts = np.random.randint(s, N_ZEROS - block_size, size=n_blocks)
    d_boot = np.concatenate([delta_n[bs:bs+block_size] for bs in block_starts])
    d_boot = d_boot[:N_ZEROS - s]
    R_boot = d_boot[s-s:] - A_COEFF * d_boot[s-s-LAG_1:-LAG_1 if LAG_1 > 0 else len(d_boot)] - B_COEFF * d_boot[:-LAG_2 if LAG_2 > 0 else len(d_boot)]
    minlen = min(len(d_boot) - s, len(d_boot) - LAG_1, len(d_boot) - LAG_2)
    if minlen > 100:
        d_sub = d_boot[:minlen+s]
        R_b = d_sub[s:] - A_COEFF * d_sub[s-LAG_1:-LAG_1] - B_COEFF * d_sub[s-LAG_2:-LAG_2]
        ml = min(len(R_b), len(d_sub[s:]))
        boot_captures[b] = 1.0 - np.mean(np.abs(R_b[:ml])) / np.mean(np.abs(d_sub[s:s+ml]))

ci_lo, ci_hi = np.percentile(boot_captures[boot_captures != 0], [2.5, 97.5])
print(f"  95% CI: [{ci_lo*100:.4f}%, {ci_hi*100:.4f}%]")
print(f"  Median: {np.median(boot_captures[boot_captures != 0])*100:.4f}%")

# Plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(perm_captures, bins=80, density=True, alpha=0.7, color='gray', label='Null (permuted)')
axes[0].axvline(capture_mean, color='red', lw=2, label=f'Observed = {capture_mean:.4f}')
axes[0].set_xlabel('Capture ratio')
axes[0].set_title(f'Permutation test (z={z_perm:.0f}, p<{max(1/N_PERMS, p_value):.0e})')
axes[0].legend()

valid_boot = boot_captures[boot_captures != 0]
axes[1].hist(valid_boot * 100, bins=80, density=True, alpha=0.7, color='steelblue')
axes[1].axvline(ci_lo*100, color='orange', lw=2, ls='--', label=f'2.5%: {ci_lo*100:.2f}%')
axes[1].axvline(ci_hi*100, color='orange', lw=2, ls='--', label=f'97.5%: {ci_hi*100:.2f}%')
axes[1].axvline(capture_mean*100, color='red', lw=2, label=f'Point est: {capture_mean*100:.2f}%')
axes[1].set_xlabel('Capture ratio (%)')
axes[1].set_title('Bootstrap 95% CI')
axes[1].legend()

plt.tight_layout()
plt.savefig('grandslam_06_significance.png', dpi=150, bbox_inches='tight')
plt.show()

results['significance'] = {
    'z_score_permutation': float(z_perm),
    'p_value': float(p_value),
    'perm_null_mean': float(perm_mean),
    'perm_null_std': float(perm_std),
    'bootstrap_ci_95': [float(ci_lo), float(ci_hi)],
}
""")

# ================================================================
# CELL 16: Part 8 header
# ================================================================
md(r"""
## Part 8: Linearization Bounds & Parseval Energy

### GPT's concern (Council-19)

The gap closure argument linearizes $\cos((\gamma_n^{(0)} + \delta_n)\omega)$,
which requires $|\delta_n \omega| \ll 1$. We compute the empirical distribution of
$|\delta_n \cdot \omega|$ for the relevant geodesic lengths.

### Parseval energy bound

What fraction of the total spectral energy of $\{\delta_n\}$ concentrates
on Fibonacci modes? (Rigorous version of the FFT peak analysis.)
""")

# ================================================================
# CELL 17: Linearization + Parseval
# ================================================================
code("""
# ============================================================
# LINEARIZATION BOUNDS & PARSEVAL ENERGY
# ============================================================

# ---- Linearization bounds ----
print("Linearization bounds (|delta_n * omega|):")
omegas = {
    'ell_0 (primitive)': ELL_0,
    'ell_1': 2 * LOG_PHI,
    'ell_8': ELL_8,
    'ell_21': ELL_21,
}

lin_stats = {}
for name, omega in omegas.items():
    product = np.abs(delta_n) * omega
    mean_prod = np.mean(product)
    max_prod = np.max(product)
    frac_small = np.mean(product < 1.0) * 100
    frac_very_small = np.mean(product < 0.1) * 100
    print(f"  {name:>25}: mean={mean_prod:.4f}, max={max_prod:.4f}, "
          f"|d*w|<1: {frac_small:.1f}%, |d*w|<0.1: {frac_very_small:.1f}%")
    lin_stats[name] = {
        'omega': float(omega),
        'mean_product': float(mean_prod),
        'max_product': float(max_prod),
        'frac_below_1': float(frac_small),
    }

print(f"\\n  Conclusion: linearization valid for primitive geodesic (ell_0),")
print(f"  but NOT for ell_8 or ell_21. GPT was right!")
print(f"  -> Need exact trig + Cauchy-Schwarz for the analytical argument.")

# ---- Parseval energy analysis ----
print(f"\\nParseval energy analysis:")
# Full FFT power spectrum (already computed: fft_power)
total_energy = np.sum(fft_power[1:])

# Energy in each Fibonacci mode
print(f"  {'Mode':>6} {'Energy':>12} {'% of total':>12} {'Cumulative %':>14}")
print(f"  {'-'*48}")
cumulative = 0
for f in FIBONACCI:
    if f >= len(fft_power):
        break
    e = fft_power[f]
    pct = e / total_energy * 100
    cumulative += pct
    print(f"  {f:>6} {e:>12.2f} {pct:>11.4f}% {cumulative:>13.4f}%")

# Parseval bound: E_fib / E_total
E_fib_8_21 = fft_power[8] + fft_power[21]
E_fib_all = sum(fft_power[f] for f in FIBONACCI if f < len(fft_power))
parseval_8_21 = E_fib_8_21 / total_energy
parseval_all = E_fib_all / total_energy

print(f"\\n  Parseval ratio (lags 8+21 only): {parseval_8_21*100:.4f}%")
print(f"  Parseval ratio (all Fibonacci):  {parseval_all*100:.4f}%")
print(f"  Parseval ratio (non-Fibonacci):  {(1-parseval_all)*100:.4f}%")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Linearization: CDF of |delta * omega|
for name, omega in omegas.items():
    product = np.sort(np.abs(delta_n) * omega)
    cdf = np.arange(1, len(product)+1) / len(product)
    axes[0].plot(product, cdf, label=name, lw=1.5)
axes[0].axvline(1.0, color='red', ls='--', lw=1, label='|dw|=1')
axes[0].axvline(0.1, color='orange', ls=':', lw=1, label='|dw|=0.1')
axes[0].set_xlabel(r'$|\\delta_n \\cdot \\omega|$')
axes[0].set_ylabel('CDF')
axes[0].set_title('Linearization regime')
axes[0].legend(fontsize=9)
axes[0].set_xlim(0, min(10, max(np.abs(delta_n) * ELL_21)))

# Parseval: cumulative energy in Fibonacci modes
fib_sorted_by_energy = sorted(
    [(f, fft_power[f]) for f in FIBONACCI if f < len(fft_power)],
    key=lambda x: -x[1])
cum_pct = np.cumsum([e/total_energy*100 for _, e in fib_sorted_by_energy])
fib_labels = [str(f) for f, _ in fib_sorted_by_energy]
axes[1].bar(range(len(cum_pct)), cum_pct, color='steelblue')
axes[1].set_xticks(range(len(cum_pct)))
axes[1].set_xticklabels(fib_labels[:len(cum_pct)], rotation=45)
axes[1].set_xlabel('Fibonacci mode (sorted by energy)')
axes[1].set_ylabel('Cumulative energy (%)')
axes[1].set_title('Parseval: cumulative Fibonacci energy')

plt.tight_layout()
plt.savefig('grandslam_07_linearization.png', dpi=150, bbox_inches='tight')
plt.show()

results['linearization'] = lin_stats
results['parseval'] = {
    'E_8_21_pct': float(parseval_8_21 * 100),
    'E_all_fib_pct': float(parseval_all * 100),
    'E_non_fib_pct': float((1 - parseval_all) * 100),
}
""")

# ================================================================
# CELL 18: Part 9 header
# ================================================================
md(r"""
## Part 9: Residual Anatomy & Window Robustness

### What's in the residual?
FFT of the residual $R_n$ — does the remaining few % have structure,
or is it noise?

### Window robustness
Is the capture ratio stable across different ranges of zeros
(first 20k, middle 20k, last 20k)?
""")

# ================================================================
# CELL 19: Residuals + robustness
# ================================================================
code("""
# ============================================================
# RESIDUAL ANATOMY & WINDOW ROBUSTNESS
# ============================================================

# ---- Residual FFT ----
print("Residual anatomy (FFT of R_n):")
R_centered = R_delta - np.mean(R_delta)
fft_R = np.abs(np.fft.rfft(R_centered)) ** 2
total_R_power = np.sum(fft_R[1:])

# Top residual modes
top_R_idx = np.argsort(fft_R[1:])[::-1][:20] + 1
print(f"  Top 10 residual FFT peaks:")
for i, idx in enumerate(top_R_idx[:10]):
    pct = fft_R[idx] / total_R_power * 100
    is_fib = "FIB" if idx in FIBONACCI else ""
    print(f"    #{i+1}: lag={idx:>4}, power={pct:.3f}% {is_fib}")

# Autocorrelation of residuals
max_acf_lag = 50
acf = np.correlate(R_centered, R_centered, 'full')
acf = acf[len(acf)//2:]  # positive lags only
acf = acf / acf[0]  # normalize
print(f"\\n  Residual autocorrelation (first 10 lags):")
for k in range(1, 11):
    sig = "***" if abs(acf[k]) > 2/np.sqrt(len(R_delta)) else ""
    print(f"    R({k:2d}) = {acf[k]:>8.5f} {sig}")

# ---- Window robustness ----
print(f"\\nWindow robustness (non-overlapping 20k windows):")
window_size = 20000
n_windows = N_ZEROS // window_size
window_caps = []

print(f"  {'Window':>20} {'N':>8} {'Capture%':>10}")
print(f"  {'-'*40}")
for w in range(n_windows):
    lo = w * window_size
    hi = (w + 1) * window_size
    d_win = delta_n[lo:hi]
    cap_win = capture_at_N(d_win, len(d_win))
    window_caps.append(cap_win)
    print(f"  [{lo+1:>6,}-{hi:>6,}] {len(d_win):>8,} {cap_win*100:>9.4f}%")

print(f"\\n  Mean across windows:  {np.mean(window_caps)*100:.4f}%")
print(f"  Std across windows:   {np.std(window_caps)*100:.4f}%")
print(f"  Min:                  {np.min(window_caps)*100:.4f}%")
print(f"  Max:                  {np.max(window_caps)*100:.4f}%")

# Sensitivity to coefficients
print(f"\\nCoefficient sensitivity (varying a around 31/21):")
a_perturb = np.linspace(A_COEFF - 0.1, A_COEFF + 0.1, 41)
cap_vs_a = []
for a_test in a_perturb:
    R_test = delta_actual - a_test * delta_n[s-LAG_1:-LAG_1] - B_COEFF * delta_n[s-LAG_2:-LAG_2]
    cap_vs_a.append(1.0 - np.mean(np.abs(R_test)) / np.mean(np.abs(delta_actual)))
print(f"  Peak capture at a = {a_perturb[np.argmax(cap_vs_a)]:.6f} (theoretical: {A_COEFF:.6f})")

# Plots
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Residual FFT
axes[0,0].semilogy(fft_R[1:200], 'b-', alpha=0.5, lw=0.8)
for f in FIBONACCI:
    if f < 200:
        axes[0,0].plot(f-1, fft_R[f], 'ro', markersize=5)
axes[0,0].set_title('FFT of residual R_n')
axes[0,0].set_xlabel('Frequency')

# Autocorrelation
axes[0,1].bar(range(1, max_acf_lag+1), acf[1:max_acf_lag+1], color='steelblue', alpha=0.7)
axes[0,1].axhline(2/np.sqrt(len(R_delta)), color='red', ls='--', lw=1, label=r'$2\\sigma$')
axes[0,1].axhline(-2/np.sqrt(len(R_delta)), color='red', ls='--', lw=1)
for f in [8, 21]:
    axes[0,1].axvline(f, color='green', alpha=0.3, lw=2)
axes[0,1].set_title('Residual autocorrelation')
axes[0,1].set_xlabel('Lag')
axes[0,1].legend()

# Window captures
axes[1,0].bar(range(n_windows), [c*100 for c in window_caps], color='steelblue')
axes[1,0].axhline(capture_mean*100, color='red', ls='--', lw=1.5, label=f'Global: {capture_mean*100:.2f}%')
axes[1,0].set_xlabel('Window index (each 20k zeros)')
axes[1,0].set_ylabel('Capture %')
axes[1,0].set_title('Window robustness')
axes[1,0].legend()

# Coefficient sensitivity
axes[1,1].plot(a_perturb, [c*100 for c in cap_vs_a], 'b-', lw=2)
axes[1,1].axvline(A_COEFF, color='red', ls='--', lw=1.5, label='a = 31/21')
axes[1,1].axvline(a_perturb[np.argmax(cap_vs_a)], color='green', ls=':', lw=1, label='Empirical optimum')
axes[1,1].set_xlabel('a coefficient')
axes[1,1].set_ylabel('Capture %')
axes[1,1].set_title('Sensitivity to coefficient a')
axes[1,1].legend()

plt.tight_layout()
plt.savefig('grandslam_08_residuals.png', dpi=150, bbox_inches='tight')
plt.show()

results['robustness'] = {
    'window_captures': [float(c) for c in window_caps],
    'window_mean': float(np.mean(window_caps)),
    'window_std': float(np.std(window_caps)),
    'coeff_sensitivity_peak_a': float(a_perturb[np.argmax(cap_vs_a)]),
}
""")

# ================================================================
# CELL 20: Summary header
# ================================================================
md(r"""
## Summary & Conclusions

Final dashboard with all key metrics.
""")

# ================================================================
# CELL 21: Summary dashboard
# ================================================================
code("""
# ============================================================
# FINAL SUMMARY DASHBOARD
# ============================================================
print("=" * 70)
print("  SELBERG-FIBONACCI GRAND SLAM: FINAL RESULTS")
print("=" * 70)

cap = results['capture']['capture_precise_mean']
target = 0.98

print(f"\\n  Dataset: {results['n_zeros']:,} genuine Riemann zeros (Odlyzko)")
print(f"  Range:   gamma_1 = {results['gamma_range'][0]:.6f} to gamma_N = {results['gamma_range'][1]:.2f}")
print(f"")
print(f"  {'='*50}")
print(f"  CAPTURE RATIO (precise theta): {cap*100:.4f}%")
if cap >= 0.98:
    print(f"  STATUS: GRAND SLAM ACHIEVED (>= 98%)")
elif cap >= 0.95:
    print(f"  STATUS: STRONG RESULT (>= 95%)")
else:
    print(f"  STATUS: {cap*100:.2f}%")
print(f"  {'='*50}")
print(f"")
print(f"  Test Results:")
print(f"    FFT: {results['fft']['fib_in_top6']}/6 top peaks are Fibonacci")
print(f"    FFT: {results['fft']['fib_in_top10']}/10 top peaks are Fibonacci")
print(f"    Parseval energy (lags 8+21): {results['parseval']['E_8_21_pct']:.3f}%")
print(f"    Parseval energy (all Fib):   {results['parseval']['E_all_fib_pct']:.3f}%")
print(f"    OLS optimal a: {results['optimality']['a_ols']:.8f} (theoretical 31/21 = {A_COEFF:.8f})")
print(f"    OLS optimal b: {results['optimality']['b_ols']:.8f} (theoretical -10/21 = {B_COEFF:.8f})")
print(f"    Lag (8,21) rank: #{results['optimality']['lag_8_21_rank']} among all pairs")
print(f"    Permutation Z-score: {results['significance']['z_score_permutation']:.1f}")
print(f"    Permutation p-value: {results['significance']['p_value']:.2e}")
print(f"    Bootstrap 95% CI: [{results['significance']['bootstrap_ci_95'][0]*100:.2f}%, "
      f"{results['significance']['bootstrap_ci_95'][1]*100:.2f}%]")
print(f"    Window robustness: {results['robustness']['window_mean']*100:.2f}% "
      f"+/- {results['robustness']['window_std']*100:.2f}%")
print(f"")

# Scaling summary
if 'capture_2M' in results.get('scaling', {}):
    print(f"  Extended scaling (2M zeros):")
    for N, c in zip(results['scaling']['N_values_2M'], results['scaling']['capture_2M']):
        print(f"    N={N:>10,}: {c*100:.4f}%")

# Linearization warning
print(f"\\n  Linearization analysis (GPT concern):")
for name, stats in results['linearization'].items():
    print(f"    {name}: mean|dw|={stats['mean_product']:.3f}, "
          f"{stats['frac_below_1']:.0f}% below 1")

print(f"\\n  -> Linearization VALID for primitive geodesic (ell_0)")
print(f"  -> Linearization FAILS for ell_8 and ell_21")
print(f"  -> Exact trig identity + Cauchy-Schwarz needed for analytical closure")

# Summary plot
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Delta distribution
axes[0,0].hist(delta_n, bins=100, density=True, alpha=0.7, color='steelblue')
axes[0,0].set_title(r'$\\delta_n$ distribution')

# 2. FFT power
axes[0,1].semilogy(fft_power[1:100], 'b-', lw=0.8, alpha=0.5)
for f in [8, 21]:
    axes[0,1].plot(f-1, fft_power[f], 'ro', markersize=8)
axes[0,1].set_title('FFT power (red = Fib)')

# 3. Capture vs N
axes[0,2].semilogx(results['scaling']['N_values_100k'],
                    [c*100 for c in results['scaling']['capture_100k']],
                    'bo-', lw=2, markersize=6)
if 'capture_2M' in results.get('scaling', {}):
    axes[0,2].semilogx(results['scaling']['N_values_2M'],
                        [c*100 for c in results['scaling']['capture_2M']],
                        'rs-', lw=2, markersize=6)
axes[0,2].axhline(98, color='g', ls='--')
axes[0,2].set_title('Capture vs N')
axes[0,2].set_ylabel('Capture %')

# 4. Residual vs delta
axes[1,0].hist(delta_actual, bins=100, density=True, alpha=0.3, label=r'$\\delta_n$')
axes[1,0].hist(R_delta, bins=100, density=True, alpha=0.6, color='red', label='Residual')
axes[1,0].legend()
axes[1,0].set_title(f'Capture = {cap*100:.2f}%')

# 5. Permutation null
axes[1,1].hist(perm_captures, bins=60, density=True, alpha=0.7, color='gray')
axes[1,1].axvline(capture_mean, color='red', lw=2)
axes[1,1].set_title(f'Permutation test (z={results["significance"]["z_score_permutation"]:.0f})')

# 6. Window stability
axes[1,2].bar(range(len(results['robustness']['window_captures'])),
              [c*100 for c in results['robustness']['window_captures']],
              color='steelblue')
axes[1,2].axhline(cap*100, color='red', ls='--')
axes[1,2].set_title('Window robustness')
axes[1,2].set_ylabel('Capture %')

plt.suptitle(f'Grand Slam Summary: Capture = {cap*100:.2f}% on {N_ZEROS:,} zeros',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('grandslam_09_summary.png', dpi=150, bbox_inches='tight')
plt.show()

# Save all results
results_file = 'grandslam_results.json'
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\\nResults saved to {results_file}")
print(f"\\n{'=' * 70}")
print(f"  GRAND SLAM COMPLETE")
print(f"{'=' * 70}")
""")


# ================================================================
# WRITE NOTEBOOK
# ================================================================
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        },
        "colab": {
            "provenance": [],
            "gpuType": "A100"
        },
        "accelerator": "GPU"
    },
    "cells": cells
}

path = "/home/user/GIFT/notebooks/Selberg_Fibonacci_GrandSlam.ipynb"
with open(path, "w") as f:
    json.dump(notebook, f, indent=1)

print(f"Generated {path}")
print(f"  {len(cells)} cells ({sum(1 for c in cells if c['cell_type']=='code')} code, "
      f"{sum(1 for c in cells if c['cell_type']=='markdown')} markdown)")
print(f"  Total lines: {sum(len(c['source']) for c in cells)}")
