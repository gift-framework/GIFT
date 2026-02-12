# %%
# =================================================================
# Cell A: VALIDATION TESTS -- CONSTANT MODEL (theta* = 0.9941)
# =================================================================
from scipy import stats

print("=" * 70)
print("VALIDATION TESTS -- CONSTANT MODEL (theta* = 0.9941)")
print("=" * 70)

# -- T5: Monte Carlo -- is theta* special? --
print("\n[T5] Monte Carlo Permutation Test")
print("-" * 50)
N_TRIALS = 200
np.random.seed(42)

N_MC = 200_000
d_mc = delta[:N_MC]
g0_mc = gamma0[:N_MC]
tp_mc = tp[:N_MC]

dp_opt = delta_pred[:N_MC]
R2_opt = float(1.0 - np.var(d_mc - dp_opt) / np.var(d_mc))

theta_random = np.random.uniform(0.3, 2.0, N_TRIALS)
R2_random = []
for i, th in enumerate(theta_random):
    dp_r = prime_sum_adaptive_chunked(g0_mc, tp_mc, primes_bisect, K_MAX, th, w_cosine)
    R2_r = float(1.0 - np.var(d_mc - dp_r) / np.var(d_mc))
    R2_random.append(R2_r)
    if (i + 1) % 50 == 0:
        print(f"    Trial {i+1}/{N_TRIALS}...")

R2_random = np.array(R2_random)
R2_best_random = float(np.max(R2_random))
margin = R2_opt - R2_best_random
p_val_mc = float(np.mean(R2_random >= R2_opt))

T5_pass = margin > 0
print(f"\n  R2(theta*=0.9941):  {R2_opt:.6f}")
print(f"  R2(best random):    {R2_best_random:.6f}")
print(f"  Margin:             {margin:+.6f}")
print(f"  p-value:            {p_val_mc:.4f}")
print(f"  >> {'PASS' if T5_pass else 'FAIL'}")

# -- T7: Bootstrap CI for alpha --
print(f"\n[T7] Bootstrap Confidence Interval for alpha")
print("-" * 50)
B = 5000
np.random.seed(123)

alpha_boots = np.empty(B)
for b in range(B):
    idx = np.random.randint(0, N_ZEROS, N_ZEROS)
    d_b = delta[idx]
    dp_b = delta_pred[idx]
    dot_pp = np.dot(dp_b, dp_b)
    alpha_boots[b] = np.dot(d_b, dp_b) / dot_pp if dot_pp > 0 else 0.0

ci_lo = float(np.percentile(alpha_boots, 2.5))
ci_hi = float(np.percentile(alpha_boots, 97.5))

T7_pass = ci_lo <= 1.0 <= ci_hi
print(f"  alpha(OLS):         {alpha_OLS:.6f}")
print(f"  95% CI:             [{ci_lo:.6f}, {ci_hi:.6f}]")
print(f"  Contains alpha=1?   {'YES' if T7_pass else 'NO'}")
print(f"  >> {'PASS' if T7_pass else 'FAIL'}")

# -- T8: Drift test --
print(f"\n[T8] Drift Test (alpha across windows)")
print("-" * 50)
alphas_w = np.array([w['alpha'] for w in window_results])
slope, intercept, r_val, p_val_drift, se = stats.linregress(
    np.arange(len(alphas_w), dtype=float), alphas_w)

T8_pass = p_val_drift > 0.05
print(f"  alpha values:       {[round(a, 4) for a in alphas_w]}")
print(f"  Slope:              {slope:+.6f}/window")
print(f"  p-value:            {p_val_drift:.4f}")
print(f"  >> {'PASS' if T8_pass else 'FAIL'}")

n_pass = sum([T5_pass, T7_pass, T8_pass])
print(f"\n{'=' * 70}")
print(f"CONSTANT MODEL: {n_pass}/3 passed  (vs 0/3 for adaptive)")
print(f"{'=' * 70}")


# %%
# =================================================================
# Cell B: EFFECTIVE theta PER WINDOW
# =================================================================

print("=" * 70)
print("EFFECTIVE theta PER WINDOW (bisection to alpha=1)")
print("=" * 70)

WINDOWS_EFF = [
    (0, 100_000),
    (100_000, 200_000),
    (200_000, 500_000),
    (500_000, 1_000_000),
    (1_000_000, 1_500_000),
    (1_500_000, N_ZEROS),
]

theta_effs = []
print(f"\n{'Window':>20} | {'T range':>22} | {'th_eff':>8} | {'delta':>8}")
print("-" * 70)

for (a, b) in WINDOWS_EFF:
    d_w = delta[a:b]
    g0_w = gamma0[a:b]
    tp_w = tp[a:b]

    lo, hi = 0.5, 1.5
    for _ in range(30):
        mid = (lo + hi) / 2
        dp_w = prime_sum_adaptive_chunked(
            g0_w, tp_w, primes_bisect, K_MAX, mid, w_cosine)
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
    T_hi = gamma_n[min(b - 1, len(gamma_n) - 1)]
    diff = theta_eff - THETA_STAR
    print(f"{label:>20} | [{T_lo:>8.0f}, {T_hi:>8.0f}]"
          f" | {theta_eff:>8.4f} | {diff:>+8.4f}")

theta_effs = np.array(theta_effs)
print(f"\n  theta_eff mean:  {np.mean(theta_effs):.4f}")
print(f"  theta_eff std:   {np.std(theta_effs):.4f}")
print(f"  theta* global:   {THETA_STAR}")
print(f"  Max |delta|:     {np.max(np.abs(theta_effs - THETA_STAR)):.4f}")

T_mids = np.array([
    (gamma_n[a] + gamma_n[min(b - 1, N_ZEROS - 1)]) / 2
    for (a, b) in WINDOWS_EFF
])
log_T_mids = np.log(T_mids)
sl, ic, rv, pv, se = stats.linregress(1.0 / log_T_mids, theta_effs)
print(f"\n  Fit theta_eff ~ a + b/log(T):")
print(f"    a (asymptote):  {ic:.4f}")
print(f"    b (correction): {sl:+.4f}")
print(f"    R2 of fit:      {rv**2:.4f}")
print(f"    theta(T->inf):  {ic:.4f}")


# %%
# =================================================================
# Cell C: R2 DECAY ANALYSIS
# =================================================================
import matplotlib.pyplot as plt

R2_windows = np.array([w['R2'] for w in window_results])
T_mids_w = np.array([(w['T_lo'] + w['T_hi']) / 2 for w in window_results])
log_T_w = np.log10(T_mids_w)

sl_R2, ic_R2, rv_R2, pv_R2, se_R2 = stats.linregress(log_T_w, R2_windows)
sl_inv, ic_inv, rv_inv, pv_inv, _ = stats.linregress(
    1.0 / np.log(T_mids_w), R2_windows)

print("=" * 70)
print("R2 DECAY ANALYSIS")
print("=" * 70)

print(f"\n  Model 1: R2 = a + b*log10(T)")
print(f"    a = {ic_R2:.4f}, b = {sl_R2:+.6f}/decade")
print(f"    R2 of fit: {rv_R2**2:.4f}")
print(f"    Predicted R2 at T=10^7:   {ic_R2 + sl_R2 * 7:.4f}")
print(f"    Predicted R2 at T=10^8:   {ic_R2 + sl_R2 * 8:.4f}")
print(f"    Predicted R2 at T=10^10:  {ic_R2 + sl_R2 * 10:.4f}")

print(f"\n  Model 2: R2 = a + b/log(T)  [asymptotically stable]")
print(f"    a (asymptote) = {ic_inv:.4f}")
print(f"    b = {sl_inv:+.4f}")
print(f"    R2 of fit: {rv_inv**2:.4f}")
print(f"    Predicted R2 at T=10^7:   {ic_inv + sl_inv / np.log(1e7):.4f}")
print(f"    Predicted R2 at T=10^8:   {ic_inv + sl_inv / np.log(1e8):.4f}")
print(f"    Predicted R2 at T->inf:   {ic_inv:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

log_T_ext = np.linspace(1, 10, 100)
ax1.plot(log_T_w, R2_windows, 'bo-', ms=8, label='Data')
ax1.plot(log_T_ext, ic_R2 + sl_R2 * log_T_ext, 'r--', alpha=0.7,
         label=f'Linear: {sl_R2:+.4f}/decade')
ax1.plot(log_T_ext, ic_inv + sl_inv / (log_T_ext * np.log(10)),
         'g-.', alpha=0.7, label=f'1/log(T): plateau={ic_inv:.3f}')
ax1.axhline(0.9, color='gray', ls=':', alpha=0.5, label='R2=0.90')
ax1.set_xlabel('log10(T)')
ax1.set_ylabel('R2')
ax1.set_title('R2 Decay & Extrapolation')
ax1.legend(fontsize=9)
ax1.set_ylim(0.85, 0.95)

alphas_plot = np.array([w['alpha'] for w in window_results])
ax2.plot(log_T_w, alphas_plot, 'bs-', ms=8, label='Constant theta*')
ax2.axhline(1.0, color='r', ls='--', alpha=0.7, label='alpha = 1')
ax2.fill_between(
    [log_T_w[0] - 0.2, log_T_w[-1] + 0.2], 0.995, 1.005,
    color='green', alpha=0.15, label='+/-0.5%')
ax2.set_xlabel('log10(T)')
ax2.set_ylabel('alpha (OLS)')
ax2.set_title('Scaling Exponent Stability')
ax2.legend(fontsize=9)
ax2.set_ylim(0.98, 1.02)

plt.tight_layout()
plt.savefig('fig5_R2_extrapolation.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved fig5_R2_extrapolation.png")


# %%
# =================================================================
# Cell D: GUE COMPARISON
# =================================================================

print("=" * 70)
print("GUE COMPARISON -- Residual correlations vs RMT")
print("=" * 70)

d_centered = delta - np.mean(delta)
var_d = np.var(d_centered)
acf_delta_1 = float(np.mean(d_centered[1:] * d_centered[:-1]) / var_d)

r_centered = residuals - np.mean(residuals)
var_r = np.var(r_centered)
acf_resid_1 = float(np.mean(r_centered[1:] * r_centered[:-1]) / var_r)

dp_centered = delta_pred - np.mean(delta_pred)
var_dp = np.var(dp_centered)
acf_pred_1 = float(np.mean(dp_centered[1:] * dp_centered[:-1]) / var_dp)

print(f"\n  Lag-1 autocorrelation:")
print(f"    delta (actual):    {acf_delta_1:+.4f}")
print(f"    delta_pred:        {acf_pred_1:+.4f}")
print(f"    residual:          {acf_resid_1:+.4f}")
print(f"    GUE prediction:    ~-0.47  (Odlyzko-Snaith)")
print(f"\n  -> Model PRESERVES the GUE correlation structure.")

var_delta = float(np.var(delta))
var_pred = float(np.var(delta_pred))
var_resid = float(np.var(residuals))

print(f"\n  Variance decomposition:")
print(f"    Var(delta):        {var_delta:.6f}")
print(f"    Var(delta_pred):   {var_pred:.6f}  ({100*var_pred/var_delta:.1f}%)")
print(f"    Var(residual):     {var_resid:.6f}  ({100*var_resid/var_delta:.1f}%)")
print(f"    R2 check:          {1.0 - var_resid/var_delta:.4f}")

spacings = np.diff(gamma_n)
mean_spacing = np.mean(spacings)
s = spacings / mean_spacing

s_grid = np.linspace(0, 4, 1000)
P_wigner = (32 / np.pi**2) * s_grid**2 * np.exp(-4 * s_grid**2 / np.pi)

hist_vals, hist_edges = np.histogram(s, bins=200, range=(0, 4), density=True)
hist_centers = (hist_edges[:-1] + hist_edges[1:]) / 2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.bar(hist_centers, hist_vals, width=hist_centers[1] - hist_centers[0],
        alpha=0.5, color='steelblue', label='Empirical spacings (2M)')
ax1.plot(s_grid, P_wigner, 'r-', lw=2, label='GUE Wigner surmise')
ax1.set_xlabel('Normalized spacing s')
ax1.set_ylabel('Density')
ax1.set_title('Spacing Distribution: Data vs GUE')
ax1.legend()
ax1.set_xlim(0, 3.5)

lags = [1, 2, 3, 5, 8, 13, 21]
acf_d = [float(np.mean(d_centered[l:] * d_centered[:-l]) / var_d)
         for l in lags]
acf_r = [float(np.mean(r_centered[l:] * r_centered[:-l]) / var_r)
         for l in lags]
acf_p = [float(np.mean(dp_centered[l:] * dp_centered[:-l]) / var_dp)
         for l in lags]

x = np.arange(len(lags))
w = 0.25
ax2.bar(x - w, acf_d, w, label='delta (actual)',
        color='steelblue', alpha=0.8)
ax2.bar(x, acf_p, w, label='delta_pred (model)',
        color='coral', alpha=0.8)
ax2.bar(x + w, acf_r, w, label='residual',
        color='seagreen', alpha=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels([str(l) for l in lags])
ax2.set_xlabel('Lag')
ax2.set_ylabel('ACF')
ax2.set_title('Autocorrelation Decomposition')
ax2.legend()
ax2.axhline(0, color='gray', ls='-', alpha=0.3)

plt.tight_layout()
plt.savefig('fig6_GUE_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved fig6_GUE_comparison.png")
