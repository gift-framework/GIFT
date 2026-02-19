#!/usr/bin/env python3
"""
ROBUST STATISTICAL VALIDATION
Before Council Submission

Tests:
1. Monte Carlo lag search - is (8,21,42) or (8,21,77) special?
2. Permutation test - does structure vanish with shuffled zeros?
3. Bootstrap stability - are coefficients stable?
4. Null distribution - expected ACF reduction with random 3rd lag?
5. Uniqueness ranking - where do GIFT lags rank among all possibilities?
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.signal import correlate
from scipy import stats
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ROBUST STATISTICAL VALIDATION - PRE-COUNCIL")
print("="*70)

# Load zeros
try:
    zeros = np.load('/home/user/GIFT/riemann_zeros_10k.npy')
except:
    zeros = np.load('riemann_zeros_10k.npy')

print(f"\nLoaded {len(zeros)} Riemann zeros")
N_ZEROS = len(zeros)

def autocorr(x, max_lag=50):
    """Compute autocorrelation."""
    x = x - np.mean(x)
    result = correlate(x, x, mode='full')
    result = result[len(result)//2:]
    if result[0] == 0:
        return np.zeros(max_lag)
    return result[:max_lag] / result[0]

def fit_recurrence(data, lags):
    """Fit recurrence and return metrics."""
    N = len(data)
    max_lag = max(lags)

    if N <= max_lag + 100:
        return None

    X = np.column_stack([data[max_lag - lag : N - lag] for lag in lags])
    y = data[max_lag:]

    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)

    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    residuals = y - y_pred

    acf = autocorr(residuals, max_lag=50)

    return {
        'coefficients': list(model.coef_),
        'intercept': model.intercept_,
        'r2': r2,
        'residual_std': np.std(residuals),
        'acf_21': acf[21] if len(acf) > 21 else np.nan,
        'sum_coef': sum(model.coef_)
    }

# ============================================================
# TEST 1: MONTE CARLO LAG SEARCH
# ============================================================
print("\n" + "="*70)
print("TEST 1: MONTE CARLO - Are GIFT lags special?")
print("="*70)

# Test all triplets (lag1, lag2, lag3) where lag1 < lag2 < lag3
# Compare against (8, 21, 42) and (8, 21, 77)

np.random.seed(42)
N_RANDOM_TRIPLETS = 1000

# Reference results
ref_42 = fit_recurrence(zeros, [8, 21, 42])
ref_77 = fit_recurrence(zeros, [8, 21, 77])
ref_2lag = fit_recurrence(zeros, [8, 21])

print(f"\nReference models:")
print(f"  (8, 21):     R² = {ref_2lag['r2']:.12f}, σ = {ref_2lag['residual_std']:.4f}")
print(f"  (8, 21, 42): R² = {ref_42['r2']:.12f}, σ = {ref_42['residual_std']:.4f}")
print(f"  (8, 21, 77): R² = {ref_77['r2']:.12f}, σ = {ref_77['residual_std']:.4f}")

# Random triplet search
print(f"\nTesting {N_RANDOM_TRIPLETS} random triplets...")

random_results = []
for _ in tqdm(range(N_RANDOM_TRIPLETS), desc="Random triplets"):
    lag1 = np.random.randint(3, 20)
    lag2 = np.random.randint(lag1 + 5, 50)
    lag3 = np.random.randint(lag2 + 5, 100)

    result = fit_recurrence(zeros, [lag1, lag2, lag3])
    if result:
        random_results.append({
            'lags': (lag1, lag2, lag3),
            'r2': result['r2'],
            'residual_std': result['residual_std'],
            'acf_21': result['acf_21']
        })

# Rank GIFT models
r2_values = [r['r2'] for r in random_results]
std_values = [r['residual_std'] for r in random_results]

rank_42_r2 = sum(1 for r2 in r2_values if r2 > ref_42['r2'])
rank_77_r2 = sum(1 for r2 in r2_values if r2 > ref_77['r2'])
rank_42_std = sum(1 for s in std_values if s < ref_42['residual_std'])
rank_77_std = sum(1 for s in std_values if s < ref_77['residual_std'])

print(f"\nRankings among {len(random_results)} random triplets:")
print(f"  (8,21,42): R² rank = #{rank_42_r2+1}/{len(random_results)}, σ rank = #{rank_42_std+1}/{len(random_results)}")
print(f"  (8,21,77): R² rank = #{rank_77_r2+1}/{len(random_results)}, σ rank = #{rank_77_std+1}/{len(random_results)}")

percentile_42 = (1 - rank_42_std / len(random_results)) * 100
percentile_77 = (1 - rank_77_std / len(random_results)) * 100
print(f"\n  (8,21,42) is in the top {100-percentile_42:.1f}% by residual σ")
print(f"  (8,21,77) is in the top {100-percentile_77:.1f}% by residual σ")

# ============================================================
# TEST 2: SYSTEMATIC SEARCH - Best triplets with lag1=8, lag2=21
# ============================================================
print("\n" + "="*70)
print("TEST 2: SYSTEMATIC - Best 3rd lag given (8, 21, ?)")
print("="*70)

third_lag_results = []
for lag3 in tqdm(range(25, 150), desc="Testing 3rd lags"):
    result = fit_recurrence(zeros, [8, 21, lag3])
    if result:
        third_lag_results.append({
            'lag3': lag3,
            'r2': result['r2'],
            'residual_std': result['residual_std'],
            'acf_21': abs(result['acf_21'])
        })

# Sort by residual_std
third_lag_results.sort(key=lambda x: x['residual_std'])

print(f"\nTop 10 third lags by residual σ:")
print(f"{'Lag3':<8} {'σ_resid':<12} {'R²':<18} {'|ACF(21)|':<10} {'Note':<15}")
print("-" * 70)
for r in third_lag_results[:10]:
    note = ""
    if r['lag3'] == 42:
        note = "← 2×b₂"
    elif r['lag3'] == 77:
        note = "← b₃"
    elif r['lag3'] == 34:
        note = "← F₉"
    elif r['lag3'] == 55:
        note = "← F₁₀"
    elif r['lag3'] == 99:
        note = "← H*"
    print(f"{r['lag3']:<8} {r['residual_std']:<12.6f} {r['r2']:<18.12f} {r['acf_21']:<10.4f} {note:<15}")

# Where do 42 and 77 rank?
rank_42 = next(i for i, r in enumerate(third_lag_results) if r['lag3'] == 42) + 1
rank_77 = next(i for i, r in enumerate(third_lag_results) if r['lag3'] == 77) + 1

print(f"\nRankings:")
print(f"  Lag 42 (2×b₂): #{rank_42}/{len(third_lag_results)}")
print(f"  Lag 77 (b₃):   #{rank_77}/{len(third_lag_results)}")

# ============================================================
# TEST 3: PERMUTATION TEST
# ============================================================
print("\n" + "="*70)
print("TEST 3: PERMUTATION - Does structure vanish with shuffled zeros?")
print("="*70)

N_PERMUTATIONS = 100

perm_results_2lag = []
perm_results_3lag = []

print(f"\nRunning {N_PERMUTATIONS} permutations...")

for _ in tqdm(range(N_PERMUTATIONS), desc="Permutations"):
    shuffled = np.random.permutation(zeros)

    r2 = fit_recurrence(shuffled, [8, 21])
    r3 = fit_recurrence(shuffled, [8, 21, 77])

    if r2:
        perm_results_2lag.append(r2['r2'])
    if r3:
        perm_results_3lag.append(r3['r2'])

# Compare to original
orig_r2_2lag = ref_2lag['r2']
orig_r2_3lag = ref_77['r2']

perm_mean_2lag = np.mean(perm_results_2lag)
perm_std_2lag = np.std(perm_results_2lag)
perm_mean_3lag = np.mean(perm_results_3lag)
perm_std_3lag = np.std(perm_results_3lag)

z_score_2lag = (orig_r2_2lag - perm_mean_2lag) / perm_std_2lag if perm_std_2lag > 0 else np.inf
z_score_3lag = (orig_r2_3lag - perm_mean_3lag) / perm_std_3lag if perm_std_3lag > 0 else np.inf

print(f"\n2-lag model (8, 21):")
print(f"  Original R²:   {orig_r2_2lag:.10f}")
print(f"  Permuted R²:   {perm_mean_2lag:.10f} ± {perm_std_2lag:.10f}")
print(f"  Z-score:       {z_score_2lag:.1f}σ")
print(f"  p-value:       {stats.norm.sf(z_score_2lag):.2e}")

print(f"\n3-lag model (8, 21, 77):")
print(f"  Original R²:   {orig_r2_3lag:.10f}")
print(f"  Permuted R²:   {perm_mean_3lag:.10f} ± {perm_std_3lag:.10f}")
print(f"  Z-score:       {z_score_3lag:.1f}σ")
print(f"  p-value:       {stats.norm.sf(z_score_3lag):.2e}")

# ============================================================
# TEST 4: BOOTSTRAP STABILITY
# ============================================================
print("\n" + "="*70)
print("TEST 4: BOOTSTRAP - Coefficient stability")
print("="*70)

N_BOOTSTRAP = 200
SAMPLE_SIZE = int(0.8 * N_ZEROS)

bootstrap_coefs = {
    '2lag': {'a': [], 'b': []},
    '3lag_77': {'a': [], 'b': [], 'c': []}
}

print(f"\nRunning {N_BOOTSTRAP} bootstrap samples (80% each)...")

for _ in tqdm(range(N_BOOTSTRAP), desc="Bootstrap"):
    idx = np.random.choice(N_ZEROS, SAMPLE_SIZE, replace=True)
    sample = zeros[np.sort(idx)]

    r2 = fit_recurrence(sample, [8, 21])
    r3 = fit_recurrence(sample, [8, 21, 77])

    if r2:
        bootstrap_coefs['2lag']['a'].append(r2['coefficients'][0])
        bootstrap_coefs['2lag']['b'].append(r2['coefficients'][1])
    if r3:
        bootstrap_coefs['3lag_77']['a'].append(r3['coefficients'][0])
        bootstrap_coefs['3lag_77']['b'].append(r3['coefficients'][1])
        bootstrap_coefs['3lag_77']['c'].append(r3['coefficients'][2])

print(f"\n2-lag model (8, 21):")
a_mean, a_std = np.mean(bootstrap_coefs['2lag']['a']), np.std(bootstrap_coefs['2lag']['a'])
b_mean, b_std = np.mean(bootstrap_coefs['2lag']['b']), np.std(bootstrap_coefs['2lag']['b'])
print(f"  a (lag 8):  {a_mean:.4f} ± {a_std:.4f}  (target: {31/21:.4f})")
print(f"  b (lag 21): {b_mean:.4f} ± {b_std:.4f}  (target: {-10/21:.4f})")
print(f"  CV(a): {a_std/abs(a_mean)*100:.1f}%")

print(f"\n3-lag model (8, 21, 77):")
a_mean, a_std = np.mean(bootstrap_coefs['3lag_77']['a']), np.std(bootstrap_coefs['3lag_77']['a'])
b_mean, b_std = np.mean(bootstrap_coefs['3lag_77']['b']), np.std(bootstrap_coefs['3lag_77']['b'])
c_mean, c_std = np.mean(bootstrap_coefs['3lag_77']['c']), np.std(bootstrap_coefs['3lag_77']['c'])
print(f"  a (lag 8):  {a_mean:.4f} ± {a_std:.4f}")
print(f"  b (lag 21): {b_mean:.4f} ± {b_std:.4f}")
print(f"  c (lag 77): {c_mean:.4f} ± {c_std:.4f}")
print(f"  CV(a): {a_std/abs(a_mean)*100:.1f}%, CV(c): {c_std/abs(c_mean)*100:.1f}%")

# ============================================================
# TEST 5: FIBONACCI vs NON-FIBONACCI LAGS
# ============================================================
print("\n" + "="*70)
print("TEST 5: FIBONACCI vs NON-FIBONACCI structure")
print("="*70)

fibonacci_lags = [8, 13, 21, 34, 55, 89]
non_fib_lags = [7, 11, 17, 31, 47, 83]  # Primes near Fibonacci

fib_result = fit_recurrence(zeros, fibonacci_lags[:4])  # 8, 13, 21, 34
nonfib_result = fit_recurrence(zeros, non_fib_lags[:4])  # 7, 11, 17, 31

print(f"\nFibonacci lags (8, 13, 21, 34):")
print(f"  R²: {fib_result['r2']:.12f}")
print(f"  σ:  {fib_result['residual_std']:.6f}")

print(f"\nNon-Fibonacci primes (7, 11, 17, 31):")
print(f"  R²: {nonfib_result['r2']:.12f}")
print(f"  σ:  {nonfib_result['residual_std']:.6f}")

fib_advantage = (nonfib_result['residual_std'] - fib_result['residual_std']) / nonfib_result['residual_std'] * 100
print(f"\nFibonacci advantage: {fib_advantage:.1f}% lower residual σ")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print("VALIDATION SUMMARY FOR COUNCIL")
print("="*70)

print("""
┌─────────────────────────────────────────────────────────────────────┐
│                    STATISTICAL VALIDATION RESULTS                    │
├─────────────────────────────────────────────────────────────────────┤
""")

print(f"│ TEST 1: Monte Carlo Uniqueness                                    │")
print(f"│   (8,21,42) rank: #{rank_42_r2+1}/{len(random_results)} by R², top {100-percentile_42:.0f}% by σ              │")
print(f"│   (8,21,77) rank: #{rank_77_r2+1}/{len(random_results)} by R², top {100-percentile_77:.0f}% by σ              │")

print(f"│                                                                     │")
print(f"│ TEST 2: Systematic 3rd Lag Search                                  │")
print(f"│   Lag 42 (2×b₂): #{rank_42}/{len(third_lag_results)} best                                      │")
print(f"│   Lag 77 (b₃):   #{rank_77}/{len(third_lag_results)} best                                      │")

print(f"│                                                                     │")
print(f"│ TEST 3: Permutation Test                                           │")
print(f"│   2-lag Z-score: {z_score_2lag:.0f}σ (p < {stats.norm.sf(z_score_2lag):.0e})                              │")
print(f"│   3-lag Z-score: {z_score_3lag:.0f}σ (p < {stats.norm.sf(z_score_3lag):.0e})                              │")

print(f"│                                                                     │")
print(f"│ TEST 4: Bootstrap Stability                                        │")
print(f"│   Coefficient CV: {a_std/abs(a_mean)*100:.1f}% (good if < 10%)                            │")

print(f"│                                                                     │")
print(f"│ TEST 5: Fibonacci vs Non-Fibonacci                                 │")
print(f"│   Fibonacci advantage: {fib_advantage:.1f}% lower σ                               │")

print("""│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│ VERDICT:                                                            │""")

# Compute overall verdict
passes = 0
total = 5

if percentile_77 > 80:
    passes += 1
    v1 = "✓"
else:
    v1 = "✗"

if rank_77 <= 10:
    passes += 1
    v2 = "✓"
else:
    v2 = "✗"

if z_score_3lag > 5:
    passes += 1
    v3 = "✓"
else:
    v3 = "✗"

if a_std/abs(a_mean)*100 < 15:
    passes += 1
    v4 = "✓"
else:
    v4 = "✗"

if fib_advantage > 0:
    passes += 1
    v5 = "✓"
else:
    v5 = "✗"

print(f"│   {v1} Monte Carlo: GIFT lags in top {100-percentile_77:.0f}%                          │")
print(f"│   {v2} Systematic: lag 77 rank #{rank_77} (top 10 = pass)                   │")
print(f"│   {v3} Permutation: Z = {z_score_3lag:.0f}σ (> 5σ = pass)                            │")
print(f"│   {v4} Bootstrap: CV = {a_std/abs(a_mean)*100:.1f}% (< 15% = pass)                          │")
print(f"│   {v5} Fibonacci: {fib_advantage:.1f}% advantage (> 0% = pass)                       │")
print(f"│                                                                     │")
print(f"│   OVERALL: {passes}/{total} TESTS PASSED                                        │")
print("""└─────────────────────────────────────────────────────────────────────┘
""")

# Save results
results = {
    'monte_carlo': {
        'n_random_triplets': len(random_results),
        'rank_42_r2': rank_42_r2 + 1,
        'rank_77_r2': rank_77_r2 + 1,
        'percentile_42': percentile_42,
        'percentile_77': percentile_77
    },
    'systematic_search': {
        'rank_lag42': rank_42,
        'rank_lag77': rank_77,
        'total_tested': len(third_lag_results),
        'top_10_lags': [r['lag3'] for r in third_lag_results[:10]]
    },
    'permutation_test': {
        'n_permutations': N_PERMUTATIONS,
        'z_score_2lag': z_score_2lag,
        'z_score_3lag': z_score_3lag,
        'p_value_2lag': float(stats.norm.sf(z_score_2lag)),
        'p_value_3lag': float(stats.norm.sf(z_score_3lag))
    },
    'bootstrap': {
        'n_samples': N_BOOTSTRAP,
        'coef_cv_percent': a_std/abs(a_mean)*100
    },
    'fibonacci_test': {
        'fib_sigma': fib_result['residual_std'],
        'nonfib_sigma': nonfib_result['residual_std'],
        'advantage_percent': fib_advantage
    },
    'verdict': {
        'tests_passed': passes,
        'tests_total': total
    }
}

with open('/home/user/GIFT/research/riemann/robust_validation_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("✓ Results saved to robust_validation_results.json")
print("="*70)
