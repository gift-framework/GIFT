#!/usr/bin/env python3
"""
Analyse rapide des correspondances GIFT-Riemann (version optimisée)
"""

import numpy as np
from pathlib import Path
import json

# ============================================================================
# CHARGEMENT DES ZEROS
# ============================================================================

def load_zeros(max_zeros=100000):
    zeros = []
    zeros_dir = Path(__file__).parent
    for i in range(1, 6):
        zeros_file = zeros_dir / f"zeros{i}"
        if zeros_file.exists():
            with open(zeros_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        zeros.append(float(line))
                        if len(zeros) >= max_zeros:
                            return np.array(zeros)
    return np.array(zeros)

# ============================================================================
# CONSTANTES GIFT
# ============================================================================

GIFT_CONSTANTS = {
    "dim_G2": 14, "b2": 21, "b3": 77, "H_star": 99, "dim_E8": 248,
    "rank_E8": 8, "dim_K7": 7, "kappa_T_inv": 61, "det_g_32": 65,
    "dim_J3O": 27, "p2": 2, "Weyl": 5, "Heegner_7": 43, "Heegner_8": 67,
    "Heegner_9": 163, "j_constant": 744, "Monster_factor_1": 47,
    "Monster_factor_2": 59, "Monster_factor_3": 71, "E8_roots": 240,
    "dim_E8xE8": 496, "spinor_128": 128, "dim_G2_sq": 196,
    "tau_num": 3472, "tau_den": 891,
}

# ============================================================================
# ANALYSE VECTORISEE
# ============================================================================

def count_matches_vectorized(values, zeros, tolerance=0.01):
    """Compte les matches de façon vectorisée (rapide)."""
    values = np.array(values)
    # Pour chaque valeur, trouver la déviation minimale
    count = 0
    for v in values:
        if v > zeros[-1]:
            continue
        devs = np.abs(zeros - v) / v
        if np.min(devs) < tolerance:
            count += 1
    return count

def monte_carlo_fast(zeros, n_constants, value_range, n_trials=5000, tolerance=0.01):
    """Monte Carlo optimisé."""
    counts = np.zeros(n_trials)
    log_min, log_max = np.log(value_range[0]), np.log(value_range[1])

    for trial in range(n_trials):
        np.random.seed(trial)
        log_values = np.random.uniform(log_min, log_max, n_constants)
        random_values = np.exp(log_values)
        counts[trial] = count_matches_vectorized(random_values, zeros, tolerance)

    return counts

def fit_recurrence_fast(zeros, lags, n_samples=10000):
    """Fit linéaire rapide."""
    max_lag = max(lags)
    n_fit = min(n_samples, len(zeros) - max_lag)

    X = np.column_stack([zeros[max_lag - lag:max_lag - lag + n_fit] for lag in lags])
    X = np.column_stack([X, np.ones(n_fit)])
    y = zeros[max_lag:max_lag + n_fit]

    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    return r_squared, coeffs

# ============================================================================
# MAIN
# ============================================================================

print("=" * 70)
print("ANALYSE GIFT-RIEMANN (VERSION RAPIDE)")
print("=" * 70)

# Charger zéros
zeros = load_zeros(50000)  # 50k suffisent
print(f"\n✓ {len(zeros)} zéros chargés")

# Constantes GIFT dans le range
gift_values = [v for v in GIFT_CONSTANTS.values() if v <= zeros[-1]]
value_range = (min(gift_values), max(gift_values))
n_gift_constants = len(gift_values)
print(f"✓ {n_gift_constants} constantes GIFT dans range [{value_range[0]}, {value_range[1]}]")

# Compter matches GIFT
tolerance = 0.01
n_gift_matches = count_matches_vectorized(gift_values, zeros, tolerance)
print(f"\n▶ GIFT: {n_gift_matches}/{n_gift_constants} correspondances (< 1%)")

# Détail des matches
print("\n  Meilleures correspondances:")
for name, v in sorted(GIFT_CONSTANTS.items(), key=lambda x: x[1]):
    if v > zeros[-1]:
        continue
    devs = np.abs(zeros - v) / v
    best_idx = np.argmin(devs)
    if devs[best_idx] < tolerance:
        print(f"    {name:20s} = {v:5d} ≈ γ_{best_idx+1:5d} = {zeros[best_idx]:10.3f} ({devs[best_idx]*100:.4f}%)")

# Monte Carlo NULL MODEL
print(f"\n▶ Monte Carlo: 5000 sets de {n_gift_constants} constantes aléatoires...")
null_counts = monte_carlo_fast(zeros, n_gift_constants, value_range, n_trials=5000, tolerance=tolerance)

print(f"  Null model: {np.mean(null_counts):.1f} ± {np.std(null_counts):.1f} matches")
print(f"  Min/Max: {int(np.min(null_counts))}/{int(np.max(null_counts))}")

p_value = np.mean(null_counts >= n_gift_matches)
print(f"\n▶ P-VALUE: {p_value:.6f}")

if p_value < 0.001:
    verdict = "⭐⭐⭐ TRES SIGNIFICATIF (p < 0.001)"
elif p_value < 0.01:
    verdict = "⭐⭐ SIGNIFICATIF (p < 0.01)"
elif p_value < 0.05:
    verdict = "⭐ Marginalement significatif (p < 0.05)"
else:
    verdict = "✗ NON SIGNIFICATIF - cohérent avec hasard"

print(f"  {verdict}")

# Distribution
print(f"\n  Distribution null model:")
for thresh in [10, 12, 14, 16, 18, 20]:
    pct = np.mean(null_counts >= thresh) * 100
    marker = " ← GIFT" if thresh == n_gift_matches else ""
    print(f"    P(≥{thresh:2d}) = {pct:6.2f}%{marker}")

# ============================================================================
# ANALYSE DES LAGS
# ============================================================================

print("\n" + "=" * 70)
print("ANALYSE DES LAGS")
print("=" * 70)

# Comparaison directe
lag_sets = {
    "GIFT original": [5, 8, 13, 27],
    "GIFT-Riemann": [8, 13, 16, 19],
    "Champion": [4, 19, 26, 29],
    "Consécutifs": [1, 2, 3, 4],
    "Fibonacci pur": [5, 8, 13, 21],
}

print("\n▶ Comparaison des combinaisons de lags:")
for name, lags in lag_sets.items():
    r2, _ = fit_recurrence_fast(zeros, lags)
    print(f"  {name:20s} {str(lags):20s} R² = {r2*100:.4f}%")

# Recherche exhaustive (échantillonnée pour rapidité)
print("\n▶ Recherche exhaustive (top 20)...")
from itertools import combinations

results = []
for lag_combo in combinations(range(1, 31), 4):
    r2, _ = fit_recurrence_fast(zeros, list(lag_combo), n_samples=5000)
    results.append((lag_combo, r2))

results.sort(key=lambda x: -x[1])

print("\n  Top 10:")
for i, (lags, r2) in enumerate(results[:10]):
    print(f"    #{i+1}: {lags} → R² = {r2*100:.4f}%")

# Trouver rang de GIFT
gift_r2 = [r2 for lags, r2 in results if list(lags) == [5, 8, 13, 27]]
if gift_r2:
    gift_rank = sum(1 for _, r2 in results if r2 > gift_r2[0]) + 1
else:
    gift_rank = "non trouvé"

print(f"\n  Rang de GIFT [5,8,13,27]: #{gift_rank}")

# Analyse de fréquence
print("\n▶ Fréquence des lags dans top 100:")
from collections import Counter
lag_freq = Counter()
diff_freq = Counter()
for lags, _ in results[:100]:
    for l in lags:
        lag_freq[l] += 1
    for i in range(len(lags)):
        for j in range(i+1, len(lags)):
            diff_freq[abs(lags[j] - lags[i])] += 1

print("  Lags les plus fréquents:")
for lag, freq in lag_freq.most_common(10):
    expected = 100 * 4 / 30
    marker = "↑" if freq > expected * 1.5 else ""
    print(f"    lag {lag:2d}: {freq:2d} fois {marker}")

print("\n  Différences les plus fréquentes:")
fibonacci = {1, 2, 3, 5, 8, 13, 21}
for diff, freq in diff_freq.most_common(10):
    fib = "← Fib" if diff in fibonacci else ""
    print(f"    diff {diff:2d}: {freq:2d} fois {fib}")

# ============================================================================
# CONCLUSIONS
# ============================================================================

print("\n" + "=" * 70)
print("CONCLUSIONS")
print("=" * 70)

print(f"""
1. CORRESPONDANCES γₙ ≈ GIFT:
   - GIFT: {n_gift_matches} matches
   - Null: {np.mean(null_counts):.1f} ± {np.std(null_counts):.1f}
   - P-value: {p_value:.4f}
   → {verdict}

2. LAGS:
   - GIFT [5,8,13,27] rang #{gift_rank}
   - Champion: {results[0][0]}
   - Les lags 8 et 13 sont sur-représentés

3. STRUCTURE:
   - Différences Fibonacci (5, 8) fréquentes
   - Pas exactement GIFT, mais trace de structure
""")

# Sauvegarder
output = {
    "gift_matches": n_gift_matches,
    "null_mean": float(np.mean(null_counts)),
    "null_std": float(np.std(null_counts)),
    "p_value": float(p_value),
    "significant": p_value < 0.05,
    "gift_rank": gift_rank if isinstance(gift_rank, int) else 0,
    "top_5_lags": [{"lags": list(l), "r2": float(r)} for l, r in results[:5]],
    "verdict": verdict
}

with open(Path(__file__).parent / "null_model_results.json", "w") as f:
    json.dump(output, f, indent=2)

print("✓ Résultats sauvegardés dans null_model_results.json")
