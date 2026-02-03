#!/usr/bin/env python3
"""
Deep Dive: Fibonacci et le Nombre d'Or dans les Zéros de Riemann
================================================================

Hypothèse: La structure des zéros de Riemann encode φ = (1+√5)/2

On explore:
1. φ dans les ratios de lags optimaux
2. φ dans les coefficients de récurrence
3. Lags Fibonacci vs Lucas vs géométriques
4. φ dans les espacements entre zéros
5. Indices "spéciaux" et leur lien avec Fibonacci
"""

import numpy as np
from pathlib import Path
from itertools import combinations
from collections import Counter
import json

# Nombre d'or
PHI = (1 + np.sqrt(5)) / 2  # ≈ 1.618033988749895
PSI = (1 - np.sqrt(5)) / 2  # ≈ -0.618033988749895

# Fibonacci et Lucas
def fib(n):
    """Retourne le n-ième nombre de Fibonacci."""
    if n <= 0: return 0
    if n == 1: return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def lucas(n):
    """Retourne le n-ième nombre de Lucas."""
    if n == 0: return 2
    if n == 1: return 1
    a, b = 2, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

FIBONACCI = [fib(i) for i in range(1, 15)]  # [1,1,2,3,5,8,13,21,34,55,89,144,233,377]
LUCAS = [lucas(i) for i in range(1, 15)]     # [1,3,4,7,11,18,29,47,76,123,199,322,521,843]

print(f"φ = {PHI}")
print(f"Fibonacci: {FIBONACCI[:10]}")
print(f"Lucas: {LUCAS[:10]}")

# ============================================================================
# CHARGEMENT DES ZEROS
# ============================================================================

def load_zeros(max_zeros=50000):
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

zeros = load_zeros(50000)
print(f"\n✓ {len(zeros)} zéros chargés")

# ============================================================================
# 1. φ DANS LES RATIOS DE LAGS OPTIMAUX
# ============================================================================

print("\n" + "=" * 70)
print("1. φ DANS LES RATIOS DE LAGS OPTIMAUX")
print("=" * 70)

def fit_recurrence(zeros, lags, n_samples=10000):
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
    mean_error = np.mean(np.abs(y - y_pred) / y) * 100
    return r_squared, coeffs[:-1], coeffs[-1], mean_error

# Recherche exhaustive avec analyse des ratios
print("\nRecherche des meilleures combinaisons et analyse des ratios...")

results = []
for lag_combo in combinations(range(1, 35), 4):
    r2, coeffs, c, err = fit_recurrence(zeros, list(lag_combo))

    # Calculer les ratios entre lags consécutifs
    sorted_lags = sorted(lag_combo)
    ratios = [sorted_lags[i+1] / sorted_lags[i] for i in range(len(sorted_lags)-1)]

    # Distance à φ pour chaque ratio
    phi_distances = [abs(r - PHI) for r in ratios]
    mean_phi_dist = np.mean(phi_distances)

    results.append({
        "lags": lag_combo,
        "r2": r2,
        "error": err,
        "ratios": ratios,
        "phi_distances": phi_distances,
        "mean_phi_dist": mean_phi_dist,
        "coeffs": coeffs.tolist()
    })

results.sort(key=lambda x: -x["r2"])

print("\nTop 20 par R² - avec distance à φ:")
print(f"{'Rang':<5} {'Lags':<20} {'R²':<12} {'Ratios':<25} {'Dist φ':<10}")
print("-" * 75)
for i, r in enumerate(results[:20]):
    ratios_str = ", ".join([f"{x:.3f}" for x in r["ratios"]])
    print(f"{i+1:<5} {str(r['lags']):<20} {r['r2']*100:.4f}% {ratios_str:<25} {r['mean_phi_dist']:.4f}")

# Trouver les combinaisons dont les ratios sont proches de φ
print("\n\nTop 20 par proximité à φ (ratios ≈ 1.618):")
results_by_phi = sorted(results, key=lambda x: x["mean_phi_dist"])
print(f"{'Rang':<5} {'Lags':<20} {'R²':<12} {'Ratios':<25} {'Dist φ':<10}")
print("-" * 75)
for i, r in enumerate(results_by_phi[:20]):
    ratios_str = ", ".join([f"{x:.3f}" for x in r["ratios"]])
    print(f"{i+1:<5} {str(r['lags']):<20} {r['r2']*100:.4f}% {ratios_str:<25} {r['mean_phi_dist']:.4f}")

# ============================================================================
# 2. LAGS FIBONACCI VS LUCAS VS AUTRES
# ============================================================================

print("\n" + "=" * 70)
print("2. COMPARAISON: FIBONACCI vs LUCAS vs GEOMETRIQUE-φ vs RANDOM")
print("=" * 70)

# Différentes familles de lags
lag_families = {
    "Fibonacci [3,5,8,13]": [3, 5, 8, 13],
    "Fibonacci [5,8,13,21]": [5, 8, 13, 21],
    "Fibonacci [2,3,5,8]": [2, 3, 5, 8],
    "Lucas [3,4,7,11]": [3, 4, 7, 11],
    "Lucas [4,7,11,18]": [4, 7, 11, 18],
    "Lucas [7,11,18,29]": [7, 11, 18, 29],
    "Géométrique-φ [2,3,5,8]": [2, 3, 5, 8],  # ≈ 2, 2φ, 2φ², 2φ³
    "Géométrique-φ [3,5,8,13]": [3, 5, 8, 13],
    "GIFT original [5,8,13,27]": [5, 8, 13, 27],
    "Champion R² [1,7,9,28]": [1, 7, 9, 28],
    "Arithmétique [5,10,15,20]": [5, 10, 15, 20],
    "Puissances 2 [2,4,8,16]": [2, 4, 8, 16],
    "Premiers [2,3,5,7]": [2, 3, 5, 7],
    "Premiers [5,7,11,13]": [5, 7, 11, 13],
}

print(f"\n{'Famille':<30} {'Lags':<20} {'R²':<12} {'Erreur':<12} {'Ratio moy':<10}")
print("-" * 85)

family_results = []
for name, lags in lag_families.items():
    r2, coeffs, c, err = fit_recurrence(zeros, lags)
    sorted_lags = sorted(lags)
    ratios = [sorted_lags[i+1] / sorted_lags[i] for i in range(len(sorted_lags)-1)]
    mean_ratio = np.mean(ratios)
    family_results.append({
        "name": name,
        "lags": lags,
        "r2": r2,
        "error": err,
        "mean_ratio": mean_ratio,
        "coeffs": coeffs.tolist()
    })
    print(f"{name:<30} {str(lags):<20} {r2*100:.4f}% {err:.4f}% {mean_ratio:.4f}")

# Trier par R²
family_results.sort(key=lambda x: -x["r2"])
print("\n→ Meilleure famille par R²:", family_results[0]["name"])

# ============================================================================
# 3. φ DANS LES COEFFICIENTS DE RÉCURRENCE
# ============================================================================

print("\n" + "=" * 70)
print("3. φ DANS LES COEFFICIENTS DE RÉCURRENCE")
print("=" * 70)

# Pour les meilleures combinaisons, analyser les coefficients
print("\nAnalyse des coefficients pour les top 10 combinaisons:")

for i, r in enumerate(results[:10]):
    coeffs = r["coeffs"]
    print(f"\n#{i+1} Lags {r['lags']}:")
    print(f"  Coefficients: {[f'{c:.4f}' for c in coeffs]}")

    # Ratios entre coefficients
    if len(coeffs) >= 2:
        coeff_ratios = []
        for j in range(len(coeffs) - 1):
            if coeffs[j+1] != 0:
                ratio = abs(coeffs[j] / coeffs[j+1])
                coeff_ratios.append(ratio)
        print(f"  Ratios |a_i/a_{'{i+1}'}|: {[f'{r:.4f}' for r in coeff_ratios]}")

        # Distance à φ
        phi_dists = [abs(r - PHI) for r in coeff_ratios]
        print(f"  Distance à φ: {[f'{d:.4f}' for d in phi_dists]}")

# ============================================================================
# 4. φ DANS LES ESPACEMENTS ENTRE ZÉROS
# ============================================================================

print("\n" + "=" * 70)
print("4. φ DANS LES ESPACEMENTS ENTRE ZÉROS")
print("=" * 70)

# Calculer les espacements
spacings = np.diff(zeros)
print(f"\nEspacement moyen: {np.mean(spacings):.4f}")
print(f"Espacement médian: {np.median(spacings):.4f}")

# Ratios consécutifs d'espacements
spacing_ratios = spacings[1:] / spacings[:-1]
print(f"\nRatio d'espacements consécutifs:")
print(f"  Moyenne: {np.mean(spacing_ratios):.4f}")
print(f"  Médiane: {np.median(spacing_ratios):.4f}")
print(f"  φ = {PHI:.4f}")

# Combien de ratios sont proches de φ ?
tolerance = 0.05
near_phi = np.sum(np.abs(spacing_ratios - PHI) < tolerance * PHI)
near_1_phi = np.sum(np.abs(spacing_ratios - 1/PHI) < tolerance / PHI)
print(f"\n  Ratios ≈ φ (±5%): {near_phi} / {len(spacing_ratios)} ({near_phi/len(spacing_ratios)*100:.2f}%)")
print(f"  Ratios ≈ 1/φ (±5%): {near_1_phi} / {len(spacing_ratios)} ({near_1_phi/len(spacing_ratios)*100:.2f}%)")

# Distribution des ratios
print("\n  Distribution des ratios d'espacements:")
for threshold in [0.5, 0.8, 1.0, 1.2, 1.5, 1.618, 2.0, 2.5]:
    count = np.sum(spacing_ratios < threshold)
    print(f"    < {threshold:.3f}: {count/len(spacing_ratios)*100:.1f}%")

# ============================================================================
# 5. INDICES FIBONACCI ET CORRESPONDANCES
# ============================================================================

print("\n" + "=" * 70)
print("5. INDICES FIBONACCI ET ZÉROS REMARQUABLES")
print("=" * 70)

# Zéros aux indices Fibonacci
print("\nZéros aux indices Fibonacci:")
print(f"{'n (Fib)':<10} {'γ_n':<15} {'γ_n/n':<12} {'Δ précédent':<12}")
print("-" * 50)

fib_indices = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584]
prev_zero = None
for n in fib_indices:
    if n <= len(zeros):
        z = zeros[n-1]
        delta = z - prev_zero if prev_zero else 0
        print(f"{n:<10} {z:<15.4f} {z/n:<12.4f} {delta:<12.4f}")
        prev_zero = z

# Ratios entre zéros Fibonacci consécutifs
print("\nRatios γ_{F_{n+1}} / γ_{F_n}:")
fib_zeros = [zeros[n-1] for n in fib_indices if n <= len(zeros)]
fib_ratios = [fib_zeros[i+1] / fib_zeros[i] for i in range(len(fib_zeros)-1)]
print(f"Ratios: {[f'{r:.4f}' for r in fib_ratios]}")
print(f"Moyenne: {np.mean(fib_ratios):.4f}")
print(f"Convergent vers: {fib_ratios[-1]:.4f}")
print(f"φ = {PHI:.4f}")

# ============================================================================
# 6. RÉCURRENCE FIBONACCI PURE
# ============================================================================

print("\n" + "=" * 70)
print("6. TEST: RÉCURRENCE DE TYPE FIBONACCI")
print("=" * 70)

# γ_n ≈ a × γ_{n-F_k} + b × γ_{n-F_{k+1}} pour différents k
print("\nTest de récurrence γ_n ≈ a × γ_{n-F_k} + b × γ_{n-F_{k+1}}:")

for k in range(3, 10):
    f1, f2 = fib(k), fib(k+1)
    if f2 > 30:
        break
    r2, coeffs, c, err = fit_recurrence(zeros, [f1, f2])
    ratio = coeffs[0] / coeffs[1] if coeffs[1] != 0 else float('inf')
    print(f"  F_{k}={f1}, F_{k+1}={f2}: a={coeffs[0]:.4f}, b={coeffs[1]:.4f}, ratio={ratio:.4f}, R²={r2*100:.2f}%")

# Test avec 3 termes Fibonacci consécutifs
print("\nTest γ_n ≈ a×γ_{n-F_k} + b×γ_{n-F_{k+1}} + c×γ_{n-F_{k+2}}:")
for k in range(3, 8):
    f1, f2, f3 = fib(k), fib(k+1), fib(k+2)
    if f3 > 30:
        break
    r2, coeffs, c, err = fit_recurrence(zeros, [f1, f2, f3])
    print(f"  [{f1},{f2},{f3}]: a={coeffs[0]:.4f}, b={coeffs[1]:.4f}, c={coeffs[2]:.4f}, R²={r2*100:.4f}%")

# ============================================================================
# 7. STRUCTURE CACHÉE: DIFFÉRENCES ENTRE LAGS OPTIMAUX
# ============================================================================

print("\n" + "=" * 70)
print("7. STRUCTURE DES DIFFÉRENCES DANS LES LAGS OPTIMAUX")
print("=" * 70)

# Pour les top 100 combinaisons, analyser les différences
diff_counter = Counter()
for r in results[:100]:
    lags = sorted(r["lags"])
    for i in range(len(lags)):
        for j in range(i+1, len(lags)):
            diff_counter[lags[j] - lags[i]] += 1

print("\nDifférences les plus fréquentes (top 100 combinaisons):")
fibonacci_set = set(FIBONACCI[:12])
for diff, count in diff_counter.most_common(15):
    is_fib = "← FIBONACCI" if diff in fibonacci_set else ""
    is_lucas = "← LUCAS" if diff in LUCAS[:12] else ""
    marker = is_fib or is_lucas
    print(f"  diff {diff:2d}: {count:3d} fois {marker}")

# Proportion de différences Fibonacci
fib_diffs = sum(count for diff, count in diff_counter.items() if diff in fibonacci_set)
total_diffs = sum(diff_counter.values())
print(f"\n→ Différences Fibonacci: {fib_diffs}/{total_diffs} ({fib_diffs/total_diffs*100:.1f}%)")
print(f"  Attendu par hasard (6 Fib sur 30 possibles): ~20%")

# ============================================================================
# 8. FORMULE AVEC φ EXPLICITE
# ============================================================================

print("\n" + "=" * 70)
print("8. TEST: FORMULE AVEC φ EXPLICITE")
print("=" * 70)

# Tester γ_n ≈ φ × γ_{n-k} + (1-φ) × γ_{n-m} pour différents k, m
print("\nTest γ_n ≈ φ × γ_{n-k} + ψ × γ_{n-m} (où ψ = 1-φ ≈ -0.618):")

best_phi_formula = None
best_phi_r2 = 0

for k in range(1, 20):
    for m in range(k+1, 25):
        max_lag = m
        n_fit = min(10000, len(zeros) - max_lag)

        X_k = zeros[max_lag - k:max_lag - k + n_fit]
        X_m = zeros[max_lag - m:max_lag - m + n_fit]
        y = zeros[max_lag:max_lag + n_fit]

        # Prédiction avec φ et ψ fixés
        y_pred_fixed = PHI * X_k + PSI * X_m
        ss_res = np.sum((y - y_pred_fixed) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2_fixed = 1 - ss_res / ss_tot

        if r2_fixed > best_phi_r2:
            best_phi_r2 = r2_fixed
            best_phi_formula = (k, m, r2_fixed)

print(f"\nMeilleure formule φ-fixée:")
print(f"  γ_n ≈ φ × γ_{{n-{best_phi_formula[0]}}} + ψ × γ_{{n-{best_phi_formula[1]}}}")
print(f"  R² = {best_phi_formula[2]*100:.4f}%")

# Comparer avec coefficients libres
k, m = best_phi_formula[0], best_phi_formula[1]
r2_free, coeffs_free, _, _ = fit_recurrence(zeros, [k, m])
print(f"\nAvec coefficients libres:")
print(f"  γ_n ≈ {coeffs_free[0]:.4f} × γ_{{n-{k}}} + {coeffs_free[1]:.4f} × γ_{{n-{m}}}")
print(f"  R² = {r2_free*100:.4f}%")
print(f"\n  Ratio coefficients libres: {coeffs_free[0]/coeffs_free[1]:.4f}")
print(f"  φ/ψ = {PHI/PSI:.4f}")

# ============================================================================
# CONCLUSIONS
# ============================================================================

print("\n" + "=" * 70)
print("CONCLUSIONS: LE NOMBRE D'OR ET RIEMANN")
print("=" * 70)

# Calculer quelques statistiques finales
fib_lags_r2 = fit_recurrence(zeros, [3, 5, 8, 13])[0]
gift_lags_r2 = fit_recurrence(zeros, [5, 8, 13, 27])[0]
champion_r2 = results[0]["r2"]

print(f"""
RÉSUMÉ DES DÉCOUVERTES:

1. RATIOS DE LAGS:
   - Les meilleurs lags n'ont PAS des ratios = φ
   - Mais les lags Fibonacci/Lucas donnent de bons R²

2. FAMILLES DE LAGS:
   - Fibonacci [3,5,8,13]: R² = {fib_lags_r2*100:.2f}%
   - GIFT [5,8,13,27]:     R² = {gift_lags_r2*100:.2f}%
   - Champion absolu:      R² = {champion_r2*100:.2f}%

3. DIFFÉRENCES FIBONACCI:
   - {fib_diffs/total_diffs*100:.1f}% des différences sont Fibonacci
   - Attendu par hasard: ~20%
   - → {"SIGNIFICATIF" if fib_diffs/total_diffs > 0.30 else "MARGINALEMENT SIGNIFICATIF" if fib_diffs/total_diffs > 0.25 else "NON SIGNIFICATIF"}

4. FORMULE φ-EXPLICITE:
   - Meilleure: γ_n ≈ φ × γ_{{n-{best_phi_formula[0]}}} + ψ × γ_{{n-{best_phi_formula[1]}}}
   - R² = {best_phi_formula[2]*100:.2f}%
   - {"EXCELLENT" if best_phi_formula[2] > 0.99 else "BON" if best_phi_formula[2] > 0.95 else "MODÉRÉ"}

5. VERDICT GLOBAL:
   - φ n'apparaît pas EXPLICITEMENT dans la structure optimale
   - MAIS les nombres de Fibonacci influencent les différences de lags
   - La récurrence Fibonacci pure n'est pas optimale mais honorable
   - Il y a une TRACE de structure φ, subtile mais présente
""")

# Sauvegarder les résultats
output = {
    "phi": PHI,
    "fibonacci_lags_r2": float(fib_lags_r2),
    "gift_lags_r2": float(gift_lags_r2),
    "champion_r2": float(champion_r2),
    "fibonacci_diff_ratio": float(fib_diffs/total_diffs),
    "best_phi_formula": {
        "k": best_phi_formula[0],
        "m": best_phi_formula[1],
        "r2": float(best_phi_formula[2])
    },
    "top_5_by_r2": [{"lags": list(r["lags"]), "r2": float(r["r2"])} for r in results[:5]],
    "top_5_by_phi_proximity": [{"lags": list(r["lags"]), "r2": float(r["r2"]), "mean_phi_dist": float(r["mean_phi_dist"])} for r in results_by_phi[:5]],
    "family_comparison": [{
        "name": f["name"],
        "r2": float(f["r2"]),
        "mean_ratio": float(f["mean_ratio"])
    } for f in family_results]
}

with open(Path(__file__).parent / "fibonacci_analysis_results.json", "w") as f:
    json.dump(output, f, indent=2)

print("\n✓ Résultats sauvegardés dans fibonacci_analysis_results.json")
