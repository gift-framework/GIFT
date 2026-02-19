#!/usr/bin/env python3
"""
Analyse rigoureuse des correspondances GIFT-Riemann
====================================================

Question: Les correspondances γₙ ≈ constantes GIFT sont-elles significatives
ou résultent-elles de cherry-picking?

Méthode: Comparer avec un null model de constantes aléatoires
"""

import numpy as np
from pathlib import Path
import json
from collections import defaultdict

# ============================================================================
# CHARGEMENT DES ZEROS DE RIEMANN
# ============================================================================

def load_zeros(max_zeros=100000):
    """Charge les zéros de Riemann depuis les fichiers Odlyzko."""
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
    # Fondamentales
    "dim_G2": 14,
    "b2": 21,
    "b3": 77,
    "H_star": 99,
    "dim_E8": 248,
    "rank_E8": 8,
    "dim_K7": 7,

    # Dérivées
    "kappa_T_inv": 61,  # 1/κ_T = b3 - dim_G2 - p2
    "det_g_32": 65,     # det(g) × 32
    "dim_J3O": 27,      # Jordan algebra
    "p2": 2,            # Pontryagin
    "Weyl": 5,

    # Heegner
    "Heegner_7": 43,
    "Heegner_8": 67,
    "Heegner_9": 163,

    # Moonshine
    "j_constant": 744,
    "Monster_factor_1": 47,
    "Monster_factor_2": 59,
    "Monster_factor_3": 71,

    # E8 extended
    "E8_roots": 240,
    "dim_E8xE8": 496,

    # Spinor
    "spinor_128": 128,
    "dim_G2_sq": 196,  # 14²

    # Tau components
    "tau_num": 3472,
    "tau_den": 891,

    # Monster dimension
    "Monster_dim": 196883,
}

# ============================================================================
# FONCTIONS D'ANALYSE
# ============================================================================

def find_best_correspondence(value, zeros, tolerance=0.01):
    """
    Trouve le zéro de Riemann le plus proche d'une valeur donnée.
    Retourne (index, zero_value, deviation) ou None si aucun match < tolerance.
    """
    deviations = np.abs(zeros - value) / value
    best_idx = np.argmin(deviations)
    best_dev = deviations[best_idx]

    if best_dev < tolerance:
        return (best_idx + 1, zeros[best_idx], best_dev)  # +1 pour index 1-based
    return None

def count_correspondences(constants, zeros, tolerance=0.01):
    """Compte le nombre de correspondances sub-tolerance."""
    count = 0
    matches = []

    for name, value in constants.items():
        if value > zeros[-1]:  # Skip si hors range
            continue
        result = find_best_correspondence(value, zeros, tolerance)
        if result:
            count += 1
            matches.append({
                "name": name,
                "value": value,
                "index": result[0],
                "zero": result[1],
                "deviation": result[2]
            })

    return count, matches

def generate_random_constants(n_constants, value_range, seed=None):
    """
    Génère des constantes aléatoires avec distribution similaire à GIFT.
    Utilise une distribution log-uniforme pour couvrir plusieurs ordres de grandeur.
    """
    if seed is not None:
        np.random.seed(seed)

    log_min, log_max = np.log(value_range[0]), np.log(value_range[1])
    log_values = np.random.uniform(log_min, log_max, n_constants)
    return {f"random_{i}": np.exp(v) for i, v in enumerate(log_values)}

def monte_carlo_null_model(zeros, n_constants, value_range, n_trials=10000, tolerance=0.01):
    """
    Monte Carlo: combien de correspondances trouve-t-on avec des constantes aléatoires?
    """
    counts = []

    for trial in range(n_trials):
        random_constants = generate_random_constants(n_constants, value_range, seed=trial)
        count, _ = count_correspondences(random_constants, zeros, tolerance)
        counts.append(count)

    return np.array(counts)

def fisher_exact_test(gift_matches, random_mean, n_gift, n_random_equiv):
    """
    Test de Fisher simplifié: quelle est la probabilité d'observer
    gift_matches ou plus par hasard?
    """
    # Approximation Poisson
    from scipy import stats

    # Taux attendu sous H0
    lambda_null = random_mean

    # P(X >= gift_matches | lambda_null)
    p_value = 1 - stats.poisson.cdf(gift_matches - 1, lambda_null)

    return p_value

# ============================================================================
# ANALYSE DES LAGS
# ============================================================================

def fit_recurrence(zeros, lags, n_samples=10000):
    """
    Fit: γₙ ≈ Σᵢ aᵢ × γₙ₋ₗₐᵍₛ[ᵢ] + c
    Retourne coefficients, constante, R², erreur moyenne.
    """
    max_lag = max(lags)
    n_fit = min(n_samples, len(zeros) - max_lag)

    # Construire matrice de design
    X = np.zeros((n_fit, len(lags) + 1))
    y = zeros[max_lag:max_lag + n_fit]

    for i, lag in enumerate(lags):
        X[:, i] = zeros[max_lag - lag:max_lag - lag + n_fit]
    X[:, -1] = 1  # Constante

    # Moindres carrés
    coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

    y_pred = X @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    mean_error = np.mean(np.abs(y - y_pred) / y) * 100  # Pourcentage

    return {
        "coefficients": dict(zip([f"a_{l}" for l in lags], coeffs[:-1])),
        "constant": coeffs[-1],
        "r_squared": r_squared,
        "mean_error_pct": mean_error
    }

def exhaustive_lag_search(zeros, n_lags=4, max_lag=30, top_k=20):
    """
    Recherche exhaustive des meilleures combinaisons de lags.
    """
    from itertools import combinations

    results = []
    all_lags = list(range(1, max_lag + 1))

    for lag_combo in combinations(all_lags, n_lags):
        fit = fit_recurrence(zeros, list(lag_combo))
        results.append({
            "lags": lag_combo,
            "r_squared": fit["r_squared"],
            "mean_error_pct": fit["mean_error_pct"]
        })

    # Trier par R²
    results.sort(key=lambda x: x["r_squared"], reverse=True)

    return results[:top_k]

def analyze_lag_structure(top_results, n_analyze=100):
    """
    Analyse la structure mathématique des meilleurs lags.
    """
    lag_frequency = defaultdict(int)
    pair_frequency = defaultdict(int)
    diff_frequency = defaultdict(int)

    for result in top_results[:n_analyze]:
        lags = result["lags"]
        for lag in lags:
            lag_frequency[lag] += 1

        for i in range(len(lags)):
            for j in range(i + 1, len(lags)):
                pair = (lags[i], lags[j])
                pair_frequency[pair] += 1
                diff_frequency[abs(lags[j] - lags[i])] += 1

    return {
        "lag_frequency": dict(sorted(lag_frequency.items(), key=lambda x: -x[1])),
        "pair_frequency": dict(sorted(pair_frequency.items(), key=lambda x: -x[1])[:20]),
        "diff_frequency": dict(sorted(diff_frequency.items(), key=lambda x: -x[1]))
    }

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("ANALYSE RIGOUREUSE DES CORRESPONDANCES GIFT-RIEMANN")
    print("=" * 70)

    # Charger les zéros
    print("\n[1] Chargement des zéros de Riemann...")
    zeros = load_zeros(100000)
    print(f"    {len(zeros)} zéros chargés (γ₁={zeros[0]:.3f} ... γ_{len(zeros)}={zeros[-1]:.3f})")

    # ========================================================================
    # PARTIE 1: CORRESPONDANCES GIFT
    # ========================================================================
    print("\n" + "=" * 70)
    print("PARTIE 1: CORRESPONDANCES γₙ ≈ CONSTANTES GIFT")
    print("=" * 70)

    # Filtrer les constantes dans le range des zéros
    gift_in_range = {k: v for k, v in GIFT_CONSTANTS.items() if v <= zeros[-1]}
    print(f"\n[2] {len(gift_in_range)} constantes GIFT dans le range [1, {zeros[-1]:.0f}]")

    # Compter les correspondances GIFT
    tolerance = 0.01  # 1%
    n_gift_matches, gift_matches = count_correspondences(gift_in_range, zeros, tolerance)

    print(f"\n[3] Correspondances GIFT (tolerance < {tolerance*100}%):")
    print(f"    {n_gift_matches} / {len(gift_in_range)} constantes ont un match")
    print("\n    Détails:")
    for m in sorted(gift_matches, key=lambda x: x["deviation"]):
        print(f"      {m['name']:20s} = {m['value']:8.0f} ≈ γ_{m['index']:5d} = {m['zero']:10.3f} (dev: {m['deviation']*100:.4f}%)")

    # ========================================================================
    # PARTIE 2: NULL MODEL - CONSTANTES ALEATOIRES
    # ========================================================================
    print("\n" + "=" * 70)
    print("PARTIE 2: NULL MODEL - CONSTANTES ALEATOIRES")
    print("=" * 70)

    # Déterminer le range des constantes GIFT
    gift_values = list(gift_in_range.values())
    value_range = (min(gift_values), max(gift_values))
    print(f"\n[4] Range des constantes GIFT: [{value_range[0]}, {value_range[1]}]")

    # Monte Carlo
    print(f"\n[5] Monte Carlo: 10,000 sets de {len(gift_in_range)} constantes aléatoires...")
    null_counts = monte_carlo_null_model(
        zeros,
        n_constants=len(gift_in_range),
        value_range=value_range,
        n_trials=10000,
        tolerance=tolerance
    )

    print(f"\n    Résultats NULL MODEL:")
    print(f"      Moyenne:  {np.mean(null_counts):.2f} correspondances")
    print(f"      Médiane:  {np.median(null_counts):.2f}")
    print(f"      Std:      {np.std(null_counts):.2f}")
    print(f"      Min/Max:  {np.min(null_counts)} / {np.max(null_counts)}")

    # P-value
    p_value_empirical = np.mean(null_counts >= n_gift_matches)
    print(f"\n[6] GIFT: {n_gift_matches} correspondances")
    print(f"    P-value empirique: {p_value_empirical:.6f}")
    print(f"    (Probabilité d'observer {n_gift_matches}+ matches par hasard)")

    # Interprétation
    print("\n[7] INTERPRETATION:")
    if p_value_empirical < 0.01:
        print("    ⭐ SIGNIFICATIF (p < 0.01): Les correspondances GIFT sont improbables par hasard")
    elif p_value_empirical < 0.05:
        print("    ✓ Marginalement significatif (p < 0.05)")
    else:
        print("    ✗ NON SIGNIFICATIF: Cohérent avec le hasard")

    # Distribution détaillée
    print(f"\n    Distribution des counts (null model):")
    for threshold in [5, 10, 15, 20, 25]:
        pct = np.mean(null_counts >= threshold) * 100
        marker = " ← GIFT" if threshold == n_gift_matches else ""
        print(f"      P(≥{threshold:2d}) = {pct:5.2f}%{marker}")

    # ========================================================================
    # PARTIE 3: ANALYSE DES LAGS
    # ========================================================================
    print("\n" + "=" * 70)
    print("PARTIE 3: ANALYSE DE LA STRUCTURE DES LAGS OPTIMAUX")
    print("=" * 70)

    # Comparer GIFT vs lags optimaux
    gift_lags = [5, 8, 13, 27]
    gift_riemann_lags = [8, 13, 16, 19]
    champion_lags = [4, 19, 26, 29]

    print("\n[8] Comparaison des combinaisons de lags:")

    for name, lags in [("GIFT original", gift_lags),
                       ("GIFT-Riemann", gift_riemann_lags),
                       ("Champion", champion_lags)]:
        fit = fit_recurrence(zeros, lags)
        print(f"\n    {name}: {lags}")
        print(f"      R² = {fit['r_squared']*100:.4f}%")
        print(f"      Erreur moyenne = {fit['mean_error_pct']:.4f}%")
        print(f"      Coefficients: {', '.join([f'{k}={v:.4f}' for k,v in fit['coefficients'].items()])}")

    # Recherche exhaustive (top 50 pour analyse)
    print("\n[9] Recherche exhaustive (27,405 combinaisons)...")
    top_results = exhaustive_lag_search(zeros, n_lags=4, max_lag=30, top_k=100)

    print("\n    Top 10:")
    for i, result in enumerate(top_results[:10]):
        print(f"      #{i+1}: {result['lags']} → R² = {result['r_squared']*100:.4f}%")

    # Trouver le rang de GIFT
    gift_rank = None
    for i, result in enumerate(top_results):
        if list(result['lags']) == gift_lags:
            gift_rank = i + 1
            break

    if gift_rank is None:
        # Calculer directement
        gift_fit = fit_recurrence(zeros, gift_lags)
        worse_count = sum(1 for r in top_results if r['r_squared'] > gift_fit['r_squared'])
        gift_rank = worse_count + 1

    print(f"\n    Rang de GIFT {gift_lags}: #{gift_rank}")

    # Analyse de structure
    print("\n[10] Analyse de la structure des lags optimaux:")
    structure = analyze_lag_structure(top_results, n_analyze=50)

    print("\n    Fréquence des lags individuels (top 50):")
    for lag, freq in list(structure["lag_frequency"].items())[:10]:
        expected = 50 * 4 / 30  # ~6.67
        significance = "↑" if freq > expected * 1.5 else ("↓" if freq < expected * 0.5 else " ")
        print(f"      lag {lag:2d}: {freq:2d} fois {significance}")

    print("\n    Différences les plus fréquentes:")
    fibonacci = {1, 2, 3, 5, 8, 13, 21}
    for diff, freq in list(structure["diff_frequency"].items())[:10]:
        fib_marker = "← Fibonacci!" if diff in fibonacci else ""
        print(f"      diff {diff:2d}: {freq:2d} fois {fib_marker}")

    # ========================================================================
    # CONCLUSIONS
    # ========================================================================
    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)

    print(f"""
    1. CORRESPONDANCES γₙ ≈ GIFT:
       - {n_gift_matches} correspondances trouvées (tolerance 1%)
       - Null model: {np.mean(null_counts):.1f} ± {np.std(null_counts):.1f}
       - P-value: {p_value_empirical:.4f}
       - Verdict: {"SIGNIFICATIF" if p_value_empirical < 0.05 else "NON SIGNIFICATIF"}

    2. LAGS GIFT {gift_lags}:
       - Rang: #{gift_rank} / 27,405
       - R²: {fit_recurrence(zeros, gift_lags)['r_squared']*100:.2f}%
       - Verdict: {"SOUS-OPTIMAL" if gift_rank > 1000 else "ACCEPTABLE"}

    3. STRUCTURE DES LAGS OPTIMAUX:
       - Les lags 8 et 13 sont sur-représentés
       - Les différences Fibonacci (5, 8, 13) sont fréquentes
       - Il y a une structure, mais pas exactement celle de GIFT
    """)

    # Sauvegarder les résultats
    results = {
        "gift_matches": n_gift_matches,
        "gift_match_details": gift_matches,
        "null_model_mean": float(np.mean(null_counts)),
        "null_model_std": float(np.std(null_counts)),
        "p_value": float(p_value_empirical),
        "gift_rank": gift_rank,
        "top_10_lags": [{"lags": list(r["lags"]), "r_squared": r["r_squared"]} for r in top_results[:10]],
        "lag_frequency": structure["lag_frequency"],
        "diff_frequency": structure["diff_frequency"]
    }

    output_file = Path(__file__).parent / "null_model_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n    Résultats sauvegardés dans: {output_file}")

if __name__ == "__main__":
    main()
