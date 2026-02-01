#!/usr/bin/env python3
"""
GIFT Phase 2.9b: Scale Convergence Analysis
============================================

Découverte: À m=5, les lags GIFT [5,8,13,27] satisfont ratio ≈ 0.96

Question: Y a-t-il une valeur m où ratio → 1 exactement ?

Hypothèse: Le ratio (8×a₈)/(13×a₁₃) suit une loi en m qui converge
vers une limite liée à φ (nombre d'or) ou un ratio GIFT.
"""

import numpy as np
from typing import List, Tuple
import json


def fit_recurrence(gamma: np.ndarray, lags: List[int],
                   start: int = None, end: int = None) -> Tuple[np.ndarray, float]:
    """Fit récurrence linéaire."""
    max_lag = max(lags)
    if start is None:
        start = max_lag
    if end is None:
        end = len(gamma)

    n_points = end - start
    n_params = len(lags) + 1

    X = np.zeros((n_points, n_params))
    for i, lag in enumerate(lags):
        X[:, i] = gamma[start - lag:end - lag]
    X[:, -1] = 1.0

    y = gamma[start:end]
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    y_pred = X @ coeffs
    errors = np.abs(y_pred - y)

    return coeffs, np.mean(errors)


def decimate(gamma: np.ndarray, m: int) -> np.ndarray:
    """Décimation: garder 1 zéro sur m."""
    return gamma[::m]


def analyze_ratio_vs_scale(gammas: np.ndarray, lags: List[int],
                           m_range: range, min_points: int = 3000) -> dict:
    """Analyse le ratio (lag[1]×a[1])/(lag[2]×a[2]) en fonction de l'échelle."""

    results = []

    for m in m_range:
        gamma_m = decimate(gammas, m)

        if len(gamma_m) < min_points:
            continue

        try:
            max_lag = max(lags)
            coeffs, error = fit_recurrence(gamma_m, lags,
                                           start=max(1000, max_lag + 100),
                                           end=min(10000, len(gamma_m)))

            # Produits lag × coeff
            products = [lags[i] * coeffs[i] for i in range(len(lags))]

            # Ratio des deux premiers Fibonacci adjacents (index 1 et 2 pour [5,8,13,27])
            if len(lags) >= 3 and products[2] != 0:
                ratio = products[1] / products[2]

                results.append({
                    'm': m,
                    'n_points': len(gamma_m),
                    'coefficients': {f'a_{lags[i]}': float(coeffs[i]) for i in range(len(lags))},
                    'products': {f'{lags[i]}_times_a': float(products[i]) for i in range(len(lags))},
                    'ratio_8_13': float(ratio),
                    'deviation_from_1': float(abs(ratio - 1.0)),
                    'error': float(error)
                })

        except Exception as e:
            continue

    return results


def fit_convergence_law(results: list) -> dict:
    """
    Fit une loi de convergence: ratio(m) = r_∞ + A/m^α

    Si le ratio converge vers 1 en m→∞, alors r_∞ = 1.
    """

    if len(results) < 5:
        return {'status': 'insufficient_data'}

    m_values = np.array([r['m'] for r in results])
    ratios = np.array([r['ratio_8_13'] for r in results])

    # Fit linéaire log-log pour la déviation: |ratio - r_∞| ~ m^(-α)
    # Essayer différentes valeurs de r_∞

    best_fit = None
    best_r2 = -float('inf')

    for r_inf in [1.0, 13/8, 8/13, (1 + np.sqrt(5))/2]:
        deviations = np.abs(ratios - r_inf)

        # Éviter log(0)
        valid = deviations > 1e-10
        if np.sum(valid) < 3:
            continue

        log_m = np.log(m_values[valid])
        log_dev = np.log(deviations[valid])

        # Régression linéaire
        slope, intercept = np.polyfit(log_m, log_dev, 1)

        # R²
        y_pred = slope * log_m + intercept
        ss_res = np.sum((log_dev - y_pred)**2)
        ss_tot = np.sum((log_dev - np.mean(log_dev))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        if r2 > best_r2:
            best_r2 = r2
            best_fit = {
                'r_infinity': float(r_inf),
                'alpha': float(-slope),
                'amplitude': float(np.exp(intercept)),
                'r_squared': float(r2)
            }

    return best_fit


def main():
    print("=" * 70)
    print("GIFT Phase 2.9b: SCALE CONVERGENCE ANALYSIS")
    print("=" * 70)

    # Charger les données
    print("\nChargement des données...")

    try:
        gammas = []
        for filename in ['zeta/zeros6', 'zeta/zeros1']:
            try:
                with open(filename, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            try:
                                gammas.append(float(line.split()[0]))
                            except:
                                continue
                print(f"   Chargé {len(gammas):,} zéros depuis {filename}")
                break
            except FileNotFoundError:
                continue

        if not gammas:
            print("   Aucun fichier de zéros trouvé")
            return

        gammas = np.array(sorted(gammas))

    except Exception as e:
        print(f"   Erreur: {e}")
        return

    # ============================================================
    # ANALYSE FINE DU RATIO EN FONCTION DE L'ÉCHELLE
    # ============================================================

    print("\n" + "=" * 70)
    print("RATIO (8×a₈)/(13×a₁₃) vs ÉCHELLE DE DÉCIMATION")
    print("=" * 70)

    gift_lags = [5, 8, 13, 27]
    results = analyze_ratio_vs_scale(gammas, gift_lags, m_range=range(1, 25))

    if not results:
        print("Pas assez de données")
        return

    print(f"\n{'m':<5} {'N':<10} {'8×a₈':<12} {'13×a₁₃':<12} {'Ratio':<10} {'|R-1|':<10}")
    print("-" * 65)

    for r in results:
        p8 = r['products']['8_times_a']
        p13 = r['products']['13_times_a']
        ratio = r['ratio_8_13']
        dev = r['deviation_from_1']
        print(f"{r['m']:<5} {r['n_points']:<10} {p8:<12.4f} {p13:<12.4f} {ratio:<10.4f} {dev:<10.4f}")

    # ============================================================
    # FIT DE CONVERGENCE
    # ============================================================

    print("\n" + "=" * 70)
    print("LOI DE CONVERGENCE: ratio(m) = r_∞ + A/m^α")
    print("=" * 70)

    convergence = fit_convergence_law(results)

    if convergence.get('status') == 'insufficient_data':
        print("Pas assez de points pour fitter")
    else:
        print(f"\n   Meilleur fit:")
        print(f"   r_∞ = {convergence['r_infinity']:.4f}")
        print(f"   α = {convergence['alpha']:.4f}")
        print(f"   A = {convergence['amplitude']:.4f}")
        print(f"   R² = {convergence['r_squared']:.4f}")

        # Interprétation
        r_inf = convergence['r_infinity']
        if abs(r_inf - 1.0) < 0.01:
            print(f"\n   → Le ratio converge vers 1 (invariant GIFT préservé)")
        elif abs(r_inf - 13/8) < 0.01:
            print(f"\n   → Le ratio converge vers 13/8 (φ² - φ ≈ 1.618)")
        elif abs(r_inf - 8/13) < 0.01:
            print(f"\n   → Le ratio converge vers 8/13 (1/φ ≈ 0.618)")

    # ============================================================
    # TEST SPÉCIFIQUE: À QUELLE ÉCHELLE ratio = 1 ?
    # ============================================================

    print("\n" + "=" * 70)
    print("RECHERCHE: ÉCHELLE OÙ RATIO = 1")
    print("=" * 70)

    # Interpolation linéaire
    m_values = [r['m'] for r in results]
    ratios = [r['ratio_8_13'] for r in results]

    # Trouver où ratio croise 1
    crossings = []
    for i in range(len(ratios) - 1):
        if (ratios[i] - 1) * (ratios[i+1] - 1) < 0:
            # Interpolation linéaire
            m_cross = m_values[i] + (1 - ratios[i]) * (m_values[i+1] - m_values[i]) / (ratios[i+1] - ratios[i])
            crossings.append(m_cross)

    if crossings:
        print(f"\n   Le ratio croise 1 à m ≈ {crossings[0]:.2f}")

        # Vérifier si c'est proche d'un nombre GIFT
        m_cross = crossings[0]
        gift_numbers = [5, 7, 8, 13, 14, 21, 27]
        closest = min(gift_numbers, key=lambda x: abs(x - m_cross))
        print(f"   Nombre GIFT le plus proche: {closest} (diff = {abs(m_cross - closest):.2f})")
    else:
        print(f"\n   Le ratio ne croise pas 1 dans la plage testée")
        # Trouver le minimum
        min_dev = min(results, key=lambda r: r['deviation_from_1'])
        print(f"   Meilleur ratio: {min_dev['ratio_8_13']:.4f} à m = {min_dev['m']}")

    # ============================================================
    # ANALYSE DES PRODUITS INDIVIDUELS
    # ============================================================

    print("\n" + "=" * 70)
    print("ÉVOLUTION DES PRODUITS INDIVIDUELS")
    print("=" * 70)

    print(f"\n{'m':<5} {'5×a₅':<10} {'8×a₈':<10} {'13×a₁₃':<10} {'27×a₂₇':<10}")
    print("-" * 50)

    for r in results:
        p5 = r['products']['5_times_a']
        p8 = r['products']['8_times_a']
        p13 = r['products']['13_times_a']
        p27 = r['products']['27_times_a']
        print(f"{r['m']:<5} {p5:<10.3f} {p8:<10.3f} {p13:<10.3f} {p27:<10.3f}")

    # ============================================================
    # TEST: SOMME DES PRODUITS
    # ============================================================

    print("\n" + "=" * 70)
    print("SOMME DES PRODUITS vs 36 (h_G₂²)")
    print("=" * 70)

    print(f"\n{'m':<5} {'8×a₈ + 13×a₁₃':<15} {'Dév/36':<10}")
    print("-" * 35)

    for r in results:
        p8 = r['products']['8_times_a']
        p13 = r['products']['13_times_a']
        sum_p = p8 + p13
        dev = abs(sum_p - 36) / 36 * 100
        marker = "←" if dev < 5 else ""
        print(f"{r['m']:<5} {sum_p:<15.3f} {dev:<10.1f}% {marker}")

    # ============================================================
    # EXPORT
    # ============================================================

    output = {
        'analysis': 'scale_convergence',
        'lags': gift_lags,
        'results_by_scale': results,
        'convergence_fit': convergence,
        'crossings_at_ratio_1': crossings
    }

    with open('phase29b_scale_convergence.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nRésultats sauvegardés dans phase29b_scale_convergence.json")


if __name__ == "__main__":
    main()
