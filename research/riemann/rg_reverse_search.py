#!/usr/bin/env python3
"""
GIFT Phase 2.9: RG Reverse Scale Search
========================================

Hypothèse: Les lags GIFT [5,8,13,27] sont des lags standards [1,2,3,4]
à une certaine échelle de décimation m.

Si γₙ^(m) = γ_{mn}, alors:
- lag 1 à l'échelle m = lag m à l'échelle 1
- lag 2 à l'échelle m = lag 2m à l'échelle 1
- etc.

Question: Existe-t-il m tel que [m, 2m, 3m, 4m] ≈ [5, 8, 13, 27] ?

Réponse directe: Non exactement, mais cherchons le meilleur compromis.

Alternative: Chercher m tel que les coefficients [a₁, a₂, a₃, a₄] sur
les zéros décimés par m satisfassent les mêmes contraintes GIFT.
"""

import numpy as np
from typing import List, Tuple, Dict
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


def compute_gift_invariants(coeffs: np.ndarray, lags: List[int]) -> Dict:
    """
    Calcule les invariants GIFT pour des coefficients donnés.

    Invariant clé: lag_i × a_i devrait être constant pour Fibonacci adjacent.
    """
    products = {f"lag_{lag}_times_a": lag * coeffs[i] for i, lag in enumerate(lags)}

    # Pour 4 lags, vérifier si lag[1]×a[1] ≈ lag[2]×a[2] (analogue à 8×a₈ = 13×a₁₃)
    if len(lags) >= 3:
        prod_1 = lags[1] * coeffs[1]
        prod_2 = lags[2] * coeffs[2]
        ratio = prod_1 / prod_2 if prod_2 != 0 else float('inf')
        products['ratio_lag12_products'] = ratio
        products['deviation_from_unity'] = abs(ratio - 1.0)

    return products


def search_optimal_scale(gammas: np.ndarray,
                         target_lags: List[int] = [5, 8, 13, 27],
                         search_lags: List[int] = [1, 2, 3, 4],
                         m_range: range = range(1, 20)) -> Dict:
    """
    Cherche l'échelle m optimale où les lags standards reproduisent GIFT.
    """

    # D'abord, calculer les invariants GIFT de référence
    print("=" * 70)
    print("RÉFÉRENCE: GIFT LAGS [5, 8, 13, 27] SUR DONNÉES ORIGINALES")
    print("=" * 70)

    coeffs_gift, error_gift = fit_recurrence(gammas, target_lags,
                                              start=10000, end=min(50000, len(gammas)))

    print(f"\nCoefficients GIFT:")
    for i, lag in enumerate(target_lags):
        print(f"   a_{lag} = {coeffs_gift[i]:.6f}")
    print(f"   c = {coeffs_gift[-1]:.4f}")
    print(f"   Erreur moyenne: {error_gift:.6f}")

    # Calculer les produits lag × coeff
    gift_products = [lag * coeffs_gift[i] for i, lag in enumerate(target_lags)]
    print(f"\nProduits lag × a:")
    for i, lag in enumerate(target_lags):
        print(f"   {lag} × a_{lag} = {gift_products[i]:.4f}")

    # Vérifier 8×a₈ ≈ 13×a₁₃
    ratio_8_13 = gift_products[1] / gift_products[2] if gift_products[2] != 0 else 0
    print(f"\n   Ratio (8×a₈)/(13×a₁₃) = {ratio_8_13:.4f}")

    # ============================================================
    # RECHERCHE PAR ÉCHELLE
    # ============================================================

    print("\n" + "=" * 70)
    print("RECHERCHE: LAGS STANDARDS SUR DONNÉES DÉCIMÉES")
    print("=" * 70)

    results = {}

    for m in m_range:
        gamma_m = decimate(gammas, m)

        if len(gamma_m) < 5000:
            continue

        # Les lags effectifs en termes de zéros originaux
        effective_lags = [lag * m for lag in search_lags]

        try:
            coeffs_m, error_m = fit_recurrence(gamma_m, search_lags,
                                                start=1000, end=min(10000, len(gamma_m)))

            # Produits lag × coeff
            products_m = [lag * coeffs_m[i] for i, lag in enumerate(search_lags)]

            # Distance aux invariants GIFT
            # Normaliser par les valeurs GIFT pour comparer

            results[m] = {
                'effective_lags': effective_lags,
                'coefficients': {f'a_{lag}': float(coeffs_m[i]) for i, lag in enumerate(search_lags)},
                'c': float(coeffs_m[-1]),
                'error': float(error_m),
                'products': {f'{lag}_times_a_{lag}': float(products_m[i]) for i, lag in enumerate(search_lags)},
            }

            # Ratio produit[1]/produit[2] - doit être proche de 1 pour GIFT
            if products_m[2] != 0:
                ratio = products_m[1] / products_m[2]
                results[m]['ratio_products_12'] = float(ratio)
                results[m]['deviation_from_unity'] = float(abs(ratio - 1.0))

        except Exception as e:
            continue

    return results, coeffs_gift, gift_products


def find_fibonacci_scale(gammas: np.ndarray) -> Dict:
    """
    Test spécifique: à quelle échelle les lags Fibonacci consécutifs
    satisfont-ils la contrainte lag₁×a₁ = lag₂×a₂ ?

    Fibonacci pairs: (1,2), (2,3), (3,5), (5,8), (8,13), ...
    """

    print("\n" + "=" * 70)
    print("RECHERCHE: ÉCHELLE OÙ FIBONACCI SATISFAIT lag×a = CONSTANT")
    print("=" * 70)

    fibonacci_pairs = [(1, 2), (2, 3), (3, 5), (5, 8), (8, 13)]

    results = {}

    for m in range(1, 15):
        gamma_m = decimate(gammas, m)

        if len(gamma_m) < 3000:
            continue

        results[m] = {'pairs': {}}

        for f1, f2 in fibonacci_pairs:
            lags = [f1, f2]

            try:
                coeffs, error = fit_recurrence(gamma_m, lags,
                                               start=max(lags) + 100,
                                               end=min(5000, len(gamma_m)))

                prod1 = f1 * coeffs[0]
                prod2 = f2 * coeffs[1]

                ratio = prod1 / prod2 if prod2 != 0 else float('inf')

                results[m]['pairs'][f'({f1},{f2})'] = {
                    f'a_{f1}': float(coeffs[0]),
                    f'a_{f2}': float(coeffs[1]),
                    f'{f1}_times_a_{f1}': float(prod1),
                    f'{f2}_times_a_{f2}': float(prod2),
                    'ratio': float(ratio),
                    'deviation': float(abs(ratio - 1.0))
                }

            except:
                continue

    return results


def main():
    print("=" * 70)
    print("GIFT Phase 2.9: RG REVERSE SCALE SEARCH")
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
    # ANALYSE 1: Lags standards sur données décimées
    # ============================================================

    results, coeffs_gift, gift_products = search_optimal_scale(
        gammas,
        target_lags=[5, 8, 13, 27],
        search_lags=[1, 2, 3, 4],
        m_range=range(1, 15)
    )

    print("\n" + "-" * 70)
    print("Résumé: Lags [1,2,3,4] à différentes échelles")
    print("-" * 70)
    print(f"\n{'m':<5} {'Lags eff.':<20} {'Ratio p1/p2':<12} {'Déviation':<12}")
    print("-" * 50)

    for m in sorted(results.keys()):
        r = results[m]
        eff = r['effective_lags']
        ratio = r.get('ratio_products_12', 0)
        dev = r.get('deviation_from_unity', 999)
        print(f"{m:<5} {str(eff):<20} {ratio:<12.3f} {dev:<12.3f}")

    # ============================================================
    # ANALYSE 2: Paires Fibonacci à différentes échelles
    # ============================================================

    fib_results = find_fibonacci_scale(gammas)

    print("\n" + "-" * 70)
    print("Paires Fibonacci: ratio (lag₁×a₁)/(lag₂×a₂) par échelle")
    print("-" * 70)

    pairs = [(1, 2), (2, 3), (3, 5), (5, 8), (8, 13)]

    print(f"\n{'m':<5}", end="")
    for f1, f2 in pairs:
        print(f"{f'({f1},{f2})':<12}", end="")
    print()
    print("-" * 65)

    for m in sorted(fib_results.keys()):
        print(f"{m:<5}", end="")
        for f1, f2 in pairs:
            key = f'({f1},{f2})'
            if key in fib_results[m]['pairs']:
                ratio = fib_results[m]['pairs'][key]['ratio']
                print(f"{ratio:<12.3f}", end="")
            else:
                print(f"{'N/A':<12}", end="")
        print()

    # ============================================================
    # ANALYSE 3: Trouver l'échelle où (8,13) donne ratio ≈ 1
    # ============================================================

    print("\n" + "=" * 70)
    print("FOCUS: Paire (8,13) - cherche l'échelle où ratio = 1")
    print("=" * 70)

    best_scale = None
    best_deviation = float('inf')

    for m in sorted(fib_results.keys()):
        if '(8,13)' in fib_results[m]['pairs']:
            r = fib_results[m]['pairs']['(8,13)']
            dev = r['deviation']
            print(f"\n   m = {m}:")
            print(f"      8 × a₈ = {r['8_times_a_8']:.4f}")
            print(f"      13 × a₁₃ = {r['13_times_a_13']:.4f}")
            print(f"      Ratio = {r['ratio']:.4f}")
            print(f"      Déviation = {dev:.4f} {'<-- MINIMUM' if dev < best_deviation else ''}")

            if dev < best_deviation:
                best_deviation = dev
                best_scale = m

    print(f"\n   MEILLEURE ÉCHELLE: m = {best_scale} (dév. = {best_deviation:.4f})")

    # ============================================================
    # ANALYSE 4: Test inverse - à quelle décimation les lags GIFT
    #            deviennent-ils équivalents à [1,2,3,4] ?
    # ============================================================

    print("\n" + "=" * 70)
    print("TEST INVERSE: GIFT lags sur données décimées")
    print("=" * 70)
    print("Question: Les lags [5,8,13,27] sur γ^(m) donnent-ils les mêmes")
    print("          invariants que [1,2,3,4] sur γ original ?")

    # Référence: lags [1,2,3,4] sur original
    ref_lags = [1, 2, 3, 4]
    coeffs_ref, error_ref = fit_recurrence(gammas, ref_lags,
                                            start=1000, end=50000)

    print(f"\nRéférence: lags [1,2,3,4] sur γ original")
    for i, lag in enumerate(ref_lags):
        print(f"   a_{lag} = {coeffs_ref[i]:.6f}")

    prod_ref = [lag * coeffs_ref[i] for i, lag in enumerate(ref_lags)]
    print(f"\nProduits:")
    for i, lag in enumerate(ref_lags):
        print(f"   {lag} × a_{lag} = {prod_ref[i]:.4f}")

    ratio_ref = prod_ref[1] / prod_ref[2] if prod_ref[2] != 0 else 0
    print(f"\n   Ratio (2×a₂)/(3×a₃) = {ratio_ref:.4f}")

    # Maintenant GIFT lags sur décimé
    print("\nGIFT lags [5,8,13,27] sur γ^(m):")

    gift_lags = [5, 8, 13, 27]

    for m in [1, 2, 3, 5]:
        gamma_m = decimate(gammas, m)
        if len(gamma_m) < 5000:
            continue

        try:
            coeffs_m, error_m = fit_recurrence(gamma_m, gift_lags,
                                               start=1000, end=min(10000, len(gamma_m)))

            print(f"\n   m = {m} (N = {len(gamma_m):,}):")
            for i, lag in enumerate(gift_lags):
                print(f"      a_{lag} = {coeffs_m[i]:.6f}")

            prod_m = [lag * coeffs_m[i] for i, lag in enumerate(gift_lags)]
            print(f"      Produits: 8×a₈={prod_m[1]:.3f}, 13×a₁₃={prod_m[2]:.3f}")
            ratio_m = prod_m[1] / prod_m[2] if prod_m[2] != 0 else 0
            print(f"      Ratio = {ratio_m:.4f}")

        except Exception as e:
            print(f"   m = {m}: Erreur - {e}")

    # ============================================================
    # EXPORT
    # ============================================================

    output = {
        'method': 'reverse_scale_search',
        'standard_lags_decimated': results,
        'fibonacci_pairs_by_scale': fib_results,
        'best_scale_for_8_13': {
            'scale': best_scale,
            'deviation': best_deviation
        },
        'gift_reference': {
            'lags': [5, 8, 13, 27],
            'products': {f'{lag}_times_a': float(gift_products[i])
                        for i, lag in enumerate([5, 8, 13, 27])}
        }
    }

    with open('phase29_reverse_search.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nRésultats sauvegardés dans phase29_reverse_search.json")


if __name__ == "__main__":
    main()
