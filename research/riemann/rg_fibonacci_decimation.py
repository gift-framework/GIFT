#!/usr/bin/env python3
"""
GIFT Phase 2.9c: Fibonacci Decimation Analysis
===============================================

Observation: Le ratio (8×a₈)/(13×a₁₃) a des minima à m = 5, 11, 14
- 5 = F₅ (nombre de Fibonacci)
- 11 = F₅ + F₆ = 5 + 6  (mais 6 n'est pas Fibonacci... 11 est proche de F₇ - 2 = 13 - 2)
- 14 = F₇ + 1 = 13 + 1

Hypothèse: La décimation par nombres de Fibonacci préserve certains invariants.

Test: Comparer décimation par Fibonacci vs non-Fibonacci.
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


def fibonacci_sequence(n_terms: int = 20) -> List[int]:
    """Génère les premiers termes de Fibonacci."""
    fib = [1, 1]
    while len(fib) < n_terms:
        fib.append(fib[-1] + fib[-2])
    return fib


def lucas_sequence(n_terms: int = 20) -> List[int]:
    """Génère les premiers termes de Lucas (2, 1, 3, 4, 7, 11, 18, ...)"""
    luc = [2, 1]
    while len(luc) < n_terms:
        luc.append(luc[-1] + luc[-2])
    return luc


def analyze_decimation_type(gammas: np.ndarray,
                            gift_lags: List[int] = [5, 8, 13, 27]) -> dict:
    """
    Compare décimation Fibonacci vs non-Fibonacci.
    """

    fib = fibonacci_sequence(15)
    luc = lucas_sequence(15)

    results = {'fibonacci': [], 'lucas': [], 'prime': [], 'other': []}

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    for m in range(1, 25):
        gamma_m = decimate(gammas, m)

        if len(gamma_m) < 3000:
            continue

        try:
            coeffs, error = fit_recurrence(gamma_m, gift_lags,
                                           start=max(1000, max(gift_lags) + 100),
                                           end=min(10000, len(gamma_m)))

            products = [gift_lags[i] * coeffs[i] for i in range(len(gift_lags))]

            if products[2] != 0:
                ratio = products[1] / products[2]
            else:
                ratio = float('inf')

            result = {
                'm': m,
                'ratio': float(ratio),
                'deviation': float(abs(ratio - 1.0)),
                '8_times_a8': float(products[1]),
                '13_times_a13': float(products[2]),
                'sum': float(products[1] + products[2])
            }

            # Classifier
            if m in fib:
                results['fibonacci'].append(result)
            elif m in luc:
                results['lucas'].append(result)
            elif m in primes:
                results['prime'].append(result)
            else:
                results['other'].append(result)

        except Exception as e:
            continue

    return results


def main():
    print("=" * 70)
    print("GIFT Phase 2.9c: FIBONACCI DECIMATION ANALYSIS")
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
    # ANALYSE PAR TYPE DE DÉCIMATION
    # ============================================================

    gift_lags = [5, 8, 13, 27]
    results = analyze_decimation_type(gammas, gift_lags)

    # Fibonacci
    print("\n" + "=" * 70)
    print("DÉCIMATION PAR NOMBRES DE FIBONACCI")
    print("=" * 70)
    print(f"Fibonacci: 1, 1, 2, 3, 5, 8, 13, 21, ...")

    print(f"\n{'m':<5} {'Ratio':<10} {'|R-1|':<10} {'8×a₈':<10} {'13×a₁₃':<10} {'Somme':<10}")
    print("-" * 60)

    fib_avg_dev = []
    for r in results['fibonacci']:
        print(f"{r['m']:<5} {r['ratio']:<10.4f} {r['deviation']:<10.4f} {r['8_times_a8']:<10.4f} {r['13_times_a13']:<10.4f} {r['sum']:<10.4f}")
        fib_avg_dev.append(r['deviation'])

    if fib_avg_dev:
        print(f"\n   Déviation moyenne (Fibonacci): {np.mean(fib_avg_dev):.4f}")

    # Lucas
    print("\n" + "=" * 70)
    print("DÉCIMATION PAR NOMBRES DE LUCAS")
    print("=" * 70)
    print(f"Lucas: 2, 1, 3, 4, 7, 11, 18, ...")

    print(f"\n{'m':<5} {'Ratio':<10} {'|R-1|':<10} {'8×a₈':<10} {'13×a₁₃':<10} {'Somme':<10}")
    print("-" * 60)

    luc_avg_dev = []
    for r in results['lucas']:
        print(f"{r['m']:<5} {r['ratio']:<10.4f} {r['deviation']:<10.4f} {r['8_times_a8']:<10.4f} {r['13_times_a13']:<10.4f} {r['sum']:<10.4f}")
        luc_avg_dev.append(r['deviation'])

    if luc_avg_dev:
        print(f"\n   Déviation moyenne (Lucas): {np.mean(luc_avg_dev):.4f}")

    # Primes
    print("\n" + "=" * 70)
    print("DÉCIMATION PAR NOMBRES PREMIERS")
    print("=" * 70)

    print(f"\n{'m':<5} {'Ratio':<10} {'|R-1|':<10} {'8×a₈':<10} {'13×a₁₃':<10} {'Somme':<10}")
    print("-" * 60)

    prime_avg_dev = []
    for r in results['prime']:
        print(f"{r['m']:<5} {r['ratio']:<10.4f} {r['deviation']:<10.4f} {r['8_times_a8']:<10.4f} {r['13_times_a13']:<10.4f} {r['sum']:<10.4f}")
        prime_avg_dev.append(r['deviation'])

    if prime_avg_dev:
        print(f"\n   Déviation moyenne (Premiers): {np.mean(prime_avg_dev):.4f}")

    # Autres
    print("\n" + "=" * 70)
    print("AUTRES DÉCIMATIONS")
    print("=" * 70)

    print(f"\n{'m':<5} {'Ratio':<10} {'|R-1|':<10} {'8×a₈':<10} {'13×a₁₃':<10} {'Somme':<10}")
    print("-" * 60)

    other_avg_dev = []
    for r in results['other']:
        print(f"{r['m']:<5} {r['ratio']:<10.4f} {r['deviation']:<10.4f} {r['8_times_a8']:<10.4f} {r['13_times_a13']:<10.4f} {r['sum']:<10.4f}")
        other_avg_dev.append(r['deviation'])

    if other_avg_dev:
        print(f"\n   Déviation moyenne (Autres): {np.mean(other_avg_dev):.4f}")

    # ============================================================
    # COMPARAISON STATISTIQUE
    # ============================================================

    print("\n" + "=" * 70)
    print("COMPARAISON STATISTIQUE")
    print("=" * 70)

    categories = [
        ('Fibonacci', fib_avg_dev),
        ('Lucas', luc_avg_dev),
        ('Premiers', prime_avg_dev),
        ('Autres', other_avg_dev)
    ]

    print(f"\n{'Catégorie':<15} {'N':<5} {'Moy |R-1|':<12} {'Min |R-1|':<12} {'Max |R-1|':<12}")
    print("-" * 60)

    for name, devs in categories:
        if devs:
            print(f"{name:<15} {len(devs):<5} {np.mean(devs):<12.4f} {np.min(devs):<12.4f} {np.max(devs):<12.4f}")
        else:
            print(f"{name:<15} {'N/A'}")

    # ============================================================
    # RECHERCHE DU PATTERN
    # ============================================================

    print("\n" + "=" * 70)
    print("RECHERCHE DE PATTERN DANS LES MINIMA")
    print("=" * 70)

    # Collecter tous les résultats
    all_results = []
    for cat in results.values():
        all_results.extend(cat)

    # Trier par déviation
    all_results.sort(key=lambda x: x['deviation'])

    print("\n   Top 10 meilleures échelles:")
    print(f"   {'Rang':<6} {'m':<5} {'|R-1|':<10} {'Catégorie':<15}")
    print("   " + "-" * 40)

    fib = fibonacci_sequence(15)
    luc = lucas_sequence(15)
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    for i, r in enumerate(all_results[:10]):
        m = r['m']
        if m in fib:
            cat = 'Fibonacci'
        elif m in luc:
            cat = 'Lucas'
        elif m in primes:
            cat = 'Premier'
        else:
            cat = 'Autre'
        print(f"   {i+1:<6} {m:<5} {r['deviation']:<10.4f} {cat:<15}")

    # ============================================================
    # TEST: DÉCIMATION RÉCURSIVE FIBONACCI
    # ============================================================

    print("\n" + "=" * 70)
    print("TEST: DÉCIMATION RÉCURSIVE PAR FIBONACCI")
    print("=" * 70)
    print("Idée: γ → γ^(5) → γ^(5×8) → γ^(5×8×13) → ...")

    fib_chain = [5, 8, 13]  # Décimation successive

    gamma_current = gammas.copy()
    print(f"\n   Étape 0: N = {len(gamma_current):,}")

    for i, f in enumerate(fib_chain):
        gamma_current = decimate(gamma_current, f)
        print(f"   Étape {i+1} (×{f}): N = {len(gamma_current):,}")

        if len(gamma_current) < 1000:
            print("   Pas assez de données pour continuer")
            break

        try:
            coeffs, error = fit_recurrence(gamma_current, gift_lags,
                                           start=max(200, max(gift_lags) + 50),
                                           end=min(2000, len(gamma_current)))

            products = [gift_lags[j] * coeffs[j] for j in range(len(gift_lags))]
            ratio = products[1] / products[2] if products[2] != 0 else float('inf')

            print(f"      8×a₈ = {products[1]:.4f}, 13×a₁₃ = {products[2]:.4f}")
            print(f"      Ratio = {ratio:.4f}, |R-1| = {abs(ratio-1):.4f}")

        except Exception as e:
            print(f"      Erreur: {e}")

    # ============================================================
    # EXPORT
    # ============================================================

    output = {
        'analysis': 'fibonacci_decimation',
        'results_by_category': {k: v for k, v in results.items()},
        'statistics': {
            'fibonacci_mean_dev': float(np.mean(fib_avg_dev)) if fib_avg_dev else None,
            'lucas_mean_dev': float(np.mean(luc_avg_dev)) if luc_avg_dev else None,
            'prime_mean_dev': float(np.mean(prime_avg_dev)) if prime_avg_dev else None,
            'other_mean_dev': float(np.mean(other_avg_dev)) if other_avg_dev else None,
        },
        'best_scales': [r['m'] for r in all_results[:5]]
    }

    with open('phase29c_fibonacci_decimation.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nRésultats sauvegardés dans phase29c_fibonacci_decimation.json")


if __name__ == "__main__":
    main()
