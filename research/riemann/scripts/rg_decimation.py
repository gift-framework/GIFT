#!/usr/bin/env python3
"""
GIFT Phase 2.8: RG by Decimation
================================

Proposition de GPT: Définir le RG explicitement par décimation/coarse-graining.

Au lieu de FITTER les β, on les DÉRIVE comme valeurs propres du Jacobien
de la transformation RG près du point fixe.

Décimation: γₙ^(m) = γ_{mn} (garder 1 zéro sur m)

Si les courbes a_i^(m)(γ) se superposent après rescaling → structure RG réelle
Les β deviennent des "scaling dimensions" calculables.
"""

import numpy as np
from typing import List, Tuple, Dict
import json

# ============================================================
# RECURRENCE FITTING
# ============================================================

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

# ============================================================
# DECIMATION RG
# ============================================================

def decimate(gamma: np.ndarray, m: int) -> np.ndarray:
    """
    Décimation: garder 1 zéro sur m.
    γₙ^(m) = γ_{mn}
    """
    return gamma[::m]

def block_average(gamma: np.ndarray, m: int) -> np.ndarray:
    """
    Moyennage par blocs de taille m.
    γ̃ₙ^(m) = (1/m) Σⱼ γ_{mn+j}
    """
    n_blocks = len(gamma) // m
    return np.array([np.mean(gamma[i*m:(i+1)*m]) for i in range(n_blocks)])

def analyze_at_scale(gamma: np.ndarray, lags: List[int],
                     window_size: int = 10000, n_windows: int = 10) -> List[Dict]:
    """
    Analyse par fenêtres glissantes à une échelle donnée.
    """
    results = []
    step = (len(gamma) - window_size) // (n_windows - 1) if n_windows > 1 else 0

    for i in range(n_windows):
        start_idx = i * step
        end_idx = start_idx + window_size

        if end_idx > len(gamma):
            break

        window = gamma[start_idx:end_idx]
        stable_start = int(window_size * 0.7)

        try:
            coeffs, error = fit_recurrence(window, lags, stable_start, window_size)
            # Estimer gamma moyen de la fenêtre
            gamma_mid = np.mean(window)

            results.append({
                'gamma_mid': float(gamma_mid),
                'coefficients': {f'a_{lag}': float(coeffs[i]) for i, lag in enumerate(lags)},
                'c': float(coeffs[-1]),
                'error': float(error)
            })
        except Exception as e:
            pass

    return results

# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():
    print("="*70)
    print("GIFT Phase 2.8: RG BY DECIMATION")
    print("="*70)

    # Charger les données
    print("\n📂 Chargement des données...")

    try:
        gammas = []
        # Try zeros6 first, then zeros1
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
            print("   ❌ Aucun fichier de zéros trouvé")
            return

        gammas = np.array(sorted(gammas))

    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return

    lags = [5, 8, 13, 27]

    # ============================================================
    # ANALYSE À DIFFÉRENTES ÉCHELLES DE DÉCIMATION
    # ============================================================

    print("\n" + "="*70)
    print("ANALYSE RG PAR DÉCIMATION")
    print("="*70)

    decimation_factors = [1, 2, 3, 5, 8, 13]  # Inclut des Fibonacci!
    all_results = {}

    for m in decimation_factors:
        print(f"\n{'─'*60}")
        print(f"📊 Échelle m = {m} (1 zéro sur {m})")
        print(f"{'─'*60}")

        # Décimer
        gamma_m = decimate(gammas, m)
        print(f"   N zéros après décimation: {len(gamma_m):,}")

        if len(gamma_m) < 5000:
            print(f"   ⚠️ Pas assez de données, skip")
            continue

        # Les lags effectifs restent les mêmes (on prédit γₙ^(m) depuis γₙ₋ₖ^(m))
        # Mais la signification physique change: lag k à l'échelle m = lag k×m en zéros originaux

        # Analyser
        results = analyze_at_scale(gamma_m, lags, window_size=min(10000, len(gamma_m)//2))

        if not results:
            print(f"   ⚠️ Pas de résultats")
            continue

        all_results[m] = results

        # Afficher les coefficients moyens
        avg_coeffs = {}
        for lag in lags:
            key = f'a_{lag}'
            values = [r['coefficients'][key] for r in results]
            avg_coeffs[key] = np.mean(values)
            std_coeffs = np.std(values)
            print(f"   <a_{lag}> = {avg_coeffs[key]:.4f} ± {std_coeffs:.4f}")

        avg_c = np.mean([r['c'] for r in results])
        print(f"   <c> = {avg_c:.4f}")

        # Calculer lag × coeff (devrait donner les invariants GIFT)
        print(f"\n   Produits lag × <a_lag>:")
        for lag in lags:
            prod = lag * avg_coeffs[f'a_{lag}']
            print(f"   {lag} × <a_{lag}> = {prod:.4f}")

    # ============================================================
    # TEST DE SUPERPOSITION (SCALING COLLAPSE)
    # ============================================================

    print("\n" + "="*70)
    print("TEST DE SUPERPOSITION (SCALING COLLAPSE)")
    print("="*70)

    if len(all_results) < 2:
        print("\n⚠️ Pas assez d'échelles pour tester la superposition")
        return

    # Pour chaque coefficient, vérifier si les courbes se superposent
    # après rescaling γ → γ/m

    print("\n📊 Coefficients moyens par échelle de décimation")
    print(f"\n   {'m':<5}", end="")
    for lag in lags:
        print(f"{'a_'+str(lag):>12}", end="")
    print(f"{'c':>12}")
    print("   " + "-"*60)

    scale_data = {}
    for m in sorted(all_results.keys()):
        results = all_results[m]
        print(f"   {m:<5}", end="")
        scale_data[m] = {}
        for lag in lags:
            key = f'a_{lag}'
            avg = np.mean([r['coefficients'][key] for r in results])
            scale_data[m][key] = avg
            print(f"{avg:>12.4f}", end="")
        avg_c = np.mean([r['c'] for r in results])
        scale_data[m]['c'] = avg_c
        print(f"{avg_c:>12.4f}")

    # ============================================================
    # ANALYSE DU FLOW RG
    # ============================================================

    print("\n" + "="*70)
    print("FLOW RG: VARIATION DES COEFFICIENTS AVEC L'ÉCHELLE")
    print("="*70)

    # Calculer les "β effectifs" depuis la variation avec m
    # Si a(m) ~ m^(-β), alors β = -d(log a)/d(log m)

    print("\n📊 Exposants de scaling (β effectifs)")
    print("   Si a(m) ~ m^(-β), alors β = -Δlog(a)/Δlog(m)")

    m_values = sorted(all_results.keys())
    if len(m_values) >= 2:
        print(f"\n   {'Coeff':<10} {'β_eff':>10} {'Interprétation':>25}")
        print("   " + "-"*50)

        for lag in lags:
            key = f'a_{lag}'
            # Régression log-log
            log_m = np.log(m_values)
            log_a = np.log([abs(scale_data[m][key]) + 1e-10 for m in m_values])

            # Fit linéaire
            if len(log_m) >= 2:
                slope, intercept = np.polyfit(log_m, log_a, 1)
                beta_eff = -slope

                # Interprétation
                interp = ""
                if abs(beta_eff) < 0.1:
                    interp = "~constant (point fixe)"
                elif abs(beta_eff - 1) < 0.2:
                    interp = "~1/m (marginal)"
                else:
                    interp = f"scaling non-trivial"

                print(f"   a_{lag:<6} {beta_eff:>10.3f} {interp:>25}")

    # ============================================================
    # TEST: LES INVARIANTS lag×a SONT-ILS PRESERVÉS?
    # ============================================================

    print("\n" + "="*70)
    print("TEST: INVARIANTS lag × a SOUS DÉCIMATION")
    print("="*70)

    print(f"\n   {'m':<5} {'5×a_5':>10} {'8×a_8':>10} {'13×a_13':>10} {'27×a_27':>10}")
    print("   " + "-"*50)

    for m in sorted(all_results.keys()):
        print(f"   {m:<5}", end="")
        for lag in lags:
            prod = lag * scale_data[m][f'a_{lag}']
            print(f"{prod:>10.3f}", end="")
        print()

    # Vérifier si 8×a_8 ≈ 13×a_13 à chaque échelle
    print(f"\n   Vérification 8×a_8 ≈ 13×a_13:")
    for m in sorted(all_results.keys()):
        prod_8 = 8 * scale_data[m]['a_8']
        prod_13 = 13 * scale_data[m]['a_13']
        diff = abs(prod_8 - prod_13)
        avg = (prod_8 + prod_13) / 2
        dev_pct = diff / avg * 100 if avg != 0 else 0
        status = "✓" if dev_pct < 5 else "✗"
        print(f"   m={m}: 8×a_8={prod_8:.3f}, 13×a_13={prod_13:.3f}, Δ={dev_pct:.1f}% {status}")

    # ============================================================
    # EXPORT
    # ============================================================

    output = {
        'decimation_analysis': {
            str(m): {
                'n_zeros': len(decimate(gammas, m)),
                'avg_coefficients': scale_data[m]
            }
            for m in all_results.keys()
        },
        'scaling_test': {
            'decimation_factors': list(all_results.keys()),
            'invariant_8_times_a8_equals_13_times_a13': {
                str(m): {
                    '8_times_a8': 8 * scale_data[m]['a_8'],
                    '13_times_a13': 13 * scale_data[m]['a_13'],
                    'deviation_pct': abs(8*scale_data[m]['a_8'] - 13*scale_data[m]['a_13']) /
                                    ((8*scale_data[m]['a_8'] + 13*scale_data[m]['a_13'])/2) * 100
                }
                for m in all_results.keys()
            }
        }
    }

    with open('phase28_rg_decimation.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n💾 Résultats sauvegardés dans phase28_rg_decimation.json")

if __name__ == "__main__":
    main()
