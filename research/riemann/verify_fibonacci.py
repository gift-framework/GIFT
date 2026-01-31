#!/usr/bin/env python3
"""
GIFT Phase 2.7: Fibonacci Constraint Verification
==================================================

Test de l'insight d'Opus: β₈/β₁₃ = 13/8 (ratio Fibonacci)

Si 8×β₈ = 13×β₁₃, alors β₈/β₁₃ = 13/8 = F₇/F₆ → φ (nombre d'or)

Ce script:
1. Vérifie toutes les contraintes de ratio entre β
2. Refitte avec contraintes Fibonacci imposées
3. Compare R² avant/après
"""

import numpy as np
from scipy.optimize import curve_fit, minimize
import json

# ============================================================
# DONNÉES DES FITS PRÉCÉDENTS
# ============================================================

# β extraits du fit power law (Phase 2.6)
BETA_MEASURED = {
    5: 0.767,
    8: 4.497,
    13: 2.764,
    27: 3.106
}

# Produits lag × β mesurés
LAG_BETA_MEASURED = {
    5: 5 * 0.767,    # = 3.835
    8: 8 * 4.497,    # = 35.976
    13: 13 * 2.764,  # = 35.932
    27: 27 * 3.106   # = 83.862
}

# Targets GIFT
LAG_BETA_GIFT = {
    5: 27/7,   # = 3.857
    8: 36,     # = h_G2²
    13: 36,    # = h_G2²
    27: 84     # = b3 + dim_K7
}

# Fibonacci
FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
PHI = (1 + np.sqrt(5)) / 2  # Nombre d'or

# ============================================================
# VÉRIFICATION DES CONTRAINTES FIBONACCI
# ============================================================

def verify_fibonacci_constraints():
    """Vérifie les contraintes de ratio Fibonacci entre β."""

    print("="*70)
    print("VÉRIFICATION DES CONTRAINTES FIBONACCI")
    print("="*70)

    # 1. Ratio β₈/β₁₃
    print("\n📊 1. Ratio β₈/β₁₃")
    print("-"*40)

    ratio_measured = BETA_MEASURED[8] / BETA_MEASURED[13]
    ratio_fibonacci = 13 / 8  # Inverse car β₈ > β₁₃

    print(f"   β₈/β₁₃ mesuré:    {ratio_measured:.4f}")
    print(f"   13/8 (Fibonacci): {ratio_fibonacci:.4f}")
    print(f"   Déviation:        {abs(ratio_measured - ratio_fibonacci)/ratio_fibonacci * 100:.2f}%")

    # Vérifier si c'est F₇/F₆
    print(f"\n   Note: 13/8 = F₇/F₆ (Fibonacci consécutifs)")
    print(f"   Limite: F_{{n+1}}/F_n → φ = {PHI:.6f}")

    # 2. Ratio β₅/β₈
    print("\n📊 2. Ratio β₅/β₈")
    print("-"*40)

    ratio_5_8 = BETA_MEASURED[5] / BETA_MEASURED[8]
    ratio_fib_5_8 = 8 / 5  # = 1.6

    print(f"   β₅/β₈ mesuré:     {ratio_5_8:.4f}")
    print(f"   8/5 (Fibonacci):  {ratio_fib_5_8:.4f}")
    print(f"   Déviation:        {abs(ratio_5_8 - 1/ratio_fib_5_8)/(1/ratio_fib_5_8) * 100:.2f}%")
    print(f"   (ou β₈/β₅ = {1/ratio_5_8:.4f} vs 8/5 = {ratio_fib_5_8:.4f})")

    # 3. Tous les ratios
    print("\n📊 3. Matrice des ratios βᵢ/βⱼ")
    print("-"*40)

    lags = [5, 8, 13, 27]
    print(f"\n   {'βᵢ/βⱼ':<8}", end="")
    for j in lags:
        print(f"{j:>10}", end="")
    print()
    print("   " + "-"*48)

    for i in lags:
        print(f"   {i:<8}", end="")
        for j in lags:
            ratio = BETA_MEASURED[i] / BETA_MEASURED[j]
            print(f"{ratio:>10.3f}", end="")
        print()

    # 4. Comparaison avec ratios de lags
    print("\n📊 4. Comparaison βᵢ/βⱼ vs j/i")
    print("-"*40)
    print(f"\n   {'Paire':<12} {'βᵢ/βⱼ':>10} {'j/i':>10} {'Dév.':>10}")
    print("   " + "-"*45)

    pairs = [(8, 13), (5, 8), (5, 13), (8, 27), (13, 27)]
    fibonacci_pairs = []

    for i, j in pairs:
        ratio_beta = BETA_MEASURED[i] / BETA_MEASURED[j]
        ratio_lag = j / i
        dev = abs(ratio_beta - ratio_lag) / ratio_lag * 100

        # Vérifier si c'est un ratio Fibonacci
        is_fib = (i in FIBONACCI and j in FIBONACCI and
                  abs(FIBONACCI.index(j) - FIBONACCI.index(i)) == 1)
        marker = "★ Fib!" if is_fib else ""

        print(f"   β_{i}/β_{j:<4} {ratio_beta:>10.4f} {ratio_lag:>10.4f} {dev:>9.2f}% {marker}")

        if is_fib:
            fibonacci_pairs.append((i, j, ratio_beta, ratio_lag, dev))

    # 5. Test de la contrainte générale
    print("\n📊 5. Test: lag_i × β_i = lag_j × β_j (pour Fibonacci adjacents)")
    print("-"*40)

    print(f"\n   {'Paire':<12} {'lag_i×β_i':>12} {'lag_j×β_j':>12} {'Dév.':>10}")
    print("   " + "-"*50)

    for i, j in [(5, 8), (8, 13)]:
        prod_i = i * BETA_MEASURED[i]
        prod_j = j * BETA_MEASURED[j]
        dev = abs(prod_i - prod_j) / ((prod_i + prod_j) / 2) * 100
        print(f"   ({i}, {j})      {prod_i:>12.4f} {prod_j:>12.4f} {dev:>9.2f}%")

    # 6. Vérification de la somme
    print("\n📊 6. Somme des β")
    print("-"*40)

    sum_beta = sum(BETA_MEASURED.values())
    target = 77/7
    print(f"   Σβᵢ = {sum_beta:.4f}")
    print(f"   b₃/dim(K₇) = 77/7 = {target:.4f}")
    print(f"   Déviation: {abs(sum_beta - target)/target * 100:.2f}%")

    return fibonacci_pairs

# ============================================================
# MODÈLE POWER LAW AVEC CONTRAINTES
# ============================================================

def power_law_free(gamma, a_ir, a_uv, gamma_c, beta):
    """Power law sans contrainte."""
    return a_uv + (a_ir - a_uv) / (1 + (gamma / gamma_c) ** beta)

def power_law_constrained_8_13(gamma, a_ir_8, a_uv_8, gamma_c_8,
                                     a_ir_13, a_uv_13, gamma_c_13,
                                     beta_8):
    """
    Power law avec contrainte β₈/β₁₃ = 13/8.
    β₁₃ est dérivé de β₈.
    """
    beta_13 = beta_8 * 8 / 13  # Contrainte Fibonacci

    pred_8 = a_uv_8 + (a_ir_8 - a_uv_8) / (1 + (gamma / gamma_c_8) ** beta_8)
    pred_13 = a_uv_13 + (a_ir_13 - a_uv_13) / (1 + (gamma / gamma_c_13) ** beta_13)

    return pred_8, pred_13

def fit_with_fibonacci_constraint(gammas, a8_values, a13_values):
    """
    Fit simultané de a₈ et a₁₃ avec contrainte β₈/β₁₃ = 13/8.
    """

    def objective(params):
        a_ir_8, a_uv_8, gamma_c_8, a_ir_13, a_uv_13, gamma_c_13, beta_8 = params

        beta_13 = beta_8 * 8 / 13  # Contrainte!

        pred_8 = a_uv_8 + (a_ir_8 - a_uv_8) / (1 + (gammas / gamma_c_8) ** beta_8)
        pred_13 = a_uv_13 + (a_ir_13 - a_uv_13) / (1 + (gammas / gamma_c_13) ** beta_13)

        # MSE combiné
        mse_8 = np.mean((pred_8 - a8_values) ** 2)
        mse_13 = np.mean((pred_13 - a13_values) ** 2)

        return mse_8 + mse_13

    # Initial guess
    x0 = [
        a8_values[0], a8_values[-1], 400000,   # a8 params
        a13_values[0], a13_values[-1], 300000, # a13 params
        4.5  # beta_8 initial
    ]

    # Bounds
    bounds = [
        (0, 1), (-1, 1), (10000, 2000000),     # a8
        (0, 1), (0, 1), (10000, 2000000),      # a13
        (0.1, 10)                               # beta_8
    ]

    result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')

    return result

# ============================================================
# MAIN
# ============================================================

def main():
    print("="*70)
    print("GIFT Phase 2.7: FIBONACCI CONSTRAINT VERIFICATION")
    print("="*70)

    # 1. Vérifier les contraintes avec les β existants
    fib_pairs = verify_fibonacci_constraints()

    # 2. Charger les données et refitter
    print("\n" + "="*70)
    print("REFIT AVEC CONTRAINTE β₈/β₁₃ = 13/8")
    print("="*70)

    try:
        with open('phase25_unfolding_results.json', 'r') as f:
            data = json.load(f)

        windows = data['window_results']['raw']
        n_mids = np.array([w['window_mid_idx'] for w in windows])
        gammas = 2 * np.pi * n_mids / np.log(n_mids / np.e + 1)

        a8_values = np.array([w['coefficients']['a_8'] for w in windows])
        a13_values = np.array([w['coefficients']['a_13'] for w in windows])

        # Fit libre (rappel des résultats)
        print("\n📊 Rappel: Fit LIBRE")
        print(f"   β₈ = 4.497, β₁₃ = 2.764")
        print(f"   β₈/β₁₃ = {4.497/2.764:.4f}")
        print(f"   13/8 = {13/8:.4f}")

        # Calcul R² du fit libre
        def calc_r2(y_true, y_pred):
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - ss_res / ss_tot

        # Prédictions avec β libres
        pred_8_free = power_law_free(gammas, a8_values[0], a8_values[-1], 386499, 4.497)
        pred_13_free = power_law_free(gammas, a13_values[0], a13_values[-1], 287669, 2.764)

        r2_8_free = calc_r2(a8_values, pred_8_free)
        r2_13_free = calc_r2(a13_values, pred_13_free)

        print(f"\n   R²(a₈) = {r2_8_free:.4f}")
        print(f"   R²(a₁₃) = {r2_13_free:.4f}")

        # Fit contraint
        print("\n📊 Fit CONTRAINT (β₈/β₁₃ = 13/8)")
        result = fit_with_fibonacci_constraint(gammas, a8_values, a13_values)

        params = result.x
        a_ir_8, a_uv_8, gamma_c_8 = params[0], params[1], params[2]
        a_ir_13, a_uv_13, gamma_c_13 = params[3], params[4], params[5]
        beta_8_constrained = params[6]
        beta_13_constrained = beta_8_constrained * 8 / 13

        print(f"\n   β₈ = {beta_8_constrained:.4f}")
        print(f"   β₁₃ = {beta_13_constrained:.4f} (= β₈ × 8/13)")
        print(f"   β₈/β₁₃ = {beta_8_constrained/beta_13_constrained:.4f} (EXACT 13/8 par construction)")

        # Prédictions contraintes
        pred_8_cons = power_law_free(gammas, a_ir_8, a_uv_8, gamma_c_8, beta_8_constrained)
        pred_13_cons = power_law_free(gammas, a_ir_13, a_uv_13, gamma_c_13, beta_13_constrained)

        r2_8_cons = calc_r2(a8_values, pred_8_cons)
        r2_13_cons = calc_r2(a13_values, pred_13_cons)

        print(f"\n   R²(a₈) = {r2_8_cons:.4f}")
        print(f"   R²(a₁₃) = {r2_13_cons:.4f}")

        # Comparaison
        print("\n📊 COMPARAISON")
        print("-"*50)
        print(f"\n   {'Métrique':<20} {'Libre':>12} {'Contraint':>12} {'Δ':>10}")
        print("   " + "-"*55)
        print(f"   {'R²(a₈)':<20} {r2_8_free:>12.4f} {r2_8_cons:>12.4f} {r2_8_cons - r2_8_free:>+10.4f}")
        print(f"   {'R²(a₁₃)':<20} {r2_13_free:>12.4f} {r2_13_cons:>12.4f} {r2_13_cons - r2_13_free:>+10.4f}")
        print(f"   {'β₈':<20} {4.497:>12.4f} {beta_8_constrained:>12.4f} {beta_8_constrained - 4.497:>+10.4f}")
        print(f"   {'β₁₃':<20} {2.764:>12.4f} {beta_13_constrained:>12.4f} {beta_13_constrained - 2.764:>+10.4f}")

        # Verdict
        print("\n" + "="*70)
        print("VERDICT")
        print("="*70)

        r2_loss = (r2_8_free + r2_13_free) / 2 - (r2_8_cons + r2_13_cons) / 2

        if r2_loss < 0.01:
            print(f"\n   ✅ La contrainte Fibonacci β₈/β₁₃ = 13/8 est COMPATIBLE!")
            print(f"   Perte de R² moyenne: {r2_loss:.4f} (< 1%)")
            print(f"\n   → La structure Fibonacci est RÉELLE, pas un artefact du fit libre.")
            verdict = "CONFIRMED"
        elif r2_loss < 0.05:
            print(f"\n   ⚠️ La contrainte est PARTIELLEMENT compatible")
            print(f"   Perte de R² moyenne: {r2_loss:.4f}")
            verdict = "PARTIAL"
        else:
            print(f"\n   ❌ La contrainte DÉGRADE significativement le fit")
            print(f"   Perte de R² moyenne: {r2_loss:.4f}")
            verdict = "REJECTED"

        # Vérification du produit lag × β
        print("\n📊 Vérification lag × β avec β contraints")
        print("-"*50)
        print(f"   8 × β₈ = 8 × {beta_8_constrained:.4f} = {8 * beta_8_constrained:.4f}")
        print(f"   13 × β₁₃ = 13 × {beta_13_constrained:.4f} = {13 * beta_13_constrained:.4f}")
        print(f"   Cible: 36 = h_G₂²")

        # Export
        output = {
            'fibonacci_constraint': {
                'beta_8_over_beta_13': {
                    'measured': 4.497 / 2.764,
                    'target': 13/8,
                    'deviation_pct': abs(4.497/2.764 - 13/8) / (13/8) * 100
                }
            },
            'constrained_fit': {
                'beta_8': float(beta_8_constrained),
                'beta_13': float(beta_13_constrained),
                'gamma_c_8': float(gamma_c_8),
                'gamma_c_13': float(gamma_c_13),
                'r2_8': float(r2_8_cons),
                'r2_13': float(r2_13_cons)
            },
            'comparison': {
                'r2_loss': float(r2_loss),
                'verdict': verdict
            },
            'lag_times_beta_constrained': {
                '8_times_beta_8': float(8 * beta_8_constrained),
                '13_times_beta_13': float(13 * beta_13_constrained)
            }
        }

        with open('phase27_fibonacci_verification.json', 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\n💾 Résultats sauvegardés dans phase27_fibonacci_verification.json")

    except FileNotFoundError:
        print("\n⚠️ Fichier phase25_unfolding_results.json non trouvé")
        print("   Lancez d'abord test_unfolding.py")

if __name__ == "__main__":
    main()
