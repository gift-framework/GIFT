#!/usr/bin/env python3
"""
GIFT Phase 2.6: RG Flow Fitting
===============================

Modéliser le drift des coefficients comme un flow de renormalisation.
Chercher si les paramètres sont liés aux constantes GIFT.

Hypothèses testées:
1. Transition tanh: a(γ) = a_UV + (a_IR - a_UV) × (1 - tanh((γ - γ_c)/Δ))/2
2. Power law: a(γ) = a_UV + (a_IR - a_UV) / (1 + (γ/γ_c)^β)
3. Logarithmique: a(γ) = a_0 + b/ln(γ/Λ)
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import json

# ============================================================
# GIFT CONSTANTS
# ============================================================
GIFT = {
    'dim_G2': 14,
    'b2': 21,
    'b3': 77,
    'H_star': 99,
    'dim_K7': 7,
    'rank_E8': 8,
    'dim_E8': 248,
    'dim_J3O': 27,
    'Weyl': 5,
    'N_gen': 3,
    'h_G2': 6,
    'p2': 2,
    'dim_F4': 52,
    'L8': 47,  # Lucas(8)
}

# Ratios GIFT calibrés (régime IR)
GIFT_IR = {
    'a_5': 8/77,      # rank(E8)/b3
    'a_8': 5/27,      # Weyl/dim(J3O)
    'a_13': 64/248,   # rank(E8)²/dim(E8)
    'a_27': 34/77,    # (27+7)/b3
    'c': 91/7,        # (b3+14)/dim(K7)
}

# ============================================================
# FLOW MODELS
# ============================================================

def model_tanh(gamma, a_ir, a_uv, gamma_c, delta):
    """Transition tanh (type phase transition)"""
    return a_uv + (a_ir - a_uv) * (1 - np.tanh((gamma - gamma_c) / delta)) / 2

def model_power(gamma, a_ir, a_uv, gamma_c, beta):
    """Power law decay"""
    return a_uv + (a_ir - a_uv) / (1 + (gamma / gamma_c) ** beta)

def model_log(gamma, a_0, b, Lambda):
    """Logarithmic running (QCD-like)"""
    return a_0 + b / np.log(gamma / Lambda)

def model_exp(gamma, a_ir, a_uv, gamma_c):
    """Exponential decay"""
    return a_uv + (a_ir - a_uv) * np.exp(-gamma / gamma_c)

# ============================================================
# FITTING FUNCTIONS
# ============================================================

def fit_coefficient(gammas, values, coef_name, gift_ir_val):
    """
    Fitte un coefficient avec plusieurs modèles.
    Retourne le meilleur fit et ses paramètres.
    """
    results = {}

    # Estimation initiale des asymptotes
    a_ir_est = values[0]  # Première valeur ≈ IR
    a_uv_est = values[-1]  # Dernière valeur ≈ UV
    gamma_c_est = 500000   # Point critique estimé

    # 1. Modèle tanh
    try:
        popt, pcov = curve_fit(
            model_tanh, gammas, values,
            p0=[a_ir_est, a_uv_est, gamma_c_est, 200000],
            bounds=([-2, -2, 10000, 1000], [2, 2, 2000000, 1000000]),
            maxfev=10000
        )
        pred = model_tanh(gammas, *popt)
        r2 = 1 - np.sum((values - pred)**2) / np.sum((values - np.mean(values))**2)
        results['tanh'] = {
            'params': {'a_IR': popt[0], 'a_UV': popt[1], 'gamma_c': popt[2], 'delta': popt[3]},
            'r2': r2,
            'pred': pred
        }
    except Exception as e:
        results['tanh'] = {'error': str(e)}

    # 2. Modèle power law
    try:
        popt, pcov = curve_fit(
            model_power, gammas, values,
            p0=[a_ir_est, a_uv_est, gamma_c_est, 1.0],
            bounds=([-2, -2, 1000, 0.01], [2, 2, 2000000, 10]),
            maxfev=10000
        )
        pred = model_power(gammas, *popt)
        r2 = 1 - np.sum((values - pred)**2) / np.sum((values - np.mean(values))**2)
        results['power'] = {
            'params': {'a_IR': popt[0], 'a_UV': popt[1], 'gamma_c': popt[2], 'beta': popt[3]},
            'r2': r2,
            'pred': pred
        }
    except Exception as e:
        results['power'] = {'error': str(e)}

    # 3. Modèle logarithmique
    try:
        popt, pcov = curve_fit(
            model_log, gammas, values,
            p0=[a_uv_est, (a_ir_est - a_uv_est) * 10, 1000],
            bounds=([-2, -100, 1], [2, 100, 100000]),
            maxfev=10000
        )
        pred = model_log(gammas, *popt)
        r2 = 1 - np.sum((values - pred)**2) / np.sum((values - np.mean(values))**2)
        results['log'] = {
            'params': {'a_0': popt[0], 'b': popt[1], 'Lambda': popt[2]},
            'r2': r2,
            'pred': pred
        }
    except Exception as e:
        results['log'] = {'error': str(e)}

    # 4. Modèle exponentiel
    try:
        popt, pcov = curve_fit(
            model_exp, gammas, values,
            p0=[a_ir_est, a_uv_est, gamma_c_est],
            bounds=([-2, -2, 10000], [2, 2, 2000000]),
            maxfev=10000
        )
        pred = model_exp(gammas, *popt)
        r2 = 1 - np.sum((values - pred)**2) / np.sum((values - np.mean(values))**2)
        results['exp'] = {
            'params': {'a_IR': popt[0], 'a_UV': popt[1], 'gamma_c': popt[2]},
            'r2': r2,
            'pred': pred
        }
    except Exception as e:
        results['exp'] = {'error': str(e)}

    # Trouver le meilleur modèle
    best_model = None
    best_r2 = -np.inf
    for name, res in results.items():
        if 'r2' in res and res['r2'] > best_r2:
            best_r2 = res['r2']
            best_model = name

    return results, best_model

def find_gift_ratios(value, tol=0.05):
    """Cherche si une valeur correspond à un ratio GIFT."""
    matches = []

    # Ratios simples
    for n1, v1 in GIFT.items():
        for n2, v2 in GIFT.items():
            if v2 != 0:
                ratio = v1 / v2
                if abs(value) > 0.001:
                    rel_err = abs(ratio - value) / abs(value)
                    if rel_err < tol:
                        matches.append((f"{n1}/{n2}", ratio, rel_err * 100))

    # Valeurs directes
    for n, v in GIFT.items():
        if abs(value) > 0.001:
            rel_err = abs(v - value) / abs(value)
            if rel_err < tol:
                matches.append((n, v, rel_err * 100))

    # Produits b3 × dim_K7
    prod_539 = GIFT['b3'] * GIFT['dim_K7']  # 539
    if abs(value) > 100:
        for n, v in GIFT.items():
            candidate = prod_539 * v
            rel_err = abs(candidate - value) / abs(value)
            if rel_err < tol:
                matches.append((f"b3×dim_K7×{n}", candidate, rel_err * 100))

    matches.sort(key=lambda x: x[2])
    return matches[:5]

# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():
    print("="*70)
    print("GIFT Phase 2.6: RG FLOW FITTING")
    print("="*70)

    # Charger les résultats de l'unfolding
    print("\n📂 Chargement des données...")
    try:
        with open('phase25_unfolding_results.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("❌ Fichier phase25_unfolding_results.json non trouvé")
        print("   Lancez d'abord test_unfolding.py")
        return

    # Extraire les données de fenêtres (raw)
    windows = data['window_results']['raw']

    # Construire les arrays
    # Utiliser gamma moyen de chaque fenêtre (approximé)
    # Window i commence à i*50000, donc gamma_mid ≈ N^{-1}((i+0.5)*50000)
    # Approximation: gamma ≈ 2π × n / ln(n) pour grand n

    n_mids = np.array([w['window_mid_idx'] for w in windows])
    # Approximation grossière de gamma à partir de l'index
    # gamma ≈ 2π × n / ln(n/e) pour n > 1000
    gammas = 2 * np.pi * n_mids / np.log(n_mids / np.e + 1)

    coefficients = {
        'a_5': np.array([w['coefficients']['a_5'] for w in windows]),
        'a_8': np.array([w['coefficients']['a_8'] for w in windows]),
        'a_13': np.array([w['coefficients']['a_13'] for w in windows]),
        'a_27': np.array([w['coefficients']['a_27'] for w in windows]),
        'c': np.array([w['coefficients']['c'] for w in windows]),
    }

    print(f"   {len(windows)} fenêtres chargées")
    print(f"   γ range: [{gammas[0]:.0f}, {gammas[-1]:.0f}]")

    # Fitter chaque coefficient
    all_results = {}

    print("\n" + "="*70)
    print("FITTING DES COEFFICIENTS")
    print("="*70)

    for coef_name, values in coefficients.items():
        print(f"\n{'─'*60}")
        print(f"📊 {coef_name}")
        print(f"{'─'*60}")

        gift_ir = GIFT_IR.get(coef_name, None)

        results, best_model = fit_coefficient(gammas, values, coef_name, gift_ir)
        all_results[coef_name] = {'fits': results, 'best_model': best_model}

        print(f"\n   Valeur initiale (IR): {values[0]:.4f}")
        print(f"   Valeur finale (UV):   {values[-1]:.4f}")
        if gift_ir:
            print(f"   GIFT calibré:         {gift_ir:.4f}")

        print(f"\n   Modèle    │ R²      │ Paramètres clés")
        print(f"   {'─'*50}")

        for model_name in ['tanh', 'power', 'log', 'exp']:
            if model_name in results:
                res = results[model_name]
                if 'error' in res:
                    print(f"   {model_name:<8} │ ERREUR  │ {res['error'][:30]}")
                else:
                    r2 = res['r2']
                    params = res['params']
                    marker = "★" if model_name == best_model else " "

                    if model_name == 'tanh':
                        print(f" {marker} {model_name:<8} │ {r2:.4f}  │ γ_c={params['gamma_c']:.0f}, Δ={params['delta']:.0f}")
                    elif model_name == 'power':
                        print(f" {marker} {model_name:<8} │ {r2:.4f}  │ γ_c={params['gamma_c']:.0f}, β={params['beta']:.3f}")
                    elif model_name == 'log':
                        print(f" {marker} {model_name:<8} │ {r2:.4f}  │ Λ={params['Lambda']:.0f}, b={params['b']:.3f}")
                    elif model_name == 'exp':
                        print(f" {marker} {model_name:<8} │ {r2:.4f}  │ γ_c={params['gamma_c']:.0f}")

        # Analyser les paramètres du meilleur modèle
        if best_model and 'params' in results[best_model]:
            params = results[best_model]['params']
            print(f"\n   🎯 Meilleur modèle: {best_model}")

            # Chercher des ratios GIFT dans les paramètres
            for pname, pval in params.items():
                if pname in ['gamma_c', 'delta', 'Lambda'] and abs(pval) > 100:
                    matches = find_gift_ratios(pval, tol=0.1)
                    if matches:
                        print(f"      {pname} = {pval:.0f} ≈ {matches[0][0]} = {matches[0][1]:.0f} ({matches[0][2]:.1f}%)")
                elif pname == 'beta':
                    # Chercher si beta est un ratio simple
                    for n, v in GIFT.items():
                        if v != 0 and abs(pval - 1/v) < 0.05:
                            print(f"      β = {pval:.4f} ≈ 1/{n} = {1/v:.4f}")
                        if abs(pval - v/77) < 0.05:
                            print(f"      β = {pval:.4f} ≈ {n}/b₃ = {v/77:.4f}")

    # Analyse du point critique γ_c
    print("\n" + "="*70)
    print("ANALYSE DU POINT CRITIQUE γ_c")
    print("="*70)

    gamma_c_values = []
    for coef_name, res in all_results.items():
        best = res['best_model']
        if best and 'params' in res['fits'][best]:
            params = res['fits'][best]['params']
            if 'gamma_c' in params:
                gamma_c_values.append((coef_name, params['gamma_c']))

    if gamma_c_values:
        print(f"\n   Coefficient │ γ_c (du fit)")
        print(f"   {'─'*30}")
        for name, gc in gamma_c_values:
            print(f"   {name:<10}   │ {gc:,.0f}")

        avg_gc = np.mean([gc for _, gc in gamma_c_values])
        std_gc = np.std([gc for _, gc in gamma_c_values])
        print(f"\n   Moyenne: {avg_gc:,.0f} ± {std_gc:,.0f}")

        # Décomposition GIFT
        print(f"\n   Décompositions GIFT de γ_c ≈ {avg_gc:,.0f}:")
        print(f"   ─────────────────────────────────────")

        b3_k7 = GIFT['b3'] * GIFT['dim_K7']  # 539
        factor = avg_gc / b3_k7
        print(f"   γ_c / (b₃ × dim_K₇) = {avg_gc:.0f} / 539 = {factor:.1f}")
        print(f"   (Rappel: 1007 = 19 × 53, et 19 - 53 = -34 = -(27+7))")

        # Test si factor proche d'un entier × quelque chose
        for mult in [1, 2, 3, 5, 7, 8, 13, 14, 21, 27]:
            if abs(factor / mult - round(factor / mult)) < 0.1:
                print(f"   {factor:.1f} ≈ {round(factor/mult)} × {mult}")

    # Chercher le point où a_27 = 0
    print("\n" + "="*70)
    print("POINT CRITIQUE: a_27 = 0")
    print("="*70)

    a27 = coefficients['a_27']
    # Trouver où a_27 change de signe
    sign_changes = np.where(np.diff(np.sign(a27)))[0]
    if len(sign_changes) > 0:
        idx = sign_changes[0]
        # Interpolation linéaire
        gamma_zero = gammas[idx] - a27[idx] * (gammas[idx+1] - gammas[idx]) / (a27[idx+1] - a27[idx])
        print(f"\n   a_27 change de signe entre fenêtres {idx} et {idx+1}")
        print(f"   γ_critique (interpolé) ≈ {gamma_zero:,.0f}")

        # Comparaison avec 542,655 (valeur précédente)
        print(f"\n   Comparaison avec analyse précédente:")
        print(f"   γ_c (précédent) = 542,655")
        print(f"   γ_c (fit)       = {gamma_zero:,.0f}")
        print(f"   Écart: {abs(gamma_zero - 542655) / 542655 * 100:.1f}%")

    # Export
    print("\n" + "="*70)
    print("EXPORT")
    print("="*70)

    output = {
        'analysis': 'RG Flow Fitting',
        'n_windows': len(windows),
        'gamma_range': [float(gammas[0]), float(gammas[-1])],
        'fits': {}
    }

    for coef_name, res in all_results.items():
        output['fits'][coef_name] = {
            'best_model': res['best_model'],
            'models': {}
        }
        for model_name, model_res in res['fits'].items():
            if 'params' in model_res:
                output['fits'][coef_name]['models'][model_name] = {
                    'r2': float(model_res['r2']),
                    'params': {k: float(v) for k, v in model_res['params'].items()}
                }

    if gamma_c_values:
        output['gamma_c_mean'] = float(avg_gc)
        output['gamma_c_std'] = float(std_gc)

    with open('phase26_rg_flow_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n💾 Résultats sauvegardés dans phase26_rg_flow_results.json")

    return output

if __name__ == "__main__":
    main()
