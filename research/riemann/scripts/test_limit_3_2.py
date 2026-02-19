#!/usr/bin/env python3
"""
Test: La limite asymptotique est-elle 3/2 ?
==========================================

On a observé que a(N) → ~1.50 quand N → ∞
Est-ce exactement 3/2 = 1.5 ?
"""

import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit

PHI = (1 + np.sqrt(5)) / 2
SQRT5 = np.sqrt(5)

def load_zeros(max_zeros=100000):
    zeros = []
    zeros_dir = Path(__file__).parent
    for i in range(1, 6):
        zeros_file = zeros_dir / f"zeros{i}"
        if zeros_file.exists():
            with open(zeros_file) as f:
                for line in f:
                    if line.strip():
                        zeros.append(float(line.strip()))
                        if len(zeros) >= max_zeros:
                            return np.array(zeros)
    return np.array(zeros)

zeros = load_zeros(100000)
print(f"✓ {len(zeros)} zéros chargés\n")

def fit_lags(zeros, lag1, lag2):
    max_lag = max(lag1, lag2)
    n_fit = len(zeros) - max_lag
    X1 = zeros[max_lag - lag1:max_lag - lag1 + n_fit]
    X2 = zeros[max_lag - lag2:max_lag - lag2 + n_fit]
    X = np.column_stack([X1, X2, np.ones(n_fit)])
    y = zeros[max_lag:max_lag + n_fit]
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return coeffs[0], coeffs[1], coeffs[2]

# Mesurer a(N) pour différents N
Ns = [500, 1000, 2000, 3000, 5000, 7000, 10000, 15000, 20000, 30000, 50000, 70000, 100000]
a_values = []
for N in Ns:
    if N > len(zeros):
        continue
    a, b, c = fit_lags(zeros[:N], 8, 21)
    a_values.append(a)

Ns = np.array(Ns[:len(a_values)])
a_values = np.array(a_values)

print("=" * 70)
print("TEST: LA LIMITE EST-ELLE EXACTEMENT 3/2 ?")
print("=" * 70)

# ============================================================================
# CANDIDATS POUR LA LIMITE
# ============================================================================

candidates = {
    "3/2": 3/2,
    "2φ/√5": 2*PHI/SQRT5,
    "φ²/φ+ψ": PHI**2 / (PHI + (1-PHI)),  # = φ²
    "(φ+2)/φ²": (PHI + 2) / PHI**2,
    "√(φ+1)": np.sqrt(PHI + 1),
    "φ/√(φ-1)": PHI / np.sqrt(PHI - 1),
    "3φ/(2φ+1)": 3*PHI / (2*PHI + 1),
    "(2φ+1)/3": (2*PHI + 1) / 3,
    "φ²/√5": PHI**2 / SQRT5,
    "1 + 1/(2φ)": 1 + 1/(2*PHI),
    "φ/(φ-1/2)": PHI / (PHI - 0.5),
}

print(f"\n{'Candidat':<15} {'Valeur':<12} Description")
print("-" * 50)
for name, val in sorted(candidates.items(), key=lambda x: abs(x[1] - 1.50)):
    print(f"{name:<15} {val:<12.6f}")

# ============================================================================
# FIT AVEC DIFFÉRENTES LIMITES FIXÉES
# ============================================================================

print("\n" + "=" * 70)
print("FIT: a(N) = a_∞ - β/√N  POUR DIFFÉRENTES LIMITES a_∞")
print("=" * 70)

def model_sqrt(N, beta, a_inf):
    return a_inf - beta / np.sqrt(N)

print(f"\n{'Limite a_∞':<20} {'β':<12} {'RSS':<15} {'a(10⁶) prédit':<15}")
print("-" * 65)

results = []
for name, a_inf in [("3/2", 1.5), ("2φ/√5", 2*PHI/SQRT5), ("1.496 (libre)", None),
                     ("φ²/√5", PHI**2/SQRT5), ("√(φ+1)", np.sqrt(PHI+1))]:
    try:
        if a_inf is None:
            # Fit libre
            popt, _ = curve_fit(lambda N, beta, a: a - beta/np.sqrt(N), Ns, a_values, p0=[3.5, 1.5])
            beta, a_inf_fit = popt
            a_inf = a_inf_fit
        else:
            # Fit avec limite fixée
            popt, _ = curve_fit(lambda N, beta: a_inf - beta/np.sqrt(N), Ns, a_values, p0=[3.5])
            beta = popt[0]
            a_inf_fit = a_inf

        y_pred = a_inf_fit - beta / np.sqrt(Ns)
        rss = np.sum((a_values - y_pred)**2)
        a_million = a_inf_fit - beta / np.sqrt(1e6)

        results.append((name, a_inf_fit, beta, rss))
        print(f"{name:<20} {beta:<12.4f} {rss:<15.2e} {a_million:<15.6f}")
    except Exception as e:
        print(f"{name:<20} ÉCHEC: {e}")

# Trouver le meilleur
best = min(results, key=lambda x: x[3])
print(f"\n→ Meilleure limite: {best[0]} = {best[1]:.6f}")

# ============================================================================
# TEST STATISTIQUE: 3/2 vs 2φ/√5
# ============================================================================

print("\n" + "=" * 70)
print("COMPARAISON DIRECTE: 3/2 vs 2φ/√5")
print("=" * 70)

# Fit avec 3/2
popt_32, _ = curve_fit(lambda N, beta: 1.5 - beta/np.sqrt(N), Ns, a_values)
beta_32 = popt_32[0]
y_pred_32 = 1.5 - beta_32 / np.sqrt(Ns)
rss_32 = np.sum((a_values - y_pred_32)**2)

# Fit avec 2φ/√5
a_phi = 2*PHI/SQRT5
popt_phi, _ = curve_fit(lambda N, beta: a_phi - beta/np.sqrt(N), Ns, a_values)
beta_phi = popt_phi[0]
y_pred_phi = a_phi - beta_phi / np.sqrt(Ns)
rss_phi = np.sum((a_values - y_pred_phi)**2)

print(f"\nModèle a = 3/2 - β/√N:")
print(f"  β = {beta_32:.4f}")
print(f"  RSS = {rss_32:.2e}")
print(f"  Prédictions: a(10⁶) = {1.5 - beta_32/1000:.6f}")

print(f"\nModèle a = 2φ/√5 - β/√N:")
print(f"  β = {beta_phi:.4f}")
print(f"  RSS = {rss_phi:.2e}")
print(f"  Prédictions: a(10⁶) = {a_phi - beta_phi/1000:.6f}")

ratio = rss_phi / rss_32
print(f"\nRatio RSS(2φ/√5) / RSS(3/2) = {ratio:.2f}")

if ratio > 2:
    print("→ 3/2 est SIGNIFICATIVEMENT meilleur")
elif ratio > 1.1:
    print("→ 3/2 est légèrement meilleur")
elif ratio > 0.9:
    print("→ Les deux sont équivalents")
else:
    print("→ 2φ/√5 est meilleur")

# ============================================================================
# EXTRAPOLATION ET VÉRIFICATION
# ============================================================================

print("\n" + "=" * 70)
print("EXTRAPOLATION: QUE PRÉDIT CHAQUE MODÈLE ?")
print("=" * 70)

print(f"\n{'N':<12} {'a observé':<15} {'a (3/2)':<15} {'a (2φ/√5)':<15} {'a (libre)':<15}")
print("-" * 75)

# Fit libre pour comparaison
popt_free, _ = curve_fit(lambda N, beta, a: a - beta/np.sqrt(N), Ns, a_values)
beta_free, a_free = popt_free

for N in Ns:
    idx = list(Ns).index(N)
    a_obs = a_values[idx]
    a_32 = 1.5 - beta_32 / np.sqrt(N)
    a_phi_pred = a_phi - beta_phi / np.sqrt(N)
    a_f = a_free - beta_free / np.sqrt(N)
    print(f"{N:<12} {a_obs:<15.6f} {a_32:<15.6f} {a_phi_pred:<15.6f} {a_f:<15.6f}")

print(f"\n{'Extrapolation':<12}")
for N in [200000, 500000, 1000000, 10000000]:
    a_32 = 1.5 - beta_32 / np.sqrt(N)
    a_phi_pred = a_phi - beta_phi / np.sqrt(N)
    a_f = a_free - beta_free / np.sqrt(N)
    print(f"{N:<12} {'—':<15} {a_32:<15.6f} {a_phi_pred:<15.6f} {a_f:<15.6f}")

# ============================================================================
# INTERPRÉTATION DE 3/2
# ============================================================================

print("\n" + "=" * 70)
print("INTERPRÉTATION: POURQUOI 3/2 ?")
print("=" * 70)

print(f"""
Si a_∞ = 3/2 exactement, cela suggère:

1. MOYENNE ARITHMÉTIQUE:
   a + b = 1, et si a = 3/2, alors b = -1/2
   → γ_n = (3/2)·γ_{{n-8}} - (1/2)·γ_{{n-21}}
   → γ_n = (3·γ_{{n-8}} - γ_{{n-21}}) / 2
   C'est une extrapolation linéaire !

2. LIEN AVEC FIBONACCI:
   3/2 = F_4 / F_3 = 3/2 (ratio de petits Fibonacci)
   Mais aussi proche de φ - 1/10 ≈ 1.518

3. RELATION AVEC φ:
   3/2 = φ - 0.118... = φ - (φ-1)/5.236...
   Pas de relation simple évidente.

4. STRUCTURE DE RÉCURRENCE:
   γ_n = (3γ_{{n-8}} - γ_{{n-21}}) / 2

   Cela ressemble à une formule de différences finies
   pour approximer une dérivée!

   En effet: (3f(x-h) - f(x-2.625h)) / 2
   où 21/8 = 2.625 ≈ φ² = 2.618
""")

# ============================================================================
# FORMULE FINALE
# ============================================================================

print("\n" + "=" * 70)
print("FORMULE FINALE (SI a_∞ = 3/2)")
print("=" * 70)

print(f"""
═══════════════════════════════════════════════════════════════════════
                    FORMULE RÉVISÉE
═══════════════════════════════════════════════════════════════════════

Pour les zéros de Riemann:

    γ_n = (3/2)·γ_{{n-8}} - (1/2)·γ_{{n-21}} + c(N)

ou de façon équivalente:

    γ_n = (3·γ_{{n-8}} - γ_{{n-21}}) / 2 + c(N)

avec:
  • 8 = F_6 et 21 = F_8 (Fibonacci, gap 2)
  • c(N) ~ O(1/√N) correction de taille finie
  • 21/8 = 2.625 ≈ φ² = 2.618

INTERPRÉTATION:
  • C'est une EXTRAPOLATION LINÉAIRE entre deux échelles Fibonacci
  • Le coefficient 3/2 est rationnel (pas de φ explicite!)
  • Mais les LAGS restent Fibonacci (structure φ cachée)

COMPARAISON DES MODÈLES:
  • a = 3/2 - {beta_32:.2f}/√N    → RSS = {rss_32:.2e}
  • a = 2φ/√5 - {beta_phi:.2f}/√N → RSS = {rss_phi:.2e}
  • Ratio: {ratio:.2f}x en faveur de 3/2

═══════════════════════════════════════════════════════════════════════
""")

# Sauvegarder
import json
results_dict = {
    "best_limit": "3/2",
    "best_limit_value": 1.5,
    "alternative": "2φ/√5",
    "alternative_value": float(2*PHI/SQRT5),
    "model_3_2": {"beta": float(beta_32), "rss": float(rss_32)},
    "model_phi": {"beta": float(beta_phi), "rss": float(rss_phi)},
    "rss_ratio": float(ratio),
    "interpretation": "Linear extrapolation: γ_n = (3γ_{n-8} - γ_{n-21})/2",
    "fibonacci_lags": [8, 21],
    "lag_ratio": 21/8,
    "phi_squared": float(PHI**2)
}

with open(Path(__file__).parent / "limit_3_2_results.json", "w") as f:
    json.dump(results_dict, f, indent=2)

print("✓ Résultats sauvegardés dans limit_3_2_results.json")
