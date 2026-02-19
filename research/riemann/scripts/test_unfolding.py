#!/usr/bin/env python3
"""
GIFT Phase 2.5: Unfolding Test
==============================

Test décisif proposé par le conseil des IA:
Si le drift des coefficients disparaît après unfolding → artefact de densité
Si le drift persiste → structure profonde réelle

Unfolding: u_n = N(γ_n) où N(T) est la fonction de comptage des zéros
"""

import numpy as np
from typing import List, Tuple, Dict
import json

# ============================================================
# UNFOLDING FUNCTIONS
# ============================================================

def N_riemann(T: float) -> float:
    """
    Smooth zero counting function (Riemann-von Mangoldt).
    N(T) ≈ (T/2π) log(T/2π) - T/2π + 7/8 + S(T)

    On ignore S(T) (terme oscillant) pour l'unfolding lisse.
    """
    if T < 14:
        return 0.0
    return (T / (2 * np.pi)) * np.log(T / (2 * np.pi)) - T / (2 * np.pi) + 7/8

def unfold_zeros(gammas: np.ndarray) -> np.ndarray:
    """
    Transforme les zéros bruts γ_n en zéros dépliés u_n = N(γ_n).
    Les u_n devraient avoir une densité uniforme (espacement moyen = 1).
    """
    return np.array([N_riemann(g) for g in gammas])

def normalized_spacings(u: np.ndarray) -> np.ndarray:
    """
    Espacements normalisés s_n = u_{n+1} - u_n.
    En GUE, ces espacements suivent la distribution de Wigner.
    """
    return np.diff(u)

# ============================================================
# RECURRENCE FITTING
# ============================================================

def fit_recurrence(data: np.ndarray, lags: List[int],
                   start: int = None, end: int = None) -> Tuple[np.ndarray, float]:
    """
    Fit récurrence linéaire sur données (γ ou u).
    Retourne: (coefficients, erreur_moyenne)
    """
    max_lag = max(lags)
    if start is None:
        start = max_lag
    if end is None:
        end = len(data)

    n_points = end - start
    n_params = len(lags) + 1

    X = np.zeros((n_points, n_params))
    for i, lag in enumerate(lags):
        X[:, i] = data[start - lag:end - lag]
    X[:, -1] = 1.0

    y = data[start:end]

    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    y_pred = X @ coeffs
    errors = np.abs(y_pred - y)

    return coeffs, np.mean(errors)

# ============================================================
# WINDOW ANALYSIS
# ============================================================

def analyze_windows(data: np.ndarray, lags: List[int],
                    window_size: int = 50000, step: int = 50000,
                    data_name: str = "data") -> List[Dict]:
    """
    Analyse par fenêtre glissante.
    """
    results = []

    for start in range(0, len(data) - window_size, step):
        window = data[start:start + window_size]
        stable_start = int(window_size * 0.7)

        coeffs, error = fit_recurrence(window, lags, stable_start, window_size)

        # Valeur centrale de la fenêtre (pour référence)
        mid_idx = start + window_size // 2

        results.append({
            'window_start': start,
            'window_mid_idx': mid_idx,
            'coefficients': {
                'a_5': float(coeffs[0]),
                'a_8': float(coeffs[1]),
                'a_13': float(coeffs[2]),
                'a_27': float(coeffs[3]),
                'c': float(coeffs[4])
            },
            'error': float(error)
        })

    return results

# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():
    print("="*70)
    print("GIFT Phase 2.5: UNFOLDING TEST")
    print("="*70)

    # Charger les données
    print("\n📂 Chargement des données...")
    try:
        gammas = []
        with open('zeta/zeros6', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        gammas.append(float(line.split()[0]))
                    except:
                        continue
        gammas = np.array(sorted(gammas))
        print(f"   {len(gammas):,} zéros chargés")
    except FileNotFoundError:
        print("   ❌ Fichier zeta/zeros6 non trouvé")
        print("   Essai avec zeros1...")
        gammas = []
        with open('zeta/zeros1', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        gammas.append(float(line.split()[0]))
                    except:
                        continue
        gammas = np.array(sorted(gammas))
        print(f"   {len(gammas):,} zéros chargés")

    # Unfolding
    print("\n📐 Unfolding des zéros...")
    u = unfold_zeros(gammas)
    print(f"   γ_1 = {gammas[0]:.2f} → u_1 = {u[0]:.2f}")
    print(f"   γ_N = {gammas[-1]:.2f} → u_N = {u[-1]:.2f}")

    # Vérification: espacement moyen devrait être ~1
    spacings = np.diff(u)
    print(f"   Espacement moyen (unfolded): {np.mean(spacings):.4f} (attendu: ~1)")

    # Deviation x_n = u_n - n
    x = u - np.arange(len(u))
    print(f"   Déviation moyenne x_n = u_n - n: {np.mean(x):.4f}")

    # Lags GIFT
    lags = [5, 8, 13, 27]

    # Analyse sur γ (raw)
    print("\n" + "="*70)
    print("ANALYSE 1: Zéros BRUTS (γ)")
    print("="*70)
    results_raw = analyze_windows(gammas, lags, window_size=50000, step=50000, data_name="gamma")

    print(f"\n{'Window':<10} {'a_5':>10} {'a_8':>10} {'a_13':>10} {'a_27':>10} {'c':>10} {'error':>10}")
    print("-"*75)
    for i, r in enumerate(results_raw[:5]):
        c = r['coefficients']
        print(f"{i+1:<10} {c['a_5']:>10.4f} {c['a_8']:>10.4f} {c['a_13']:>10.4f} {c['a_27']:>10.4f} {c['c']:>10.2f} {r['error']:>10.4f}")
    print("...")
    for r in results_raw[-2:]:
        c = r['coefficients']
        print(f"{r['window_start']//50000+1:<10} {c['a_5']:>10.4f} {c['a_8']:>10.4f} {c['a_13']:>10.4f} {c['a_27']:>10.4f} {c['c']:>10.2f} {r['error']:>10.4f}")

    # Analyse sur u (unfolded)
    print("\n" + "="*70)
    print("ANALYSE 2: Zéros DÉPLIÉS (u = N(γ))")
    print("="*70)
    results_unf = analyze_windows(u, lags, window_size=50000, step=50000, data_name="u")

    print(f"\n{'Window':<10} {'a_5':>10} {'a_8':>10} {'a_13':>10} {'a_27':>10} {'c':>10} {'error':>10}")
    print("-"*75)
    for i, r in enumerate(results_unf[:5]):
        c = r['coefficients']
        print(f"{i+1:<10} {c['a_5']:>10.4f} {c['a_8']:>10.4f} {c['a_13']:>10.4f} {c['a_27']:>10.4f} {c['c']:>10.2f} {r['error']:>10.4f}")
    print("...")
    for r in results_unf[-2:]:
        c = r['coefficients']
        print(f"{r['window_start']//50000+1:<10} {c['a_5']:>10.4f} {c['a_8']:>10.4f} {c['a_13']:>10.4f} {c['a_27']:>10.4f} {c['c']:>10.2f} {r['error']:>10.4f}")

    # Analyse sur x = u - n (deviation)
    print("\n" + "="*70)
    print("ANALYSE 3: DÉVIATION (x = u - n)")
    print("="*70)
    results_dev = analyze_windows(x, lags, window_size=50000, step=50000, data_name="x")

    print(f"\n{'Window':<10} {'a_5':>10} {'a_8':>10} {'a_13':>10} {'a_27':>10} {'c':>10} {'error':>10}")
    print("-"*75)
    for i, r in enumerate(results_dev[:5]):
        c = r['coefficients']
        print(f"{i+1:<10} {c['a_5']:>10.4f} {c['a_8']:>10.4f} {c['a_13']:>10.4f} {c['a_27']:>10.4f} {c['c']:>10.4f} {r['error']:>10.4f}")
    print("...")
    for r in results_dev[-2:]:
        c = r['coefficients']
        print(f"{r['window_start']//50000+1:<10} {c['a_5']:>10.4f} {c['a_8']:>10.4f} {c['a_13']:>10.4f} {c['a_27']:>10.4f} {c['c']:>10.4f} {r['error']:>10.4f}")

    # Comparaison du drift
    print("\n" + "="*70)
    print("VERDICT: Le drift persiste-t-il après unfolding?")
    print("="*70)

    def compute_drift(results):
        """Calcule la variation relative des coefficients."""
        if len(results) < 2:
            return {}
        first = results[0]['coefficients']
        last = results[-1]['coefficients']
        drift = {}
        for key in first:
            if abs(first[key]) > 0.001:
                drift[key] = (last[key] - first[key]) / abs(first[key]) * 100
            else:
                drift[key] = float('inf') if last[key] != first[key] else 0
        return drift

    drift_raw = compute_drift(results_raw)
    drift_unf = compute_drift(results_unf)
    drift_dev = compute_drift(results_dev)

    print(f"\n{'Coeff':<8} {'Drift γ (%)':>15} {'Drift u (%)':>15} {'Drift x (%)':>15}")
    print("-"*55)
    for key in ['a_5', 'a_8', 'a_13', 'a_27', 'c']:
        d_raw = drift_raw.get(key, 0)
        d_unf = drift_unf.get(key, 0)
        d_dev = drift_dev.get(key, 0)
        print(f"{key:<8} {d_raw:>15.1f} {d_unf:>15.1f} {d_dev:>15.1f}")

    # Verdict
    print("\n" + "="*70)
    avg_drift_raw = np.mean([abs(v) for v in drift_raw.values() if abs(v) < 1000])
    avg_drift_unf = np.mean([abs(v) for v in drift_unf.values() if abs(v) < 1000])
    avg_drift_dev = np.mean([abs(v) for v in drift_dev.values() if abs(v) < 1000])

    print(f"Drift moyen absolu:")
    print(f"   γ (raw):     {avg_drift_raw:.1f}%")
    print(f"   u (unfolded): {avg_drift_unf:.1f}%")
    print(f"   x (deviation): {avg_drift_dev:.1f}%")

    if avg_drift_unf < avg_drift_raw * 0.3:
        print("\n✅ Le drift DISPARAÎT après unfolding!")
        print("   → Le drift observé était un ARTEFACT de la densité variable")
        print("   → GIFT décrit les SPACINGS, pas les hauteurs brutes")
        verdict = "ARTIFACT"
    elif avg_drift_unf < avg_drift_raw * 0.7:
        print("\n⚠️ Le drift est RÉDUIT mais persiste partiellement")
        print("   → Structure mixte: partie artefact + partie réelle")
        verdict = "PARTIAL"
    else:
        print("\n🔥 Le drift PERSISTE après unfolding!")
        print("   → Le drift est une STRUCTURE RÉELLE")
        print("   → Possible RG flow / transition de phase")
        verdict = "REAL_STRUCTURE"

    # Export results
    output = {
        'verdict': verdict,
        'drift_comparison': {
            'raw_gamma': drift_raw,
            'unfolded_u': drift_unf,
            'deviation_x': drift_dev
        },
        'avg_drift': {
            'raw': float(avg_drift_raw),
            'unfolded': float(avg_drift_unf),
            'deviation': float(avg_drift_dev)
        },
        'window_results': {
            'raw': results_raw,
            'unfolded': results_unf,
            'deviation': results_dev
        }
    }

    with open('phase25_unfolding_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n💾 Résultats sauvegardés dans phase25_unfolding_results.json")

    return output

if __name__ == "__main__":
    main()
