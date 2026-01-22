"""
Matrice de Robustesse pour K₇ (H*=99)

OBJECTIF:
Tester toutes les combinaisons (N, k, laplacian_type) pour:
1. Identifier si λ₁×H* = 13 est un plateau stable
2. Déterminer les paramètres optimaux
3. Vérifier l'indépendance aux choix méthodologiques

MÉTHODE:
- Construction TCS: K₇ ≈ S¹ × S³ × S³
- Distance TCS avec ratio = H*/84 (formule standard)
- Grille: N ∈ {1000,2000,5000,10000}, k ∈ {15,25,40,60}

Reference: ROADMAP_SPECTRAL_VALIDATION.md Phase 2
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import itertools
import sys
from datetime import datetime

# Ajouter le parent pour importer config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    TOPOLOGY, ROBUSTNESS_GRID, ROBUSTNESS_GRID_QUICK, TOLERANCE, SEEDS,
    get_output_path
)


# =============================================================================
# CONSTANTES K₇
# =============================================================================

DET_G = 65/32  # Déterminant de la métrique G₂
H_STAR = TOPOLOGY["H_star"]  # 99
B2 = TOPOLOGY["b2"]  # 21
B3 = TOPOLOGY["b3"]  # 77


# =============================================================================
# TCS CONSTRUCTION
# =============================================================================

def sample_S3(n: int, seed: int) -> np.ndarray:
    """Échantillonnage uniforme sur S³."""
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def geodesic_S3(Q: np.ndarray) -> np.ndarray:
    """Matrice de distances géodésiques sur S³."""
    dot = np.clip(np.abs(Q @ Q.T), 0, 1)
    return 2 * np.arccos(dot)


def tcs_distance_matrix(n: int, ratio: float, seed: int) -> np.ndarray:
    """
    Calcule la matrice de distances TCS pour K₇.

    Construction TCS: K₇ ≈ S¹ × S³ × S³
    Distance: d² = α * d_S1² + d_S3_1² + ratio² * d_S3_2²

    avec α = det(G) / ratio³

    Args:
        n: Nombre de points
        ratio: Paramètre d'anisotropie (typiquement H*/84)
        seed: Graine aléatoire

    Returns:
        Matrice (n, n) de distances
    """
    rng = np.random.default_rng(seed)

    # S¹ component
    theta = rng.uniform(0, 2*np.pi, n)
    theta_diff = np.abs(theta[:, None] - theta[None, :])
    d_S1_sq = np.minimum(theta_diff, 2*np.pi - theta_diff)**2

    # Deux S³ components
    q1 = sample_S3(n, seed + 1000)
    q2 = sample_S3(n, seed + 2000)
    d1 = geodesic_S3(q1)
    d2 = geodesic_S3(q2)

    # Combinaison TCS
    alpha = DET_G / (ratio**3)
    d_sq = alpha * d_S1_sq + d1**2 + (ratio**2) * d2**2

    return np.sqrt(np.maximum(d_sq, 0))


def ratio_formula(H_star: int, formula: str = "84") -> float:
    """
    Calcule le ratio pour la construction TCS.

    Formules validées dans V11:
    - "84": ratio = H*/84 = H*/(6×14) - formule standard
    - "78": ratio = H*/78 = H*/(6×13) - formule alternative

    Note: Paradoxalement, "84" donne de meilleurs résultats pour target=13
    """
    if formula == "84":
        return max(H_star / 84, 0.8)
    elif formula == "78":
        return max(H_star / 78, 0.8)
    else:
        raise ValueError(f"Unknown formula: {formula}")


# =============================================================================
# GRAPH LAPLACIAN
# =============================================================================

def compute_lambda1_from_distance(D: np.ndarray, k: int = 25,
                                   laplacian_type: str = "symmetric") -> float:
    """
    Calcule λ₁ à partir d'une matrice de distances.

    Reproduit exactement le pipeline V11.
    """
    n = D.shape[0]
    k = min(k, n - 1)

    # Sigma = médiane des k plus proches distances
    knn_dists = np.partition(D, k, axis=1)[:, :k]
    sigma = max(np.median(knn_dists), 1e-10)

    # Matrice de poids gaussiens
    W = np.exp(-D**2 / (2 * sigma**2))
    np.fill_diagonal(W, 0)

    # Garder seulement les k plus proches voisins
    for i in range(n):
        idx = np.argpartition(W[i], -k)[-k:]
        mask = np.ones(n, dtype=bool)
        mask[idx] = False
        W[i, mask] = 0

    # Symétriser
    W = (W + W.T) / 2

    # Degrés
    d = W.sum(axis=1)
    d = np.maximum(d, 1e-10)

    # Construire Laplacien
    if laplacian_type == "unnormalized":
        L = np.diag(d) - W
    elif laplacian_type == "random_walk":
        d_inv = 1.0 / d
        L = np.eye(n) - np.diag(d_inv) @ W
    elif laplacian_type == "symmetric":
        d_inv_sqrt = 1.0 / np.sqrt(d)
        L = np.eye(n) - np.diag(d_inv_sqrt) @ W @ np.diag(d_inv_sqrt)
    else:
        raise ValueError(f"Unknown laplacian_type: {laplacian_type}")

    # Sparse pour eigsh
    L_sparse = sp.csr_matrix(L)

    # Calculer les 5 plus petites valeurs propres
    eigs, _ = eigsh(L_sparse, k=5, which='SM', tol=1e-8)
    eigs = np.sort(np.real(eigs))

    # λ₁ = première valeur propre non-nulle
    for ev in eigs:
        if ev > 1e-8:
            return ev

    return eigs[1] if len(eigs) > 1 else 0.0


# =============================================================================
# ROBUSTNESS MATRIX
# =============================================================================

@dataclass
class RobustnessResult:
    """Résultat d'un test de robustesse."""
    N: int
    k: int
    laplacian_type: str
    seed: int
    ratio: float
    lambda1: float
    lambda1_x_Hstar: float
    deviation_from_13_pct: float
    deviation_from_14_pct: float
    closer_to: str


def run_single_test(N: int, k: int, laplacian_type: str, seed: int,
                    H_star: int = 99, ratio_formula_name: str = "84") -> RobustnessResult:
    """Exécute un test de robustesse."""
    ratio = ratio_formula(H_star, ratio_formula_name)

    # Calculer matrice de distances TCS
    D = tcs_distance_matrix(N, ratio, seed)

    # Calculer λ₁
    lambda1 = compute_lambda1_from_distance(D, k, laplacian_type)

    # Produit
    product = lambda1 * H_star

    # Déviations
    dev_13 = abs(product - 13) / 13 * 100
    dev_14 = abs(product - 14) / 14 * 100

    return RobustnessResult(
        N=N, k=k, laplacian_type=laplacian_type, seed=seed,
        ratio=ratio, lambda1=lambda1, lambda1_x_Hstar=product,
        deviation_from_13_pct=dev_13, deviation_from_14_pct=dev_14,
        closer_to="13" if dev_13 < dev_14 else "14"
    )


def run_robustness_matrix(grid: Dict = None, seeds: List[int] = None,
                           verbose: bool = True) -> Dict:
    """
    Exécute la grille complète de robustesse.

    Args:
        grid: Grille de paramètres (N, k, laplacian)
        seeds: Liste de graines pour moyennage
        verbose: Afficher progression

    Returns:
        Dict avec résultats et analyse
    """
    if grid is None:
        grid = ROBUSTNESS_GRID

    if seeds is None:
        seeds = [SEEDS["robustness"]]

    results = []
    total = (len(grid["N"]) * len(grid["k"]) *
             len(grid["laplacian"]) * len(seeds))
    current = 0

    for N in grid["N"]:
        for k in grid["k"]:
            for lap_type in grid["laplacian"]:
                for seed in seeds:
                    current += 1
                    if verbose:
                        print(f"[{current}/{total}] N={N}, k={k}, {lap_type}...", end=" ")

                    try:
                        result = run_single_test(N, k, lap_type, seed)
                        results.append(asdict(result))
                        if verbose:
                            print(f"λ₁×H*={result.lambda1_x_Hstar:.2f} ({result.closer_to})")
                    except Exception as e:
                        if verbose:
                            print(f"ERROR: {e}")
                        results.append({
                            "N": N, "k": k, "laplacian_type": lap_type,
                            "seed": seed, "error": str(e)
                        })

    # Analyse
    analysis = analyze_results(results, grid)

    return {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "H_star": H_STAR,
            "b2": B2,
            "b3": B3,
            "grid": grid,
            "seeds": seeds,
        },
        "results": results,
        "analysis": analysis,
    }


def analyze_results(results: List[Dict], grid: Dict) -> Dict:
    """Analyse la matrice de résultats."""
    # Filtrer erreurs
    valid = [r for r in results if "error" not in r]

    if not valid:
        return {"error": "No valid results"}

    # Statistiques globales
    products = [r["lambda1_x_Hstar"] for r in valid]
    global_stats = {
        "count": len(valid),
        "mean": float(np.mean(products)),
        "std": float(np.std(products)),
        "min": float(np.min(products)),
        "max": float(np.max(products)),
        "median": float(np.median(products)),
    }

    # Par type de Laplacien
    by_type = {}
    for lap_type in grid["laplacian"]:
        subset = [r for r in valid if r["laplacian_type"] == lap_type]
        if subset:
            prods = [r["lambda1_x_Hstar"] for r in subset]
            by_type[lap_type] = {
                "count": len(subset),
                "mean": float(np.mean(prods)),
                "std": float(np.std(prods)),
                "closer_to_13": sum(1 for r in subset if r["closer_to"] == "13"),
                "closer_to_14": sum(1 for r in subset if r["closer_to"] == "14"),
            }

    # Analyse du plateau (N >= 5000)
    high_N = [r for r in valid if r["N"] >= TOLERANCE["plateau_N_min"]]
    if high_N:
        prods_high = [r["lambda1_x_Hstar"] for r in high_N]
        plateau_mean = float(np.mean(prods_high))
        plateau_std = float(np.std(prods_high))
        plateau_variation = plateau_std / plateau_mean * 100 if plateau_mean > 0 else float('inf')

        plateau_analysis = {
            "N_min": TOLERANCE["plateau_N_min"],
            "count": len(high_N),
            "mean": plateau_mean,
            "std": plateau_std,
            "variation_pct": plateau_variation,
            "is_plateau": plateau_variation < TOLERANCE["convergence_plateau"] * 100,
            "plateau_value": 13 if abs(plateau_mean - 13) < abs(plateau_mean - 14) else 14,
            "deviation_from_13_pct": abs(plateau_mean - 13) / 13 * 100,
            "deviation_from_14_pct": abs(plateau_mean - 14) / 14 * 100,
        }
    else:
        plateau_analysis = {"error": "No high-N results"}

    # Convergence: λ₁×H* vs N (k=25, symmetric)
    convergence = []
    for N in sorted(grid["N"]):
        subset = [r for r in valid
                  if r["N"] == N and r["k"] == 25 and r["laplacian_type"] == "symmetric"]
        if subset:
            convergence.append({
                "N": N,
                "mean": float(np.mean([r["lambda1_x_Hstar"] for r in subset])),
                "std": float(np.std([r["lambda1_x_Hstar"] for r in subset])) if len(subset) > 1 else 0,
            })

    # Verdict
    if plateau_analysis.get("is_plateau"):
        if plateau_analysis["plateau_value"] == 13:
            verdict = "PLATEAU_13"
        else:
            verdict = "PLATEAU_14"
    elif plateau_analysis.get("variation_pct", 100) < 10:
        verdict = "QUASI_PLATEAU"
    else:
        verdict = "NO_PLATEAU"

    return {
        "global": global_stats,
        "by_laplacian_type": by_type,
        "plateau_analysis": plateau_analysis,
        "convergence": convergence,
        "verdict": verdict,
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  K₇ ROBUSTNESS MATRIX - Spectral Validation Phase 2")
    print(f"  H* = {H_STAR}, Target: λ₁×H* = 13")
    print("=" * 70)
    print()

    # Utiliser grille rapide pour test, grille complète pour production
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Run full grid")
    parser.add_argument("--seeds", type=int, default=1, help="Number of seeds")
    args = parser.parse_args()

    if args.full:
        grid = ROBUSTNESS_GRID
        seeds = [42, 123, 456][:args.seeds]
        print("Running FULL grid...")
    else:
        grid = ROBUSTNESS_GRID_QUICK
        seeds = [42]
        print("Running QUICK grid (use --full for complete analysis)...")

    print(f"Grid: N={grid['N']}, k={grid['k']}, laplacian={grid['laplacian']}")
    print(f"Seeds: {seeds}")
    print()

    # Run
    output = run_robustness_matrix(grid, seeds)

    # Save
    output_path = get_output_path("robustness", "K7_robustness_matrix.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Print summary
    print()
    print("=" * 70)
    print("RÉSULTATS K₇ ROBUSTESSE")
    print("=" * 70)

    analysis = output["analysis"]

    print(f"\nStatistiques globales:")
    print(f"  λ₁×H* mean: {analysis['global']['mean']:.4f} ± {analysis['global']['std']:.4f}")
    print(f"  Range: [{analysis['global']['min']:.2f}, {analysis['global']['max']:.2f}]")

    print(f"\nPar type de Laplacien:")
    for lap_type, stats in analysis["by_laplacian_type"].items():
        print(f"  {lap_type}:")
        print(f"    Mean: {stats['mean']:.4f}")
        print(f"    Closer to 13: {stats['closer_to_13']}, to 14: {stats['closer_to_14']}")

    print(f"\nAnalyse du plateau (N ≥ {TOLERANCE['plateau_N_min']}):")
    pa = analysis["plateau_analysis"]
    if "error" not in pa:
        print(f"  Mean: {pa['mean']:.4f} ± {pa['std']:.4f}")
        print(f"  Variation: {pa['variation_pct']:.2f}%")
        print(f"  Is plateau: {pa['is_plateau']}")
        print(f"  Closer to: {pa['plateau_value']}")
        print(f"  Dev from 13: {pa['deviation_from_13_pct']:.2f}%")
        print(f"  Dev from 14: {pa['deviation_from_14_pct']:.2f}%")

    print(f"\nConvergence (k=25, symmetric):")
    for c in analysis["convergence"]:
        print(f"  N={c['N']}: λ₁×H*={c['mean']:.4f}")

    print(f"\nVERDICT: {analysis['verdict']}")
    print(f"\nSaved: {output_path}")
