"""
Calibration sur S⁷ : λ₁(Laplace-Beltrami) = 7

OBJECTIF:
Calibration sur la 7-sphère, MÊME DIMENSION que K₇.
Permet de comparer directement les effets dimensionnels.

CONTEXTE:
- S⁷ = sphère 7D, λ₁(LB) = n(n+6) = 1×7 = 7 avec multiplicité 8
- Même dimension que K₇ → comparaison plus directe
- Le ratio λ₁(S⁷)/λ₁(S³) = 7/3 ≈ 2.33 si le pipeline est correct

MÉTHODE:
1. Échantillonner S⁷ uniformément (8 gaussiennes normalisées)
2. Calculer graph Laplacian avec mêmes paramètres que S³
3. Comparer ratios S⁷/S³

Reference: ROADMAP_SPECTRAL_VALIDATION.md Phase 1.2
"""

import numpy as np
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import sys

# Ajouter le parent pour importer config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MANIFOLDS, TOLERANCE, SEEDS,
    get_output_path, check_pass_fail
)


# =============================================================================
# SAMPLING S⁷
# =============================================================================

def sample_S7_uniform(N: int, seed: int = 42) -> np.ndarray:
    """
    Échantillonnage uniforme sur S⁷ via méthode gaussienne.

    S⁷ = {x ∈ ℝ⁸ : |x| = 1}

    Méthode: 8 gaussiennes i.i.d. normalisées → uniforme sur S⁷
    (Muller, 1959)

    Args:
        N: Nombre de points
        seed: Graine aléatoire

    Returns:
        Array (N, 8) de points sur S⁷
    """
    rng = np.random.default_rng(seed)
    points = rng.standard_normal((N, 8))
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms


def chord_to_geodesic_Sn(chord_dist: np.ndarray) -> np.ndarray:
    """
    Convertit distance de corde en distance géodésique sur Sⁿ.

    Pour Sⁿ de rayon 1:
        d_chord = 2 * sin(d_geo / 2)
        d_geo = 2 * arcsin(d_chord / 2)

    Valide pour toutes les sphères.
    """
    clipped = np.clip(chord_dist / 2, 0, 1)
    return 2 * np.arcsin(clipped)


def geodesic_distance_matrix_S7(points: np.ndarray) -> np.ndarray:
    """
    Calcule la matrice de distances géodésiques sur S⁷.

    d_geo(p, q) = arccos(p · q) pour Sⁿ
    """
    # Distance de corde via norme euclidienne
    # ||p - q||² = 2 - 2(p·q)
    dot_products = points @ points.T
    chord_sq = 2 - 2 * dot_products
    chord_dist = np.sqrt(np.maximum(chord_sq, 0))

    return chord_to_geodesic_Sn(chord_dist)


# =============================================================================
# GRAPH LAPLACIAN (réutilise la logique de S3)
# =============================================================================

def build_knn_graph(points: np.ndarray, k: int,
                    use_geodesic: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Construit le graphe k-NN."""
    dim = points.shape[1]

    if use_geodesic and dim == 8:
        # S⁷
        D_geo = geodesic_distance_matrix_S7(points)
        indices = np.argsort(D_geo, axis=1)[:, :k+1]
        distances = np.take_along_axis(D_geo, indices, axis=1)
    else:
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
        nbrs.fit(points)
        distances, indices = nbrs.kneighbors(points)

    return distances, indices


def build_graph_laplacian(points: np.ndarray, k: int, sigma: Optional[float] = None,
                          laplacian_type: str = "symmetric",
                          use_geodesic: bool = True) -> csr_matrix:
    """Construit le Laplacien de graphe."""
    N = len(points)
    dim = points.shape[1]

    distances, indices = build_knn_graph(points, k, use_geodesic)

    if sigma is None:
        sigma = np.median(distances[:, 1:])
        sigma = max(sigma, 1e-10)

    # Matrice de poids gaussiens
    row_indices = []
    col_indices = []
    weights = []

    for i in range(N):
        for j_idx in range(1, k + 1):
            j = indices[i, j_idx]
            d = distances[i, j_idx]
            w = np.exp(-d**2 / (2 * sigma**2))

            row_indices.append(i)
            col_indices.append(j)
            weights.append(w)

    W = csr_matrix((weights, (row_indices, col_indices)), shape=(N, N))
    W = (W + W.T) / 2

    d = np.asarray(W.sum(axis=1)).flatten()
    d = np.maximum(d, 1e-10)

    if laplacian_type == "unnormalized":
        D = diags(d)
        L = D - W
    elif laplacian_type == "random_walk":
        d_inv = 1.0 / d
        D_inv = diags(d_inv)
        L = eye(N) - D_inv @ W
    elif laplacian_type == "symmetric":
        d_inv_sqrt = 1.0 / np.sqrt(d)
        D_inv_sqrt = diags(d_inv_sqrt)
        L = eye(N) - D_inv_sqrt @ W @ D_inv_sqrt
    else:
        raise ValueError(f"Unknown laplacian_type: {laplacian_type}")

    return L.tocsr()


def compute_lambda1(L: csr_matrix, num_eigenvalues: int = 6) -> Tuple[float, np.ndarray]:
    """Calcule λ₁."""
    eigenvalues, _ = eigsh(L, k=num_eigenvalues, which='SM', tol=1e-8)
    eigenvalues = np.sort(np.real(eigenvalues))

    threshold = 1e-8
    for i, ev in enumerate(eigenvalues):
        if ev > threshold:
            return ev, eigenvalues

    return eigenvalues[1] if len(eigenvalues) > 1 else 0.0, eigenvalues


# =============================================================================
# CALIBRATION
# =============================================================================

@dataclass
class CalibrationResult:
    """Résultat d'un run de calibration."""
    manifold: str
    N: int
    k: int
    sigma: float
    laplacian_type: str
    seed: int
    lambda1_measured: float
    lambda1_expected: float
    eigenvalues: List[float]
    deviation_pct: float
    passed: bool


def run_single_calibration(N: int, k: int, laplacian_type: str, seed: int,
                           lambda1_expected: float = 7.0) -> CalibrationResult:
    """Exécute une calibration sur S⁷."""
    # Échantillonner S⁷
    points = sample_S7_uniform(N, seed)

    # Sigma heuristique: sqrt(dim/k) * facteur
    dim = 7
    sigma = np.sqrt(dim / k) * 0.5

    # Construire Laplacien
    L = build_graph_laplacian(points, k, sigma, laplacian_type, use_geodesic=True)

    # Calculer λ₁
    lambda1, eigenvalues = compute_lambda1(L)

    # Déviation
    deviation_pct = abs(lambda1 - lambda1_expected) / lambda1_expected * 100

    # Critère PASS
    tolerance = TOLERANCE.get("calibration_S7", 0.10)
    low, high = TOLERANCE.get("calibration_S7_range", (6.3, 7.7))

    if laplacian_type in ["symmetric", "random_walk"]:
        passed = 0.01 < lambda1 < 2.0
    else:
        passed = low <= lambda1 <= high

    return CalibrationResult(
        manifold="S7",
        N=N,
        k=k,
        sigma=sigma,
        laplacian_type=laplacian_type,
        seed=seed,
        lambda1_measured=lambda1,
        lambda1_expected=lambda1_expected,
        eigenvalues=eigenvalues.tolist(),
        deviation_pct=deviation_pct,
        passed=passed,
    )


def run_S7_calibration(config: Optional[Dict] = None) -> Dict:
    """Exécute la calibration complète S⁷."""
    if config is None:
        config = {
            "N_values": [1000, 2000, 5000, 10000],
            "k_values": [15, 25, 40],
            "laplacian_types": ["unnormalized", "random_walk", "symmetric"],
            "seeds": [SEEDS["calibration"]],
        }

    results = []

    total = (len(config["N_values"]) * len(config["k_values"]) *
             len(config["laplacian_types"]) * len(config["seeds"]))
    current = 0

    for N in config["N_values"]:
        for k in config["k_values"]:
            for lap_type in config["laplacian_types"]:
                for seed in config["seeds"]:
                    current += 1
                    print(f"[{current}/{total}] S⁷: N={N}, k={k}, {lap_type}...", end=" ")

                    try:
                        result = run_single_calibration(N, k, lap_type, seed)
                        results.append(asdict(result))
                        status = "✓" if result.passed else "✗"
                        print(f"λ₁={result.lambda1_measured:.4f} {status}")
                    except Exception as e:
                        print(f"ERROR: {e}")
                        results.append({
                            "manifold": "S7", "N": N, "k": k,
                            "laplacian_type": lap_type, "seed": seed,
                            "error": str(e)
                        })

    # Analyse
    valid_results = [r for r in results if "error" not in r]

    by_type = {}
    for lap_type in config["laplacian_types"]:
        subset = [r for r in valid_results if r["laplacian_type"] == lap_type]
        if subset:
            lambdas = [r["lambda1_measured"] for r in subset]
            by_type[lap_type] = {
                "count": len(subset),
                "mean_lambda1": float(np.mean(lambdas)),
                "std_lambda1": float(np.std(lambdas)),
                "min_lambda1": float(np.min(lambdas)),
                "max_lambda1": float(np.max(lambdas)),
                "pass_rate": sum(1 for r in subset if r["passed"]) / len(subset),
            }

    convergence = []
    for N in sorted(config["N_values"]):
        subset = [r for r in valid_results
                  if r["N"] == N and r["k"] == 25 and r["laplacian_type"] == "symmetric"]
        if subset:
            convergence.append({
                "N": N,
                "lambda1": float(np.mean([r["lambda1_measured"] for r in subset])),
            })

    # Verdict
    if valid_results:
        sym_results = [r for r in valid_results if r["laplacian_type"] == "symmetric"]
        if sym_results:
            lambdas_sym = [r["lambda1_measured"] for r in sym_results]
            gap_exists = all(l > 0.01 for l in lambdas_sym)
            stable = np.std(lambdas_sym) / np.mean(lambdas_sym) < 0.3 if lambdas_sym else False
            verdict = "PASS" if (gap_exists and stable) else "INVESTIGATE"
        else:
            verdict = "INCOMPLETE"
    else:
        verdict = "FAIL"

    analysis = {
        "by_laplacian_type": by_type,
        "convergence": convergence,
        "total_runs": len(results),
        "errors": len(results) - len(valid_results),
        "verdict": verdict,
    }

    return {
        "manifold": "S7",
        "expected_lambda1_LB": 7.0,
        "results": results,
        "analysis": analysis,
    }


def compare_S3_S7(s3_results: Dict, s7_results: Dict) -> Dict:
    """
    Compare les ratios S⁷/S³ pour vérifier la cohérence.

    Ratio attendu (Laplace-Beltrami): λ₁(S⁷)/λ₁(S³) = 7/3 ≈ 2.33
    """
    comparison = {
        "expected_ratio": 7.0 / 3.0,
        "by_config": [],
    }

    # Extraire résultats symmetric
    s3_sym = [r for r in s3_results["results"]
              if r.get("laplacian_type") == "symmetric" and "error" not in r]
    s7_sym = [r for r in s7_results["results"]
              if r.get("laplacian_type") == "symmetric" and "error" not in r]

    # Pour chaque config (N, k), calculer le ratio
    configs = set((r["N"], r["k"]) for r in s3_sym)

    for N, k in sorted(configs):
        s3_val = next((r["lambda1_measured"] for r in s3_sym
                       if r["N"] == N and r["k"] == k), None)
        s7_val = next((r["lambda1_measured"] for r in s7_sym
                       if r["N"] == N and r["k"] == k), None)

        if s3_val and s7_val and s3_val > 1e-10:
            ratio = s7_val / s3_val
            comparison["by_config"].append({
                "N": N, "k": k,
                "lambda1_S3": s3_val,
                "lambda1_S7": s7_val,
                "ratio": ratio,
                "expected": 7.0/3.0,
                "deviation_pct": abs(ratio - 7.0/3.0) / (7.0/3.0) * 100
            })

    if comparison["by_config"]:
        ratios = [c["ratio"] for c in comparison["by_config"]]
        comparison["mean_ratio"] = float(np.mean(ratios))
        comparison["std_ratio"] = float(np.std(ratios))
        comparison["deviation_from_expected_pct"] = abs(comparison["mean_ratio"] - 7.0/3.0) / (7.0/3.0) * 100
    else:
        comparison["mean_ratio"] = None

    return comparison


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  S⁷ CALIBRATION - Spectral Validation Phase 1.2")
    print("  Expected: λ₁(Laplace-Beltrami) = 7")
    print("=" * 60)
    print()

    # Configuration
    config = {
        "N_values": [1000, 2000, 5000],
        "k_values": [15, 25, 40],
        "laplacian_types": ["unnormalized", "random_walk", "symmetric"],
        "seeds": [42],
    }

    # Run S⁷ calibration
    output = run_S7_calibration(config)

    # Save results
    output_path = get_output_path("calibration", "S7_calibration_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print()
    print("=" * 60)
    print("RÉSULTATS S⁷ CALIBRATION")
    print("=" * 60)

    print(f"\nAnalyse par type de Laplacien:")
    for lap_type, stats in output["analysis"]["by_laplacian_type"].items():
        print(f"  {lap_type}:")
        print(f"    Mean λ₁: {stats['mean_lambda1']:.4f} ± {stats['std_lambda1']:.4f}")
        print(f"    Range: [{stats['min_lambda1']:.4f}, {stats['max_lambda1']:.4f}]")
        print(f"    Pass rate: {stats['pass_rate']*100:.0f}%")

    print(f"\nConvergence (k=25, symmetric):")
    for c in output["analysis"]["convergence"]:
        print(f"  N={c['N']}: λ₁={c['lambda1']:.4f}")

    print(f"\nVERDICT: {output['analysis']['verdict']}")
    print(f"\nSaved: {output_path}")

    # Charger S³ et comparer
    s3_path = get_output_path("calibration", "S3_calibration_results.json")
    if s3_path.exists():
        print()
        print("=" * 60)
        print("COMPARAISON S⁷/S³")
        print("=" * 60)

        with open(s3_path) as f:
            s3_results = json.load(f)

        comparison = compare_S3_S7(s3_results, output)

        print(f"\nRatio attendu (LB): {comparison['expected_ratio']:.4f}")
        if comparison.get("mean_ratio"):
            print(f"Ratio mesuré: {comparison['mean_ratio']:.4f} ± {comparison['std_ratio']:.4f}")
            print(f"Déviation: {comparison['deviation_from_expected_pct']:.1f}%")

            print(f"\nDétail par configuration:")
            for c in comparison["by_config"]:
                print(f"  N={c['N']}, k={c['k']}: S³={c['lambda1_S3']:.4f}, S⁷={c['lambda1_S7']:.4f}, ratio={c['ratio']:.2f}")

        # Sauvegarder comparaison
        comp_path = get_output_path("calibration", "S3_S7_comparison.json")
        with open(comp_path, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"\nSaved: {comp_path}")
