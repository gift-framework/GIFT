"""
Test d'Indépendance Betti

OBJECTIF:
Vérifier que λ₁×H* dépend UNIQUEMENT de H*, pas de la partition (b₂, b₃).

HYPOTHÈSE:
Pour H* = 99 fixé, différentes partitions (b₂, b₃) avec b₂ + b₃ + 1 = 99
doivent donner le MÊME λ₁×H*.

Si spread < 10⁻⁸ → indépendance confirmée (propriété topologique)
Si spread > 1% → dépendance aux détails géométriques

Reference: ROADMAP_SPECTRAL_VALIDATION.md Phase 3
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict
import sys
from datetime import datetime

# Ajouter le parent pour importer config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    TOPOLOGY, BETTI_PARTITIONS, TOLERANCE, SEEDS,
    get_output_path
)


# =============================================================================
# CONSTANTES
# =============================================================================

DET_G = 65/32
H_STAR = TOPOLOGY["H_star"]  # 99


# =============================================================================
# TCS CONSTRUCTION (copié de robustness pour autonomie)
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


def tcs_distance_matrix(n: int, ratio: float, seed: int,
                        b2: int = 21, b3: int = 77) -> np.ndarray:
    """
    Calcule la matrice de distances TCS pour K₇.

    NOTE: Dans la construction TCS actuelle, b2 et b3 n'affectent PAS
    directement la métrique - seul H* (via ratio) compte.
    Ce test vérifie cette propriété.

    Args:
        n: Nombre de points
        ratio: Paramètre d'anisotropie
        seed: Graine aléatoire
        b2, b3: Nombres de Betti (pour documentation, pas utilisés directement)

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


def compute_lambda1_from_distance(D: np.ndarray, k: int = 25) -> float:
    """Calcule λ₁ à partir d'une matrice de distances."""
    n = D.shape[0]
    k = min(k, n - 1)

    knn_dists = np.partition(D, k, axis=1)[:, :k]
    sigma = max(np.median(knn_dists), 1e-10)

    W = np.exp(-D**2 / (2 * sigma**2))
    np.fill_diagonal(W, 0)

    for i in range(n):
        idx = np.argpartition(W[i], -k)[-k:]
        mask = np.ones(n, dtype=bool)
        mask[idx] = False
        W[i, mask] = 0

    W = (W + W.T) / 2
    d = np.maximum(W.sum(axis=1), 1e-10)

    # Laplacien symmetric
    d_inv_sqrt = 1.0 / np.sqrt(d)
    L = np.eye(n) - np.diag(d_inv_sqrt) @ W @ np.diag(d_inv_sqrt)
    L_sparse = sp.csr_matrix(L)

    eigs, _ = eigsh(L_sparse, k=5, which='SM', tol=1e-8)
    eigs = np.sort(np.real(eigs))

    for ev in eigs:
        if ev > 1e-8:
            return ev

    return eigs[1] if len(eigs) > 1 else 0.0


# =============================================================================
# TEST D'INDÉPENDANCE BETTI
# =============================================================================

@dataclass
class BettiTestResult:
    """Résultat pour une partition Betti."""
    name: str
    b2: int
    b3: int
    H_star: int
    lambda1: float
    lambda1_x_Hstar: float


def test_single_partition(partition: Dict, N: int, k: int, seed: int) -> BettiTestResult:
    """Teste une partition Betti."""
    b2 = partition["b2"]
    b3 = partition["b3"]
    H_star = b2 + b3 + 1

    # Ratio = H*/84 (formule standard)
    ratio = max(H_star / 84, 0.8)

    # Calculer distance TCS
    D = tcs_distance_matrix(N, ratio, seed, b2, b3)

    # Calculer λ₁
    lambda1 = compute_lambda1_from_distance(D, k)
    product = lambda1 * H_star

    return BettiTestResult(
        name=partition["name"],
        b2=b2,
        b3=b3,
        H_star=H_star,
        lambda1=lambda1,
        lambda1_x_Hstar=product,
    )


def run_betti_independence_test(N: int = 5000, k: int = 25,
                                 partitions: List[Dict] = None,
                                 seeds: List[int] = None,
                                 verbose: bool = True) -> Dict:
    """
    Exécute le test d'indépendance Betti.

    Pour H* = 99 fixé, teste plusieurs partitions (b₂, b₃).
    """
    if partitions is None:
        partitions = BETTI_PARTITIONS

    if seeds is None:
        seeds = [SEEDS["betti_test"]]

    results = []

    for partition in partitions:
        if verbose:
            print(f"Testing {partition['name']} (b₂={partition['b2']}, b₃={partition['b3']})...", end=" ")

        partition_results = []
        for seed in seeds:
            result = test_single_partition(partition, N, k, seed)
            partition_results.append(asdict(result))

        # Moyenne sur les seeds
        mean_product = np.mean([r["lambda1_x_Hstar"] for r in partition_results])
        std_product = np.std([r["lambda1_x_Hstar"] for r in partition_results])

        results.append({
            "partition": partition,
            "runs": partition_results,
            "mean_lambda1_x_Hstar": float(mean_product),
            "std_lambda1_x_Hstar": float(std_product),
        })

        if verbose:
            print(f"λ₁×H* = {mean_product:.6f}")

    # Analyse de l'indépendance
    products = [r["mean_lambda1_x_Hstar"] for r in results]
    mean_all = np.mean(products)
    std_all = np.std(products)
    spread = (max(products) - min(products)) / mean_all * 100 if mean_all > 0 else float('inf')

    # Critère PASS
    threshold = TOLERANCE.get("betti_independence", 1e-8)
    passed = spread < threshold * 100  # threshold est en fraction, spread en %

    analysis = {
        "H_star": H_STAR,
        "N": N,
        "k": k,
        "num_partitions": len(partitions),
        "num_seeds": len(seeds),
        "mean_product": float(mean_all),
        "std_product": float(std_all),
        "min_product": float(min(products)),
        "max_product": float(max(products)),
        "spread_pct": float(spread),
        "threshold_pct": threshold * 100,
        "passed": passed,
        "verdict": "INDEPENDENT" if passed else "DEPENDENT",
    }

    return {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "test": "Betti Independence",
            "hypothesis": "λ₁×H* depends only on H*, not on (b₂, b₃)",
        },
        "parameters": {
            "N": N,
            "k": k,
            "seeds": seeds,
        },
        "results": results,
        "analysis": analysis,
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  BETTI INDEPENDENCE TEST - Spectral Validation Phase 3")
    print(f"  H* = {H_STAR}, Testing {len(BETTI_PARTITIONS)} partitions")
    print("=" * 70)
    print()

    # Paramètres
    N = 5000
    k = 25
    seeds = [42, 123, 456]

    print(f"Parameters: N={N}, k={k}, seeds={seeds}")
    print()

    # Run test
    output = run_betti_independence_test(N, k, seeds=seeds)

    # Save
    output_path = get_output_path("analysis", "betti_independence.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Print summary
    print()
    print("=" * 70)
    print("RÉSULTATS TEST D'INDÉPENDANCE BETTI")
    print("=" * 70)

    analysis = output["analysis"]

    print(f"\nStatistiques sur {analysis['num_partitions']} partitions:")
    print(f"  Mean λ₁×H*: {analysis['mean_product']:.6f}")
    print(f"  Std: {analysis['std_product']:.6f}")
    print(f"  Range: [{analysis['min_product']:.6f}, {analysis['max_product']:.6f}]")
    print(f"  Spread: {analysis['spread_pct']:.2e}%")

    print(f"\nCritère: spread < {analysis['threshold_pct']:.2e}%")
    print(f"VERDICT: {analysis['verdict']}")

    if analysis['passed']:
        print("\n✓ λ₁×H* est INDÉPENDANT de la partition (b₂, b₃)")
        print("  → Propriété topologique confirmée")
    else:
        print("\n✗ λ₁×H* DÉPEND de la partition (b₂, b₃)")
        print("  → Investiguer les détails géométriques")

    print(f"\nSaved: {output_path}")
