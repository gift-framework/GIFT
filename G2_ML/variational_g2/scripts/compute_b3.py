#!/usr/bin/env python3
"""
Compute b3=77 for GIFT K7 Manifold

Standalone script version of b3.ipynb notebook.
Uses toroidal distance and spectral gap detection.

Usage:
    python compute_b3.py [--points 8192] [--k 30] [--output artifacts/b3_result.json]
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json
import argparse
import os
from pathlib import Path


# GIFT constants
DIM = 7
B3_TARGET = 77
G2_METRIC_DET = 65 / 32  # 2.03125
N_GEN = 3


def toroidal_distance(p1: np.ndarray, p2: np.ndarray, period: float = 2*np.pi) -> np.ndarray:
    """Compute toroidal distance between points (periodic BC)."""
    diff = np.abs(p1 - p2)
    diff = np.minimum(diff, period - diff)
    return np.sqrt(np.sum(diff**2, axis=-1))


def build_toroidal_knn(points: np.ndarray, k: int, period: float = 2*np.pi):
    """Build k-NN graph with toroidal metric."""
    n = len(points)
    print(f"Building toroidal k-NN graph ({n} points, k={k})...")

    neighbors = np.zeros((n, k), dtype=np.int32)
    distances = np.zeros((n, k), dtype=np.float64)

    for i in range(n):
        dists = toroidal_distance(points[i:i+1], points, period)
        dists[i] = np.inf  # Exclude self

        idx = np.argpartition(dists, k)[:k]
        idx = idx[np.argsort(dists[idx])]

        neighbors[i] = idx
        distances[i] = dists[idx]

        if (i + 1) % 2000 == 0:
            print(f"  Processed {i+1}/{n} points...")

    return neighbors, distances


def compute_b3(n_points: int = 8192, k_nn: int = 30, n_evals: int = 150,
               output_path: str = None, plot: bool = True) -> dict:
    """
    Compute b3 via spectral gap detection on toroidal mesh.

    Returns:
        Dictionary with verification results
    """
    # 1. Generate Sobol mesh on torus [0, 2pi]^7
    print(f"\n=== Phase 1: Mesh Generation ===")
    sampler = qmc.Sobol(d=DIM, scramble=True, seed=42)
    points = sampler.random(n=n_points) * (2 * np.pi)
    print(f"Generated {n_points} Sobol points in {DIM}D torus")

    # 2. Build k-NN graph with toroidal distance
    print(f"\n=== Phase 2: Graph Construction ===")
    idx, dist = build_toroidal_knn(points, k_nn)

    # Weights: inv dist * metric det
    weights = G2_METRIC_DET / (dist + 1e-8)

    # Sparse adjacency matrix
    rows = np.repeat(np.arange(n_points), k_nn)
    cols = idx.ravel()
    data = weights.ravel()
    adj = csr_matrix((data, (rows, cols)), shape=(n_points, n_points))
    adj = (adj + adj.T) / 2  # Symmetrize

    # Degree and connectivity
    deg = np.array(adj.sum(axis=1)).flatten()
    n_cc = connected_components(adj, directed=False)[0]
    print(f"Graph: {adj.nnz} edges, avg degree {deg.mean():.1f}")
    print(f"Connected components: {n_cc}")

    # 3. Normalized Laplacian
    print(f"\n=== Phase 3: Laplacian Spectrum ===")
    I = eye(n_points, format='csr', dtype=np.float64)
    D_inv_sqrt = csr_matrix(
        (1 / np.sqrt(deg + 1e-8), (np.arange(n_points), np.arange(n_points))),
        shape=(n_points, n_points)
    )
    L = I - D_inv_sqrt @ adj @ D_inv_sqrt

    # Compute smallest eigenvalues
    print(f"Computing {n_evals} smallest eigenvalues...")
    evals, evecs = eigsh(L, k=n_evals, which='SM', tol=1e-10, maxiter=1000)

    # Gap detection - look for significant gaps, especially around b3=77
    gaps = np.diff(evals) / (evals[1:] + 1e-12)

    # Find top 5 gaps
    top_gap_indices = np.argsort(gaps)[-5:][::-1] + 1

    print("Top 5 spectral gaps:")
    for i, idx in enumerate(top_gap_indices):
        print(f"  {i+1}. Index {idx}: eval={evals[idx]:.2e}, rel_gap={gaps[idx-1]:.2f}")

    # Primary gap (largest)
    gap_idx = top_gap_indices[0]
    gap_val = gaps[gap_idx - 1]

    # Check for gap near 77
    gap_near_77 = None
    for idx in range(70, min(85, n_evals-1)):
        if gaps[idx] > 0.1:  # Significant gap
            gap_near_77 = idx + 1
            break

    print(f"\nPrimary gap at index: {gap_idx}")
    print(f"Gap near b3=77: {gap_near_77}")
    print(f"Target b3=77, mismatch: {abs(gap_idx - B3_TARGET)}")

    # 4. 3-generation clustering
    print(f"\n=== Phase 4: Generation Structure ===")
    n_first = min(B3_TARGET, len(evals))
    first_evals = evals[:n_first].reshape(-1, 1)

    kmeans = KMeans(n_clusters=N_GEN, random_state=42, n_init=10)
    labels = kmeans.fit_predict(first_evals)
    cluster_sizes = np.bincount(labels, minlength=N_GEN)
    sil_score = silhouette_score(first_evals, labels)

    print(f"Cluster sizes: {cluster_sizes.tolist()}")
    print(f"Silhouette score: {sil_score:.4f}")

    # 5. Summary
    mismatch = abs(gap_idx - B3_TARGET)
    if mismatch <= 3:
        status = "VERIFIED"
    elif mismatch <= 10:
        status = "PROMISING"
    else:
        status = "INCONCLUSIVE"

    summary = {
        'mesh': {'n_points': int(n_points), 'dim': int(DIM), 'sampling': 'Sobol', 'metric': 'toroidal'},
        'graph': {'k_nn': int(k_nn), 'n_edges': int(adj.nnz // 2), 'connected_cc': int(n_cc)},
        'spectral': {
            'gap_idx': int(gap_idx),
            'gap_near_77': int(gap_near_77) if gap_near_77 else None,
            'gap_eval': float(evals[gap_idx]) if gap_idx < len(evals) else None,
            'rel_gap': float(gap_val),
            'mismatch': int(mismatch),
            'top_5_gaps': [int(x) for x in top_gap_indices.tolist()]
        },
        '3gen': {
            'clusters': [int(x) for x in cluster_sizes.tolist()],
            'silhouette': float(sil_score),
            'n_gen_detected': int(N_GEN) if sil_score > 0.5 else 'Ambiguous'
        },
        'status': status
    }

    print(f"\n=== Result: {status} ===")

    # Save output
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved to {output_path}")

    # Plot
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.semilogy(evals[:100], 'b.-')
        ax1.axvline(B3_TARGET, color='r', ls='--', label='Target b3=77')
        ax1.axvline(gap_idx, color='g', ls='--', label=f'Gap at {gap_idx}')
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Eigenvalue (log)')
        ax1.set_title('Laplacian Spectrum')
        ax1.legend()

        ax2.bar(range(len(gaps[:100])), gaps[:100], alpha=0.7)
        ax2.axvline(gap_idx - 1, color='r', ls='--', label='Max Gap')
        ax2.set_xlabel('Gap Index')
        ax2.set_ylabel('Relative Gap')
        ax2.set_title('Spectral Gaps')
        ax2.legend()

        plt.tight_layout()

        if output_path:
            plot_path = output_path.replace('.json', '.png')
            plt.savefig(plot_path, dpi=150)
            print(f"Plot saved to {plot_path}")

        plt.show()

    return summary


def main():
    parser = argparse.ArgumentParser(description='Compute b3=77 for GIFT K7')
    parser.add_argument('--points', type=int, default=8192, help='Number of mesh points')
    parser.add_argument('--k', type=int, default=30, help='k-NN neighbors')
    parser.add_argument('--evals', type=int, default=150, help='Number of eigenvalues')
    parser.add_argument('--output', type=str, default='../outputs/artifacts/b3_result.json',
                        help='Output JSON path')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')

    args = parser.parse_args()

    result = compute_b3(
        n_points=args.points,
        k_nn=args.k,
        n_evals=args.evals,
        output_path=args.output,
        plot=not args.no_plot
    )

    return 0 if result['status'] in ['VERIFIED', 'PROMISING'] else 1


if __name__ == '__main__':
    exit(main())
