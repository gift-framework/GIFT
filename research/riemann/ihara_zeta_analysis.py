#!/usr/bin/env python3
"""
Ihara Zeta Function Analysis for GIFT-Related Graphs
=====================================================

The Ihara zeta function ζ_G(u) for a graph G encodes spectral information
analogous to how ζ(s) encodes prime information.

Key question: Do graphs with GIFT symmetry (E₈, G₂, Weyl groups)
have Ihara zeros with GIFT structure?

Ihara-Bass formula:
    ζ_G(u)^{-1} = (1-u²)^{r-1} det(I - Au + (D-I)u²)

where:
    - A = adjacency matrix
    - D = degree matrix (diagonal)
    - r = rank = |E| - |V| + 1 (cyclomatic number)

Author: GIFT Research
Date: February 2026
"""

import numpy as np
from numpy.linalg import det, eigvals
from typing import List, Tuple, Dict
import json

# GIFT Constants
GIFT = {
    'p2': 2, 'N_gen': 3, 'Weyl': 5, 'dim_K7': 7, 'rank_E8': 8,
    'D_bulk': 11, 'F7': 13, 'dim_G2': 14, 'b2': 21, 'dim_J3O': 27,
    'h_G2_sq': 36, 'b3': 77, 'H_star': 99, 'dim_E8': 248
}


def cycle_graph(n: int) -> np.ndarray:
    """Create adjacency matrix for cycle graph C_n."""
    A = np.zeros((n, n))
    for i in range(n):
        A[i, (i+1) % n] = 1
        A[i, (i-1) % n] = 1
    return A


def complete_graph(n: int) -> np.ndarray:
    """Create adjacency matrix for complete graph K_n."""
    A = np.ones((n, n)) - np.eye(n)
    return A


def petersen_graph() -> np.ndarray:
    """Create the Petersen graph (10 vertices, 3-regular)."""
    # Famous graph with interesting spectral properties
    A = np.zeros((10, 10))
    # Outer pentagon
    for i in range(5):
        A[i, (i+1) % 5] = 1
        A[(i+1) % 5, i] = 1
    # Inner pentagram
    for i in range(5):
        A[5+i, 5+(i+2) % 5] = 1
        A[5+(i+2) % 5, 5+i] = 1
    # Spokes
    for i in range(5):
        A[i, 5+i] = 1
        A[5+i, i] = 1
    return A


def e8_root_graph() -> np.ndarray:
    """
    Create a graph based on E₈ root system adjacency.

    E₈ has 240 roots. Two roots are connected if their
    inner product is ±1 (adjacent in root system).

    For computational tractability, we use the E₈ Dynkin diagram
    (8 vertices) extended to capture some root structure.
    """
    # E₈ Dynkin diagram (8 nodes)
    # Structure: 1-2-3-4-5-6-7 with 8 connected to 4
    A = np.zeros((8, 8))
    edges = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (3,7)]
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1
    return A


def g2_root_graph() -> np.ndarray:
    """
    Create a graph based on G₂ root system.

    G₂ has 12 roots (6 short, 6 long).
    The Dynkin diagram is just 2 nodes with a triple edge.

    We create a 12-vertex graph from the actual root adjacency.
    """
    # G₂ roots in 2D (simplified representation)
    # 6 short roots + 6 long roots
    # Connect roots at 60° angles
    A = np.zeros((12, 12))
    # Short roots form a hexagon
    for i in range(6):
        A[i, (i+1) % 6] = 1
        A[(i+1) % 6, i] = 1
    # Long roots form another hexagon
    for i in range(6):
        A[6+i, 6+(i+1) % 6] = 1
        A[6+(i+1) % 6, 6+i] = 1
    # Connect short to adjacent long roots
    for i in range(6):
        A[i, 6+i] = 1
        A[6+i, i] = 1
        A[i, 6+(i+1) % 6] = 1
        A[6+(i+1) % 6, i] = 1
    return A


def weyl_a_n_graph(n: int) -> np.ndarray:
    """Cayley graph of symmetric group S_{n+1} (Weyl group of A_n)."""
    # Too large for n > 4, use Dynkin diagram instead
    # A_n Dynkin: linear chain of n nodes
    A = np.zeros((n, n))
    for i in range(n-1):
        A[i, i+1] = 1
        A[i+1, i] = 1
    return A


def ihara_zeta_inverse(A: np.ndarray, u: complex) -> complex:
    """
    Compute ζ_G(u)^{-1} using Ihara-Bass formula.

    ζ_G(u)^{-1} = (1-u²)^{r-1} det(I - Au + (D-I)u²)

    where r = |E| - |V| + 1 is the cyclomatic number.
    """
    n = A.shape[0]

    # Degree matrix
    degrees = np.sum(A, axis=1)
    D = np.diag(degrees)

    # Number of edges
    num_edges = int(np.sum(A) / 2)

    # Cyclomatic number (first Betti number of graph)
    r = num_edges - n + 1

    # Identity
    I = np.eye(n)

    # Ihara-Bass determinant
    M = I - A * u + (D - I) * (u ** 2)
    det_M = det(M)

    # Full formula
    zeta_inv = ((1 - u**2) ** (r - 1)) * det_M

    return zeta_inv


def find_ihara_zeros(A: np.ndarray, num_points: int = 1000,
                      r_max: float = 1.5) -> List[complex]:
    """
    Find zeros of ζ_G(u)^{-1} by grid search.

    For regular graphs, zeros lie on |u| = 1/sqrt(q-1) where q = degree.
    """
    zeros = []

    # Search in complex plane
    for real in np.linspace(-r_max, r_max, num_points):
        for imag in np.linspace(-r_max, r_max, num_points):
            u = complex(real, imag)
            if abs(u) < 0.01:  # Skip origin
                continue

            try:
                val = ihara_zeta_inverse(A, u)
                if abs(val) < 0.01:  # Near zero
                    # Refine
                    zeros.append(u)
            except:
                pass

    # Remove duplicates (within tolerance)
    unique_zeros = []
    for z in zeros:
        is_dup = False
        for uz in unique_zeros:
            if abs(z - uz) < 0.05:
                is_dup = True
                break
        if not is_dup:
            unique_zeros.append(z)

    return unique_zeros


def spectral_analysis(A: np.ndarray) -> Dict:
    """Analyze adjacency spectrum of graph."""
    eigenvalues = eigvals(A)
    eigenvalues = np.sort(np.real(eigenvalues))[::-1]  # Descending

    n = len(eigenvalues)

    return {
        'num_vertices': n,
        'num_edges': int(np.sum(A) / 2),
        'max_eigenvalue': eigenvalues[0],
        'min_eigenvalue': eigenvalues[-1],
        'spectral_gap': eigenvalues[0] - eigenvalues[1] if n > 1 else 0,
        'eigenvalues': eigenvalues.tolist()
    }


def analyze_graph(name: str, A: np.ndarray) -> Dict:
    """Full analysis of a graph for GIFT patterns."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {name}")
    print('='*60)

    # Basic properties
    n = A.shape[0]
    m = int(np.sum(A) / 2)
    degrees = np.sum(A, axis=1)
    is_regular = len(set(degrees)) == 1

    print(f"Vertices: {n}")
    print(f"Edges: {m}")
    print(f"Regular: {is_regular}" + (f" (degree {int(degrees[0])})" if is_regular else ""))

    # Cyclomatic number (first Betti)
    r = m - n + 1
    print(f"Cyclomatic number (β₁): {r}")

    # Check GIFT
    gift_matches = []
    for const_name, val in GIFT.items():
        if n == val:
            gift_matches.append(f"n = {const_name}")
        if m == val:
            gift_matches.append(f"m = {const_name}")
        if r == val:
            gift_matches.append(f"β₁ = {const_name}")

    if gift_matches:
        print(f"GIFT matches: {', '.join(gift_matches)}")

    # Spectral analysis
    spec = spectral_analysis(A)
    print(f"\nSpectral properties:")
    print(f"  λ_max = {spec['max_eigenvalue']:.6f}")
    print(f"  λ_min = {spec['min_eigenvalue']:.6f}")
    print(f"  Spectral gap = {spec['spectral_gap']:.6f}")

    # Check if spectral gap or eigenvalues match GIFT
    print(f"\nEigenvalues: {[f'{e:.4f}' for e in spec['eigenvalues'][:10]]}")

    # GIFT eigenvalue check
    for i, ev in enumerate(spec['eigenvalues']):
        for const_name, val in GIFT.items():
            if abs(ev - val) < 0.1:
                print(f"  λ_{i+1} ≈ {const_name} = {val}")
            if val > 0 and abs(ev - val/10) < 0.05:
                print(f"  λ_{i+1} ≈ {const_name}/10 = {val/10}")

    # Ihara zeros (simplified)
    print(f"\nIhara zeta analysis:")

    # For regular graphs, poles are at u = ±1/√(q-1)
    if is_regular:
        q = int(degrees[0])
        if q > 1:
            pole_radius = 1 / np.sqrt(q - 1)
            print(f"  Expected pole radius: 1/√({q}-1) = {pole_radius:.6f}")

    # Evaluate ζ^{-1} at some test points
    test_points = [0.1, 0.5, 1/np.sqrt(2), 0.9]
    print(f"  ζ_G(u)^{{-1}} at test points:")
    for u in test_points:
        val = ihara_zeta_inverse(A, u)
        print(f"    u = {u:.3f}: ζ^{{-1}} = {val:.6f}")

    return {
        'name': name,
        'vertices': n,
        'edges': m,
        'cyclomatic': r,
        'is_regular': is_regular,
        'spectral': spec,
        'gift_matches': gift_matches
    }


def main():
    print("=" * 70)
    print("IHARA ZETA FUNCTION ANALYSIS FOR GIFT-RELATED GRAPHS")
    print("=" * 70)

    results = []

    # Analyze various graphs
    graphs = [
        ("Cycle C_7 (dim K₇)", cycle_graph(7)),
        ("Cycle C_8 (rank E₈)", cycle_graph(8)),
        ("Cycle C_14 (dim G₂)", cycle_graph(14)),
        ("Cycle C_21 (b₂)", cycle_graph(21)),
        ("Complete K_7", complete_graph(7)),
        ("Complete K_8", complete_graph(8)),
        ("Petersen Graph", petersen_graph()),
        ("E₈ Dynkin Graph", e8_root_graph()),
        ("G₂ Root Graph (12 vertices)", g2_root_graph()),
        ("A_7 Dynkin (Weyl)", weyl_a_n_graph(7)),
    ]

    for name, A in graphs:
        result = analyze_graph(name, A)
        results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: GIFT PATTERNS IN GRAPH INVARIANTS")
    print("=" * 70)

    print("\n| Graph | n | m | β₁ | GIFT Matches |")
    print("|-------|---|---|----|--------------| ")
    for r in results:
        matches = ", ".join(r['gift_matches']) if r['gift_matches'] else "-"
        print(f"| {r['name'][:20]:20s} | {r['vertices']:3d} | {r['edges']:3d} | {r['cyclomatic']:3d} | {matches} |")

    # Save results
    with open('/home/user/GIFT/research/riemann/ihara_analysis_results.json', 'w') as f:
        # Convert numpy types
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(x) for x in obj]
            return obj
        json.dump(convert(results), f, indent=2)

    print(f"\n✓ Results saved to ihara_analysis_results.json")


if __name__ == "__main__":
    main()
