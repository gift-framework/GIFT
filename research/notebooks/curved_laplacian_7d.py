"""
Sparse Curved Laplacian Operator on 7D Domain with Diagonal Metric

This module constructs the curved Laplacian operator:
    Δ_g f = (1/√g) ∂_i(√g g^{ii} ∂_i f)

for a diagonal metric g_ij = diag(g_11, ..., g_77) on a 7-dimensional domain.

Uses finite differences on a regular grid with scipy.sparse for memory efficiency.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh, LinearOperator
from typing import Callable, Tuple, Optional, Literal
import warnings


def build_curved_laplacian(
    grid_points_per_dim: int,
    domain_size: np.ndarray,
    metric_func: Callable[[np.ndarray], np.ndarray],
    boundary: Literal['periodic', 'dirichlet'] = 'periodic'
) -> Tuple[sparse.csr_matrix, np.ndarray]:
    """
    Build sparse matrix for curved Laplacian on 7D domain with diagonal metric.

    The curved Laplacian for diagonal metric g_ij = diag(g_11, ..., g_77) is:
        Δ_g f = (1/√g) ∂_i(√g g^{ii} ∂_i f)

    where g = det(g_ij) = ∏_i g_ii and g^{ii} = 1/g_ii

    Discretization uses conservative finite differences:
        ∂_i(A_i ∂_i f) ≈ [A_{i+1/2}(f_{i+1} - f_i) - A_{i-1/2}(f_i - f_{i-1})] / h²

    where A_i = √g g^{ii} and half-point values are interpolated.

    Args:
        grid_points_per_dim: N (total points = N^7)
        domain_size: array of 7 domain lengths [L_0, ..., L_6]
        metric_func: function(x) -> g_diag, where x is (7,) position
                     and g_diag is (7,) diagonal metric components g_ii
        boundary: 'periodic' or 'dirichlet' boundary conditions

    Returns:
        L: sparse matrix (N^7 x N^7) in CSR format
        points: array of grid points (N^7, 7)
    """
    N = grid_points_per_dim
    dim = 7
    domain_size = np.asarray(domain_size, dtype=np.float64)

    if len(domain_size) != dim:
        raise ValueError(f"domain_size must have {dim} elements")

    # Grid spacing in each dimension
    if boundary == 'periodic':
        h = domain_size / N  # N points cover [0, L) periodically
    else:
        h = domain_size / (N - 1)  # N points cover [0, L] with endpoints

    # Total number of grid points
    total_points = N ** dim

    print(f"Building curved Laplacian: {N}^{dim} = {total_points:,} grid points")
    print(f"Grid spacing: h = {h}")

    # Generate all grid points using efficient indexing
    if boundary == 'periodic':
        coords_1d = [np.linspace(0, domain_size[d], N, endpoint=False) for d in range(dim)]
    else:
        coords_1d = [np.linspace(0, domain_size[d], N) for d in range(dim)]

    # Create flattened coordinate arrays (memory-efficient)
    # Point index i corresponds to multi-index (i0, i1, ..., i6)
    # where i = i0*N^6 + i1*N^5 + ... + i6
    points = np.zeros((total_points, dim), dtype=np.float64)

    strides = np.array([N**(dim - 1 - d) for d in range(dim)])

    for d in range(dim):
        # Replicate coordinate values according to the stride pattern
        pattern_size = N ** (dim - d)
        repeat_count = N ** d
        tile = np.tile(np.repeat(coords_1d[d], N ** (dim - 1 - d)), repeat_count)
        points[:, d] = tile

    print("Computing metric at all grid points...")

    # Precompute metric at all points
    metric_diag = np.zeros((total_points, dim), dtype=np.float64)
    for i in range(total_points):
        metric_diag[i] = metric_func(points[i])

    # Compute √g = √(∏_i g_ii) at each point
    sqrt_g = np.sqrt(np.prod(metric_diag, axis=1))

    # Compute A^d = √g * g^{dd} = √g / g_dd at each point for each dimension
    A = sqrt_g[:, np.newaxis] / metric_diag  # shape (total_points, dim)

    print("Building sparse matrix...")

    # Build sparse matrix using COO format (efficient for construction)
    # Maximum non-zeros: 15 * total_points (self + 2*dim neighbors)
    max_nnz = (1 + 2 * dim) * total_points
    rows = np.zeros(max_nnz, dtype=np.int64)
    cols = np.zeros(max_nnz, dtype=np.int64)
    data = np.zeros(max_nnz, dtype=np.float64)
    nnz_count = 0

    # Precompute stride multipliers for index conversion
    # multi_idx -> linear: sum(multi_idx[d] * strides[d])
    # linear -> multi_idx[d]: (linear // strides[d]) % N

    h_sq = h * h

    # Iterate over all grid points
    for lin_idx in range(total_points):
        # Convert to multi-index
        multi_idx = np.array([(lin_idx // strides[d]) % N for d in range(dim)])

        sqrt_g_here = sqrt_g[lin_idx]
        diagonal_val = 0.0

        # Loop over dimensions
        for d in range(dim):
            # Neighbor indices in dimension d
            i_plus = multi_idx[d] + 1
            i_minus = multi_idx[d] - 1

            # Handle boundary conditions
            if boundary == 'periodic':
                i_plus = i_plus % N
                i_minus = i_minus % N
                has_plus = True
                has_minus = True
            else:  # Dirichlet
                has_plus = (i_plus < N)
                has_minus = (i_minus >= 0)

            # Linear index of neighbors
            if has_plus:
                lin_plus = lin_idx + (1 if i_plus == multi_idx[d] + 1 else 1 - N) * strides[d]
                if boundary == 'periodic':
                    lin_plus = lin_idx - multi_idx[d] * strides[d] + i_plus * strides[d]
                else:
                    lin_plus = lin_idx + strides[d]

                # A at half-point (i + 1/2) - interpolate
                A_plus_half = 0.5 * (A[lin_idx, d] + A[lin_plus, d])

                coeff_plus = A_plus_half / (sqrt_g_here * h_sq[d])

                rows[nnz_count] = lin_idx
                cols[nnz_count] = lin_plus
                data[nnz_count] = coeff_plus
                nnz_count += 1

                diagonal_val -= coeff_plus

            if has_minus:
                if boundary == 'periodic':
                    lin_minus = lin_idx - multi_idx[d] * strides[d] + i_minus * strides[d]
                else:
                    lin_minus = lin_idx - strides[d]

                # A at half-point (i - 1/2) - interpolate
                A_minus_half = 0.5 * (A[lin_idx, d] + A[lin_minus, d])

                coeff_minus = A_minus_half / (sqrt_g_here * h_sq[d])

                rows[nnz_count] = lin_idx
                cols[nnz_count] = lin_minus
                data[nnz_count] = coeff_minus
                nnz_count += 1

                diagonal_val -= coeff_minus

        # Add diagonal entry
        rows[nnz_count] = lin_idx
        cols[nnz_count] = lin_idx
        data[nnz_count] = diagonal_val
        nnz_count += 1

    # Trim arrays to actual size
    rows = rows[:nnz_count]
    cols = cols[:nnz_count]
    data = data[:nnz_count]

    # Create sparse matrix in COO format, then convert to CSR
    L = sparse.coo_matrix((data, (rows, cols)), shape=(total_points, total_points))
    L = L.tocsr()

    print(f"Matrix built: {L.nnz:,} non-zeros, density = {L.nnz / total_points**2:.2e}")

    return L, points


def build_curved_laplacian_optimized(
    grid_points_per_dim: int,
    domain_size: np.ndarray,
    metric_func: Callable[[np.ndarray], np.ndarray],
    boundary: Literal['periodic', 'dirichlet'] = 'periodic'
) -> Tuple[sparse.csr_matrix, np.ndarray]:
    """
    Optimized version using vectorized operations where possible.

    More memory-efficient for larger grids.
    """
    N = grid_points_per_dim
    dim = 7
    domain_size = np.asarray(domain_size, dtype=np.float64)

    if boundary == 'periodic':
        h = domain_size / N
    else:
        h = domain_size / (N - 1)

    total_points = N ** dim
    strides = np.array([N**(dim - 1 - d) for d in range(dim)], dtype=np.int64)

    print(f"Building optimized curved Laplacian: {N}^{dim} = {total_points:,} points")

    # Generate points
    if boundary == 'periodic':
        coords_1d = [np.linspace(0, domain_size[d], N, endpoint=False) for d in range(dim)]
    else:
        coords_1d = [np.linspace(0, domain_size[d], N) for d in range(dim)]

    points = np.zeros((total_points, dim), dtype=np.float64)
    for d in range(dim):
        pattern = np.repeat(coords_1d[d], strides[d])
        points[:, d] = np.tile(pattern, N ** d)

    # Compute metric
    print("Computing metric...")
    metric_diag = np.array([metric_func(points[i]) for i in range(total_points)])
    sqrt_g = np.sqrt(np.prod(metric_diag, axis=1))
    A = sqrt_g[:, np.newaxis] / metric_diag

    # Build matrix dimension by dimension
    print("Building sparse matrix by dimension...")
    h_sq = h * h

    # Start with zero diagonal
    diag_data = np.zeros(total_points, dtype=np.float64)

    # Lists to collect off-diagonal entries
    all_rows = []
    all_cols = []
    all_data = []

    for d in range(dim):
        print(f"  Processing dimension {d}...")
        stride = strides[d]

        # Compute multi-index for dimension d
        multi_idx_d = (np.arange(total_points) // stride) % N

        # Forward neighbors (i -> i+1 in dimension d)
        if boundary == 'periodic':
            # All points have forward neighbors
            forward_mask = np.ones(total_points, dtype=bool)
            neighbor_idx = (multi_idx_d + 1) % N
            lin_plus = np.arange(total_points) - multi_idx_d * stride + neighbor_idx * stride
        else:
            forward_mask = multi_idx_d < (N - 1)
            lin_plus = np.arange(total_points) + stride

        if np.any(forward_mask):
            idx_here = np.arange(total_points)[forward_mask]
            idx_plus = lin_plus[forward_mask] if boundary == 'periodic' else (np.arange(total_points) + stride)[forward_mask]

            A_plus_half = 0.5 * (A[idx_here, d] + A[idx_plus, d])
            coeff_plus = A_plus_half / (sqrt_g[idx_here] * h_sq[d])

            all_rows.append(idx_here)
            all_cols.append(idx_plus)
            all_data.append(coeff_plus)

            # Update diagonal
            np.subtract.at(diag_data, idx_here, coeff_plus)

        # Backward neighbors (i -> i-1 in dimension d)
        if boundary == 'periodic':
            backward_mask = np.ones(total_points, dtype=bool)
            neighbor_idx = (multi_idx_d - 1) % N
            lin_minus = np.arange(total_points) - multi_idx_d * stride + neighbor_idx * stride
        else:
            backward_mask = multi_idx_d > 0
            lin_minus = np.arange(total_points) - stride

        if np.any(backward_mask):
            idx_here = np.arange(total_points)[backward_mask]
            idx_minus = lin_minus[backward_mask] if boundary == 'periodic' else (np.arange(total_points) - stride)[backward_mask]

            A_minus_half = 0.5 * (A[idx_here, d] + A[idx_minus, d])
            coeff_minus = A_minus_half / (sqrt_g[idx_here] * h_sq[d])

            all_rows.append(idx_here)
            all_cols.append(idx_minus)
            all_data.append(coeff_minus)

            # Update diagonal
            np.subtract.at(diag_data, idx_here, coeff_minus)

    # Combine all entries
    rows = np.concatenate(all_rows + [np.arange(total_points)])
    cols = np.concatenate(all_cols + [np.arange(total_points)])
    data = np.concatenate(all_data + [diag_data])

    L = sparse.coo_matrix((data, (rows, cols)), shape=(total_points, total_points))
    L = L.tocsr()

    print(f"Matrix built: {L.nnz:,} non-zeros")

    return L, points


def compute_eigenvalues(
    L: sparse.spmatrix,
    k: int = 10,
    sigma: Optional[float] = 0.0,
    which: str = 'LM',
    tol: float = 1e-8,
    maxiter: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the first few eigenvalues of the Laplacian.

    For the Laplacian, eigenvalues are non-positive. The smallest magnitude
    eigenvalue (closest to 0) corresponds to constant modes (for periodic BC)
    or the ground state (for Dirichlet BC).

    Args:
        L: sparse Laplacian matrix
        k: number of eigenvalues to compute
        sigma: shift for shift-invert mode. Use sigma=0 with which='LM'
               to find eigenvalues closest to 0.
        which: 'LM' (largest magnitude after shift), 'SM' (smallest magnitude),
               'LA' (largest algebraic), 'SA' (smallest algebraic)
        tol: convergence tolerance
        maxiter: maximum iterations

    Returns:
        eigenvalues: array of k eigenvalues (sorted from largest to smallest,
                     i.e., closest to zero first)
        eigenvectors: array of shape (N^7, k) with eigenvectors as columns
    """
    n = L.shape[0]

    if k >= n - 1:
        warnings.warn(f"Reducing k from {k} to {n-2} (matrix size is {n})")
        k = n - 2

    print(f"Computing {k} eigenvalues using shift-invert (sigma={sigma})...")

    try:
        eigenvalues, eigenvectors = eigsh(
            L, k=k, sigma=sigma, which=which,
            tol=tol, maxiter=maxiter
        )
    except Exception as e:
        print(f"eigsh with shift-invert failed: {e}")
        print("Trying without shift-invert...")
        eigenvalues, eigenvectors = eigsh(
            L, k=k, which='SA',  # Smallest algebraic
            tol=tol, maxiter=maxiter
        )

    # Sort by eigenvalue (most negative to least negative, or closest to 0)
    idx = np.argsort(-eigenvalues)  # Descending order (closest to 0 first)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors


def compute_spectrum_without_matrix(
    grid_points_per_dim: int,
    domain_size: np.ndarray,
    metric_func: Callable[[np.ndarray], np.ndarray],
    k: int = 10,
    boundary: Literal['periodic', 'dirichlet'] = 'periodic',
    tol: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute eigenvalues using matrix-free LinearOperator (more memory efficient).

    This approach doesn't store the full sparse matrix, only the metric data.
    Useful when even the sparse matrix is too large.

    Args:
        grid_points_per_dim: N
        domain_size: domain lengths
        metric_func: diagonal metric function
        k: number of eigenvalues
        boundary: boundary condition type
        tol: convergence tolerance

    Returns:
        eigenvalues: array of k eigenvalues
        eigenvectors: array of shape (N^7, k)
        points: grid points array
    """
    N = grid_points_per_dim
    dim = 7
    domain_size = np.asarray(domain_size, dtype=np.float64)

    if boundary == 'periodic':
        h = domain_size / N
    else:
        h = domain_size / (N - 1)

    total_points = N ** dim
    strides = np.array([N**(dim - 1 - d) for d in range(dim)], dtype=np.int64)
    h_sq = h * h

    print(f"Matrix-free eigenvalue computation: {N}^{dim} = {total_points:,} points")

    # Generate points
    if boundary == 'periodic':
        coords_1d = [np.linspace(0, domain_size[d], N, endpoint=False) for d in range(dim)]
    else:
        coords_1d = [np.linspace(0, domain_size[d], N) for d in range(dim)]

    points = np.zeros((total_points, dim), dtype=np.float64)
    for d in range(dim):
        pattern = np.repeat(coords_1d[d], strides[d])
        points[:, d] = np.tile(pattern, N ** d)

    # Precompute metric data
    print("Precomputing metric data...")
    metric_diag = np.array([metric_func(points[i]) for i in range(total_points)])
    sqrt_g = np.sqrt(np.prod(metric_diag, axis=1))
    A = sqrt_g[:, np.newaxis] / metric_diag

    # Precompute multi-indices for all dimensions
    multi_idx = np.zeros((total_points, dim), dtype=np.int64)
    for d in range(dim):
        multi_idx[:, d] = (np.arange(total_points) // strides[d]) % N

    def laplacian_matvec(v):
        """Apply Laplacian to vector v."""
        result = np.zeros_like(v)

        for d in range(dim):
            stride = strides[d]
            m_idx = multi_idx[:, d]

            # Forward difference contribution
            if boundary == 'periodic':
                neighbor_idx = (m_idx + 1) % N
                lin_plus = np.arange(total_points) - m_idx * stride + neighbor_idx * stride
                valid_plus = np.ones(total_points, dtype=bool)
            else:
                lin_plus = np.arange(total_points) + stride
                valid_plus = m_idx < (N - 1)

            if np.any(valid_plus):
                idx_h = np.arange(total_points)[valid_plus]
                idx_p = lin_plus[valid_plus]

                A_half = 0.5 * (A[idx_h, d] + A[idx_p, d])
                coeff = A_half / (sqrt_g[idx_h] * h_sq[d])

                result[idx_h] += coeff * (v[idx_p] - v[idx_h])

            # Backward difference contribution
            if boundary == 'periodic':
                neighbor_idx = (m_idx - 1) % N
                lin_minus = np.arange(total_points) - m_idx * stride + neighbor_idx * stride
                valid_minus = np.ones(total_points, dtype=bool)
            else:
                lin_minus = np.arange(total_points) - stride
                valid_minus = m_idx > 0

            if np.any(valid_minus):
                idx_h = np.arange(total_points)[valid_minus]
                idx_m = lin_minus[valid_minus]

                A_half = 0.5 * (A[idx_h, d] + A[idx_m, d])
                coeff = A_half / (sqrt_g[idx_h] * h_sq[d])

                result[idx_h] += coeff * (v[idx_m] - v[idx_h])

        return result

    # Create LinearOperator
    L_op = LinearOperator(
        shape=(total_points, total_points),
        matvec=laplacian_matvec,
        dtype=np.float64
    )

    print(f"Computing {k} eigenvalues...")
    eigenvalues, eigenvectors = eigsh(
        L_op, k=k, sigma=0.0, which='LM', tol=tol
    )

    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors, points


# ============================================================================
# Example metric functions
# ============================================================================

def flat_metric(x: np.ndarray) -> np.ndarray:
    """Flat (Euclidean) metric: g_ii = 1."""
    return np.ones(7)


def g2_inspired_metric(x: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    G2-inspired diagonal metric with position-dependent components.

    This is a simplified diagonal approximation inspired by G2 holonomy manifolds.
    The metric components vary smoothly with position.
    """
    r = np.linalg.norm(x)

    # Smooth variation with radial distance
    # g_ii = 1 + scale * exp(-r²/2) * cos(2π x_i / L)
    # This creates a smoothly varying metric that's close to flat
    g_diag = np.ones(7)
    if r > 1e-10:
        for i in range(7):
            g_diag[i] = 1.0 + scale * np.exp(-r**2 / 2) * (1 + 0.1 * np.cos(x[i]))

    return g_diag


def joyce_orbifold_metric(x: np.ndarray, params: dict = None) -> np.ndarray:
    """
    Simplified diagonal metric inspired by Joyce orbifold construction.

    The Joyce construction for compact G2 manifolds uses T^7/Γ with
    singularity resolution. This provides a diagonal approximation.
    """
    if params is None:
        params = {'epsilon': 0.1, 'a': 1.0}

    eps = params['epsilon']
    a = params['a']

    # Base metric with small perturbation
    g_diag = np.ones(7)

    # Add smooth perturbation based on torus coordinates
    for i in range(7):
        # Periodic perturbation
        g_diag[i] = 1.0 + eps * np.sin(2 * np.pi * x[i] / a) ** 2

    return g_diag


def conformal_metric(x: np.ndarray, omega_func: Callable = None) -> np.ndarray:
    """
    Conformal metric: g_ij = Ω(x)² δ_ij

    Args:
        x: position
        omega_func: conformal factor function. Default: Ω = 1 + 0.1*r²
    """
    if omega_func is None:
        r_sq = np.sum(x**2)
        omega = 1.0 + 0.1 * r_sq
    else:
        omega = omega_func(x)

    return omega**2 * np.ones(7)


# ============================================================================
# Validation and testing
# ============================================================================

def validate_flat_space(N: int = 5) -> bool:
    """
    Validate against known flat-space eigenvalues.

    For flat space on [0, L]^7 with periodic BC, the eigenvalues are:
        λ = -4π² Σ_i (n_i / L_i)²
    where n_i are integers.
    """
    print("=" * 60)
    print("Validation: Flat 7-torus eigenvalues")
    print("=" * 60)

    L = np.ones(7)  # Unit domain

    L_mat, points = build_curved_laplacian_optimized(
        N, L, flat_metric, boundary='periodic'
    )

    k = min(15, N**7 - 2)
    eigenvalues, _ = compute_eigenvalues(L_mat, k=k)

    print(f"\nComputed eigenvalues (N={N}):")
    for i, ev in enumerate(eigenvalues):
        print(f"  λ_{i} = {ev:.6f}")

    # Theoretical eigenvalues for flat torus
    # λ = -4π² Σ_i n_i² where n_i ∈ Z
    print("\nTheoretical lowest eigenvalues for unit 7-torus:")
    print("  λ_0 = 0 (constant mode)")
    print("  λ_1 = -4π² ≈ -39.478 (7-fold degenerate: n_i = ±1 for one i)")
    print("  λ_2 = -8π² ≈ -78.957 (21-fold degenerate: n_i = ±1 for two i's)")

    # Check zero mode
    zero_mode_error = abs(eigenvalues[0])
    print(f"\nZero mode error: {zero_mode_error:.2e}")

    return zero_mode_error < 1.0  # Coarse grid won't be very accurate


def test_curved_metric(N: int = 5) -> None:
    """Test with a non-trivial curved metric."""
    print("=" * 60)
    print("Test: G2-inspired curved metric")
    print("=" * 60)

    L = 2 * np.pi * np.ones(7)  # Domain [0, 2π]^7

    def metric(x):
        return g2_inspired_metric(x, scale=0.3)

    L_mat, points = build_curved_laplacian_optimized(
        N, L, metric, boundary='periodic'
    )

    k = min(10, N**7 - 2)
    eigenvalues, eigenvectors = compute_eigenvalues(L_mat, k=k)

    print(f"\nEigenvalues for G2-inspired metric (N={N}):")
    for i, ev in enumerate(eigenvalues):
        print(f"  λ_{i} = {ev:.6f}")

    # Compute spectral gap
    if len(eigenvalues) >= 2:
        gap = eigenvalues[0] - eigenvalues[1]
        print(f"\nSpectral gap (λ_0 - λ_1): {gap:.6f}")


def run_full_example():
    """Run a complete example with eigenvalue computation."""
    print("=" * 60)
    print("Complete Example: Curved Laplacian on 7D Domain")
    print("=" * 60)

    # Parameters
    N = 5  # 5^7 = 78,125 points
    L = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # Unit cube

    # Define a metric that varies smoothly
    def varying_metric(x):
        """Metric with smooth spatial variation."""
        center = 0.5 * L
        r = np.linalg.norm(x - center)

        # Metric components vary with distance from center
        # g_ii = 1 + 0.2 * exp(-2r²)
        factor = 1.0 + 0.2 * np.exp(-2 * r**2)
        return factor * np.ones(7)

    # Build the Laplacian
    print("\n1. Building sparse Laplacian matrix...")
    L_mat, points = build_curved_laplacian_optimized(
        N, L, varying_metric, boundary='periodic'
    )

    # Matrix properties
    print(f"\nMatrix properties:")
    print(f"  Shape: {L_mat.shape}")
    print(f"  Non-zeros: {L_mat.nnz:,}")
    print(f"  Storage (approx): {L_mat.data.nbytes / 1e6:.2f} MB")

    # Compute eigenvalues
    print("\n2. Computing eigenvalues...")
    k = 8
    eigenvalues, eigenvectors = compute_eigenvalues(L_mat, k=k)

    print(f"\nFirst {k} eigenvalues:")
    for i, ev in enumerate(eigenvalues):
        print(f"  λ_{i} = {ev:12.6f}")

    # Verify Laplacian is negative semi-definite
    print(f"\nMaximum eigenvalue: {np.max(eigenvalues):.6e}")
    print(f"(Should be ≤ 0 for negative semi-definite Laplacian)")

    return L_mat, points, eigenvalues, eigenvectors


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--validate":
        validate_flat_space(N=5)
    elif len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_curved_metric(N=5)
    else:
        run_full_example()
