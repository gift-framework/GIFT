"""Yukawa tensor computation from harmonic forms.

The Yukawa couplings arise from the topological triple product:
    Y_ijk = integral_{K7} omega_i wedge omega_j wedge Phi_k

where:
- omega_i, omega_j in H^2(K7) are harmonic 2-forms (21 each)
- Phi_k in H^3(K7) is a harmonic 3-form (77 total)

The resulting tensor Y has shape (21, 21, 77) and encodes:
- Fermion mass matrices (via eigenvalues)
- Mixing angles (via eigenvectors)
- CP violation phases (via complex structure)

Reference: Candelas et al., "Yukawa couplings in heterotic compactifications"
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import math

import torch
import torch.nn as nn

from .config import HarmonicConfig, default_harmonic_config
from .harmonic_extraction import HarmonicBasis
from .wedge_product import WedgeProduct, wedge_2_2_3


@dataclass
class YukawaResult:
    """Container for Yukawa tensor computation results."""
    tensor: torch.Tensor            # (21, 21, 77) Yukawa couplings
    gram_matrix: torch.Tensor       # (77, 77) = Y^T Y (for mass eigenvalues)
    eigenvalues: torch.Tensor       # (77,) eigenvalues of Gram matrix
    eigenvectors: torch.Tensor      # (77, 77) eigenvectors
    trace: float                    # Tr(Y^T Y)
    det: float                      # det(Y^T Y)^{1/77}

    @property
    def effective_rank(self) -> int:
        """Number of significant eigenvalues (> 0.01 * max)."""
        threshold = 0.01 * self.eigenvalues.max()
        return int((self.eigenvalues > threshold).sum().item())

    @property
    def hierarchy_ratio(self) -> float:
        """Ratio of largest to smallest nonzero eigenvalue."""
        nonzero = self.eigenvalues[self.eigenvalues > 1e-10]
        if len(nonzero) < 2:
            return float('inf')
        return (nonzero.max() / nonzero.min()).item()

    def mass_spectrum(self, scale: float = 1.0) -> torch.Tensor:
        """Convert eigenvalues to mass spectrum.

        The Yukawa eigenvalues squared give mass eigenvalues.
        Actual masses = sqrt(eigenvalues) * scale.

        Args:
            scale: Overall mass scale (e.g., v/sqrt(2) for SM)

        Returns:
            masses: (77,) mass values
        """
        return scale * torch.sqrt(self.eigenvalues.clamp(min=0))

    def mixing_matrix(self) -> torch.Tensor:
        """Extract mixing matrix from eigenvectors.

        For fermion mixing (CKM, PMNS), the mixing matrix
        relates mass eigenstates to flavor eigenstates.

        Returns:
            V: (77, 77) unitary mixing matrix
        """
        return self.eigenvectors


class YukawaTensor:
    """Compute Yukawa tensor from harmonic basis.

    The Yukawa tensor Y_ijk encodes triple couplings between
    harmonic forms. It is computed via Monte Carlo integration:

        Y_ijk = (1/V) sum_{x in samples} omega_i(x) wedge omega_j(x) wedge Phi_k(x) * vol(x)

    where V = sum vol(x) is the total volume.
    """

    def __init__(self, config: HarmonicConfig = None, device: str = 'cpu'):
        self.config = config or default_harmonic_config
        self.device = device
        self.wedge = WedgeProduct()

    def compute(self, basis: HarmonicBasis) -> YukawaResult:
        """Compute full Yukawa tensor from harmonic basis.

        Args:
            basis: HarmonicBasis with H^2 and H^3 forms

        Returns:
            YukawaResult with tensor and derived quantities
        """
        n_points = basis.sample_points.shape[0]

        # Volume weights
        det_g = torch.det(basis.metric_at_points)
        vol = torch.sqrt(det_g.abs().clamp(min=self.config.eps))
        total_vol = vol.sum()

        # Initialize Yukawa tensor
        Y = torch.zeros(self.config.b2, self.config.b2, self.config.b3, device=self.device)

        print(f"Computing Yukawa tensor ({self.config.b2}x{self.config.b2}x{self.config.b3})...")

        # Compute in batches for memory efficiency
        batch_size = self.config.yukawa_batch_size

        for i in range(self.config.b2):
            for j in range(self.config.b2):
                # Get forms i and j at all points
                omega_i = basis.h2_forms[:, i, :]  # (n_points, 21)
                omega_j = basis.h2_forms[:, j, :]  # (n_points, 21)

                # Compute 2-wedge-2 once
                eta = self.wedge.wedge_2_2(omega_i, omega_j)  # (n_points, 35)

                for k in range(self.config.b3):
                    Phi_k = basis.h3_forms[:, k, :]  # (n_points, 35)

                    # 4-wedge-3 gives scalar
                    integrand = self.wedge.wedge_4_3(eta, Phi_k)  # (n_points,)

                    # Integrate with volume weights
                    Y[i, j, k] = (integrand * vol).sum() / total_vol

            if (i + 1) % 5 == 0:
                print(f"  Completed row {i + 1}/{self.config.b2}")

        # Compute Gram matrix M = Y^T Y (contracting over i,j indices)
        # M_kl = sum_{i,j} Y_ijk Y_ijl
        Y_flat = Y.reshape(-1, self.config.b3)  # (21*21, 77)
        gram_matrix = Y_flat.T @ Y_flat  # (77, 77)

        # Eigendecomposition of Gram matrix
        eigenvalues, eigenvectors = torch.linalg.eigh(gram_matrix)

        # Sort by eigenvalue (descending)
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Compute statistics
        trace = eigenvalues.sum().item()
        det_val = eigenvalues.prod().item()
        det_normalized = det_val ** (1.0 / self.config.b3) if det_val > 0 else 0.0

        return YukawaResult(
            tensor=Y,
            gram_matrix=gram_matrix,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            trace=trace,
            det=det_normalized
        )

    def compute_symmetric(self, basis: HarmonicBasis) -> YukawaResult:
        """Compute Yukawa tensor with symmetry Y_ijk = Y_jik enforced.

        The physical Yukawa tensor has this symmetry by construction
        (wedge product of 2-forms is symmetric up to sign).

        Args:
            basis: HarmonicBasis

        Returns:
            YukawaResult with symmetrized tensor
        """
        result = self.compute(basis)

        # Symmetrize: Y_ijk -> (Y_ijk + Y_jik) / 2
        Y_sym = (result.tensor + result.tensor.transpose(0, 1)) / 2

        # Recompute Gram matrix
        Y_flat = Y_sym.reshape(-1, self.config.b3)
        gram_matrix = Y_flat.T @ Y_flat

        eigenvalues, eigenvectors = torch.linalg.eigh(gram_matrix)
        idx = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        return YukawaResult(
            tensor=Y_sym,
            gram_matrix=gram_matrix,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            trace=eigenvalues.sum().item(),
            det=eigenvalues.prod().item() ** (1.0 / self.config.b3) if eigenvalues.prod().item() > 0 else 0.0
        )

    def verify_antisymmetry(self, basis: HarmonicBasis) -> dict:
        """Verify expected antisymmetry properties.

        The wedge product omega_i ^ omega_j should be antisymmetric
        under i <-> j for 2-forms.

        Returns:
            Dictionary of symmetry violation metrics
        """
        Y = self.compute(basis).tensor

        # Check Y_ijk vs Y_jik
        diff = Y - Y.transpose(0, 1)
        antisym_violation = diff.abs().max().item()

        # For identical forms, should be zero
        diag_nonzero = torch.tensor([Y[i, i, :].abs().max() for i in range(self.config.b2)])

        return {
            "antisymmetry_violation": antisym_violation,
            "diagonal_nonzero_max": diag_nonzero.max().item(),
            "diagonal_nonzero_mean": diag_nonzero.mean().item(),
        }
