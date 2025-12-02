"""Configuration for Harmonic Forms and Yukawa Pipeline.

All constants derive from GIFT v2.2 topological structure.
"""
from dataclasses import dataclass, field
from typing import List
import math


@dataclass
class HarmonicConfig:
    """Configuration for harmonic extraction and Yukawa computation.

    Topological constants from GIFT v2.2:
    - b2(K7) = 21: Harmonic 2-forms (gauge moduli)
    - b3(K7) = 77: Harmonic 3-forms (35 local + 42 global)
    - det(g) = 65/32: Metric determinant target
    - kappa_T = 1/61: Torsion magnitude
    """

    # Cohomological dimensions (TOPOLOGICAL)
    b2: int = 21                    # dim H^2(K7)
    b3: int = 77                    # dim H^3(K7)
    b3_local: int = 35              # dim Lambda^3(R^7)
    b3_global: int = 42             # TCS modes: b3 - b3_local

    # Component dimensions
    dim_2form: int = 21             # C(7,2) components per 2-form
    dim_3form: int = 35             # C(7,3) components per 3-form
    dim_K7: int = 7                 # Manifold dimension

    # Metric constraints
    det_g_target: float = 65/32     # = 2.03125 (TOPOLOGICAL)
    kappa_T: float = 1/61           # Torsion magnitude

    # Harmonic extraction parameters
    n_sample_points: int = 10000    # Monte Carlo integration points
    n_eigenvalues: int = 100        # Number of eigenvalues to compute
    harmonic_threshold: float = 1e-4  # Eigenvalue threshold for "zero"

    # Yukawa integration
    n_yukawa_samples: int = 50000   # Points for Yukawa integration
    yukawa_batch_size: int = 1000   # Batch size for memory efficiency

    # Training for learned harmonics
    harmonic_epochs: int = 5000
    harmonic_lr: float = 1e-4
    orthonormality_weight: float = 10.0
    closedness_weight: float = 5.0
    coclosedness_weight: float = 5.0

    # Numerical precision
    eps: float = 1e-10

    @property
    def total_harmonic_dim(self) -> int:
        """Total dimension of harmonic cohomology: b2 + b3."""
        return self.b2 + self.b3

    @property
    def h_star(self) -> int:
        """Effective cohomological dimension: b2 + b3 + 1 = 99."""
        return self.b2 + self.b3 + 1


default_harmonic_config = HarmonicConfig()
