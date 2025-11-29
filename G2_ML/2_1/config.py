"""Configuration for GIFT v2.2 Variational G2 Metric Extraction.

This module defines all structural parameters from GIFT v2.2.
These are NOT fitted parameters - they are topologically determined.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import math


@dataclass
class GIFTConfig:
    """GIFT v2.2 structural parameters.

    All values derive from:
    - E8 x E8 gauge group (dimension 496)
    - K7 manifold with G2 holonomy

    No continuous adjustable parameters.
    """

    # ==========================================================================
    # Fundamental dimensions
    # ==========================================================================
    dim: int = 7                    # Manifold dimension
    dim_E8: int = 248               # E8 Lie algebra dimension
    dim_E8xE8: int = 496            # Total gauge dimension

    # ==========================================================================
    # Cohomological structure (TOPOLOGICAL - fixed by manifold choice)
    # ==========================================================================
    b2_K7: int = 21                 # Second Betti number: harmonic 2-forms
    b3_K7: int = 77                 # Third Betti number: harmonic 3-forms
    b3_local: int = 35              # Local 3-forms from phi (C(7,3))
    b3_global: int = 42             # Global 3-forms from topology (77-35)
    h_star: int = 99                # Total cohomology: b2 + b3 + 1

    # ==========================================================================
    # G2 structure constants
    # ==========================================================================
    dim_G2: int = 14                # G2 Lie algebra dimension
    rank_E8: int = 8                # E8 rank

    # ==========================================================================
    # GIFT v2.2 derived constants (TOPOLOGICAL/PROVEN)
    # ==========================================================================

    # Binary duality: dim(G2)/dim(K7)
    p2: int = 2                     # = 14/7

    # Weinberg angle: b2 / (b3 + dim_G2) = 21 / (77 + 14) = 21/91 = 3/13
    sin2_theta_W: float = 3/13      # PROVEN

    # Hierarchy parameter: (496 * 21) / (27 * 99) = 10416/2673 = 3472/891
    tau: float = 3472/891           # PROVEN (exact rational)

    # Metric determinant: derived from h* = 99
    det_g_target: float = 65/32     # = 2.03125 (TOPOLOGICAL)

    # Global torsion magnitude: 1/(b3 - dim_G2 - p2) = 1/(77-14-2) = 1/61
    kappa_T: float = 1/61           # TOPOLOGICAL

    # Angular quantization
    beta_0: float = math.pi / 8     # = pi / rank(E8)

    # Correlation parameter: (Weyl_factor / p2) * beta_0
    Weyl_factor: int = 5            # Pentagonal symmetry
    xi: float = 5 * math.pi / 16    # = (5/2) * (pi/8)

    # ==========================================================================
    # Network architecture
    # ==========================================================================
    hidden_dim: int = 256           # Hidden layer dimension
    n_layers: int = 6               # Number of hidden layers
    fourier_features: int = 64      # Number of Fourier features
    fourier_scale: float = 2.0      # Fourier feature scale

    # ==========================================================================
    # Training hyperparameters
    # ==========================================================================
    total_epochs: int = 10000
    batch_size: int = 2048
    grid_resolution: int = 16       # Points per dimension for validation
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    # ==========================================================================
    # Constraint tolerances
    # ==========================================================================
    det_tolerance: float = 0.001    # |det(g) - 65/32| < 0.1%
    kappa_tolerance: float = 0.05   # |kappa_T - 1/61| < 5%
    positivity_eps: float = 1e-6    # Minimum eigenvalue threshold

    # ==========================================================================
    # Phase configuration
    # ==========================================================================
    phases: List[Dict] = field(default_factory=lambda: [
        {
            "name": "initialization",
            "epochs": 2000,
            "focus": "establish G2 structure",
            "weights": {"torsion": 1.0, "det": 0.5, "positivity": 2.0, "cohomology": 0.1}
        },
        {
            "name": "constraint_satisfaction",
            "epochs": 3000,
            "focus": "satisfy det(g) = 65/32",
            "weights": {"torsion": 1.0, "det": 2.0, "positivity": 1.0, "cohomology": 0.5}
        },
        {
            "name": "torsion_targeting",
            "epochs": 3000,
            "focus": "achieve kappa_T = 1/61",
            "weights": {"torsion": 3.0, "det": 1.0, "positivity": 1.0, "cohomology": 1.0}
        },
        {
            "name": "cohomology_refinement",
            "epochs": 2000,
            "focus": "refine (b2, b3) = (21, 77)",
            "weights": {"torsion": 2.0, "det": 1.0, "positivity": 0.5, "cohomology": 2.0}
        }
    ])

    def __post_init__(self):
        """Validate configuration values."""
        # Check topological constraints
        assert self.b2_K7 == 21, "b2(K7) must be 21"
        assert self.b3_K7 == 77, "b3(K7) must be 77"
        assert self.b3_local + self.b3_global == self.b3_K7
        assert self.h_star == self.b2_K7 + self.b3_K7 + 1

        # Check derived values
        assert abs(self.sin2_theta_W - 3/13) < 1e-10
        assert abs(self.tau - 3472/891) < 1e-10
        assert abs(self.det_g_target - 65/32) < 1e-10
        assert abs(self.kappa_T - 1/61) < 1e-10

    @property
    def n_phi_components(self) -> int:
        """Number of independent components in 3-form phi."""
        return 35  # C(7,3)

    @property
    def n_2form_components(self) -> int:
        """Number of independent components in a 2-form."""
        return 21  # C(7,2)

    @property
    def total_samples_per_epoch(self) -> int:
        """Total grid points for training."""
        return self.grid_resolution ** self.dim


@dataclass
class TrainingState:
    """Mutable training state."""
    epoch: int = 0
    phase_idx: int = 0
    best_loss: float = float('inf')
    det_history: List[float] = field(default_factory=list)
    kappa_history: List[float] = field(default_factory=list)
    loss_history: List[float] = field(default_factory=list)


# Default configuration instance
default_config = GIFTConfig()
