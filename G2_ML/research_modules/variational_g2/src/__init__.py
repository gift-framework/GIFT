"""
GIFT v2.2 Variational G2 Metric Extraction

Physics-Informed Neural Network for solving the constrained G2 variational problem.
This module implements numerical resolution of a minimization problem whose solution,
if it exists, defines the K7 geometry consistent with GIFT v2.2 constraints.

Mathematical Formulation:
    Find phi in Lambda^3_+(R^7) minimizing:
        F[phi] = ||d*phi||^2_{L^2} + ||d*phi||^2_{L^2}

    Subject to GIFT v2.2 constraints:
        - b_2 = 21 (topological, from E8 decomposition)
        - b_3 = 77 (topological, cohomology split)
        - det(g) = 65/32 (metric constraint)
        - kappa_T = 1/61 (torsion magnitude)
        - phi positive (in G2 cone)

Modules:
    - constraints: Constraint functions (det, torsion, positivity, cohomology)
    - model: G2VariationalNet neural network architecture
    - loss: Variational loss composition
    - harmonic: Cohomology extraction via Hodge decomposition
    - training: Phased training protocol
    - validation: Metric computation and validation
"""

__version__ = "1.0.0"
__author__ = "GIFT Framework Team"

from .constraints import (
    metric_from_phi,
    det_constraint_loss,
    torsion_loss,
    g2_positivity_check,
    exterior_derivative,
    codifferential,
)
from .model import G2VariationalNet, FourierFeatures
from .loss import VariationalLoss, LossWeights
from .harmonic import extract_betti_numbers, hodge_decomposition
from .training import Trainer, TrainingConfig
from .validation import Validator, ValidationMetrics

__all__ = [
    # Constraints
    "metric_from_phi",
    "det_constraint_loss",
    "torsion_loss",
    "g2_positivity_check",
    "exterior_derivative",
    "codifferential",
    # Model
    "G2VariationalNet",
    "FourierFeatures",
    # Loss
    "VariationalLoss",
    "LossWeights",
    # Harmonic
    "extract_betti_numbers",
    "hodge_decomposition",
    # Training
    "Trainer",
    "TrainingConfig",
    # Validation
    "Validator",
    "ValidationMetrics",
]
