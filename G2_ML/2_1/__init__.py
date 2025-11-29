"""GIFT v2.2 Variational G2 Metric Extraction.

This module implements a Physics-Informed Neural Network (PINN) to solve
a constrained variational problem on G2 geometry.

Mathematical Formulation
------------------------
Find phi in Lambda^3_+(R^7) minimizing:
    F[phi] = ||d phi||^2 + ||d* phi||^2

Subject to GIFT v2.2 constraints:
    - det(g) = 65/32 (metric determinant from h* = 99)
    - kappa_T = 1/61 (global torsion magnitude)
    - phi positive (phi in G2 cone)
    - (b2, b3) = (21, 77) (cohomology from E8 decomposition)

Key Insight
-----------
The constraints are PRIMARY (inputs from GIFT theory).
The metric is EMERGENT (output from variational solution).

We do NOT assume TCS or Joyce construction - the geometry emerges
from satisfying the constraint system.

Usage
-----
    from G2_ML.2_1 import GIFTConfig, G2VariationalNet, train_gift_g2

    # Quick training
    model, history = train_gift_g2(device='cuda')

    # Custom configuration
    config = GIFTConfig(total_epochs=5000)
    model = G2VariationalNet(config)
    # ... custom training loop

Modules
-------
- config: GIFT v2.2 structural parameters (topologically determined)
- g2_geometry: G2 structure operations (metric from phi, positivity)
- model: Neural network architectures
- constraints: Constraint enforcement functions
- loss: Variational loss composition
- training: Phased training protocol
- validation: Metric computation and verification

References
----------
- Joyce (2000): "Compact Manifolds with Special Holonomy"
- GIFT v2.2: publications/gift_2_2_main.md
"""
from .config import GIFTConfig, default_config, TrainingState
from .g2_geometry import (
    MetricFromPhi,
    G2Positivity,
    TorsionComputation,
    standard_phi_coefficients,
    random_phi_near_standard,
    normalize_phi,
)
from .model import (
    G2VariationalNet,
    HarmonicFormsNet,
    GIFTVariationalModel,
    FourierFeatures,
)
from .constraints import (
    DeterminantConstraint,
    TorsionConstraint,
    PositivityConstraint,
    CohomologyConstraint,
    GIFTConstraints,
)
from .loss import (
    TorsionFunctional,
    VariationalLoss,
    PhasedLossManager,
    format_loss_dict,
    log_constraints,
)
from .training import (
    Trainer,
    TrainingHistory,
    train_gift_g2,
    sample_coordinates,
    sample_grid,
)
from .validation import (
    Validator,
    ValidationResult,
    FullValidationReport,
    CohomologyValidator,
    StabilityAnalyzer,
    generate_validation_report,
)

__version__ = "2.1.0"
__all__ = [
    # Config
    "GIFTConfig",
    "default_config",
    "TrainingState",
    # Geometry
    "MetricFromPhi",
    "G2Positivity",
    "TorsionComputation",
    "standard_phi_coefficients",
    "random_phi_near_standard",
    "normalize_phi",
    # Model
    "G2VariationalNet",
    "HarmonicFormsNet",
    "GIFTVariationalModel",
    "FourierFeatures",
    # Constraints
    "DeterminantConstraint",
    "TorsionConstraint",
    "PositivityConstraint",
    "CohomologyConstraint",
    "GIFTConstraints",
    # Loss
    "TorsionFunctional",
    "VariationalLoss",
    "PhasedLossManager",
    "format_loss_dict",
    "log_constraints",
    # Training
    "Trainer",
    "TrainingHistory",
    "train_gift_g2",
    "sample_coordinates",
    "sample_grid",
    # Validation
    "Validator",
    "ValidationResult",
    "FullValidationReport",
    "CohomologyValidator",
    "StabilityAnalyzer",
    "generate_validation_report",
]
