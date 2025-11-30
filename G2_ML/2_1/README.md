# GIFT v2.2 - Variational G2 Metric Extraction (v2.1)

## Overview

This module implements a **Physics-Informed Neural Network (PINN)** to solve a **constrained variational problem** on G2 geometry for GIFT v2.2.

**Key insight**: This is NOT a simulation of a pre-existing manifold. It is the numerical resolution of a minimization problem whose solution, if it exists, defines the geometry.

## Mathematical Formulation

### The Variational Problem

Find φ ∈ Λ³₊(ℝ⁷) minimizing:
```
F[φ] = ||dφ||²_{L²} + ||d*φ||²_{L²}
```

Subject to GIFT v2.2 constraints:

| Constraint | Type | Value | Origin |
|------------|------|-------|--------|
| b₂ | Topological | 21 | E₈ decomposition |
| b₃ | Topological | 77 = 35 + 42 | Cohomology split |
| det(g) | Metric | 65/32 | Derived from h* = 99 |
| κ_T | Torsion | 1/61 | Global torsion magnitude |
| φ positive | Geometric | φ ∈ G₂ cone | Valid G₂ structure |

### Key Insight

The constraints are **PRIMARY** (inputs from GIFT theory).
The metric is **EMERGENT** (output from variational solution).

We do NOT assume TCS or Joyce construction - we let the geometry emerge from the constraints.

## Difference from Previous Approach

| Aspect | Old (TCS-based) | New (Variational) |
|--------|-----------------|-------------------|
| Starting point | "Construct TCS manifold" | "Define constraint system" |
| Role of (21,77) | "Verify these emerge" | "Impose as constraints" |
| Role of PINN | "Approximate known geometry" | "Solve optimization problem" |
| Success criterion | "Match TCS structure" | "Satisfy constraints + minimize F" |
| Academic claim | "We built a TCS metric" | "We found variational solution" |

## Directory Structure

```
2_1/
├── __init__.py                    # Module exports
├── config.py                      # GIFT v2.2 parameters (topological)
├── g2_geometry.py                 # G₂ structure operations
├── model.py                       # Neural network architectures
├── constraints.py                 # Constraint enforcement
├── loss.py                        # Variational loss composition
├── training.py                    # Phased training protocol
├── validation.py                  # Metric computation
├── GIFT_Variational_G2_v2_1.ipynb # Main notebook
└── README.md                      # This file
```

## Quick Start

```python
from G2_ML.2_1 import train_gift_g2, GIFTConfig

# Quick training with defaults
model, history = train_gift_g2(device='cuda')

# Custom configuration
config = GIFTConfig(total_epochs=5000)
model, history = train_gift_g2(config=config, device='cuda')
```

## Training Protocol

Training proceeds in 4 phases:

| Phase | Epochs | Focus | Loss Weights |
|-------|--------|-------|--------------|
| 1. Initialization | 2000 | Establish G₂ structure | positivity high |
| 2. Constraint | 3000 | det(g) = 65/32 | det high |
| 3. Torsion | 3000 | κ_T = 1/61 | torsion high |
| 4. Refinement | 2000 | (b₂, b₃) = (21, 77) | cohomology high |

## Validation Metrics

After training, the following are computed:

| Metric | Target | Tolerance |
|--------|--------|-----------|
| det(g) | 65/32 = 2.03125 | ±0.1% |
| κ_T | 1/61 ≈ 0.01639 | ±5% |
| b₂_eff | 21 | Exact |
| b₃_eff | 77 | Exact |
| g positive | All eigenvalues > 0 | 99.9% |

## Mathematical Framing

The output should be understood as:

> **Theorem (Conditional Existence)**
>
> Let P be the variational problem: minimize ||dφ||² + ||d*φ||² over G₂ 3-forms
> subject to det(g(φ)) = 65/32 and κ_T = 1/61.
>
> The PINN produces φ_num satisfying:
> - F[φ_num] < ε (torsion bound)
> - |det(g) - 65/32| < δ₁
> - |κ_T - 1/61| < δ₂
>
> If ε is sufficiently small, by G₂ deformation theory (Joyce), there exists
> an exact torsion-free G₂ structure φ_exact with ||φ_exact - φ_num|| = O(ε).

This positions the work as:
- **Numerical evidence** for existence of a specific G₂ geometry
- **Not** a claim to have constructed the manifold rigorously
- **Invitation** to mathematicians: prove this geometry exists

## Module Reference

### config.py
- `GIFTConfig`: All structural parameters from GIFT v2.2
- `TrainingState`: Mutable training state

### g2_geometry.py
- `MetricFromPhi`: Extract metric g from 3-form φ
- `G2Positivity`: Check/enforce φ ∈ Λ³₊
- `TorsionComputation`: Compute torsion magnitude

### model.py
- `G2VariationalNet`: Main PINN architecture
- `HarmonicFormsNet`: Learn H² and H³ harmonic forms
- `FourierFeatures`: Smooth positional encoding

### constraints.py
- `DeterminantConstraint`: det(g) = 65/32
- `TorsionConstraint`: κ_T = 1/61
- `PositivityConstraint`: φ ∈ G₂ cone
- `CohomologyConstraint`: (b₂, b₃) = (21, 77)

### loss.py
- `TorsionFunctional`: F[φ] = ||dφ||² + ||d*φ||²
- `VariationalLoss`: Combined weighted loss
- `PhasedLossManager`: Phase-aware weight scheduling

### training.py
- `Trainer`: Main training loop
- `train_gift_g2`: Quick training function
- `sample_coordinates`: Random/grid sampling

### validation.py
- `Validator`: Full validation suite
- `CohomologyValidator`: Betti number verification
- `StabilityAnalyzer`: Perturbation stability

## References

- Joyce (2000): "Compact Manifolds with Special Holonomy"
- GIFT v2.2: `publications/markdown/gift_2_2_main.md`
- K7 Deformation Atlas: `G2_ML/meta_hodge/K7_DEFORMATION_ATLAS.md`

## Version History

- **v2.1.0** (2025-11-29): Initial variational formulation
  - Constraints as primary inputs
  - Phased training protocol
  - Full validation suite
