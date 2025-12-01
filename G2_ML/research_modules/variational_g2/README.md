# GIFT v2.2 Variational G2 Metric Extraction

Physics-Informed Neural Network for solving the constrained G2 variational problem.

## Overview

This module implements numerical resolution of a minimization problem whose solution, if it exists, defines the K7 geometry consistent with GIFT v2.2 constraints. This is **NOT** a simulation of a pre-existing manifold - it is the numerical resolution of a variational problem.

## Mathematical Formulation

### The Variational Problem

Find φ ∈ Λ³₊(ℝ⁷) minimizing:

```
F[φ] = ||dφ||²_{L²} + ||d*φ||²_{L²}
```

subject to GIFT v2.2 constraints:

| Constraint | Type | Value | Origin |
|------------|------|-------|--------|
| b₂ | Topological | 21 | E₈ decomposition |
| b₃ | Topological | 77 = 35 + 42 | Cohomology split |
| det(g) | Metric | 65/32 | Derived from h* = 99 |
| κ_T | Torsion | 1/61 | Global torsion magnitude |
| φ positive | Geometric | φ ∈ G₂ cone | Valid G₂ structure |

### Key Insight

The constraints are **primary** (inputs), the metric is **emergent** (output).
We do NOT assume TCS or Joyce construction - we let the geometry emerge from the constraints.

## Directory Structure

```
variational_g2/
├── README.md                 # This file
├── src/
│   ├── __init__.py          # Package initialization
│   ├── constraints.py       # Constraint functions (det, torsion, positivity)
│   ├── model.py             # G2VariationalNet neural network
│   ├── loss.py              # Variational loss composition
│   ├── harmonic.py          # Cohomology extraction (Betti numbers)
│   ├── training.py          # Phased training protocol
│   └── validation.py        # Metric computation and validation
├── config/
│   └── gift_v22.yaml        # GIFT v2.2 parameters and training config
├── notebooks/
│   └── analysis.ipynb       # Post-training analysis
└── outputs/
    ├── checkpoints/         # Model weights during training
    ├── metrics/             # Validation results
    └── artifacts/           # φ(x), g(x), harmonic bases
```

## Installation

```bash
# From the GIFT repository root
pip install -r requirements.txt

# Navigate to this directory
cd G2_ML/variational_g2
```

## Usage

### Training

```python
from src.training import train_from_config

# Train from configuration file
results = train_from_config('config/gift_v22.yaml', output_dir='outputs')
```

Or via command line:

```bash
python -m src.training --config config/gift_v22.yaml --output outputs
```

### Validation

```python
from src.validation import validate_model
from src.model import create_model
import torch

# Load trained model
model = create_model(config)
model.load_state_dict(torch.load('outputs/checkpoints/final_model.pt')['model_state_dict'])

# Validate
results = validate_model(model, output_path='outputs/metrics/validation.json')
print(results.summary)
```

### Using Individual Components

```python
import torch
from src.model import G2VariationalNet
from src.constraints import metric_from_phi, det_constraint_loss
from src.loss import VariationalLoss, LossWeights

# Create model
model = G2VariationalNet(
    hidden_dims=[256, 512, 512, 256],
    num_frequencies=64,
)

# Sample point
x = torch.randn(1, 7)

# Get 3-form and metric
output = model(x, return_full=True, return_metric=True)
phi = output['phi_full']      # Shape: (1, 7, 7, 7)
metric = output['metric']     # Shape: (1, 7, 7)

# Compute determinant
det_g = torch.det(metric)
print(f"det(g) = {det_g.item():.6f}, target = {65/32:.6f}")
```

## Training Protocol

Training proceeds in four phases with different loss weight configurations:

### Phase 1: Initialization (2000 epochs)
- **Focus**: Establish valid G₂ structure
- **Weights**: positivity=2.0, torsion=1.0, det=0.5
- Ensures φ is in the G₂ cone (positive definite metric)

### Phase 2: Constraint Satisfaction (3000 epochs)
- **Focus**: Achieve det(g) = 65/32
- **Weights**: det=2.0, torsion=1.0, positivity=1.0
- Drives metric determinant toward target value

### Phase 3: Torsion Targeting (3000 epochs)
- **Focus**: Achieve κ_T = 1/61
- **Weights**: torsion=3.0, det=1.0, positivity=1.0
- Fine-tunes torsion magnitude to GIFT target

### Phase 4: Cohomology Refinement (2000 epochs)
- **Focus**: Refine (b₂, b₃) = (21, 77)
- **Weights**: cohomology=2.0, torsion=2.0, det=1.0
- Adjusts harmonic content of the geometry

## Validation Metrics

After training, the following metrics are computed:

| Metric | Target | Tolerance | How to compute |
|--------|--------|-----------|----------------|
| det(g) | 65/32 = 2.03125 | ±0.1% | Mean over grid |
| κ_T | 1/61 ≈ 0.01639 | ±5% | ||dφ|| + ||d*φ|| |
| b₂_eff | 21 | Exact | Rank of H² projection |
| b₃_eff | 77 | Exact | Rank of H³ projection |
| ||φ||²_g | 7 | ±1% | G₂ identity |
| g positive | Yes | Binary | All eigenvalues > 0 |

## Output Artifacts

The trained model produces:

1. **φ(x)**: The 3-form as a function (neural network weights)
2. **g(x)**: The induced metric (derived from φ)
3. **Harmonic bases**: {ω_α} for H², {Ω_k} for H³
4. **Validation report**: All metrics with uncertainties
5. **Training history**: Loss curves and metric evolution

## Mathematical Framing

The output documentation states:

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

## Key Differences from Previous Approaches

| Aspect | Old (TCS-based) | New (Variational) |
|--------|-----------------|-------------------|
| Starting point | "Construct TCS manifold" | "Define constraint system" |
| Role of (21,77) | "Verify these emerge" | "Impose as constraints" |
| Role of PINN | "Approximate known geometry" | "Solve optimization problem" |
| Success criterion | "Match TCS structure" | "Satisfy constraints + minimize F" |
| Academic claim | "We built a TCS metric" | "We found variational solution" |

## References

- Joyce (2000): Deformation theory for G₂ structures
- GIFT v2.2 main paper: Constraint derivations
- Bryant (1987): Metrics with exceptional holonomy

## Citation

If using this code, please cite the GIFT framework:

```
GIFT Framework v2.2
https://github.com/gift-framework/GIFT
```

## License

MIT License - See repository root for details.
