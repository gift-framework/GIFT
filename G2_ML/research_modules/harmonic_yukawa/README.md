# Harmonic-Yukawa Pipeline

Extract fermion mass predictions from trained G2 PINN via harmonic forms and Yukawa couplings.

## The Pipeline

```
    PINN           Metric          Harmonic         Yukawa           Masses
   phi(x)    -->   g(x)    -->   H^2, H^3    -->   Y_ijk    -->   m_f
  (trained)       (7x7)        (21, 77)        (21x21x77)       (GeV)
```

## Mathematical Foundation

### Stage 1: Metric Extraction
From the trained PINN, we have phi(x) -> g_{ij}(x) via the G2 structure equation.

### Stage 2: Harmonic Forms
Solve the Hodge eigenvalue problem on the metric:
- Delta omega = lambda omega
- Harmonic forms have lambda = 0
- b2(K7) = 21 harmonic 2-forms
- b3(K7) = 77 harmonic 3-forms

### Stage 3: Yukawa Tensor
Compute the triple product integral:
```
Y_ijk = integral_{K7} omega_i wedge omega_j wedge Phi_k
```
where omega_i, omega_j are 2-forms and Phi_k is a 3-form.

### Stage 4: Mass Spectrum
The Yukawa Gram matrix M = Y^T Y has eigenvalues related to fermion masses:
```
m_f = v * sqrt(lambda_f) / sqrt(2)
```
where v = 246 GeV is the Higgs VEV.

## GIFT Predictions (Proven)

| Quantity | Value | Origin |
|----------|-------|--------|
| N_gen | 3 | Topological constraint |
| m_tau/m_e | 3477 | Cohomology structure |
| m_s/m_d | 20 | b2 - 1 = 21 - 1 |
| Q_Koide | 2/3 | 1 - 1/N_gen |
| tau | 3472/891 | (496*21)/(27*99) |

## Quick Start

```python
import torch
from G2_ML.harmonic_yukawa import HarmonicYukawaPipeline

# Mock metric function (replace with trained PINN)
def mock_metric(x):
    batch = x.shape[0]
    return torch.eye(7).unsqueeze(0).expand(batch, 7, 7)

# Run pipeline
pipeline = HarmonicYukawaPipeline(mock_metric, device='cpu')
result = pipeline.run()

# Get predictions
print(result.mass_report)

# Save for Lean verification
result.save("./output")
```

## Module Structure

```
harmonic_yukawa/
├── __init__.py           # Module exports
├── config.py             # Configuration
├── hodge_laplacian.py    # Hodge Laplacian on PINN metric
├── harmonic_extraction.py # Extract H^2, H^3 forms
├── wedge_product.py      # Wedge product computations
├── yukawa.py             # Yukawa tensor Y_ijk
├── mass_spectrum.py      # Mass extraction & PDG comparison
├── pipeline.py           # End-to-end pipeline
└── README.md             # This file
```

## Output Files

After running `result.save(output_dir)`:

| File | Description |
|------|-------------|
| `yukawa_tensor.pt` | Full (21,21,77) Yukawa tensor |
| `eigenvalues.json` | Eigenvalue spectrum with statistics |
| `masses.json` | Extracted fermion masses |
| `bounds.json` | Numerical bounds for Lean verification |
| `report.txt` | Human-readable analysis report |

## Lean Integration

The `bounds.json` file contains numerical bounds that can be imported into Lean:

```lean
-- From bounds.json
def det_g_computed : Float := 2.03125
def tau_computed : Float := 3.896
-- etc.
```

Use `result.export_lean_bounds()` to generate Lean theorem statements.

## Dependencies

- torch >= 2.0
- numpy

## References

- Joyce (2000): "Compact Manifolds with Special Holonomy"
- Candelas et al.: "Yukawa couplings in heterotic compactifications"
- GIFT v2.2: `publications/markdown/gift_2_2_main.md`
