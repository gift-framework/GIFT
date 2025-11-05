# G₂ Metric Learning via Physics-Informed Neural Networks

**TL;DR:** Neural network learns G₂ holonomy metrics on 7-manifolds through pure geometric constraints – **no training data required**.

## Quick Start

### Results at a Glance
```
✓ G₂ closure: ||dφ||², ||d*φ||² < 10⁻⁶
✓ Normalization: ||φ||² = 7.000001
✓ Volume: det(g) = 1.000004  
✓ Ricci-flat: ||Ric||² < 10⁻⁶ (after polish)
✓ Topology: b₃ = 77 (matches GIFT theory)
```

### Key Innovation

Traditional approach:
```
Need known G₂ manifold → Discretize → Solve PDEs → Get metric
```

Our approach:
```
Define loss functions → Train neural net → Get continuous metric
```

**No training data.** Network learns by minimizing:
- Ricci curvature: ||Ric(g)||²
- G₂ closure: ||dφ||² + ||d*φ||²  
- Normalization: (||φ||² - 7)²
- Volume: (det g - 1)²

All computed via automatic differentiation.

## Architecture

```
7D coords → Fourier features (64D) → MLP [256,256,128] → 7×7 metric tensor
```

**120k parameters** trained for 6000 epochs (~5 hours on A100).

## Files

- `Complete_G₂_Metric_Training_v0_1.ipynb` - Full training + analysis notebook
- `TECHNICAL_DOCUMENTATION.md` - Detailed mathematical/computational documentation
- `G2_final_model.pt` - Trained model weights (1.4MB)
- `G2_metric_samples.npz` - 100 sample metric tensors for validation
- `G2_metric_analysis.json` - Quantitative results summary

## Validation

**Torsion Classes:** dφ = 0, d*φ = 0 satisfied to 10⁻⁶  
**Ricci Curvature:** ||Ric||² < 10⁻⁶ across 2000 test points  
**Positivity:** min(eigenvalues) = 0.431 (strongly positive-definite)  
**Stability:** Robust across different random seeds  
**Topology:** b₃ = 77 consistent with K₇ twisted connected sum  

## Running the Code

### Option 1: View Results (No GPU needed)
```python
import torch
import numpy as np

# Load trained model
checkpoint = torch.load('G2_final_model.pt', map_location='cpu')

# Load sample metrics
data = np.load('G2_metric_samples.npz')
metrics = data['metric']      # (100, 7, 7)
coords = data['coordinates']  # (100, 7)
phi = data['phi']             # (100, 35)

# Verify properties
import json
with open('G2_metric_analysis.json') as f:
    results = json.load(f)
    print(results)
```

### Option 2: Full Training (GPU required)
Open `Complete_G₂_Metric_Training_v0_1.ipynb` in Google Colab or Jupyter with GPU.

Estimated runtime: 5-6 hours on A100, 10-12 hours on T4.

## Method Overview

**Step 1:** Initialize random neural network  
**Step 2:** Sample random points in ℝ⁷  
**Step 3:** Compute metric g = NN(x)  
**Step 4:** Compute geometric losses (Ricci, G₂ closure, etc.)  
**Step 5:** Backpropagate and update weights  
**Step 6:** Repeat for 6000 epochs with curriculum learning  

**Curriculum:** Start with Ricci-flatness → Gradually increase G₂ weight → Polish

## Comparison with Traditional Methods

| Method | Data Needed | Time | Flexibility |
|--------|-------------|------|-------------|
| Finite Elements | Mesh | Hours | Fixed discretization |
| Spectral Methods | Basis | Hours | Limited to specific geometries |
| **Neural Net (ours)** | **None** | **5 hours** | **Any query point** |

## Questions & Contact

**For mathematicians:**  
- See `TECHNICAL_DOCUMENTATION.md` for full loss function derivations
- All geometric quantities computed via automatic differentiation
- G₂ 3-form construction follows twisted connected sum ansatz

**For ML practitioners:**  
- Architecture: Standard MLP with Fourier features
- Loss: Custom physics-informed (no labels needed)
- Training: Curriculum learning with 6 phases

**For physicists:**  
- Target: K₇ Calabi-Yau with G₂ holonomy
- Validation: Betti numbers, Ricci-flatness, torsion classes
- Future: Yukawa couplings, matter curves (in progress)

## Citation

```
@misc{g2_neural_metric_2025,
  title={Physics-Informed Neural Networks for G₂ Holonomy Metrics},
  author={Brieuc},
  year={2025},
  note={Version 0.1}
}
```

## Limitations & Future Work

**Current limitations:**
- Betti number b₂ computation via discrete Laplacian (approximation)
- Yukawa couplings estimated (need full harmonic basis)
- Single manifold tested (K₇)

**Planned improvements:**
- Test on other G₂ manifolds
- Higher precision polish (L-BFGS to ~10⁻¹⁰)
- Full Yukawa computation with matter curves

## License

Research code – free to use with attribution.

---

**Last updated:** November 2025  
**Status:** Initial results, validation complete  
**Feedback welcome:** Especially from differential geometers!

