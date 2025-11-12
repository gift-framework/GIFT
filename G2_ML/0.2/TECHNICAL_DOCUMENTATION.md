# G₂ Metric Learner v0.2 - Technical Documentation

## Overview

Version 0.2 represents a fundamental redesign of the G₂ metric learning approach. Instead of learning metric tensors directly, this version learns the G₂ 3-form φ(x) and reconstructs the metric algebraically, while enforcing true torsion-free conditions through proper exterior derivatives.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Module Documentation](#module-documentation)
4. [Training Pipeline](#training-pipeline)
5. [Validation Methodology](#validation-methodology)
6. [Performance Optimization](#performance-optimization)
7. [Scientific Considerations](#scientific-considerations)

---

## Architecture Overview

### Core Design Principles

1. **φ-based Representation**
   - Primary field: 3-form φ(x) with 35 components
   - Metric reconstruction via algebraic identity: g_ij = (1/144) φ_imn φ_jpq φ_rst ε^mnpqrst
   - Ensures G₂ structure compatibility

2. **Manifold Structure**
   - Training domain: 7-torus T⁷ with periodic boundary conditions
   - Coordinate domain: [0, 2π]⁷
   - Enables proper derivative computation with boundary handling

3. **Dual Encoding Options**
   - **Fourier Features**: Explicit periodicity, good for T⁷
   - **SIREN**: Implicit smoothness, better for general manifolds

4. **True Geometric Losses**
   - Torsion-free: L_torsion = ||dφ||² + ||d*φ||²
   - Volume normalization: L_volume = ||det(g) - 1||²
   - Harmonic gauge: Prevents drift without suppressing curvature

---

## Mathematical Foundation

### G₂ Structures

A G₂ structure on a 7-manifold M is defined by a positive 3-form φ satisfying:

1. **Positivity**: φ defines a Riemannian metric via the algebraic formula
2. **Torsion-free**: dφ = 0 (closure)
3. **Co-closure**: d*φ = 0 (where * is the Hodge star)

### Metric Reconstruction

The metric is reconstructed from φ using:

```
g_ij = (1/144) φ_imn φ_jpq φ_rst ε^mnpqrst
```

where ε is the Levi-Civita symbol. In practice, we use an approximate formula:

```python
def metric_from_phi_approximate(phi):
    # Diagonal from phi norm
    metric_diag = ||phi|| / sqrt(7)
    
    # Off-diagonal from cross-terms
    metric_ij ~ sum_k phi_ijk^2 / count
    
    return metric
```

### Exterior Derivatives

For a 3-form φ in 7D:

```
dφ = ∂φ_jkl/∂x^i dx^i ∧ dx^j ∧ dx^k ∧ dx^l
```

We implement two methods:

1. **Autograd** (accurate, slow):
   - Uses PyTorch automatic differentiation
   - Antisymmetrizes derivatives
   - Good for validation

2. **Optimized** (fast, approximate):
   - Finite differences on T⁷ lattice
   - Exploits periodicity
   - ~10x faster for training

---

## Module Documentation

### G2_phi_network.py

**Purpose**: Neural network architectures for φ(x).

**Key Classes**:

#### `FourierFeatures`
```python
class FourierFeatures(nn.Module):
    def __init__(self, input_dim=7, n_modes=16, scale=1.0)
```
- Random Fourier feature encoding
- Maps x ∈ R⁷ to [cos(2πBx), sin(2πBx)] ∈ R^(2*n_modes)
- Explicit periodicity for T⁷

#### `SirenLayer`
```python
class SirenLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0=30.0, is_first=False)
```
- Sinusoidal activation: sin(ω(Wx + b))
- Smooth implicit representation
- Special weight initialization

#### `G2PhiNetwork`
```python
class G2PhiNetwork(nn.Module):
    def __init__(self, encoding_type='fourier', hidden_dims=[256,256,128], ...)
```
- Main network: coordinates → φ
- Output: 35 components (C(7,3) = 35)
- Optional normalization: ||φ||² = 7

**Architecture**:
```
Input (7D) → Encoding → MLP → Output (35D)
```

### G2_geometry.py

**Purpose**: Differential geometry operators.

**Key Functions**:

#### `project_spd(metric, epsilon=1e-6)`
- Projects symmetric matrix to positive definite
- Method: eigenvalue decomposition + clamping
- Returns: SPD metric

#### `exterior_derivative_3form(phi, coords, method='autograd')`
- Computes d(φ) for 3-form φ
- Returns: (d_phi, ||d_phi||²)
- Methods: 'autograd' or 'optimized'

#### `hodge_star(phi, metric)`
- Computes Hodge dual *φ: 3-form → 4-form
- Uses metric for index raising/lowering
- Returns: φ_dual

#### `ricci_tensor(metric, coords)`
- Computes Ricci curvature tensor
- Simplified approximation via Christoffel symbols
- Returns: Ric_ij

### G2_manifold.py

**Purpose**: Training domain management.

**Key Classes**:

#### `TorusT7`
```python
class TorusT7:
    def __init__(self, radii=None, device='cpu')
```
- 7-dimensional torus with periodic boundaries
- Default radii: [2π, 2π, ..., 2π]

**Methods**:
- `sample_points(n_batch, method='uniform')`: Random sampling
- `enforce_periodicity(coords)`: Wrap to [0, L_i]
- `create_validation_grid(points_per_dim)`: Regular grid

### G2_losses.py

**Purpose**: Loss functions and curriculum learning.

**Key Functions**:

#### `torsion_loss(phi, metric, coords, method='autograd')`
```python
L_torsion = ||dφ||² + ||d*φ||²
```
- Primary geometric loss
- Enforces torsion-free condition
- Returns: (loss, info_dict)

#### `volume_loss(metric)`
```python
L_volume = ||det(g) - 1||²
```
- Normalizes volume form
- Target: det(g) = 1

#### `harmonic_gauge_loss(metric, coords)`
```python
L_gauge = ||∇^i g_ij||²
```
- Prevents metric drift
- Doesn't suppress curvature (unlike simple regularization)

#### `CurriculumScheduler`
```python
class CurriculumScheduler:
    def __init__(self, phase_epochs=[500, 2000, 3000], ...)
```
- Progressive weight scheduling
- Phase 1: Learn φ structure (high volume, low torsion)
- Phase 2: Balance (equal weights)
- Phase 3: Torsion-free refinement (high torsion, low volume)

### G2_validation.py

**Purpose**: Comprehensive validation and analysis.

**Key Functions**:

#### `validate_torsion_free(model, manifold, n_samples=1000)`
- Checks ||dφ||² < ε and ||d*φ||² < ε
- Target: < 10⁻⁶ for torsion-free

#### `validate_curvature(model, manifold, n_samples=500)`
- Ricci flatness check
- Eigenvalue spectrum analysis
- Condition number monitoring

#### `validate_metric_quality(model, manifold, n_samples=1000)`
- Positive definiteness
- Symmetry
- Volume normalization
- Phi norm

#### `comprehensive_validation(model, manifold, ...)`
- Runs all validation checks
- Compiles comprehensive report
- Returns: quality score and detailed metrics

---

## Training Pipeline

### Standard Training

```bash
python G2_train.py --encoding fourier --epochs 3000 --batch-size 512
```

**Configuration**:
- Encoding: fourier or siren
- Hidden dimensions: [256, 256, 128]
- Batch size: 512 points
- Learning rate: 1e-4
- Weight decay: 1e-4
- Gradient clipping: 1.0

**Curriculum Phases**:

| Phase | Epochs | λ_torsion | λ_volume | Focus |
|-------|--------|-----------|----------|-------|
| 1 | 0-500 | 0.1 | 10.0 | Structure Learning |
| 2 | 500-2000 | 1.0 | 1.0 | Balance |
| 3 | 2000-3000 | 10.0 | 0.1 | Torsion-Free |

### Monitoring

During training, monitor:
- Total loss (should decrease steadily)
- Torsion loss (target: < 10⁻⁶)
- ||φ||² (should stabilize at 7.0)
- det(g) (should stabilize at 1.0)

### Checkpointing

Checkpoints saved every 500 epochs containing:
- Model state dict
- Optimizer state dict
- Training configuration
- Current loss values

---

## Validation Methodology

### Torsion-Free Verification

**Criterion**: ||dφ||² + ||d*φ||² < 10⁻⁶

**Method**:
1. Sample 1000-2000 points on T⁷
2. Compute φ at each point
3. Evaluate exterior derivatives
4. Check norm squared

**Interpretation**:
- < 10⁻⁶: Excellent (true torsion-free)
- 10⁻⁶ - 10⁻⁴: Good (approximate torsion-free)
- > 10⁻⁴: Poor (significant torsion)

### Curvature Analysis

**Ricci Flatness**: For torsion-free G₂, should have Ric(g) = 0

**Non-Flatness**: Riemann tensor should be non-zero (verify curvature exists)

**Eigenvalue Spectrum**:
- All positive: metric is positive definite ✓
- Condition number < 100: well-conditioned ✓
- Large spread: might need better normalization

### Metric Quality

**Checks**:
1. Positive definiteness: min(eigenvalues) > 0
2. Symmetry: ||g - g^T|| < 10⁻⁵
3. Volume: |det(g) - 1| < 0.1
4. Phi norm: |||φ||² - 7| < 0.1

**G₂ Identity**: φ ∧ *φ = (7/6) vol_g
- Relative error < 10%: Good
- Relative error > 20%: Poor structure

---

## Performance Optimization

### Computational Complexity

**Forward Pass**:
- Encoding: O(7 × n_modes)
- MLP: O(sum of layer dimensions)
- Total: ~120k parameters → ~0.5ms per batch (GPU)

**Backward Pass**:
- Autograd derivatives: O(35 × 7 × batch_size)
- Dominant cost: exterior derivative computation
- Optimization: use 'optimized' method for training

### Memory Requirements

**Per Batch** (batch_size=512):
- Input coords: 512 × 7 × 4 bytes = 14 KB
- Phi: 512 × 35 × 4 bytes = 70 KB
- Metric: 512 × 7 × 7 × 4 bytes = 100 KB
- Gradients: ~2x forward pass
- **Total: ~400 KB per batch**

**Training** (3000 epochs, batch=512):
- Model: ~0.5 MB (parameters)
- Checkpoints: ~1.5 MB each
- History: ~1 MB (CSV)
- **Total: ~10 MB**

### GPU Utilization

**Recommended Hardware**:
- Minimum: RTX 3090 (24 GB VRAM)
- Recommended: A100 (40-80 GB VRAM)
- Training time: 4-6 hours (3000 epochs on A100)

**Optimization Tips**:
1. Use mixed precision training (torch.cuda.amp)
2. Increase batch size if memory allows
3. Use 'optimized' derivative method
4. Profile with torch.profiler for bottlenecks

---

## Scientific Considerations

### Reproducibility

**Fixed Random Seeds**:
```python
torch.manual_seed(42)
np.random.seed(42)
```

**Deterministic Algorithms**:
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**Documented Hyperparameters**: All settings saved in config.json

### Verification

**Autograd vs Optimized**:
- Train with autograd for small tests
- Validate optimized version against autograd
- Report differences in documentation

**Cross-Validation**:
- Multiple random initializations
- Different encoding types (Fourier vs SIREN)
- Various batch sizes

### Limitations

**Known Issues**:
1. Metric reconstruction is approximate (not exact algebraic formula)
2. Hodge star computation simplified
3. Ricci tensor uses first-order approximation
4. Topological Betti numbers are rough estimates

**Future Work**:
1. Implement exact metric reconstruction
2. Extend to twisted connected sum (K3 × T³)
3. Add field of moduli for deformations
4. Rigorous topological analysis

---

## References

### Mathematical Background

1. Joyce, D. D. (2000). *Compact Manifolds with Special Holonomy*
2. Bryant, R. L. (1987). "Metrics with exceptional holonomy"
3. Karigiannis, S. (2009). "Flows of G₂ structures"

### Implementation

1. PyTorch Documentation: https://pytorch.org/docs/
2. SIREN: Sitzmann et al. (2020). "Implicit Neural Representations with Periodic Activation Functions"
3. Fourier Features: Tancik et al. (2020). "Fourier Features Let Networks Learn High Frequency Functions"

---

## Appendix: File Structure

```
outputs/0.2/
├── G2_phi_network.py          # Neural architectures
├── G2_geometry.py              # Differential operators
├── G2_losses.py                # Loss functions
├── G2_manifold.py              # Domain management
├── G2_validation.py            # Validation suite
├── G2_train.py                 # Training script
├── G2_eval.py                  # Evaluation utilities
├── G2_export.py                # ONNX export
├── Complete_G2_Metric_Training_v0_2.ipynb  # Interactive notebook
├── TECHNICAL_DOCUMENTATION.md  # This file
├── README.md                   # User guide
└── requirements.txt            # Dependencies
```

---

**GIFT Project - Geometric Inference Framework Theory**

*Last updated: 2025-01-07*






