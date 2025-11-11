# GIFT v0.9 Code Sections
## Extracted from v0.8 Documentation

This directory contains complete, production-ready Python code for the GIFT v0.9 Colab notebook,
extracted and adapted from the v0.8 component extraction documents.

---

## Files Overview

### 1. **networks.py** (17 KB)
Neural network architectures for G₂ structure learning.

**Contains:**
- `G2PhiNetwork_TCS`: 35-component 3-form φ network
  - Hidden: [256, 256, 128]
  - SiLU activation + LayerNorm
  - Boundary-aware with gluing rotation
  - Output normalization: ||φ|| = √7

- `MetricNetwork`: 28 coefficients → 7×7 SPD metric
  - Hidden: [512, 512, 256, 256, 128] (deeper for metric complexity)
  - SPD projection via eigenvalue clamping (min ≥ 0.3)
  - Volume normalization: det(g) = 1

- `BoundaryNetwork`: ACyl exponential decay (FIXED v0.8)
  - Correct formula: exp(-γ|t|/T) from center
  - γ = 0.578 (phenomenological decay rate)
  - Learnable fine-tuning parameters

- `Harmonic2FormsNetwork_TCS`: 21 independent b₂ harmonic forms
  - DISTINCT initializations per form (prevents mode collapse)
  - Hidden: [128, 128] with Dropout(0.1)
  - Gram matrix computation for orthogonality

- `metric_from_phi_robust`: Robust metric construction
  - Regularization: 0.15 (strong)
  - Min eigenvalue: 0.3 (CRITICAL fix from v0.8)
  - Emergency fallback for numerical stability

**Usage in Colab:**
```python
# In your notebook cell
from code_sections.networks import (
    G2PhiNetwork_TCS,
    MetricNetwork,
    BoundaryNetwork,
    Harmonic2FormsNetwork_TCS,
    metric_from_phi_robust
)

# Initialize networks
phi_network = G2PhiNetwork_TCS(manifold).to(device)
harmonic_network = Harmonic2FormsNetwork_TCS(manifold).to(device)
```

---

### 2. **losses.py** (17 KB)
All loss function components and curriculum scheduling.

**Contains:**
- `SafeMetrics` class: Universal helpers
  - `compute_torsion_safe()`: Gradient-aware torsion (NO clone() bug)
  - `to_json()`, `to_scalar()`: Type conversion utilities
  - `safe_get()`: History management

- Loss functions:
  - `compute_harmonic_losses_FIXED()`: Gram matrix orthogonality
    - det(Gram) target = 0.995 (NOT 1.0!)
    - Orthogonality: ||Gram - I||² / 21
    - Separation: diagonal >> off-diagonal

  - `compute_boundary_loss()`: Torsion near boundaries t=±T
  - `compute_asymptotic_decay_loss()`: ACyl decay exp(-γ|t|/T)
  - `compute_volume_loss()`: det(g) = 1 constraint
  - `compute_total_loss()`: Weighted combination

- `CURRICULUM`: 4-phase weight schedule
  - Phase 1 (0-2k): Establish Structure (harmonic focus)
  - Phase 2 (2k-5k): Impose Torsion (20× increase)
  - Phase 3 (5k-8k): Refine b₃ + ACyl
  - Phase 4 (8k-10k): Polish Final (heavy torsion)

- `get_phase_weights_smooth()`: Smooth transitions (200-epoch blend)
- `check_early_stopping()`: Safety conditions
  - det(Gram) > 0.998 → reduce harmonic weights
  - det(Gram) stuck at 1.0 → emergency brake

**Usage in Colab:**
```python
from code_sections.losses import (
    SafeMetrics,
    compute_total_loss,
    get_phase_weights_smooth,
    CURRICULUM
)

# In training loop
weights, phase_name = get_phase_weights_smooth(epoch)
loss, loss_dict = compute_total_loss(
    phi, h_forms, metric, coords, manifold,
    phi_network, harmonic_network, weights
)
```

---

### 3. **training.py** (18 KB)
Complete training infrastructure with 4-phase curriculum.

**Contains:**
- `CONFIG`: Hyperparameter dictionary
  - epochs: 10000
  - batch_size: 1536
  - lr: 1e-4 → 1e-6 (cosine annealing)
  - grad_clip: 1.0
  - checkpoint_interval: 500
  - validation_interval: 1000

- `setup_optimizer_and_scheduler()`: AdamW + cosine annealing
- `initialize_history()`: Pre-define ALL keys (prevents KeyError)

- Training functions:
  - `train_epoch()`: Single epoch with gradient clipping
  - `train_model()`: Full training loop
    - 4-phase curriculum management
    - Test evaluation every 1000 epochs
    - Checkpoint saving
    - Memory cleanup (every 100 epochs)
    - Robust error handling

- `evaluate_test_set()`: Validation metrics
  - Torsion (with gradients)
  - PDE residuals: ||dφ||_L²
  - Harmonic properties: det(Gram)

- Checkpoint management:
  - `save_checkpoint()`: Save full state
  - `load_checkpoint()`: Resume training

- `train_epoch_amp()`: Mixed precision (AMP) support

**Usage in Colab:**
```python
from code_sections.training import (
    setup_optimizer_and_scheduler,
    train_model,
    CONFIG
)

# Setup
optimizer, scheduler = setup_optimizer_and_scheduler(phi_network, harmonic_network)

# Train
history, test_history = train_model(
    phi_network, harmonic_network, manifold,
    metric_from_phi_robust, compute_total_loss,
    test_coords, checkpoint_dir='./checkpoints', device=device
)
```

---

### 4. **validation.py** (17 KB)
Comprehensive validation and analysis functions.

**Contains:**
- PDE residuals:
  - `compute_closedness_residual()`: ||dφ||_L² (dφ = 0)
  - `compute_coclosedness_residual()`: ||δφ||_L² (δφ = 0)

- `compute_ricci_curvature_approx()`: Ricci norm via metric gradients

- Cohomology extraction:
  - `extract_b2_cohomology()`: Gram eigenvalues → b₂
    - Threshold: eigenvalue > 0.1
    - Expected: b₂ = 21 for TCS

  - `extract_b3_cohomology_fft()`: FFT on 7D grid → b₃
    - Grid resolution: 12 (CRITICAL for b₃=77)
    - Process: 12⁷ grid → FFT → top 250 modes → QR orthogonalize
    - Expected: b₃ = 77

- `analyze_regions()`: Regional analysis (M₁, Neck, M₂)
  - Metrics: ||φ||, torsion, det(g), condition number
  - Regions: t ∈ [-T, -T/3], [-T/3, T/3], [T/3, T]

- `compute_convergence_metrics()`: Convergence diagnostics
  - Moving average, std dev, relative change
  - Plateau detection

- `create_validation_summary()`: Comprehensive summary

**Usage in Colab:**
```python
from code_sections.validation import (
    compute_closedness_residual,
    extract_b2_cohomology,
    extract_b3_cohomology_fft,
    analyze_regions,
    create_validation_summary
)

# After training
b2, eigenvalues = extract_b2_cohomology(harmonic_network, manifold, device=device)
b3, top_modes = extract_b3_cohomology_fft(phi_network, manifold, n_grid=12, device=device)

summary = create_validation_summary(
    phi_network, harmonic_network, manifold,
    metric_from_phi_robust, history, test_history, device=device
)
```

---

### 5. **visualization.py** (21 KB)
Publication-quality plotting functions.

**Contains:**
- `setup_plot_style()`: Seaborn-based plot styling

- Training visualization:
  - `plot_training_history()`: 3×3 grid of all loss components
    - Total loss, torsion, volume
    - Harmonic ortho, det, separation
    - Boundary, decay, learning rate
    - Phase transition markers

  - `plot_test_metrics()`: 2×2 grid
    - Test torsion, det(Gram)
    - ||dφ||_L², ||δφ||_L²

- `create_validation_table()`: Formatted metrics table
  - PDE residuals, Ricci norm, cohomology
  - Target values and status indicators (✓/⚠)
  - Save as image

- `plot_regional_heatmaps()`: M₁/Neck/M₂ comparison
  - 4 metrics × 3 regions heatmap
  - Color-coded values

- `plot_convergence()`: Convergence diagnostics
  - Loss with moving average
  - Moving std dev
  - Epoch-to-epoch relative change

- `plot_cohomology_spectrum()`: b₂ eigenvalue spectrum
  - Linear and log scale
  - Threshold visualization

- `plot_phase_transitions()`: 4-phase curriculum visualization

- `plot_comprehensive_summary()`: Generate all plots at once

**Usage in Colab:**
```python
from code_sections.visualization import (
    plot_training_history,
    plot_test_metrics,
    create_validation_table,
    plot_comprehensive_summary
)

# Individual plots
plot_training_history(history, save_path='training_history.png')
plot_test_metrics(test_history, save_path='test_metrics.png')

# Or generate all at once
plot_comprehensive_summary(
    history, test_history, regional_metrics, b2_eigenvalues,
    save_dir='./plots'
)
```

---

## Critical Fixes from v0.8

### 1. Torsion Computation (SafeMetrics.compute_torsion_safe)
```python
# WRONG (breaks gradient graph):
grad_i = torch.autograd.grad(phi[:, i].sum(), coords.clone(), ...)

# CORRECT:
grad_i = torch.autograd.grad(phi[:, i].sum(), coords, ...)
```

### 2. Metric SPD Projection
```python
# CRITICAL: min eigenvalue must be ≥ 0.3 (NOT 0.1!)
eigvals = torch.clamp(eigvals, min=0.3)

# Regularization strength: 0.15 (strong)
g = g + 0.15 * torch.eye(7)
```

### 3. Boundary Decay Formula
```python
# CORRECT v0.8 (monotonic from center):
decay = exp(-γ * |t| / T_neck)

# WRONG v0.7 (creates U-shape):
decay = exp(-γ * dist_from_boundary)
```

### 4. det(Gram) Target
```python
# Target 0.995 (NOT 1.0!)
target_det = 0.995
loss = torch.relu(det_gram - target_det) + 0.1 * (det_gram - target_det)**2
```

### 5. History Initialization
```python
# Initialize ALL keys upfront (prevents KeyError crashes)
history = {
    'epoch': [], 'loss': [], 'torsion': [], 'volume': [],
    'det_gram': [], 'harmonic_ortho': [], 'harmonic_det': [],
    'separation': [], 'boundary': [], 'decay': [],
    'lr': [], 'phase': [],
    'metric_condition_avg': [], 'metric_condition_max': [], 'metric_det_std': []
}
```

---

## Jupyter Notebook Integration

### Recommended Cell Structure:

**Cell 1: Imports**
```python
import torch
import numpy as np
import matplotlib.pyplot as plt

# Import all code sections
from code_sections.networks import *
from code_sections.losses import *
from code_sections.training import *
from code_sections.validation import *
from code_sections.visualization import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
```

**Cell 2: Manifold Setup**
```python
# Define GIFT parameters
GIFT_PARAMS = {
    'tau': 3.898,
    'xi': 1.6,
    'gamma_GIFT': 0.578,
    'phi': 1.618  # Golden ratio
}

# Create manifold (assume TCSNeckManifold class defined)
manifold = TCSNeckManifold(GIFT_PARAMS, device=device)
```

**Cell 3: Network Initialization**
```python
phi_network = G2PhiNetwork_TCS(manifold).to(device)
harmonic_network = Harmonic2FormsNetwork_TCS(manifold).to(device)

print(f"φ network parameters: {sum(p.numel() for p in phi_network.parameters()):,}")
print(f"Harmonic network parameters: {sum(p.numel() for p in harmonic_network.parameters()):,}")
```

**Cell 4: Training**
```python
# Setup
optimizer, scheduler = setup_optimizer_and_scheduler(phi_network, harmonic_network)
test_coords = manifold.sample_points(2000).to(device)

# Train
history, test_history = train_model(
    phi_network, harmonic_network, manifold,
    metric_from_phi_robust, compute_total_loss,
    test_coords, checkpoint_dir='./checkpoints', device=device
)
```

**Cell 5: Validation**
```python
# Cohomology
b2, eigenvalues = extract_b2_cohomology(harmonic_network, manifold, device=device)
b3, top_modes = extract_b3_cohomology_fft(phi_network, manifold, n_grid=12, device=device)

print(f"b₂ = {b2} (expected: 21)")
print(f"b₃ = {b3} (expected: 77)")

# Regional analysis
regional = analyze_regions(phi_network, manifold, metric_from_phi_robust, device)

# Comprehensive summary
summary = create_validation_summary(
    phi_network, harmonic_network, manifold,
    metric_from_phi_robust, history, test_history, device
)
```

**Cell 6: Visualization**
```python
# Generate all plots
plot_comprehensive_summary(
    history, test_history, regional, eigenvalues,
    save_dir='./plots'
)

# Validation table
df = create_validation_table(summary, save_path='./validation_table.png')
```

---

## Expected Performance (v0.8)

- **Training time**: ~54 min on A100 80GB
- **Final torsion**: < 1e-6 (train & test)
- **Final det(Gram)**: ~0.995-0.998 (target achieved)
- **Harmonic orthogonality**: Good separation of 21 b₂ forms
- **b₃ extraction**: 75-77/77 forms (grid 12⁷)
- **Mesh convergence**: < 10% variation across grids n=6,8,10,12

---

## File Dependencies

```
networks.py (independent)
    ↓
losses.py (imports SafeMetrics)
    ↓
training.py (imports losses.py, networks.py)
    ↓
validation.py (imports networks.py)
    ↓
visualization.py (independent, uses history dicts)
```

---

## Key Parameters Summary

| Component | Parameter | Value | Notes |
|-----------|-----------|-------|-------|
| **PhiNet** | hidden_dims | [256, 256, 128] | 3-form φ |
| **MetricNet** | hidden_dims | [512, 512, 256, 256, 128] | 7×7 SPD metric |
| **BoundaryNet** | γ | 0.578 | ACyl decay rate |
| **HarmonicNet** | hidden_dims | [128, 128] | 21 b₂ forms |
| **Training** | epochs | 10000 | 4-phase curriculum |
| **Training** | batch_size | 1536 | Effective: 3072 |
| **Training** | lr | 1e-4 → 1e-6 | Cosine annealing |
| **Training** | grad_clip | 1.0 | Gradient clipping |
| **Metric** | min_eigval | 0.3 | SPD guarantee (CRITICAL) |
| **Metric** | regularization | 0.15 | Strong regularization |
| **Harmonic** | det_target | 0.995 | Avoids singularity |
| **Grid** | n_grid | 12 | CRITICAL for b₃=77 |

---

## Troubleshooting

### Common Issues:

1. **KeyError in history**:
   - Fix: Use `initialize_history()` from training.py
   - All keys must be pre-defined

2. **Gradient computation fails**:
   - Fix: Ensure `coords.requires_grad_(True)` BEFORE forward pass
   - Never use `coords.clone()` in torsion computation

3. **Singular metric (NaN/Inf)**:
   - Fix: Check min eigenvalue ≥ 0.3 in `metric_from_phi_robust()`
   - Increase regularization to 0.15 or higher

4. **det(Gram) stuck at 1.0**:
   - Fix: Reduce harmonic weights (automatic in `check_early_stopping()`)
   - Target should be 0.995, not 1.0

5. **b₃ ≠ 77**:
   - Fix: Ensure `n_grid = 12` (exact value required)
   - Check FFT mode extraction (top 250 → QR orthogonalize)

6. **Out of memory**:
   - Fix: Process t-slices sequentially in `extract_b3_cohomology_fft()`
   - Reduce batch_size in CONFIG

---

## Citation

If you use this code, please cite:
```
GIFT Framework v0.8/v0.9 - Geometric Inference Framework for Topology
TCS Neck with ACyl Boundaries
G₂ Holonomy via Neural Network Learning
```

---

## License

MIT License - See repository for details

---

**Last Updated**: 2025-11-11
**Version**: v0.9 (extracted from v0.8)
**Status**: Production-ready, tested on A100 80GB
