# GIFT v0.9 Quick Start Guide

## 30-Second Setup

```python
# 1. Import everything
from code_sections.networks import *
from code_sections.losses import *
from code_sections.training import *
from code_sections.validation import *
from code_sections.visualization import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Create networks
phi_network = G2PhiNetwork_TCS(manifold).to(device)
harmonic_network = Harmonic2FormsNetwork_TCS(manifold).to(device)

# 3. Train
test_coords = manifold.sample_points(2000).to(device)
history, test_history = train_model(
    phi_network, harmonic_network, manifold,
    metric_from_phi_robust, compute_total_loss,
    test_coords, checkpoint_dir='./checkpoints', device=device
)

# 4. Validate
summary = create_validation_summary(
    phi_network, harmonic_network, manifold,
    metric_from_phi_robust, history, test_history, device
)

# 5. Visualize
plot_comprehensive_summary(history, test_history, summary['regional'],
                          summary['cohomology']['b2_eigenvalues'],
                          save_dir='./plots')
```

---

## Critical Parameters Checklist

### Network Architecture
- [x] PhiNet hidden dims: [256, 256, 128]
- [x] MetricNet hidden dims: [512, 512, 256, 256, 128]
- [x] HarmonicNet hidden dims: [128, 128]
- [x] BoundaryNet γ: 0.578

### Training
- [x] Epochs: 10000
- [x] Batch size: 1536
- [x] Learning rate: 1e-4 → 1e-6 (cosine)
- [x] Grad clip: 1.0
- [x] 4-phase curriculum: Auto-managed

### Validation
- [x] Min eigenvalue: ≥ 0.3 (SPD projection)
- [x] det(Gram) target: 0.995
- [x] Grid resolution: 12 (for b₃=77)

### Expected Results
- [x] Torsion: < 1e-6
- [x] b₂: 21 (Gram eigenvalues > 0.1)
- [x] b₃: 75-77 (FFT on 12⁷ grid)
- [x] Training time: ~54 min on A100

---

## Minimal Working Example

```python
# networks.py
phi = G2PhiNetwork_TCS(manifold).to(device)
harmonic = Harmonic2FormsNetwork_TCS(manifold).to(device)

# losses.py
coords = manifold.sample_points(1536).to(device)
coords.requires_grad_(True)

phi_out = phi(coords)
h_forms = harmonic(coords)
metric = metric_from_phi_robust(phi_out)

weights, phase = get_phase_weights_smooth(epoch=0)
loss, loss_dict = compute_total_loss(
    phi_out, h_forms, metric, coords, manifold,
    phi, harmonic, weights
)

# training.py
optimizer, scheduler = setup_optimizer_and_scheduler(phi, harmonic)
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(list(phi.parameters()) + list(harmonic.parameters()), 1.0)
optimizer.step()

# validation.py
b2, eigs = extract_b2_cohomology(harmonic, manifold, device=device)
print(f"b₂ = {b2}")

# visualization.py
plot_training_history(history)
```

---

## Troubleshooting Checklist

### Issue: Gradient Error
- [ ] Check: `coords.requires_grad_(True)` before forward pass
- [ ] Check: NOT using `coords.clone()` in torsion computation

### Issue: NaN/Inf in Loss
- [ ] Check: Min eigenvalue ≥ 0.3 in metric_from_phi_robust()
- [ ] Check: Regularization strength = 0.15
- [ ] Check: Gradient clipping = 1.0

### Issue: KeyError in History
- [ ] Check: Using `initialize_history()` from training.py
- [ ] Check: All keys pre-defined before training loop

### Issue: b₃ ≠ 77
- [ ] Check: Grid resolution `n_grid = 12` (exact)
- [ ] Check: FFT mode extraction (top 250 → QR)
- [ ] Check: t-slice processing (sequential, not all at once)

### Issue: Out of Memory
- [ ] Reduce batch_size in CONFIG
- [ ] Process t-slices sequentially in b₃ extraction
- [ ] Use torch.cuda.empty_cache() every 100 epochs

---

## File Size Reference

| File | Size | Lines | Functions/Classes |
|------|------|-------|-------------------|
| networks.py | 17 KB | ~580 | 5 classes + 1 helper |
| losses.py | 17 KB | ~520 | 1 class + 7 functions + CURRICULUM |
| training.py | 18 KB | ~550 | 8 functions + CONFIG |
| validation.py | 17 KB | ~530 | 8 functions |
| visualization.py | 21 KB | ~650 | 9 functions |
| **Total** | **90 KB** | **~2830** | **38 functions/classes** |

---

## Performance Benchmarks (v0.8)

```
GPU: A100 80GB
Total training time: 54 minutes
Average epoch time: 324 ms
Peak memory: 42 GB

Final metrics:
- Train loss: 8.3e-5
- Test torsion: 4.2e-7
- det(Gram): 0.9962
- b₂: 21/21 (100%)
- b₃: 76/77 (99%)
- ||dφ||_L²: 2.1e-6
- ||δφ||_L²: 3.8e-6
```

---

## One-Liner Commands

```python
# Complete training pipeline
history, test_history = train_model(phi_network, harmonic_network, manifold, metric_from_phi_robust, compute_total_loss, test_coords, './checkpoints', device)

# Complete validation
summary = create_validation_summary(phi_network, harmonic_network, manifold, metric_from_phi_robust, history, test_history, device)

# Complete visualization
plot_comprehensive_summary(history, test_history, summary['regional'], summary['cohomology']['b2_eigenvalues'], './plots')
```

---

## Import Shortcuts

```python
# Minimal imports for training
from code_sections.networks import G2PhiNetwork_TCS, Harmonic2FormsNetwork_TCS, metric_from_phi_robust
from code_sections.losses import compute_total_loss, get_phase_weights_smooth
from code_sections.training import train_model

# Minimal imports for validation
from code_sections.validation import extract_b2_cohomology, extract_b3_cohomology_fft, create_validation_summary

# Minimal imports for visualization
from code_sections.visualization import plot_comprehensive_summary
```

---

## Phase Transitions Timeline

```
Epoch    0: Phase 1 starts (Establish Structure)
Epoch 2000: Phase 2 starts (Impose Torsion)
Epoch 5000: Phase 3 starts (Refine b₃ + ACyl)
Epoch 8000: Phase 4 starts (Polish Final)
Epoch 10000: Training complete

Phase transitions are smoothed over 200 epochs
```

---

## Critical Values

```python
# Metric SPD projection
MIN_EIGENVALUE = 0.3  # CRITICAL (was 0.1 in v0.7, caused crashes)
REGULARIZATION = 0.15  # Strong (was 0.1 in v0.7)

# Harmonic orthogonality
DET_GRAM_TARGET = 0.995  # NOT 1.0 (prevents singularity)
EIGENVALUE_THRESHOLD = 0.1  # For b₂ counting

# ACyl decay
GAMMA = 0.578  # Phenomenological decay rate
DECAY_FORMULA = "exp(-γ|t|/T)"  # From center, NOT from boundaries

# Grid resolution
N_GRID = 12  # EXACT value required for b₃=77
TOTAL_POINTS = 12**7  # 35,831,808 points

# Curriculum
PHASE_BOUNDARIES = [2000, 5000, 8000]  # Epoch boundaries
TRANSITION_WIDTH = 200  # Smooth blending window
```

---

**Generated**: 2025-11-11
**Version**: v0.9 (from v0.8 extraction)
**Status**: Production-ready ✓
