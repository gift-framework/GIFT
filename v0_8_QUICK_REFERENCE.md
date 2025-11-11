# GIFT v0.8 → v0.9: Quick Reference Card

## CRITICAL FIXES FOR v0.9

### 1. Torsion Computation Bug (MUST FIX)
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

# γ = 0.578 (phenomenological)
# T_neck ≈ 24.48
```

### 4. det(Gram) Target
```python
# Target 0.995 (NOT 1.0!)
target_det = 0.995
loss = torch.relu(det_gram - target_det) + 0.1 * (det_gram - target_det)**2

# REASON: Prevents singularity by avoiding exact 1.0
```

### 5. History Initialization
```python
# Initialize ALL keys upfront (prevents KeyError crashes)
history = {
    'epoch': [],
    'loss': [],
    'torsion': [],
    'volume': [],
    'det_gram': [],
    'harmonic_ortho': [],
    'harmonic_det': [],
    'separation': [],
    'boundary': [],
    'decay': [],
    'lr': [],
    'phase': [],
    'metric_condition_avg': [],
    'metric_condition_max': [],
    'metric_det_std': []
}
```

---

## NETWORK DIMENSIONS

| Network | Input | Hidden | Output | Notes |
|---------|-------|--------|--------|-------|
| PhiNetwork | Fourier feat. | [256,256,128] | 35 | 3-form φ |
| MetricNetwork | Fourier feat. | [512,512,256,256,128] | 28 | 7×7 SPD metric |
| BoundaryNetwork | coords(7D) | Learnable params | 1 | ACyl decay [0,1] |
| HarmonicNetwork | Fourier feat. | [128,128] | 21 | b₂ forms (×21 networks) |

---

## TRAINING HYPERPARAMETERS

```python
CONFIG = {
    'epochs': 10000,
    'batch_size': 1536,
    'lr': 1e-4,              # Starting lr
    'weight_decay': 1e-4,
    'grad_clip': 1.0,
    'scheduler': 'cosine',   # Cosine annealing
    'eta_min': 1e-6,         # Final lr (100× decay)
    'seed': 47,
    'checkpoint_interval': 500,
    'validation_interval': 1000,
    'b3_grid_resolution': 12, # CRITICAL for b₃=77
}
```

---

## 4-PHASE CURRICULUM (Loss Weights)

| Phase | Epochs | Torsion | Volume | Harmonic | Boundary | Decay | ACyl |
|-------|--------|---------|--------|----------|----------|-------|------|
| 1 | 0-2k | 0.1 | 0.6 | 6.0 | 0.05 | 0.05 | 0.0 |
| 2 | 2k-5k | 2.0 | 0.4 | 3.0 | 0.5 | 0.3 | 0.1 |
| 3 | 5k-8k | 5.0 | 0.2 | 2.0 | 1.0 | 0.5 | 0.3 |
| 4 | 8k-10k | 20.0 | 0.1 | 1.0 | 1.5 | 1.0 | 0.5 |

---

## MANIFOLD PARAMETERS

```python
GIFT_PARAMS = {
    'tau': 3.898,        # Modulus (T_neck = tau × 2π ≈ 24.48)
    'xi': 1.6,           # Gluing angle (radians)
    'gamma_GIFT': 0.578, # ACyl decay rate
    'phi': 1.618,        # Golden ratio (K3 hierarchy)
}

# Derived:
T_neck = 3.898 * 2π ≈ 24.48
Fiber radii: [2π, 2π]
K3 radii: [2π, 2π, 2π/φ, 2π/φ]
```

---

## MESH GENERATION (b₃ Extraction)

```python
# 7D Grid parameters:
n_grid = 12  # CRITICAL! Must be 12
total_points = 12**7 = 35,831,808

# Grid structure:
# dim 0 (t):     [-T_neck, +T_neck],  12 points
# dim 1-2 (θ,φ): [0, 2π],             12 points each
# dim 3-6 (x):   [0, radius],         12 points each

# Processing:
# - Process 12 t-slices sequentially (save VRAM)
# - Each t-slice: 12^6 ≈ 2.9M points
# - Batch size: 8192 per iteration
# - FFT: 35 components (streaming)
# - Mode selection: Top 250 candidates → orthogonalize to 77
```

---

## LOSS FUNCTIONS (Formulas)

### Torsion (Gradient Norm Proxy)
```
T ≈ ||∇φ|| = sqrt(sum_i ||∂φ_i/∂x_j||²)
```

### Metric from φ
```
g_ij = φ_k * 0.1 + δ_ij + 0.15*δ_ij
(then SPD projection via eigenvalue clamping)
```

### Harmonic Orthogonality
```
L_ortho = ||Gram - I||_F / 21
where Gram[α,β] = <ω_α, ω_β>_L²
```

### Harmonic Determinant
```
L_det = ReLU(det(Gram) - 0.995) + 0.1*(det(Gram) - 0.995)²
```

### Boundary Decay
```
expected_decay = exp(-γ|t|/T_neck)
L_decay = |φ_amplitude - expected_decay|_avg
```

### Separation (Gram)
```
L_sep = ReLU(0.5 - (diag_mean - off_diag_mean))
```

---

## TEST METRICS (Every 1000 epochs)

```python
# Test set: 2000 fixed points (with requires_grad=True)

# Logged metrics:
'test_torsion':   SafeMetrics.compute_torsion_safe(phi, coords, metric, use_grad=True)
'test_det_gram':  Determinant of harmonic Gram matrix
'test_dphi_L2':   ||∇φ||_L² (exterior derivative)
'test_dstar_phi_L2': Codifferential residual (approximated)

# Health checks:
'metric_condition_avg': mean(λ_max / λ_min) over batch
'metric_condition_max': max(λ_max / λ_min) over batch
'metric_det_std':       std(det(metric)) over batch
```

---

## STOPPING CONDITIONS

```python
# EARLY STOPPING (if det(Gram) too perfect):
if epoch > 2000 and det_gram.item() > 0.998:
    weights['harmonic_ortho'] *= 0.5
    weights['harmonic_det'] *= 0.5

# EMERGENCY BRAKE (over-optimization):
if epoch > 3000 and all(abs(det_gram - 1.0) < 1e-6 for last 5 test epochs):
    print("EARLY STOPPING: det(Gram) stuck at 1.0")
    break
```

---

## FILES TO PRESERVE FROM v0.8

1. **SafeMetrics class** - Universal type converter + history helpers
2. **metric_from_phi_robust()** - SPD projection with eigenvalue clamping
3. **compute_harmonic_losses_FIXED()** - Gram matrix orthogonality
4. **BoundaryNetwork class** - FIXED ACyl decay formula
5. **TCSNeckManifold class** - 7D manifold structure
6. **Optimizer + Scheduler setup** - AdamW + cosine annealing
7. **4-phase curriculum** - Loss weight scheduling
8. **Training loop structure** - Checkpoint management + history tracking

---

## EXPECTED PERFORMANCE (v0.8)

- **Training time**: ~54 min on A100 80GB
- **Final torsion**: < 1e-6 (train & test)
- **Final det(Gram)**: ~0.995-0.998 (target achieved)
- **Harmonic orthogonality**: Good separation of 21 b₂ forms
- **b₃ extraction**: 75-77/77 forms (grid 12⁷)
- **Mesh convergence**: < 10% variation across grids n=6,8,10,12

