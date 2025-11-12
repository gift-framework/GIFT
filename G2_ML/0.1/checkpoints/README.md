# G2 Metric Training Checkpoints

This folder contains checkpoints from different stages of the G2 metric training process, enabling reproducibility and analysis of training dynamics.

## Checkpoints

### 1. Epoch 1500 - G2 Emphasis Phase
**File:** `k7_g2_checkpoint_epoch_1500.pt`  
**Training Phase:** Phase 3 (G2 emphasis)  
**Curriculum Weights:** Ricci=0.2, G2=1.0, Reg=0.05

**Status at this point:**
- ||phi||^2 = 7.000000 (perfect normalization)
- det(g) = 1.002410
- Condition number: 5.36
- G2 loss: ~3.2e-06 (entering high precision)

**Key milestone:** Transition from Ricci-dominance to G2-structure emphasis. Metric becomes well-structured.

---

### 2. Epoch 3000 - G2 Dominance Phase
**File:** `k7_g2_checkpoint_epoch_3000.pt`  
**Training Phase:** Phase 4 (G2 dominance)  
**Curriculum Weights:** Ricci=0.05, G2=2.0, Reg=0.02

**Status at this point:**
- ||phi||^2 = 7.000000 (maintained)
- det(g) = 1.000032 (excellent volume)
- Condition number: 5.37
- G2 loss: ~2.9e-07 (very low)

**Key milestone:** G2 structure dominates. Volume normalization excellent. Ready for aggressive refinement.

---

### 3. Epoch 5500 - Aggressive G2 Phase
**File:** `k7_g2_checkpoint_epoch_5500.pt`  
**Training Phase:** Phase 5 (Aggressive G2)  
**Curriculum Weights:** Ricci=0.02, G2=3.0, Reg=0.01

**Status at this point:**
- ||phi||^2 = 7.000000 (stable)
- det(g) = 1.000004 (near-perfect)
- Condition number: 5.37
- G2 loss: ~6.9e-08 (extremely low)
- Ricci loss: ~4.3e-05 (before polish)

**Key milestone:** Final state before Ricci polish. G2 structure at highest precision.

---

### 4. Final Model - After Ricci Polish
**File:** `G2_final_model.pt` (in parent directory)  
**Training:** 6000 epochs + 500 polish epochs  
**Polish Weights:** Ricci=10.0, G2=1.0, Reg=0.005

**Final status:**
- ||phi||^2 = 7.000001 (perfect)
- det(g) = 1.000004 (excellent, slight trade-off from polish)
- Condition number: 5.37 (stable)
- G2 closure: ||d(phi)||^2, ||d(*phi)||^2 < 1e-08 (spectacul ar)
- Ricci: ||Ric||^2 = 1.4e-05 (one order above target)

**Key milestone:** Publication-ready metric. High-precision G2 holonomy structure confirmed.

## Usage

### Loading a Checkpoint

```python
import torch
from G2_phi_wrapper import CompactG2Network

# Load checkpoint
checkpoint = torch.load('checkpoints/k7_g2_checkpoint_epoch_1500.pt')

# Create model
model = CompactG2Network(hidden_dims=[256, 256, 128], num_freq=32)

# Load weights
if 'model' in checkpoint:
    model.load_state_dict(checkpoint['model'])
elif 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.eval()

# Use model
coords = torch.randn(10, 7) * 5.0
metric = model(coords)
```

### Comparing Checkpoints

```python
# Evaluate all checkpoints to see training progression
checkpoints = [
    'checkpoints/k7_g2_checkpoint_epoch_1500.pt',
    'checkpoints/k7_g2_checkpoint_epoch_3000.pt',
    'checkpoints/k7_g2_checkpoint_epoch_5500.pt',
    '../G2_final_model.pt'
]

for cp_path in checkpoints:
    model = load_model(cp_path)
    # Run evaluation...
```

## Training History

The complete training history is available in:
- `g2_training_history.csv` - Loss values for all 6000 epochs
- `G2_training_history_complete.csv` - Complete training log

## Curriculum Schedule

| Phase | Epochs | Ricci | G2 | Reg | Description |
|-------|--------|-------|----|----|-------------|
| 1 | 0-200 | 1.0 | 0.0 | 0.1 | Ricci-flat approximation |
| 2 | 200-500 | 0.5 | 0.5 | 0.1 | G2 structure introduction |
| 3 | 500-1500 | 0.2 | 1.0 | 0.05 | G2 emphasis |
| 4 | 1500-3000 | 0.05 | 2.0 | 0.02 | G2 dominance |
| 5 | 3000-6000 | 0.02 | 3.0 | 0.01 | Aggressive G2 |
| 6 | 6000-6500 | 10.0 | 1.0 | 0.005 | Ricci polish (optional) |

## Checkpoint File Structure

Each checkpoint file (.pt) contains:
- `model` or `model_state_dict`: Network weights (120,220 parameters)
- `optimizer`: Optimizer state (if available)
- `scheduler`: Learning rate scheduler state (if available)
- `history`: Training history up to that point (if available)

## Reproducibility

All checkpoints use:
- Architecture: CompactG2Network with [256, 256, 128] hidden dims, 32 Fourier frequencies
- Random seed: As specified in training (varies by run)
- Domain size: 5.0 (sampling from [-5, 5]^7)
- Batch size: 512

## File Sizes

- Epoch 1500: ~1.4 MB
- Epoch 3000: ~1.4 MB
- Epoch 5500: ~1.4 MB
- Final model: ~1.4 MB

Total: ~5.6 MB for all checkpoints

## Notes

- Checkpoints from notebook/ and outputs/ directories have been consolidated here
- All checkpoints are CPU-compatible (no CUDA required for loading)
- Use `map_location='cpu'` when loading to avoid CUDA errors on CPU-only systems
- Polish step (epoch 6000-6500) is optional but recommended for highest Ricci precision

## Contact

For questions about checkpoint usage or training reproduction, refer to the main documentation or contact the authors.









