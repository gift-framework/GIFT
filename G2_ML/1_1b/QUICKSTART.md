# G2_ML v1.1b Quick Start Guide

## Ready to Run in 5 Minutes

This guide gets you training the complete GIFT 2.1 RG flow model immediately.

---

## Prerequisites

```bash
# Installed
✓ Python 3.8+
✓ PyTorch 2.0+
✓ CUDA (for GPU training)
✓ NumPy, SciPy, Pandas

# Recommended GPU
- NVIDIA A100 (best)
- NVIDIA V100
- RTX 3090
- Minimum: 8 GB VRAM
```

---

## Option 1: Google Colab (Easiest)

1. Open the notebook:
   ```
   G2_ML/1_1b/K7_G2_TCS_RGFlow_v1_1b.ipynb
   ```

2. Click "Open in Colab" badge at top

3. Enable GPU:
   - Runtime → Change runtime type → GPU → Save

4. Run all cells:
   - Runtime → Run all

**Estimated Time**: 10-12 hours on Colab GPU

---

## Option 2: Local Jupyter

```bash
# Navigate to directory
cd G2_ML/1_1b

# Launch Jupyter
jupyter notebook K7_G2_TCS_RGFlow_v1_1b.ipynb

# In notebook: Cell → Run All
```

**Estimated Time**: 8-10 hours on A100

---

## Option 3: Command Line Training (Advanced)

```bash
# Convert notebook to script (one-time)
jupyter nbconvert --to script K7_G2_TCS_RGFlow_v1_1b.ipynb

# Run in background
nohup python K7_G2_TCS_RGFlow_v1_1b.py > training.log 2>&1 &

# Monitor progress
tail -f training.log
```

---

## Quick Verification (5 Minutes)

Before full training, verify everything works:

```python
# In first code cell, modify CONFIG:
CONFIG['n_epochs_per_phase'] = 10  # Quick test
CONFIG['checkpoint_freq'] = 5

# Run cells up through training initialization
# Should see:
# ✓ Device: cuda
# ✓ RGFlowGIFT class ready
# ✓ SmartEarlyStopping class ready
# ✓ RGFlowMonitor class ready
```

Expected output:
```
Device: cuda
Training grid: 16^7
GIFT 2.1 RG Flow Configuration:
  λ_max=39.44, Δα target=-0.9
  Components enabled: ∇·T=True, ∂ε g=True, fractality=True
  Adaptive geodesic frequency: True
RGFlowGIFT class ready
  Initial coefficients: A=-4.68, B=15.17, D=2.50
SmartEarlyStopping class ready
RGFlowMonitor class ready
```

---

## Training Progress Monitoring

### Real-Time (During Training)

Watch for:
```
[████████████░░░░░░░░░░░░░░░░░░] 40.0% | 
Epoch 600/2000 | 
Loss=0.363222 | 
||T||=1.6499e-02 (target=1.6400e-02) | 
RG=0.0000e+00 | 
ETA: 6.2h
```

### Phase Transitions

```
============================================================
PHASE 1: TCS_Neck
Torsion target: ||T|| = 0.001
============================================================
...
============================================================
PHASE 5: RG_Calibration
Torsion target: ||T|| = 0.0164
============================================================
```

### Calibration Event (Epoch 5000)

```
============================================================
CALIBRATING RG FLOW COEFFICIENTS (Epoch 5000)
  Updating RGFlowGIFT coefficients based on current trajectory
============================================================
  Coefficients: A=-4.68, B=15.17
============================================================
```

---

## Checking Results (After Training)

### 1. Load Validation Results

```python
import json

with open('G2_ML/1_1b/metadata.json', 'r') as f:
    results = json.load(f)

# Check key metrics
print(f"Torsion error: {results['torsion_validation']['error_percent']:.2f}%")
print(f"RG flow error: {results['rg_flow_validation']['error_percent']:.2f}%")
print(f"Geometry passed: {results['geometric_validation']['passed']}")
```

### 2. Analyze RG Flow Components

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load component log
rg_log = pd.read_csv('G2_ML/1_1b/rg_flow_log.csv')

# Plot evolution
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

rg_log.plot(x='epoch', y='A_div', ax=axes[0,0], title='A: Divergence')
rg_log.plot(x='epoch', y='B_norm', ax=axes[0,1], title='B: Norm')
rg_log.plot(x='epoch', y='C_eps', ax=axes[1,0], title='C: Epsilon')
rg_log.plot(x='epoch', y='D_frac', ax=axes[1,1], title='D: Fractality')

plt.tight_layout()
plt.savefig('rg_components.png')
plt.show()
```

### 3. Check Success Criteria

```python
# Success if:
torsion_ok = results['torsion_validation']['error_percent'] < 5.0
geometry_ok = results['geometric_validation']['passed']
rg_ok = results['rg_flow_validation']['error_percent'] < 30.0  # <20% target, <30% acceptable

if torsion_ok and geometry_ok and rg_ok:
    print("✓ ALL CRITERIA MET - SUCCESS!")
else:
    print("Some criteria not met, may need iteration")
    if not rg_ok:
        print(f"  RG flow error: {results['rg_flow_validation']['error_percent']:.1f}% (target < 30%)")
```

---

## Common Issues & Quick Fixes

### "CUDA out of memory"

```python
# Reduce batch size
CONFIG['batch_size'] = 512  # was 1024

# Or reduce grid
CONFIG['n_grid'] = 12  # was 16
```

### "Training too slow"

```python
# Increase checkpoint frequency (save less often)
CONFIG['checkpoint_freq'] = 1000  # was 500

# Reduce validation samples
CONFIG['yukawa_samples'] = 100000  # was 200000
```

### "RG flow not improving"

```python
# Increase RG flow weight in Phase 5
CONFIG['phases'][5]['weights']['rg_flow'] = 5.0  # was 3.0

# Extend Phase 5
CONFIG['n_epochs_per_phase'] = 2500  # was 2000
```

---

## Expected Timeline

### Phase-by-Phase

```
Phase 1 (TCS_Neck):          0.5-1 hour   [████████░░]
Phase 2 (ACyl_Matching):     1-1.5 hours  [██████████]
Phase 3 (Cohomology):        1.5-2 hours  [██████████]
Phase 4 (Harmonic):          2-3 hours    [██████████]
Phase 5 (RG_Calibration):    3-4 hours    [██████████]
Post-processing:             0.5 hour     [████░░░░░░]

Total: 8-12 hours on A100 GPU
```

### Milestones

- **Hour 1**: Phases 1-2 complete, geometry established
- **Hour 3**: Phase 3 complete, cohomology working
- **Hour 5**: Epoch 5000, RG flow calibration event
- **Hour 7**: Phase 4 complete, harmonics extracted
- **Hour 10**: Phase 5 complete, validation running
- **Hour 11**: All results saved, ready for analysis

---

## Resuming Interrupted Training

If training stops (disconnect, crash, etc.):

```python
# The notebook automatically resumes from latest checkpoint
# Just run: Cell → Run All

# Or manually:
checkpoint = torch.load('checkpoints_v1_1b/checkpoint_latest.pt')
phi_net.load_state_dict(checkpoint['model_state'])
optimizer.load_state_dict(checkpoint['optimizer_state'])
start_phase = checkpoint['phase']
start_epoch = checkpoint['epoch'] + 1

print(f"Resuming from Phase {start_phase}, Epoch {start_epoch}")
```

---

## Output Files

After training completes, you'll have:

```
G2_ML/1_1b/
├── checkpoints_v1_1b/
│   ├── checkpoint_latest.pt           (Latest model, ~3.5 MB)
│   ├── checkpoint_phase5_epoch_999.pt (Phase 5 final, ~3.5 MB)
│   └── ...
├── training_history.csv               (All epochs, ~2 MB)
├── rg_flow_log.csv                    (RG components, ~500 KB)
├── metadata.json                      (Results summary, ~50 KB)
├── harmonic_2forms.npy                (H² basis, ~20 MB)
├── harmonic_3forms.npy                (H³ basis, ~80 MB)
├── yukawa_tensor.npy                  (Couplings, ~2 MB)
├── phi_samples.npy                    (Field samples, ~50 MB)
└── metric_samples.npy                 (Geometry samples, ~50 MB)

Total: ~210 MB
```

---

## Next Steps After Training

1. **Verify Results**
   ```bash
   python test_gift21_components.py
   # Should see: ✓ All unit tests passed!
   ```

2. **Analyze Components**
   ```python
   # See "Checking Results" section above
   # Generate component plots
   # Verify contributions
   ```

3. **Compare with v1.1a**
   ```python
   # Load v1.1a results
   with open('../1_1a/metadata.json', 'r') as f:
       v11a = json.load(f)
   
   # Compare
   print(f"v1.1a RG error: {v11a['rg_flow_validation']['error_percent']:.1f}%")
   print(f"v1.1b RG error: {results['rg_flow_validation']['error_percent']:.1f}%")
   improvement = v11a['rg_flow_validation']['error_percent'] - results['rg_flow_validation']['error_percent']
   print(f"Improvement: {improvement:.1f} percentage points")
   ```

4. **Share Results**
   - Document final RG flow error
   - Plot component contributions
   - Note any unexpected behaviors
   - Compare with expected results in README

---

## Support

### Documentation

- **README_v1_1b.md**: Complete technical documentation
- **IMPLEMENTATION_SUMMARY.md**: Implementation details
- **QUICKSTART.md**: This file

### Files

- **K7_G2_TCS_RGFlow_v1_1b.ipynb**: Main notebook
- **test_gift21_components.py**: Unit tests

### Getting Help

If you encounter issues:

1. Check troubleshooting in README_v1_1b.md
2. Verify unit tests pass: `python test_gift21_components.py`
3. Review training log for NaN or unusual values
4. Check `rg_flow_log.csv` for component balance

---

## Summary: 3 Commands to Start

```bash
# 1. Navigate
cd G2_ML/1_1b

# 2. Verify
python test_gift21_components.py

# 3. Train
jupyter notebook K7_G2_TCS_RGFlow_v1_1b.ipynb
# Then: Cell → Run All
```

**That's it!** Training will run for 8-12 hours. Check back for results.

---

**Good luck with your GIFT 2.1 RG flow training!** 🚀

