# K₇ v1.0 Standalone Notebook

## Overview

**K7_v1_0_STANDALONE_FINAL.ipynb** is a completely self-contained, shareable Jupyter notebook for G₂ metric reconstruction on K₇ manifolds.

## Key Features

### 100% Self-Contained
- **No external files required** - all code is inline
- **No imports from local modules** - everything embedded in notebook
- All 4 modules (losses, training, validation, yukawa) included inline (~1440 lines)

### Local Storage Only
- Uses **`/content/K7_v1_0_training/`** (Colab local storage)
- **No Google Drive mounting** required
- You manage external backups yourself
- All checkpoints and results in `/content/`

### Robust Checkpoint System
- **Auto-resume** from latest checkpoint on restart
- **Atomic saves** to prevent corruption
- **Fallback recovery** if checkpoint is corrupted
- Keeps best 5 checkpoints by torsion metric
- Saves every 500 epochs

### Complete Implementation
- Torsion-free G₂ structure learning
- Harmonic form extraction (b₂=21, b₃=77)
- Yukawa coupling computation
- Five-phase curriculum training
- Adaptive loss scheduling

## Usage

### On Google Colab

1. **Upload notebook to Colab**
   - Go to https://colab.research.google.com
   - File → Upload notebook
   - Select `K7_v1_0_STANDALONE_FINAL.ipynb`

2. **Set GPU runtime**
   - Runtime → Change runtime type
   - Hardware accelerator: GPU (T4 or better)
   - Save

3. **Run all cells**
   - Runtime → Run all
   - Training starts automatically
   - Auto-resumes from checkpoint if restarting

4. **Download results before session ends**
   ```python
   # Run this in a cell to download checkpoint
   from google.colab import files
   files.download('/content/K7_v1_0_training/checkpoints/checkpoint_epoch_15000.pt')
   ```

### On Local Jupyter

```bash
jupyter notebook K7_v1_0_STANDALONE_FINAL.ipynb
```

Training will use `/content/K7_v1_0_training/` even locally (or modify WORK_DIR in cell 2).

## File Structure

### Notebook Cells

1. **Title and Introduction**
2. **Environment Setup** - Install packages, create directories
3. **Imports** - All Python dependencies
4. **Configuration** - Training parameters and GIFT constants
5. **Complete Implementation** - ALL MODULE CODE INLINE (1440+ lines)
   - Checkpoint management
   - Loss functions (losses.py)
   - Training loop and curriculum (training.py)
   - Geometric validation (validation.py)
   - Yukawa computation (yukawa.py)
6. **Training Execution** - Start/resume training

### Output Directory Structure

```
/content/K7_v1_0_training/
├── checkpoints/
│   ├── checkpoint_epoch_500.pt
│   ├── checkpoint_epoch_1000.pt
│   └── ... (best 5 kept)
├── results/
│   ├── training_history.npz
│   ├── final_validation.json
│   └── yukawa_tensor.npy
└── config.json
```

## Training Details

### Default Configuration

- **Total epochs**: 15,000
- **Batch size**: 2048
- **Learning rate**: 1×10⁻⁴
- **Checkpoint interval**: 500 epochs
- **Auto-resume**: Enabled

### Five Training Phases

1. **Phase 1 (0-2000)**: Neck stability
2. **Phase 2 (2000-5000)**: ACyl matching
3. **Phase 3 (5000-8000)**: Cohomology refinement
4. **Phase 4 (8000-10000)**: Harmonic extraction
5. **Phase 5 (10000-15000)**: Calibration fine-tuning

### Expected Results

- **Torsion**: < 0.1% (target)
- **Harmonic bases**: Full rank (21 and 77)
- **Yukawa deviation**: < 10% vs GIFT predictions
- **Training time**: 6-8 hours on A100, 12-15 hours on T4

## Checkpoint Management

### Auto-Resume Example

Session 1:
```
Epoch 0-2500 → crashes/disconnects
Checkpoint saved at epoch 2500
```

Session 2 (restart):
```
Auto-detects checkpoint_epoch_2500.pt
Resumes from epoch 2501
Continues to 5000
```

### Manual Checkpoint Loading

To resume from a specific checkpoint instead of latest:

```python
# In training execution cell, modify:
checkpoint = torch.load('/content/K7_v1_0_training/checkpoints/checkpoint_epoch_5000.pt')
```

### Download Checkpoints

Important: Download checkpoints periodically as Colab sessions expire!

```python
from google.colab import files

# Download best checkpoint
files.download('/content/K7_v1_0_training/checkpoints/checkpoint_epoch_10000.pt')

# Download results
files.download('/content/K7_v1_0_training/results/yukawa_tensor.npy')
```

## Sharing the Notebook

This notebook is designed to be **fully shareable**:

1. **Share the .ipynb file directly** - recipient can run immediately
2. **No setup required** - all dependencies installed automatically
3. **No external files needed** - completely self-contained
4. **Works on any Colab account** - no Drive access required

Simply send the `K7_v1_0_STANDALONE_FINAL.ipynb` file and the recipient can:
- Upload to their Colab
- Click Runtime → Run all
- Training begins automatically

## Customization

### Change Training Duration

Modify in config cell:
```python
'training': {'total_epochs': 10000, ...}  # Reduce to 10k
```

### Adjust Checkpoint Frequency

```python
'checkpointing': {'interval': 250, ...}  # Save every 250 epochs
```

### Change Batch Size (for GPU memory)

```python
'training': {'batch_size': 1024, ...}  # Reduce for smaller GPU
```

## Troubleshooting

### Out of Memory
- Reduce `batch_size` from 2048 to 1024 or 512
- Use smaller GPU runtime (T4 instead of trying L4)

### Training Slow
- Verify GPU is enabled (Runtime → Change runtime type)
- Check `DEVICE = cuda` in output
- Consider reducing to 10k epochs initially

### Checkpoint Not Loading
- The fallback system will try previous checkpoints automatically
- Worst case: training restarts from epoch 0 (not a data loss issue)

### Session Expired
- Colab free sessions last ~12 hours
- Download checkpoints periodically
- Resume from latest checkpoint in new session

## Advanced Usage

### Extract Specific Epoch Results

```python
# Load checkpoint
ckpt = torch.load('/content/K7_v1_0_training/checkpoints/checkpoint_epoch_10000.pt')

# Inspect metrics
print(ckpt['metrics'])
# {'torsion_closure': 0.0008, 'torsion_coclosure': 0.0009, ...}

# Load model weights
models = {...}  # Initialize models
for name, model in models.items():
    model.load_state_dict(ckpt['models'][name])
```

### Visualize Training Progress

```python
import numpy as np
import matplotlib.pyplot as plt

# Load history
history = np.load('/content/K7_v1_0_training/results/training_history.npz')

# Plot torsion evolution
epochs, torsion = history['torsion_closure']
plt.plot(epochs, torsion)
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Torsion Closure')
plt.title('Training Progress')
plt.show()
```

## Technical Details

### Inline Code Structure

The main implementation cell contains:

1. **Checkpoint Manager** (~50 lines)
   - Atomic save/load operations
   - Auto-resume logic
   - Fallback recovery

2. **Losses Module** (~320 lines)
   - Torsion closure/coclosure
   - Gram matrix orthonormalization
   - Calibration constraints
   - Adaptive loss scheduling

3. **Training Module** (~390 lines)
   - Curriculum scheduler
   - Training loop
   - Gradient accumulation
   - LR scheduling

4. **Validation Module** (~360 lines)
   - Ricci-flatness checks
   - Holonomy testing
   - Geometric validation

5. **Yukawa Module** (~420 lines)
   - Dual integration (MC + Grid)
   - Tucker decomposition
   - Mass ratio extraction

Total: ~1,540 lines of production-ready code

### Storage Requirements

- **Checkpoints**: ~50 MB each × 5 = ~250 MB
- **Results**: ~100 MB (Yukawa tensors, histories)
- **Total**: ~350-400 MB for complete training run

Colab provides sufficient storage for this.

## Version History

- **v1.0 Standalone** (2025-01-17)
  - Initial standalone release
  - All modules inline
  - Local storage only
  - Full checkpoint system

## License

MIT License - Same as GIFT Framework

## Citation

If using this implementation:

```bibtex
@software{gift_k7_v1_standalone,
  title={K₇ Metric Reconstruction v1.0: Self-Contained Training System},
  author={GIFT Framework Team},
  year={2025},
  url={https://github.com/gift-framework/GIFT/tree/main/G2_ML/1.0}
}
```

## Support

For questions or issues:
- GitHub Issues: https://github.com/gift-framework/GIFT/issues
- Main documentation: See GIFT repository README

---

**Ready to share and run anywhere with zero setup!**
