# Complete G2 Metric Training v0.3 - Rigorous Implementation

## Overview

This directory contains the v0.3 implementation of G2 metric training on T^7, featuring mathematically rigorous differential geometry and A100 GPU optimization.

## Key Improvements Over v0.2

1. **Discrete Fourier Modes**: Integer frequency grid for TRUE T^7 periodicity (not random features)
2. **Rigorous Exterior Derivatives**: Proper antisymmetric structure with Levi-Civita symbols
3. **Dual Implementations**: Both exact (rigorous) and optimized (fast) exterior derivative methods
4. **Algebraic Metric Reconstruction**: Bryant's formula + standard G2 contraction methods
5. **Explicit b_2=21 Construction**: Network for 21 linearly independent harmonic 2-forms
6. **A100 Optimization**: Mixed precision (AMP), batch=2048, gradient accumulation, TF32

## Files

- `Complete_G2_Metric_Training_v0_3_Rigorous.ipynb` - Main training notebook
- `README.md` - This file

## Usage

### Google Colab (Recommended)

1. Upload the notebook to Google Colab
2. Select Runtime > Change runtime type > GPU (preferably A100)
3. Run all cells

The notebook will:
- Automatically install dependencies
- Detect GPU capabilities and optimize settings
- Train for 6000 epochs (~5 hours on A100)
- Save checkpoints every 500 epochs
- Generate comprehensive visualizations

### Local Execution

Requires:
- Python 3.8+
- PyTorch 2.0+ with CUDA
- GPU with 16+ GB VRAM (A100 recommended)

```bash
# Install dependencies
pip install torch torchvision matplotlib scipy numpy pandas

# Open notebook
jupyter notebook Complete_G2_Metric_Training_v0_3_Rigorous.ipynb
```

## Configuration Options

Key settings in the CONFIG dictionary:

### Exterior Derivative Mode
```python
'exterior_derivative_mode': 'optimized'  # or 'rigorous'
```
- **optimized**: Fast approximation (~1.5s per batch on A100)
- **rigorous**: Exact antisymmetric formula (~15s per batch on A100)

Recommendation: Start with 'optimized' for initial training, then optionally polish with 'rigorous'.

### Metric Reconstruction Mode
```python
'metric_reconstruction_mode': 'contraction'  # or 'bryant'
```
- **contraction**: Standard g(v,w) = (v⌟φ) ∧ (w⌟φ) ∧ φ formula
- **bryant**: Bryant's algebraic formula from G2 theory

Both are mathematically valid; contraction is more standard in literature.

### Training Parameters
```python
'epochs': 6000            # Total epochs
'batch_size': 2048        # 2048 for A100, 512 for other GPUs
'lr': 1e-4                # Learning rate
'max_frequency': 8        # Discrete Fourier modes cutoff
```

## Outputs

After training, the following files are saved to `g2_v03_outputs/`:

- `final_model.pt` - Final trained model
- `best_model.pt` - Best model (lowest loss)
- `checkpoint_epoch_*.pt` - Checkpoints every 500 epochs
- `training_history.csv` - All loss components over time
- `validation_results.json` - Validation metrics
- `config.json` - Training configuration
- `final_analysis.png` - Comprehensive visualization

## Validation Criteria

The trained G2 structure is validated against:

1. **Torsion-free**: ||d(φ)|| < 1e-4 and ||d(*φ)|| < 1e-4
2. **Volume normalization**: |det(g) - 1| < 1e-3
3. **Harmonic orthonormality**: ||G - I|| < 0.1
4. **Linear independence**: det(G) > 0.5
5. **Positive definite**: All eigenvalues λ_i > 1e-6
6. **Periodicity**: ||φ(x) - φ(x+2π)|| < 1e-3

## Mathematical Background

### G2 Structures

A G2 structure on a 7-manifold is specified by a 3-form φ. The structure is torsion-free if:
- d(φ) = 0 (exterior derivative vanishes)
- d(*φ) = 0 (codifferential vanishes)

The metric g can be algebraically reconstructed from φ.

### Discrete Fourier Periodicity

For T^7 = (S^1)^7, we use:
```
φ(x) = Σ [a_n cos(n·x) + b_n sin(n·x)]
```
where n = (n_1,...,n_7) are integer frequency vectors with ||n||^2 ≤ max_freq^2.

This ensures φ(x + 2πe_i) = φ(x) exactly for all i.

### Harmonic 2-Forms

The K7 manifold (twisted connected sum) has b_2 = 21, meaning 21 linearly independent harmonic 2-forms. We explicitly construct these with a dedicated network and enforce:
- Orthonormality: ∫ ω_α ∧ *ω_β = δ_αβ
- Linear independence: det(Gram) ≈ 1

## Performance Estimates

| GPU | Batch Size | Time per Epoch | Total Time (6000 epochs) |
|-----|------------|----------------|--------------------------|
| A100 (optimized) | 2048 | 1.5s | ~2.5 hours |
| A100 (rigorous) | 2048 | 15s | ~25 hours |
| V100 | 512 | 3s | ~5 hours |
| T4 | 512 | 6s | ~10 hours |

## Troubleshooting

### Out of Memory
- Reduce `batch_size` (try 512, 256, or 128)
- Reduce `max_frequency` (try 6 or 4)
- Ensure mixed precision is enabled: `'use_amp': True`

### Slow Training
- Ensure GPU is being used: check device detection in first cell
- Use 'optimized' exterior derivative mode
- Reduce checkpoint/plot intervals

### Poor Convergence
- Check validation: some tests may fail early but pass later
- Increase epochs (try 8000-10000)
- Adjust loss weights in curriculum learning
- Try different random seed

## Scientific Context

This implementation supports numerical exploration of G2 metrics on compact 7-manifolds in the context of Geometric Information Field Theory (GIFT). The discrete Fourier approach ensures mathematical rigor while maintaining computational tractability.

For theoretical background, see:
- Bryant, R. L. (1987). Metrics with exceptional holonomy
- Joyce, D. D. (2000). Compact manifolds with special holonomy
- Corti, A. et al. (2013). G2-manifolds and associative submanifolds

## Contact

For questions about GIFT theory, see the main GIFT documentation in `publication/gift_main.md`.



