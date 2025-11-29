# G2_ML v1.9 - Hodge Pure

**Goal**: Learn true harmonic forms H2 and H3 on the fixed v1.8 metric, then compute proper Yukawa tensor.

## Motivation

The spectral analysis from v1.8 showed:
- det(g) = 2.03125 EXACT
- kappa_T = 0.0163 (0.59% error)
- A significant gap at position 42->43 in the Yukawa Gram matrix

However, tau = 3472/891 did not emerge because we used proxy constructions instead of true harmonic forms.

## Architecture

### Phase 1: H2 Training (21 Harmonic 2-Forms)

On the **frozen** v1.8 metric, learn:

- **H2 modes (21)**: Harmonic 2-forms omega_i satisfying
  - d(omega) = 0 (closed)
  - d*(omega) = 0 (co-closed)
  - Gram(omega) ~ I_21 (orthonormal)

### Phase 2: H3 Training (77 Harmonic 3-Forms)

- **H3 modes (77 = 35 local + 42 global)**:
  - d(Phi) = 0 (closed)
  - d*(Phi) = 0 (co-closed)
  - Gram(Phi) ~ I_77 (orthonormal)
  - G2 compatibility for local modes

### Phase 3: Yukawa Computation

With proper harmonic bases:

```
Y_ijk = integral_{K7} omega_i ^ omega_j ^ Phi_k
```

Compute Gram matrix M = Y Y^T and check:
- Does rank = 43?
- Does the eigenvalue ratio give tau = 3472/891?

## Files

| File | Description |
|------|-------------|
| `config.json` | Training configuration |
| `hodge_forms.py` | H2Network, H3Network, loss functions |
| `exterior_calculus.py` | d, d*, Hodge star, wedge product operators |
| `yukawa_integral.py` | Yukawa tensor computation and analysis |
| `train_hodge.py` | CLI training script |
| `K7_Hodge_Pure_v1_9.ipynb` | **All-in-one Colab notebook** |

## Colab Notebook Features

The `K7_Hodge_Pure_v1_9.ipynb` notebook includes:

- **Auto-resume**: Checkpoints saved every 500 epochs, automatic resume on restart
- **Live visualization**: Training progress plots update in real-time
- **Multi-format output**:
  - `yukawa.npz`: Yukawa tensor, Gram matrix, eigenvalues
  - `models.pt`: Trained H2 and H3 model weights
  - `final_metrics.json`: Analysis summary
  - `eigenvalues.csv`: Eigenvalue table
  - `samples.npz`: Sample evaluations for further analysis
  - `yukawa_spectrum.png`: Visualization

## Usage

### Colab (Recommended)

1. Upload `K7_Hodge_Pure_v1_9.ipynb` to Colab
2. Upload `samples.npz` from v1.8 (or let it generate synthetic data)
3. Run all cells
4. Download outputs from `outputs_v1_9/`

### Local

```bash
# Full training
python train_hodge.py

# Phase by phase
python train_hodge.py --phase h2
python train_hodge.py --phase h3
python train_hodge.py --phase yukawa
```

## Expected Results

If the theory is correct:
- The Yukawa Gram matrix should have rank 43
- The ratio sum(lambda_visible) / sum(lambda_hidden) should approach 3472/891
- The 43/77 split becomes a geometric invariant, not imposed by hand

## Dependencies

- PyTorch
- NumPy, Matplotlib
- v1.8 trained metric (`../1_8/samples.npz`)

## Status

**READY FOR TRAINING**

- Exterior calculus operators implemented (d, d*, Hodge star, wedge)
- H2 and H3 networks with orthonormality constraints
- Yukawa integral via Monte Carlo wedge product
- Checkpointing and resume
- Multi-format output export
