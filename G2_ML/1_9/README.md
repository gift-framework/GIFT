# G2_ML v1.9 - Hodge Pure

**Goal**: Learn true harmonic forms H² and H³ on the fixed v1.8 metric, then compute proper Yukawa.

## Motivation

The spectral analysis from v1.8 showed:
- det(g) = 2.03125 EXACT
- kappa_T = 0.0163 (0.59% error)
- A significant gap at position 42→43 in the Yukawa Gram matrix

However, tau = 3472/891 did not emerge because we used proxy constructions instead of true harmonic forms.

## Approach

### Phase 1: Hodge Training

On the **frozen** v1.8 metric, learn:

1. **H² modes (21)**: Harmonic 2-forms ω_i satisfying
   - dω = 0 (closed)
   - d*ω = 0 (co-closed)
   - Gram(ω) ≈ I₂₁ (orthonormal)

2. **H³ modes (77)**: Harmonic 3-forms Φ_k satisfying
   - dΦ = 0 (closed)
   - d*Φ = 0 (co-closed)
   - Gram(Φ) ≈ I₇₇ (orthonormal)
   - G₂ compatibility (local modes ≈ phi structure)

### Phase 2: Yukawa Computation

With proper harmonic bases:

```
Y_ijk = ∫_{K7} ω_i ∧ ω_j ∧ Φ_k
```

### Phase 3: Spectral Verification

Compute Gram matrix M = Y Y^T and check:
- Does rank = 43?
- Does the eigenvalue ratio give tau = 3472/891?

## Files

- `config.json`: Configuration
- `hodge_forms.py`: Network architectures for H² and H³
- `train_hodge.py`: Training script

## Usage

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
- The ratio sum(λ_visible) / sum(λ_hidden) should approach 3472/891 ≈ 3.897
- The 43/77 split becomes a geometric invariant, not imposed by hand

## Dependencies

- v1.8 trained metric (from `../1_8/samples.npz`)
- PyTorch

## Status

**Scaffold ready** - Needs proper exterior calculus implementation for dω and d*ω.
