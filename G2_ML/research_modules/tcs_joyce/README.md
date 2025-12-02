# TCS/Joyce Global Mode Construction for K7 Manifolds

**Version 2.0.0** - GIFT Framework

## Overview

This package implements geometrically meaningful global H3 modes for K7 manifolds based on Twisted Connected Sum (TCS) and Joyce G2 construction principles.

The key insight is that the 77 harmonic 3-forms on K7 decompose as:

```
b3(K7) = 77 = 35 (local) + 42 (global)
```

where:
- **35 local modes**: Fiber direction Lambda^3(R^7) - constant G2 backbone
- **42 global modes**: Base direction - TCS gluing/profile functions

## Motivation

In v1.9b, the 42 global modes were constructed using artificial polynomial and trigonometric functions. While this allowed the pipeline to run, the resulting Yukawa spectrum showed only 35 active modes (the local ones), with the 42 global modes contributing little.

The TCS/Joyce construction replaces this with geometrically meaningful modes:

1. **14 left-weighted modes**: Forms concentrated in the left CY3 region
2. **14 right-weighted modes**: Forms concentrated in the right CY3 region
3. **14 neck-coupled modes**: Forms localized around the gluing neck

## Module Structure

```
tcs_joyce/
    __init__.py           # Package exports
    profiles.py           # Smooth profile functions (left/right/neck)
    basis_forms.py        # Canonical G2 3-form templates
    tcs_global_modes.py   # Main global mode builder
    config.py             # Configuration options
    README.md             # This file
```

## Usage

### Basic Usage

```python
import torch
from G2_ML.tcs_joyce import build_tcs_global_modes

# Sample coordinates on K7
coords = torch.rand(1000, 7)

# Build 42 global modes using TCS construction
global_modes = build_tcs_global_modes(coords)
print(global_modes.shape)  # (1000, 42)
```

### With CandidateLibrary

```python
from G2_ML.meta_hodge.candidate_library import (
    CandidateLibrary,
    GlobalModeStrategy
)

# Create library with TCS mode construction
library = CandidateLibrary(
    global_mode_strategy=GlobalModeStrategy.TCS_JOYCE
)

# Build all 77 H3 modes
phi = ...  # Your phi values, shape (N, 35)
coords = torch.rand(N, 7)
h3_modes = library.collect_b3_77(phi, coords)
print(h3_modes.shape)  # (N, 77)
```

### Configuration

```python
from G2_ML.tcs_joyce import TCSJoyceConfig, GlobalModeConstruction

# Create custom configuration
config = TCSJoyceConfig(
    construction=GlobalModeConstruction.TCS_JOYCE,
    domain=(0.0, 1.0),
    neck_fraction=0.5,      # Neck at center of domain
    transition_width=0.15,   # Width of profile transitions
    n_left=14,
    n_right=14,
    n_neck=14,
    orthonormalize=True,
    include_xi_weighting=True,
)

# Use with mode builder
from G2_ML.tcs_joyce import get_global_mode_builder
builder = get_global_mode_builder(config)
global_modes = builder(coords)
```

### Comparing Constructions

```python
from G2_ML.tcs_joyce.tcs_global_modes import compare_mode_constructions

# Compare TCS vs legacy modes
comparison = compare_mode_constructions(coords)
print("TCS modes:", comparison["tcs"].shape)
print("Legacy modes:", comparison["legacy"].shape)
print("Correlation:", comparison["correlation"].shape)
```

## Profile Functions

The TCS geometry is encoded via smooth profile functions:

### Left Plateau
```python
from G2_ML.tcs_joyce import left_plateau

# ~1 on left (CY3_L), ~0 on right
f_L = left_plateau(lambda_coord, lambda_L=0.0, lambda_neck=0.5, sigma=0.1)
```

### Right Plateau
```python
from G2_ML.tcs_joyce import right_plateau

# ~0 on left, ~1 on right (CY3_R)
f_R = right_plateau(lambda_coord, lambda_R=1.0, lambda_neck=0.5, sigma=0.1)
```

### Neck Bump
```python
from G2_ML.tcs_joyce import neck_bump

# Localized around neck, vanishing at ends
g_neck = neck_bump(lambda_coord, lambda_neck=0.5, sigma=0.1)
```

## Expected Improvements

With TCS/Joyce modes, we expect:

1. **Better spectral structure**: The Yukawa Gram matrix eigenspectrum should show:
   - ~43 significant eigenvalues (visible sector)
   - ~34 near-zero eigenvalues (hidden sector)

2. **Cleaner 43/77 gap**: The eigenvalue gap at position 43 should be more pronounced

3. **Emergent tau structure**: The ratio of visible/hidden eigenvalue sums may approach tau = 3472/891

## Theory Background

### TCS Construction

Joyce and Kovalev showed that compact G2 manifolds can be constructed as:

```
K7 = (S^1 x CY3_L) cup_neck (S^1 x CY3_R)
```

where two asymptotically cylindrical Calabi-Yau 3-folds are glued along a common neck region.

### Mode Decomposition

The cohomology of K7 decomposes as:
- H^2(K7): Forms from the base CY3 structure (b2 = 21)
- H^3(K7): Forms from Lambda^3 plus TCS gluing data (b3 = 77)

The 77 = 35 + 42 split reflects:
- 35 = C(7,3): Local fiber forms (constant over manifold)
- 42 = 14 + 14 + 14: Left/Right/Neck gluing contributions

## References

1. Joyce, D. "Compact Riemannian 7-manifolds with holonomy G2" (1996)
2. Kovalev, A. "Twisted connected sums and special Riemannian holonomy" (2003)
3. Corti et al. "G2-manifolds and associative submanifolds via semi-Fano 3-folds" (2015)

## Version History

- **v2.0.0** (2024): Initial TCS/Joyce implementation
- Replaces legacy polynomial/trig construction from v1.9b
- Integrated with meta_hodge pipeline

## See Also

- `G2_ML/meta_hodge/PHI_ANALYTICAL_STRUCTURE.md`: Analysis of phi decomposition
- `G2_ML/meta_hodge/K7_GIFT_ATLAS.md`: K7 manifold overview
- `G2_ML/1_9b/README.md`: Previous version with legacy modes
