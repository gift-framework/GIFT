# G2 Machine Learning Framework

Neural network extraction of harmonic forms on compact G₂ manifolds for the GIFT framework.

## Quick Links

- **[STATUS.md](STATUS.md)** - Current implementation status, what works now, what's in progress
- **[FUTURE_WORK.md](FUTURE_WORK.md)** - Detailed technical plan for remaining work
- **Latest Version**: [2_1/](2_1/) - Current development version
- **Archived Versions**: [archived/](archived/) - Historical development versions

## Overview

The G2_ML framework uses deep learning to extract harmonic forms from the compact 7-dimensional K₇ manifold with G₂ holonomy. These harmonic forms are essential for dimensional reduction in the GIFT theoretical framework.

**Target**: Extract complete harmonic basis:
- **b₂=21**: 21 harmonic 2-forms - Complete (v0.7, v0.9a)
- **b₃=77**: 77 harmonic 3-forms - In Progress (v0.8 planned)

## Current Status

**Completion**: 90% (b₂ complete, b₃ in progress)

See **[STATUS.md](STATUS.md)** for detailed current status.

## Quick Start

### Run Latest Implementation

Check the latest version directories (1_9b/, 2_0/, 2_1/) for current implementations.

For archived stable versions:

```bash
cd archived/early_development/0.7/
jupyter notebook Complete_G2_Metric_Training_v0_7.ipynb
```

### Requirements

```bash
pip install -r ../requirements.txt
```

GPU recommended (training takes 6-8 hours on A100, much longer on CPU).

## Directory Structure

```
G2_ML/
├── README.md                  # This file
├── STATUS.md                  # Current implementation status (read this first!)
├── FUTURE_WORK.md             # Detailed plan for remaining work
├── VERSIONS.md                # Version history and changelog
│
├── 1_9b/                      # Version 1.9b
├── 2_0/                       # Version 2.0
├── 2_1/                       # Version 2.1 (latest)
│
├── archived/                  # Historical versions
│   ├── early_development/    # Versions 0.1 through 0.9
│   └── v1_iterations/        # Versions 1.0 through 1.x
│
├── research_modules/          # Specialized research modules
│   ├── meta_hodge/           # Hodge theory implementations
│   ├── tcs_joyce/            # Joyce's construction methods
│   └── variational_g2/       # Variational approaches
│
└── tests/                     # Test suite

**Note**: Lean formal verification is now in `/Lean/` at repository root (17 modules, 0 sorry).
```

Each version directory contains notebooks, models, and validation data specific to that version.

## What You Can Do Now

With current implementation (v0.7, v0.9a):

- Train harmonic 2-forms extraction
```python
# See notebooks for complete examples
from G2_phi_network import PhiNetwork
from G2_train import train_harmonic_network

model = PhiNetwork(input_dim=7, hidden_dims=[384, 384, 256])
trained_model = train_harmonic_network(model, epochs=10000)
```

- Validate G₂ geometry
```python
from G2_eval import validate_harmonic_forms

results = validate_harmonic_forms(trained_model)
# Check: Gram matrix determinant ≈ 1.0
# Check: All eigenvalues > 0.5
```

- Generate K₇ metrics
```python
from G2_manifold import K7Manifold

manifold = K7Manifold(trained_model)
metric = manifold.compute_metric(point)
```

## Architecture

### Neural Networks

**PhiNetwork**: Learns the φ function defining K₇ metric
- Input: 7D coordinates on K₇
- Architecture: [384, 384, 256] (configurable)
- Output: Metric components

**HarmonicNetwork**: Extracts harmonic forms from metric
- Input: Metric from PhiNetwork
- Output: 21 harmonic 2-forms (b₂=21)
- Loss: Orthonormality + Closedness + Coclosedness

### Training

Curriculum learning schedule:
1. **Phase 1**: Orthonormality (epochs 0-3000)
2. **Phase 2**: Add closedness (epochs 3000-6000)
3. **Phase 3**: Add coclosedness (epochs 6000-10000)
4. **Phase 4**: Full loss refinement (epochs 10000+)

Training time: ~6-8 hours on A100 GPU

### Validation

All trained models validated via:
- Gram matrix determinant: det(G) ∈ [0.9, 1.1] (verified)
- Eigenvalue spectrum: all λ_i > 0.5 (verified)
- Closedness: dω_i ≈ 0 (verified)
- Coclosedness: δω_i ≈ 0 (verified)

## Versions

| Version | Status | Location | Features |
|---------|--------|----------|----------|
| 0.1-0.9 | Archived | archived/early_development/ | Early development iterations |
| 1.0-1.x | Archived | archived/v1_iterations/ | Version 1 iterations |
| **1.9b** | Stable | 1_9b/ | Stable version |
| **2.0** | Stable | 2_0/ | Version 2.0 |
| **2.1** | **Current** | 2_1/ | **Latest development version** |

**Recommendation**: Use **2.1/** for current work. See [VERSIONS.md](VERSIONS.md) for detailed version history.

## What's Next

See **[FUTURE_WORK.md](FUTURE_WORK.md)** for detailed plans on ongoing research directions.

For specialized research modules, see [research_modules/README.md](research_modules/README.md).

## Scientific Context

### Role in GIFT Framework

Harmonic forms on K₇ are essential for:
- **Dimensional reduction**: 496D → 99D → 4D
- **Gauge coupling unification**: Related to b₂=21 moduli
- **Yukawa couplings**: Triple products determine fermion masses
- **CP violation**: Topological phases from 3-forms

### Publications

Results from this framework appear in:
- GIFT v2 Supplement F: K₇ Metric Construction
- Statistical validation notebooks
- Experimental predictions

Standalone publication planned after b₃ completion.

## Technical Details

### G₂ Holonomy Manifolds

K₇ is a compact 7-dimensional Riemannian manifold with:
- **Holonomy group**: G₂ ⊂ SO(7)
- **Defining 3-form**: φ (parallel under ∇φ = 0)
- **Hodge dual**: ⋆φ (4-form)
- **Betti numbers**: b₂(K₇) = 21, b₃(K₇) = 77

### Harmonic Forms

Forms ω satisfying:
- **Closedness**: dω = 0 (exact forms modulo boundaries)
- **Coclosedness**: δω = d⋆ω = 0 (divergence-free)
- **Harmonic**: Δω = (dδ + δd)ω = 0

Harmonic forms are:
- Topologically non-trivial
- Orthogonal under L² inner product
- Basis for cohomology H^p(K₇)

### Loss Functions

```python
L_total = w1·L_orthonormality + w2·L_closedness + w3·L_coclosedness

L_orthonormality = ||G - I||²  # Gram matrix ≈ identity
L_closedness = ||dω_i||²       # Exterior derivative ≈ 0
L_coclosedness = ||δω_i||²     # Codifferential ≈ 0
```

Weights w1, w2, w3 vary during curriculum learning.

## Performance

### b₂=21 Extraction (v0.9a)

- **Training time**: 6-8 hours (A100 GPU)
- **Final loss**: ~1e-4
- **Gram determinant**: 0.98-1.02 (excellent)
- **Eigenvalues**: All in [0.85, 1.15] (acceptable)
- **Success rate**: >90% (most training runs converge)

### Computational Requirements

- **GPU memory**: 16 GB minimum (24 GB recommended)
- **Training samples**: 100K-1M K₇ points per epoch
- **Batch size**: 2048-4096
- **Learning rate**: 1e-4 (with cosine annealing)

## Code Quality

- **Modular design**: Separate files for geometry, networks, training, evaluation
- **Type hints**: All functions annotated
- **Docstrings**: Complete documentation
- **Notebooks**: Executable demonstrations
- **Validation**: Comprehensive checks included

## Limitations

### Current

- Only b₂=21 complete (b₃=77 in progress)
- No Yukawa tensor computation yet
- Architecture not fully optimized
- Numerical approximation (not exact forms)

### Theoretical

- Metric ansatz dependence
- Indirect validation (no analytical comparison available)
- Assumes smooth K₇ (no singularities)

## Support

**Quick questions**: See [STATUS.md](STATUS.md) first

**Technical details**: See [FUTURE_WORK.md](FUTURE_WORK.md)

**Code issues**: Each version directory has inline documentation

**GIFT framework**: See main repository [README.md](../README.md)

## License

MIT License (same as GIFT framework)

## Citation

When using this framework:

```bibtex
@software{gift_g2ml_2025,
  title={G2 Machine Learning Framework: Neural Network Extraction of Harmonic Forms},
  author={{GIFT Framework Team}},
  year={2025},
  url={https://github.com/gift-framework/GIFT/tree/main/G2_ML},
  note={Version 0.9a, b₂=21 complete}
}
```

---

**Status**: Active development (90% complete)
**Latest version**: 2.1
**Last updated**: 2025-12-03
**Framework**: GIFT v2.3.1
**Lean Proofs**: See `/Lean/` for formal verification
