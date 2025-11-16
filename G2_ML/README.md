# G2 Machine Learning Framework

Neural network extraction of harmonic forms on compact G₂ manifolds for the GIFT framework.

## Quick Links

- **[STATUS.md](STATUS.md)** - Current implementation status, what works now, what's in progress
- **[COMPLETION_PLAN.md](COMPLETION_PLAN.md)** - Detailed technical plan for remaining work
- **Latest Version**: [0.9a/](0.9a/) - Production-ready b₂=21 implementation

## Overview

The G2_ML framework uses deep learning to extract harmonic forms from the compact 7-dimensional K₇ manifold with G₂ holonomy. These harmonic forms are essential for dimensional reduction in the GIFT theoretical framework.

**Target**: Extract complete harmonic basis:
- **b₂=21**: 21 harmonic 2-forms ✅ **Complete** (v0.7, v0.9a)
- **b₃=77**: 77 harmonic 3-forms 🔨 **In Progress** (v0.8 planned)

## Current Status

**Completion**: 90% (b₂ complete, b₃ in progress)

See **[STATUS.md](STATUS.md)** for detailed current status.

## Quick Start

### Run Latest Implementation (b₂=21)

```bash
cd 0.9a/
jupyter notebook Complete_G2_Metric_Training_v0_9a.ipynb
```

Or use stable production version:

```bash
cd 0.7/
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
├── STATUS.md                  # Current implementation status (read this first!)
├── COMPLETION_PLAN.md         # Detailed plan for remaining work
├── README.md                  # This file
│
├── 0.1/ through 0.6/         # Archived development versions
├── 0.7/                      # Production: b₂=21 complete ✅
├── 0.8/                      # Planned: b₃=77 extraction 🔨
├── 0.9/                      # Future: Yukawa tensors 📋
└── 0.9a/                     # Latest: b₂=21 with refinements ✅
```

Each version directory contains:
- Complete training notebook
- Python modules (geometry, manifold, networks, losses, training, evaluation)
- Results and validation data

## What You Can Do Now

With current implementation (v0.7, v0.9a):

✅ **Train harmonic 2-forms extraction**
```python
# See notebooks for complete examples
from G2_phi_network import PhiNetwork
from G2_train import train_harmonic_network

model = PhiNetwork(input_dim=7, hidden_dims=[384, 384, 256])
trained_model = train_harmonic_network(model, epochs=10000)
```

✅ **Validate G₂ geometry**
```python
from G2_eval import validate_harmonic_forms

results = validate_harmonic_forms(trained_model)
# Check: Gram matrix determinant ≈ 1.0
# Check: All eigenvalues > 0.5
```

✅ **Generate K₇ metrics**
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
- Gram matrix determinant: det(G) ∈ [0.9, 1.1] ✅
- Eigenvalue spectrum: all λ_i > 0.5 ✅
- Closedness: dω_i ≈ 0 ✅
- Coclosedness: δω_i ≈ 0 ✅

## Versions

| Version | Status | Features |
|---------|--------|----------|
| 0.1-0.6 | Archived | Development iterations |
| **0.7** | **Production** | **b₂=21 complete, validated** |
| 0.8 | Planned | b₃=77 extraction (in progress) |
| 0.9 | Future | Yukawa tensor computation |
| **0.9a** | **Latest** | **b₂=21 with improvements** |
| 1.0 | Target | Complete framework |

**Recommendation**: Use **v0.9a** for new work (latest refinements) or **v0.7** for stability.

## What's Next

See **[COMPLETION_PLAN.md](COMPLETION_PLAN.md)** for detailed plans:

1. **b₃=77 extraction** (v0.8) - $150-300, 1-2 days
2. **Yukawa tensors** (v0.9) - $40-60, 6-12 hours
3. **Architecture optimization** - $50-100, 1 day

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

**Technical details**: See [COMPLETION_PLAN.md](COMPLETION_PLAN.md)

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
**Latest version**: 0.9a
**Last updated**: 2025-11-16
**Framework**: GIFT v2.0.0
