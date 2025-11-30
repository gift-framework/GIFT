# G2 Machine Learning Framework - Version Index

**Framework Completion**: 93%
**Latest Production**: v0.9a (b₂=21), v0.8 (Yukawa)
**In Progress**: v0.9b (full b₃=77), variational_g2 (PINN approach)
**Last Updated**: 2025-11-30

## Quick Version Guide

| Version | Status | Key Features | Use Case |
|---------|--------|--------------|----------|
| **variational_g2** | 🔨 WIP | PINN variational approach | **New paradigm** |
| **2_1** | 🔨 WIP | Constraints-first formulation | New approach |
| **2_0** | 🔨 WIP | Weighted Yukawa analysis | TCS exploration |
| **0.9b** | 🔨 Training | Full b₃=77 extraction | Next milestone |
| **0.9a** | ✅ Production | b₂=21 latest refinements | **Recommended for b₂** |
| **0.8** | ✅ Complete | Yukawa + partial b₃ (20/77) | **Yukawa analysis** |
| **0.7** | ✅ Production | b₂=21 stable | Alternative to 0.9a |
| 1.x series | ✅/⚠️ | Extended exploration | Milestones kept |
| 0.1-0.6c | ⚠️ Archived | Early development | See archived/ |

## Folder Structure

```
G2_ML/
├── Production (v0.x)
│   ├── 0.7/          b₂=21 first stable
│   ├── 0.8/          Yukawa computed
│   ├── 0.9a/         Latest production
│   └── 0.9b/         Full b₃=77 (in progress)
├── Extended (v1.x milestones)
│   ├── 1.0f, 1_1c, 1_2c, 1_3c/   Series finals
│   ├── 1_4 - 1_8/                Exploration
│   └── 1_9b/                     Fixed hodge
├── New Paradigm (v2.x)
│   ├── 2_0/          Weighted Yukawa
│   └── 2_1/          Variational formulation
├── Specialized Modules
│   ├── variational_g2/   PINN-based metric extraction
│   ├── meta_hodge/       Historical data mining
│   └── tcs_joyce/        TCS global modes
└── archived/
    ├── early_development/   0.1-0.6c, 0.9
    └── v1_iterations/       Intermediate 1.x versions
```

---

## Version Details

### v0.9b - Full b₃=77 Extraction (In Progress) 🔨

**Status**: Training currently running
**Date**: 2025-11
**Focus**: Complete b₃=77 harmonic 3-forms extraction

**Expected Deliverables**:
- Full 77 harmonic 3-forms basis
- Gram matrix det(G) ∈ [0.9, 1.1]
- All 77 eigenvalues λ_i > 0.5
- Complete harmonic basis for H³(K₇)

**Timeline**: Completion expected soon

**Scientific Impact**: Enables publication "Complete Harmonic Form Basis from Machine Learning"

---

### v0.9a - Latest Production (b₂=21) ✅

**Status**: ✅ Production Ready
**Date**: 2025-11
**Focus**: Latest refinements for b₂=21 extraction

**Features**:
- ✅ All b₂=21 features from v0.7
- ✅ Code improvements and optimizations
- ✅ Enhanced documentation
- ✅ Better modularity

**Key Files**:
- `Complete_G2_Metric_Training_v0_9a.ipynb` - Full pipeline
- `README.md` - Version documentation
- Python modules (G2_*.py)

**Validation**:
- Gram matrix det ≈ 1.0 ✓
- Eigenvalues in [0.9, 1.1] ✓
- Training success rate >90% ✓

**Recommended For**:
- New b₂=21 work
- Production applications
- Publication-quality results

**Publications**: "Neural Network Extraction of Harmonic 2-Forms on G₂ Manifolds"

---

### v0.8 - Yukawa + Partial b₃ ✅

**Status**: ✅ Complete
**Date**: 2025-11
**Focus**: Yukawa coupling tensor + partial b₃ extraction

**Achievements**:
- ✅ **Yukawa tensor computed**: Complete 21×21×21 triple products
- ✅ **Partial b₃**: 20/77 harmonic 3-forms (26%)
- ✅ Hitchin metric construction
- ✅ Torsion decomposition (τ₀, τ₁, τ₂, τ₃)
- ✅ Final torsion: 0.000146 (excellent)

**Key Files**:
- `Complete_G2_Metric_Training_v0_8b.ipynb`
- `yukawa_couplings.json` (19KB) - Complete Yukawa data
- `summary.json` - Training summary
- `training_history.csv` - Metrics over time
- `training_results.png` - Visualization

**Yukawa Tensor Details**:
- 9,261 triple wedge products computed
- Y_αβγ = ∫_{K₇} ω_α ∧ ω_β ∧ ω_γ
- Values: ~1e-5 to ~1e-4
- Multiplicity structure preserved

**Recommended For**:
- Yukawa coupling analysis
- Fermion mass hierarchy studies
- Phenomenological research

**Publications**: "Yukawa Couplings from Compact G₂ Geometry"

---

### v0.7 - First Production Release (b₂=21) ✅

**Status**: ✅ Production Ready (Stable)
**Date**: 2025-11
**Focus**: First production-ready b₂=21 implementation

**Features**:
- ✅ Complete b₂=21 harmonic 2-forms extraction
- ✅ Validated Gram matrix
- ✅ Reliable training convergence (90%+ success)
- ✅ Complete documentation

**Key Files**:
- `Complete_G2_Metric_Training_v0_7.ipynb`
- Python modules for reuse

**Validation**:
- det(Gram) ≈ 1.0 ✓
- All eigenvalues in [0.9, 1.1] ✓

**Recommended For**:
- Stable b₂=21 work
- When preferring proven stability over latest features
- Alternative to v0.9a

**Note**: v0.9a is recommended for new work, but v0.7 remains a stable fallback

---

### Archived Versions (0.1-0.6c, 0.9) ⚠️

**Location**: `archived/early_development/`

Early development versions preserved for historical reference.
See `archived/README.md` for details.

| Version | Focus | Key Innovation |
|---------|-------|----------------|
| 0.1 | Initial prototype | Proof of feasibility |
| 0.2 | Architecture | 2x faster convergence |
| 0.3 | Loss functions | Improved training |
| 0.4 | Curriculum learning | 70% -> 85% success |
| 0.5 | b₃ exploration | Preliminary investigation |
| 0.6-0.6c | Validation | Gram matrix analysis |
| 0.9 | Refinement | Superseded by 0.9a |

**Migration**: Use v0.7 or v0.9a for production work.

---

### New Paradigm: variational_g2 🔨

**Status**: Work in Progress
**Location**: `variational_g2/`
**Approach**: Physics-Informed Neural Network (PINN)

**Key Shift**: Constraints as PRIMARY inputs, metric as EMERGENT output.
Does NOT assume TCS/Joyce - lets geometry emerge from GIFT constraints.

**Constraints enforced**:
- det(g) = 65/32 (GIFT topological)
- kappa_T = 1/61 (torsion magnitude)
- b₂ = 21, b₃ = 77 (cohomology)
- Metric positivity

**Training phases**:
1. Initialization (warm start)
2. Constraint enforcement
3. Torsion minimization
4. Refinement

**Output**: `outputs/rigorous_certificate.json`
- det(g) verified to 1.5e-13% relative error
- Status: NUMERICALLY_PROMISING

---

### New Paradigm: meta_hodge

**Location**: `meta_hodge/`
**Purpose**: Historical data mining from all versions

Aggregates learned metrics from v0.1 through v1.9b to:
- Build candidate library
- Run unified Hodge analysis
- Extract Yukawa couplings across versions
- Analyze stability patterns

**Key outputs**: K7_GIFT_ATLAS.md, K7_DEFORMATION_ATLAS.md

---

### New Paradigm: tcs_joyce

**Location**: `tcs_joyce/`
**Purpose**: Geometrically-motivated TCS global modes

Replaces artificial polynomial/trig modes with proper TCS construction:
- 42 global modes = 14 left + 14 right + 14 neck
- Profile functions for CY3 regions
- Expected: better eigenvalue gap, 43/77 structure

---

## Choosing a Version

### For Production Work

**b₂=21 harmonic 2-forms**:
- ✅ **Primary**: v0.9a (latest production)
- ✅ **Alternative**: v0.7 (stable, proven)

**Yukawa coupling analysis**:
- ✅ **Use**: v0.8 (only version with Yukawa computed)

**b₃=77 harmonic 3-forms**:
- 🔨 **Wait for**: v0.9b (currently training)
- ⚠️ **Partial only**: v0.8 (20/77 forms)

### For Historical Research

All archived versions (0.1-0.6c) available for historical reference. Each contains README.md with migration guidance.

### For Development

- **Latest code**: v0.9a
- **Bleeding edge**: v0.9b (when complete)

---

## Version Progression Timeline

```
2025-09: 0.1 → 0.2 → 0.3  (Architecture development)        [archived]
2025-10: 0.4 → 0.5 → 0.6  (Curriculum learning, validation) [archived]
2025-11: 0.6b → 0.6c → 0.7 ✅ (Production milestone)
2025-11: 0.8 ✅  (Yukawa + partial b₃)
2025-11: 0.9a ✅ (Latest production)
2025-11: 0.9b 🔨 (Full b₃=77 in progress)
2025-11: 1.x series (Extended exploration, milestones kept)
2025-11: 2.x + variational_g2 🔨 (New paradigm - constraints first)
```

---

## Scientific Output by Version

| Version | Publication Status | Title |
|---------|-------------------|-------|
| 0.7, 0.9a | ✅ Ready | "Neural Network Extraction of Harmonic 2-Forms on G₂ Manifolds" |
| 0.8 | ✅ Data Ready | "Yukawa Couplings from Compact G₂ Geometry" |
| 0.9b | 🔨 Awaiting | "Complete Harmonic Form Basis from Machine Learning" |
| 1.0 | 📋 Future | Complete framework methodology paper |

---

## Version Support Policy

**Production Versions** (✅):
- v0.9a, v0.8, v0.7: Fully supported, maintained
- Recommended for all new work

**In Progress** (🔨):
- v0.9b: Active development

**Archived Versions** (⚠️):
- v0.1-0.6c, 0.9: Located in `archived/early_development/`
- v1.x iterations: Located in `archived/v1_iterations/`
- See `archived/README.md` for full inventory

---

## Related Documentation

- **[STATUS.md](STATUS.md)** - Current implementation status (detailed)
- **[README.md](README.md)** - Framework overview
- **[FUTURE_WORK.md](FUTURE_WORK.md)** - Planned enhancements
- **[archived/README.md](archived/README.md)** - Archived versions guide
- **Individual version READMEs** - Version-specific documentation

---

## Summary

**Current Recommended Versions**:
1. **v0.9a** - Latest b₂=21 (primary recommendation)
2. **v0.8** - Yukawa tensors (unique capability)
3. **v0.7** - Stable b₂=21 (proven alternative)

**New Paradigm (WIP)**:
- **variational_g2** - PINN-based constraints-first approach
- **meta_hodge** - Cross-version analysis pipeline
- **tcs_joyce** - TCS global modes

**Coming Soon**: v0.9b with full b₃=77 extraction

**For Details**: See version-specific README.md in each directory

---

**Last Updated**: 2025-11-30
**Framework**: GIFT v2.2.0
**Maintained by**: GIFT Framework Team
**License**: MIT

