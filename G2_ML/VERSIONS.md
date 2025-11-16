# G2 Machine Learning Framework - Version Index

**Framework Completion**: 93%
**Latest Production**: v0.9a (b₂=21), v0.8 (Yukawa)
**In Progress**: v0.9b (full b₃=77 training)
**Last Updated**: 2025-11-16

## Quick Version Guide

| Version | Status | Key Features | Use Case |
|---------|--------|--------------|----------|
| **0.9b** | 🔨 Training | Full b₃=77 extraction | **Next milestone** |
| **0.9a** | ✅ Production | b₂=21 latest refinements | **Recommended for b₂** |
| **0.8** | ✅ Complete | Yukawa + partial b₃ (20/77) | **Yukawa analysis** |
| **0.7** | ✅ Production | b₂=21 stable | Alternative to 0.9a |
| 0.6c | ⚠️ Archived | Validation refinements | Historical |
| 0.6b | ⚠️ Archived | Validation improvements | Historical |
| 0.6 | ⚠️ Archived | Enhanced validation | Historical |
| 0.5 | ⚠️ Archived | b₃ exploration (prelim) | Historical |
| 0.4 | ⚠️ Archived | Curriculum learning | Historical |
| 0.3 | ⚠️ Archived | Loss improvements | Historical |
| 0.2 | ⚠️ Archived | Architecture refinement | Historical |
| 0.1 | ⚠️ Archived | Initial prototype | Historical |

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

### v0.6c - Validation Refinements ⚠️

**Status**: ⚠️ Archived
**Date**: 2025-11
**Focus**: Third iteration of validation improvements

**Historical Note**: Final stepping stone before v0.7 production release

**Migration**: Use v0.7 or v0.9a

---

### v0.6b - Validation Improvements ⚠️

**Status**: ⚠️ Archived
**Date**: 2025-11
**Focus**: Second validation iteration

**Migration**: Use v0.7 or v0.9a

---

### v0.6 - Enhanced Validation ⚠️

**Status**: ⚠️ Archived
**Date**: 2025-11
**Focus**: Comprehensive validation metrics

**Improvements**:
- Gram matrix analysis
- Eigenvalue spectrum validation
- Torsion measurements
- Hitchin functional evaluation

**Migration**: Validation methods integrated into v0.7+

---

### v0.5 - b₃ Exploration (Preliminary) ⚠️

**Status**: ⚠️ Archived
**Date**: 2025-10
**Focus**: Preliminary b₃ harmonic 3-forms exploration

**Historical Note**: Early investigation into b₃=77 extraction

**Migration**: For b₃ work, see v0.8 (partial) or v0.9b (in progress)

---

### v0.4 - Curriculum Learning ⚠️

**Status**: ⚠️ Archived
**Date**: 2025-10
**Focus**: Multi-phase training schedule implementation

**Key Innovation**: Curriculum learning approach
- Phase 1: Orthonormality
- Phase 2: + Closedness
- Phase 3: + Coclosedness

**Impact**: Improved training success rate from 70% to 85%

**Key Files**:
- `K7_G2_Metric_Publication_v04.md` (archived draft)
- `K7_G2_Metric_Supplementary_v04.md` (archived)

**Migration**: Curriculum approach adopted in v0.7+

---

### v0.3 - Loss Function Improvements ⚠️

**Status**: ⚠️ Archived
**Date**: 2025-10
**Focus**: Enhanced loss functions

**Migration**: Use v0.7 or v0.9a

---

### v0.2 - Architecture Refinement ⚠️

**Status**: ⚠️ Archived
**Date**: 2025-09
**Focus**: Network architecture improvements

**Improvements**:
- Refined PhiNetwork architecture
- Better numerical stability
- Enhanced gradient flow
- 2× faster convergence vs. v0.1

**Key Files**:
- `TECHNICAL_DOCUMENTATION.md` (archived)

**Migration**: Use v0.7 or v0.9a

---

### v0.1 - Initial Prototype ⚠️

**Status**: ⚠️ Archived
**Date**: 2025-09
**Focus**: Proof-of-concept

**Features**:
- Basic PhiNetwork architecture
- Initial harmonic 2-form extraction
- Foundational loss functions

**Key Files**:
- `TECHNICAL_DOCUMENTATION.md` (archived)

**Historical Significance**: Demonstrated feasibility of ML approach to G₂ metric learning

**Migration**: Use v0.7 or v0.9a

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
2025-09: 0.1 → 0.2 → 0.3  (Architecture development)
2025-10: 0.4 → 0.5 → 0.6  (Curriculum learning, validation)
2025-11: 0.6b → 0.6c → 0.7 ✅ (Production milestone)
2025-11: 0.8 ✅  (Yukawa + partial b₃)
2025-11: 0.9a ✅ (Latest production)
2025-11: 0.9b 🔨 (Full b₃=77 in progress)
Future:  1.0 🎯 (Complete framework target)
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
- v0.1-0.6c: No active support
- Available for historical reference only
- Documented migration paths to production versions

---

## Related Documentation

- **[STATUS.md](STATUS.md)** - Current implementation status (detailed)
- **[README.md](README.md)** - Framework overview
- **[FUTURE_WORK.md](FUTURE_WORK.md)** - Planned enhancements
- **Individual version READMEs** - Version-specific documentation

---

## Summary

**Current Recommended Versions**:
1. **v0.9a** - Latest b₂=21 (primary recommendation)
2. **v0.8** - Yukawa tensors (unique capability)
3. **v0.7** - Stable b₂=21 (proven alternative)

**Coming Soon**: v0.9b with full b₃=77 extraction

**For Details**: See version-specific README.md in each directory

---

**Last Updated**: 2025-11-16
**Framework**: GIFT v2.0.0
**Maintained by**: GIFT Framework Team
**License**: MIT

