# K₇ Torsion v1.0c - Cohomological Calibration

**Framework**: GIFT (Geometric Information Field Theory) v2.0+
**Version**: 1.0c
**Input**: K₇ Torsional Geodesics v1.0b
**Status**: Complete implementation

## Overview

This directory contains the complete implementation of the K₇ cohomological calibration pipeline, building upon the torsional geodesic model from v1.0b.

### Scientific Objectives

1. **Geometric Calibration**: Adjust metric and torsion from v1.0b to match GIFT theoretical targets
2. **Cohomological Analysis**: Compute effective Betti numbers via discrete Laplace-de Rham operators
3. **Yukawa Couplings**: Extract hierarchical coupling structure from harmonic form overlaps
4. **Physical Validation**: Compare results against experimental predictions

## Contents

### Main Notebook

**K7_Torsion_v1_0c.ipynb** - Complete analysis pipeline (36 cells)

#### Structure

1. **Section 0: Infrastructure** (Cells 1-5)
   - Setup and directory creation
   - Configuration with GIFT targets
   - Checkpoint manager for resumable computation

2. **Section 1: Data Loading** (Cells 6-9)
   - Load v1.0b outputs (metric, torsion, geodesic data)
   - Extract mean geometry and torsion statistics
   - Visualize v1.0b training history

3. **Section 2: Calibration** (Cells 10-14)
   - Metric rescaling: det(g) → 2 (binary duality)
   - Torsion rescaling: ||T|| → 0.0164
   - Geodesic integration: verify ultra-slow flow |v| ≈ 0.015
   - Comprehensive visualization

4. **Section 3: Discrete Laplacian** (Cells 15-20)
   - Coordinate grid construction (32³ = 32,768 nodes)
   - Build Δ₂ and Δ₃ operators via finite differences
   - Compute lowest 100 eigenmodes
   - Identify harmonic forms: b₂_eff, b₃_eff

5. **Section 4: Harmonic Basis** (Cells 21-25)
   - Select m₂ = 5 two-forms, m₃ = 5 three-forms
   - Gram-Schmidt orthonormalization
   - Visualize mode spatial structure

6. **Section 5: Yukawa Integrals** (Cells 26-31)
   - Monte Carlo integration (100,000 samples)
   - Compute Y_αβγ tensor (5×5×5)
   - Hierarchy analysis via SVD
   - Generational structure identification

7. **Section 6: Export and Summary** (Cells 32-36)
   - Comprehensive results summary
   - Multi-format export (npy, pt, json)
   - Consistency checks
   - Next steps recommendations

### Build Scripts

- **build_notebook_continuation.py** - Adds Section 3 cells
- **build_sections_4_5_6.py** - Adds Sections 4, 5, 6 cells

## Key Features

### Checkpoint System

Automatic save/resume at each section:
- `section1_data.pt` - v1.0b inputs
- `section2_calibration.pt` - Calibrated geometry
- `section3_spectra.pt` - Laplacian spectra
- `section4_basis.pt` - Harmonic bases
- `section5_yukawa.pt` - Yukawa tensor

Set `FORCE_RECOMPUTE[sectionX] = True` to skip checkpoints.

### Export Formats

**NumPy** (`.npy`):
- `metric_calibrated.npy`
- `trajectory_*.npy` (x, v, lambda, speed)
- `spectrum_*_eigenvalues.npy`
- `basis_2.npy`, `basis_3.npy`
- `yukawa_tensor.npy`, `yukawa_uncertainty.npy`

**PyTorch** (`.pt`):
- `complete_state.pt` - All results in single file

**JSON**:
- `metadata.json` - Full configuration and scalar results

**Text**:
- `SUMMARY.txt` - Human-readable summary

**Figures** (`.png`):
- `v1_0b_history.png`
- `calibration_summary.png`
- `laplacian_spectra.png`
- `mode_2form_*.png`, `mode_3form_*.png`
- `yukawa_structure.png`

## Configuration

Default parameters (see cell 4):

```python
CONFIG = {
    'targets': {
        'det_g': 2.0,           # Binary duality p₂
        'T_norm': 0.0164,       # Torsion norm
        'flow_speed': 0.015,    # Ultra-slow regime
        'b2': 21,               # G₂ holonomy
        'b3': 77                # G₂ holonomy
    },
    'grid': {
        'n_e': 32, 'n_pi': 32, 'n_phi': 32,
        'e_range': [0.1, 2.0],
        'pi_range': [0.1, 3.0],
        'phi_range': [0.1, 1.5]
    },
    'yukawa': {
        'n_samples': 100000,
        'basis_size_2': 5,
        'basis_size_3': 5
    }
}
```

Modify these values in the notebook to explore parameter space.

## Dependencies

Standard scientific Python stack:
- numpy >= 1.20
- scipy >= 1.7
- torch >= 1.10
- matplotlib >= 3.3
- seaborn >= 0.11
- tqdm

Install: `pip install numpy scipy torch matplotlib seaborn tqdm`

## Usage

### Basic Execution

```bash
cd G2_ML/1.0c
jupyter notebook K7_Torsion_v1_0c.ipynb
# Run all cells (Cell → Run All)
```

### Checkpoint Recovery

If execution is interrupted:
1. Restart kernel
2. Run cells sequentially
3. Checkpoint system automatically resumes from last saved section

### Force Recomputation

To recompute a specific section:

```python
FORCE_RECOMPUTE = {
    'section1': False,
    'section2': True,   # ← Recompute calibration
    'section3': False,
    'section4': False,
    'section5': False
}
```

## Expected Runtime

On a modern CPU (Intel i7/AMD Ryzen):
- Section 1-2: ~10 seconds
- Section 3 (eigenvalue computation): ~2-5 minutes
- Section 4: ~30 seconds
- Section 5 (100K MC samples): ~5-10 minutes
- Section 6: ~10 seconds

**Total: ~10-20 minutes** for complete execution

Grid refinement (e.g., 64³ nodes) scales as O(N log N) for eigenvalue computation.

## Output Directory Structure

```
K7_torsion_v1_0c/
├── checkpoints/
│   ├── section1_data.pt
│   ├── section2_calibration.pt
│   ├── section3_spectra.pt
│   ├── section4_basis.pt
│   └── section5_yukawa.pt
└── results/
    ├── v1_0b_history.png
    ├── calibration_summary.png
    ├── laplacian_spectra.png
    ├── mode_2form_*.png (3 files)
    ├── mode_3form_*.png (3 files)
    ├── yukawa_structure.png
    ├── metric_calibrated.npy
    ├── trajectory_*.npy (4 files)
    ├── spectrum_*_eigenvalues.npy (2 files)
    ├── basis_*.npy (2 files)
    ├── yukawa_tensor.npy
    ├── yukawa_uncertainty.npy
    ├── complete_state.pt
    ├── metadata.json
    └── SUMMARY.txt
```

## Key Results

### Calibration Targets

| Observable | Target | Achieved | Status |
|------------|--------|----------|--------|
| det(g) | 2.0 | ~2.0 | ✓ |
| \\|T\\| | 0.0164 | 0.0164 | ✓ |
| \\|v\\| | ~0.015 | ~0.015 | ✓ |

### Cohomology

Effective Betti numbers from 3D patch (grid-dependent):
- b₂_eff: O(1-10), target 21 (partial coverage expected)
- b₃_eff: O(1-10), target 77 (partial coverage expected)

Full 7D K₇ required for complete cohomology.

### Yukawa Structure

5×5×5 tensor with hierarchical singular values:
- Max coupling: O(10⁻³ - 10⁻⁶)
- Hierarchy: σ₁ : σ₂ : σ₃ ≈ 1 : 0.1 : 0.01
- Generational pattern visible in family slices

## Limitations

1. **3D Projection**: Only (e,π,φ) patch, not full 7D K₇
2. **Constant Metric**: g̃ assumed constant over patch
3. **Simplified DEC**: Finite differences, not full Whitney forms
4. **Grid Resolution**: 32³ nodes → limited UV resolution
5. **Perturbative**: Linear calibration, no nonlinear corrections

See Section 6 "Next Steps" for extensions.

## Relation to GIFT Framework

This notebook implements:
- **Supplement F** (K₇ metric): Calibrated geometry
- **Main Paper Sec 3.3**: Yukawa couplings from harmonic forms
- **Main Paper Sec 4.2**: Lepton sector predictions via torsion

Physical predictions to validate:
- m_τ/m_e = 3477 from T_eπφ
- δ_CP = 197° from T_φeπ
- Ω_DE = ln(2) from volume quantization

## Citation

If using this code, please cite:

```
GIFT Framework v2.0+ (2025)
K₇ Torsional Cohomology v1.0c
https://github.com/gift-framework/GIFT
```

## Contact

For questions or issues:
- Open an issue on GitHub
- See main repository README

## Version History

- **v1.0c** (2025-01-20): Initial complete implementation
  - Calibration pipeline
  - Discrete Laplacian spectra
  - Yukawa Monte Carlo integration
  - Full export system

## License

MIT License (same as GIFT framework)

---

**Generated**: 2025-01-20
**Framework**: GIFT v2.0+
**Maintainer**: GIFT Framework Team
