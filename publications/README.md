# GIFT Framework Publications

This directory contains the theoretical publications for the Geometric Information Field Theory (GIFT) framework.

## Directory Structure

### v2.0/ - Published Version
The current stable version (v2.0) of the GIFT framework publications. This version focuses on static topological structure and observable predictions.

**Contents**:
- `gift_main.md` - Core theoretical paper
- `gift_extensions.md` - Dimensional observables and temporal framework
- `gift_technical.md` - Technical supplement with detailed derivations
- `README_experimental_predictions.md` - Experimental collaboration reference
- `supplements/` - Six detailed mathematical supplements (A-F)

**Status**: Published, stable, reference version

**Key Features**:
- 34 dimensionless observables from 3 topological parameters
- Mean precision: 0.13%
- 9 proven exact relations
- Static topological framework

### v2.1/ - Development Version (Torsional Dynamics)
The next version (v2.1) extending the framework with **torsional geodesic dynamics**, connecting static topology to dynamic evolution (RG flow).

**Contents**:
- All v2.0 documents (updated)
- **NEW**: `supplements/G_torsional_dynamics.md` - Complete torsional dynamics formulation

**Status**: Work in progress, branch `feature/v2.1-torsional-dynamics`

**Key Additions**:

#### 1. Torsional Geodesic Flow Framework
- Explicit metric tensor g in (e,π,φ) coordinates
- Volume quantization: det(g) ≈ 2 (binary duality)
- Torsion tensor T from non-closure: |dφ| ≈ 0.0164
- Geodesic equation: d²x^k/dλ² = (1/2) g^kl T_ijl (dx^i/dλ)(dx^j/dλ)

#### 2. Physical Applications
- **Mass hierarchies**: m_τ/m_e = 3477 from geodesic length driven by T_{eφ,π} ≈ -4.89
- **CP violation**: δ_CP = 197° from torsional twist via T_{πφ,e} ≈ -0.45
- **Constant variation**: α̇/α ~ 10⁻¹⁶ yr⁻¹ from ultra-slow flow |v| ≈ 0.015
- **Hubble constant**: H₀² ∝ R·|T|² from curvature-torsion relation

#### 3. RG Flow Connection
- Flow parameter identification: λ = ln(μ/μ₀)
- β-functions as geodesic velocities: β_i = dx^i/dλ
- Geometric origin of renormalization group equations

#### 4. Main Changes

**gift_main.md**:
- New Section 6: "Dynamic Evolution and Torsional Geodesic Flow"
- Renumbered subsequent sections (6→7, 7→8)

**supplements/F_K7_metric.md**:
- Removed torsion-free assumption (dφ = 0, d★φ = 0)
- Added measured torsion values (|dφ| ≈ 0.0164, |d★φ| ≈ 0.0140)
- Updated geometric constraints

**supplements/G_torsional_dynamics.md** (NEW):
- Complete mathematical formulation (11 sections, ~100 pages)
- Metric derivation, torsion tensor calculation
- Geodesic equation from action principle
- Lagrangian formulation and physical applications

## Version History

| Version | Date | Key Features | Status |
|---------|------|-------------|---------|
| v2.0 | 2024 | Static topological framework, 34 observables | Published |
| v2.1 | 2025 | + Torsional dynamics, RG flow connection | In development |

## Citation

### v2.0 (Current)
```bibtex
@article{GIFT-v2.0-2024,
  title={Geometric Information Field Theory: Topological Unification of Standard Model Parameters},
  author={[Author]},
  year={2024},
  note={Version 2.0}
}
```

### v2.1 (Development)
```bibtex
@article{GIFT-v2.1-2025,
  title={Geometric Information Field Theory: Topological Unification with Torsional Geodesic Dynamics},
  author={[Author]},
  year={2025},
  note={Version 2.1 (in preparation)}
}
```

## Development Workflow

1. **v2.0/**: Do NOT modify (reference version)
2. **v2.1/**: Active development on branch `feature/v2.1-torsional-dynamics`
3. Once v2.1 is complete and validated:
   - Merge to `main`
   - Tag as `v2.1.0`
   - v2.1 becomes the new reference
   - Create v2.2/ for future work

## Related Code

Machine learning implementations for K₇ metric and torsion reconstruction:
- `G2_ML/1.0/` - K₇ TCS (Torsional Calibration System)
- `G2_ML/1.0b/` - K₇ Torsional Geodesics
- `G2_ML/1.0c/` - K₇ Torsion with cohomological calibration
- `G2_ML/1.0d-e/` - Extended TCS implementations

## License

MIT License (see repository root)

---

**Note**: The v2.1 development is based on discoveries from numerical K₇ metric reconstruction (G2_ML/) and theoretical insights documented in `wip/MISSING_V2_1.md`.
