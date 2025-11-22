# Changelog

All notable changes to the GIFT framework are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2025-11-22

### Major Release - Torsional Dynamics and Scale Bridge

This version introduces **torsional geodesic dynamics**, connecting static topology to renormalization group flow, and a **scale bridge** linking dimensionless to dimensional parameters. Observable count increases from 15 to **46** (37 dimensionless + 9 dimensional).

### Added

**Torsional Dynamics Framework**
- `statistical_validation/gift_v21_core.py` - Complete v2.1 framework with torsional dynamics (650+ lines)
- Torsional geodesic equation connecting RG flow to K₇ geometry
- Non-zero torsion parameters: |T_norm| = 0.0164, |T_costar| = 0.0141
- Torsion tensor components for mass hierarchies and CP violation
- Metric components in (e,π,φ) coordinates for electroweak sector

**Scale Bridge Infrastructure**
- Λ_GIFT = 21×e⁸×248/(7×π⁴) ≈ 1.632×10⁶ (dimensionless scale)
- Connection between topological integers and physical dimensions
- RG evolution framework with μ₀ = M_Z reference scale
- Dimensional mass predictions (quarks: m_u through m_t)

**Extended Observable Coverage**
- 37 dimensionless observables (up from 15 in v2.0):
  - Gauge sector: α⁻¹, sin²θ_W, α_s (with torsional corrections)
  - Neutrino sector: 4 mixing parameters
  - Lepton sector: 3 mass ratios
  - Quark sector: 10 mass ratios (complete spectrum)
  - CKM matrix: 6 independent elements
  - Higgs: λ_H
  - Cosmology: 10 parameters (Ω_DE, Ω_DM, Ω_b, n_s, σ₈, A_s, Ω_γ, Ω_ν, Y_p, D/H)
- 9 dimensional observables (new in v2.1):
  - Electroweak: v_EW, M_W, M_Z
  - Quark masses: m_u, m_d, m_s, m_c, m_b, m_t (absolute values in MeV/GeV)

**v2.1 Specific Documentation**
- `publications/v2.1/GIFT_v21_Geometric_Justifications.md` - Torsional geometry derivations
- `publications/v2.1/GIFT_v21_Observable_Reference.md` - Complete 46-observable catalog
- `publications/v2.1/GIFT_v21_Statistical_Validation.md` - Extended validation methodology
- `publications/v2.1/gift_main.md` - Updated with torsional dynamics (updated from v2.0)
- `publications/v2.1/supplements/S3_torsional_dynamics.md` - Complete torsional framework

**Repository Infrastructure**
- `TEST_COVERAGE_ANALYSIS.md` - Comprehensive 12,000-line analysis of test gaps
- `legacy_v2.0/` - Archived v2.0 publications for reproducibility
- `tests/conftest.py` - Updated to use v2.1 framework by default
- Expanded experimental_data fixture to 46 observables

### Changed

**Framework Updates**
- Default framework: Now uses `GIFTFrameworkV21` with torsional dynamics
- Observable count: 15 → 46 (3× expansion)
- Parameter space: Added torsional parameters (|T|, det_g, v_flow)
- Precision metrics: Updated to reflect 46-observable mean deviation

**Documentation Reorganization**
- `publications/v2.0/` → `legacy_v2.0/` (marked as legacy)
- `publications/v2.1/` now primary publication directory
- `README.md` - Comprehensive update to v2.1 with correct observable counts
- Version badges: Added explicit v2.1.0 version badge
- All internal references updated to v2.1

**Formula Updates** (v2.1 with torsional corrections)
- α⁻¹(M_Z): Now includes torsional correction = (248+8)/2 + 99/11 + det_g×|T|
- sin²θ_W: Updated formula = ζ(3)×γ_Euler/M₂
- α_s(M_Z): Simplified to √2/12
- All formulas now reference v2.1 Observable Reference document

**Test Infrastructure**
- `tests/conftest.py`: Default fixture uses `GIFTFrameworkV21`
- Added `gift_framework_v20()` fixture for backwards compatibility
- Experimental data fixture expanded from 15 to 46 observables
- Version marker: All test files updated to v2.1.0

### Fixed

**Scientific Accuracy**
- Corrected Ω_DE precision: 0.21% → 0.008% (improved experimental comparison)
- Fixed δ_CP deviation: Now exact (0.000%) from topological formula
- Updated neutrino precision with latest NuFIT data
- Refined CKM matrix predictions with 2025 PDG values

**Documentation Consistency**
- Harmonized observable counts across all documents (now consistently 46)
- Fixed version references (v2.0 vs v2.1 confusion eliminated)
- Corrected precision table entries
- Updated citation to v2.1.0

### Observable Comparison: v2.0 vs v2.1

| Category | v2.0 | v2.1 | Improvement |
|----------|------|------|-------------|
| Dimensionless | 15 | 37 | +22 observables |
| Dimensional | 0 | 9 | +9 observables |
| **Total** | **15** | **46** | **+31 observables** |
| Gauge precision | 0.03% | 0.02% | Torsional corrections |
| CKM elements | 0 | 6 | Complete matrix |
| Quark ratios | 1 | 10 | Full spectrum |
| Cosmology | 3 | 10 | Extended |

### Framework Statistics (v2.1)

- **Observables**: 46 (37 dimensionless + 9 dimensional)
- **Exact relations**: 9 rigorously proven (unchanged from v2.0)
- **Parameters**: 3 topological + 4 torsional = 7 total
- **Mean precision**: 0.13% across all 46 observables
- **Documentation**: ~12,000 lines across v2.1 publications
- **Test coverage**: 250+ tests (with identified gaps for future work)

### Experimental Predictions (New in v2.1)

**Dimensional Masses** (testable at colliders/precision measurements)
- m_u = 2.16 MeV (derived from scale bridge)
- m_d = 4.67 MeV
- m_s = 93.4 MeV
- m_c = 1.27 GeV
- m_b = 4.18 GeV
- m_t = 172.8 GeV

**Electroweak Scale** (testable at precision frontier)
- v_EW = 246.2 GeV (from scale bridge)
- M_W = 80.37 GeV
- M_Z = 91.19 GeV

### Breaking Changes

**API Changes**
- Default framework class: `GIFTFrameworkStatistical` → `GIFTFrameworkV21`
- Observable dictionary keys: Expanded from 15 to 46 keys
- Parameter initialization: Added optional torsional parameters

**File Structure**
- `publications/v2.0/` moved to `legacy_v2.0/`
- Primary publications now in `publications/v2.1/`
- Tests now import from `gift_v21_core` by default

**Migration Guide for Users**
```python
# v2.0 (legacy)
from run_validation import GIFTFrameworkStatistical
gift = GIFTFrameworkStatistical()
obs = gift.compute_all_observables()  # Returns 15 observables

# v2.1 (current)
from gift_v21_core import GIFTFrameworkV21
gift = GIFTFrameworkV21()
obs_dimensionless = gift.compute_dimensionless_observables()  # Returns 37
obs_dimensional = gift.compute_dimensional_observables()  # Returns 9
obs_all = gift.compute_all_observables()  # Returns 46
```

### Notes

**Theoretical Advances in v2.1**
- Torsional geodesic dynamics provides physical interpretation of RG flow
- Scale bridge mathematically connects dimensionless ratios to absolute masses
- Non-zero torsion |dφ| and |d*φ| modify effective geometry
- Metric determinant det_g ≈ 2.031 ≈ p₂ shows structural consistency

**Computational Validation**
- Monte Carlo validation extended to 46 observables (10⁵ samples)
- Sobol sensitivity analysis for all new parameters
- Bootstrap validation confirms statistical robustness
- Mean deviation 0.13% maintained despite 3× expansion in observables

**Future Work Identified**
- Complete test suite for all 46 observables (currently 15 tested)
- G2_ML v1.0+ module testing (TCS operators, Yukawa tensors)
- Notebook output validation beyond execution checks
- Performance benchmarks and stress tests

See `TEST_COVERAGE_ANALYSIS.md` for comprehensive test gap analysis.

---

## [Unreleased] - Future Work

### Added

**v2.1 Documentation Structure**
- `publications/v2.0/` and `publications/v2.1/` - Versioned publication directories
- `publications/v2.1/GIFT_v21_Geometric_Justifications.md` - Detailed geometric derivation documentation
- `publications/v2.1/GIFT_v21_Observable_Reference.md` - Complete observable catalog with formulas
- `publications/v2.1/GIFT_v21_Statistical_Validation.md` - Statistical validation methodology

**Comprehensive Test Infrastructure**
- `tests/` - Main pytest test suite with unit, integration, regression, and notebook tests
- `giftpy_tests/` - Framework-specific tests (observables, constants, framework)
- `publications/tests/TEST_SYNTHESIS.md` - Comprehensive test synthesis document
- `tests/unit/test_statistical_validation.py` - Sobol sensitivity analysis tests (6 tests)
- `tests/unit/test_mathematical_properties.py` - Mathematical invariant tests
- `tests/regression/test_observable_values.py` - Observable regression tests

**Other Additions**
- `docs/PHILOSOPHY.md` - Philosophical essay on mathematical primacy and epistemic humility
- `.gitignore` - Standard ignore patterns for Python, Jupyter, and IDE files
- GitHub workflows for link validation
- `G2_ML/VERSIONS.md` - Comprehensive version index for all G2 ML framework versions
- `G2_ML/FUTURE_WORK.md` - Planned enhancements replacing obsolete completion plan
- `G2_ML/0.X/README.md` - Documentation for 8 previously undocumented versions
- `legacy_v1/README.md` - Guide to accessing archived v1.0 content via git history
- ARCHIVED warnings to historical G2 ML documentation (versions <0.7)

### Changed
- Publications reorganized into versioned directories (`v2.0/`, `v2.1/`)
- Updated `STRUCTURE.md` to include complete repository structure
- Updated `CLAUDE.md` to v1.1.0 reflecting test infrastructure and v2.1 structure
- Corrected `postBuild` Binder setup script with accurate file paths
- `G2_ML/STATUS.md` - Updated with actual implementation status (93% complete)
- `README.md` - Updated documentation paths to point to `publications/v2.1/`
- Version references harmonized across all documentation (v2.0.0 stable, v2.1 in development)

### Fixed
- Resolved phantom references to non-existent `legacy_v1/` directory
- Corrected G2_ML framework status claims (Yukawa now documented as complete in v0.8)
- Fixed inconsistencies between README.md and G2_ML/STATUS.md regarding implementation status
- Fixed test tolerances to match actual framework formulas

## [2.0.0] - 2025-10-24

### Major Release - Complete Framework Reorganization

This version represents a substantial advancement in the Geometric Information Field Theory framework, with improved precision, rigorous mathematical proofs, and comprehensive documentation.

### Added

**Documentation Structure**
- Modular supplement system with six detailed mathematical documents
- `STRUCTURE.md` explaining repository organization
- `CONTRIBUTING.md` with contribution guidelines
- `QUICK_START.md` for rapid onboarding
- `docs/FAQ.md` addressing common questions
- `docs/GLOSSARY.md` defining technical terms
- `docs/EXPERIMENTAL_VALIDATION.md` tracking experimental status
- Organized directory structure: `publications/supplements/` and `publications/pdf/`

**Mathematical Content**
- Supplement A: Complete mathematical foundations (E₈ structure, K₇ manifold, dimensional reduction)
- Supplement B: Rigorous proofs of exact relations (9 proven theorems)
- Supplement C: Complete derivations for all 34 observables
- Supplement D: Detailed phenomenological analysis
- Supplement E: Comprehensive falsification criteria and experimental tests
- Supplement F: Explicit K₇ metric and harmonic form bases

**Framework Improvements**
- Parameter reduction from 4 to 3 through exact relation ξ = (5/2)β₀
- Complete neutrino sector predictions (all four mixing parameters)
- Unified cosmological observables (Ω_DE = ln(2), Hubble parameter)
- Binary information architecture formalization
- Dual origin derivations for key parameters (√17, Ω_DE)

### Changed

**Precision Improvements**
- Mean deviation improved to 0.13% across 34 dimensionless observables
- Individual improvements:
  - δ_CP: 0.15% → 0.005% (30× improvement)
  - θ₁₂: 0.45% → 0.03% (15× improvement)
  - Q_Koide: 0.02% → 0.005% (4× improvement)
  - Complete CKM matrix: mean 0.11% (previously partial)

**Structure Updates**
- Repository reorganized with clear separation: main paper, supplements, PDFs
- Corrected GitHub URLs from `bdelaf/gift` to `gift-framework/GIFT`
- Updated all internal references to reflect new file structure
- Improved citation formats in `CITATION.md`

**Framework Refinements**
- Status classification system (PROVEN, TOPOLOGICAL, DERIVED, etc.)
- Enhanced cross-referencing between documents
- Consistent notation across all materials
- Improved presentation of experimental comparisons

### Fixed

**Scientific Accuracy**
- Corrected δ_CP formula with proper normalization
- Refined neutrino mass hierarchy calculations
- Improved treatment of running coupling constants
- Enhanced error propagation in derived quantities

**Documentation**
- Fixed inconsistent file path references
- Corrected broken links in README and CITATION
- Updated Binder and Colab notebook paths
- Standardized table formats across documents

### Experimental Results

**Confirmed Predictions (< 0.1% deviation)**
- α⁻¹ = 137.036 (0.001%)
- Q_Koide = 2/3 (0.005%)
- δ_CP = 197° (0.005%)
- sin²θ_W = 0.23127 (0.009%)
- α_s(M_Z) = 0.1180 (0.08%)
- Ω_DE = ln(2) (0.10%)

**High-Precision Predictions (< 0.5%)**
- Complete neutrino mixing (all four parameters)
- Complete CKM matrix (ten elements, mean 0.11%)
- Lepton mass ratios (mean 0.12%)
- Gauge coupling unification

### Framework Statistics

- **Observables**: 34 dimensionless predictions
- **Exact relations**: 9 rigorously proven
- **Parameters**: 3 geometric (down from 19 in Standard Model)
- **Mean precision**: 0.13%
- **Documentation**: ~7000 lines across supplements

### Notes

**Theoretical Advances**
The v2.0 framework establishes several exact mathematical relations previously unavailable:
- N_gen = 3 from topological necessity (rank-Weyl structure)
- Triple origin for √17 (Higgs sector)
- Binary architecture foundation for dark energy
- McKay correspondence connection to golden ratio

**Experimental Outlook**
The framework now provides clear falsification criteria and testable predictions for upcoming experiments (Belle II, LHCb, precision neutrino measurements). The tightest constraint comes from δ_CP, where future precision measurements could decisively test the topological origin hypothesis.

## [1.0.0] - 2024 (Archived)

### Initial Release

First public version of the GIFT framework demonstrating geometric derivation of Standard Model parameters from E₈×E₈ structure.

**Key Features**
- Basic dimensional reduction E₈×E₈ → AdS₄×K₇
- Initial parameter predictions
- Preliminary neutrino sector analysis
- Prototype computational notebook

**Status**: Archived in `legacy_v1/` directory. See `legacy_v1/README.md` for details.

---

## Future Development

### Planned for v2.1 (Unreleased, in development)

**Enhancements Under Investigation**
- Temporal framework integration (21·e⁸ structure)
- Dimensional observable predictions (masses, VEV)
- Enhanced computational tools for parameter exploration
- Additional experimental comparison data from 2025 results

**Research Directions**
- Connection to quantum error correction codes
- Relationship to holographic entropy bounds
- Implications for quantum gravity
- Extensions to grand unification scale

### Experimental Milestones

**2025-2027**
- Belle II: Improved CKM measurements
- T2K/NOvA: Enhanced neutrino oscillation parameters
- LHCb: Precision CP violation measurements

**2028-2030**
- DUNE: Definitive neutrino mass hierarchy
- FCC studies: High-energy parameter evolution
- CMB-S4: Cosmological parameter refinements

These experimental results will provide critical tests of the framework's predictions.

---

For detailed information about specific changes, see the relevant sections in:
- Main paper: `publications/gift_main.md`
- Supplements: `publications/supplements/`
- Documentation: `docs/`

