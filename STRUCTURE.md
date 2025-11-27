# Repository Structure

This document describes the organization of the GIFT v2.2 framework repository.

## Overview

The repository is organized to separate theoretical content, mathematical derivations, documentation, and computational tools while maintaining clear navigation paths.

## Directory Layout

```
gift/
├── README.md                          # Main entry point and overview
├── CITATION.md                        # Citation formats and references
├── LICENSE                            # MIT License
├── CHANGELOG.md                       # Version history and changes
├── STRUCTURE.md                       # This file
├── CONTRIBUTING.md                    # Contribution guidelines
├── QUICK_START.md                     # Fast onboarding guide
├── requirements.txt                   # Python dependencies
│
├── publications/                      # Main theoretical documents (v2.2)
│   ├── gift_2_2_main.md               # Core theoretical paper (~1400 lines)
│   ├── summary.txt                    # Executive summary (5-min read)
│   ├── READING_GUIDE.md               # Navigation by time/interest
│   ├── GLOSSARY.md                    # Terminology definitions
│   ├── GIFT_v22_Observable_Reference.md   # 39-observable catalog
│   ├── GIFT_v22_Geometric_Justifications.md # Derivation details
│   ├── GIFT_v22_Statistical_Validation.md  # Validation methods
│   │   # (Interactive notebooks now in assets/visualizations/)
│   │
│   ├── supplements/                   # 7 detailed mathematical supplements
│   │   ├── S1_mathematical_architecture.md  # E₈, G₂, cohomology
│   │   ├── S2_K7_manifold_construction.md   # TCS, ML metrics
│   │   ├── S3_torsional_dynamics.md         # Geodesics, RG flow
│   │   ├── S4_complete_derivations.md       # 13 proofs + all derivations
│   │   ├── S5_experimental_validation.md    # Data comparison, falsification
│   │   ├── S6_theoretical_extensions.md     # QG, info theory
│   │   └── S7_dimensional_observables.md    # Masses, scale bridge
│   │
│   └── pdf/                           # PDF versions (generated)
│
├── docs/                              # Additional documentation
│   ├── FAQ.md                         # Frequently asked questions
│   ├── GLOSSARY.md                    # Technical terms and notation
│   ├── PHILOSOPHY.md                  # Philosophical perspectives
│   ├── EXPERIMENTAL_VALIDATION.md     # Current experimental status
│   └── *.md                           # Additional guides
│
├── G2_ML/                             # Machine learning for K₇ metrics
│   ├── README.md                      # Framework overview
│   ├── STATUS.md                      # Current implementation status
│   └── [versioned dirs]               # Neural network training
│
├── statistical_validation/            # Statistical analysis tools
│   ├── run_validation.py              # Monte Carlo validation
│   └── full_results/                  # Validation output data
│
├── assets/                            # Interactive assets and tools
│   └── visualizations/                # Interactive Jupyter visualizations
│
├── tests/                             # Test suite
│   ├── unit/                          # Unit tests
│   ├── integration/                   # Integration tests
│   └── regression/                    # Regression tests
│
└── legacy/                            # Archived versions
    ├── legacy_v1/                     # v1.0 archive
    ├── legacy_v2.0/                   # v2.0 archive
    └── legacy_v2.1/                   # v2.1 archive
```

## Document Hierarchy

### Entry Points

For different audiences and purposes:

**Quick Overview** → `README.md`
- Framework summary
- Key results
- Installation instructions
- Links to all resources

**Getting Started** → `QUICK_START.md`
- Fast onboarding
- Immediate results
- Guided tour of capabilities

**Scientific Details** → `publications/gift_2_2_main.md`
- Complete theoretical framework
- Main predictions and results
- References to detailed supplements

### Main Theoretical Content

The core framework is presented in a modular structure:

1. **Main Paper** (`publications/gift_2_2_main.md`)
   - Overview of GIFT framework
   - Key results and predictions
   - Summary of mathematical structure
   - Experimental validation
   - ~1400 lines, self-contained introduction

2. **Seven Supplements** (`publications/supplements/`)
   - **Supplement S1**: Mathematical architecture (E₈, K₇, cohomology)
   - **Supplement S2**: K₇ manifold construction (TCS, G₂ holonomy, ML metrics)
   - **Supplement S3**: Torsional dynamics (torsion tensor, geodesic flow, RG)
   - **Supplement S4**: Complete derivations (13 proven relations + all observables)
   - **Supplement S5**: Experimental validation (data comparison, falsification protocol)
   - **Supplement S6**: Theoretical extensions (quantum gravity, information theory)
   - **Supplement S7**: Dimensional observables (absolute masses, scale bridge, cosmology)

3. **Reference Documents** (`publications/`)
   - `GIFT_v22_Observable_Reference.md`: Complete 39-observable catalog
   - `GIFT_v22_Geometric_Justifications.md`: Geometric derivation details
   - `GIFT_v22_Statistical_Validation.md`: Statistical validation methods

### Computational Tools

**Interactive Visualizations** (`assets/visualizations/`)
- `e8_root_system_3d.ipynb`: E8 240-root structure visualization
- `precision_dashboard.ipynb`: All observables vs experiment
- `dimensional_reduction_flow.ipynb`: 496D -> 99D -> 4D animation
- Runs on Binder or Google Colab without local installation

### Supporting Documentation

**Technical References**
- `docs/GLOSSARY.md`: Definitions of all technical terms
- `docs/FAQ.md`: Common questions and answers
- `docs/EXPERIMENTAL_VALIDATION.md`: Current experimental status

**Project Management**
- `CONTRIBUTING.md`: How to contribute
- `CHANGELOG.md`: Version history
- `CITATION.md`: Citation formats

## File Naming Conventions

### Markdown Documents

**Main papers**: Versioned names
- `gift_2_2_main.md`
- `GIFT_v22_*.md` (reference documents)

**Supplements**: S-prefix with descriptive names
- `S1_mathematical_architecture.md`
- `S2_K7_manifold_construction.md`
- `S3_torsional_dynamics.md`
- `S4_complete_derivations.md`
- `S5_experimental_validation.md`
- `S6_theoretical_extensions.md`
- `S7_dimensional_observables.md`

**Documentation**: Purpose-based names
- `FAQ.md`
- `GLOSSARY.md`
- `EXPERIMENTAL_VALIDATION.md`

**Project files**: ALL_CAPS for visibility
- `README.md`
- `CHANGELOG.md`
- `STRUCTURE.md`

### PDF Files

PDF versions are in `publications/pdf/` with matching names.

## Navigation Guide

### Finding Specific Information

**Parameter Predictions**
→ `publications/gift_2_2_main.md` (Section 8: Observable Predictions)
→ `publications/supplements/S4_complete_derivations.md` (detailed derivations)
→ `publications/GIFT_v22_Observable_Reference.md` (complete catalog)

**Mathematical Proofs**
→ `publications/supplements/S4_complete_derivations.md` (13 proven relations)
→ `publications/supplements/S1_mathematical_architecture.md` (underlying mathematics)

**Experimental Comparison**
→ `publications/gift_2_2_main.md` (summary tables)
→ `publications/supplements/S5_experimental_validation.md` (detailed analysis)
→ `docs/EXPERIMENTAL_VALIDATION.md` (current status)

**Geometric Construction**
→ `publications/supplements/S1_mathematical_architecture.md` (K₇ manifold overview)
→ `publications/supplements/S2_K7_manifold_construction.md` (TCS construction, ML metrics)

**Falsification Tests**
→ `publications/supplements/S5_experimental_validation.md` (comprehensive criteria)
→ `docs/EXPERIMENTAL_VALIDATION.md` (experimental timeline)

**Definitions and Notation**
→ `publications/GLOSSARY.md` (all technical terms)
→ `publications/gift_2_2_main.md` (Section 1.4: Conventions)

### Cross-References

Documents use internal cross-referencing:
- Equations: `(#eq:delta-cp)`
- Figures: `{#fig:e8-roots}`
- Sections: `(#sec:foundations)`
- External documents: Direct markdown links

## Document Status

All documents include status classifications for results:
- **PROVEN**: Exact topological identity with rigorous proof
- **TOPOLOGICAL**: Direct consequence of topological structure
- **DERIVED**: Calculated from proven relations
- **THEORETICAL**: Has theoretical justification, proof in progress
- **PHENOMENOLOGICAL**: Empirically accurate, derivation in progress
- **EXPLORATORY**: Preliminary formula, mechanism under investigation

## Maintenance

### Updating Documents

When modifying the framework:

1. Update relevant section in main paper or supplements
2. Update `CHANGELOG.md` with changes
3. Regenerate PDF versions if needed
4. Update cross-references if structure changes
5. Run computational notebook to verify calculations
6. Update `docs/EXPERIMENTAL_VALIDATION.md` with new data

### Adding New Content

For new predictions or results:

1. Add to appropriate supplement (or create new section)
2. Summarize in main paper if significant
3. Update tables in Section 4 (Dimensionless Observable Predictions)
4. Add to `CHANGELOG.md` under appropriate version
5. Update `docs/FAQ.md` if commonly asked

## Version Control

The repository uses semantic versioning:
- Major (v2.0): Substantial framework changes
- Minor (v2.1): New features, additional predictions
- Patch (v2.0.1): Bug fixes, documentation corrections

See `CHANGELOG.md` for complete version history.

## External Links

All external links (GitHub, arXiv, DOIs) use the canonical form:
- GitHub: `https://github.com/gift-framework/GIFT`
- Repository issues: `https://github.com/gift-framework/GIFT/issues`
- Binder: Links to specific notebooks in `publications/`

## Archival Policy

Previous versions:
- **v1.0**: Archived in `legacy_v1/`, accessible via git history
- **v2.x**: Current active version
- Future versions will maintain backward compatibility in document structure

## Questions

For questions about repository organization:
- See `docs/FAQ.md` for common questions
- Open an issue: `https://github.com/gift-framework/GIFT/issues`
- Consult `CONTRIBUTING.md` for contribution guidelines

---

This structure is designed to facilitate both casual exploration and deep technical study while maintaining scientific rigor and clarity.

