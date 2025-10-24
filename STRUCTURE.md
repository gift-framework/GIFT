# Repository Structure

This document describes the organization of the GIFT v2 framework repository.

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
├── runtime.txt                        # Python version specification
├── postBuild                          # Binder configuration
│
├── publications/                      # Main theoretical documents
│   ├── gift_main.md                   # Core theoretical paper (~1100 lines)
│   ├── gift_extensions.md             # Dimensional observables and temporal framework
│   ├── gift_v2_notebook.ipynb         # Interactive computational notebook
│   │
│   ├── supplements/                   # Detailed mathematical supplements
│   │   ├── A_math_foundations.md      # E₈ structure, K₇ manifold, reduction
│   │   ├── B_rigorous_proofs.md       # Complete proofs of exact relations
│   │   ├── C_complete_derivations.md  # All 34 observable derivations
│   │   ├── D_phenomenology.md         # Experimental comparison
│   │   ├── E_falsification.md         # Testability and falsification criteria
│   │   └── F_K7_metric.md             # Explicit geometric constructions
│   │
│   └── pdf/                           # PDF versions of all documents
│       ├── gift-main.pdf              # Main paper PDF
│       ├── gift_extensions.pdf        # Extensions PDF
│       ├── Supp_A.pdf                 # Mathematical foundations PDF
│       ├── Supp_B.pdf                 # Rigorous proofs PDF
│       ├── Supp_C.pdf                 # Complete derivations PDF
│       ├── Supp_D.pdf                 # Phenomenology PDF
│       ├── Supp_E.pdf                 # Falsification PDF
│       └── Supp_F.pdf                 # K₇ metric PDF
│
├── docs/                              # Additional documentation
│   ├── FAQ.md                         # Frequently asked questions
│   ├── GLOSSARY.md                    # Technical terms and notation
│   └── EXPERIMENTAL_VALIDATION.md     # Current experimental status
│
└── legacy_v1/                         # Archived v1.0 version
    └── README.md                      # Guide to v1 archive
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

**Scientific Details** → `publications/gift_main.md`
- Complete theoretical framework
- Main predictions and results
- References to detailed supplements

### Main Theoretical Content

The core framework is presented in a modular structure:

1. **Main Paper** (`publications/gift_main.md`)
   - Overview of GIFT framework
   - Key results and predictions
   - Summary of mathematical structure
   - Experimental validation
   - ~1100 lines, self-contained introduction

2. **Six Supplements** (`publications/supplements/`)
   - **Supplement A**: Mathematical foundations (E₈, K₇, dimensional reduction)
   - **Supplement B**: Rigorous proofs (9 exact relations with complete derivations)
   - **Supplement C**: Complete derivations (all 34 observables)
   - **Supplement D**: Phenomenology (detailed experimental comparison)
   - **Supplement E**: Falsification criteria (testability and experimental program)
   - **Supplement F**: Explicit constructions (K₇ metric, harmonic forms)

3. **Extensions** (`publications/gift_extensions.md`)
   - Dimensional observables (masses, VEV, Hubble parameter)
   - Temporal framework (21·e⁸ structure)
   - Advanced topics and ongoing research

### Computational Tools

**Interactive Notebook** (`publications/gift_v2_notebook.ipynb`)
- All calculations reproduced computationally
- Visualization of geometric structures
- Parameter exploration tools
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

**Main papers**: Descriptive names
- `gift_main.md`
- `gift_extensions.md`

**Supplements**: Letter prefix indicating order
- `A_math_foundations.md`
- `B_rigorous_proofs.md`
- etc.

**Documentation**: Purpose-based names
- `FAQ.md`
- `GLOSSARY.md`
- `EXPERIMENTAL_VALIDATION.md`

**Project files**: ALL_CAPS for visibility
- `README.md`
- `CHANGELOG.md`
- `STRUCTURE.md`

### PDF Files

PDF versions mirror markdown filenames:
- `gift-main.pdf` ← `gift_main.md`
- `gift_extensions.pdf` ← `gift_extensions.md`
- `Supp_A.pdf` ← `A_math_foundations.md`

## Navigation Guide

### Finding Specific Information

**Parameter Predictions**
→ `publications/gift_main.md` (Section 4: Dimensionless Observable Predictions)
→ `publications/supplements/C_complete_derivations.md` (detailed derivations)

**Mathematical Proofs**
→ `publications/supplements/B_rigorous_proofs.md` (9 exact relations)
→ `publications/supplements/A_math_foundations.md` (underlying mathematics)

**Experimental Comparison**
→ `publications/gift_main.md` (summary tables)
→ `publications/supplements/D_phenomenology.md` (detailed analysis)
→ `docs/EXPERIMENTAL_VALIDATION.md` (current status)

**Geometric Construction**
→ `publications/supplements/A_math_foundations.md` (K₇ manifold overview)
→ `publications/supplements/F_K7_metric.md` (explicit metric and harmonic forms)

**Falsification Tests**
→ `publications/supplements/E_falsification.md` (comprehensive criteria)
→ `docs/EXPERIMENTAL_VALIDATION.md` (experimental timeline)

**Definitions and Notation**
→ `docs/GLOSSARY.md` (all technical terms)
→ `publications/gift_main.md` (Section 1.4: Conventions)

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

