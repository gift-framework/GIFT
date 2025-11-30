# Repository Navigation

Quick reference for finding specific content in the GIFT framework.

## Directory Layout

```
gift/
├── publications/                      # Theoretical documents
│   ├── README.md                     # Overview, reading guide, summary
│   ├── markdown/                     # Main documents
│   │   ├── gift_2_2_main.md         # Core paper (~1400 lines)
│   │   ├── S1_mathematical_architecture.md   # E₈, K₇, cohomology
│   │   ├── S2_K7_manifold_construction.md    # TCS, G₂, ML metrics
│   │   ├── S3_torsional_dynamics.md          # Geodesics, RG flow
│   │   ├── S4_complete_derivations.md        # 13 proofs + all derivations
│   │   ├── S5_experimental_validation.md     # Data, falsification
│   │   ├── S6_theoretical_extensions.md      # QG, info theory
│   │   └── S7_dimensional_observables.md     # Masses, cosmology
│   ├── references/                   # Quick reference documents
│   │   ├── GIFT_v22_Observable_Reference.md
│   │   ├── GIFT_v22_Geometric_Justifications.md
│   │   └── GIFT_v22_Statistical_Validation.md
│   ├── tex/                          # LaTeX sources
│   ├── pdf/                          # Generated PDFs
│   └── template/                     # Document templates
│
├── docs/                             # Supporting documentation
│   ├── FAQ.md                        # Common questions
│   ├── GLOSSARY.md                   # Technical terms
│   ├── PHILOSOPHY.md                 # Philosophical perspectives
│   ├── EXPERIMENTAL_VALIDATION.md    # Current status
│   └── tests/                        # Test documentation
│
├── assets/visualizations/            # Interactive notebooks
│   ├── precision_dashboard.ipynb     # 39 observables vs experiment
│   ├── e8_root_system_3d.ipynb      # 240-root visualization
│   └── dimensional_reduction_flow.ipynb
│
├── statistical_validation/           # Monte Carlo tools
├── G2_ML/                           # Neural network for K₇
├── tests/                           # pytest suite
└── legacy/                          # v1, v2.0, v2.1 archives
```

## Find What You Need

| Looking for... | Go to |
|----------------|-------|
| Framework overview | `README.md` |
| 5-minute summary | `publications/README.md` |
| Complete theory | `publications/markdown/gift_2_2_main.md` |
| All 39 observables | `publications/references/GIFT_v22_Observable_Reference.md` |
| Proofs (13 exact relations) | `publications/markdown/S4_complete_derivations.md` |
| Experimental comparison | `publications/markdown/S5_experimental_validation.md` |
| Falsification criteria | `publications/markdown/S5_experimental_validation.md` |
| Mathematical foundations | `publications/markdown/S1_mathematical_architecture.md` |
| Technical definitions | `docs/GLOSSARY.md` |
| Common questions | `docs/FAQ.md` |
| Interactive visualizations | `assets/visualizations/` |
| Validation code | `statistical_validation/` |

## Status Classifications

Results use these status labels:

| Status | Meaning |
|--------|---------|
| **PROVEN** | Exact topological identity with rigorous proof |
| **TOPOLOGICAL** | Direct consequence of manifold structure |
| **DERIVED** | Calculated from proven/topological relations |
| **THEORETICAL** | Theoretical justification, proof in progress |
| **PHENOMENOLOGICAL** | Empirically accurate, derivation in progress |

## File Naming

- **Main papers**: `gift_2_2_main.md`, `GIFT_v22_*.md`
- **Supplements**: `S1_*.md` through `S7_*.md`
- **Project files**: ALL_CAPS (`README.md`, `CHANGELOG.md`)

## Maintenance

When updating the framework:

1. Update relevant section in main paper or supplements
2. Update `CHANGELOG.md`
3. Regenerate PDFs if needed (`publications/tex/` → `publications/pdf/`)
4. Run notebooks to verify calculations
5. Update `docs/EXPERIMENTAL_VALIDATION.md` with new data
