# CLAUDE.md - AI Assistant Guide for GIFT Framework

> **Complete Guide** (22,647 lines). For a quick reference, see [CLAUDE_QUICK.md](CLAUDE_QUICK.md) (~400 lines).

## Repository Overview

**GIFT (Geometric Information Field Theory)** is a theoretical physics framework that derives Standard Model parameters from E₈×E₈ exceptional Lie algebras via dimensional reduction. This repository contains:

- Theoretical publications and mathematical proofs
- Computational implementations in Python/Jupyter
- Statistical validation tools
- Machine learning for K₇ manifold metrics
- Interactive visualizations
- Documentation and educational materials

**Key Metrics**:
- Mean precision: 0.128% across 39 observables
- Parameter status: All quantities structurally determined (no continuous adjustable parameters)
- Mathematical rigor: 13 proven exact relations
- Version: 2.2.0 (current stable)

## Quick Reference

### Essential Files to Understand First

1. `README.md` - Complete framework overview with quick start
2. `STRUCTURE.md` - Repository navigation
3. `CONTRIBUTING.md` - Scientific standards and contribution process
4. `publications/gift_2_2_main.md` - Core theoretical paper (~1400 lines)

### Key Directories

```
GIFT/
├── publications/           # Main theoretical documents (v2.2)
│   ├── gift_2_2_main.md   # Core paper
│   ├── summary.txt        # Executive summary
│   ├── GIFT_v22_*.md      # Reference documents
│   ├── READING_GUIDE.md   # Navigation guide
│   ├── GLOSSARY.md        # Terminology
│   ├── supplements/       # 7 detailed mathematical documents (S1-S7)
│   └── *.ipynb            # Interactive notebooks
├── tests/                 # Main test suite (pytest)
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── regression/       # Regression tests for observable values
├── docs/                  # Additional documentation
│   ├── FAQ.md            # Common questions
│   ├── GLOSSARY.md       # Technical definitions
│   └── EXPERIMENTAL_VALIDATION.md
├── G2_ML/                 # Machine learning for K₇ metrics
├── statistical_validation/ # Monte Carlo & validation
├── assets/                # Visualizations, agents, tools
├── legacy/                # Archived versions (v1, v2.0, v2.1)
├── requirements.txt       # Python dependencies
└── quick_start.py        # Interactive launcher
```

## Codebase Structure and Organization

### Publication Structure

**Main Documents** (`publications/`):
- `gift_2_2_main.md` - Core theoretical framework with key results (~1400 lines)
- `summary.txt` - Executive summary (5-minute read)
- `GIFT_v22_Observable_Reference.md` - Complete 39-observable catalog
- `GIFT_v22_Geometric_Justifications.md` - Geometric derivation details
- `GIFT_v22_Statistical_Validation.md` - Statistical validation methods
- `READING_GUIDE.md` - Navigation by time and interest
- `GLOSSARY.md` - Terminology definitions

**Supplements** (`publications/supplements/`):
- `S1_mathematical_architecture.md` - E₈ structure, K₇ manifold, cohomology
- `S2_K7_manifold_construction.md` - TCS construction, G₂ holonomy, ML metrics
- `S3_torsional_dynamics.md` - Torsion tensor, geodesic flow, RG connection
- `S4_complete_derivations.md` - 13 proven relations + all observable derivations
- `S5_experimental_validation.md` - Data comparison, falsification protocol
- `S6_theoretical_extensions.md` - Quantum gravity, information theory, speculative
- `S7_dimensional_observables.md` - Absolute masses, scale bridge, cosmology

**Computational** (`assets/visualizations/`):
- `e8_root_system_3d.ipynb` - E8 240-root 3D visualization
- `precision_dashboard.ipynb` - All observables vs experiment
- `dimensional_reduction_flow.ipynb` - 496D -> 99D -> 4D animation

### Code Organization

**Python Modules**:
- `quick_start.py` - Interactive launcher for visualizations, docs, agents
- `statistical_validation/gift_v22_core.py` - Core validation module
- `G2_ML/` - Neural network training for K₇ metric extraction
- `assets/agents/` - Automated verification and maintenance tools

**Key Python Components**:
```
G2_ML/[version]/
├── G2_geometry.py       # Geometric calculations
├── G2_manifold.py       # K₇ manifold implementation
├── G2_phi_network.py    # Neural network architectures
├── G2_losses.py         # Loss functions for training
├── G2_train.py          # Training loops
├── G2_eval.py           # Evaluation and validation
└── G2_export.py         # Model export utilities
```

### Documentation Hierarchy

**Entry points for different audiences**:
- Quick overview: `README.md`
- Navigation: `STRUCTURE.md`
- Scientific details: `publications/gift_2_2_main.md`
- Mathematical rigor: `publications/supplements/S4_complete_derivations.md`
- Experimental validation: `docs/EXPERIMENTAL_VALIDATION.md`
- Definitions: `docs/GLOSSARY.md`
- Common questions: `docs/FAQ.md`

## Development Workflows

### Git Workflow

**Branch Strategy**:
- `main` - Stable releases only
- Feature branches: `feature/description`
- Bug fixes: `fix/description`
- Documentation: `docs/description`

**Commit Standards**:
- Clear, descriptive messages
- Reference issues when applicable
- One logical change per commit
- Scientific results require validation

**Pull Request Process**:
1. Fork repository and create feature branch
2. Make changes following scientific standards
3. Update `CHANGELOG.md` under "Unreleased"
4. Ensure notebooks run without errors
5. Submit PR with template filled out
6. Address review feedback
7. Maintain scientific rigor throughout

### CI/CD Workflows

**GitHub Actions** (`.github/workflows/`):
- `ci.yml` - Continuous integration tests
- `codeql-analysis.yml` - Security analysis
- `notebook-check.yml` - Notebook validation
- `validate-links.yml` - Link checking

**Pre-commit Checks**:
- Python code style (PEP 8)
- Notebook cell execution
- Link validation
- File organization

### Scientific Standards

**Status Classifications** (use consistently):
- **PROVEN**: Rigorous mathematical proof with complete derivation
- **TOPOLOGICAL**: Direct consequence of manifold structure
- **DERIVED**: Calculated from proven/topological results
- **THEORETICAL**: Theoretical justification, proof incomplete
- **PHENOMENOLOGICAL**: Empirical fit, theory in progress
- **EXPLORATORY**: Preliminary investigation

**Precision Reporting**:
- Use appropriate significant figures
- Include uncertainty estimates
- Compare with experimental values
- State assumptions clearly
- Document numerical methods

**Documentation Tone**:
- Sober and humble, avoiding hype
- Speculative where appropriate
- Balanced presentation of strengths AND limitations
- Report both successes and failures
- Maintain epistemic humility

## Key Conventions

### File Naming

**Markdown Documents**:
- Main papers: Descriptive names (`gift_main.md`)
- Supplements: Letter prefix (`A_math_foundations.md`)
- Documentation: Purpose-based (`FAQ.md`, `GLOSSARY.md`)
- Project files: ALL_CAPS (`README.md`, `CHANGELOG.md`)

**Python Files**:
- Modules: lowercase_with_underscores (`run_validation.py`)
- Classes: CamelCase (in code)
- Versioned: `G2_ML/0.X/` directories

**Notebooks**:
- Descriptive with version: `gift_v2_notebook.ipynb`
- Specific purpose: `gift_statistical_validation.ipynb`

### Notation and Mathematical Conventions

**Follow** `publications/gift_2_2_main.md` Section 1.4 and `docs/GLOSSARY.md`:

- E₈: Exceptional Lie algebra (dim 248)
- K₇: Compact 7-dimensional manifold with G₂ holonomy
- b₂(K₇) = 21: Second Betti number (harmonic 2-forms)
- b₃(K₇) = 77: Third Betti number (harmonic 3-forms)
- H* = 99: Total effective cohomological dimension (b₂ + b₃ + 1)
- p₂ = 2: Binary duality (dim(G₂)/dim(K₇) = 14/7)
- β₀ = π/8: Angular quantization (π/rank(E₈))
- Weyl_factor = 5: Pentagonal symmetry (from |W(E₈)| = 2¹⁴ × 3⁵ × 5² × 7)
- ξ = 5π/16: Correlation parameter (derived: (Weyl/p₂) × β₀)
- τ = 3472/891: Hierarchy parameter (exact rational, (496×21)/(27×99))
- det(g) = 65/32: K₇ metric determinant (TOPOLOGICAL)
- κ_T = 1/61: Global torsion magnitude (TOPOLOGICAL)
- sin²θ_W = 3/13: Weinberg angle (PROVEN)
- N_gen = 3: Number of fermion generations
- δ_CP = 197°: CP violation phase (PROVEN)
- α_s = √2/12: Strong coupling constant (TOPOLOGICAL)

**Always define new symbols on first use.**

### Code Style

**Python** (PEP 8):
- 4 spaces for indentation
- Max line length: 88 characters (Black formatter compatible)
- Docstrings for all functions
- Type hints where helpful
- Clear variable names

**Jupyter Notebooks**:
- Markdown cells explain each section
- Code cells are self-contained when possible
- Clear outputs for reproducibility
- Section headers with hierarchy
- Avoid very long cells (split for readability)

**Dependencies**:
- Listed in `requirements.txt`
- Request new dependencies with justification
- Prefer standard scientific stack (numpy, scipy, matplotlib)

## Working with Different Components

### Publications and Supplements

**When editing theoretical content**:
1. Maintain consistent notation (see `docs/GLOSSARY.md`)
2. Apply appropriate status classification
3. Include proofs or derivations
4. Update cross-references
5. Verify numerical calculations
6. Update related documents (main paper, supplements)

**Cross-referencing**:
- Equations: `(#eq:delta-cp)`
- Figures: `{#fig:e8-roots}`
- Sections: `(#sec:foundations)`
- External docs: `[Supplement S4](publications/supplements/S4_complete_derivations.md)`

### Jupyter Notebooks

**Running notebooks**:
```bash
# Install dependencies
pip install -r requirements.txt

# Launch specific notebook
jupyter notebook publications/gift_v2_notebook.ipynb

# Or use quick start
python quick_start.py
```

**Notebook organization**:
- Section 1: Parameters and setup
- Section 2: Core calculations
- Section 3: Sector-specific predictions
- Section 4: Experimental comparison
- Section 5: Visualizations

**Important**: Notebooks should be run top-to-bottom without errors.

### G2 Machine Learning Framework

**Current status** (see `G2_ML/COMPLETION_PLAN.md`):
- Versions 0.1 through 0.9a implemented
- b₂=21 harmonic 2-forms: ✓ Completed
- b₃=77 harmonic 3-forms: In progress (v0.8+)
- Yukawa tensor computation: Planned
- Hyperparameter optimization: Planned

**Working with G2_ML**:
```bash
cd G2_ML/0.9a/
# Each version is self-contained with its own notebooks
jupyter notebook Complete_G2_Metric_Training_v0_9a.ipynb
```

**Key files per version**:
- Training notebook: Complete training pipeline
- `G2_*.py` modules: Reusable components
- Results: Saved weights and validation metrics

### Statistical Validation

**Running validation** (`statistical_validation/`):
```bash
cd statistical_validation/
pip install -r requirements.txt
python run_validation.py
```

**Features**:
- Monte Carlo uncertainty propagation (1M samples)
- Sobol global sensitivity analysis
- Bootstrap validation
- Uncertainty quantification for all observables

**Output**: JSON files in `full_results/` with detailed statistics.

### Visualization Tools

**Interactive visualizations** (`assets/visualizations/`):
- `e8_root_system_3d.ipynb` - E₈ 240 roots in 3D
- `precision_dashboard.ipynb` - All observables vs experiment
- `dimensional_reduction_flow.ipynb` - 496D → 99D → 4D animation

**Launch**:
```bash
python quick_start.py
# Select option 1 for visualizations
```

**Or run directly**:
```bash
cd assets/visualizations/
jupyter notebook
```

### Automated Agents

**Available agents** (`assets/agents/`):
```bash
python quick_start.py
# Select option 3 for agents

# Or run directly:
python -m assets.agents.cli verify        # Verification
python -m assets.agents.cli unicode       # Unicode sanitizer
python -m assets.agents.cli docs          # Docs integrity
python -m assets.agents.cli notebooks     # Notebook discovery
python -m assets.agents.cli canonical     # Canonical monitor
```

**Purpose**: Automated maintenance, verification, and integrity checking.

## Common Tasks for AI Assistants

### Understanding the Framework

**To understand GIFT fundamentals**:
1. Read `README.md` - Overview and key results
2. Read `publications/summary.txt` - 5-minute executive summary
3. Review `publications/gift_2_2_main.md` Sections 1-4
4. Check `docs/GLOSSARY.md` for unfamiliar terms
5. Explore `docs/FAQ.md` for common questions

**To understand specific predictions**:
1. Check `publications/gift_2_2_main.md` Section 8 (summary tables)
2. Read detailed derivations in `publications/supplements/S4_complete_derivations.md`
3. Review mathematical foundations in `publications/supplements/S1_mathematical_architecture.md`
4. Check experimental status in `docs/EXPERIMENTAL_VALIDATION.md`

### Making Changes

**Adding new predictions**:
1. Derive from geometric structure with clear status classification
2. Add to appropriate supplement (usually Supplement S4)
3. Update summary in `publications/gift_2_2_main.md` Section 8
4. Verify numerically in notebook
5. Compare with experiment in Supplement S5
6. Update `CHANGELOG.md`

**Improving documentation**:
1. Identify unclear sections
2. Add explanations, examples, or cross-references
3. Maintain consistent notation
4. Update related documents
5. Check links remain valid

**Refining code**:
1. Follow PEP 8 style
2. Add docstrings and comments
3. Verify calculations match analytical expressions
4. Test edge cases
5. Update notebooks if core functions change

### Verification and Validation

**Before committing changes**:
- [ ] Notebooks run without errors
- [ ] Numerical precision maintained or improved
- [ ] Documentation updated
- [ ] Links validated
- [ ] Status classifications correct
- [ ] Cross-references intact
- [ ] `CHANGELOG.md` updated

**Scientific validation checklist**:
- [ ] Mathematical derivation is sound
- [ ] Computational results reproduce analytical formulas
- [ ] Experimental comparison included
- [ ] Uncertainties quantified
- [ ] Limitations stated clearly

### Updating with New Experimental Data

**When new measurements appear**:
1. Update experimental values in relevant supplements
2. Recalculate deviations
3. Update tables in `publications/gift_2_2_main.md` Section 8
4. Update precision metrics in `README.md`
5. Check `docs/EXPERIMENTAL_VALIDATION.md`
6. Re-run statistical validation if needed
7. Update `CHANGELOG.md` with new data sources
8. Note changes in precision/agreement

## Testing and Validation

### Running Tests with pytest

**Full test suite**:
```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test category
pytest tests/unit/
pytest tests/integration/
pytest tests/regression/
pytest giftpy_tests/
```

**Specific test files**:
```bash
pytest tests/unit/test_statistical_validation.py  # Sobol & MC tests
pytest tests/unit/test_mathematical_properties.py  # Mathematical invariants
pytest tests/regression/test_observable_values.py  # Observable regression tests
```

**Test documentation**: See `publications/tests/TEST_SYNTHESIS.md` for comprehensive test coverage analysis.

### Manual Testing

**Notebooks**:
```bash
# Run all notebooks to verify they execute
jupyter nbconvert --to notebook --execute publications/gift_v2_notebook.ipynb
```

**Python scripts**:
```bash
# Run validation
cd statistical_validation/
python run_validation.py

# Run agents
python -m assets.agents.cli verify
```

### Automated Testing

**CI Pipeline** runs on every push:
- Python syntax checking
- Notebook execution tests
- Link validation
- Code quality analysis (CodeQL)

**Local checks before pushing**:
```bash
# Validate links
npm install -g markdown-link-check
find . -name "*.md" -exec markdown-link-check {} \;

# Check notebooks execute
jupyter nbconvert --to notebook --execute [notebook].ipynb
```

### Validation Scripts

**Available validation tools**:
- `statistical_validation/run_validation.py` - Full statistical analysis
- `assets/agents/verifier.py` - General verification
- `assets/agents/docs_integrity.py` - Documentation checks
- `assets/agents/notebook_exec.py` - Notebook execution

## Special Considerations for AI Assistants

### Scientific Rigor

**Always maintain**:
- Mathematical correctness above all
- Honest reporting of both agreements and tensions
- Clear status classifications
- Appropriate level of uncertainty
- Speculative tone where warranted

**Never**:
- Make unsupported claims
- Hide limitations or failures
- Overstate significance
- Skip derivation steps
- Ignore experimental discrepancies

### Working with Theoretical Content

**When deriving new results**:
1. Start from established mathematical foundations
2. Show all steps in derivation
3. Assign appropriate status (PROVEN requires rigorous proof!)
4. Verify numerically
5. Compare with experiment
6. Identify falsification tests

**When improving existing derivations**:
1. Understand current derivation completely
2. Identify specific improvement
3. Maintain or increase rigor
4. Update precision if applicable
5. Document changes in `CHANGELOG.md`

### Numerical Precision

**Standards**:
- Use appropriate significant figures (typically 4-6 for predictions)
- Include uncertainty estimates
- Document numerical methods
- Verify against analytical expressions when possible
- Report absolute and relative deviations

**Example format**:
```
Observable: δ_CP
Experimental: 197° ± 24°
GIFT prediction: 197.3°
Absolute deviation: 0.3°
Relative deviation: 0.005%
Status: PROVEN (exact topological formula)
```

### Documentation Updates

**When updating docs**:
- Maintain existing structure and organization
- Use consistent formatting (markdown, no emojis)
- Update cross-references if structure changes
- Keep tone professional and humble
- Add to `CHANGELOG.md`

**Cross-document consistency**:
- Check that changes propagate to related documents
- Verify numerical values match across all files
- Update version numbers if applicable
- Maintain notation consistency

## Dependencies and Environment

### Python Environment

**Requirements** (`requirements.txt`):
```
Core: numpy, scipy, matplotlib, pandas, sympy
Statistical: scikit-learn, statsmodels
Visualization: plotly, seaborn, kaleido
Computing: jupyter, ipykernel, notebook, ipywidgets
```

**Python version**: 3.11 or higher (see `runtime.txt`)

**Setup**:
```bash
# Clone repository
git clone https://github.com/gift-framework/GIFT.git
cd GIFT

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### External Tools

**Binder**: Cloud Jupyter environment (no installation)
- Main notebook: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gift-framework/GIFT/main)

**Google Colab**: Alternative cloud environment
- Requires Google account
- Links in `README.md`

## File Organization Principles

### Directory Structure Rules

**Publications** (`publications/`):
- Main theoretical documents only
- Supplements in `supplements/` subdirectory
- PDFs in `pdf/` subdirectory (when available)
- Notebooks that implement theoretical content

**Documentation** (`docs/`):
- Supporting documentation only
- FAQ, glossary, guides
- No theoretical derivations (those go in publications)

**Code** (various):
- Python scripts in logical directories
- Versioned components in subdirectories
- Standalone tools at root level

### Version Control

**Semantic Versioning**:
- Major (2.0): Substantial framework changes
- Minor (2.1): New features, additional predictions
- Patch (2.0.1): Bug fixes, documentation corrections

**See `CHANGELOG.md`** for version history.

**Branching**:
- Create feature branches from `main`
- Never commit directly to `main`
- Use descriptive branch names

## Project-Specific Guidance

### The GIFT Philosophy

**Core principles** (see `docs/PHILOSOPHY.md`):
1. Mathematical primacy - geometry determines physics
2. Epistemic humility - speculative but testable
3. Information-theoretic foundations
4. Topological necessity over phenomenological fitting
5. Falsifiability is essential

**Maintain this philosophy** in all contributions.

### Zero-Parameter Paradigm

Version 2.2 achieves structural determination: all quantities derive from fixed topological structure with **no continuous adjustable parameters** (see `publications/gift_2_2_main.md` Section 1.4).

**Structural inputs** (discrete mathematical choices):
- E₈×E₈ gauge group (dimension 496)
- K₇ manifold with G₂ holonomy (b₂=21, b₃=77)

**Key topological constants** (derived, not fitted):
- **sin²θ_W = 3/13**: Weinberg angle (b₂/(b₃ + dim(G₂)))
- **κ_T = 1/61**: Torsion magnitude (1/(b₃ - dim(G₂) - p₂))
- **det(g) = 65/32**: Metric determinant (topological formula)
- **τ = 3472/891**: Hierarchy parameter (exact rational)

### Exact Relations

**Thirteen proven exact relations** (from Supplement S4):
1. N_gen = 3 (generation number)
2. p₂ = 2 (binary duality)
3. Q_Koide = 2/3 (Koide parameter)
4. m_s/m_d = 20 (quark mass ratio)
5. δ_CP = 197° (CP violation phase)
6. m_τ/m_e = 3477 (lepton mass ratio)
7. Ω_DE = ln(2)×98/99 (dark energy density)
8. n_s = ζ(11)/ζ(5) (spectral index)
9. ξ = 5π/16 (correlation parameter)
10. λ_H = √17/32 (Higgs coupling)
11. sin²θ_W = 3/13 (Weinberg angle)
12. τ = 3472/891 (hierarchy parameter)
13. det(g) = 65/32 (metric determinant)

**These must remain exact** in any framework modifications.

### Experimental Validation

**Current status** (`docs/EXPERIMENTAL_VALIDATION.md`):
- 39 observables predicted
- Mean deviation: 0.128%
- Best predictions: 0.00% (δ_CP, m_s/m_d, n_s)
- All predictions: <1.0%

**Update regularly** with new experimental data from:
- Particle Data Group (PDG)
- NuFIT (neutrino parameters)
- Planck/WMAP (cosmology)
- LHC, Belle II, DUNE (ongoing experiments)

## Quick Command Reference

```bash
# Setup
git clone https://github.com/gift-framework/GIFT.git
cd GIFT
pip install -r requirements.txt

# Interactive launcher
python quick_start.py

# Run main notebook
jupyter notebook publications/gift_v2_notebook.ipynb

# Statistical validation
cd statistical_validation/
python run_validation.py

# Agents
python -m assets.agents.cli verify
python -m assets.agents.cli docs

# Documentation
open docs/index.html  # or your browser

# Testing
jupyter nbconvert --to notebook --execute [notebook].ipynb
```

## Resources and Links

### Internal Documentation
- `README.md` - Overview
- `STRUCTURE.md` - Organization
- `CONTRIBUTING.md` - Guidelines
- `docs/FAQ.md` - Questions
- `docs/GLOSSARY.md` - Definitions
- `CHANGELOG.md` - History

### External Links
- Repository: https://github.com/gift-framework/GIFT
- Issues: https://github.com/gift-framework/GIFT/issues
- Binder: https://mybinder.org/v2/gh/gift-framework/GIFT/main

### Key Papers (in `publications/`)
- `gift_2_2_main.md` - Core framework
- `supplements/S1_mathematical_architecture.md` - Mathematics
- `supplements/S4_complete_derivations.md` - Proofs + all derivations
- `supplements/S5_experimental_validation.md` - Testability

## Contact and Support

**For questions**:
1. Check `docs/FAQ.md`
2. Review relevant supplements
3. Open an issue: https://github.com/gift-framework/GIFT/issues

**For contributions**:
1. Read `CONTRIBUTING.md`
2. Check `STRUCTURE.md`
3. Follow scientific standards
4. Submit PR with template

## Summary for AI Assistants

**When working with this repository**:
1. **Understand the science** - This is theoretical physics, not just code
2. **Maintain rigor** - Mathematical correctness is paramount
3. **Document thoroughly** - Show all steps in derivations
4. **Validate numerically** - Verify predictions computationally
5. **Compare with experiment** - Always include experimental comparison
6. **Be humble** - Framework is speculative, maintain appropriate tone
7. **Follow conventions** - Notation, status classifications, file organization
8. **Update related files** - Changes propagate across documents
9. **Test thoroughly** - Notebooks must execute without errors
10. **Respect the philosophy** - Mathematical primacy and epistemic humility

**This is not just a software project** - it's a scientific framework making testable predictions about fundamental physics. Maintain the highest standards of scientific integrity, reproducibility, intellectual honesty and no emojis.

---

**Version**: 1.2.0 (2025-11-27)
**For**: GIFT Framework v2.2.0 (zero-parameter paradigm)
**Maintained by**: GIFT Framework Team
**License**: MIT (same as repository)

For updates to this guide, see repository commits and CHANGELOG.md.
