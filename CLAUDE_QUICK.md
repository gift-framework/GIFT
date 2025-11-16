# CLAUDE_QUICK.md - Quick Reference for AI Assistants

> **Condensed guide for the GIFT Framework**. For complete details, see [CLAUDE.md](CLAUDE.md).

## What is GIFT?

**GIFT (Geometric Information Field Theory)** derives Standard Model parameters from E₈×E₈ topology.

**Key Metrics**:
- 0.13% mean precision across 34 observables
- 19 parameters (SM) → 3 parameters (GIFT)
- 9 rigorously proven exact relations
- Version: 2.0.0

## Essential Files (Read These First)

1. **README.md** - Framework overview
2. **STRUCTURE.md** - Repository organization
3. **publications/gift_main.md** - Core theoretical paper (~1100 lines)
4. **docs/GLOSSARY.md** - Technical definitions
5. **CONTRIBUTING.md** - Scientific standards

## Repository Structure

```
GIFT/
├── publications/           # Main theoretical documents
│   ├── gift_main.md       # Core paper
│   ├── supplements/       # 6 detailed supplements (A-F)
│   └── *.ipynb           # Interactive notebooks
├── docs/                  # FAQ, glossary, validation
├── G2_ML/                 # Machine learning (90% complete)
├── statistical_validation/ # Monte Carlo validation
├── assets/                # Visualizations, agents, tools
├── tests/                 # 200+ tests
└── requirements.txt       # Dependencies
```

## Quick Navigation

**For theorists**: Start with `publications/gift_main.md`
**For experimentalists**: See `docs/EXPERIMENTAL_VALIDATION.md`
**For coders**: Check `statistical_validation/run_validation.py`
**For questions**: Read `docs/FAQ.md`

## Key Conventions

### Notation (see docs/GLOSSARY.md)

- **E₈**: Exceptional Lie algebra (dim 248)
- **K₇**: Compact 7D manifold with G₂ holonomy
- **β₀ = 1/(4π²)**: Base coupling
- **ξ = 5β₀/2**: Correlation parameter (DERIVED, not free!)
- **ε₀ = 1/8**: Symmetry breaking scale
- **N_gen = 3**: Number of generations

### Status Classifications

Use these consistently:

- **PROVEN**: Rigorous mathematical proof
- **TOPOLOGICAL**: Direct consequence of manifold structure
- **DERIVED**: Calculated from proven/topological results
- **THEORETICAL**: Theory incomplete
- **PHENOMENOLOGICAL**: Empirical fit
- **EXPLORATORY**: Preliminary

### File Naming

- Publications: `gift_main.md`, `gift_extensions.md`
- Supplements: `A_math_foundations.md`, `B_rigorous_proofs.md`, etc.
- Project files: ALL_CAPS (`README.md`, `CHANGELOG.md`)
- Python: lowercase_with_underscores

## Common Tasks

### Understanding the Framework

1. Read `README.md` for overview
2. Read `publications/gift_main.md` Sections 1-3
3. Check `docs/GLOSSARY.md` for terms
4. See `docs/FAQ.md` for questions

### Making Changes

1. Follow scientific standards (see CONTRIBUTING.md)
2. Use appropriate status classification
3. Update related documents (cross-references!)
4. Update `CHANGELOG.md`
5. Verify numerically

### Running Code

```bash
# Install dependencies
pip install -r requirements.txt

# Interactive launcher
python quick_start.py

# Main notebook
jupyter notebook publications/gift_v2_notebook.ipynb

# Statistical validation
cd statistical_validation/
python run_validation.py

# Tests
pytest tests/
```

## Scientific Standards

**Always**:
- Maintain mathematical correctness
- Report both successes AND limitations
- Use appropriate status classifications
- Include uncertainty estimates
- Show derivation steps

**Never**:
- Make unsupported claims
- Hide failures or tensions
- Skip proofs (for PROVEN status)
- Overstate significance

## Nine Exact Relations (PROVEN)

1. **N_gen = 3**: Three generations (topological)
2. **Q_Koide = 2/3**: Koide formula parameter
3. **m_s/m_d = 20**: Quark mass ratio
4. **δ_CP = 197°**: CP violation phase
5. **m_τ/m_e = 3477**: Lepton mass ratio
6. **Ω_DE = ln(2)**: Dark energy density
7. **ξ = 5β₀/2**: Parameter relation
8-9. Dual derivations (see Supplement B)

All have precision <0.01% vs experiment.

## Three Parameters

GIFT uses only 3 geometric parameters:

1. **β₀ = 1/(4π²)**: From E₈ normalization
2. **ξ = 5β₀/2**: DERIVED (not free!)
3. **ε₀ = 1/8**: From G₂ structure

Effectively 2 free parameters (ξ derived).

## Document Hierarchy

### Publications

- **gift_main.md**: Core framework (1,120 lines)
- **gift_extensions.md**: Dimensional observables (695 lines)
- **gift_technical.md**: Technical details (4,686 lines)

### Supplements (publications/supplements/)

- **A**: Mathematical foundations (E₈, K₇, reduction)
- **B**: Rigorous proofs (9 exact relations)
- **C**: Complete derivations (all 34 observables)
- **D**: Phenomenology (experimental comparison)
- **E**: Falsification (testability criteria)
- **F**: K₇ metric (explicit construction)

## Code Organization

### Python Modules

- `quick_start.py` - Interactive launcher
- `statistical_validation/run_validation.py` - Validation script
- `G2_ML/` - Neural networks for K₇ metric (90% complete)
- `assets/agents/` - Automated verification tools

### Notebooks

- `gift_v2_notebook.ipynb` - Main implementation
- `gift_statistical_validation.ipynb` - 1M Monte Carlo samples
- `gift_experimental_predictions.ipynb` - DUNE predictions

## Git Workflow

**Branches**:
- `main` - Stable releases only
- Feature: `feature/description`
- Fixes: `fix/description`
- Docs: `docs/description`

**Commits**:
- Clear, descriptive messages
- One logical change per commit
- Scientific results require validation

**Pull Requests**:
1. Create feature branch
2. Make changes following standards
3. Update `CHANGELOG.md`
4. Run tests (`pytest tests/`)
5. Submit PR

## Testing

**Run tests**:
```bash
pytest tests/                    # All tests
pytest tests/unit/              # Unit tests only
pytest tests/integration/       # Integration tests
pytest -m "not slow"            # Fast tests only
```

**200+ tests** covering:
- 34 observables
- 9 PROVEN exact relations (100% coverage)
- G2 geometry
- Error handling

## G2_ML Status

**Version**: 0.9a (90% complete)

| Component | Status |
|-----------|--------|
| b₂=21 harmonic 2-forms | ✅ Complete |
| b₃=77 harmonic 3-forms | 🔨 In progress |
| Yukawa tensor | 📋 Planned |
| Optimization | 📋 Planned |

See `G2_ML/STATUS.md` for details.

## Pattern Explorer

**20 scripts** organized in 6 categories:

Use unified CLI:
```bash
cd assets/pattern_explorer
python3 pattern_explorer_cli.py
```

See `assets/pattern_explorer/SCRIPTS_GUIDE.md` for workflows.

## Validation Checklist

Before committing:

- [ ] Notebooks run without errors
- [ ] Numerical precision maintained
- [ ] Documentation updated
- [ ] Links validated
- [ ] Status classifications correct
- [ ] Cross-references intact
- [ ] `CHANGELOG.md` updated
- [ ] Tests pass (`pytest tests/`)

## Experimental Validation

**Current status**: 34 observables predicted

**Best predictions** (<0.01%):
- α (fine structure constant)
- δ_CP (CP violation)
- Q_Koide (Koide formula)

**All predictions**: <1.0% deviation

See `docs/EXPERIMENTAL_VALIDATION.md` for updates.

## Common Questions

**Q: How many parameters?**
A: 3 (β₀, ξ, ε₀), but ξ is derived, so effectively 2.

**Q: What's PROVEN vs DERIVED?**
A: PROVEN = rigorous proof. DERIVED = calculated from PROVEN results.

**Q: Can I modify predictions?**
A: Yes, but clearly note changes. Don't claim PROVEN without proof.

**Q: How to update experimental data?**
A: Update supplements, recalculate deviations, update README metrics.

**Q: Where are the tests?**
A: `tests/` directory, 200+ tests, run with `pytest`.

## Tools and Automation

**Agents** (automated tools):
```bash
python -m assets.agents.cli verify        # Verification
python -m assets.agents.cli docs          # Docs integrity
python -m assets.agents.cli notebooks     # Notebook discovery
```

**PDF sync check**:
```bash
python3 tools/check_pdf_sync.py
```

## Version Control

**Current stable**: v2.0.0 (2025-10-24)
**In development**: v2.1 (unreleased)

See `CHANGELOG.md` for history.

**Semantic versioning**:
- Major (2.0): Framework changes
- Minor (2.1): New features
- Patch (2.0.1): Bug fixes

## Dependencies

```bash
pip install -r requirements.txt
```

**Core**: numpy, scipy, matplotlib, pandas
**Statistical**: scikit-learn, statsmodels
**Visualization**: plotly, seaborn
**Computing**: jupyter, notebook

Python ≥3.11 recommended.

## Philosophy

From `docs/PHILOSOPHY.md`:

1. **Mathematical primacy** - Geometry determines physics
2. **Epistemic humility** - Speculative but testable
3. **Information-theoretic foundations**
4. **Topological necessity** over fitting
5. **Falsifiability is essential**

Maintain this in all contributions.

## Tone and Style

**Professional**:
- Sober and humble
- Speculative where appropriate
- Balanced (strengths AND limitations)
- Honest about uncertainties

**Avoid**:
- Hype or overclaiming
- Hiding limitations
- Excessive enthusiasm
- Unqualified assertions

## Contact and Support

**Issues**: https://github.com/gift-framework/GIFT/issues
**Repository**: https://github.com/gift-framework/GIFT
**Documentation**: Start with `README.md`

## Quick Command Reference

```bash
# Setup
git clone https://github.com/gift-framework/GIFT.git
cd GIFT
pip install -r requirements.txt

# Interactive
python quick_start.py

# Notebooks
jupyter notebook publications/gift_v2_notebook.ipynb

# Validation
cd statistical_validation/
python run_validation.py

# Tests
pytest tests/

# Agents
python -m assets.agents.cli verify

# Pattern explorer
cd assets/pattern_explorer
python3 pattern_explorer_cli.py
```

## Most Important Rules

1. **Read GLOSSARY.md** for notation
2. **Use correct status** classifications
3. **Update cross-references** when changing structure
4. **Run tests** before committing
5. **Update CHANGELOG.md**
6. **Maintain scientific rigor** above all
7. **Be honest** about limitations
8. **Show your work** (derivations!)
9. **Verify numerically**
10. **Check experimental data**

## File Size Reference

- CLAUDE.md: 22,647 lines (complete guide)
- CLAUDE_QUICK.md: This file (quick reference)
- gift_main.md: 1,120 lines (core paper)
- gift_technical.md: 4,686 lines (detailed)
- All supplements: ~6,640 lines total

## When to Use Full CLAUDE.md

Use the complete guide when:
- Setting up development environment (detailed instructions)
- Understanding specific workflows (complete examples)
- Working with notebooks (execution details)
- Debugging issues (comprehensive troubleshooting)
- Contributing code (detailed standards)

Otherwise, this quick reference should suffice.

## Summary

**GIFT is**:
- Theoretical physics framework
- E₈×E₈ → Standard Model parameters
- 0.13% mean precision, 34 observables
- 3 parameters (2 effective)
- 9 proven exact relations

**When working with GIFT**:
- Maintain mathematical rigor
- Use correct status classifications
- Update related files
- Test thoroughly
- Be humble about limitations
- Show all derivation steps

**This repository is**:
- World-class documentation
- Professional code quality
- Rigorous scientific standards
- Honest uncertainty reporting
- Reproducible and testable

---

**Quick Reference Version**: 1.0.0
**Full Guide**: CLAUDE.md (22,647 lines)
**For**: GIFT Framework v2.0.0+
**Last Updated**: 2025-11-16

**Remember**: Mathematical correctness and intellectual honesty above all else.
