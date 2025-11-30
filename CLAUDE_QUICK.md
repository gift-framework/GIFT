# CLAUDE_QUICK.md - Quick Reference for AI Assistants

> **Condensed guide for the GIFT Framework**. For complete details, see [CLAUDE.md](CLAUDE.md).

## What is GIFT?

**GIFT (Geometric Information Field Theory)** derives Standard Model parameters from E₈×E₈ topology.

**Key Metrics**:
- 0.128% mean precision across 39 observables
- 19 parameters (SM) → 0 continuous adjustable parameters (GIFT)
- 13 rigorously proven exact relations
- Version: 2.2.0

## Essential Files (Read These First)

1. **README.md** - Framework overview
2. **STRUCTURE.md** - Repository organization
3. **publications/markdown/gift_2_2_main.md** - Core theoretical paper (~1400 lines)
4. **docs/GLOSSARY.md** - Technical definitions
5. **CONTRIBUTING.md** - Scientific standards

## Repository Structure

```
GIFT/
├── publications/           # Main theoretical documents (v2.2)
│   ├── README.md          # Overview, reading guide, summary
│   ├── markdown/          # Core paper + S1-S7 supplements
│   ├── references/        # Observable reference docs
│   ├── pdf/               # Generated PDFs
│   └── tex/               # LaTeX sources
├── docs/                  # FAQ, glossary, validation
├── G2_ML/                 # Machine learning (90% complete)
├── statistical_validation/ # Monte Carlo validation
├── assets/                # Visualizations, agents, tools
├── tests/                 # 200+ tests
└── requirements.txt       # Dependencies
```

## Quick Navigation

**For theorists**: Start with `publications/markdown/gift_2_2_main.md`
**For experimentalists**: See `docs/EXPERIMENTAL_VALIDATION.md`
**For coders**: Check `statistical_validation/run_validation.py`
**For questions**: Read `docs/FAQ.md`

## Key Conventions

### Notation (see docs/GLOSSARY.md)

- **E₈**: Exceptional Lie algebra (dim 248)
- **K₇**: Compact 7D manifold with G₂ holonomy (b₂=21, b₃=77)
- **H* = 99**: Total effective cohomological dimension (b₂ + b₃ + 1)
- **β₀ = π/8**: Angular quantization (π/rank(E₈))
- **ξ = 5π/16**: Correlation parameter (DERIVED: Weyl/p₂ × β₀)
- **τ = 3472/891**: Hierarchy parameter (exact rational)
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

- Publications: `gift_2_2_main.md`, `GIFT_v22_*.md`
- Supplements: `S1_mathematical_architecture.md`, `S2_K7_manifold_construction.md`, etc.
- Project files: ALL_CAPS (`README.md`, `CHANGELOG.md`)
- Python: lowercase_with_underscores

## Common Tasks

### Understanding the Framework

1. Read `README.md` for overview
2. Read `publications/markdown/gift_2_2_main.md` Sections 1-4
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

# Visualization notebooks
jupyter notebook assets/visualizations/

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

## Thirteen Proven Exact Relations

From S4_complete_derivations.md (authoritative list):

1. **N_gen = 3**: Three generations
2. **p₂ = 2**: Binary duality (dim(G₂)/dim(K₇))
3. **Q_Koide = 2/3**: Koide formula parameter
4. **m_s/m_d = 20**: Quark mass ratio
5. **δ_CP = 197°**: CP violation phase
6. **m_τ/m_e = 3477**: Lepton mass ratio
7. **Ω_DE = ln(2)×98/99**: Dark energy density
8. **n_s = ζ(11)/ζ(5)**: Spectral index
9. **ξ = 5π/16**: Correlation parameter
10. **λ_H = √17/32**: Higgs coupling
11. **sin²θ_W = 3/13**: Weinberg angle
12. **τ = 3472/891**: Hierarchy parameter
13. **det(g) = 65/32**: K₇ metric determinant

## Zero-Parameter Framework

GIFT v2.2 achieves **zero continuous adjustable parameters**:

- **Structural inputs**: E₈×E₈ gauge group, K₇ with G₂ holonomy (discrete choices)
- **All constants derived**: β₀, ξ, τ, det(g), κ_T follow from topology
- **No fitting**: All 39 observables are structurally determined

## Document Hierarchy

### Publications (publications/)

- **README.md**: Overview, reading guide, executive summary
- **markdown/gift_2_2_main.md**: Core framework (~1,400 lines)
- **markdown/S1-S7**: Supplements (mathematical details)
- **references/GIFT_v22_Observable_Reference.md**: Complete 39-observable catalog
- **references/GIFT_v22_Geometric_Justifications.md**: Geometric derivation details
- **references/GIFT_v22_Statistical_Validation.md**: Statistical validation methods

### Supplements (publications/markdown/)

- **S1**: Mathematical architecture (E₈, K₇, cohomology)
- **S2**: K₇ manifold construction (TCS, G₂ holonomy)
- **S3**: Torsional dynamics (torsion tensor, RG)
- **S4**: Complete derivations (13 proven + all observables)
- **S5**: Experimental validation (data comparison, falsification)
- **S6**: Theoretical extensions (quantum gravity, speculative)
- **S7**: Dimensional observables (absolute masses, cosmology)

## Code Organization

### Python Modules

- `quick_start.py` - Interactive launcher
- `statistical_validation/run_validation.py` - Validation script
- `G2_ML/` - Neural networks for K₇ metric (90% complete)
- `assets/agents/` - Automated verification tools

### Visualizations (`assets/visualizations/`)

- `e8_root_system_3d.ipynb` - E8 root structure
- `precision_dashboard.ipynb` - All observables
- `dimensional_reduction_flow.ipynb` - 496D -> 4D flow

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
- 39 observables
- 13 PROVEN exact relations (100% coverage)
- G2 geometry
- Error handling

## G2_ML Status

**Version**: 0.9a (90% complete)

| Component | Status |
|-----------|--------|
| b₂=21 harmonic 2-forms | Complete |
| b₃=77 harmonic 3-forms | In progress |
| Yukawa tensor | Planned |
| Optimization | Planned |

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

**Current status**: 39 observables predicted (27 dimensionless + 12 dimensional)

**Best predictions** (0.00% deviation):
- δ_CP = 197° (CP violation)
- m_s/m_d = 20 (quark ratio)
- n_s (spectral index)

**All predictions**: <1.0% deviation, mean 0.128%

See `docs/EXPERIMENTAL_VALIDATION.md` for updates.

## Common Questions

**Q: How many parameters?**
A: Zero continuous adjustable parameters. All quantities are structurally determined from E₈×E₈ and K₇ topology.

**Q: What's PROVEN vs TOPOLOGICAL?**
A: PROVEN = rigorous mathematical proof. TOPOLOGICAL = direct consequence of manifold structure.

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

**Current stable**: v2.2.0 (2025-11-27)

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

# Visualizations
jupyter notebook assets/visualizations/

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

- CLAUDE.md: ~22,000 lines (complete guide)
- CLAUDE_QUICK.md: This file (quick reference)
- gift_2_2_main.md: ~1,400 lines (core paper)
- All supplements (S1-S7): ~8,000 lines total

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
- 0.128% mean precision, 39 observables
- Zero continuous adjustable parameters
- 13 proven exact relations

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

**Quick Reference Version**: 2.2.0
**Full Guide**: CLAUDE.md
**For**: GIFT Framework v2.2.0
**Last Updated**: 2025-11-27

**Remember**: Mathematical correctness and intellectual honesty above all else. (and never use emojis in markdown or python files)
