# GIFT v1.0 - Archived Version

**Status**: Archived
**Date**: 2024
**Current Version**: See main repository for GIFT v2.0.0+

## About This Directory

This directory placeholder exists to resolve documentation references to GIFT v1.0. The actual v1.0 codebase and documentation are not stored in the current repository structure but are accessible through git history.

## Accessing v1.0

GIFT v1.0 was the initial public release demonstrating geometric derivation of Standard Model parameters from E₈×E₈ structure. It has been superseded by v2.0.0 with substantial improvements.

### Via Git History

To access v1.0 code and documentation:

```bash
# Find v1.0 commits
git log --all --grep="v1.0" --oneline

# Or search by date (v1.0 was released in 2024)
git log --all --before="2025-01-01" --oneline

# Check out specific commit
git checkout <commit-hash>

# Or view specific file from v1.0
git show <commit-hash>:path/to/file
```

## What Changed in v2.0

v2.0.0 (released 2025-10-24) represents a major advancement:

### Key Improvements

**Precision**:
- v1.0: ~1-2% mean deviation
- v2.0: 0.13% mean deviation (10× improvement)

**Rigor**:
- v1.0: Preliminary derivations
- v2.0: 9 proven exact relations with complete mathematical proofs

**Coverage**:
- v1.0: ~20 observables
- v2.0: 34 dimensionless observables

**Documentation**:
- v1.0: Single document
- v2.0: Modular system with 6 mathematical supplements

**Parameters**:
- v1.0: 4 geometric parameters
- v2.0: 3 parameters (ξ = 5β₀/2 proven relation)

### Structural Changes

v2.0 introduced:
- Complete neutrino sector (all 4 mixing parameters)
- Cosmological observables (Ω_DE = ln(2), Hubble parameter)
- Binary information architecture
- Dual origin derivations for key parameters
- G2 machine learning framework for K₇ metrics
- Comprehensive statistical validation (Monte Carlo, Sobol, Bootstrap)
- Full test suite (200+ tests, ~85% coverage)

## Why v1.0 is Not in Main Tree

**Rationale**:
1. **Superseded**: v2.0 provides strictly better results
2. **Incompatible**: Different parameter conventions and notation
3. **Confusing**: Keeping both would confuse users
4. **Maintainability**: Single version reduces maintenance burden

**Historical Record**: v1.0 remains in git history for:
- Scientific reproducibility
- Historical context
- Attribution and credit
- Comparison of methodological evolution

## Migration from v1.0 to v2.0

If you were using v1.0 code or notebooks:

### Parameter Mapping

| v1.0 Parameter | v2.0 Equivalent | Notes |
|----------------|-----------------|-------|
| β₀ | β₀ = 1/(4π²) | Same, now with proven normalization |
| ξ | ξ = 5β₀/2 | Now **derived**, not free! |
| ε₀ | ε₀ = 1/8 | Same |
| κ (if used) | Removed | Replaced by exact relation |

### Notebook Updates

v1.0 notebooks should be replaced with:
- `publications/gift_v2_notebook.ipynb` - Main implementation
- `publications/gift_statistical_validation.ipynb` - Statistical analysis
- `publications/gift_experimental_predictions.ipynb` - Future predictions

### Formula Changes

Many v2.0 formulas have been refined. Consult:
- `publications/supplements/C_complete_derivations.md` - All 34 observables
- `publications/supplements/B_rigorous_proofs.md` - Proven exact relations

### Code APIs

v1.0 Python code (if any existed) is incompatible. Use v2.0:
- Statistical validation: `statistical_validation/run_validation.py`
- G2 ML framework: `G2_ML/` (new in v2.0)
- Interactive launcher: `quick_start.py` (new in v2.0)

## Key v1.0 Features (Historical)

For historical reference, v1.0 included:

**Core Framework**:
- E₈×E₈ → AdS₄×K₇ dimensional reduction
- Basic parameter predictions
- Preliminary neutrino sector
- Prototype computational notebook

**Predictions** (~20 observables):
- Fine structure constant α
- Weak mixing angle sin²θ_W
- Strong coupling α_s
- Some quark/lepton mass ratios
- Partial CKM matrix
- Preliminary neutrino mixing

**Limitations** (addressed in v2.0):
- ~1-2% precision (now 0.13%)
- Limited mathematical rigor (now 9 proven relations)
- Incomplete phenomenology (now comprehensive)
- No statistical validation (now rigorous)
- Single monolithic document (now modular)

## Citation

If citing historical v1.0 work, use:

```bibtex
@article{gift_v1_2024,
  title={GIFT: Geometric Information Field Theory (v1.0)},
  author={{GIFT Framework Team}},
  year={2024},
  note={Archived version, see v2.0+ for current framework},
  url={https://github.com/gift-framework/GIFT}
}
```

For current work, cite v2.0:

```bibtex
@article{gift_v2_2025,
  title={GIFT: Geometric Information Field Theory},
  author={{GIFT Framework Team}},
  year={2025},
  version={2.0.0},
  url={https://github.com/gift-framework/GIFT}
}
```

See `CITATION.md` in main repository for complete citation information.

## Support

**For v1.0 questions**: Best effort only, v2.0 is recommended

**For current framework**: See main `README.md`, `QUICK_START.md`, and documentation in `docs/`

**Issues**: https://github.com/gift-framework/GIFT/issues

## See Also

- `CHANGELOG.md` - Complete version history
- `README.md` - Current framework overview
- `publications/gift_main.md` - v2.0 theoretical paper
- `publications/supplements/` - Mathematical details

---

**Bottom line**: v1.0 is archived in git history. Use v2.0.0+ for all current work.

**Last Updated**: 2025-11-16
**Repository**: https://github.com/gift-framework/GIFT
**License**: MIT

