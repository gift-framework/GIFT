# Pattern Explorer Scripts Guide

## Overview

The Pattern Explorer contains **20 specialized scripts** organized into 6 logical categories. Instead of running individual scripts, use the **unified CLI** for easier access.

## Quick Start

```bash
cd assets/pattern_explorer
python3 pattern_explorer_cli.py
```

This launches an interactive menu organizing all 20 scripts by category.

## Script Organization

### 1. Zeta Function Analysis (5 scripts)

Systematic exploration of zeta function patterns in GIFT observables.

| Script | Purpose | Output |
|--------|---------|--------|
| `refined_zeta_analysis.py` | Core zeta pattern analysis | Refined pattern rankings |
| `zeta_ratio_discovery.py` | Discover zeta value ratios | Ratio patterns with significance |
| `extended_zeta_analysis.py` | Higher-order zeta analysis | Extended pattern catalog |
| `odd_zeta_systematic_search.py` | Focus on odd zeta values | Odd zeta discoveries |
| `visualize_zeta_patterns.py` | Generate visualizations | PNG plots in `visualizations/` |

**Key discoveries**: Multiple observables match odd zeta combinations with >5σ significance.

### 2. Mathematical Constants (2 scripts)

Search for fundamental mathematical constants in observable values.

| Script | Purpose | Output |
|--------|---------|--------|
| `golden_ratio_search.py` | Golden ratio (φ) patterns | φ-based formulas |
| `feigenbaum_analysis.py` | Feigenbaum constant (δ) | Chaos theory connections |

**Status**: Exploratory. Some promising patterns but not yet PROVEN.

### 3. Number Theory (3 scripts)

Explore number-theoretic patterns and integer relationships.

| Script | Purpose | Output |
|--------|---------|--------|
| `integer_factorization_search.py` | Integer factorization patterns | Factor-based expressions |
| `number_theory_search.py` | Comprehensive NT search | Prime, divisor, modular patterns |
| `binary_modular_search.py` | Binary/modular arithmetic | Powers of 2, modular relations |

**Key insight**: Binary patterns appear frequently (powers of 2, factors of 8).

### 4. Systematic Exploration (5 scripts)

General-purpose pattern discovery across multiple mathematical domains.

| Script | Purpose | Scope |
|--------|---------|-------|
| `systematic_explorer.py` | General systematic search | All pattern types |
| `deep_dive_explorer.py` | In-depth specific patterns | Focused deep analysis |
| `extended_pattern_search.py` | Extended multi-type search | Broader coverage |
| `higher_order_systematic_search.py` | Higher-order patterns | Complex expressions |
| `quick_explorer.py` | Fast preliminary scan | Quick overview |

**Usage**: Start with `quick_explorer.py`, then use `systematic_explorer.py` for comprehensive search.

### 5. Validation & Statistics (3 scripts)

Validate discovered patterns and assess statistical significance.

| Script | Purpose | Output |
|--------|---------|--------|
| `comprehensive_validator.py` | Full validation suite | Validation reports |
| `statistical_validation.py` | Statistical testing | Significance metrics |
| `statistical_significance_analyzer.py` | Detailed significance analysis | P-values, confidence intervals |

**Critical**: Always validate patterns before promoting to PROVEN status.

### 6. Output Generation (2 scripts)

Generate reports and documentation from discoveries.

| Script | Purpose | Output |
|--------|---------|--------|
| `experimental_predictions_generator.py` | Generate prediction reports | Markdown reports |
| `scientific_paper_generator.py` | Draft paper sections | Paper-ready text |

**Usage**: Run after validation to generate documentation.

## Workflow Recommendations

### Discovery Workflow

```
1. Quick scan:          quick_explorer.py
2. Focused search:      [category-specific script]
3. Validation:          comprehensive_validator.py
4. Generate report:     experimental_predictions_generator.py
```

### Zeta Function Workflow

```
1. Refined analysis:    refined_zeta_analysis.py
2. Find ratios:         zeta_ratio_discovery.py
3. Visualize:           visualize_zeta_patterns.py
4. Validate:            statistical_validation.py
```

### Complete Analysis Workflow

```
1. Systematic search:   systematic_explorer.py
2. Deep dive:           deep_dive_explorer.py
3. Validate all:        comprehensive_validator.py
4. Statistical check:   statistical_significance_analyzer.py
5. Generate paper:      scientific_paper_generator.py
```

## Output Locations

All scripts follow consistent output conventions:

- **Data**: `data/` - CSV files with raw results
- **Visualizations**: `visualizations/` - PNG plots and figures
- **Reports**: `reports/` - Markdown analysis reports
- **Discoveries**: `discoveries/` - High-confidence patterns
- **Validation**: `validation_results/` - Validation metrics

## Script Details

### Total Code Size

- **20 scripts**
- **~350 KB** total
- **Average**: ~17 KB per script
- **Largest**: `comprehensive_validator.py` (44 KB)
- **Smallest**: `quick_explorer.py` (8.4 KB)

### Dependencies

All scripts use standard scientific Python stack:
- numpy
- scipy
- matplotlib
- pandas (for data output)
- mpmath (for high-precision calculations)

Install with:
```bash
pip install -r ../../requirements.txt
```

## Running Scripts Individually

If you prefer direct execution (not recommended - use CLI instead):

```bash
cd assets/pattern_explorer/scripts
python3 refined_zeta_analysis.py
```

Scripts generate output in parent directory (`../data/`, `../reports/`, etc.).

## Script Status

| Status | Count | Scripts |
|--------|-------|---------|
| Production | 8 | Core analysis and validation tools |
| Exploratory | 8 | Experimental pattern searches |
| Utility | 4 | Output generation and visualization |

## Consolidation Strategy

**Previous issue**: 20 separate scripts appeared complex and redundant.

**Solution**: Unified CLI (`pattern_explorer_cli.py`) provides:
- Single entry point for all tools
- Logical categorization
- Interactive menu
- Consistent interface

**Result**: Complexity reduced from "20 scripts to remember" to "1 CLI to run".

## Best Practices

### For New Users

1. Start with the CLI: `python3 pattern_explorer_cli.py`
2. Run Quick Explorer (#15) first
3. Check existing reports in `reports/`
4. Read `QUICK_REFERENCE.md` for summary

### For Pattern Discovery

1. Use systematic search tools first
2. Validate immediately (don't accumulate unvalidated patterns)
3. Check statistical significance (require >3σ minimum)
4. Document theoretical mechanism if found
5. Add to appropriate supplement if PROVEN

### For Validation

1. Run comprehensive validator on all new patterns
2. Check cross-validation across multiple observables
3. Verify statistical independence
4. Test against experimental uncertainties
5. Generate validation report

## Future Improvements

Planned consolidations:

1. **Merge similar scripts**:
   - Combine 3 zeta analysis scripts → 1 unified zeta analyzer
   - Merge 2 statistical validators → 1 comprehensive tool

2. **Create modular library**:
   - Extract common pattern matching code
   - Build reusable `pattern_matcher` module
   - Reduce code duplication

3. **Add configuration files**:
   - YAML/TOML configs for search parameters
   - Avoid hardcoded thresholds

4. **Automated workflow**:
   - Single command: "discover → validate → report"
   - CI/CD integration for continuous pattern discovery

## Questions?

- **CLI issues**: Check Python version (≥3.9 required)
- **Script errors**: Verify dependencies installed
- **Pattern questions**: See `QUICK_REFERENCE.md`
- **Validation**: Consult `VALIDATION_SUMMARY.md`

## Summary

The Pattern Explorer scripts are organized into a **coherent ecosystem**:

- **Unified CLI** for easy access
- **Logical categories** for navigation
- **Clear workflows** for different tasks
- **Consistent outputs** for reproducibility

**Use `pattern_explorer_cli.py` instead of running scripts individually.**

---

**Last updated**: 2025-11-16
**Total scripts**: 20
**Categories**: 6
**Lines of code**: ~8,000
