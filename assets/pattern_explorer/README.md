# GIFT Pattern Explorer

This directory contains systematic pattern discovery tools and results for the GIFT framework.

## Directory Structure

```
pattern_explorer/
├── reports/              # Pattern discovery reports and summaries
├── scripts/              # Analysis and search scripts
├── data/                 # CSV data files and raw results
├── visualizations/       # Generated plots and figures
├── discoveries/          # High-confidence pattern discoveries
├── validation_results/   # Validation reports and consistency checks
├── EXPERIMENTAL_PREDICTIONS.md
├── FRAMEWORK_STATUS_SUMMARY.md
├── QUICK_REFERENCE.md
└── VALIDATION_SUMMARY.md
```

## Contents

### Reports (`reports/`)
Comprehensive analysis reports from systematic pattern searches:
- Zeta function patterns (odd zeta values, ratios)
- Feigenbaum constant analysis
- Golden ratio relationships
- Binary modular patterns
- Number theory explorations
- Statistical validation results

### Scripts (`scripts/`)
Python scripts for systematic pattern discovery:
- `*_search.py`: Pattern search implementations
- `*_analysis.py`: Statistical and theoretical analysis
- `comprehensive_validator.py`: Validation framework
- `experimental_predictions_generator.py`: Prediction tools

### Data (`data/`)
Raw results and discoveries in CSV/TXT format:
- Pattern match results
- Statistical rankings
- Validated discoveries
- Binary rational expressions

### Visualizations (`visualizations/`)
Generated figures showing:
- Pattern distributions by observable
- Statistical significance plots
- Network graphs of relationships
- Top discoveries rankings

## Quick Reference

For a quick overview of discoveries and current status:
1. Start with `QUICK_REFERENCE.md` for high-level summary
2. See `FRAMEWORK_STATUS_SUMMARY.md` for derivation status
3. Check `VALIDATION_SUMMARY.md` for validation metrics
4. Browse `reports/` for detailed analysis

## Running Scripts

**Recommended**: Use the unified CLI for easy access to all tools:

```bash
cd assets/pattern_explorer
python3 pattern_explorer_cli.py
```

The CLI organizes all 20 scripts into logical categories with an interactive menu.

**Alternative**: Run scripts directly (not recommended):

```bash
cd assets/pattern_explorer/scripts
python3 golden_ratio_search.py
python3 zeta_ratio_discovery.py
```

Scripts generate results in `../data/` and visualizations in `../visualizations/`.

**Documentation**: See `SCRIPTS_GUIDE.md` for complete script organization and workflows.

## Key Discoveries

See `discoveries/high_confidence/` for patterns with:
- Multiple independent derivations
- Statistical significance > 5σ
- Theoretical mechanisms identified

## Validation

Validation tools ensure pattern robustness:
- Cross-validation across observables
- Statistical significance testing
- Consistency with known physics
- Independence checks

See `validation_results/` for detailed reports.
