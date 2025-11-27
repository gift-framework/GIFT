# Legacy Tests

This folder contains archived tests from previous versions of the GIFT framework.

## Structure

- `v21/` - GIFT v2.1 specific tests (archived with v2.2 release)
- `giftpy/` - giftpy library tests (will be synced with g2-forge project)
- `g2_ml/` - G2_ML notebook and integration tests (for g2-forge project)

## Contents

### v21/ (13 files)
- `test_gift_v21_framework.py` - v2.1 framework tests
- `test_v21_all_observables.py` - v2.1 46-observable tests
- `test_statistical_validation_v21.py` - v2.1 statistical validation
- `test_gift_framework.py` - Generic framework tests (v2.1)
- `test_mathematical_properties.py` - Mathematical property tests
- `test_statistical_validation.py` - Statistical validation tests
- `test_observables_parametrized.py` - Parametrized observable tests
- `test_observable_values.py` - Observable value regression
- `test_gift_pipeline.py` - Pipeline integration tests
- `test_input_validation.py` - Input validation tests
- `test_edge_cases_errors.py` - Edge case tests
- `test_property_based.py` - Property-based tests
- `test_benchmarks.py` - Performance benchmarks

### giftpy/ (1 file)
- `test_giftpy_comprehensive.py` - giftpy library tests

### g2_ml/ (4 files)
- `test_g2_ml_losses.py` - G2 ML loss function tests
- `test_g2_ml_networks.py` - G2 ML network architecture tests
- `test_g2_ml_integration.py` - G2 ML integration tests
- `test_notebook_execution_comprehensive.py` - Notebook execution tests

## Why Archived?

### v2.1 Tests
These tests were specific to the v2.1 framework implementation which used different formulas than v2.2's zero-parameter paradigm. Key differences:

- v2.1 used `zeta(2) - sqrt(2)` for sin^2(theta_W)
- v2.2 uses `3/13 = b2/(b3+dim(G2))` (PROVEN)

### giftpy Tests
The giftpy library will be synchronized with the new g2-forge project, which extends G2 metric handling universally. These tests are preserved for reference.

### G2_ML Tests
The G2_ML module will be developed further in the g2-forge project. These tests are preserved for migration.

## Running Legacy Tests

```bash
# Run v2.1 legacy tests (requires gift_v21_core)
pytest tests/legacy/v21/ -v

# Run giftpy legacy tests (requires giftpy)
pytest tests/legacy/giftpy/ -v

# Run G2_ML legacy tests (requires torch)
pytest tests/legacy/g2_ml/ -v
```

## Note

These tests are NOT included in the main test suite. They are preserved for:
- Historical reference
- Backwards compatibility testing
- Migration guides
- g2-forge development

---
*Archived: 2025-11-27*
*GIFT Framework v2.2.0*
