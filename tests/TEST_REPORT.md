# GIFT Framework - Test Execution Report

**Date**: 2025-11-27
**Branch**: claude/testing-mihq64gp4vga9o86-013NSSuHBB9UeSxa3ecrLVxb
**Python Version**: 3.11.14
**pytest Version**: 9.0.1

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Total Tests Collected** | 1,773 |
| **Passed** | 1,612 (90.9%) |
| **Failed** | 92 (5.2%) |
| **Skipped** | 70 (3.9%) |
| **Execution Time** | 15.62s |

---

## Test Categories Overview

### Unit Tests
| File | Status | Notes |
|------|--------|-------|
| `test_run_validation.py` | PASSED | New - validation runner tests |
| `test_agents_cli.py` | PASSED | New - CLI and registry tests |
| `test_canonical_monitor.py` | PASSED | New - state tracking tests |
| `test_json_schema_validation.py` | PASSED | New - schema validation tests |
| `test_agents.py` | PASSED | Agent functionality |
| `test_agent_utilities.py` | PASSED | Utility functions |
| `test_api_contracts.py` | PASSED | API contracts |
| `test_error_handling.py` | PASSED | Error handling |
| `test_observable_precision.py` | PASSED | Precision validation |
| `test_topological_invariance.py` | PASSED | Topological constants |
| `test_gift_v22_comprehensive.py` | PASSED | v2.2 framework |
| `test_gift_v22_statistical_validation.py` | PASSED | v2.2 stats |
| `test_gift_v21_framework.py` | PARTIAL | 9 failures (CKM, scale) |
| `test_giftpy_comprehensive.py` | PARTIAL | 7 failures (formula diffs) |
| `test_smoke.py` | PARTIAL | 4 failures |
| `test_edge_cases_errors.py` | PARTIAL | 2 NaN/Inf handling |
| `test_input_validation.py` | PARTIAL | 2 negative param |
| `test_v21_all_observables.py` | PARTIAL | 5 CKM deviations |

### Integration Tests
| File | Status | Notes |
|------|--------|-------|
| `test_cross_version_compatibility.py` | PASSED | 14 skipped (v2.0 missing) |
| `test_publication_code_consistency.py` | PARTIAL | 5 formula mismatches |

### Regression Tests
| File | Status | Notes |
|------|--------|-------|
| `test_enhanced_observable_regression.py` | FAILED | 45 baseline mismatches |
| `test_v22_observables.py` | PASSED | All v2.2 exact relations |

### Notebook Tests
| File | Status | Notes |
|------|--------|-------|
| `test_notebook_execution_comprehensive.py` | PASSED | Structure validation |

### Property-Based Tests
| File | Status | Notes |
|------|--------|-------|
| `test_property_based.py` | PARTIAL | 1 symmetry failure |

### Performance Tests
| File | Status | Notes |
|------|--------|-------|
| `test_benchmarks.py` | PARTIAL | 1 overhead test failure |

### Documentation Tests
| File | Status | Notes |
|------|--------|-------|
| `test_docstring_validation.py` | PASSED | All docstrings valid |

---

## Failure Analysis

### Category 1: Regression Baseline Mismatches (45 failures)
**File**: `test_enhanced_observable_regression.py`

These tests compare computed values against hardcoded baselines. The failures indicate the computed values differ from expected baselines:
- Most observable values differ slightly from v2.0 baselines
- This is expected behavior for v2.1/v2.2 which use updated formulas

**Recommendation**: Update regression baselines to v2.2 values or mark as expected differences.

### Category 2: CKM Matrix Deviations (8 failures)
**Files**: `test_gift_v21_framework.py`, `test_v21_all_observables.py`

CKM matrix elements (`V_cb`, `V_ub`, `V_cs`, `V_td`) show deviations beyond tolerance:
```
V_cb: expected 0.0422, got different value
V_ub: expected 0.00394, tolerance exceeded
```

**Recommendation**: Review CKM calculation formulas in v2.1 framework.

### Category 3: Formula Implementation Differences (7 failures)
**Files**: `test_giftpy_comprehensive.py`, `test_publication_code_consistency.py`

Differences between giftpy implementation and v2.1 core:
- `m_tau_m_e`: Different computation methods
- `beta0`: Formula verification failure
- `H_star`: Sum verification

**Recommendation**: Align giftpy formulas with v2.2 publications.

### Category 4: Scale and Electroweak Sector (6 failures)
**Files**: `test_gift_v21_framework.py`

Scale-related observables show deviations:
- `v_EW`: Electroweak VEV
- `M_Z`: Z boson mass
- `H0`: Hubble constant

**Recommendation**: Verify dimensional analysis in scale bridge.

### Category 5: Numerical Edge Cases (4 failures)
**Files**: `test_edge_cases_errors.py`, `test_input_validation.py`

- NaN parameter handling
- Inf parameter handling
- Negative parameter validation

**Recommendation**: Add explicit guards for edge case inputs.

### Category 6: Other (22 failures)
Miscellaneous failures including:
- Property-based symmetry tests
- Performance overhead benchmarks
- Bootstrap uncertainty estimation

---

## Tests Skipped (70)

### By Category:
| Reason | Count |
|--------|-------|
| v2.0 framework unavailable | 14 |
| Documentation files not found | 5 |
| torch not installed | 4 |
| Other dependencies | 47 |

---

## New Tests Added (This Session)

### test_run_validation.py (34 tests)
- Framework initialization
- Topological invariants
- Observable computations
- Numerical stability
- Parameter variations

### test_agents_cli.py (38 tests)
- Registry functions
- Agent class loading
- CLI argument parsing
- Report saving

### test_canonical_monitor.py (33 tests)
- Status ranking
- Baseline persistence
- Upgrade detection
- Addon generation

### test_json_schema_validation.py (32 tests)
- Agent report schema
- Observable data schema
- Training history schema
- JSON serialization edge cases

---

## Recommendations

### High Priority
1. **Update regression baselines** to v2.2 computed values
2. **Fix CKM matrix calculations** to match experimental values within tolerance
3. **Align giftpy with v2.2** formula implementations

### Medium Priority
4. **Add input validation** for NaN/Inf parameters
5. **Review scale bridge** dimensional analysis
6. **Install torch** for full G2_ML test coverage

### Low Priority
7. **Clean up v2.0 references** in skipped tests
8. **Update documentation paths** in test fixtures
9. **Add performance baseline** for overhead tests

---

## Test Artifacts

- `tests/test_results.xml` - JUnit XML report
- `tests/test_output.log` - Full console output

---

## Commands to Reproduce

```bash
# Run all tests (requires full dependencies)
pytest tests/ -v --tb=short

# Run only passing tests
pytest tests/ -v --ignore=tests/regression/test_enhanced_observable_regression.py

# Run new tests only
pytest tests/unit/test_run_validation.py \
       tests/unit/test_agents_cli.py \
       tests/unit/test_canonical_monitor.py \
       tests/unit/test_json_schema_validation.py -v

# Generate HTML report
pytest tests/ --html=tests/report.html --self-contained-html
```

---

*Report generated: 2025-11-27*
*GIFT Framework Test Suite*
