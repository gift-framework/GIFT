# GIFT Framework Test Suite

Comprehensive test suite for the GIFT (Geometric Information Field Theory) framework.

## Overview

This test suite provides extensive coverage of all major components of the GIFT framework, including:

- Core physics calculations (34 dimensionless observables)
- G2 machine learning framework
- Statistical validation tools
- Automated agents
- Jupyter notebooks
- Integration pipelines

**Target Coverage**: 80-90%
**Current Status**: Initial implementation complete

## Test Structure

```
tests/
├── unit/                          # Unit tests for individual components
│   ├── test_gift_framework.py     # Core physics calculations (300+ tests)
│   ├── test_agents.py             # Automated agents
│   └── test_error_handling.py     # Error handling & edge cases
│
├── integration/                   # Integration tests for workflows
│   └── test_gift_pipeline.py      # End-to-end pipelines
│
├── regression/                    # Regression tests
│   └── test_observable_values.py  # Lock in known good results
│
├── notebooks/                     # Notebook validation
│   └── test_notebooks.py          # Jupyter notebook execution tests
│
├── fixtures/                      # Test fixtures and reference data
│   └── reference_observables.json # Reference values for regression
│
└── conftest.py                    # Shared pytest fixtures
```

## Running Tests

### Quick Start

```bash
# Install test dependencies
pip install -r requirements.txt

# Run all tests
pytest

# Run with coverage report
pytest --cov=. --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Specific Test Categories

```bash
# Unit tests only
pytest tests/unit

# Integration tests
pytest tests/integration

# Regression tests
pytest tests/regression

# Notebook tests (slow)
pytest tests/notebooks

# G2 ML tests
pytest G2_ML/tests
```

### Test Markers

```bash
# Run only fast tests (skip slow tests)
pytest -m "not slow"

# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only regression tests
pytest -m regression

# Run only notebook tests
pytest -m notebook
```

### Parallel Execution

```bash
# Run tests in parallel (faster)
pytest -n auto

# Run with specific number of workers
pytest -n 4
```

## Test Categories

### Unit Tests

#### Core GIFT Framework (`test_gift_framework.py`)

Tests all 34 dimensionless observable calculations:

**Gauge Sector**:
- `test_alpha_inv_MZ` - Fine structure constant at MZ
- `test_sin2thetaW` - Weak mixing angle
- `test_alpha_s_MZ` - Strong coupling constant

**Neutrino Sector**:
- `test_theta12` - Solar mixing angle
- `test_theta13` - Reactor mixing angle
- `test_theta23` - Atmospheric mixing angle
- `test_delta_CP` - CP violation phase (PROVEN exact)

**Lepton Sector**:
- `test_Q_Koide` - Koide formula parameter (PROVEN exact)
- `test_m_mu_m_e` - Muon to electron mass ratio
- `test_m_tau_m_e` - Tau to electron mass ratio (PROVEN exact)

**Quark Sector**:
- `test_m_s_m_d` - Strange to down quark mass ratio (PROVEN exact)

**Higgs Sector**:
- `test_lambda_H` - Higgs quartic coupling (PROVEN exact)

**Cosmology**:
- `test_Omega_DE` - Dark energy density (PROVEN exact)
- `test_n_s` - Scalar spectral index
- `test_H0` - Hubble constant

**Test Classes**:
- `TestGIFTFrameworkInitialization` - Parameter initialization
- `TestGaugeSectorObservables` - Gauge observable calculations
- `TestNeutrinoSectorObservables` - Neutrino parameters
- `TestLeptonSectorObservables` - Lepton mass ratios
- `TestQuarkSectorObservables` - Quark parameters
- `TestHiggsSectorObservables` - Higgs coupling
- `TestCosmologicalObservables` - Cosmological parameters
- `TestNumericalPrecision` - Precision and stability
- `TestParameterSensitivity` - Parameter variations
- `TestExperimentalComparison` - Comparison with data

#### G2 Geometry (`G2_ML/tests/test_geometry.py`)

Tests differential geometry operations:

- `TestSPDProjection` - Symmetric positive definite projection
- `TestVolumeForm` - Volume form computation
- `TestMetricInverse` - Metric inverse with stability
- `TestGeometryGradients` - Gradient computations
- `TestNumericalAccuracy` - Numerical precision

#### G2 Manifold (`G2_ML/tests/test_manifold.py`)

Tests manifold operations:

- `TestManifoldCreation` - Manifold initialization
- `TestPointSampling` - Point sampling on T^7
- `TestManifoldProperties` - Geometric properties

#### Agents (`test_agents.py`)

Tests automated maintenance agents:

- `TestVerifierAgent` - Link and status tag verification
- `TestDocsIntegrityAgent` - Documentation integrity
- `TestUnicodeSanitizerAgent` - Unicode detection
- `TestMarkdownUtils` - Markdown parsing utilities
- `TestFileSystemUtils` - File discovery utilities

#### Error Handling (`test_error_handling.py`)

Tests error cases and edge conditions:

- `TestGIFTFrameworkErrorHandling` - Invalid parameters
- `TestG2GeometryErrorHandling` - Singular matrices, NaN, Inf
- `TestNumericalEdgeCases` - Overflow, underflow protection
- `TestFileIOErrors` - Missing files, invalid JSON
- `TestMemoryManagement` - Memory leak detection
- `TestDivisionByZero` - Division by zero protection

### Integration Tests

#### Full Pipeline (`test_gift_pipeline.py`)

Tests complete workflows:

- `TestFullObservablePipeline` - Parameters → observables → comparison
- `TestStatisticalValidationPipeline` - Monte Carlo, bootstrap
- `TestExperimentalComparisonPipeline` - Precision calculation
- `TestMultiVersionCompatibility` - Cross-version consistency

### Regression Tests

#### Observable Values (`test_observable_values.py`)

Locks in known good results:

- `TestObservableRegression` - All observables vs reference
- `TestNumericalStability` - Reproducibility over runs
- `TestBackwardCompatibility` - API stability

Reference data in `fixtures/reference_observables.json`

### Notebook Tests

#### Notebook Validation (`test_notebooks.py`)

Tests Jupyter notebooks:

- `TestPublicationNotebooks` - Main publication notebooks execute
- `TestNotebookStructure` - Valid format, markdown cells
- `TestG2Notebooks` - G2 ML notebooks
- `TestVisualizationNotebooks` - Visualization notebooks
- `TestNotebookOutputs` - Expected outputs present

## Coverage Goals

### Target Coverage by Component

| Component | Target | Priority |
|-----------|--------|----------|
| Core physics (`run_validation.py`) | 95% | Critical |
| G2 geometry (`G2_geometry.py`) | 90% | High |
| G2 manifold (`G2_manifold.py`) | 85% | High |
| Agents | 80% | Medium |
| Utilities | 75% | Medium |
| Overall | 80-85% | - |

### PROVEN Exact Relations

All 9 PROVEN exact relations must have 100% test coverage:

1. ✓ N_gen = 3
2. ✓ Q_Koide = 2/3
3. ✓ m_s/m_d = 20
4. ✓ δ_CP = 197°
5. ✓ m_τ/m_e = 3477
6. ✓ Ω_DE = ln(2) × 98/99
7. ✓ ξ = 5β₀/2
8. ✓ λ_H = √17/32

## Test Fixtures

### Reference Observable Data

`fixtures/reference_observables.json` contains:

- Framework version: 2.0.0
- Default parameters
- All 34 observable predictions
- Experimental values
- Deviation percentages

Used for regression testing to ensure calculations remain consistent.

## Continuous Integration

### GitHub Actions

Tests run automatically on:
- Every push to `main`
- Every pull request
- Python versions: 3.9, 3.10, 3.11

### CI Workflow

1. **Lint** - Code quality with flake8
2. **Unit Tests** - Fast unit tests with coverage
3. **Integration Tests** - Workflow tests
4. **Regression Tests** - Value stability
5. **Coverage Upload** - To Codecov
6. **Notebook Validation** - Basic execution check

### Coverage Reporting

Coverage reports uploaded to Codecov with each CI run.

View at: `https://codecov.io/gh/gift-framework/GIFT`

## Writing New Tests

### Test Structure

```python
import pytest
import sys
from pathlib import Path

# Add module to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent / "module_dir"))

from module import function_to_test


class TestFeature:
    """Test a specific feature."""

    def test_basic_functionality(self):
        """Test basic use case."""
        result = function_to_test()
        assert result == expected_value

    def test_edge_case(self):
        """Test edge case."""
        with pytest.raises(ValueError):
            function_to_test(invalid_input)

    @pytest.mark.slow
    def test_expensive_operation(self):
        """Test expensive operation (marked slow)."""
        # Long-running test
        pass
```

### Using Fixtures

```python
def test_with_framework(gift_framework):
    """Use shared fixture from conftest.py."""
    obs = gift_framework.compute_all_observables()
    assert 'alpha_inv_MZ' in obs
```

### Parametrized Tests

```python
@pytest.mark.parametrize("input,expected", [
    (2.0, 20.0),
    (2.5, 31.25),
    (3.0, 45.0),
])
def test_multiple_inputs(input, expected):
    """Test multiple inputs efficiently."""
    result = input ** 2 * 5
    assert result == expected
```

## Best Practices

### Test Naming

- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`
- Be descriptive: `test_delta_CP_exact_formula` not `test_obs_4`

### Test Organization

- One test file per module
- Group related tests in classes
- Order tests from simple to complex
- Mark slow tests with `@pytest.mark.slow`

### Assertions

- Use specific assertions: `assert x == y` not `assert True`
- Include helpful messages: `assert x == y, f"Expected {y}, got {x}"`
- Test both success and failure cases
- Check for NaN/Inf in numerical tests

### Coverage

- Aim for 80%+ line coverage
- 100% coverage for PROVEN exact relations
- Don't sacrifice test quality for coverage percentage
- Use `# pragma: no cover` sparingly

### Performance

- Unit tests should be fast (< 1s each)
- Mark slow tests: `@pytest.mark.slow`
- Use fixtures for expensive setup
- Run slow tests separately in CI

## Troubleshooting

### Common Issues

**Import Errors**:
```bash
# Ensure you're in project root
cd /path/to/GIFT

# Install in development mode
pip install -e .
```

**Missing Dependencies**:
```bash
# Install all test dependencies
pip install -r requirements.txt
```

**CUDA Errors in G2 Tests**:
```bash
# Run CPU-only tests
pytest -k "not cuda"
```

**Slow Test Execution**:
```bash
# Skip slow tests
pytest -m "not slow"

# Run in parallel
pytest -n auto
```

**Coverage Not Generated**:
```bash
# Ensure pytest-cov installed
pip install pytest-cov

# Run with coverage
pytest --cov=. --cov-report=html
```

## Contributing

When adding new features:

1. **Write tests first** (TDD approach)
2. **Ensure tests pass** locally before PR
3. **Maintain coverage** at 80%+
4. **Add docstrings** to test functions
5. **Update this README** if adding new test categories

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [GIFT Framework Documentation](../README.md)
- [Contributing Guidelines](../CONTRIBUTING.md)

## Contact

For test-related questions:
- Open an issue: https://github.com/gift-framework/GIFT/issues
- Check existing tests for examples
- Review `conftest.py` for available fixtures

---

**Version**: 1.0.0
**Last Updated**: 2025-11-16
**Test Count**: 200+ tests
**Target Coverage**: 80-90%
