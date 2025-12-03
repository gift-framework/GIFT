# G2_ML Test Suite

Unit tests for the G2 Machine Learning module of the GIFT framework.

## Test Organization

The tests are organized into two categories:

### Legacy Tests (archived/0.2)

These tests cover the **archived** implementation from `G2_ML/archived/early_development/0.2/`:

| File | Tests |
|------|-------|
| `test_geometry.py` | SPD projection, volume form, metric inverse |
| `test_losses.py` | Torsion loss, volume loss, curriculum scheduler |
| `test_phi_network.py` | Fourier features, SIREN layers, G2PhiNetwork |
| `test_manifold.py` | Manifold creation, point sampling |

These tests import from the **archived** code and are kept for backwards compatibility verification.

### Active Tests (2_1/)

These tests cover the **current production** implementation in `G2_ML/2_1/`:

| File | Tests |
|------|-------|
| `test_g2_v21_geometry.py` | MetricFromPhi, G2Positivity, standard G2 structure |
| `test_g2_v21_model.py` | FourierFeatures, G2VariationalNet, HarmonicFormsNet |
| `test_g2_v21_loss.py` | TorsionFunctional, VariationalLoss, PhasedLossManager |

These tests import from `G2_ML/2_1/` which is the active, maintained codebase.

## Running Tests

```bash
# Run all G2_ML tests
pytest G2_ML/tests -v

# Run only v2.1 (active) tests
pytest G2_ML/tests/test_g2_v21_*.py -v

# Run only legacy (archived) tests
pytest G2_ML/tests/test_geometry.py G2_ML/tests/test_losses.py G2_ML/tests/test_phi_network.py G2_ML/tests/test_manifold.py -v

# Skip slow tests
pytest G2_ML/tests -v -m "not slow"

# Run with coverage
pytest G2_ML/tests --cov=G2_ML --cov-report=term-missing
```

## G2_ML Module Structure

```
G2_ML/
├── 2_1/                    # ACTIVE - Production code
│   ├── config.py           # GIFT v2.2 configuration
│   ├── constraints.py      # Physics constraints
│   ├── g2_geometry.py      # G2 geometry operations
│   ├── loss.py             # Variational loss functions
│   ├── model.py            # Neural network architectures
│   ├── training.py         # Training pipeline
│   └── validation.py       # Validation utilities
│
├── archived/               # ARCHIVED - Historical implementations
│   ├── early_development/
│   │   ├── 0.1/           # Initial version
│   │   └── 0.2/           # Second iteration
│   └── v1_iterations/     # Research iterations
│
├── research_modules/       # RESEARCH - Experimental code
│   ├── meta_hodge/        # Hodge theory exploration
│   ├── variational_g2/    # Variational methods
│   ├── tcs_joyce/         # TCS construction
│   └── harmonic_yukawa/   # Yukawa coupling analysis
│
├── G2_Lean/               # Lean 4 integration
│   └── numerical/         # Export to Lean proofs
│
└── tests/                 # THIS DIRECTORY
    ├── test_g2_v21_*.py   # Tests for 2_1/ (active)
    └── test_*.py          # Tests for archived (legacy)
```

## Test Dependencies

Tests require:
- `pytest`
- `torch` (PyTorch)
- `numpy`

Optional (for GPU tests):
- CUDA-capable GPU
- `torch` with CUDA support

## Adding New Tests

When adding tests for **new features**, add them to `test_g2_v21_*.py` files.

When fixing tests for **archived code**, modify `test_geometry.py`, `test_losses.py`, etc.

### Test Naming Convention

- `test_g2_v21_<module>.py` - Tests for active 2_1/ code
- `test_<module>.py` - Tests for archived 0.2/ code

### Markers

- `@pytest.mark.slow` - Tests that take >1 second
- `@pytest.mark.skipif(not torch.cuda.is_available())` - GPU tests
