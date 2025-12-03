# Harmonic-Yukawa Pipeline Tests

Comprehensive test suite for the Yukawa tensor computation pipeline in GIFT v2.2.

## Test Organization

| File | Module | Tests |
|------|--------|-------|
| `test_config.py` | `config.py` | GIFT topological constants, configuration validation |
| `test_wedge_product.py` | `wedge_product.py` | Wedge products: 2∧2→4, 4∧3→7, antisymmetry |
| `test_yukawa_tensor.py` | `yukawa.py` | YukawaResult, YukawaTensor, eigenvalue analysis |
| `test_mass_spectrum.py` | `mass_spectrum.py` | FermionMasses, PDG comparisons, GIFT predictions |
| `test_harmonic_forms.py` | `harmonic_extraction.py`, `hodge_laplacian.py` | HarmonicBasis, neural networks, Laplacian |

## Running Tests

```bash
# Run all harmonic_yukawa tests
pytest G2_ML/research_modules/harmonic_yukawa/tests -v

# Run specific test file
pytest G2_ML/research_modules/harmonic_yukawa/tests/test_wedge_product.py -v

# Skip slow tests
pytest G2_ML/research_modules/harmonic_yukawa/tests -v -m "not slow"

# Run with coverage
pytest G2_ML/research_modules/harmonic_yukawa/tests --cov=G2_ML/research_modules/harmonic_yukawa
```

## GIFT v2.2 Constants Tested

The tests verify these PROVEN topological constants:

| Constant | Value | Meaning |
|----------|-------|---------|
| b₂(K7) | 21 | Harmonic 2-forms (gauge moduli) |
| b₃(K7) | 77 | Harmonic 3-forms (35 local + 42 global) |
| det(g) | 65/32 | Metric determinant |
| κ_T | 1/61 | Torsion magnitude |
| τ | 3472/891 | Visible/hidden eigenvalue ratio |
| N_gen | 3 | Number of generations |

## Key Physical Tests

### Wedge Product Properties
- Antisymmetry: ω ∧ η = -η ∧ ω
- Self-wedge zero: ω ∧ ω = 0
- Bilinearity

### Yukawa Tensor Properties
- Gram matrix M = Y^T Y is symmetric, positive semi-definite
- 43/77 split (visible/hidden sectors)
- τ parameter computation

### Mass Spectrum
- Koide parameter Q ≈ 2/3
- m_τ/m_e ≈ 3477
- m_s/m_d ≈ 20

## Test Markers

- `@pytest.mark.slow` - Tests requiring full Yukawa tensor computation
- `@pytest.mark.skipif(not torch.cuda.is_available())` - GPU tests

## Adding New Tests

When adding tests, follow these patterns:

1. **Import with fallback**: Handle import errors gracefully
2. **Use fixtures**: Share setup code via pytest fixtures
3. **Test shapes first**: Verify tensor shapes before values
4. **Check numerical stability**: Test edge cases (zeros, large values)
5. **Mark slow tests**: Use `@pytest.mark.slow` for expensive computations
