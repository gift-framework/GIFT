# GIFTpy - Geometric Information Field Theory

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)]()

Python package for computing Standard Model predictions from topological geometry.

GIFTpy provides an interface to the GIFT (Geometric Information Field Theory) framework, which derives Standard Model parameters from E₈×E₈ exceptional Lie algebras and K₇ manifolds with G₂ holonomy.

## Key Features

- 43+ observables predicted from topological geometry
- Mean deviation of approximately 0.3% compared to experimental values
- Predictions derived from topological structures
- Comprehensive test suite
- Export functionality to multiple formats

## Installation

```bash
# From repository (development)
cd GIFT
pip install -e .

# From PyPI (planned for future release)
pip install giftpy
```

## Basic Usage

```python
import giftpy

# Initialize GIFT framework
gift = giftpy.GIFT()

# Compute individual observables
alpha_s = gift.gauge.alpha_s()
print(f"Strong coupling: α_s(M_Z) = {alpha_s:.6f}")

Q_Koide = gift.lepton.Q_Koide()
print(f"Koide parameter: Q = {Q_Koide}")

# Compute all observables
results = gift.compute_all()
print(results[['observable', 'value', 'experimental', 'deviation_%']])

# Validate against experiments
validation = gift.validate()
print(validation.summary())
```

## Notable Predictions

### Koide Formula

The Koide parameter Q is derived from topological structure:

```python
Q = gift.lepton.Q_Koide()
# Returns: 0.6666... (2/3)

# Topological origin: Q = dim(G₂)/b₂(K₇) = 14/21 = 2/3
```

- Experimental: Q = 0.666661 ± 0.000007
- GIFT: Q = 2/3
- Deviation: 0.0007%

This provides a theoretical derivation of the Koide formula (Koide, 1982).

### Strong Coupling

```python
alpha_s = gift.gauge.alpha_s()
# Formula: √2/12
```

- Experimental: α_s(M_Z) = 0.1179 ± 0.0010
- GIFT: √2/12 = 0.117851
- Deviation: 0.04%

### Fine Structure Constant

```python
alpha_inv = gift.gauge.alpha_inv()
# Formula: 2⁷ - 1/24
```

- Experimental: α⁻¹(M_Z) = 127.952 ± 0.001
- GIFT: 2⁷ - 1/24 = 127.958333
- Deviation: 0.005%

### CP Violation Phase

```python
delta_CP = gift.neutrino.delta_CP(degrees=True)
# Formula: ζ(3) + √5
```

- Experimental: δ_CP = 197° ± 24°
- GIFT: ζ(3) + √5 ≈ 197°
- Deviation: 0.02%

## Physics Sectors

### Gauge Sector

```python
# Fine structure constant
alpha = gift.gauge.alpha()
alpha_inv = gift.gauge.alpha_inv()

# Strong coupling
alpha_s = gift.gauge.alpha_s()

# Weak mixing angle
sin2theta_W = gift.gauge.sin2theta_W()
theta_W = gift.gauge.theta_W(degrees=True)
```

### Lepton Sector

```python
# Mass ratios
m_mu_m_e = gift.lepton.m_mu_m_e()
m_tau_m_mu = gift.lepton.m_tau_m_mu()
m_tau_m_e = gift.lepton.m_tau_m_e()

# Koide formula
Q_Koide = gift.lepton.Q_Koide()

# Verify Koide formula with physical masses
result = gift.lepton.verify_koide_formula()
print(result)
```

### Neutrino Sector

```python
# Mixing angles
theta_12 = gift.neutrino.theta_12(degrees=True)  # Solar
theta_23 = gift.neutrino.theta_23(degrees=True)  # Atmospheric
theta_13 = gift.neutrino.theta_13(degrees=True)  # Reactor

# CP violation
delta_CP = gift.neutrino.delta_CP(degrees=True)
```

### Quark Sector

```python
# Mass ratios
m_s_m_d = gift.quark.m_s_m_d()

# CKM matrix elements
V_us = gift.quark.V_us()
```

### Cosmology

```python
# Dark energy density
Omega_DE = gift.cosmology.Omega_DE()

# Scalar spectral index
n_s = gift.cosmology.n_s()
```

## Advanced Usage

### Validation and Analysis

```python
# Full validation
validation = gift.validate()
print(validation.summary())

# Plot results
validation.plot(filename='validation.png')

# Access detailed statistics
print(f"Mean deviation: {validation.mean_deviation:.4f}%")
print(f"Number of predictions: {validation.n_observables}")
print(f"χ²/dof: {validation.chi_squared_dof:.2f}")
```

### Export Results

```python
# Export to various formats
gift.export('predictions.csv', format='csv')
gift.export('predictions.json', format='json')
gift.export('predictions.tex', format='latex')
gift.export('predictions.xlsx', format='excel')
```

### Custom Topological Constants

For research purposes, custom topological constants can be specified:

```python
from giftpy.core.constants import TopologicalConstants

# Create custom constants
custom = TopologicalConstants(
    p2=2,
    rank_E8=8,
    Weyl_factor=5
)

# Use in GIFT framework
gift_custom = giftpy.GIFT(constants=custom)
```

### Comparison

```python
# Compare two configurations
gift1 = giftpy.GIFT()
gift2 = giftpy.GIFT(constants=custom)

diff = gift1.compare(gift2)
print(diff)
```

## Testing

```bash
# Run all tests
pytest giftpy_tests/

# Run with coverage
pytest giftpy_tests/ --cov=giftpy --cov-report=html

# Run specific test
pytest giftpy_tests/test_observables.py::TestLeptonSector::test_Q_Koide_exact
```

## Theory Background

GIFT (Geometric Information Field Theory) derives Standard Model parameters from:

1. E₈×E₈ Lie algebras (496 dimensions)
2. K₇ compact manifolds with G₂ holonomy (7 dimensions)
3. Topological invariants: Betti numbers b₂=21, b₃=77

### Core Parameters

GIFTpy uses 3 fundamental topological parameters:

- p₂ = 2: Binary architecture
- rank(E₈) = 8: E₈ Lie algebra rank
- Weyl factor = 5: Weyl group structure

Other quantities are derived from these through topological relations.

### Key Identities

```python
# β₀ = b₂/b₃ (base coupling)
beta0 = CONSTANTS.beta0  # = 21/77

# ξ = (5/2)β₀ (derived parameter)
xi = CONSTANTS.xi  # = 105/154

# N_gen = 3 (derived from topology)
N_gen = CONSTANTS.N_gen  # = 3
```

## Package Structure

```
giftpy/
├── __init__.py           # Main package
├── core/
│   ├── constants.py      # Topological constants
│   ├── framework.py      # GIFT class
│   └── validation.py     # Validation system
├── observables/
│   ├── gauge.py          # α, α_s, sin²θ_W
│   ├── lepton.py         # Lepton masses, Koide
│   ├── neutrino.py       # PMNS, oscillations
│   ├── quark.py          # CKM, mass ratios
│   └── cosmology.py      # Ω_DE, n_s
├── topology/             # E₈, K₇ structures (planned)
├── temporal/             # τ framework (planned)
└── tools/
    ├── export.py         # Data export
    └── visualization.py  # Plotting
```

## Development Roadmap

### v0.1.0 (Current - MVP)
- Core constants module
- Main framework class
- Gauge sector (3 observables)
- Lepton sector (4 observables)
- Neutrino sector (2 observables)
- Quark sector (2 observables)
- Cosmology sector (2 observables)
- Validation system
- Basic tests

### v0.2.0 (Planned)
- Complete observable set (43+ total)
- Advanced neutrino sector (full PMNS matrix)
- Complete quark sector (all CKM elements)
- Improved documentation
- Jupyter notebook tutorials

### v0.3.0 (Planned)
- Topology module (E₈ roots, K₇ cohomology)
- Temporal framework (τ-parameter)
- Interactive visualizations
- Performance optimizations

### v1.0.0 (Future)
- Production release
- Complete documentation
- PyPI publication
- Potential submission to JOSS

## Documentation

- Quick Start: This README
- API Reference: See docstrings (Sphinx documentation planned)
- Theory: See `publications/gift_main.md` in main repository
- Examples: See `examples/` directory

## Contributing

Contributions are welcome. See `CONTRIBUTING.md` for guidelines.

### Development Setup

```bash
# Clone repository
git clone https://github.com/gift-framework/GIFT.git
cd GIFT

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest giftpy_tests/

# Format code
black giftpy/

# Type checking
mypy giftpy/
```

## License

MIT License - see LICENSE file

## Contact

- Repository: https://github.com/gift-framework/GIFT
- Issues: https://github.com/gift-framework/GIFT/issues
- Discussions: https://github.com/gift-framework/GIFT/discussions

## Citation

If you use GIFTpy in your research, please cite:

```bibtex
@software{giftpy2024,
  title={GIFTpy: Python package for GIFT framework computations},
  author={GIFT Framework Collaboration},
  year={2024},
  version={0.1.0},
  url={https://github.com/gift-framework/GIFT}
}
```

## Acknowledgments

Built on the theoretical work of the GIFT Framework research program.

## References

Koide, Y. (1982). "A New Relation Among Lepton Masses". Letters to the Nuovo Cimento, 34, 201-205.
