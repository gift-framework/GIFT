# Supplement S6: Numerical Methods

## Algorithms, Implementation, and Validation

*This supplement documents the computational framework, algorithms, and validation procedures used in GIFT numerical calculations.*

---

## 1. Computational Framework

### 1.1 Software Stack

The GIFT computational framework relies on standard scientific Python libraries:

```python
# Core numerical libraries
numpy>=1.24.0      # Array operations
scipy>=1.10.0      # Scientific computing
sympy>=1.11.0      # Symbolic mathematics

# Machine learning
torch>=2.0.0       # Neural networks for K7 metric

# Visualization
matplotlib>=3.7.0  # Plotting
plotly>=5.14.0     # Interactive 3D

# Statistical analysis
pandas>=2.0.0      # Data manipulation
statsmodels>=0.14  # Statistical models
```

### 1.2 Hardware Requirements

**Minimum configuration**:
- CPU: 8 cores
- RAM: 32 GB
- Storage: 100 GB SSD

**Recommended for ML training**:
- GPU: NVIDIA A100 (40GB) or equivalent
- RAM: 128 GB
- Storage: 1 TB NVMe

### 1.3 Installation

```bash
# Clone repository
git clone https://github.com/gift-framework/GIFT.git
cd GIFT

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Core Algorithms

### 2.1 Topological Parameter Computation

All topological parameters are computed exactly from integer arithmetic:

```python
import numpy as np

# E8 parameters
dim_E8 = 248
rank_E8 = 8

# K7 cohomology
b2_K7 = 21
b3_K7 = 77
H_star = b2_K7 + b3_K7 + 1  # = 99

# G2 parameters
dim_G2 = 14
dim_K7 = 7

# Derived parameters
p2 = dim_G2 / dim_K7  # = 2 (exact)
Wf = 5  # Weyl factor

# Base coupling
beta_0 = np.pi / rank_E8
xi = (Wf / p2) * beta_0  # = 5*pi/16
```

### 2.2 Observable Computation

Example: Computing gauge couplings

```python
def compute_gauge_couplings():
    """Compute the three gauge coupling constants."""

    # Fine structure constant inverse
    alpha_inv = (dim_E8 + rank_E8) / 2  # = 128

    # Weinberg angle
    zeta_3 = 1.2020569031595943  # Apery's constant
    gamma_euler = 0.5772156649015329
    M2 = 3  # Second Mersenne prime
    sin2_theta_W = zeta_3 * gamma_euler / M2

    # Strong coupling
    W_G2 = 12  # |W(G2)| Weyl group order
    alpha_s = np.sqrt(2) / W_G2

    return {
        'alpha_inv_MZ': alpha_inv,
        'sin2_theta_W': sin2_theta_W,
        'alpha_s_MZ': alpha_s
    }
```

### 2.3 Neutrino Mixing Angles

```python
def compute_neutrino_mixing():
    """Compute PMNS mixing parameters."""

    # Reactor angle (exact)
    theta_13 = np.pi / b2_K7  # = pi/21

    # Atmospheric angle
    theta_23_rad = (rank_E8 + b3_K7) / H_star  # = 85/99
    theta_23 = np.degrees(theta_23_rad)

    # Solar angle
    delta = 2 * np.pi / (Wf ** 2)  # = 2pi/25
    gamma_GIFT = 511 / 884
    theta_12 = np.degrees(np.arctan(np.sqrt(delta / gamma_GIFT)))

    # CP phase (exact)
    delta_CP = dim_K7 * dim_G2 + H_star  # = 197 degrees

    return {
        'theta_12': theta_12,
        'theta_13': np.degrees(theta_13),
        'theta_23': theta_23,
        'delta_CP': delta_CP
    }
```

### 2.4 Heat Kernel Coefficient

The GIFT heat kernel coefficient gamma_GIFT is computed as:

```python
def compute_gamma_GIFT():
    """Compute heat kernel coefficient gamma_GIFT."""
    numerator = 2 * rank_E8 + 5 * H_star
    denominator = 10 * dim_G2 + 3 * dim_E8
    return numerator / denominator  # = 511/884

# Verification
gamma_GIFT = 511 / 884
gamma_euler = 0.5772156649015329
print(f"gamma_GIFT = {gamma_GIFT:.16f}")
print(f"gamma_Euler = {gamma_euler:.16f}")
print(f"Difference = {abs(gamma_GIFT - gamma_euler):.6f}")
# Difference = 0.000839 (0.145%)
```

---

## 3. Statistical Validation

### 3.1 Monte Carlo Uncertainty Propagation

```python
def monte_carlo_validation(n_samples=1_000_000):
    """
    Monte Carlo propagation of experimental uncertainties.
    """
    import numpy as np

    # Experimental values with uncertainties (example: gauge sector)
    exp_values = {
        'alpha_inv': (127.955, 0.016),
        'sin2_theta_W': (0.23122, 0.00004),
        'alpha_s': (0.1179, 0.0010)
    }

    results = {}
    for name, (mean, sigma) in exp_values.items():
        # Sample from Gaussian distribution
        samples = np.random.normal(mean, sigma, n_samples)

        # Compute statistics
        results[name] = {
            'mean': np.mean(samples),
            'std': np.std(samples),
            'median': np.median(samples),
            'percentile_16': np.percentile(samples, 16),
            'percentile_84': np.percentile(samples, 84)
        }

    return results
```

### 3.2 Chi-Square Analysis

```python
def chi_square_test(gift_values, exp_values, exp_uncertainties):
    """
    Compute chi-square statistic for GIFT predictions.
    """
    chi2 = 0
    n_obs = len(gift_values)

    for obs in gift_values:
        gift = gift_values[obs]
        exp = exp_values[obs]
        sigma = exp_uncertainties[obs]
        chi2 += ((gift - exp) / sigma) ** 2

    dof = n_obs - 3  # 3 free parameters
    p_value = 1 - scipy.stats.chi2.cdf(chi2, dof)

    return chi2, dof, p_value
```

### 3.3 Sobol Sensitivity Analysis

Global sensitivity analysis identifies which parameters most affect predictions:

```python
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

def sobol_analysis():
    """
    Sobol global sensitivity analysis for GIFT parameters.
    """
    problem = {
        'num_vars': 3,
        'names': ['p2', 'rank_E8', 'Wf'],
        'bounds': [[1.9, 2.1], [7.9, 8.1], [4.9, 5.1]]
    }

    # Generate samples
    param_values = sobol_sample.sample(problem, 2048)

    # Evaluate model at each sample point
    Y = np.array([evaluate_model(p) for p in param_values])

    # Analyze sensitivity
    Si = sobol_analyze.analyze(problem, Y)

    return Si
```

---

## 4. K7 Metric Computation

### 4.1 Neural Network Architecture

The K7 metric is approximated using neural networks:

```python
import torch
import torch.nn as nn

class K7MetricNetwork(nn.Module):
    """Neural network for K7 metric tensor components."""

    def __init__(self, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 28)  # 7x7 symmetric -> 28 components
        )

    def forward(self, x):
        """
        Args:
            x: Coordinates on K7, shape (batch, 7)
        Returns:
            Metric tensor components, shape (batch, 28)
        """
        return self.net(x)

    def get_metric(self, x):
        """Reconstruct full metric tensor."""
        components = self.forward(x)
        # Unpack to symmetric 7x7 matrix
        metric = torch.zeros(x.shape[0], 7, 7)
        idx = 0
        for i in range(7):
            for j in range(i, 7):
                metric[:, i, j] = components[:, idx]
                metric[:, j, i] = components[:, idx]
                idx += 1
        return metric
```

### 4.2 Training Procedure

```python
def train_k7_metric(model, dataloader, epochs=1000):
    """Train K7 metric network with G2 holonomy constraint."""

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        for batch in dataloader:
            coords = batch['coordinates']

            # Get predicted metric
            g = model.get_metric(coords)

            # Loss 1: Ricci-flat condition
            loss_ricci = compute_ricci_loss(g, coords)

            # Loss 2: G2 holonomy constraint
            loss_g2 = compute_g2_loss(g, coords)

            # Loss 3: Determinant = p2 = 2
            det_g = torch.det(g)
            loss_det = ((det_g - 2.0) ** 2).mean()

            # Total loss
            loss = loss_ricci + 0.1 * loss_g2 + 0.01 * loss_det

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### 4.3 Harmonic Form Extraction

```python
def extract_harmonic_forms(metric_network, n_points=10000):
    """
    Extract harmonic 2-forms and 3-forms from trained metric.

    Returns:
        h2: Array of shape (21, n_points, 21) for b2=21 forms
        h3: Array of shape (77, n_points, 35) for b3=77 forms
    """
    # Sample points on K7
    coords = sample_k7_manifold(n_points)

    # Get metric at each point
    g = metric_network.get_metric(coords)

    # Compute Hodge star operator
    hodge = compute_hodge_star(g)

    # Solve eigenvalue problem for harmonic forms
    # Laplacian eigenvalue = 0 for harmonic forms

    # 2-forms: 21 harmonic forms
    h2 = solve_harmonic_eigenvalue(hodge, degree=2, n_forms=21)

    # 3-forms: 77 harmonic forms
    h3 = solve_harmonic_eigenvalue(hodge, degree=3, n_forms=77)

    return h2, h3
```

---

## 5. Validation Suite

### 5.1 Unit Tests

```python
import pytest

class TestTopologicalConstants:
    """Unit tests for topological constants."""

    def test_betti_numbers(self):
        assert b2_K7 == 21
        assert b3_K7 == 77
        assert b2_K7 + b3_K7 == 98

    def test_dimensions(self):
        assert dim_E8 == 248
        assert rank_E8 == 8
        assert dim_G2 == 14
        assert dim_K7 == 7

    def test_p2_dual_origin(self):
        p2_local = dim_G2 / dim_K7
        p2_global = 496 / 248
        assert p2_local == 2
        assert p2_global == 2
        assert p2_local == p2_global

    def test_generation_number(self):
        N_gen = 168 / 56
        assert N_gen == 3

class TestExactRelations:
    """Unit tests for exact topological relations."""

    def test_tau_electron_ratio(self):
        ratio = 7 + 10*248 + 10*99
        assert ratio == 3477

    def test_strange_down_ratio(self):
        ratio = 4 * 5  # p2^2 * Wf
        assert ratio == 20

    def test_koide_parameter(self):
        Q = 14 / 21
        assert Q == pytest.approx(2/3, rel=1e-10)

    def test_cp_phase(self):
        delta_CP = 7 * 14 + 99
        assert delta_CP == 197
```

### 5.2 Integration Tests

```python
class TestFullPipeline:
    """Integration tests for complete calculation pipeline."""

    def test_gauge_sector(self):
        results = compute_gauge_couplings()
        assert results['alpha_inv_MZ'] == 128
        assert abs(results['sin2_theta_W'] - 0.23122) < 0.001
        assert abs(results['alpha_s_MZ'] - 0.1179) < 0.001

    def test_neutrino_sector(self):
        results = compute_neutrino_mixing()
        assert abs(results['theta_12'] - 33.44) < 0.5
        assert abs(results['theta_13'] - 8.57) < 0.1
        assert abs(results['theta_23'] - 49.2) < 0.5
        assert results['delta_CP'] == 197

    def test_all_observables(self):
        """Verify all 37 observables compute without error."""
        results = compute_all_observables()
        assert len(results) == 37
        for name, value in results.items():
            assert np.isfinite(value), f"{name} is not finite"
```

### 5.3 Numerical Stability

```python
def test_numerical_stability():
    """Test numerical stability across different precisions."""

    # Single precision
    result_32 = compute_observables(dtype=np.float32)

    # Double precision
    result_64 = compute_observables(dtype=np.float64)

    # Extended precision (if available)
    result_128 = compute_observables(dtype=np.float128)

    # Check consistency
    for obs in result_64:
        rel_diff = abs(result_64[obs] - result_128[obs]) / result_128[obs]
        assert rel_diff < 1e-10, f"{obs} unstable: {rel_diff}"
```

---

## 6. Performance Benchmarks

### 6.1 Computation Times

| Operation | Time (ms) | Notes |
|-----------|-----------|-------|
| Topological constants | < 0.1 | Integer arithmetic |
| Gauge couplings | < 1 | Simple formulas |
| All 37 observables | < 10 | Full computation |
| Monte Carlo (10^6) | ~5000 | Statistical validation |
| K7 metric training | ~3600000 | 1 hour on A100 |

### 6.2 Memory Usage

| Dataset | Memory (MB) |
|---------|-------------|
| Topological parameters | < 1 |
| Full observable set | < 10 |
| K7 metric network | ~100 |
| Training batch | ~1000 |

### 6.3 Scaling

The Monte Carlo validation scales linearly with sample size:
- 10^4 samples: 50 ms
- 10^5 samples: 500 ms
- 10^6 samples: 5 s
- 10^7 samples: 50 s

---

## 7. Reproducibility

### 7.1 Random Seed Management

```python
def set_reproducibility(seed=42):
    """Set all random seeds for reproducibility."""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

### 7.2 Version Control

All numerical results are tagged with:
- Git commit hash
- Python version
- Library versions
- Hardware specification
- Timestamp

### 7.3 Result Caching

```python
import hashlib
import json

def cache_result(params, result, cache_dir='./cache'):
    """Cache computation result with parameter hash."""
    param_hash = hashlib.md5(
        json.dumps(params, sort_keys=True).encode()
    ).hexdigest()

    cache_file = f"{cache_dir}/{param_hash}.json"
    with open(cache_file, 'w') as f:
        json.dump({'params': params, 'result': result}, f)
```

---

## References

1. NumPy documentation: https://numpy.org/doc/
2. SciPy reference: https://docs.scipy.org/doc/scipy/reference/
3. PyTorch documentation: https://pytorch.org/docs/
4. SALib for sensitivity analysis: https://salib.readthedocs.io/

---

*GIFT Framework v2.1 - Supplement S6*
*Numerical Methods*
