"""
Pytest configuration and shared fixtures for GIFT framework tests.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "statistical_validation"))
sys.path.insert(0, str(PROJECT_ROOT / "assets" / "agents"))


@pytest.fixture
def gift_framework():
    """Create a GIFTFrameworkStatistical instance with default parameters."""
    from run_validation import GIFTFrameworkStatistical
    return GIFTFrameworkStatistical()


@pytest.fixture
def experimental_data():
    """Provide experimental data for validation tests."""
    return {
        'alpha_inv_MZ': (127.955, 0.01),
        'sin2thetaW': (0.23122, 0.00004),
        'alpha_s_MZ': (0.1179, 0.0011),
        'theta12': (33.44, 0.77),
        'theta13': (8.61, 0.12),
        'theta23': (49.2, 1.1),
        'delta_CP': (197.0, 24.0),
        'Q_Koide': (0.6667, 0.0001),
        'm_mu_m_e': (206.768, 0.001),
        'm_tau_m_e': (3477.0, 0.1),
        'm_s_m_d': (20.0, 1.0),
        'lambda_H': (0.129, 0.002),
        'Omega_DE': (0.6847, 0.0056),
        'n_s': (0.9649, 0.0042),
        'H0': (73.04, 1.04)
    }


@pytest.fixture
def random_seed():
    """Set random seed for reproducible tests."""
    np.random.seed(42)
    return 42


@pytest.fixture
def project_root():
    """Provide project root path."""
    return PROJECT_ROOT


@pytest.fixture
def tolerance():
    """Standard numerical tolerance for tests."""
    return 1e-6


@pytest.fixture
def relative_tolerance():
    """Relative tolerance for observable comparisons."""
    return 0.01  # 1%
