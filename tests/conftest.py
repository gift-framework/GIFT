"""
Pytest configuration and shared fixtures for GIFT framework tests.

Version: 2.1.0
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
    """
    Create a GIFT Framework v2.1 instance with default parameters.

    Returns GIFTFrameworkV21 with torsional dynamics and scale bridge.
    """
    from gift_v21_core import GIFTFrameworkV21
    return GIFTFrameworkV21()


@pytest.fixture
def gift_framework_v20():
    """
    Legacy v2.0 framework for backwards compatibility tests.

    Returns GIFTFrameworkStatistical (v2.0) with static topology only.
    """
    from run_validation import GIFTFrameworkStatistical
    return GIFTFrameworkStatistical()


@pytest.fixture
def experimental_data():
    """
    Provide experimental data for v2.1 validation tests (46 observables).

    Returns dictionary with (value, uncertainty) tuples.
    """
    return {
        # === DIMENSIONLESS (37 total) ===
        # Gauge sector (3)
        'alpha_inv_MZ': (127.955, 0.01),
        'sin2thetaW': (0.23122, 0.00004),
        'alpha_s_MZ': (0.1179, 0.0011),

        # Neutrino mixing (4)
        'theta12': (33.44, 0.77),
        'theta13': (8.61, 0.12),
        'theta23': (49.2, 1.1),
        'delta_CP': (197.0, 24.0),

        # Lepton mass ratios (3)
        'Q_Koide': (0.666661, 0.000007),
        'm_mu_m_e': (206.768, 0.001),
        'm_tau_m_e': (3477.15, 0.12),

        # Quark mass ratios (10)
        'm_s_m_d': (20.0, 1.0),
        'm_c_m_s': (13.60, 0.30),
        'm_b_m_u': (1935.19, 40.0),
        'm_t_m_b': (41.3, 1.2),
        'm_d_m_u': (2.16, 0.04),
        'm_c_m_u': (589.35, 15.0),
        'm_b_m_d': (894.0, 25.0),
        'm_t_m_s': (1848.0, 50.0),
        'm_t_m_d': (36960.0, 1000.0),
        'm_t_m_c': (136.0, 3.0),

        # CKM matrix elements (6)
        'V_us': (0.2243, 0.0005),
        'V_cb': (0.0422, 0.0008),
        'V_ub': (0.00394, 0.00036),
        'V_cd': (0.218, 0.004),
        'V_cs': (0.997, 0.017),
        'V_td': (0.0081, 0.0006),

        # Higgs sector (1)
        'lambda_H': (0.129, 0.002),

        # Cosmological (10)
        'Omega_DE': (0.6847, 0.0056),
        'Omega_DM': (0.265, 0.007),
        'Omega_b': (0.0493, 0.0006),
        'n_s': (0.9649, 0.0042),
        'sigma_8': (0.811, 0.006),
        'A_s': (2.1e-9, 0.03e-9),
        'Omega_gamma': (5.38e-5, 0.15e-5),
        'Omega_nu': (0.00064, 0.00014),
        'Y_p': (0.2449, 0.0040),
        'D_H': (2.547e-5, 0.025e-5),

        # === DIMENSIONAL (9 total) ===
        # Electroweak scale (3)
        'v_EW': (246.22, 0.03),  # GeV
        'M_W': (80.369, 0.023),  # GeV
        'M_Z': (91.188, 0.002),  # GeV

        # Quark masses (6)
        'm_u_MeV': (2.16, 0.04),
        'm_d_MeV': (4.67, 0.04),
        'm_s_MeV': (93.4, 0.8),
        'm_c_MeV': (1270.0, 20.0),
        'm_b_MeV': (4180.0, 30.0),
        'm_t_GeV': (172.76, 0.30),

        # Cosmological scale (1)
        'H0': (70.0, 2.0),  # km/s/Mpc (compromise value)
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
