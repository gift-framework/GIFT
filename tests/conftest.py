"""
Pytest configuration and shared fixtures for GIFT framework tests.

Version: 2.2.0 (updated from 2.1.0)
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root and submodules to path BEFORE any imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "statistical_validation"))
sys.path.insert(0, str(PROJECT_ROOT / "assets" / "agents"))
sys.path.insert(0, str(PROJECT_ROOT / "giftpy"))


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "notebook: marks tests that run notebooks")


@pytest.fixture
def gift_framework_v22():
    """
    Create a GIFT Framework v2.2 instance with zero-parameter paradigm.

    Returns GIFTFrameworkV22 with all topological constants derived.
    """
    try:
        from gift_v22_core import GIFTFrameworkV22
        return GIFTFrameworkV22()
    except ImportError:
        pytest.skip("GIFTFrameworkV22 not available")


@pytest.fixture
def gift_params_v22():
    """
    Create GIFTParametersV22 instance.

    Returns the v2.2 parameters dataclass.
    """
    try:
        from gift_v22_core import GIFTParametersV22
        return GIFTParametersV22()
    except ImportError:
        pytest.skip("GIFTParametersV22 not available")


@pytest.fixture
def gift_framework():
    """
    Create a GIFT Framework v2.1 instance (legacy).

    Returns GIFTFrameworkV21 with torsional dynamics and scale bridge.
    """
    try:
        from gift_v21_core import GIFTFrameworkV21
        return GIFTFrameworkV21()
    except ImportError:
        pytest.skip("GIFTFrameworkV21 not available")


@pytest.fixture
def gift_framework_v20():
    """
    Legacy v2.0 framework for backwards compatibility tests.

    Returns GIFTFrameworkStatistical (v2.0) with static topology only.
    """
    try:
        from run_validation import GIFTFrameworkStatistical
        return GIFTFrameworkStatistical()
    except ImportError:
        pytest.skip("GIFTFrameworkStatistical (v2.0) not available")


@pytest.fixture
def experimental_data():
    """
    Provide experimental data for validation tests (39 observables for v2.2).

    Returns dictionary with (value, uncertainty) tuples (PDG 2024, NuFIT 5.3).
    """
    return {
        # === GAUGE SECTOR (3) ===
        'alpha_inv': (137.036, 0.000001),
        'sin2thetaW': (0.23122, 0.00003),
        'alpha_s_MZ': (0.1179, 0.0009),

        # === NEUTRINO SECTOR (4) ===
        'theta12': (33.41, 0.75),
        'theta13': (8.54, 0.12),
        'theta23': (49.3, 1.0),
        'delta_CP': (197.0, 24.0),

        # === LEPTON SECTOR (3) ===
        'Q_Koide': (0.666661, 0.000007),
        'm_mu_m_e': (206.768, 0.001),
        'm_tau_m_e': (3477.15, 0.01),

        # === QUARK RATIOS (9) ===
        'm_s_m_d': (20.0, 1.0),
        'm_c_m_s': (13.60, 0.5),
        'm_b_m_u': (1935.2, 10.0),
        'm_t_m_b': (41.3, 0.5),
        'm_c_m_d': (272.0, 12.0),
        'm_b_m_d': (893.0, 10.0),
        'm_t_m_c': (136.0, 2.0),
        'm_t_m_s': (1848.0, 60.0),
        'm_d_m_u': (2.16, 0.10),

        # === CKM MATRIX (6) ===
        'V_us': (0.2243, 0.0005),
        'V_cb': (0.0422, 0.0008),
        'V_ub': (0.00394, 0.00036),
        'V_td': (0.00867, 0.00031),
        'V_ts': (0.0415, 0.0009),
        'V_tb': (0.999105, 0.000032),

        # === ELECTROWEAK SCALE (3) ===
        'v_EW': (246.22, 0.01),
        'M_W': (80.369, 0.019),
        'M_Z': (91.188, 0.002),

        # === HIGGS SECTOR (1) ===
        'lambda_H': (0.126, 0.008),

        # === QUARK MASSES (6) ===
        'm_u_MeV': (2.16, 0.49),
        'm_d_MeV': (4.67, 0.48),
        'm_s_MeV': (93.4, 8.6),
        'm_c_MeV': (1270.0, 20.0),
        'm_b_MeV': (4180.0, 30.0),
        'm_t_GeV': (172.76, 0.30),

        # === COSMOLOGICAL (3) ===
        'Omega_DE': (0.6889, 0.0056),
        'n_s': (0.9649, 0.0042),
        'H0': (70.0, 2.0),

        # === v2.2 NEW (2) ===
        'kappa_T': (0.0164, 0.002),
        'tau': (3.8967, 0.0001),
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


# =============================================================================
# v2.2 Specific Fixtures
# =============================================================================

@pytest.fixture
def v22_proven_values():
    """
    Return v2.2 PROVEN exact values for testing.

    These are the 13 proven relations from v2.2.
    """
    from fractions import Fraction

    return {
        'N_gen': 3,
        'Q_Koide': Fraction(2, 3),
        'm_s_m_d': 20,
        'delta_CP': 197,
        'm_tau_m_e': 3477,
        'Omega_DE': np.log(2) * 98/99,
        'xi': 5 * np.pi / 16,
        'lambda_H': np.sqrt(17) / 32,
        'sin2thetaW': Fraction(3, 13),
        'tau': Fraction(3472, 891),
        'kappa_T': Fraction(1, 61),
        'det_g': Fraction(65, 32),
        'n_s': 1.0004941886041195 / 1.0369277551433699,  # zeta(11)/zeta(5)
    }


@pytest.fixture
def v22_topological_integers():
    """
    Return v2.2 topological integer values.
    """
    return {
        'b2_K7': 21,
        'b3_K7': 77,
        'H_star': 99,
        'dim_E8': 248,
        'dim_E8xE8': 496,
        'dim_G2': 14,
        'dim_K7': 7,
        'dim_J3O': 27,
        'rank_E8': 8,
        'N_gen': 3,
        'D_bulk': 11,
        'p2': 2,
        'Weyl_factor': 5,
    }
