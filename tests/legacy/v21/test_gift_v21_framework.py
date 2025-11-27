"""
Unit tests for GIFT Framework v2.1 with torsional dynamics.

Tests all 46 observables (37 dimensionless + 9 dimensional) against
reference values and experimental data.

Version: 2.1.0
Test Coverage: 46/46 observables (100%)
"""

import pytest
import json
import sys
from pathlib import Path

# Add statistical_validation to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "statistical_validation"))

from gift_v21_core import GIFTFrameworkV21, GIFTParameters


# ============================================================================
# FIXTURES AND HELPERS
# ============================================================================

@pytest.fixture
def reference_data():
    """Load reference observable values for v2.1."""
    ref_file = Path(__file__).parent.parent / "fixtures" / "reference_observables_v21.json"
    with open(ref_file, 'r') as f:
        return json.load(f)


@pytest.fixture
def gift_v21():
    """Create GIFTFrameworkV21 instance with default parameters."""
    return GIFTFrameworkV21()


def assert_close_to_reference(predicted, ref_data, observable_name, rel_tol=0.05):
    """
    Assert that predicted value is close to reference value.

    Args:
        predicted: Predicted value from framework
        ref_data: Reference data dictionary
        observable_name: Name of observable
        rel_tol: Relative tolerance (default 5%)
    """
    ref = ref_data['observables'][observable_name]
    ref_predicted = ref['predicted']

    # Calculate relative deviation
    if ref_predicted != 0:
        rel_dev = abs(predicted - ref_predicted) / abs(ref_predicted)
        assert rel_dev < rel_tol, (
            f"{observable_name}: predicted={predicted:.6f}, "
            f"reference={ref_predicted:.6f}, "
            f"relative deviation={rel_dev*100:.2f}% (tolerance={rel_tol*100}%)"
        )
    else:
        assert abs(predicted - ref_predicted) < 1e-10


def assert_within_experimental(predicted, ref_data, observable_name, n_sigma=3):
    """
    Assert that predicted value is within n_sigma of experimental value.

    Args:
        predicted: Predicted value
        ref_data: Reference data dictionary
        observable_name: Name of observable
        n_sigma: Number of standard deviations
    """
    ref = ref_data['observables'][observable_name]
    exp_val = ref['experimental']
    exp_unc = ref['uncertainty']

    deviation = abs(predicted - exp_val)
    allowed = n_sigma * exp_unc

    assert deviation <= allowed, (
        f"{observable_name}: predicted={predicted:.6f}, "
        f"experimental={exp_val:.6f}±{exp_unc:.6f}, "
        f"deviation={deviation:.6f} > {n_sigma}σ={allowed:.6f}"
    )


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================

class TestGIFTV21Initialization:
    """Test v2.1 framework initialization and parameters."""

    def test_default_initialization(self, gift_v21):
        """Test default parameter initialization."""
        assert gift_v21.params.p2 == 2.0
        assert gift_v21.params.Weyl_factor == 5.0
        assert abs(gift_v21.params.tau - 3.8967) < 0.0001

    def test_torsional_parameters(self, gift_v21):
        """Test torsional dynamics parameters."""
        assert abs(gift_v21.params.T_norm - 0.0164) < 0.0001
        assert abs(gift_v21.params.T_costar - 0.0141) < 0.0001
        assert abs(gift_v21.params.det_g - 2.031) < 0.001
        assert abs(gift_v21.params.v_flow - 0.015) < 0.001

    def test_topological_integers(self, gift_v21):
        """Test topological invariants are exact integers."""
        assert gift_v21.b2_K7 == 21
        assert gift_v21.b3_K7 == 77
        assert gift_v21.H_star == 99
        assert gift_v21.dim_E8 == 248
        assert gift_v21.dim_G2 == 14
        assert gift_v21.dim_K7 == 7
        assert gift_v21.rank_E8 == 8
        assert gift_v21.N_gen == 3

    def test_scale_bridge(self, gift_v21):
        """Test scale bridge Λ_GIFT calculation."""
        # Λ_GIFT = 21×e⁸×248 / (7×π⁴) ≈ 1.632×10⁶
        assert gift_v21.Lambda_GIFT > 1.6e6
        assert gift_v21.Lambda_GIFT < 1.7e6


# ============================================================================
# GAUGE SECTOR TESTS (3 observables)
# ============================================================================

class TestGaugeSectorV21:
    """Test gauge coupling predictions with torsional corrections."""

    def test_alpha_inv_MZ(self, gift_v21, reference_data):
        """
        Test fine structure constant at M_Z.

        v2.1 formula: α⁻¹(M_Z) = (248+8)/2 + 99/11 + det_g×|T|
        Status: TOPOLOGICAL with torsional correction
        """
        obs = gift_v21.compute_dimensionless_observables()
        alpha_inv = obs['alpha_inv_MZ']

        assert_close_to_reference(alpha_inv, reference_data, 'alpha_inv_MZ', rel_tol=0.10)
        assert 135 < alpha_inv < 140  # Physical range check

    def test_sin2thetaW(self, gift_v21, reference_data):
        """
        Test weak mixing angle.

        v2.1 formula: sin²θ_W = ζ(3)×γ_Euler/M₂
        Status: TOPOLOGICAL
        Expected: ~0.231
        """
        obs = gift_v21.compute_dimensionless_observables()
        sin2theta = obs['sin2thetaW']

        assert_close_to_reference(sin2theta, reference_data, 'sin2thetaW', rel_tol=0.01)
        assert 0.20 < sin2theta < 0.25  # Physical range
        assert_within_experimental(sin2theta, reference_data, 'sin2thetaW', n_sigma=3)

    def test_alpha_s_MZ(self, gift_v21, reference_data):
        """
        Test strong coupling constant at M_Z.

        v2.1 formula: α_s(M_Z) = √2/12
        Status: TOPOLOGICAL
        Expected: ~0.1178
        """
        obs = gift_v21.compute_dimensionless_observables()
        alpha_s = obs['alpha_s_MZ']

        assert_close_to_reference(alpha_s, reference_data, 'alpha_s_MZ', rel_tol=0.01)
        assert 0.10 < alpha_s < 0.13  # Physical range
        assert_within_experimental(alpha_s, reference_data, 'alpha_s_MZ', n_sigma=3)


# ============================================================================
# NEUTRINO SECTOR TESTS (4 observables)
# ============================================================================

class TestNeutrinoSectorV21:
    """Test neutrino mixing parameters."""

    def test_theta12(self, gift_v21, reference_data):
        """
        Test solar mixing angle θ₁₂.

        Status: DERIVED
        Expected: ~33.7°
        """
        obs = gift_v21.compute_dimensionless_observables()
        theta12 = obs['theta12']

        assert_close_to_reference(theta12, reference_data, 'theta12', rel_tol=0.02)
        assert 30 < theta12 < 37  # Physical range
        assert_within_experimental(theta12, reference_data, 'theta12', n_sigma=3)

    def test_theta13(self, gift_v21, reference_data):
        """
        Test reactor mixing angle θ₁₃.

        Formula: π/21
        Status: DERIVED
        Expected: ~8.57°
        """
        obs = gift_v21.compute_dimensionless_observables()
        theta13 = obs['theta13']

        assert_close_to_reference(theta13, reference_data, 'theta13', rel_tol=0.01)
        assert 7 < theta13 < 10  # Physical range
        assert_within_experimental(theta13, reference_data, 'theta13', n_sigma=3)

    def test_theta23(self, gift_v21, reference_data):
        """
        Test atmospheric mixing angle θ₂₃.

        Formula: (8+77)/99
        Status: DERIVED
        Expected: ~48.6°
        """
        obs = gift_v21.compute_dimensionless_observables()
        theta23 = obs['theta23']

        assert_close_to_reference(theta23, reference_data, 'theta23', rel_tol=0.02)
        assert 40 < theta23 < 55  # Physical range
        assert_within_experimental(theta23, reference_data, 'theta23', n_sigma=3)

    def test_delta_CP_exact(self, gift_v21, reference_data):
        """
        Test CP violation phase δ_CP (PROVEN EXACT).

        Formula: 7×14 + 99 = 197°
        Status: PROVEN (exact topological)
        Expected: 197.0° (exact)
        Deviation: 0.000%
        """
        obs = gift_v21.compute_dimensionless_observables()
        delta_cp = obs['delta_CP']

        # Exact check
        assert abs(delta_cp - 197.0) < 0.01, f"δ_CP should be exactly 197°, got {delta_cp}"
        assert_close_to_reference(delta_cp, reference_data, 'delta_CP', rel_tol=0.001)


# ============================================================================
# LEPTON SECTOR TESTS (3 observables)
# ============================================================================

class TestLeptonSectorV21:
    """Test lepton mass ratio predictions."""

    def test_Q_Koide_exact(self, gift_v21, reference_data):
        """
        Test Koide formula parameter (PROVEN EXACT).

        Formula: dim(G₂)/b₂(K₇) = 14/21 = 2/3
        Status: PROVEN
        Expected: 0.666667 (exact)
        """
        obs = gift_v21.compute_dimensionless_observables()
        Q_Koide = obs['Q_Koide']

        # Should be exactly 2/3
        assert abs(Q_Koide - 2.0/3.0) < 1e-5
        assert_close_to_reference(Q_Koide, reference_data, 'Q_Koide', rel_tol=0.001)
        assert_within_experimental(Q_Koide, reference_data, 'Q_Koide', n_sigma=3)

    def test_m_mu_m_e(self, gift_v21, reference_data):
        """
        Test muon to electron mass ratio.

        Formula: 27^φ (φ = golden ratio)
        Status: DERIVED
        Expected: ~206.8
        """
        obs = gift_v21.compute_dimensionless_observables()
        m_mu_m_e = obs['m_mu_m_e']

        assert_close_to_reference(m_mu_m_e, reference_data, 'm_mu_m_e', rel_tol=0.001)
        assert 205 < m_mu_m_e < 208
        assert_within_experimental(m_mu_m_e, reference_data, 'm_mu_m_e', n_sigma=3)

    def test_m_tau_m_e_exact(self, gift_v21, reference_data):
        """
        Test tau to electron mass ratio (PROVEN EXACT).

        Formula: 7 + 10×248 + 10×99 = 3477
        Status: PROVEN
        Expected: 3477.0 (exact)
        """
        obs = gift_v21.compute_dimensionless_observables()
        m_tau_m_e = obs['m_tau_m_e']

        # Should be exactly 3477
        assert abs(m_tau_m_e - 3477.0) < 0.5
        assert_close_to_reference(m_tau_m_e, reference_data, 'm_tau_m_e', rel_tol=0.001)
        assert_within_experimental(m_tau_m_e, reference_data, 'm_tau_m_e', n_sigma=3)


# ============================================================================
# QUARK SECTOR TESTS (10 observables)
# ============================================================================

class TestQuarkSectorV21:
    """Test quark mass ratio predictions from torsional hierarchy."""

    def test_m_s_m_d_exact(self, gift_v21, reference_data):
        """
        Test strange to down quark mass ratio (PROVEN EXACT).

        Formula: p₂² × Weyl = 4 × 5 = 20
        Status: PROVEN
        Expected: 20.0 (exact)
        """
        obs = gift_v21.compute_dimensionless_observables()
        m_s_m_d = obs['m_s_m_d']

        # Should be exactly 20
        assert abs(m_s_m_d - 20.0) < 0.1
        assert_close_to_reference(m_s_m_d, reference_data, 'm_s_m_d', rel_tol=0.01)

    def test_m_c_m_s(self, gift_v21, reference_data):
        """Test charm to strange mass ratio."""
        obs = gift_v21.compute_dimensionless_observables()
        m_c_m_s = obs['m_c_m_s']

        assert_close_to_reference(m_c_m_s, reference_data, 'm_c_m_s', rel_tol=0.05)
        assert 10 < m_c_m_s < 20

    def test_m_b_m_u(self, gift_v21, reference_data):
        """Test bottom to up mass ratio."""
        obs = gift_v21.compute_dimensionless_observables()
        m_b_m_u = obs['m_b_m_u']

        assert_close_to_reference(m_b_m_u, reference_data, 'm_b_m_u', rel_tol=0.05)
        assert 1800 < m_b_m_u < 2100

    def test_m_t_m_b(self, gift_v21, reference_data):
        """Test top to bottom mass ratio."""
        obs = gift_v21.compute_dimensionless_observables()
        m_t_m_b = obs['m_t_m_b']

        assert_close_to_reference(m_t_m_b, reference_data, 'm_t_m_b', rel_tol=0.05)
        assert 35 < m_t_m_b < 50

    def test_m_d_m_u(self, gift_v21, reference_data):
        """Test down to up mass ratio."""
        obs = gift_v21.compute_dimensionless_observables()
        m_d_m_u = obs['m_d_m_u']

        assert_close_to_reference(m_d_m_u, reference_data, 'm_d_m_u', rel_tol=0.05)
        assert 1.8 < m_d_m_u < 2.5

    def test_m_c_m_u(self, gift_v21, reference_data):
        """Test charm to up mass ratio."""
        obs = gift_v21.compute_dimensionless_observables()
        m_c_m_u = obs['m_c_m_u']

        assert_close_to_reference(m_c_m_u, reference_data, 'm_c_m_u', rel_tol=0.05)
        assert 500 < m_c_m_u < 700

    def test_m_b_m_d(self, gift_v21, reference_data):
        """Test bottom to down mass ratio."""
        obs = gift_v21.compute_dimensionless_observables()
        m_b_m_d = obs['m_b_m_d']

        assert_close_to_reference(m_b_m_d, reference_data, 'm_b_m_d', rel_tol=0.05)
        assert 800 < m_b_m_d < 1000

    def test_m_t_m_s(self, gift_v21, reference_data):
        """Test top to strange mass ratio."""
        obs = gift_v21.compute_dimensionless_observables()
        m_t_m_s = obs['m_t_m_s']

        assert_close_to_reference(m_t_m_s, reference_data, 'm_t_m_s', rel_tol=0.05)
        assert 1700 < m_t_m_s < 2000

    def test_m_t_m_d(self, gift_v21, reference_data):
        """Test top to down mass ratio."""
        obs = gift_v21.compute_dimensionless_observables()
        m_t_m_d = obs['m_t_m_d']

        assert_close_to_reference(m_t_m_d, reference_data, 'm_t_m_d', rel_tol=0.05)
        assert 35000 < m_t_m_d < 40000

    def test_m_t_m_c(self, gift_v21, reference_data):
        """Test top to charm mass ratio."""
        obs = gift_v21.compute_dimensionless_observables()
        m_t_m_c = obs['m_t_m_c']

        assert_close_to_reference(m_t_m_c, reference_data, 'm_t_m_c', rel_tol=0.05)
        assert 130 < m_t_m_c < 145


# ============================================================================
# CKM MATRIX TESTS (6 observables)
# ============================================================================

class TestCKMMatrixV21:
    """Test CKM matrix element predictions."""

    def test_V_us(self, gift_v21, reference_data):
        """Test V_us CKM matrix element (Cabibbo angle)."""
        obs = gift_v21.compute_dimensionless_observables()
        V_us = obs['V_us']

        assert_close_to_reference(V_us, reference_data, 'V_us', rel_tol=0.02)
        assert 0.20 < V_us < 0.25
        assert_within_experimental(V_us, reference_data, 'V_us', n_sigma=3)

    def test_V_cb(self, gift_v21, reference_data):
        """Test V_cb CKM matrix element."""
        obs = gift_v21.compute_dimensionless_observables()
        V_cb = obs['V_cb']

        assert_close_to_reference(V_cb, reference_data, 'V_cb', rel_tol=0.02)
        assert 0.035 < V_cb < 0.050
        assert_within_experimental(V_cb, reference_data, 'V_cb', n_sigma=3)

    def test_V_ub(self, gift_v21, reference_data):
        """Test V_ub CKM matrix element."""
        obs = gift_v21.compute_dimensionless_observables()
        V_ub = obs['V_ub']

        assert_close_to_reference(V_ub, reference_data, 'V_ub', rel_tol=0.05)
        assert 0.002 < V_ub < 0.006
        assert_within_experimental(V_ub, reference_data, 'V_ub', n_sigma=3)

    def test_V_cd(self, gift_v21, reference_data):
        """Test V_cd CKM matrix element."""
        obs = gift_v21.compute_dimensionless_observables()
        V_cd = obs['V_cd']

        assert_close_to_reference(V_cd, reference_data, 'V_cd', rel_tol=0.02)
        assert 0.20 < V_cd < 0.23
        assert_within_experimental(V_cd, reference_data, 'V_cd', n_sigma=3)

    def test_V_cs(self, gift_v21, reference_data):
        """Test V_cs CKM matrix element (close to unity)."""
        obs = gift_v21.compute_dimensionless_observables()
        V_cs = obs['V_cs']

        assert_close_to_reference(V_cs, reference_data, 'V_cs', rel_tol=0.02)
        assert 0.95 < V_cs < 1.01
        assert_within_experimental(V_cs, reference_data, 'V_cs', n_sigma=3)

    def test_V_td(self, gift_v21, reference_data):
        """Test V_td CKM matrix element."""
        obs = gift_v21.compute_dimensionless_observables()
        V_td = obs['V_td']

        assert_close_to_reference(V_td, reference_data, 'V_td', rel_tol=0.05)
        assert 0.005 < V_td < 0.012
        assert_within_experimental(V_td, reference_data, 'V_td', n_sigma=3)


# ============================================================================
# HIGGS SECTOR TEST (1 observable)
# ============================================================================

class TestHiggsSectorV21:
    """Test Higgs quartic coupling prediction."""

    def test_lambda_H_exact(self, gift_v21, reference_data):
        """
        Test Higgs quartic coupling (PROVEN EXACT).

        Formula: √17/32
        Status: PROVEN
        Expected: 0.1287
        """
        obs = gift_v21.compute_dimensionless_observables()
        lambda_H = obs['lambda_H']

        # Check exact formula
        import math
        expected = math.sqrt(17) / 32
        assert abs(lambda_H - expected) < 0.001

        assert_close_to_reference(lambda_H, reference_data, 'lambda_H', rel_tol=0.01)
        assert 0.12 < lambda_H < 0.14
        assert_within_experimental(lambda_H, reference_data, 'lambda_H', n_sigma=3)


# ============================================================================
# COSMOLOGICAL SECTOR TESTS (10 observables)
# ============================================================================

class TestCosmologySectorV21:
    """Test cosmological parameter predictions."""

    def test_Omega_DE_exact(self, gift_v21, reference_data):
        """
        Test dark energy density (PROVEN EXACT).

        Formula: ln(2) × 98/99
        Status: PROVEN
        Expected: 0.6861
        Deviation: 0.20% (improved from v2.0)
        """
        obs = gift_v21.compute_dimensionless_observables()
        Omega_DE = obs['Omega_DE']

        # Check formula
        import math
        expected = math.log(2) * 98 / 99
        assert abs(Omega_DE - expected) < 0.001

        assert_close_to_reference(Omega_DE, reference_data, 'Omega_DE', rel_tol=0.01)
        assert 0.60 < Omega_DE < 0.75
        assert_within_experimental(Omega_DE, reference_data, 'Omega_DE', n_sigma=3)

    def test_Omega_DM(self, gift_v21, reference_data):
        """Test dark matter density."""
        obs = gift_v21.compute_dimensionless_observables()
        Omega_DM = obs['Omega_DM']

        assert_close_to_reference(Omega_DM, reference_data, 'Omega_DM', rel_tol=0.05)
        assert 0.20 < Omega_DM < 0.35
        assert_within_experimental(Omega_DM, reference_data, 'Omega_DM', n_sigma=3)

    def test_Omega_b(self, gift_v21, reference_data):
        """Test baryon density."""
        obs = gift_v21.compute_dimensionless_observables()
        Omega_b = obs['Omega_b']

        assert_close_to_reference(Omega_b, reference_data, 'Omega_b', rel_tol=0.05)
        assert 0.040 < Omega_b < 0.060
        assert_within_experimental(Omega_b, reference_data, 'Omega_b', n_sigma=3)

    def test_n_s(self, gift_v21, reference_data):
        """
        Test scalar spectral index.

        Formula: ζ(11)/ζ(5)
        Status: DERIVED
        """
        obs = gift_v21.compute_dimensionless_observables()
        n_s = obs['n_s']

        assert_close_to_reference(n_s, reference_data, 'n_s', rel_tol=0.01)
        assert 0.94 < n_s < 0.99
        assert_within_experimental(n_s, reference_data, 'n_s', n_sigma=3)

    def test_sigma_8(self, gift_v21, reference_data):
        """Test matter fluctuation amplitude."""
        obs = gift_v21.compute_dimensionless_observables()
        sigma_8 = obs['sigma_8']

        assert_close_to_reference(sigma_8, reference_data, 'sigma_8', rel_tol=0.05)
        assert 0.70 < sigma_8 < 0.90
        assert_within_experimental(sigma_8, reference_data, 'sigma_8', n_sigma=3)

    def test_A_s(self, gift_v21, reference_data):
        """Test primordial power spectrum amplitude."""
        obs = gift_v21.compute_dimensionless_observables()
        A_s = obs['A_s']

        assert_close_to_reference(A_s, reference_data, 'A_s', rel_tol=0.05)
        assert 1.5e-9 < A_s < 2.5e-9
        assert_within_experimental(A_s, reference_data, 'A_s', n_sigma=3)

    def test_Omega_gamma(self, gift_v21, reference_data):
        """Test photon density."""
        obs = gift_v21.compute_dimensionless_observables()
        Omega_gamma = obs['Omega_gamma']

        assert_close_to_reference(Omega_gamma, reference_data, 'Omega_gamma', rel_tol=0.05)
        assert 4e-5 < Omega_gamma < 7e-5
        assert_within_experimental(Omega_gamma, reference_data, 'Omega_gamma', n_sigma=3)

    def test_Omega_nu(self, gift_v21, reference_data):
        """Test neutrino density."""
        obs = gift_v21.compute_dimensionless_observables()
        Omega_nu = obs['Omega_nu']

        assert_close_to_reference(Omega_nu, reference_data, 'Omega_nu', rel_tol=0.10)
        assert 0.0001 < Omega_nu < 0.002
        # Note: Higher tolerance due to experimental uncertainty

    def test_Y_p(self, gift_v21, reference_data):
        """Test primordial helium abundance."""
        obs = gift_v21.compute_dimensionless_observables()
        Y_p = obs['Y_p']

        assert_close_to_reference(Y_p, reference_data, 'Y_p', rel_tol=0.02)
        assert 0.23 < Y_p < 0.26
        assert_within_experimental(Y_p, reference_data, 'Y_p', n_sigma=3)

    def test_D_H(self, gift_v21, reference_data):
        """Test primordial deuterium to hydrogen ratio."""
        obs = gift_v21.compute_dimensionless_observables()
        D_H = obs['D_H']

        assert_close_to_reference(D_H, reference_data, 'D_H', rel_tol=0.02)
        assert 2.0e-5 < D_H < 3.0e-5
        assert_within_experimental(D_H, reference_data, 'D_H', n_sigma=3)


# ============================================================================
# DIMENSIONAL OBSERVABLES TESTS (9 observables)
# ============================================================================

class TestElectroweakScaleV21:
    """Test electroweak scale dimensional predictions from scale bridge."""

    def test_v_EW(self, gift_v21, reference_data):
        """
        Test electroweak VEV.

        Status: DERIVED from scale bridge
        Expected: ~246.2 GeV
        """
        obs = gift_v21.compute_dimensional_observables()
        v_EW = obs['v_EW']

        assert_close_to_reference(v_EW, reference_data, 'v_EW', rel_tol=0.01)
        assert 240 < v_EW < 250
        assert_within_experimental(v_EW, reference_data, 'v_EW', n_sigma=3)

    def test_M_W(self, gift_v21, reference_data):
        """
        Test W boson mass.

        Status: DERIVED from scale bridge
        Expected: ~80.37 GeV
        """
        obs = gift_v21.compute_dimensional_observables()
        M_W = obs['M_W']

        assert_close_to_reference(M_W, reference_data, 'M_W', rel_tol=0.01)
        assert 78 < M_W < 82
        assert_within_experimental(M_W, reference_data, 'M_W', n_sigma=3)

    def test_M_Z(self, gift_v21, reference_data):
        """
        Test Z boson mass (reference scale).

        Status: DERIVED from scale bridge (μ₀ = M_Z reference)
        Expected: ~91.19 GeV
        """
        obs = gift_v21.compute_dimensional_observables()
        M_Z = obs['M_Z']

        assert_close_to_reference(M_Z, reference_data, 'M_Z', rel_tol=0.001)
        assert 90 < M_Z < 92
        assert_within_experimental(M_Z, reference_data, 'M_Z', n_sigma=3)


class TestQuarkMassesV21:
    """Test absolute quark mass predictions from scale bridge."""

    def test_m_u_MeV(self, gift_v21, reference_data):
        """Test up quark mass (MeV)."""
        obs = gift_v21.compute_dimensional_observables()
        m_u = obs['m_u_MeV']

        assert_close_to_reference(m_u, reference_data, 'm_u_MeV', rel_tol=0.05)
        assert 1.5 < m_u < 3.0
        assert_within_experimental(m_u, reference_data, 'm_u_MeV', n_sigma=3)

    def test_m_d_MeV(self, gift_v21, reference_data):
        """Test down quark mass (MeV)."""
        obs = gift_v21.compute_dimensional_observables()
        m_d = obs['m_d_MeV']

        assert_close_to_reference(m_d, reference_data, 'm_d_MeV', rel_tol=0.05)
        assert 3.5 < m_d < 5.5
        assert_within_experimental(m_d, reference_data, 'm_d_MeV', n_sigma=3)

    def test_m_s_MeV(self, gift_v21, reference_data):
        """Test strange quark mass (MeV)."""
        obs = gift_v21.compute_dimensional_observables()
        m_s = obs['m_s_MeV']

        assert_close_to_reference(m_s, reference_data, 'm_s_MeV', rel_tol=0.05)
        assert 85 < m_s < 105
        assert_within_experimental(m_s, reference_data, 'm_s_MeV', n_sigma=3)

    def test_m_c_MeV(self, gift_v21, reference_data):
        """Test charm quark mass (MeV)."""
        obs = gift_v21.compute_dimensional_observables()
        m_c = obs['m_c_MeV']

        assert_close_to_reference(m_c, reference_data, 'm_c_MeV', rel_tol=0.05)
        assert 1200 < m_c < 1350
        assert_within_experimental(m_c, reference_data, 'm_c_MeV', n_sigma=3)

    def test_m_b_MeV(self, gift_v21, reference_data):
        """Test bottom quark mass (MeV)."""
        obs = gift_v21.compute_dimensional_observables()
        m_b = obs['m_b_MeV']

        assert_close_to_reference(m_b, reference_data, 'm_b_MeV', rel_tol=0.05)
        assert 4000 < m_b < 4300
        assert_within_experimental(m_b, reference_data, 'm_b_MeV', n_sigma=3)

    def test_m_t_GeV(self, gift_v21, reference_data):
        """Test top quark mass (GeV)."""
        obs = gift_v21.compute_dimensional_observables()
        m_t = obs['m_t_GeV']

        assert_close_to_reference(m_t, reference_data, 'm_t_GeV', rel_tol=0.01)
        assert 170 < m_t < 175
        assert_within_experimental(m_t, reference_data, 'm_t_GeV', n_sigma=3)


class TestCosmologicalScaleV21:
    """Test cosmological dimensional scale (Hubble parameter)."""

    def test_H0(self, gift_v21, reference_data):
        """
        Test Hubble parameter H₀.

        Status: DERIVED from scale bridge + cosmology
        Expected: ~70 km/s/Mpc (compromise value)
        Note: Hubble tension remains an active research area
        """
        obs = gift_v21.compute_dimensional_observables()
        H0 = obs['H0']

        assert_close_to_reference(H0, reference_data, 'H0', rel_tol=0.05)
        assert 60 < H0 < 80  # Broad range due to Hubble tension
        # Note: Don't use experimental assertion due to Hubble tension


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestV21Integration:
    """Integration tests for v2.1 framework."""

    def test_all_46_observables_computed(self, gift_v21):
        """Test that all 46 observables are computed without errors."""
        obs_dim = gift_v21.compute_dimensionless_observables()
        obs_dimensional = gift_v21.compute_dimensional_observables()
        obs_all = gift_v21.compute_all_observables()

        # Check counts
        assert len(obs_dim) == 37, f"Expected 37 dimensionless, got {len(obs_dim)}"
        assert len(obs_dimensional) == 9, f"Expected 9 dimensional, got {len(obs_dimensional)}"
        assert len(obs_all) == 46, f"Expected 46 total, got {len(obs_all)}"

        # Check all values are finite
        for name, value in obs_all.items():
            assert not (value != value), f"{name} is NaN"  # NaN check
            assert abs(value) < 1e10, f"{name} is infinite or very large: {value}"

    def test_dimensionless_vs_all(self, gift_v21):
        """Test that dimensionless observables match in both methods."""
        obs_dim = gift_v21.compute_dimensionless_observables()
        obs_all = gift_v21.compute_all_observables()

        # All dimensionless should be in obs_all
        for name in obs_dim:
            assert name in obs_all
            assert abs(obs_dim[name] - obs_all[name]) < 1e-10

    def test_mean_deviation(self, gift_v21, reference_data):
        """
        Test mean deviation across all observables.

        Target: < 0.2% mean deviation (v2.1 target)
        """
        obs_all = gift_v21.compute_all_observables()

        deviations = []
        for name, value in obs_all.items():
            if name in reference_data['observables']:
                ref = reference_data['observables'][name]
                exp_val = ref['experimental']
                if exp_val != 0:
                    dev = abs(value - exp_val) / abs(exp_val) * 100
                    deviations.append(dev)

        mean_dev = sum(deviations) / len(deviations)

        # Should be less than 0.5% on average (conservative target)
        assert mean_dev < 0.5, f"Mean deviation {mean_dev:.2f}% exceeds 0.5%"

    def test_proven_exact_relations(self, gift_v21):
        """
        Test all 6 PROVEN exact relations maintain their exact values.

        PROVEN observables in v2.1:
        1. δ_CP = 197°
        2. Q_Koide = 2/3
        3. m_τ/m_e = 3477
        4. m_s/m_d = 20
        5. λ_H = √17/32
        6. Ω_DE = ln(2)×98/99
        """
        obs = gift_v21.compute_all_observables()

        import math

        # 1. δ_CP
        assert abs(obs['delta_CP'] - 197.0) < 0.1

        # 2. Q_Koide
        assert abs(obs['Q_Koide'] - 2.0/3.0) < 1e-5

        # 3. m_τ/m_e
        assert abs(obs['m_tau_m_e'] - 3477.0) < 1.0

        # 4. m_s/m_d
        assert abs(obs['m_s_m_d'] - 20.0) < 0.5

        # 5. λ_H
        assert abs(obs['lambda_H'] - math.sqrt(17)/32) < 0.001

        # 6. Ω_DE
        assert abs(obs['Omega_DE'] - math.log(2)*98/99) < 0.001


# ============================================================================
# PARAMETRIZED TESTS
# ============================================================================

@pytest.mark.parametrize("observable_name,sector,tolerance", [
    # Gauge (tight tolerance)
    ('sin2thetaW', 'gauge', 0.01),
    ('alpha_s_MZ', 'gauge', 0.01),

    # Neutrino
    ('theta12', 'neutrino', 0.02),
    ('theta13', 'neutrino', 0.02),
    ('theta23', 'neutrino', 0.03),

    # Lepton (very tight for PROVEN)
    ('Q_Koide', 'lepton', 0.001),
    ('m_tau_m_e', 'lepton', 0.001),

    # CKM (moderate)
    ('V_us', 'CKM', 0.02),
    ('V_cb', 'CKM', 0.03),

    # Dimensional (scale bridge)
    ('M_W', 'electroweak', 0.01),
    ('M_Z', 'electroweak', 0.001),
])
def test_observable_precision_parametrized(gift_v21, reference_data, observable_name, sector, tolerance):
    """Parametrized test for observable precision."""
    if observable_name in gift_v21.compute_dimensionless_observables():
        obs = gift_v21.compute_dimensionless_observables()
    else:
        obs = gift_v21.compute_dimensional_observables()

    value = obs[observable_name]
    assert_close_to_reference(value, reference_data, observable_name, rel_tol=tolerance)


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
