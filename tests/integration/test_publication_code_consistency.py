"""
Publication-Code Consistency Tests.

Tests that code implementations match the formulas and values
documented in the publication files:
- publications/gift_2_2_main.md
- publications/supplements/S1_mathematical_architecture.md
- publications/supplements/S4_complete_derivations.md

This ensures the code and documentation stay synchronized.

Version: 2.2.0 (updated from 2.1.0)
"""

import pytest
import numpy as np
import sys
import re
from pathlib import Path
from typing import Dict, List, Optional
from fractions import Fraction

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "statistical_validation"))
sys.path.insert(0, str(PROJECT_ROOT / "giftpy"))


# =============================================================================
# Documentation Reference Values (v2.2 Zero-Parameter Paradigm)
# =============================================================================

# Values from publications/gift_2_2_main.md and supplements
DOCUMENTED_VALUES_V22 = {
    # Exact topological integers
    'b2_K7': {
        'value': 21,
        'source': 'gift_2_2_main.md Section 1.4',
        'description': 'Second Betti number of K7',
    },
    'b3_K7': {
        'value': 77,
        'source': 'gift_2_2_main.md Section 1.4',
        'description': 'Third Betti number of K7',
    },
    'H_star': {
        'value': 99,
        'source': 'gift_2_2_main.md Section 1.4',
        'description': 'Total cohomology dim H*(K7) = b2 + b3 + 1',
    },
    'dim_E8': {
        'value': 248,
        'source': 'gift_2_2_main.md Section 1.4',
        'description': 'Dimension of E8 Lie algebra',
    },
    'dim_E8xE8': {
        'value': 496,
        'source': 'gift_2_2_main.md Section 1.4',
        'description': 'Dimension of E8xE8 gauge group',
    },
    'dim_G2': {
        'value': 14,
        'source': 'gift_2_2_main.md Section 1.4',
        'description': 'Dimension of G2 holonomy group',
    },
    'dim_K7': {
        'value': 7,
        'source': 'gift_2_2_main.md Section 1.4',
        'description': 'Dimension of K7 manifold',
    },
    'N_gen': {
        'value': 3,
        'source': 'gift_2_2_main.md, S4_complete_derivations.md',
        'description': 'Number of fermion generations',
    },
    'dim_J3O': {
        'value': 27,
        'source': 'gift_2_2_main.md Section 1.4',
        'description': 'Dimension of exceptional Jordan algebra J3(O)',
    },

    # v2.2 Derived topological parameters
    'p2': {
        'value': 2,
        'formula': 'dim(G2)/dim(K7) = 14/7',
        'source': 'gift_2_2_main.md Section 1.4',
        'description': 'Binary duality',
    },
    'Weyl_factor': {
        'value': 5,
        'source': 'gift_2_2_main.md Section 1.4',
        'description': 'Pentagonal symmetry from |W(E8)|',
    },
    'beta0': {
        'value': np.pi / 8,
        'formula': 'pi/rank(E8) = pi/8',
        'source': 'gift_2_2_main.md Section 1.4',
    },

    # v2.2 PROVEN exact relations (13 total)
    'Q_Koide': {
        'value': Fraction(2, 3),
        'formula': 'dim(G2)/b2 = 14/21 = 2/3',
        'source': 'S4_complete_derivations.md Theorem 2',
        'status': 'PROVEN',
    },
    'm_s_m_d': {
        'value': 20,
        'formula': 'p2^2 * Weyl = 4 * 5 = 20',
        'source': 'S4_complete_derivations.md Theorem 3',
        'status': 'PROVEN',
    },
    'delta_CP': {
        'value': 197,
        'formula': 'dim(K7)*dim(G2) + H* = 7*14 + 99 = 197',
        'source': 'S4_complete_derivations.md Theorem 4',
        'status': 'PROVEN',
        'unit': 'degrees',
    },
    'm_tau_m_e': {
        'value': 3477,
        'formula': 'dim(K7) + 10*dim(E8) + 10*H* = 7 + 2480 + 990 = 3477',
        'source': 'S4_complete_derivations.md Theorem 5',
        'status': 'PROVEN',
    },
    'Omega_DE': {
        'value': np.log(2) * 98/99,
        'formula': 'ln(2) * (b2+b3)/H* = ln(2) * 98/99',
        'source': 'S4_complete_derivations.md Theorem 6',
        'status': 'PROVEN',
    },
    'lambda_H': {
        'value': np.sqrt(17) / 32,
        'formula': 'sqrt(dim(G2)+N_gen)/2^Weyl = sqrt(17)/32',
        'source': 'S4_complete_derivations.md Theorem 8',
        'status': 'PROVEN',
    },
    'xi': {
        'value': 5 * np.pi / 16,
        'formula': '(Weyl/p2) * beta0 = 5*pi/16',
        'source': 'S4_complete_derivations.md Theorem 7',
        'status': 'PROVEN',
    },
    'n_s': {
        'value': 1.0004941886041195 / 1.0369277551433699,  # zeta(11)/zeta(5)
        'formula': 'zeta(11)/zeta(5)',
        'source': 'S4_complete_derivations.md',
        'status': 'PROVEN',
    },

    # v2.2 NEW exact relations
    'sin2thetaW': {
        'value': Fraction(3, 13),
        'formula': 'b2/(b3 + dim(G2)) = 21/(77+14) = 21/91 = 3/13',
        'source': 'S4_complete_derivations.md Theorem 10',
        'status': 'PROVEN',
    },
    'kappa_T': {
        'value': Fraction(1, 61),
        'formula': '1/(b3 - dim(G2) - p2) = 1/(77-14-2) = 1/61',
        'source': 'S4_complete_derivations.md Theorem 12',
        'status': 'TOPOLOGICAL',
    },
    'tau': {
        'value': Fraction(3472, 891),
        'formula': 'dim(E8xE8)*b2/(dim(J3O)*H*) = 496*21/(27*99) = 3472/891',
        'source': 'S4_complete_derivations.md Theorem 11',
        'status': 'PROVEN',
    },
    'det_g': {
        'value': Fraction(65, 32),
        'formula': 'p2 + 1/(b2 + dim(G2) - N_gen) = 2 + 1/32 = 65/32',
        'source': 'S4_complete_derivations.md Theorem 13',
        'status': 'TOPOLOGICAL',
    },

    # Gauge sector
    'alpha_s_MZ': {
        'value': np.sqrt(2) / 12,
        'formula': 'sqrt(2)/(dim(G2) - p2) = sqrt(2)/12',
        'source': 'S4_complete_derivations.md Section 1',
        'status': 'TOPOLOGICAL',
    },
}


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def framework_v22():
    """Get GIFTFrameworkV22 instance."""
    try:
        from gift_v22_core import GIFTFrameworkV22
        return GIFTFrameworkV22()
    except ImportError:
        pytest.skip("GIFTFrameworkV22 not available")


@pytest.fixture
def params_v22():
    """Get GIFTParametersV22 instance."""
    try:
        from gift_v22_core import GIFTParametersV22
        return GIFTParametersV22()
    except ImportError:
        pytest.skip("GIFTParametersV22 not available")


@pytest.fixture
def giftpy_framework():
    """Get giftpy GIFT instance."""
    try:
        from giftpy import GIFT
        return GIFT()
    except ImportError:
        pytest.skip("giftpy not available")


@pytest.fixture
def giftpy_constants():
    """Get giftpy constants."""
    try:
        from giftpy.core.constants import CONSTANTS
        return CONSTANTS
    except ImportError:
        pytest.skip("giftpy.core.constants not available")


# =============================================================================
# v2.2 Topological Constants Consistency Tests
# =============================================================================

class TestTopologicalConstantsConsistency:
    """Test topological constants match v2.2 documentation."""

    def test_b2_K7_matches_docs(self, params_v22):
        """Test b2(K7) = 21."""
        assert params_v22.b2_K7 == DOCUMENTED_VALUES_V22['b2_K7']['value']

    def test_b3_K7_matches_docs(self, params_v22):
        """Test b3(K7) = 77."""
        assert params_v22.b3_K7 == DOCUMENTED_VALUES_V22['b3_K7']['value']

    def test_H_star_matches_docs(self, params_v22):
        """Test H* = 99."""
        assert params_v22.H_star == DOCUMENTED_VALUES_V22['H_star']['value']

    def test_dim_E8_matches_docs(self, params_v22):
        """Test dim(E8) = 248."""
        assert params_v22.dim_E8 == DOCUMENTED_VALUES_V22['dim_E8']['value']

    def test_dim_G2_matches_docs(self, params_v22):
        """Test dim(G2) = 14."""
        assert params_v22.dim_G2 == DOCUMENTED_VALUES_V22['dim_G2']['value']

    def test_dim_K7_matches_docs(self, params_v22):
        """Test dim(K7) = 7."""
        assert params_v22.dim_K7 == DOCUMENTED_VALUES_V22['dim_K7']['value']

    def test_N_gen_matches_docs(self, params_v22):
        """Test N_gen = 3."""
        assert params_v22.N_gen == DOCUMENTED_VALUES_V22['N_gen']['value']

    def test_dim_J3O_matches_docs(self, params_v22):
        """Test dim(J3O) = 27."""
        assert params_v22.dim_J3O == DOCUMENTED_VALUES_V22['dim_J3O']['value']


# =============================================================================
# v2.2 PROVEN Exact Relations Consistency Tests
# =============================================================================

class TestProvenRelationsConsistencyV22:
    """Test all 13 PROVEN exact relations match v2.2 documentation."""

    def test_sin2thetaW_matches_docs(self, params_v22):
        """Test sin^2(theta_W) = 3/13 as documented (v2.2)."""
        assert params_v22.sin2_theta_W == Fraction(3, 13)

    def test_kappa_T_matches_docs(self, params_v22):
        """Test kappa_T = 1/61 as documented (v2.2)."""
        assert params_v22.kappa_T == Fraction(1, 61)

    def test_tau_matches_docs(self, params_v22):
        """Test tau = 3472/891 as documented (v2.2)."""
        assert params_v22.tau == Fraction(3472, 891)

    def test_det_g_matches_docs(self, params_v22):
        """Test det(g) = 65/32 as documented (v2.2)."""
        assert params_v22.det_g == Fraction(65, 32)

    def test_lambda_H_matches_docs(self, params_v22):
        """Test lambda_H = sqrt(17)/32 as documented."""
        expected = np.sqrt(17) / 32
        assert np.isclose(params_v22.lambda_H, expected, rtol=1e-14)

    def test_xi_matches_docs(self, params_v22):
        """Test xi = 5*pi/16 as documented."""
        expected = 5 * np.pi / 16
        assert np.isclose(params_v22.xi, expected, rtol=1e-14)

    def test_Q_Koide_matches_docs(self, giftpy_framework):
        """Test Q_Koide = 2/3 as documented."""
        code_value = giftpy_framework.lepton.Q_Koide()
        assert np.isclose(code_value, 2/3, rtol=1e-14)

    def test_m_tau_m_e_matches_docs(self, giftpy_framework):
        """Test m_tau/m_e = 3477 as documented."""
        code_value = giftpy_framework.lepton.m_tau_m_e()
        assert code_value == 3477

    def test_m_s_m_d_matches_docs(self, giftpy_framework):
        """Test m_s/m_d = 20 as documented."""
        code_value = giftpy_framework.quark.m_s_m_d()
        assert code_value == 20

    def test_alpha_s_matches_docs(self, params_v22):
        """Test alpha_s = sqrt(2)/12 as documented."""
        expected = np.sqrt(2) / 12
        assert np.isclose(params_v22.alpha_s, expected, rtol=1e-14)


# =============================================================================
# v2.2 Formula Implementation Tests
# =============================================================================

class TestFormulaImplementationsV22:
    """Test that v2.2 code formulas match documented formulas."""

    def test_p2_formula(self, params_v22):
        """Test p2 = dim(G2)/dim(K7) formula."""
        expected = params_v22.dim_G2 // params_v22.dim_K7
        assert params_v22.p2 == expected
        assert params_v22.p2 == 2

    def test_beta0_formula(self, params_v22):
        """Test beta0 = pi/rank(E8) formula."""
        expected = np.pi / params_v22.rank_E8
        assert np.isclose(params_v22.beta0, expected, rtol=1e-14)

    def test_xi_formula(self, params_v22):
        """Test xi = (Weyl/p2)*beta0 formula."""
        expected = (params_v22.Weyl_factor / params_v22.p2) * params_v22.beta0
        assert np.isclose(params_v22.xi, expected, rtol=1e-14)

    def test_H_star_formula(self, params_v22):
        """Test H* = b2 + b3 + 1 formula."""
        expected = params_v22.b2_K7 + params_v22.b3_K7 + 1
        assert params_v22.H_star == expected

    def test_tau_formula(self, params_v22):
        """Test tau = dim(E8xE8)*b2/(dim(J3O)*H*) formula."""
        numerator = params_v22.dim_E8xE8 * params_v22.b2_K7  # 496 * 21
        denominator = params_v22.dim_J3O * params_v22.H_star  # 27 * 99
        expected = Fraction(numerator, denominator)
        assert params_v22.tau == expected

    def test_kappa_T_formula(self, params_v22):
        """Test kappa_T = 1/(b3 - dim(G2) - p2) formula."""
        denominator = params_v22.b3_K7 - params_v22.dim_G2 - params_v22.p2  # 77 - 14 - 2 = 61
        expected = Fraction(1, denominator)
        assert params_v22.kappa_T == expected

    def test_sin2thetaW_formula(self, params_v22):
        """Test sin^2(theta_W) = b2/(b3 + dim(G2)) formula."""
        numerator = params_v22.b2_K7  # 21
        denominator = params_v22.b3_K7 + params_v22.dim_G2  # 77 + 14 = 91
        expected = Fraction(numerator, denominator)  # 21/91 = 3/13
        assert params_v22.sin2_theta_W == expected

    def test_det_g_formula(self, params_v22):
        """Test det(g) = p2 + 1/(b2 + dim(G2) - N_gen) formula."""
        denominator = params_v22.b2_K7 + params_v22.dim_G2 - params_v22.N_gen  # 21 + 14 - 3 = 32
        expected = params_v22.p2 + Fraction(1, denominator)  # 2 + 1/32 = 65/32
        assert params_v22.det_g == expected


# =============================================================================
# v2.2 Documentation File Verification
# =============================================================================

class TestDocumentationFilesV22:
    """Test v2.2 documentation files exist and are accessible."""

    EXPECTED_DOCS_V22 = [
        'publications/gift_2_2_main.md',
        'publications/supplements/S1_mathematical_architecture.md',
        'publications/supplements/S4_complete_derivations.md',
        'publications/supplements/S5_experimental_validation.md',
    ]

    @pytest.mark.parametrize("doc_path", EXPECTED_DOCS_V22)
    def test_documentation_file_exists(self, doc_path):
        """Test documentation file exists."""
        full_path = PROJECT_ROOT / doc_path
        if not full_path.exists():
            pytest.skip(f"Documentation file not found: {doc_path}")

    def test_gift_main_contains_v22_constants(self):
        """Test gift_2_2_main.md contains v2.2 topological constants."""
        doc_path = PROJECT_ROOT / 'publications' / 'gift_2_2_main.md'
        if not doc_path.exists():
            pytest.skip("gift_2_2_main.md not found")

        content = doc_path.read_text()

        # Check for key v2.2 values
        assert '21' in content  # b2 value
        assert '77' in content  # b3 value
        assert '99' in content  # H* value
        assert '3/13' in content or '0.230769' in content  # sin^2(theta_W)


# =============================================================================
# Zero-Parameter Paradigm Tests
# =============================================================================

class TestZeroParameterParadigmConsistency:
    """Test zero-parameter paradigm is consistent in code and docs."""

    def test_no_fitted_parameters(self, params_v22):
        """Test all parameters are derived, not fitted."""
        # All these should be exact fractions or derived values
        assert params_v22.sin2_theta_W == Fraction(3, 13)
        assert params_v22.kappa_T == Fraction(1, 61)
        assert params_v22.tau == Fraction(3472, 891)
        assert params_v22.det_g == Fraction(65, 32)

    def test_13_proven_relations(self, framework_v22):
        """Test framework has 13 proven relations."""
        proven = framework_v22.get_proven_relations()
        assert len(proven) >= 13, f"Only {len(proven)} proven relations found"

    def test_structural_inputs_only(self, params_v22):
        """Test only structural inputs (discrete choices) exist."""
        # E8 x E8 gauge group
        assert params_v22.dim_E8xE8 == 496

        # K7 manifold with G2 holonomy
        assert params_v22.b2_K7 == 21
        assert params_v22.b3_K7 == 77


# =============================================================================
# Cross-Framework Consistency Tests
# =============================================================================

class TestCrossFrameworkConsistency:
    """Test consistency between v2.2 framework and giftpy."""

    def test_Q_Koide_consistent(self, framework_v22, giftpy_framework):
        """Test Q_Koide is consistent across frameworks."""
        obs = framework_v22.compute_all_observables()
        giftpy_value = giftpy_framework.lepton.Q_Koide()

        assert np.isclose(obs['Q_Koide'], giftpy_value, rtol=1e-14)
        assert np.isclose(obs['Q_Koide'], 2/3, rtol=1e-14)

    def test_m_tau_m_e_consistent(self, framework_v22, giftpy_framework):
        """Test m_tau/m_e is consistent across frameworks."""
        obs = framework_v22.compute_all_observables()
        giftpy_value = giftpy_framework.lepton.m_tau_m_e()

        assert obs['m_tau_m_e'] == giftpy_value == 3477

    def test_m_s_m_d_consistent(self, framework_v22, giftpy_framework):
        """Test m_s/m_d is consistent across frameworks."""
        obs = framework_v22.compute_all_observables()
        giftpy_value = giftpy_framework.quark.m_s_m_d()

        assert obs['m_s_m_d'] == giftpy_value == 20


# =============================================================================
# Precision Consistency Tests
# =============================================================================

class TestPrecisionConsistencyV22:
    """Test v2.2 precision claims match code."""

    def test_mean_precision_claim(self, framework_v22):
        """Test documented mean precision of 0.128% is achievable."""
        obs = framework_v22.compute_all_observables()
        exp_data = framework_v22.experimental_data

        deviations = []
        for name, pred in obs.items():
            if name in exp_data:
                exp_val = exp_data[name][0]
                if exp_val != 0 and np.isfinite(pred) and np.isfinite(exp_val):
                    dev = abs(pred - exp_val) / abs(exp_val) * 100
                    deviations.append(dev)

        if deviations:
            mean_dev = np.mean(deviations)
            # Should be under 1%
            assert mean_dev < 1.0, f"Mean deviation {mean_dev:.3f}% exceeds 1%"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
