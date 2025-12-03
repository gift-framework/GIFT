"""
Unit tests for mass spectrum extraction from Yukawa tensor.

Tests:
- FermionMasses dataclass
- MassSpectrum analysis
- GIFT predictions verification
- PDG comparisons
"""

import pytest
import torch
import numpy as np
import math
import sys
from pathlib import Path

# Add harmonic_yukawa to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from mass_spectrum import (
        PDG_2024,
        GIFT_PREDICTIONS,
        FermionMasses,
        MassSpectrum,
    )
    from yukawa import YukawaResult
    from config import HarmonicConfig, default_harmonic_config
    MASS_AVAILABLE = True
except ImportError as e:
    MASS_AVAILABLE = False
    MASS_IMPORT_ERROR = str(e)


pytestmark = pytest.mark.skipif(
    not MASS_AVAILABLE,
    reason=f"mass_spectrum module not available: {MASS_IMPORT_ERROR if not MASS_AVAILABLE else ''}"
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def config():
    return default_harmonic_config


@pytest.fixture
def realistic_eigenvalues():
    """Create eigenvalues with realistic hierarchy."""
    # Create eigenvalues spanning several orders of magnitude
    # This mimics the fermion mass hierarchy
    eigenvalues = torch.zeros(77)

    # Top sector (heaviest): eigenvalues 0-8
    eigenvalues[0] = 1e4      # Top-like
    eigenvalues[1] = 1e2      # Bottom-like
    eigenvalues[2] = 1e4      # Top-like
    eigenvalues[3] = 1e0      # Charm-like
    eigenvalues[4] = 1e-1     # Strange-like
    eigenvalues[5] = 1e0      # Charm-like
    eigenvalues[6] = 1e-4     # Up-like
    eigenvalues[7] = 1e-3     # Down-like
    eigenvalues[8] = 1e-4     # Up-like

    # Fill remaining with small values (hidden sector)
    eigenvalues[9:43] = torch.logspace(-5, -2, 34)
    eigenvalues[43:] = torch.logspace(-8, -6, 34)

    # Sort descending
    eigenvalues, _ = torch.sort(eigenvalues, descending=True)
    return eigenvalues


@pytest.fixture
def mock_yukawa_result(realistic_eigenvalues):
    """Create mock YukawaResult with realistic eigenvalues."""
    return YukawaResult(
        tensor=torch.zeros(21, 21, 77),
        gram_matrix=torch.diag(realistic_eigenvalues),
        eigenvalues=realistic_eigenvalues,
        eigenvectors=torch.eye(77),
        trace=realistic_eigenvalues.sum().item(),
        det=realistic_eigenvalues.prod().item() ** (1/77) if realistic_eigenvalues.prod().item() > 0 else 0.0
    )


# =============================================================================
# PDG Constants Tests
# =============================================================================

class TestPDGConstants:
    """Test PDG 2024 constants are correctly defined."""

    def test_pdg_has_leptons(self):
        """PDG should have charged lepton masses."""
        assert "m_e" in PDG_2024
        assert "m_mu" in PDG_2024
        assert "m_tau" in PDG_2024

    def test_pdg_has_quarks(self):
        """PDG should have quark masses."""
        for q in ["m_u", "m_c", "m_t", "m_d", "m_s", "m_b"]:
            assert q in PDG_2024

    def test_pdg_has_ratios(self):
        """PDG should have mass ratios."""
        assert "m_tau_m_e" in PDG_2024
        assert "m_s_m_d" in PDG_2024

    def test_pdg_lepton_ordering(self):
        """m_e < m_mu < m_tau."""
        assert PDG_2024["m_e"] < PDG_2024["m_mu"] < PDG_2024["m_tau"]

    def test_pdg_quark_ordering(self):
        """Light < heavy ordering within generations."""
        assert PDG_2024["m_u"] < PDG_2024["m_c"] < PDG_2024["m_t"]
        assert PDG_2024["m_d"] < PDG_2024["m_s"] < PDG_2024["m_b"]


# =============================================================================
# GIFT Predictions Tests
# =============================================================================

class TestGIFTPredictions:
    """Test GIFT predicted values."""

    def test_gift_tau_e_ratio(self):
        """GIFT predicts m_tau/m_e = 3477."""
        assert GIFT_PREDICTIONS["m_tau_m_e"] == 3477

    def test_gift_s_d_ratio(self):
        """GIFT predicts m_s/m_d = 20."""
        assert GIFT_PREDICTIONS["m_s_m_d"] == 20

    def test_gift_koide(self):
        """GIFT predicts Q_Koide = 2/3."""
        assert abs(GIFT_PREDICTIONS["Q_Koide"] - 2/3) < 1e-10

    def test_gift_generations(self):
        """GIFT predicts N_gen = 3."""
        assert GIFT_PREDICTIONS["N_gen"] == 3


# =============================================================================
# FermionMasses Tests
# =============================================================================

class TestFermionMasses:
    """Test FermionMasses dataclass."""

    def test_from_eigenvalues_creates_instance(self, realistic_eigenvalues):
        """from_eigenvalues should create valid instance."""
        masses = FermionMasses.from_eigenvalues(realistic_eigenvalues, scale=246.0)

        assert isinstance(masses, FermionMasses)
        assert masses.m_e > 0
        assert masses.m_tau > 0

    def test_from_eigenvalues_all_positive(self, realistic_eigenvalues):
        """All masses should be positive."""
        masses = FermionMasses.from_eigenvalues(realistic_eigenvalues)

        for attr in ["m_e", "m_mu", "m_tau", "m_u", "m_c", "m_t", "m_d", "m_s", "m_b"]:
            assert getattr(masses, attr) >= 0

    def test_from_eigenvalues_scale_dependence(self, realistic_eigenvalues):
        """Masses should scale linearly with scale parameter."""
        masses_246 = FermionMasses.from_eigenvalues(realistic_eigenvalues, scale=246.0)
        masses_492 = FermionMasses.from_eigenvalues(realistic_eigenvalues, scale=492.0)

        # Doubling scale should double masses
        assert abs(masses_492.m_e / masses_246.m_e - 2.0) < 0.01

    def test_tau_e_ratio_computed(self, realistic_eigenvalues):
        """tau_e_ratio should be computed correctly."""
        masses = FermionMasses.from_eigenvalues(realistic_eigenvalues)

        expected = masses.m_tau / masses.m_e if masses.m_e > 0 else float('inf')
        assert abs(masses.tau_e_ratio - expected) < 1e-6

    def test_s_d_ratio_computed(self, realistic_eigenvalues):
        """s_d_ratio should be computed correctly."""
        masses = FermionMasses.from_eigenvalues(realistic_eigenvalues)

        expected = masses.m_s / masses.m_d if masses.m_d > 0 else float('inf')
        assert abs(masses.s_d_ratio - expected) < 1e-6

    def test_koide_formula(self):
        """Test Koide parameter computation."""
        # Use actual PDG-like values
        m_e, m_mu, m_tau = 0.000511, 0.1057, 1.777

        sqrt_sum = math.sqrt(m_e) + math.sqrt(m_mu) + math.sqrt(m_tau)
        mass_sum = m_e + m_mu + m_tau
        koide = mass_sum / (sqrt_sum ** 2)

        # Should be close to 2/3
        assert abs(koide - 2/3) < 0.01

    def test_compare_pdg_returns_dict(self, realistic_eigenvalues):
        """compare_pdg should return dictionary."""
        masses = FermionMasses.from_eigenvalues(realistic_eigenvalues)
        comparison = masses.compare_pdg()

        assert isinstance(comparison, dict)
        assert "m_e" in comparison
        assert "m_tau/m_e" in comparison
        assert "Q_Koide" in comparison

    def test_compare_pdg_has_structure(self, realistic_eigenvalues):
        """Each comparison entry should have expected keys."""
        masses = FermionMasses.from_eigenvalues(realistic_eigenvalues)
        comparison = masses.compare_pdg()

        for key, value in comparison.items():
            assert "experimental" in value
            assert "predicted" in value
            assert "deviation_percent" in value


# =============================================================================
# MassSpectrum Tests
# =============================================================================

class TestMassSpectrum:
    """Test MassSpectrum analysis class."""

    def test_initialization(self, config):
        """MassSpectrum should initialize correctly."""
        ms = MassSpectrum(config)

        assert ms.config == config
        assert ms.tau == 3472 / 891

    def test_tau_value(self, config):
        """GIFT tau parameter should be 3472/891."""
        ms = MassSpectrum(config)

        expected = 3472 / 891
        assert abs(ms.tau - expected) < 1e-10

    def test_extract_masses(self, mock_yukawa_result, config):
        """extract_masses should return FermionMasses."""
        ms = MassSpectrum(config)
        masses = ms.extract_masses(mock_yukawa_result)

        assert isinstance(masses, FermionMasses)

    def test_analyze_spectrum_keys(self, mock_yukawa_result, config):
        """analyze_spectrum should return expected keys."""
        ms = MassSpectrum(config)
        analysis = ms.analyze_spectrum(mock_yukawa_result)

        expected_keys = [
            "n_eigenvalues",
            "eigenvalue_range",
            "visible_sum",
            "hidden_sum",
            "computed_tau",
            "expected_tau",
            "tau_deviation",
            "max_gap_position",
            "max_gap_ratio",
            "effective_rank",
            "hierarchy_ratio",
            "generation_structure",
        ]

        for key in expected_keys:
            assert key in analysis

    def test_analyze_spectrum_n_eigenvalues(self, mock_yukawa_result, config):
        """analyze_spectrum should report 77 eigenvalues."""
        ms = MassSpectrum(config)
        analysis = ms.analyze_spectrum(mock_yukawa_result)

        assert analysis["n_eigenvalues"] == 77

    def test_analyze_spectrum_43_77_split(self, mock_yukawa_result, config):
        """visible + hidden should sum to total."""
        ms = MassSpectrum(config)
        analysis = ms.analyze_spectrum(mock_yukawa_result)

        total = mock_yukawa_result.eigenvalues.sum().item()
        split_sum = analysis["visible_sum"] + analysis["hidden_sum"]

        assert abs(total - split_sum) < 1e-4

    def test_generation_structure(self, mock_yukawa_result, config):
        """generation_structure should have expected fields."""
        ms = MassSpectrum(config)
        analysis = ms.analyze_spectrum(mock_yukawa_result)
        gen = analysis["generation_structure"]

        assert "n_bands" in gen
        assert "band_sizes" in gen
        assert "gaps_found" in gen

    def test_verify_gift_predictions_keys(self, mock_yukawa_result, config):
        """verify_gift_predictions should check expected predictions."""
        ms = MassSpectrum(config)
        masses = ms.extract_masses(mock_yukawa_result)
        checks = ms.verify_gift_predictions(masses)

        assert "tau_e_ratio" in checks
        assert "s_d_ratio" in checks
        assert "koide_q" in checks

    def test_verify_gift_predictions_returns_bool(self, mock_yukawa_result, config):
        """verify_gift_predictions values should be boolean."""
        ms = MassSpectrum(config)
        masses = ms.extract_masses(mock_yukawa_result)
        checks = ms.verify_gift_predictions(masses)

        for key, value in checks.items():
            assert isinstance(value, bool)

    def test_full_report_returns_string(self, mock_yukawa_result, config):
        """full_report should return formatted string."""
        ms = MassSpectrum(config)
        report = ms.full_report(mock_yukawa_result)

        assert isinstance(report, str)
        assert len(report) > 100  # Should be substantial

    def test_full_report_contains_sections(self, mock_yukawa_result, config):
        """full_report should contain expected sections."""
        ms = MassSpectrum(config)
        report = ms.full_report(mock_yukawa_result)

        assert "Fermion Masses" in report
        assert "Mass Ratios" in report
        assert "Spectrum Analysis" in report
        assert "PDG" in report
        assert "GIFT Predictions" in report


# =============================================================================
# Edge Cases
# =============================================================================

class TestMassSpectrumEdgeCases:
    """Test edge cases and numerical stability."""

    def test_zero_eigenvalues(self, config):
        """Handle case with many zero eigenvalues."""
        eigenvalues = torch.zeros(77)
        eigenvalues[0] = 1.0  # Only one nonzero

        yukawa = YukawaResult(
            tensor=torch.zeros(21, 21, 77),
            gram_matrix=torch.diag(eigenvalues),
            eigenvalues=eigenvalues,
            eigenvectors=torch.eye(77),
            trace=1.0,
            det=0.0
        )

        ms = MassSpectrum(config)
        masses = ms.extract_masses(yukawa)

        # Should handle without error
        assert masses.m_e >= 0

    def test_uniform_eigenvalues(self, config):
        """Handle case with no hierarchy."""
        eigenvalues = torch.ones(77)

        yukawa = YukawaResult(
            tensor=torch.zeros(21, 21, 77),
            gram_matrix=torch.eye(77),
            eigenvalues=eigenvalues,
            eigenvectors=torch.eye(77),
            trace=77.0,
            det=1.0
        )

        ms = MassSpectrum(config)
        analysis = ms.analyze_spectrum(yukawa)

        # tau should be close to 43/34
        expected_tau = 43 / 34
        assert abs(analysis["computed_tau"] - expected_tau) < 0.1

    def test_extreme_hierarchy(self, config):
        """Handle extreme mass hierarchy."""
        eigenvalues = torch.zeros(77)
        eigenvalues[0] = 1e20
        eigenvalues[76] = 1e-20

        yukawa = YukawaResult(
            tensor=torch.zeros(21, 21, 77),
            gram_matrix=torch.diag(eigenvalues),
            eigenvalues=eigenvalues,
            eigenvectors=torch.eye(77),
            trace=eigenvalues.sum().item(),
            det=0.0
        )

        ms = MassSpectrum(config)
        analysis = ms.analyze_spectrum(yukawa)

        # Should handle without NaN/Inf
        assert not np.isnan(analysis["hierarchy_ratio"])


# =============================================================================
# GIFT-Specific Tests
# =============================================================================

class TestGIFTSpecific:
    """Test GIFT-specific predictions and constraints."""

    def test_tau_parameter_target(self, config):
        """GIFT tau = 3472/891 should be approximately 3.897."""
        ms = MassSpectrum(config)
        assert abs(ms.tau - 3.897) < 0.001

    def test_koide_target(self):
        """Koide Q should be close to 2/3 for SM leptons."""
        # PDG values
        m_e = PDG_2024["m_e"]
        m_mu = PDG_2024["m_mu"]
        m_tau = PDG_2024["m_tau"]

        sqrt_sum = math.sqrt(m_e) + math.sqrt(m_mu) + math.sqrt(m_tau)
        mass_sum = m_e + m_mu + m_tau
        koide = mass_sum / (sqrt_sum ** 2)

        # Empirically close to 2/3
        assert abs(koide - 2/3) < 0.001

    def test_tau_e_ratio_target(self):
        """m_tau/m_e should be close to 3477."""
        ratio = PDG_2024["m_tau_m_e"]
        assert abs(ratio - 3477) < 1  # Within 1

    def test_s_d_ratio_target(self):
        """m_s/m_d should be close to 20."""
        ratio = PDG_2024["m_s_m_d"]
        assert abs(ratio - 20) < 1  # Within 1


@pytest.mark.slow
class TestMassSpectrumLargeScale:
    """Large scale mass spectrum tests."""

    def test_many_analyses(self, config):
        """Test stability over many analyses."""
        ms = MassSpectrum(config)

        for _ in range(50):
            # Random eigenvalues
            eigenvalues = torch.rand(77)
            eigenvalues, _ = torch.sort(eigenvalues, descending=True)

            yukawa = YukawaResult(
                tensor=torch.zeros(21, 21, 77),
                gram_matrix=torch.diag(eigenvalues),
                eigenvalues=eigenvalues,
                eigenvectors=torch.eye(77),
                trace=eigenvalues.sum().item(),
                det=eigenvalues.prod().item() ** (1/77)
            )

            analysis = ms.analyze_spectrum(yukawa)
            assert not np.isnan(analysis["computed_tau"])
