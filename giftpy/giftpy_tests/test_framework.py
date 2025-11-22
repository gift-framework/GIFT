"""Tests for GIFT framework core."""
import pytest
import pandas as pd
from giftpy import GIFT
from giftpy.core.constants import CONSTANTS


class TestGIFTFramework:
    """Test main GIFT framework class."""

    @pytest.fixture
    def gift(self):
        """Create GIFT instance for testing."""
        return GIFT()

    def test_initialization(self):
        """Test GIFT initialization."""
        gift = GIFT()
        assert gift.constants == CONSTANTS
        assert gift._gauge is None  # Lazy loading

    def test_gauge_sector_access(self, gift):
        """Test accessing gauge sector."""
        gauge = gift.gauge
        assert gauge is not None
        assert hasattr(gauge, "alpha_s")
        assert hasattr(gauge, "sin2theta_W")
        assert hasattr(gauge, "alpha_inv")

    def test_lepton_sector_access(self, gift):
        """Test accessing lepton sector."""
        lepton = gift.lepton
        assert lepton is not None
        assert hasattr(lepton, "m_mu_m_e")
        assert hasattr(lepton, "Q_Koide")

    def test_neutrino_sector_access(self, gift):
        """Test accessing neutrino sector."""
        neutrino = gift.neutrino
        assert neutrino is not None
        assert hasattr(neutrino, "theta_12")
        assert hasattr(neutrino, "delta_CP")

    def test_quark_sector_access(self, gift):
        """Test accessing quark sector."""
        quark = gift.quark
        assert quark is not None
        assert hasattr(quark, "m_s_m_d")

    def test_cosmology_sector_access(self, gift):
        """Test accessing cosmology sector."""
        cosmo = gift.cosmology
        assert cosmo is not None
        assert hasattr(cosmo, "Omega_DE")

    def test_compute_all(self, gift):
        """Test computing all observables."""
        results = gift.compute_all()

        # Check it's a DataFrame
        assert isinstance(results, pd.DataFrame)

        # Check required columns
        assert "observable" in results.columns
        assert "value" in results.columns
        assert "sector" in results.columns
        assert "deviation_%" in results.columns

        # Check we have observables
        assert len(results) > 0

        # Check all sectors represented
        sectors = results["sector"].unique()
        assert "gauge" in sectors
        assert "lepton" in sectors

    def test_validation(self, gift):
        """Test validation system."""
        validation = gift.validate(verbose=False)

        # Check validation result attributes
        assert hasattr(validation, "mean_deviation")
        assert hasattr(validation, "n_observables")
        assert hasattr(validation, "all_under_1_percent")

        # Check observables count
        assert validation.n_observables > 0

        # Check summary generation
        summary = validation.summary()
        assert isinstance(summary, str)
        assert "GIFT" in summary

    def test_export_csv(self, gift, tmp_path):
        """Test CSV export."""
        output_file = tmp_path / "predictions.csv"
        gift.export(str(output_file), format="csv")
        assert output_file.exists()

        # Verify CSV can be read
        import pandas as pd

        df = pd.read_csv(output_file)
        assert len(df) > 0
        assert "observable" in df.columns

    def test_export_json(self, gift, tmp_path):
        """Test JSON export."""
        output_file = tmp_path / "predictions.json"
        gift.export(str(output_file), format="json")
        assert output_file.exists()

    def test_compare_instances(self):
        """Test comparing two GIFT instances."""
        gift1 = GIFT()
        gift2 = GIFT()

        diff = gift1.compare(gift2)

        # Same constants should give zero differences
        assert all(diff["difference"].abs() < 1e-10)

    def test_cache_clearing(self, gift):
        """Test cache clearing."""
        # Compute once (fills cache)
        results1 = gift.compute_all()

        # Clear cache
        gift.clear_cache()

        # Compute again
        results2 = gift.compute_all()

        # Results should be identical
        pd.testing.assert_frame_equal(results1, results2)

    def test_info(self, gift):
        """Test info() method."""
        info = gift.info()
        assert isinstance(info, str)
        assert "GIFT Framework" in info
        assert "β₀" in info

    def test_repr(self, gift):
        """Test string representation."""
        repr_str = repr(gift)
        assert "GIFT" in repr_str

    def test_version(self, gift):
        """Test version access."""
        version = gift.version
        assert isinstance(version, str)
        assert len(version) > 0


def test_reproducibility():
    """Test that results are reproducible."""
    gift1 = GIFT()
    gift2 = GIFT()

    # Same inputs should give identical outputs
    assert gift1.gauge.alpha_s() == gift2.gauge.alpha_s()
    assert gift1.lepton.Q_Koide() == gift2.lepton.Q_Koide()

    # DataFrames should be identical
    df1 = gift1.compute_all()
    df2 = gift2.compute_all()
    pd.testing.assert_frame_equal(df1, df2)
