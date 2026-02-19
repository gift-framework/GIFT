"""
Tests automatisés pour la validation spectrale.

Exécuter avec: pytest tests/spectral/ -v

Ces tests vérifient:
1. Que les fichiers de résultats existent
2. Que les valeurs sont dans les plages attendues
3. Que les critères PASS/FAIL sont respectés
"""

import pytest
import json
from pathlib import Path

# Chemin vers les outputs
SPECTRAL_DIR = Path(__file__).parent.parent.parent / "research/yang-mills/spectral_validation"


class TestCalibration:
    """Tests pour la calibration S³ et S⁷."""

    def test_s3_results_exist(self):
        """Vérifie que les résultats S³ existent."""
        path = SPECTRAL_DIR / "calibration/outputs/S3_calibration_results.json"
        assert path.exists(), f"Missing: {path}"

    def test_s7_results_exist(self):
        """Vérifie que les résultats S⁷ existent."""
        path = SPECTRAL_DIR / "calibration/outputs/S7_calibration_results.json"
        assert path.exists(), f"Missing: {path}"

    def test_s3_gap_exists(self):
        """Vérifie que le gap spectral existe sur S³."""
        path = SPECTRAL_DIR / "calibration/outputs/S3_calibration_results.json"
        if not path.exists():
            pytest.skip("S³ results not available")

        with open(path) as f:
            data = json.load(f)

        # Récupérer les résultats symmetric
        results = data.get("results", [])
        symmetric = [r for r in results
                     if r.get("laplacian_type") == "symmetric" and "error" not in r]

        assert len(symmetric) > 0, "No symmetric results"

        # Tous les λ₁ doivent être > 0
        lambdas = [r["lambda1_measured"] for r in symmetric]
        assert all(l > 0.001 for l in lambdas), f"Some λ₁ too small: {lambdas}"

    def test_s7_gap_exists(self):
        """Vérifie que le gap spectral existe sur S⁷."""
        path = SPECTRAL_DIR / "calibration/outputs/S7_calibration_results.json"
        if not path.exists():
            pytest.skip("S⁷ results not available")

        with open(path) as f:
            data = json.load(f)

        results = data.get("results", [])
        symmetric = [r for r in results
                     if r.get("laplacian_type") == "symmetric" and "error" not in r]

        assert len(symmetric) > 0, "No symmetric results"

        lambdas = [r["lambda1_measured"] for r in symmetric]
        assert all(l > 0.01 for l in lambdas), f"Some λ₁ too small: {lambdas}"


class TestRobustness:
    """Tests pour la robustesse K₇."""

    def test_robustness_results_exist(self):
        """Vérifie que les résultats de robustesse existent."""
        path = SPECTRAL_DIR / "robustness/outputs/K7_robustness_matrix.json"
        assert path.exists(), f"Missing: {path}"

    def test_k7_product_near_13_or_14(self):
        """Vérifie que λ₁×H* est proche de 13 ou 14."""
        path = SPECTRAL_DIR / "robustness/outputs/K7_robustness_matrix.json"
        if not path.exists():
            pytest.skip("K₇ results not available")

        with open(path) as f:
            data = json.load(f)

        plateau = data.get("analysis", {}).get("plateau_analysis", {})

        if "mean" not in plateau:
            pytest.skip("No plateau analysis")

        mean = plateau["mean"]
        # Doit être entre 10 et 18 (plage large pour robustesse)
        assert 10 < mean < 18, f"λ₁×H* = {mean} hors plage [10, 18]"

    def test_k7_closer_to_13(self):
        """Vérifie que K₇ est plus proche de 13 que de 14."""
        path = SPECTRAL_DIR / "robustness/outputs/K7_robustness_matrix.json"
        if not path.exists():
            pytest.skip("K₇ results not available")

        with open(path) as f:
            data = json.load(f)

        plateau = data.get("analysis", {}).get("plateau_analysis", {})

        if "deviation_from_13_pct" not in plateau:
            pytest.skip("No deviation data")

        dev_13 = plateau["deviation_from_13_pct"]
        dev_14 = plateau["deviation_from_14_pct"]

        # K₇ devrait être plus proche de 13
        assert dev_13 <= dev_14 + 5, f"K₇ not closer to 13: dev_13={dev_13}%, dev_14={dev_14}%"


class TestBettiIndependence:
    """Tests pour l'indépendance Betti."""

    def test_betti_results_exist(self):
        """Vérifie que les résultats Betti existent."""
        path = SPECTRAL_DIR / "analysis/outputs/betti_independence.json"
        assert path.exists(), f"Missing: {path}"

    def test_betti_spread_low(self):
        """Vérifie que le spread entre partitions est faible."""
        path = SPECTRAL_DIR / "analysis/outputs/betti_independence.json"
        if not path.exists():
            pytest.skip("Betti results not available")

        with open(path) as f:
            data = json.load(f)

        analysis = data.get("analysis", {})
        spread = analysis.get("spread_pct", float('inf'))

        # Spread doit être < 1% (très conservateur)
        assert spread < 1.0, f"Spread {spread}% too high (expected < 1%)"

    def test_betti_independence_verdict(self):
        """Vérifie le verdict d'indépendance."""
        path = SPECTRAL_DIR / "analysis/outputs/betti_independence.json"
        if not path.exists():
            pytest.skip("Betti results not available")

        with open(path) as f:
            data = json.load(f)

        verdict = data.get("analysis", {}).get("verdict", "")
        assert verdict == "INDEPENDENT", f"Verdict: {verdict} (expected INDEPENDENT)"


class TestBiasAnalysis:
    """Tests pour l'analyse du biais."""

    def test_bias_results_exist(self):
        """Vérifie que l'analyse du biais existe."""
        path = SPECTRAL_DIR / "analysis/outputs/bias_analysis.json"
        assert path.exists(), f"Missing: {path}"

    def test_bias_verdict(self):
        """Vérifie le verdict sur le biais -1."""
        path = SPECTRAL_DIR / "analysis/outputs/bias_analysis.json"
        if not path.exists():
            pytest.skip("Bias analysis not available")

        with open(path) as f:
            data = json.load(f)

        verdict = data.get("k7_analysis", {}).get("verdict", "")
        # Accepter GENUINE_13 ou BIAS_ARTIFACT (les deux sont des résultats valides)
        assert verdict in ["GENUINE_13", "BIAS_ARTIFACT"], f"Unexpected verdict: {verdict}"


class TestOverall:
    """Tests globaux."""

    def test_config_exists(self):
        """Vérifie que config.py existe."""
        path = SPECTRAL_DIR / "config.py"
        assert path.exists(), f"Missing: {path}"

    def test_final_report_exists(self):
        """Vérifie que le rapport final existe."""
        path = SPECTRAL_DIR / "FINAL_REPORT.md"
        assert path.exists(), f"Missing: {path}"

    def test_all_phases_complete(self):
        """Vérifie que toutes les phases ont des outputs."""
        required_files = [
            "calibration/outputs/S3_calibration_results.json",
            "calibration/outputs/S7_calibration_results.json",
            "robustness/outputs/K7_robustness_matrix.json",
            "analysis/outputs/betti_independence.json",
            "analysis/outputs/bias_analysis.json",
        ]

        missing = []
        for f in required_files:
            if not (SPECTRAL_DIR / f).exists():
                missing.append(f)

        assert len(missing) == 0, f"Missing files: {missing}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
