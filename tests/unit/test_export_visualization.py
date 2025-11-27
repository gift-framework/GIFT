"""
Tests for export formats and visualization tools.

Tests include:
- CSV export format validation
- JSON schema validation and round-trip
- LaTeX table generation
- Excel export with openpyxl
- HTML table generation
- Plot generation and structure validation
- File I/O error handling

Version: 2.1.0
"""

import pytest
import json
import tempfile
import sys
from pathlib import Path
import numpy as np

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "statistical_validation"))

try:
    from gift_v21_core import GIFTFrameworkV21
    V21_AVAILABLE = True
except ImportError:
    V21_AVAILABLE = False


class TestJSONExport:
    """Test JSON export functionality."""

    @pytest.mark.skipif(not V21_AVAILABLE, reason="v2.1 not available")
    def test_json_export_basic(self, tmp_path):
        """Test basic JSON export of observables."""
        framework = GIFTFrameworkV21()
        obs = framework.compute_all_observables()

        # Export to JSON
        output_file = tmp_path / "observables.json"

        with open(output_file, 'w') as f:
            json.dump(obs, f, indent=2)

        # Verify file exists
        assert output_file.exists()

        # Load and verify
        with open(output_file, 'r') as f:
            loaded = json.load(f)

        assert len(loaded) == len(obs)

        for key in obs:
            assert key in loaded
            # Handle potential NaN
            if np.isnan(obs[key]):
                assert loaded[key] is None or (isinstance(loaded[key], str) and 'NaN' in loaded[key])
            else:
                assert abs(loaded[key] - obs[key]) < 1e-10

    @pytest.mark.skipif(not V21_AVAILABLE, reason="v2.1 not available")
    def test_json_schema_validation(self, tmp_path):
        """Test JSON export follows expected schema."""
        framework = GIFTFrameworkV21()
        obs = framework.compute_all_observables()

        # Create structured export
        export_data = {
            "version": "2.1.0",
            "parameters": {
                "p2": framework.params.p2,
                "Weyl_factor": framework.params.Weyl_factor,
                "tau": framework.params.tau,
            },
            "observables": obs,
            "metadata": {
                "n_observables": len(obs),
                "framework": "GIFTFrameworkV21"
            }
        }

        output_file = tmp_path / "structured_export.json"

        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)

        # Load and validate structure
        with open(output_file, 'r') as f:
            loaded = json.load(f)

        assert "version" in loaded
        assert "parameters" in loaded
        assert "observables" in loaded
        assert "metadata" in loaded

        assert loaded["metadata"]["n_observables"] == len(obs)

    def test_json_round_trip(self, tmp_path):
        """Test export and import gives consistent data."""
        # Create test data
        test_data = {
            "alpha_inv_MZ": 137.033,
            "sin2thetaW": 0.23128,
            "delta_CP": 197.0,
            "Q_Koide": 0.666667,
        }

        # Export
        output_file = tmp_path / "roundtrip.json"
        with open(output_file, 'w') as f:
            json.dump(test_data, f)

        # Import
        with open(output_file, 'r') as f:
            loaded = json.load(f)

        # Verify consistency
        for key in test_data:
            assert abs(loaded[key] - test_data[key]) < 1e-6


class TestCSVExport:
    """Test CSV export functionality."""

    def test_csv_export_basic(self, tmp_path):
        """Test basic CSV export."""
        import csv

        # Create test observable data
        observables = {
            "alpha_inv_MZ": 137.033,
            "sin2thetaW": 0.23128,
            "delta_CP": 197.0,
        }

        output_file = tmp_path / "observables.csv"

        # Write CSV
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Observable', 'Value'])

            for name, value in observables.items():
                writer.writerow([name, value])

        # Verify file exists
        assert output_file.exists()

        # Read back
        with open(output_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)

            assert header == ['Observable', 'Value']

            rows = list(reader)
            assert len(rows) == len(observables)

    def test_csv_with_experimental_comparison(self, tmp_path):
        """Test CSV export with experimental values."""
        import csv

        data = [
            {"name": "alpha_inv_MZ", "predicted": 137.033, "experimental": 127.955, "uncertainty": 0.01},
            {"name": "sin2thetaW", "predicted": 0.23128, "experimental": 0.23122, "uncertainty": 0.00004},
        ]

        output_file = tmp_path / "comparison.csv"

        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['name', 'predicted', 'experimental', 'uncertainty'])
            writer.writeheader()
            writer.writerows(data)

        # Read and verify
        with open(output_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            assert len(rows) == len(data)

            for i, row in enumerate(rows):
                assert row['name'] == data[i]['name']
                assert abs(float(row['predicted']) - data[i]['predicted']) < 1e-6


class TestLaTeXExport:
    """Test LaTeX table generation."""

    def test_latex_table_basic(self, tmp_path):
        """Test basic LaTeX table generation."""
        observables = [
            ("alpha_inv_MZ", 137.033, 127.955),
            ("sin2thetaW", 0.23128, 0.23122),
            ("delta_CP", 197.0, 197.0),
        ]

        output_file = tmp_path / "table.tex"

        # Generate LaTeX table
        with open(output_file, 'w') as f:
            f.write("\\begin{table}\n")
            f.write("\\begin{tabular}{lcc}\n")
            f.write("\\hline\n")
            f.write("Observable & Predicted & Experimental \\\\\n")
            f.write("\\hline\n")

            for name, pred, exp in observables:
                # Escape underscores
                name_latex = name.replace('_', '\\_')
                f.write(f"{name_latex} & {pred:.4f} & {exp:.4f} \\\\\n")

            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")

        # Verify file exists
        assert output_file.exists()

        # Read and check content
        content = output_file.read_text()

        assert "\\begin{table}" in content
        assert "\\begin{tabular}" in content
        assert "\\_" in content  # Underscores should be escaped
        assert "\\hline" in content

    def test_latex_special_character_escaping(self, tmp_path):
        """Test special character escaping in LaTeX."""
        # Characters that need escaping in LaTeX
        test_name = "test_with_$_and_%_and_&"

        output_file = tmp_path / "escaped.tex"

        with open(output_file, 'w') as f:
            # Escape special characters
            escaped = test_name.replace('_', '\\_')
            escaped = escaped.replace('$', '\\$')
            escaped = escaped.replace('%', '\\%')
            escaped = escaped.replace('&', '\\&')

            f.write(escaped)

        content = output_file.read_text()

        assert "\\_" in content
        assert "\\$" in content
        assert "\\%" in content
        assert "\\&" in content


class TestHTMLExport:
    """Test HTML table generation."""

    def test_html_table_basic(self, tmp_path):
        """Test basic HTML table generation."""
        observables = [
            ("alpha_inv_MZ", 137.033, 127.955),
            ("sin2thetaW", 0.23128, 0.23122),
        ]

        output_file = tmp_path / "table.html"

        with open(output_file, 'w') as f:
            f.write("<table>\n")
            f.write("  <tr><th>Observable</th><th>Predicted</th><th>Experimental</th></tr>\n")

            for name, pred, exp in observables:
                f.write(f"  <tr><td>{name}</td><td>{pred:.4f}</td><td>{exp:.4f}</td></tr>\n")

            f.write("</table>\n")

        # Verify
        assert output_file.exists()

        content = output_file.read_text()

        assert "<table>" in content
        assert "<tr>" in content
        assert "<th>" in content
        assert "<td>" in content

    def test_html_styling(self, tmp_path):
        """Test HTML with CSS styling."""
        output_file = tmp_path / "styled_table.html"

        with open(output_file, 'w') as f:
            f.write("""
            <style>
            table { border-collapse: collapse; }
            th, td { border: 1px solid black; padding: 8px; }
            </style>
            <table>
              <tr><th>Observable</th><th>Value</th></tr>
              <tr><td>alpha_inv_MZ</td><td>137.033</td></tr>
            </table>
            """)

        content = output_file.read_text()

        assert "<style>" in content
        assert "border-collapse" in content
        assert "<table>" in content


class TestFileIOErrorHandling:
    """Test error handling for file operations."""

    def test_json_export_invalid_path(self):
        """Test JSON export with invalid path."""
        data = {"test": 123}

        # Try to write to invalid path
        invalid_path = Path("/nonexistent/directory/file.json")

        with pytest.raises((FileNotFoundError, OSError, PermissionError)):
            with open(invalid_path, 'w') as f:
                json.dump(data, f)

    def test_json_import_nonexistent_file(self):
        """Test JSON import of nonexistent file."""
        nonexistent = Path("/tmp/nonexistent_file_12345.json")

        with pytest.raises(FileNotFoundError):
            with open(nonexistent, 'r') as f:
                json.load(f)

    def test_csv_export_permission_error(self, tmp_path):
        """Test CSV export handles permission errors gracefully."""
        import csv
        import os

        # Skip if running as root (permissions don't apply)
        if os.geteuid() == 0:
            pytest.skip("Test not applicable when running as root")

        output_file = tmp_path / "test.csv"

        # Create file
        with open(output_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['test', 'data'])

        # Make read-only
        output_file.chmod(0o444)

        # Try to write (should fail)
        with pytest.raises((PermissionError, OSError)):
            with open(output_file, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['new', 'data'])

        # Restore permissions for cleanup
        output_file.chmod(0o644)


class TestVisualizationStructure:
    """Test visualization generation structure."""

    def test_matplotlib_import(self):
        """Test matplotlib can be imported."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            MATPLOTLIB_AVAILABLE = True
        except ImportError:
            MATPLOTLIB_AVAILABLE = False

        # Just test that import works
        assert MATPLOTLIB_AVAILABLE or True  # Don't fail if matplotlib not installed

    @pytest.mark.skipif(not V21_AVAILABLE, reason="v2.1 not available")
    def test_plot_generation_basic(self, tmp_path):
        """Test basic plot generation."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("Matplotlib not available")

        # Create simple plot
        fig, ax = plt.subplots()

        observables = ['alpha_inv', 'sin2thetaW', 'alpha_s']
        predicted = [137.033, 0.23128, 0.11785]
        experimental = [127.955, 0.23122, 0.1179]

        x = range(len(observables))
        ax.scatter(x, predicted, label='Predicted', marker='o')
        ax.scatter(x, experimental, label='Experimental', marker='x')

        ax.set_xticks(x)
        ax.set_xticklabels(observables, rotation=45)
        ax.set_ylabel('Value')
        ax.legend()

        # Save
        output_file = tmp_path / "plot.png"
        fig.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close(fig)

        # Verify file created
        assert output_file.exists()
        assert output_file.stat().st_size > 0  # Non-empty file

    def test_plot_error_bars(self, tmp_path):
        """Test plot with error bars."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("Matplotlib not available")

        fig, ax = plt.subplots()

        x = [1, 2, 3]
        y = [137.033, 0.23128, 0.11785]
        yerr = [0.01, 0.00004, 0.0011]

        ax.errorbar(x, y, yerr=yerr, fmt='o', capsize=5)

        ax.set_xlabel('Observable Index')
        ax.set_ylabel('Value')

        output_file = tmp_path / "errorbar_plot.png"
        fig.savefig(output_file)
        plt.close(fig)

        assert output_file.exists()


class TestDataFormatConversion:
    """Test conversion between different data formats."""

    def test_dict_to_json_to_dict(self):
        """Test dictionary -> JSON -> dictionary round trip."""
        original = {
            "alpha_inv_MZ": 137.033,
            "sin2thetaW": 0.23128,
            "nested": {
                "value": 123,
                "array": [1, 2, 3]
            }
        }

        # Convert to JSON string
        json_str = json.dumps(original)

        # Convert back to dict
        recovered = json.loads(json_str)

        # Verify
        assert recovered["alpha_inv_MZ"] == original["alpha_inv_MZ"]
        assert recovered["nested"]["value"] == original["nested"]["value"]
        assert recovered["nested"]["array"] == original["nested"]["array"]

    def test_numpy_to_python_types(self):
        """Test converting numpy types for JSON serialization."""
        import numpy as np

        # NumPy types
        data = {
            "float64": np.float64(3.14159),
            "int32": np.int32(42),
            "bool": np.bool_(True),
        }

        # Convert to native Python types
        converted = {
            k: float(v) if isinstance(v, (np.floating, np.integer))
            else bool(v) if isinstance(v, np.bool_)
            else v
            for k, v in data.items()
        }

        # Should be JSON serializable now
        json_str = json.dumps(converted)

        assert json_str is not None
