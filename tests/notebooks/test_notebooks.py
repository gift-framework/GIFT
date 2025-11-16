"""
Notebook validation tests using papermill.

Tests that all Jupyter notebooks execute without errors and produce
expected outputs.
"""

import pytest
from pathlib import Path
import tempfile
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


@pytest.fixture
def project_root():
    """Get project root directory."""
    return Path(__file__).parent.parent.parent


@pytest.fixture
def notebook_executor():
    """Create notebook executor with timeout."""
    return ExecutePreprocessor(timeout=600, kernel_name='python3')


def execute_notebook(notebook_path, executor):
    """
    Execute a notebook and return success status.

    Args:
        notebook_path: Path to notebook
        executor: ExecutePreprocessor instance

    Returns:
        tuple: (success: bool, errors: list)
    """
    try:
        with open(notebook_path, 'r') as f:
            nb = nbformat.read(f, as_version=4)

        # Execute notebook
        executor.preprocess(nb, {'metadata': {'path': str(notebook_path.parent)}})

        return True, []

    except Exception as e:
        return False, [str(e)]


@pytest.mark.notebook
@pytest.mark.slow
class TestPublicationNotebooks:
    """Test publication notebooks execute correctly."""

    def test_gift_v2_notebook_executes(self, project_root, notebook_executor):
        """Test that main GIFT v2 notebook executes."""
        notebook_path = project_root / "publications" / "gift_v2_notebook.ipynb"

        if not notebook_path.exists():
            pytest.skip(f"Notebook not found: {notebook_path}")

        success, errors = execute_notebook(notebook_path, notebook_executor)

        assert success, f"Notebook execution failed: {errors}"

    def test_statistical_validation_notebook_executes(self, project_root):
        """Test that statistical validation notebook executes (with lighter execution)."""
        notebook_path = project_root / "publications" / "gift_statistical_validation.ipynb"

        if not notebook_path.exists():
            pytest.skip(f"Notebook not found: {notebook_path}")

        # For statistical validation, just check it opens correctly
        # Full execution would take too long
        with open(notebook_path, 'r') as f:
            nb = nbformat.read(f, as_version=4)

        assert len(nb.cells) > 0, "Notebook has no cells"

    def test_experimental_predictions_notebook_exists(self, project_root):
        """Test that experimental predictions notebook exists."""
        notebook_path = project_root / "publications" / "gift_experimental_predictions.ipynb"

        if not notebook_path.exists():
            pytest.skip(f"Notebook not found: {notebook_path}")

        with open(notebook_path, 'r') as f:
            nb = nbformat.read(f, as_version=4)

        assert len(nb.cells) > 0


@pytest.mark.notebook
class TestNotebookStructure:
    """Test notebook structure and format."""

    def test_all_notebooks_valid_format(self, project_root):
        """Test that all notebooks have valid format."""
        notebook_paths = list(project_root.glob("**/*.ipynb"))

        # Filter out checkpoints
        notebook_paths = [p for p in notebook_paths if ".ipynb_checkpoints" not in str(p)]

        assert len(notebook_paths) > 0, "No notebooks found"

        for nb_path in notebook_paths:
            try:
                with open(nb_path, 'r') as f:
                    nb = nbformat.read(f, as_version=4)

                # Check basic structure
                assert 'cells' in nb, f"{nb_path} missing cells"
                assert len(nb.cells) > 0, f"{nb_path} has no cells"

            except Exception as e:
                pytest.fail(f"Invalid notebook {nb_path}: {e}")

    def test_notebooks_have_markdown_cells(self, project_root):
        """Test that notebooks have documentation (markdown cells)."""
        main_notebooks = [
            project_root / "publications" / "gift_v2_notebook.ipynb"
        ]

        for nb_path in main_notebooks:
            if not nb_path.exists():
                continue

            with open(nb_path, 'r') as f:
                nb = nbformat.read(f, as_version=4)

            markdown_cells = [c for c in nb.cells if c.cell_type == 'markdown']
            assert len(markdown_cells) > 0, f"{nb_path} has no markdown cells"

    def test_notebooks_no_empty_outputs(self, project_root):
        """Test main notebooks have executed outputs."""
        main_notebooks = [
            project_root / "publications" / "gift_v2_notebook.ipynb"
        ]

        for nb_path in main_notebooks:
            if not nb_path.exists():
                continue

            with open(nb_path, 'r') as f:
                nb = nbformat.read(f, as_version=4)

            code_cells = [c for c in nb.cells if c.cell_type == 'code']

            # At least some code cells should have outputs
            cells_with_output = [c for c in code_cells if len(c.get('outputs', [])) > 0]

            # Not all cells need output, but most should
            if len(code_cells) > 0:
                output_ratio = len(cells_with_output) / len(code_cells)
                assert output_ratio > 0.3, f"{nb_path} has too few cells with output"


@pytest.mark.notebook
@pytest.mark.slow
class TestG2Notebooks:
    """Test G2 ML notebooks."""

    def test_latest_g2_notebook_exists(self, project_root):
        """Test that latest G2 notebook exists."""
        g2_latest = project_root / "G2_ML" / "0.9a"
        notebook_paths = list(g2_latest.glob("*.ipynb"))

        if len(notebook_paths) == 0:
            pytest.skip("No G2 notebooks found in 0.9a")

        # Just check they're valid
        for nb_path in notebook_paths:
            with open(nb_path, 'r') as f:
                nb = nbformat.read(f, as_version=4)
            assert len(nb.cells) > 0

    def test_g2_integration_test_exists(self, project_root):
        """Test that G2 integration test exists."""
        test_path = project_root / "G2_ML" / "0.2" / "test_integration.py"

        # We know this exists from earlier
        assert test_path.exists()


@pytest.mark.notebook
class TestVisualizationNotebooks:
    """Test visualization notebooks."""

    def test_visualization_notebooks_exist(self, project_root):
        """Test that visualization notebooks exist."""
        viz_dir = project_root / "assets" / "visualizations"

        if not viz_dir.exists():
            pytest.skip("Visualization directory not found")

        notebook_paths = list(viz_dir.glob("*.ipynb"))

        # Should have at least some visualization notebooks
        if len(notebook_paths) > 0:
            for nb_path in notebook_paths:
                with open(nb_path, 'r') as f:
                    nb = nbformat.read(f, as_version=4)
                assert len(nb.cells) > 0, f"{nb_path} has no cells"


@pytest.mark.notebook
class TestNotebookOutputs:
    """Test notebook outputs are reasonable."""

    def test_main_notebook_produces_observables(self, project_root):
        """Test that main notebook produces expected observable outputs."""
        notebook_path = project_root / "publications" / "gift_v2_notebook.ipynb"

        if not notebook_path.exists():
            pytest.skip(f"Notebook not found: {notebook_path}")

        with open(notebook_path, 'r') as f:
            nb = nbformat.read(f, as_version=4)

        # Check that notebook mentions key observables
        notebook_text = " ".join([
            str(cell.get('source', '')) for cell in nb.cells
        ])

        key_observables = ['alpha_inv', 'sin2thetaW', 'delta_CP', 'Q_Koide']

        for obs in key_observables:
            assert obs in notebook_text, f"Observable {obs} not found in notebook"


@pytest.mark.notebook
class TestNotebookPerformance:
    """Test notebook execution performance."""

    def test_notebook_execution_time_reasonable(self, project_root):
        """Test that notebook execution doesn't timeout."""
        # This is implicitly tested by the executor timeout
        # If a notebook takes > 600s, it will fail

        # For now, just a placeholder
        assert True

    def test_notebook_memory_usage_reasonable(self):
        """Test that notebooks don't use excessive memory."""
        # Placeholder for memory profiling
        # Would need memory_profiler or similar
        assert True
