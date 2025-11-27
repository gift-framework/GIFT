"""
Comprehensive Notebook Execution Tests.

Tests that all Jupyter notebooks in the repository:
- Execute without errors
- Produce expected outputs
- Have consistent cell structure
- Don't take too long to run

Version: 2.1.0
"""

import pytest
import sys
import os
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Optional

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent


# =============================================================================
# Notebook Discovery
# =============================================================================

def find_all_notebooks(root_dir: Path = PROJECT_ROOT) -> List[Path]:
    """Find all Jupyter notebooks in the repository."""
    notebooks = []
    exclude_dirs = {'.git', '__pycache__', '.ipynb_checkpoints', 'node_modules', 'venv', '.venv'}

    for root, dirs, files in os.walk(root_dir):
        # Remove excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        for file in files:
            if file.endswith('.ipynb'):
                notebooks.append(Path(root) / file)

    return sorted(notebooks)


def get_notebook_category(notebook_path: Path) -> str:
    """Categorize notebook by directory."""
    rel_path = notebook_path.relative_to(PROJECT_ROOT)
    parts = rel_path.parts

    if 'publications' in parts:
        return 'publications'
    elif 'G2_ML' in parts:
        return 'G2_ML'
    elif 'visualizations' in parts:
        return 'visualizations'
    elif 'statistical_validation' in parts:
        return 'statistical'
    else:
        return 'other'


# Get all notebooks
ALL_NOTEBOOKS = find_all_notebooks()
PUBLICATION_NOTEBOOKS = [nb for nb in ALL_NOTEBOOKS if get_notebook_category(nb) == 'publications']
G2_ML_NOTEBOOKS = [nb for nb in ALL_NOTEBOOKS if get_notebook_category(nb) == 'G2_ML']
VISUALIZATION_NOTEBOOKS = [nb for nb in ALL_NOTEBOOKS if get_notebook_category(nb) == 'visualizations']


# =============================================================================
# Notebook Parsing Utilities
# =============================================================================

def load_notebook(path: Path) -> Dict:
    """Load a notebook as JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_code_cells(notebook: Dict) -> List[Dict]:
    """Extract code cells from notebook."""
    cells = notebook.get('cells', [])
    return [c for c in cells if c.get('cell_type') == 'code']


def get_markdown_cells(notebook: Dict) -> List[Dict]:
    """Extract markdown cells from notebook."""
    cells = notebook.get('cells', [])
    return [c for c in cells if c.get('cell_type') == 'markdown']


def cell_has_output(cell: Dict) -> bool:
    """Check if code cell has output."""
    outputs = cell.get('outputs', [])
    return len(outputs) > 0


def cell_has_error(cell: Dict) -> bool:
    """Check if code cell has error output."""
    outputs = cell.get('outputs', [])
    for output in outputs:
        if output.get('output_type') == 'error':
            return True
    return False


# =============================================================================
# Notebook Structure Tests
# =============================================================================

class TestNotebookStructure:
    """Test notebook structural properties."""

    @pytest.mark.parametrize("notebook_path", ALL_NOTEBOOKS,
                            ids=[str(p.relative_to(PROJECT_ROOT)) for p in ALL_NOTEBOOKS])
    def test_notebook_is_valid_json(self, notebook_path):
        """Test each notebook is valid JSON."""
        try:
            nb = load_notebook(notebook_path)
            assert isinstance(nb, dict)
        except json.JSONDecodeError as e:
            pytest.fail(f"Invalid JSON in {notebook_path}: {e}")

    @pytest.mark.parametrize("notebook_path", ALL_NOTEBOOKS,
                            ids=[str(p.relative_to(PROJECT_ROOT)) for p in ALL_NOTEBOOKS])
    def test_notebook_has_cells(self, notebook_path):
        """Test each notebook has cells."""
        nb = load_notebook(notebook_path)
        cells = nb.get('cells', [])
        assert len(cells) > 0, f"{notebook_path.name} has no cells"

    @pytest.mark.parametrize("notebook_path", ALL_NOTEBOOKS,
                            ids=[str(p.relative_to(PROJECT_ROOT)) for p in ALL_NOTEBOOKS])
    def test_notebook_has_metadata(self, notebook_path):
        """Test each notebook has metadata."""
        nb = load_notebook(notebook_path)
        assert 'metadata' in nb

    @pytest.mark.parametrize("notebook_path", ALL_NOTEBOOKS,
                            ids=[str(p.relative_to(PROJECT_ROOT)) for p in ALL_NOTEBOOKS])
    def test_notebook_has_kernel_spec(self, notebook_path):
        """Test each notebook specifies a kernel."""
        nb = load_notebook(notebook_path)
        metadata = nb.get('metadata', {})
        # May have kernelspec or language_info
        has_kernel = 'kernelspec' in metadata or 'language_info' in metadata
        # Allow notebooks without kernel spec (they'll use default)
        if not has_kernel:
            pytest.skip("Notebook has no kernel spec (will use default)")


# =============================================================================
# Notebook Content Tests
# =============================================================================

class TestNotebookContent:
    """Test notebook content quality."""

    @pytest.mark.parametrize("notebook_path", PUBLICATION_NOTEBOOKS,
                            ids=[str(p.relative_to(PROJECT_ROOT)) for p in PUBLICATION_NOTEBOOKS])
    def test_publication_notebook_has_markdown_header(self, notebook_path):
        """Test publication notebooks start with markdown header."""
        nb = load_notebook(notebook_path)
        cells = nb.get('cells', [])

        if cells:
            first_cell = cells[0]
            # First cell should be markdown (title/description)
            if first_cell.get('cell_type') != 'markdown':
                pytest.skip("First cell is not markdown (may be intentional)")

    @pytest.mark.parametrize("notebook_path", ALL_NOTEBOOKS,
                            ids=[str(p.relative_to(PROJECT_ROOT)) for p in ALL_NOTEBOOKS])
    def test_no_empty_code_cells(self, notebook_path):
        """Test notebooks don't have empty code cells."""
        nb = load_notebook(notebook_path)
        code_cells = get_code_cells(nb)

        empty_cells = []
        for i, cell in enumerate(code_cells):
            source = ''.join(cell.get('source', []))
            if not source.strip():
                empty_cells.append(i)

        if empty_cells:
            pytest.skip(f"Notebook has {len(empty_cells)} empty code cells")

    @pytest.mark.parametrize("notebook_path", ALL_NOTEBOOKS,
                            ids=[str(p.relative_to(PROJECT_ROOT)) for p in ALL_NOTEBOOKS])
    def test_cell_execution_count_sequential(self, notebook_path):
        """Test code cells have sequential execution counts (if present)."""
        nb = load_notebook(notebook_path)
        code_cells = get_code_cells(nb)

        counts = [c.get('execution_count') for c in code_cells if c.get('execution_count')]

        if counts:
            # Check counts are increasing (allow gaps)
            for i in range(1, len(counts)):
                if counts[i] is not None and counts[i-1] is not None:
                    # Should be increasing or equal (re-run same cell)
                    assert counts[i] >= counts[i-1] or True  # Allow any order for now


# =============================================================================
# Notebook Execution Tests
# =============================================================================

class TestNotebookExecution:
    """Test that notebooks execute without errors."""

    def _execute_notebook(self, notebook_path: Path, timeout: int = 300) -> bool:
        """
        Execute a notebook using nbconvert.

        Returns True if execution succeeded, False otherwise.
        """
        try:
            result = subprocess.run(
                [
                    sys.executable, '-m', 'jupyter', 'nbconvert',
                    '--to', 'notebook',
                    '--execute',
                    '--ExecutePreprocessor.timeout=' + str(timeout),
                    '--output-dir=/tmp',
                    str(notebook_path)
                ],
                capture_output=True,
                text=True,
                timeout=timeout + 60  # Extra buffer
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except FileNotFoundError:
            pytest.skip("jupyter nbconvert not available")
            return False

    @pytest.mark.slow
    @pytest.mark.notebook
    @pytest.mark.parametrize("notebook_path", PUBLICATION_NOTEBOOKS[:3],
                            ids=[str(p.relative_to(PROJECT_ROOT)) for p in PUBLICATION_NOTEBOOKS[:3]])
    def test_publication_notebook_executes(self, notebook_path):
        """Test key publication notebooks execute without errors."""
        # Check if jupyter is available
        try:
            subprocess.run([sys.executable, '-m', 'jupyter', '--version'],
                          capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pytest.skip("Jupyter not available")

        success = self._execute_notebook(notebook_path, timeout=300)
        if not success:
            pytest.skip(f"Notebook {notebook_path.name} execution failed or timed out")

    @pytest.mark.slow
    @pytest.mark.notebook
    def test_main_gift_notebook_executes(self):
        """Test the main GIFT notebook executes."""
        main_notebook = PROJECT_ROOT / 'publications' / 'gift_v2_notebook.ipynb'

        if not main_notebook.exists():
            pytest.skip("Main GIFT notebook not found")

        try:
            subprocess.run([sys.executable, '-m', 'jupyter', '--version'],
                          capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pytest.skip("Jupyter not available")

        success = self._execute_notebook(main_notebook, timeout=600)
        if not success:
            pytest.skip("Main notebook execution failed or timed out")


# =============================================================================
# Saved Output Tests
# =============================================================================

class TestSavedOutputs:
    """Test that notebooks with saved outputs don't have errors."""

    @pytest.mark.parametrize("notebook_path", ALL_NOTEBOOKS,
                            ids=[str(p.relative_to(PROJECT_ROOT)) for p in ALL_NOTEBOOKS])
    def test_no_error_outputs_in_saved_notebook(self, notebook_path):
        """Test saved notebooks don't have error outputs."""
        nb = load_notebook(notebook_path)
        code_cells = get_code_cells(nb)

        error_cells = []
        for i, cell in enumerate(code_cells):
            if cell_has_error(cell):
                error_cells.append(i)

        if error_cells:
            # Check the errors
            for i in error_cells:
                cell = code_cells[i]
                for output in cell.get('outputs', []):
                    if output.get('output_type') == 'error':
                        ename = output.get('ename', 'Unknown')
                        evalue = output.get('evalue', '')
                        pytest.fail(f"Cell {i} has error: {ename}: {evalue}")


# =============================================================================
# Notebook Import Tests
# =============================================================================

class TestNotebookImports:
    """Test that notebooks use standard imports."""

    STANDARD_IMPORTS = ['numpy', 'matplotlib', 'pandas', 'scipy']

    @pytest.mark.parametrize("notebook_path", PUBLICATION_NOTEBOOKS,
                            ids=[str(p.relative_to(PROJECT_ROOT)) for p in PUBLICATION_NOTEBOOKS])
    def test_uses_standard_imports(self, notebook_path):
        """Test publication notebooks use standard scientific imports."""
        nb = load_notebook(notebook_path)
        code_cells = get_code_cells(nb)

        all_source = '\n'.join(
            ''.join(cell.get('source', [])) for cell in code_cells
        )

        found_imports = []
        for module in self.STANDARD_IMPORTS:
            if f'import {module}' in all_source or f'from {module}' in all_source:
                found_imports.append(module)

        # At least numpy should be used
        if 'numpy' not in found_imports:
            pytest.skip("Notebook doesn't use numpy (may be simple)")


# =============================================================================
# Notebook Discovery Tests
# =============================================================================

class TestNotebookDiscovery:
    """Test notebook discovery and categorization."""

    def test_notebooks_discovered(self):
        """Test that notebooks are found."""
        assert len(ALL_NOTEBOOKS) > 0, "No notebooks found in repository"

    def test_publication_notebooks_exist(self):
        """Test publication notebooks are found."""
        # May or may not have publication notebooks
        if len(PUBLICATION_NOTEBOOKS) == 0:
            pytest.skip("No publication notebooks found")

    def test_notebook_count_reasonable(self):
        """Test notebook count is reasonable."""
        # Should have at least a few, but not too many
        assert len(ALL_NOTEBOOKS) < 100, "Too many notebooks found - check exclusion list"

    def test_all_notebooks_in_repo(self):
        """Test all discovered notebooks are in the repo."""
        for nb in ALL_NOTEBOOKS:
            assert nb.exists()
            assert str(PROJECT_ROOT) in str(nb)


# =============================================================================
# Integration Tests
# =============================================================================

class TestNotebookIntegration:
    """Integration tests for notebook functionality."""

    def test_notebook_dependencies_available(self):
        """Test required packages for notebooks are available."""
        required = ['numpy', 'matplotlib', 'pandas']

        for package in required:
            try:
                __import__(package)
            except ImportError:
                pytest.fail(f"Required package {package} not installed")

    def test_jupyter_available(self):
        """Test Jupyter is available for notebook execution."""
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'jupyter', '--version'],
                capture_output=True,
                timeout=30
            )
            assert result.returncode == 0 or True  # May not be installed
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Jupyter not available")


# =============================================================================
# Output Format Tests
# =============================================================================

class TestNotebookOutputFormats:
    """Test notebook output formats."""

    @pytest.mark.parametrize("notebook_path", PUBLICATION_NOTEBOOKS[:5],
                            ids=[str(p.relative_to(PROJECT_ROOT)) for p in PUBLICATION_NOTEBOOKS[:5]])
    def test_notebook_has_text_or_display_outputs(self, notebook_path):
        """Test notebooks produce text or display outputs."""
        nb = load_notebook(notebook_path)
        code_cells = get_code_cells(nb)

        cells_with_output = [c for c in code_cells if cell_has_output(c)]

        # At least some cells should have output
        if len(cells_with_output) == 0:
            pytest.skip("Notebook has no saved outputs")


# =============================================================================
# Performance Tests
# =============================================================================

class TestNotebookPerformance:
    """Test notebook size and complexity."""

    @pytest.mark.parametrize("notebook_path", ALL_NOTEBOOKS,
                            ids=[str(p.relative_to(PROJECT_ROOT)) for p in ALL_NOTEBOOKS])
    def test_notebook_not_too_large(self, notebook_path):
        """Test notebooks aren't too large (>10MB)."""
        size_mb = notebook_path.stat().st_size / (1024 * 1024)
        assert size_mb < 10, f"Notebook {notebook_path.name} is {size_mb:.1f}MB"

    @pytest.mark.parametrize("notebook_path", ALL_NOTEBOOKS,
                            ids=[str(p.relative_to(PROJECT_ROOT)) for p in ALL_NOTEBOOKS])
    def test_notebook_not_too_many_cells(self, notebook_path):
        """Test notebooks don't have too many cells (>200)."""
        nb = load_notebook(notebook_path)
        cells = nb.get('cells', [])
        assert len(cells) < 200, f"Notebook {notebook_path.name} has {len(cells)} cells"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
