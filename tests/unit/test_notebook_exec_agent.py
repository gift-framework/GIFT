"""
Unit tests for Notebook Execution Agent.

Tests notebook discovery and execution status reporting
for Jupyter notebooks in the project.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import shutil
import sys

# Add agents to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "assets"))

from agents.notebook_exec import NotebookExecutionAgent
from agents.base import AgentResult


# =============================================================================
# NotebookExecutionAgent Tests
# =============================================================================

class TestNotebookExecutionAgent:
    """Test the NotebookExecutionAgent class."""

    @pytest.fixture
    def agent(self):
        """Create agent instance."""
        return NotebookExecutionAgent()

    def test_agent_name(self, agent):
        """Test agent has correct name."""
        assert agent.name == "notebooks"

    def test_run_returns_agent_result(self, agent, tmp_path):
        """Test that run returns AgentResult."""
        result = agent.run(tmp_path)

        assert isinstance(result, AgentResult)

    def test_run_with_no_notebooks(self, agent, tmp_path):
        """Test run when no notebooks exist."""
        result = agent.run(tmp_path)

        assert result.ok is True
        assert "No notebooks" in result.summary

    def test_run_with_notebooks_deps_missing(self, agent, tmp_path):
        """Test that when deps are missing, info is reported."""
        pubs_dir = tmp_path / "publications"
        pubs_dir.mkdir()

        # Create a minimal notebook file
        notebook_content = '{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}'
        (pubs_dir / "test_notebook.ipynb").write_text(notebook_content)

        with patch("shutil.which", return_value=None):
            result = agent.run(tmp_path)

        assert result.ok is True
        # When deps missing, issues contains info about skipping
        assert len(result.issues) > 0
        assert "info" in result.issues[0]

    def test_run_with_notebooks_papermill_available(self, agent, tmp_path):
        """Test that notebooks are listed when papermill is available."""
        viz_dir = tmp_path / "assets" / "visualizations"
        viz_dir.mkdir(parents=True)

        notebook_content = '{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}'
        (viz_dir / "viz_notebook.ipynb").write_text(notebook_content)

        with patch("shutil.which") as mock_which:
            mock_which.side_effect = lambda x: "/usr/bin/papermill" if x == "papermill" else None
            result = agent.run(tmp_path)

        assert result.ok is True
        # When deps available, notebooks are listed
        assert any("viz_notebook" in str(info) for info in result.issues)

    def test_run_without_papermill_or_jupyter(self, agent, tmp_path):
        """Test run when neither papermill nor jupyter is installed."""
        pubs_dir = tmp_path / "publications"
        pubs_dir.mkdir()

        notebook_content = '{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}'
        (pubs_dir / "test.ipynb").write_text(notebook_content)

        with patch("shutil.which", return_value=None):
            result = agent.run(tmp_path)

        assert result.ok is True
        # Should skip execution
        assert "dependencies" in result.summary.lower() or "skipping" in result.summary.lower()

    def test_run_with_papermill_available(self, agent, tmp_path):
        """Test run when papermill is available."""
        pubs_dir = tmp_path / "publications"
        pubs_dir.mkdir()

        notebook_content = '{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}'
        (pubs_dir / "test.ipynb").write_text(notebook_content)

        with patch("shutil.which") as mock_which:
            mock_which.side_effect = lambda x: "/usr/bin/papermill" if x == "papermill" else None
            result = agent.run(tmp_path)

        assert result.ok is True

    def test_run_with_jupyter_available(self, agent, tmp_path):
        """Test run when jupyter is available."""
        pubs_dir = tmp_path / "publications"
        pubs_dir.mkdir()

        notebook_content = '{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}'
        (pubs_dir / "test.ipynb").write_text(notebook_content)

        with patch("shutil.which") as mock_which:
            mock_which.side_effect = lambda x: "/usr/bin/jupyter" if x == "jupyter" else None
            result = agent.run(tmp_path)

        assert result.ok is True

    def test_run_reports_notebook_count_with_deps(self, agent, tmp_path):
        """Test that summary reports notebook count when deps available."""
        pubs_dir = tmp_path / "publications"
        pubs_dir.mkdir()

        notebook_content = '{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}'
        (pubs_dir / "nb1.ipynb").write_text(notebook_content)
        (pubs_dir / "nb2.ipynb").write_text(notebook_content)
        (pubs_dir / "nb3.ipynb").write_text(notebook_content)

        with patch("shutil.which") as mock_which:
            mock_which.side_effect = lambda x: "/usr/bin/papermill" if x == "papermill" else None
            result = agent.run(tmp_path)

        assert "3" in result.summary

    def test_run_issues_contain_notebook_paths_with_deps(self, agent, tmp_path):
        """Test that issues contain notebook paths when deps available."""
        pubs_dir = tmp_path / "publications"
        pubs_dir.mkdir()

        notebook_content = '{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}'
        (pubs_dir / "my_notebook.ipynb").write_text(notebook_content)

        with patch("shutil.which") as mock_which:
            mock_which.side_effect = lambda x: "/usr/bin/papermill" if x == "papermill" else None
            result = agent.run(tmp_path)

        # Check that notebook path is in issues when deps are available
        notebook_paths = [info.get("notebook", "") for info in result.issues if "notebook" in info]
        assert any("my_notebook.ipynb" in path for path in notebook_paths)

    def test_run_with_nested_notebooks(self, agent, tmp_path):
        """Test finding notebooks in nested directories."""
        nested_dir = tmp_path / "publications" / "examples" / "tutorials"
        nested_dir.mkdir(parents=True)

        notebook_content = '{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}'
        (nested_dir / "tutorial.ipynb").write_text(notebook_content)

        with patch("shutil.which", return_value=None):
            result = agent.run(tmp_path)

        # Note: depends on glob patterns used - check if nested is included

    def test_run_ignores_non_ipynb_files(self, agent, tmp_path):
        """Test that non-.ipynb files are ignored."""
        pubs_dir = tmp_path / "publications"
        pubs_dir.mkdir()

        (pubs_dir / "not_a_notebook.txt").write_text("text file")
        (pubs_dir / "script.py").write_text("python file")

        result = agent.run(tmp_path)

        assert result.ok is True
        assert "No notebooks" in result.summary


# =============================================================================
# Dependency Detection Tests
# =============================================================================

class TestDependencyDetection:
    """Test dependency detection for execution."""

    @pytest.fixture
    def agent(self):
        return NotebookExecutionAgent()

    def test_papermill_detection(self, agent, tmp_path):
        """Test papermill availability detection."""
        pubs_dir = tmp_path / "publications"
        pubs_dir.mkdir()
        (pubs_dir / "nb.ipynb").write_text('{"cells": []}')

        with patch("shutil.which") as mock_which:
            mock_which.side_effect = lambda x: "/path/to/papermill" if x == "papermill" else None
            result = agent.run(tmp_path)

        assert result.ok is True

    def test_jupyter_detection(self, agent, tmp_path):
        """Test jupyter availability detection."""
        pubs_dir = tmp_path / "publications"
        pubs_dir.mkdir()
        (pubs_dir / "nb.ipynb").write_text('{"cells": []}')

        with patch("shutil.which") as mock_which:
            mock_which.side_effect = lambda x: "/path/to/jupyter" if x == "jupyter" else None
            result = agent.run(tmp_path)

        assert result.ok is True

    def test_no_execution_tools(self, agent, tmp_path):
        """Test behavior when no execution tools available."""
        pubs_dir = tmp_path / "publications"
        pubs_dir.mkdir()
        (pubs_dir / "nb.ipynb").write_text('{"cells": []}')

        with patch("shutil.which", return_value=None):
            result = agent.run(tmp_path)

        assert result.ok is True
        # Should indicate skipping
        assert any("info" in str(info) for info in result.issues) or "skip" in result.summary.lower()


# =============================================================================
# Edge Cases
# =============================================================================

class TestNotebookExecutionEdgeCases:
    """Test edge cases for notebook execution agent."""

    @pytest.fixture
    def agent(self):
        return NotebookExecutionAgent()

    def test_empty_publications_directory(self, agent, tmp_path):
        """Test with empty publications directory."""
        (tmp_path / "publications").mkdir()

        result = agent.run(tmp_path)

        assert result.ok is True
        assert "No notebooks" in result.summary

    def test_empty_visualizations_directory(self, agent, tmp_path):
        """Test with empty visualizations directory."""
        (tmp_path / "assets" / "visualizations").mkdir(parents=True)

        result = agent.run(tmp_path)

        assert result.ok is True

    def test_notebook_with_special_characters_in_name(self, agent, tmp_path):
        """Test notebook with special characters in filename."""
        pubs_dir = tmp_path / "publications"
        pubs_dir.mkdir()

        notebook_content = '{"cells": [], "metadata": {}, "nbformat": 4}'
        (pubs_dir / "my notebook (v2).ipynb").write_text(notebook_content)

        with patch("shutil.which", return_value=None):
            result = agent.run(tmp_path)

        assert result.ok is True

    def test_many_notebooks(self, agent, tmp_path):
        """Test with many notebooks when deps available."""
        pubs_dir = tmp_path / "publications"
        pubs_dir.mkdir()

        notebook_content = '{"cells": []}'
        for i in range(20):
            (pubs_dir / f"notebook_{i}.ipynb").write_text(notebook_content)

        with patch("shutil.which") as mock_which:
            mock_which.side_effect = lambda x: "/usr/bin/papermill" if x == "papermill" else None
            result = agent.run(tmp_path)

        assert result.ok is True
        assert "20" in result.summary


# =============================================================================
# Integration Tests
# =============================================================================

class TestNotebookExecutionIntegration:
    """Integration tests for notebook execution agent."""

    @pytest.fixture
    def agent(self):
        return NotebookExecutionAgent()

    def test_realistic_project_structure(self, agent, tmp_path):
        """Test with realistic project structure when deps available."""
        # Create GIFT-like structure
        pubs_dir = tmp_path / "publications"
        viz_dir = tmp_path / "assets" / "visualizations"
        pubs_dir.mkdir()
        viz_dir.mkdir(parents=True)

        notebook_content = '{"cells": [], "metadata": {"kernelspec": {"name": "python3"}}}'

        # Publications notebooks
        (pubs_dir / "gift_v2_notebook.ipynb").write_text(notebook_content)
        (pubs_dir / "statistical_validation.ipynb").write_text(notebook_content)

        # Visualization notebooks
        (viz_dir / "e8_root_system_3d.ipynb").write_text(notebook_content)
        (viz_dir / "precision_dashboard.ipynb").write_text(notebook_content)

        with patch("shutil.which") as mock_which:
            mock_which.side_effect = lambda x: "/usr/bin/papermill" if x == "papermill" else None
            result = agent.run(tmp_path)

        assert result.ok is True
        assert "4" in result.summary  # Should find 4 notebooks

    def test_mixed_file_types(self, agent, tmp_path):
        """Test with mixed file types in directories when deps available."""
        pubs_dir = tmp_path / "publications"
        pubs_dir.mkdir()

        notebook_content = '{"cells": []}'
        (pubs_dir / "notebook.ipynb").write_text(notebook_content)
        (pubs_dir / "paper.md").write_text("# Paper")
        (pubs_dir / "data.csv").write_text("a,b,c")
        (pubs_dir / "script.py").write_text("print('hello')")

        with patch("shutil.which") as mock_which:
            mock_which.side_effect = lambda x: "/usr/bin/papermill" if x == "papermill" else None
            result = agent.run(tmp_path)

        # Should only report the notebook
        assert result.ok is True
        notebook_paths = [info.get("notebook", "") for info in result.issues if "notebook" in info]
        assert len(notebook_paths) == 1
        assert "notebook.ipynb" in notebook_paths[0]
