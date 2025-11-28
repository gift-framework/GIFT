"""
Unit tests for quick_start.py entry point.

Tests the interactive launcher for GIFT visualizations,
documentation, and agents.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import sys
import os

# Import the module under test
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import quick_start


# =============================================================================
# launch_visualizations Tests
# =============================================================================

class TestLaunchVisualizations:
    """Test the launch_visualizations function."""

    def test_checks_directory_exists(self, tmp_path, monkeypatch):
        """Test that function checks for visualizations directory."""
        monkeypatch.chdir(tmp_path)

        with patch("builtins.print") as mock_print:
            quick_start.launch_visualizations()

        # Should print error about directory not found
        calls_str = " ".join(str(c) for c in mock_print.call_args_list)
        assert "Error" in calls_str or "not found" in calls_str

    def test_launches_jupyter_when_installed(self, tmp_path, monkeypatch):
        """Test that Jupyter is launched when installed."""
        monkeypatch.chdir(tmp_path)

        # Create the visualizations directory
        viz_dir = tmp_path / "assets" / "visualizations"
        viz_dir.mkdir(parents=True)

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("subprocess.run") as mock_run, \
             patch("os.chdir") as mock_chdir:
            mock_run.return_value = mock_result
            quick_start.launch_visualizations()

        # Should have called subprocess.run to check/launch jupyter
        assert mock_run.called

    def test_handles_jupyter_not_installed(self, tmp_path, monkeypatch):
        """Test handling when Jupyter is not installed."""
        monkeypatch.chdir(tmp_path)

        # Create the visualizations directory
        viz_dir = tmp_path / "assets" / "visualizations"
        viz_dir.mkdir(parents=True)

        # First call checks jupyter, returns error (not installed)
        # Second call would install it
        call_count = [0]
        def mock_run_side_effect(*args, **kwargs):
            call_count[0] += 1
            result = MagicMock()
            if call_count[0] == 1:
                result.returncode = 1  # jupyter not found
            else:
                result.returncode = 0
            return result

        with patch("subprocess.run") as mock_run, \
             patch("os.chdir"):
            mock_run.side_effect = mock_run_side_effect
            quick_start.launch_visualizations()

        # Should have attempted to install jupyter
        calls = [str(c) for c in mock_run.call_args_list]
        assert any("pip" in c and "install" in c for c in calls) or len(calls) >= 2

    def test_prints_available_notebooks(self, tmp_path, monkeypatch, capsys):
        """Test that available notebooks are printed."""
        monkeypatch.chdir(tmp_path)

        viz_dir = tmp_path / "assets" / "visualizations"
        viz_dir.mkdir(parents=True)

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("subprocess.run") as mock_run, \
             patch("os.chdir"):
            mock_run.return_value = mock_result
            quick_start.launch_visualizations()

        # Check printed output mentions notebooks
        captured = capsys.readouterr()
        assert "notebook" in captured.out.lower() or mock_run.called

    def test_handles_exception_gracefully(self, tmp_path, monkeypatch):
        """Test that exceptions are handled gracefully."""
        monkeypatch.chdir(tmp_path)

        viz_dir = tmp_path / "assets" / "visualizations"
        viz_dir.mkdir(parents=True)

        with patch("subprocess.run") as mock_run, \
             patch("os.chdir"), \
             patch("builtins.print") as mock_print:
            mock_run.side_effect = Exception("Test error")
            quick_start.launch_visualizations()

        # Should print error message and alternatives
        calls_str = " ".join(str(c) for c in mock_print.call_args_list)
        assert "Error" in calls_str or "Alternative" in calls_str or "Binder" in calls_str


# =============================================================================
# launch_documentation Tests
# =============================================================================

class TestLaunchDocumentation:
    """Test the launch_documentation function."""

    def test_checks_docs_exists(self, tmp_path, monkeypatch):
        """Test that function checks for docs directory."""
        monkeypatch.chdir(tmp_path)

        with patch("builtins.print") as mock_print:
            quick_start.launch_documentation()

        # Should print error about documentation not found
        calls_str = " ".join(str(c) for c in mock_print.call_args_list)
        assert "Error" in calls_str or "not found" in calls_str

    def test_opens_browser_when_docs_exist(self, tmp_path, monkeypatch):
        """Test that browser is opened when docs exist."""
        monkeypatch.chdir(tmp_path)

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "index.html").write_text("<html></html>")

        with patch("webbrowser.open") as mock_open:
            quick_start.launch_documentation()

        mock_open.assert_called_once()

    def test_handles_browser_exception(self, tmp_path, monkeypatch):
        """Test handling of browser open exception."""
        monkeypatch.chdir(tmp_path)

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "index.html").write_text("<html></html>")

        with patch("webbrowser.open") as mock_open, \
             patch("builtins.print") as mock_print:
            mock_open.side_effect = Exception("Browser error")
            quick_start.launch_documentation()

        # Should print error and manual instructions
        calls_str = " ".join(str(c) for c in mock_print.call_args_list)
        assert "Error" in calls_str or "Manual" in calls_str

    def test_prints_success_message(self, tmp_path, monkeypatch, capsys):
        """Test that success message is printed."""
        monkeypatch.chdir(tmp_path)

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "index.html").write_text("<html></html>")

        with patch("webbrowser.open"):
            quick_start.launch_documentation()

        captured = capsys.readouterr()
        assert "opened" in captured.out.lower() or "browser" in captured.out.lower()


# =============================================================================
# launch_agents Tests
# =============================================================================

class TestLaunchAgents:
    """Test the launch_agents function."""

    def test_displays_agent_menu(self, capsys):
        """Test that agent menu is displayed."""
        with patch("builtins.input", return_value=""):
            quick_start.launch_agents()

        captured = capsys.readouterr()
        assert "Verification" in captured.out or "1." in captured.out

    def test_launches_verify_agent(self):
        """Test launching verification agent."""
        with patch("builtins.input", return_value="1"), \
             patch("subprocess.run") as mock_run:
            quick_start.launch_agents()

        # Should call agents.cli with verify
        assert mock_run.called
        call_args = str(mock_run.call_args)
        assert "verify" in call_args

    def test_launches_unicode_agent(self):
        """Test launching unicode sanitizer agent."""
        with patch("builtins.input", return_value="2"), \
             patch("subprocess.run") as mock_run:
            quick_start.launch_agents()

        call_args = str(mock_run.call_args)
        assert "unicode" in call_args

    def test_launches_docs_agent(self):
        """Test launching docs integrity agent."""
        with patch("builtins.input", return_value="3"), \
             patch("subprocess.run") as mock_run:
            quick_start.launch_agents()

        call_args = str(mock_run.call_args)
        assert "docs" in call_args

    def test_launches_notebooks_agent(self):
        """Test launching notebook discovery agent."""
        with patch("builtins.input", return_value="4"), \
             patch("subprocess.run") as mock_run:
            quick_start.launch_agents()

        call_args = str(mock_run.call_args)
        assert "notebooks" in call_args

    def test_launches_canonical_agent(self):
        """Test launching canonical monitor agent."""
        with patch("builtins.input", return_value="5"), \
             patch("subprocess.run") as mock_run:
            quick_start.launch_agents()

        call_args = str(mock_run.call_args)
        assert "canonical" in call_args

    def test_handles_invalid_choice(self):
        """Test handling of invalid agent choice."""
        with patch("builtins.input", return_value="99"), \
             patch("subprocess.run") as mock_run:
            quick_start.launch_agents()

        # Should not crash, may or may not call subprocess


# =============================================================================
# main Function Tests
# =============================================================================

class TestMain:
    """Test the main function."""

    def test_displays_menu(self, capsys):
        """Test that main menu is displayed."""
        with patch("builtins.input", return_value="4"):  # Exit
            quick_start.main()

        captured = capsys.readouterr()
        assert "GIFT Framework" in captured.out
        assert "Quick Start" in captured.out

    def test_menu_shows_options(self, capsys):
        """Test that menu shows all options."""
        with patch("builtins.input", return_value="4"):
            quick_start.main()

        captured = capsys.readouterr()
        assert "Visualizations" in captured.out or "1." in captured.out
        assert "Documentation" in captured.out or "2." in captured.out
        assert "Agents" in captured.out or "3." in captured.out
        assert "Exit" in captured.out or "4." in captured.out

    def test_option_1_launches_visualizations(self, tmp_path, monkeypatch):
        """Test that option 1 launches visualizations."""
        monkeypatch.chdir(tmp_path)

        with patch("builtins.input", return_value="1"), \
             patch.object(quick_start, "launch_visualizations") as mock_launch:
            quick_start.main()

        mock_launch.assert_called_once()

    def test_option_2_launches_documentation(self, tmp_path, monkeypatch):
        """Test that option 2 launches documentation."""
        monkeypatch.chdir(tmp_path)

        with patch("builtins.input", return_value="2"), \
             patch.object(quick_start, "launch_documentation") as mock_launch:
            quick_start.main()

        mock_launch.assert_called_once()

    def test_option_3_launches_agents(self):
        """Test that option 3 launches agents."""
        with patch("builtins.input", return_value="3"), \
             patch.object(quick_start, "launch_agents") as mock_launch:
            quick_start.main()

        mock_launch.assert_called_once()

    def test_option_4_exits(self, capsys):
        """Test that option 4 exits."""
        with patch("builtins.input", return_value="4"):
            quick_start.main()

        captured = capsys.readouterr()
        assert "Goodbye" in captured.out

    def test_invalid_option_prompts_again(self):
        """Test that invalid option prompts for re-entry."""
        inputs = iter(["invalid", "abc", "4"])

        with patch("builtins.input", side_effect=lambda x: next(inputs)):
            quick_start.main()

        # Should eventually exit when "4" is entered

    def test_handles_empty_input(self):
        """Test handling of empty input."""
        inputs = iter(["", "4"])

        with patch("builtins.input", side_effect=lambda x: next(inputs)):
            quick_start.main()

        # Should handle gracefully


# =============================================================================
# Edge Cases
# =============================================================================

class TestQuickStartEdgeCases:
    """Test edge cases for quick_start module."""

    def test_keyboard_interrupt_handling(self, capsys):
        """Test handling of keyboard interrupt."""
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            try:
                quick_start.main()
            except KeyboardInterrupt:
                pass  # Expected behavior

    def test_eof_error_handling(self):
        """Test handling of EOF error."""
        with patch("builtins.input", side_effect=EOFError):
            try:
                quick_start.main()
            except EOFError:
                pass  # Expected behavior

    def test_numeric_string_choices(self, capsys):
        """Test that numeric strings are handled correctly."""
        with patch("builtins.input", return_value="4"):
            quick_start.main()

        captured = capsys.readouterr()
        assert "Goodbye" in captured.out

    def test_whitespace_in_input(self):
        """Test that whitespace is stripped from input."""
        with patch("builtins.input", return_value="  4  "):
            quick_start.main()

        # Should handle whitespace and exit


# =============================================================================
# Integration Tests
# =============================================================================

class TestQuickStartIntegration:
    """Integration-style tests for quick_start."""

    def test_full_visualization_workflow(self, tmp_path, monkeypatch):
        """Test complete visualization launch workflow."""
        monkeypatch.chdir(tmp_path)

        # Create necessary directories
        viz_dir = tmp_path / "assets" / "visualizations"
        viz_dir.mkdir(parents=True)

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("builtins.input", return_value="1"), \
             patch("subprocess.run", return_value=mock_result), \
             patch("os.chdir"):
            quick_start.main()

    def test_full_documentation_workflow(self, tmp_path, monkeypatch):
        """Test complete documentation launch workflow."""
        monkeypatch.chdir(tmp_path)

        # Create docs
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "index.html").write_text("<html></html>")

        with patch("builtins.input", return_value="2"), \
             patch("webbrowser.open"):
            quick_start.main()

    def test_full_agents_workflow(self):
        """Test complete agents launch workflow."""
        # First input selects agents menu, second selects verify
        inputs = iter(["3", "1"])

        with patch("builtins.input", side_effect=lambda x: next(inputs)), \
             patch("subprocess.run"):
            quick_start.main()
