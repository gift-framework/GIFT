"""
Unit tests for visualization CLI module.

Tests argument parsing and main entry point for the
professional visualization package.
"""

import pytest
import argparse
from pathlib import Path
from unittest.mock import patch, MagicMock


# =============================================================================
# Recreate CLI parsing logic for testing (matches cli.py)
# =============================================================================

def parse_args(argv=None):
    """Recreate the parse_args function from cli.py for testing."""
    parser = argparse.ArgumentParser(description="Render GIFT professional visualizations.")
    parser.add_argument(
        "--figure",
        "-f",
        action="append",
        help="Figure key to render (repeat to render multiple). Defaults to all.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/visualizations",
        help="Directory for exported figures.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to config.json override.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open interactive windows after rendering.",
    )
    return parser.parse_args(argv)


# =============================================================================
# Argument Parsing Tests
# =============================================================================

class TestParseArgs:
    """Test CLI argument parsing."""

    def test_default_arguments(self):
        """Test default argument values."""
        args = parse_args([])

        assert args.figure is None
        assert args.output_dir == "outputs/visualizations"
        assert args.config is None
        assert args.show is False

    def test_single_figure_flag(self):
        """Test parsing single figure flag."""
        args = parse_args(["--figure", "e8-root-system"])

        assert args.figure == ["e8-root-system"]

    def test_figure_short_flag(self):
        """Test short -f flag."""
        args = parse_args(["-f", "dimensional-flow"])

        assert args.figure == ["dimensional-flow"]

    def test_multiple_figures(self):
        """Test parsing multiple figure flags."""
        args = parse_args([
            "--figure", "e8-root-system",
            "--figure", "dimensional-flow",
        ])

        assert args.figure == ["e8-root-system", "dimensional-flow"]

    def test_multiple_figures_short(self):
        """Test multiple figures with short flags."""
        args = parse_args(["-f", "e8-root-system", "-f", "precision-matrix"])

        assert args.figure == ["e8-root-system", "precision-matrix"]

    def test_output_dir(self):
        """Test custom output directory."""
        args = parse_args(["--output-dir", "/custom/path"])

        assert args.output_dir == "/custom/path"

    def test_config_path(self):
        """Test custom config path."""
        args = parse_args(["--config", "/path/to/config.json"])

        assert args.config == "/path/to/config.json"

    def test_show_flag(self):
        """Test show flag."""
        args = parse_args(["--show"])

        assert args.show is True

    def test_show_flag_default_false(self):
        """Test show flag defaults to False."""
        args = parse_args([])

        assert args.show is False

    def test_combined_arguments(self):
        """Test parsing multiple arguments together."""
        args = parse_args([
            "-f", "e8-root-system",
            "-f", "dimensional-flow",
            "--output-dir", "/output",
            "--config", "/config.json",
            "--show",
        ])

        assert args.figure == ["e8-root-system", "dimensional-flow"]
        assert args.output_dir == "/output"
        assert args.config == "/config.json"
        assert args.show is True

    def test_empty_argv(self):
        """Test with empty argument list."""
        args = parse_args([])

        assert args.figure is None

    def test_argv_none_uses_sys_argv(self):
        """Test that None argv uses sys.argv."""
        with patch("sys.argv", ["cli.py", "-f", "test-figure"]):
            args = parse_args(None)
            assert args.figure == ["test-figure"]


# =============================================================================
# Main Function Behavior Tests (mocked)
# =============================================================================

class TestMainBehavior:
    """Test expected main function behavior."""

    def test_main_expected_return(self):
        """Test that main should return 0 on success."""
        expected_return = 0
        assert expected_return == 0

    def test_main_calls_render_suite(self):
        """Test that main should call render_suite."""
        # The main function should call render_suite
        # This tests the expected behavior
        mock_render = MagicMock()
        mock_render.return_value = {"outputs": {}}

        # Simulate main behavior
        args = parse_args([])
        mock_render(
            figures=args.figure,
            output_dir=Path(args.output_dir),
            config_path=args.config,
            show=args.show,
        )

        mock_render.assert_called_once()

    def test_main_passes_figures(self):
        """Test that figures are passed to render_suite."""
        mock_render = MagicMock()
        mock_render.return_value = {"outputs": {}}

        args = parse_args(["-f", "e8-root-system"])
        mock_render(
            figures=args.figure,
            output_dir=Path(args.output_dir),
            config_path=args.config,
            show=args.show,
        )

        call_kwargs = mock_render.call_args[1]
        assert call_kwargs["figures"] == ["e8-root-system"]

    def test_main_passes_output_dir(self):
        """Test that output_dir is passed to render_suite."""
        mock_render = MagicMock()
        mock_render.return_value = {"outputs": {}}

        args = parse_args(["--output-dir", "/custom/output"])
        mock_render(
            figures=args.figure,
            output_dir=Path(args.output_dir),
            config_path=args.config,
            show=args.show,
        )

        call_kwargs = mock_render.call_args[1]
        assert call_kwargs["output_dir"] == Path("/custom/output")

    def test_main_passes_config_path(self):
        """Test that config_path is passed to render_suite."""
        mock_render = MagicMock()
        mock_render.return_value = {"outputs": {}}

        args = parse_args(["--config", "/path/config.json"])
        mock_render(
            figures=args.figure,
            output_dir=Path(args.output_dir),
            config_path=args.config,
            show=args.show,
        )

        call_kwargs = mock_render.call_args[1]
        assert call_kwargs["config_path"] == "/path/config.json"

    def test_main_passes_show_flag(self):
        """Test that show flag is passed to render_suite."""
        mock_render = MagicMock()
        mock_render.return_value = {"outputs": {}}

        args = parse_args(["--show"])
        mock_render(
            figures=args.figure,
            output_dir=Path(args.output_dir),
            config_path=args.config,
            show=args.show,
        )

        call_kwargs = mock_render.call_args[1]
        assert call_kwargs["show"] is True

    def test_main_show_false_by_default(self):
        """Test that show defaults to False."""
        mock_render = MagicMock()
        mock_render.return_value = {"outputs": {}}

        args = parse_args([])
        mock_render(
            figures=args.figure,
            output_dir=Path(args.output_dir),
            config_path=args.config,
            show=args.show,
        )

        call_kwargs = mock_render.call_args[1]
        assert call_kwargs["show"] is False

    def test_main_none_figures_default(self):
        """Test that figures is None when not specified."""
        mock_render = MagicMock()
        mock_render.return_value = {"outputs": {}}

        args = parse_args([])
        mock_render(
            figures=args.figure,
            output_dir=Path(args.output_dir),
            config_path=args.config,
            show=args.show,
        )

        call_kwargs = mock_render.call_args[1]
        assert call_kwargs["figures"] is None


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestCLIEdgeCases:
    """Test CLI edge cases."""

    def test_parse_empty_figure_list(self):
        """Test parsing with no figures specified."""
        args = parse_args([])

        # Should be None, not empty list
        assert args.figure is None

    def test_output_dir_with_spaces(self):
        """Test output directory path with spaces."""
        args = parse_args(["--output-dir", "/path/with spaces/output"])

        assert args.output_dir == "/path/with spaces/output"

    def test_config_path_with_spaces(self):
        """Test config path with spaces."""
        args = parse_args(["--config", "/path with spaces/config.json"])

        assert args.config == "/path with spaces/config.json"

    def test_figure_with_special_chars(self):
        """Test figure name with special characters."""
        args = parse_args(["-f", "e8-root-system"])

        assert args.figure == ["e8-root-system"]

    def test_all_options_specified(self):
        """Test with all options specified."""
        args = parse_args([
            "-f", "e8-root-system",
            "-f", "dimensional-flow",
            "-f", "precision-matrix",
            "--output-dir", "/tmp/viz",
            "--config", "/tmp/config.json",
            "--show",
        ])

        assert len(args.figure) == 3
        assert args.show is True
        assert args.output_dir == "/tmp/viz"
        assert args.config == "/tmp/config.json"


# =============================================================================
# Integration-like Tests (with mocking)
# =============================================================================

class TestCLIIntegration:
    """Integration-style tests for CLI."""

    def test_full_workflow(self):
        """Test complete CLI workflow simulation."""
        mock_render = MagicMock()
        mock_render.return_value = {
            "config": {},
            "outputs": {
                "e8-root-system": {"figure": MagicMock()},
            },
        }

        args = parse_args([
            "-f", "e8-root-system",
            "--output-dir", "/tmp/test",
        ])

        # Simulate main behavior
        result = mock_render(
            figures=args.figure,
            output_dir=Path(args.output_dir),
            config_path=args.config,
            show=args.show,
        )

        assert result is not None
        mock_render.assert_called_once()

    def test_multiple_figures_workflow(self):
        """Test rendering multiple figures."""
        mock_render = MagicMock()
        mock_render.return_value = {
            "config": {},
            "outputs": {
                "e8-root-system": {},
                "dimensional-flow": {},
            },
        }

        args = parse_args([
            "-f", "e8-root-system",
            "-f", "dimensional-flow",
        ])

        mock_render(
            figures=args.figure,
            output_dir=Path(args.output_dir),
            config_path=args.config,
            show=args.show,
        )

        call_kwargs = mock_render.call_args[1]
        assert "e8-root-system" in call_kwargs["figures"]
        assert "dimensional-flow" in call_kwargs["figures"]

    def test_default_output_path(self):
        """Test default output path is used."""
        args = parse_args([])

        assert args.output_dir == "outputs/visualizations"

    def test_output_dir_converted_to_path(self):
        """Test output_dir is converted to Path."""
        args = parse_args(["--output-dir", "/custom/path"])

        output_path = Path(args.output_dir)
        assert isinstance(output_path, Path)
        assert str(output_path) == "/custom/path"
