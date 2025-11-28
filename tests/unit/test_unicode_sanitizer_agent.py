"""
Unit tests for Unicode Sanitizer Agent.

Tests transliteration of Greek letters, subscripts, and
special characters, as well as the agent's file scanning functionality.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add agents to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "assets"))

from agents.unicode_sanitizer import (
    transliterate_identifier,
    UnicodeSanitizerAgent,
    REPLACEMENTS,
    ASCII_LINE,
)
from agents.base import AgentResult


# =============================================================================
# REPLACEMENTS Dictionary Tests
# =============================================================================

class TestReplacementsDict:
    """Test the REPLACEMENTS mapping."""

    def test_greek_letters_present(self):
        """Test that common Greek letters are in replacements."""
        greek_letters = ["theta", "tau", "mu", "nu", "pi", "alpha", "beta", "gamma", "delta", "epsilon", "xi"]
        greek_symbols = ["θ", "τ", "μ", "ν", "π", "α", "β", "γ", "δ", "ε", "ξ"]

        for symbol, expected in zip(greek_symbols, greek_letters):
            assert symbol in REPLACEMENTS, f"Missing {symbol}"
            assert REPLACEMENTS[symbol] == expected

    def test_uppercase_greek_present(self):
        """Test uppercase Greek letters."""
        assert REPLACEMENTS["Ω"] == "Omega"
        assert REPLACEMENTS["Δ"] == "Delta"

    def test_subscript_numbers_present(self):
        """Test subscript number replacements."""
        for i in range(10):
            subscript = chr(0x2080 + i)  # Unicode subscript digits
            assert subscript in REPLACEMENTS
            assert REPLACEMENTS[subscript] == str(i)

    def test_special_characters_present(self):
        """Test special character replacements."""
        assert REPLACEMENTS["×"] == "x"
        assert REPLACEMENTS["–"] == "-"  # en-dash
        assert REPLACEMENTS["—"] == "-"  # em-dash
        assert REPLACEMENTS["'"] == "'"  # curly quote
        assert REPLACEMENTS["°"] == "deg"


# =============================================================================
# transliterate_identifier Tests
# =============================================================================

class TestTransliterateIdentifier:
    """Test the transliterate_identifier function."""

    def test_greek_theta(self):
        """Test transliteration of theta."""
        result = transliterate_identifier("θ")
        assert result == "theta"

    def test_greek_tau(self):
        """Test transliteration of tau."""
        result = transliterate_identifier("τ")
        assert result == "tau"

    def test_greek_mu(self):
        """Test transliteration of mu."""
        result = transliterate_identifier("μ")
        assert result == "mu"

    def test_greek_pi(self):
        """Test transliteration of pi."""
        result = transliterate_identifier("π")
        assert result == "pi"

    def test_greek_alpha(self):
        """Test transliteration of alpha."""
        result = transliterate_identifier("α")
        assert result == "alpha"

    def test_greek_beta(self):
        """Test transliteration of beta."""
        result = transliterate_identifier("β")
        assert result == "beta"

    def test_uppercase_omega(self):
        """Test transliteration of uppercase Omega."""
        result = transliterate_identifier("Ω")
        assert result == "Omega"

    def test_uppercase_delta(self):
        """Test transliteration of uppercase Delta."""
        result = transliterate_identifier("Δ")
        assert result == "Delta"

    def test_subscript_zero(self):
        """Test transliteration of subscript 0."""
        result = transliterate_identifier("x₀")
        assert "0" in result
        assert "x" in result

    def test_subscript_numbers(self):
        """Test transliteration of various subscripts."""
        for i in range(10):
            subscript = chr(0x2080 + i)
            result = transliterate_identifier(f"x{subscript}")
            assert str(i) in result

    def test_multiplication_sign(self):
        """Test transliteration of multiplication sign."""
        result = transliterate_identifier("E₈×E₈")
        assert "x" in result
        assert "8" in result

    def test_en_dash(self):
        """Test transliteration of en-dash."""
        result = transliterate_identifier("a–b")
        assert "-" in result or "a_b" in result

    def test_em_dash(self):
        """Test transliteration of em-dash."""
        result = transliterate_identifier("a—b")
        assert "-" in result or "a_b" in result

    def test_degree_symbol(self):
        """Test transliteration of degree symbol."""
        result = transliterate_identifier("90°")
        assert "deg" in result

    def test_curly_quote(self):
        """Test transliteration of curly quote."""
        result = transliterate_identifier("it's")
        # Curly quote becomes straight quote, then may be stripped
        assert "it" in result

    def test_empty_string_returns_id(self):
        """Test that empty string returns 'id'."""
        result = transliterate_identifier("")
        assert result == "id"

    def test_pure_ascii_unchanged(self):
        """Test that pure ASCII is mostly preserved."""
        result = transliterate_identifier("hello_world")
        assert "hello" in result
        assert "world" in result

    def test_complex_identifier(self):
        """Test complex identifier with multiple Greek letters."""
        result = transliterate_identifier("sin²θ_W")
        assert "theta" in result
        assert "W" in result

    def test_weinberg_angle(self):
        """Test Weinberg angle notation."""
        result = transliterate_identifier("θ_W")
        assert "theta" in result
        assert "W" in result

    def test_betti_number_notation(self):
        """Test Betti number notation."""
        result = transliterate_identifier("b₂")
        assert "b" in result
        assert "2" in result

    def test_removes_non_word_chars(self):
        """Test that non-word characters become underscores."""
        result = transliterate_identifier("a!b@c#d")
        # Should become something like a_b_c_d
        assert "_" in result or result in ["a_b_c_d", "abcd"]

    def test_strips_leading_trailing_underscores(self):
        """Test that leading/trailing underscores are stripped."""
        result = transliterate_identifier("_test_")
        assert not result.startswith("_")
        assert not result.endswith("_")

    def test_unicode_normalization(self):
        """Test Unicode NFKD normalization."""
        # Composed character (e with acute)
        result = transliterate_identifier("cafe\u0301")
        # Should normalize and potentially strip diacritic
        assert "cafe" in result or "caf" in result


# =============================================================================
# ASCII_LINE Regex Tests
# =============================================================================

class TestASCIILineRegex:
    """Test the ASCII_LINE regex pattern."""

    def test_pure_ascii_matches(self):
        """Test that pure ASCII lines match."""
        assert ASCII_LINE.match("Hello, World!")
        assert ASCII_LINE.match("def function():")
        assert ASCII_LINE.match("# This is a comment")

    def test_empty_string_no_match(self):
        """Test that empty string doesn't match (regex requires at least one char)."""
        # The regex uses + which requires at least one character
        assert not ASCII_LINE.match("")

    def test_greek_letter_no_match(self):
        """Test that Greek letters don't match."""
        assert not ASCII_LINE.match("theta = θ")
        assert not ASCII_LINE.match("α + β = γ")

    def test_subscript_no_match(self):
        """Test that subscripts don't match."""
        assert not ASCII_LINE.match("x₀ = 0")
        assert not ASCII_LINE.match("H₂O")

    def test_mixed_content_no_match(self):
        """Test that mixed ASCII/Unicode doesn't match."""
        assert not ASCII_LINE.match("The angle θ is 90°")


# =============================================================================
# UnicodeSanitizerAgent Tests
# =============================================================================

class TestUnicodeSanitizerAgent:
    """Test the UnicodeSanitizerAgent class."""

    @pytest.fixture
    def agent(self):
        """Create agent instance."""
        return UnicodeSanitizerAgent()

    def test_agent_name(self, agent):
        """Test agent has correct name."""
        assert agent.name == "unicode"

    def test_run_with_ascii_only_files(self, agent, tmp_path):
        """Test run with only ASCII files."""
        # Create ASCII-only file
        (tmp_path / "test.py").write_text("def hello():\n    return 'world'\n")
        (tmp_path / "test.md").write_text("# Hello World\n\nThis is a test.\n")

        result = agent.run(tmp_path)

        assert isinstance(result, AgentResult)
        assert result.ok is True
        assert "No non-ASCII" in result.summary

    def test_run_detects_non_ascii(self, agent, tmp_path):
        """Test run detects non-ASCII content."""
        # Create file with Greek letter
        (tmp_path / "test.py").write_text("theta = θ\n")

        result = agent.run(tmp_path)

        assert result.ok is False
        assert len(result.issues) > 0

    def test_run_reports_file_with_issues(self, agent, tmp_path):
        """Test that issue reports include file path."""
        (tmp_path / "greek.py").write_text("α = 1\n")

        result = agent.run(tmp_path)

        assert result.ok is False
        assert any("greek.py" in str(issue.get("file", "")) for issue in result.issues)

    def test_run_reports_line_numbers(self, agent, tmp_path):
        """Test that issues include line numbers."""
        (tmp_path / "test.py").write_text("line1 = 1\ntheta = θ\nline3 = 3\n")

        result = agent.run(tmp_path)

        assert result.ok is False
        # Check that line info is reported
        for issue in result.issues:
            if "non_ascii_lines" in issue:
                lines = issue["non_ascii_lines"]
                assert any(line.get("line") == 2 for line in lines)

    def test_run_includes_snippet(self, agent, tmp_path):
        """Test that issues include line snippets."""
        (tmp_path / "test.py").write_text("theta = θ\n")

        result = agent.run(tmp_path)

        for issue in result.issues:
            if "non_ascii_lines" in issue:
                for line_info in issue["non_ascii_lines"]:
                    assert "snippet" in line_info
                    assert "θ" in line_info["snippet"]

    def test_run_with_empty_directory(self, agent, tmp_path):
        """Test run with empty directory."""
        result = agent.run(tmp_path)

        assert isinstance(result, AgentResult)
        assert result.ok is True

    def test_run_only_scans_py_and_md(self, agent, tmp_path):
        """Test that only .py and .md files are scanned."""
        # Create files with various extensions
        (tmp_path / "test.py").write_text("θ = 1\n")  # Should be scanned
        (tmp_path / "test.md").write_text("θ is theta\n")  # Should be scanned
        (tmp_path / "test.txt").write_text("θ ignored\n")  # Should NOT be scanned
        (tmp_path / "test.json").write_text('{"θ": 1}\n')  # Should NOT be scanned

        result = agent.run(tmp_path)

        # Should find issues in .py and .md files
        assert result.ok is False
        file_names = [issue.get("file", "") for issue in result.issues]
        assert any("test.py" in f for f in file_names)
        assert any("test.md" in f for f in file_names)
        # Should not find issues in .txt or .json
        assert not any("test.txt" in f for f in file_names)
        assert not any("test.json" in f for f in file_names)

    def test_run_handles_read_errors(self, agent, tmp_path):
        """Test that read errors are handled gracefully."""
        # Create a file but mock read to fail
        test_file = tmp_path / "test.py"
        test_file.write_text("content")

        with patch.object(Path, "read_text", side_effect=IOError("Read failed")):
            # This should not raise, just report the error
            result = agent.run(tmp_path)
            # Should have an error issue
            assert any("error" in str(issue) for issue in result.issues)

    def test_run_summary_message(self, agent, tmp_path):
        """Test summary message content."""
        (tmp_path / "test.py").write_text("θ\nα\nβ\n")

        result = agent.run(tmp_path)

        assert "non-ASCII" in result.summary or "1 file" in result.summary

    def test_run_multiple_files_with_issues(self, agent, tmp_path):
        """Test run with multiple files having issues."""
        (tmp_path / "file1.py").write_text("θ = 1\n")
        (tmp_path / "file2.py").write_text("α = 2\n")
        (tmp_path / "file3.md").write_text("# Title with π\n")

        result = agent.run(tmp_path)

        assert result.ok is False
        assert len(result.issues) == 3


# =============================================================================
# Integration Tests
# =============================================================================

class TestUnicodeSanitizerIntegration:
    """Integration tests for Unicode sanitizer."""

    @pytest.fixture
    def agent(self):
        return UnicodeSanitizerAgent()

    def test_real_world_physics_content(self, agent, tmp_path):
        """Test with real-world physics content."""
        content = """
# GIFT Framework Constants

sin²θ_W = 3/13  # Weinberg angle
α_s = √2/12      # Strong coupling
τ = 3472/891     # Hierarchy parameter
Ω_DE = ln(2)×98/99  # Dark energy
"""
        (tmp_path / "constants.md").write_text(content)

        result = agent.run(tmp_path)

        # Should detect non-ASCII
        assert result.ok is False
        assert len(result.issues) > 0

    def test_clean_python_code(self, agent, tmp_path):
        """Test with clean Python code."""
        content = '''
def calculate_weinberg_angle():
    """Calculate sin^2(theta_W)."""
    return 3 / 13

def calculate_hierarchy():
    """Calculate tau parameter."""
    return 3472 / 891
'''
        (tmp_path / "physics.py").write_text(content)

        result = agent.run(tmp_path)

        # Should pass - all ASCII
        assert result.ok is True
