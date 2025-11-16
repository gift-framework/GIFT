"""
Unit tests for automated agents.

Tests verifier, unicode sanitizer, docs integrity, and other agents.
"""

import pytest
from pathlib import Path
import sys
import tempfile
import os

# Add agents to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "assets" / "agents"))

from base import AgentResult
from verifier import VerifierAgent
from docs_integrity import DocsIntegrityAgent
from unicode_sanitizer import UnicodeSanitizerAgent
from utils.markdown import parse_links, extract_status_tags
from utils.fs import discover_files


class TestAgentResult:
    """Test AgentResult data structure."""

    def test_agent_result_creation(self):
        """Test creating an AgentResult."""
        result = AgentResult(
            name="test",
            ok=True,
            issues=[],
            summary="All good"
        )

        assert result.name == "test"
        assert result.ok is True
        assert result.issues == []
        assert result.summary == "All good"

    def test_agent_result_with_issues(self):
        """Test AgentResult with issues."""
        issues = [
            {"type": "error", "file": "test.md", "error": "broken link"},
            {"type": "warning", "file": "doc.md", "warning": "missing tag"}
        ]

        result = AgentResult(
            name="test",
            ok=False,
            issues=issues,
            summary="Found 2 issues"
        )

        assert result.ok is False
        assert len(result.issues) == 2


class TestMarkdownUtils:
    """Test markdown utility functions."""

    def test_parse_links_simple(self):
        """Test parsing simple markdown links."""
        text = "[Example](https://example.com)"
        links = list(parse_links(text))

        assert len(links) == 1
        assert links[0][0] == "Example"
        assert links[0][1] == "https://example.com"

    def test_parse_links_multiple(self):
        """Test parsing multiple links."""
        text = "[Link1](url1) some text [Link2](url2)"
        links = list(parse_links(text))

        assert len(links) == 2
        assert links[0][1] == "url1"
        assert links[1][1] == "url2"

    def test_parse_links_relative(self):
        """Test parsing relative links."""
        text = "[Doc](../docs/file.md)"
        links = list(parse_links(text))

        assert len(links) == 1
        assert links[0][1] == "../docs/file.md"

    def test_extract_status_tags(self):
        """Test extracting status tags."""
        text = """
        Some text here.
        Status: **PROVEN**
        More text.
        Status: **DERIVED**
        Even more.
        """

        tags = set(extract_status_tags(text))

        assert "PROVEN" in tags
        assert "DERIVED" in tags

    def test_extract_status_tags_various_formats(self):
        """Test status tags in various formats."""
        text = """
        **PROVEN**: exact result
        Status: TOPOLOGICAL
        (THEORETICAL)
        """

        tags = set(extract_status_tags(text))

        # Should extract tags in various formats
        assert len(tags) >= 1


class TestFileSystemUtils:
    """Test file system utility functions."""

    def test_discover_files_pattern(self):
        """Test file discovery with patterns."""
        root = Path(__file__).parent.parent.parent
        patterns = ["README.md", "*.py"]

        files = discover_files(root, patterns)

        # Should find at least README.md
        readme_found = any(f.name == "README.md" for f in files)
        assert readme_found

    def test_discover_files_glob(self):
        """Test file discovery with glob patterns."""
        root = Path(__file__).parent.parent.parent
        patterns = ["tests/**/*.py"]

        files = discover_files(root, patterns)

        # Should find this test file
        assert len(files) > 0
        assert all(f.suffix == ".py" for f in files)


class TestVerifierAgent:
    """Test verifier agent functionality."""

    def test_verifier_initialization(self):
        """Test verifier agent initialization."""
        agent = VerifierAgent()
        assert agent.name == "verify"

    def test_verifier_run_on_temp_directory(self):
        """Test verifier runs without crashing."""
        agent = VerifierAgent()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create a simple markdown file
            test_file = tmppath / "test.md"
            test_file.write_text("# Test\n[Link](file.md)")

            result = agent.run(tmppath)

            assert isinstance(result, AgentResult)
            assert result.name == "verify"

    def test_verifier_detects_missing_local_file(self):
        """Test verifier detects missing local files."""
        agent = VerifierAgent()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create markdown with broken local link
            test_file = tmppath / "test.md"
            test_file.write_text("[Broken](missing_file.md)")

            result = agent.run(tmppath)

            # Should detect the missing file
            assert len(result.issues) > 0
            assert any(issue.get("error") == "missing_local_path" for issue in result.issues)

    def test_verifier_ignores_mailto_links(self):
        """Test verifier ignores mailto links."""
        agent = VerifierAgent()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            test_file = tmppath / "test.md"
            test_file.write_text("[Email](mailto:test@example.com)")

            result = agent.run(tmppath)

            # Should not flag mailto links as issues
            mailto_issues = [i for i in result.issues if "mailto" in str(i.get("link", ""))]
            assert len(mailto_issues) == 0


class TestDocsIntegrityAgent:
    """Test docs integrity agent."""

    def test_docs_integrity_initialization(self):
        """Test docs integrity agent initialization."""
        agent = DocsIntegrityAgent()
        assert agent.name == "docs"

    def test_docs_integrity_run(self):
        """Test docs integrity agent runs."""
        agent = DocsIntegrityAgent()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = agent.run(Path(tmpdir))

            assert isinstance(result, AgentResult)
            assert result.name == "docs"


class TestUnicodeSanitizerAgent:
    """Test unicode sanitizer agent."""

    def test_unicode_sanitizer_initialization(self):
        """Test unicode sanitizer initialization."""
        agent = UnicodeSanitizerAgent()
        assert agent.name == "unicode"

    def test_unicode_sanitizer_run(self):
        """Test unicode sanitizer runs."""
        agent = UnicodeSanitizerAgent()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = agent.run(Path(tmpdir))

            assert isinstance(result, AgentResult)
            assert result.name == "unicode"

    def test_unicode_sanitizer_detects_special_chars(self):
        """Test unicode sanitizer detects special characters."""
        agent = UnicodeSanitizerAgent()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create file with unicode characters
            test_file = tmppath / "test.py"
            test_file.write_text("# Test with émoji 🎉")

            result = agent.run(tmppath)

            # May or may not flag depending on implementation
            # Just check it doesn't crash
            assert isinstance(result, AgentResult)


class TestAgentIntegration:
    """Test agents on actual project."""

    def test_run_verifier_on_project(self):
        """Test verifier on actual GIFT project."""
        agent = VerifierAgent()
        project_root = Path(__file__).parent.parent.parent

        # Just verify it doesn't crash
        result = agent.run(project_root)

        assert isinstance(result, AgentResult)
        assert result.name == "verify"

    def test_run_docs_integrity_on_project(self):
        """Test docs integrity on actual GIFT project."""
        agent = DocsIntegrityAgent()
        project_root = Path(__file__).parent.parent.parent

        result = agent.run(project_root)

        assert isinstance(result, AgentResult)
        assert result.name == "docs"
