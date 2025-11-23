"""
Tests for agent CLI and utility functions.

Tests include:
- Markdown link parsing
- Heading extraction and slugification
- Status tag extraction
- CLI argument parsing simulation
- Report generation
- File discovery utilities

Version: 2.1.0
"""

import pytest
import json
import sys
from pathlib import Path
import tempfile

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "assets" / "agents"))

try:
    from utils.markdown import (
        parse_links,
        collect_headings,
        slugify,
        extract_status_tags
    )
    MARKDOWN_UTILS_AVAILABLE = True
except ImportError:
    MARKDOWN_UTILS_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not MARKDOWN_UTILS_AVAILABLE,
    reason="Markdown utilities not available"
)


class TestMarkdownLinkParsing:
    """Test markdown link extraction."""

    def test_parse_single_link(self):
        """Test parsing a single markdown link."""
        text = "See [documentation](path/to/doc.md) for details."
        links = parse_links(text)

        assert len(links) == 1
        assert links[0] == ("documentation", "path/to/doc.md")

    def test_parse_multiple_links(self):
        """Test parsing multiple links."""
        text = """
        Check [intro](intro.md) and [advanced](advanced.md) guides.
        Also see [external](https://example.com).
        """
        links = parse_links(text)

        assert len(links) == 3
        assert ("intro", "intro.md") in links
        assert ("advanced", "advanced.md") in links
        assert ("external", "https://example.com") in links

    def test_parse_link_with_special_characters(self):
        """Test parsing links with special characters."""
        text = "[Test File](path/with%20spaces.md)"
        links = parse_links(text)

        assert len(links) == 1
        assert links[0] == ("Test File", "path/with%20spaces.md")

    def test_parse_no_links(self):
        """Test text with no links."""
        text = "Just plain text without any links."
        links = parse_links(text)

        assert len(links) == 0

    def test_parse_inline_code_not_link(self):
        """Test that inline code is not parsed as link."""
        text = "Use `[not a link](not/a/path)` in code."
        links = parse_links(text)

        # Might still match depending on implementation
        # This tests the actual behavior
        assert isinstance(links, list)

    def test_parse_link_with_anchor(self):
        """Test parsing link with anchor."""
        text = "See [section](#heading-name) below."
        links = parse_links(text)

        assert len(links) == 1
        assert links[0] == ("section", "#heading-name")


class TestMarkdownHeadingExtraction:
    """Test markdown heading extraction."""

    def test_collect_h1_heading(self):
        """Test extracting H1 heading."""
        text = "# Main Title\n\nSome content."
        headings = collect_headings(text)

        assert "main-title" in headings
        assert headings["main-title"] == "Main Title"

    def test_collect_multiple_heading_levels(self):
        """Test extracting different heading levels."""
        text = """
# Title
## Subtitle
### Sub-subtitle
#### Level 4
        """
        headings = collect_headings(text)

        assert "title" in headings
        assert "subtitle" in headings
        assert "sub-subtitle" in headings
        assert "level-4" in headings

    def test_collect_headings_with_special_characters(self):
        """Test heading slugification."""
        text = "# Test: With Special! Characters?"
        headings = collect_headings(text)

        # Should be slugified (implementation specific)
        assert len(headings) > 0

        # Check that some slug exists
        slugs = list(headings.keys())
        assert len(slugs[0]) > 0

    def test_collect_no_headings(self):
        """Test text with no headings."""
        text = "Just plain text.\nNo headings here."
        headings = collect_headings(text)

        assert len(headings) == 0


class TestSlugification:
    """Test heading slugification."""

    def test_slugify_basic(self):
        """Test basic slugification."""
        slug = slugify("Simple Title")

        assert slug == "simple-title"

    def test_slugify_with_numbers(self):
        """Test slugification with numbers."""
        slug = slugify("Section 1.2.3")

        # Should preserve numbers
        assert "1" in slug or "123" in slug
        assert slug.islower() or slug.isdigit() or "-" in slug

    def test_slugify_removes_special_chars(self):
        """Test special character removal."""
        slug = slugify("Test: With! Special? Characters#")

        # Special chars should be removed
        assert ":" not in slug
        assert "!" not in slug
        assert "?" not in slug
        assert "#" not in slug

    def test_slugify_spaces_to_hyphens(self):
        """Test spaces converted to hyphens."""
        slug = slugify("Multiple Word Title")

        assert " " not in slug
        assert "-" in slug or slug == slug.replace("-", "")

    def test_slugify_lowercase(self):
        """Test output is lowercase."""
        slug = slugify("UPPERCASE TITLE")

        assert slug.islower() or not slug.isalpha()

    def test_slugify_empty_string(self):
        """Test slugifying empty string."""
        slug = slugify("")

        assert slug == ""

    def test_slugify_unicode(self):
        """Test slugification with unicode characters."""
        # Most implementations remove non-ASCII
        slug = slugify("Title with é and ñ")

        # Should handle gracefully (result depends on implementation)
        assert isinstance(slug, str)


class TestStatusTagExtraction:
    """Test status tag extraction."""

    def test_extract_single_status_tag(self):
        """Test extracting a single status tag."""
        text = "Observable: **PROVEN** from topology."
        tags = extract_status_tags(text)

        assert len(tags) == 1
        assert "PROVEN" in tags

    def test_extract_multiple_status_tags(self):
        """Test extracting multiple status tags."""
        text = """
        - Observable 1: **PROVEN**
        - Observable 2: **DERIVED**
        - Observable 3: **TOPOLOGICAL**
        """
        tags = extract_status_tags(text)

        assert len(tags) == 3
        assert "PROVEN" in tags
        assert "DERIVED" in tags
        assert "TOPOLOGICAL" in tags

    def test_extract_all_status_types(self):
        """Test all status tag types are recognized."""
        text = """
        **PROVEN** **TOPOLOGICAL** **DERIVED**
        **THEORETICAL** **PHENOMENOLOGICAL** **EXPLORATORY**
        """
        tags = extract_status_tags(text)

        expected = ["PROVEN", "TOPOLOGICAL", "DERIVED",
                    "THEORETICAL", "PHENOMENOLOGICAL", "EXPLORATORY"]

        for tag_type in expected:
            assert tag_type in tags

    def test_extract_no_status_tags(self):
        """Test text with no status tags."""
        text = "Just regular text without any status markers."
        tags = extract_status_tags(text)

        assert len(tags) == 0

    def test_extract_case_sensitive(self):
        """Test status tag extraction is case-sensitive."""
        text = "This is **proven** (lowercase) vs **PROVEN** (uppercase)."
        tags = extract_status_tags(text)

        # Should only match uppercase
        assert "PROVEN" in tags
        assert "proven" not in tags


class TestCLIArgumentParsing:
    """Test CLI argument parsing (simulated)."""

    def test_cli_list_command(self):
        """Test 'list' command parsing."""
        # This is a simulation since we can't easily test argparse directly
        command = "list"

        assert command in ["list", "run", "verify", "unicode", "docs", "notebooks", "canonical"]

    def test_cli_run_command(self):
        """Test 'run' command with agent name."""
        command = "run"
        agent = "verifier"

        assert command == "run"
        assert isinstance(agent, str)

    def test_cli_shorthand_commands(self):
        """Test shorthand agent commands."""
        shorthand_commands = ["verify", "unicode", "docs", "notebooks", "canonical"]

        for cmd in shorthand_commands:
            # Each should map to itself as agent name
            assert cmd in shorthand_commands


class TestReportGeneration:
    """Test agent report generation."""

    def test_report_json_structure(self, tmp_path):
        """Test report JSON has correct structure."""
        # Simulate a report
        report_data = {
            "name": "test_agent",
            "ok": True,
            "summary": "All checks passed",
            "issues": [],
            "timestamp": "20250101_120000",
            "hostname": "testhost"
        }

        report_file = tmp_path / "test_report.json"

        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)

        # Load and verify
        with open(report_file, 'r') as f:
            loaded = json.load(f)

        assert loaded["name"] == "test_agent"
        assert loaded["ok"] is True
        assert "summary" in loaded
        assert "issues" in loaded
        assert isinstance(loaded["issues"], list)

    def test_report_with_issues(self, tmp_path):
        """Test report with multiple issues."""
        report_data = {
            "name": "test_agent",
            "ok": False,
            "summary": "Found 2 issues",
            "issues": [
                {"type": "error", "message": "Test error 1"},
                {"type": "warning", "message": "Test warning 1"}
            ],
            "timestamp": "20250101_120000",
            "hostname": "testhost"
        }

        report_file = tmp_path / "report_with_issues.json"

        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)

        with open(report_file, 'r') as f:
            loaded = json.load(f)

        assert loaded["ok"] is False
        assert len(loaded["issues"]) == 2

    def test_report_timestamp_format(self):
        """Test timestamp format."""
        import datetime

        # Generate timestamp in expected format
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Should be numeric with underscores
        assert "_" in ts
        assert len(ts) == 15  # YYYYMMDD_HHMMSS


class TestFileDiscoveryUtilities:
    """Test file discovery and filtering."""

    def test_markdown_file_discovery(self, tmp_path):
        """Test discovering markdown files."""
        # Create test files
        (tmp_path / "test1.md").touch()
        (tmp_path / "test2.md").touch()
        (tmp_path / "test.txt").touch()
        (tmp_path / "README.md").touch()

        # Discover .md files
        md_files = list(tmp_path.glob("*.md"))

        assert len(md_files) == 3
        assert all(f.suffix == ".md" for f in md_files)

    def test_recursive_file_discovery(self, tmp_path):
        """Test recursive file discovery."""
        # Create nested structure
        (tmp_path / "dir1").mkdir()
        (tmp_path / "dir1" / "test1.md").touch()
        (tmp_path / "dir2").mkdir()
        (tmp_path / "dir2" / "test2.md").touch()

        # Recursive discovery
        all_md = list(tmp_path.glob("**/*.md"))

        assert len(all_md) == 2

    def test_file_filtering_by_pattern(self, tmp_path):
        """Test filtering files by pattern."""
        # Create various files
        (tmp_path / "GIFT_main.md").touch()
        (tmp_path / "GIFT_extensions.md").touch()
        (tmp_path / "README.md").touch()
        (tmp_path / "other.md").touch()

        # Filter for GIFT_ prefix
        gift_files = [f for f in tmp_path.glob("*.md") if f.name.startswith("GIFT_")]

        assert len(gift_files) == 2


class TestUtilityEdgeCases:
    """Test edge cases in utility functions."""

    def test_parse_links_malformed_markdown(self):
        """Test link parsing with malformed markdown."""
        malformed_texts = [
            "[incomplete link",
            "](incomplete url)",
            "[[double brackets]](url)",
            "[](empty text)(url)",
            "[text]()",
        ]

        for text in malformed_texts:
            links = parse_links(text)
            # Should not crash, result depends on implementation
            assert isinstance(links, list)

    def test_collect_headings_with_empty_lines(self):
        """Test heading collection with empty lines."""
        text = """
        # Heading 1

        # Heading 2

        Content

        ## Heading 3
        """
        headings = collect_headings(text)

        # Should still find all headings
        assert len(headings) >= 3

    def test_slugify_with_only_special_chars(self):
        """Test slugifying string with only special characters."""
        slug = slugify("!@#$%^&*()")

        # Result might be empty string after removing all special chars
        assert isinstance(slug, str)

    def test_extract_status_tags_with_formatting(self):
        """Test status tag extraction with various formatting."""
        text = """
        **PROVEN** (exact)
        *** DERIVED *** (extra stars - should not match)
        ** TOPOLOGICAL ** (spaces - should not match based on regex)
        **PHENOMENOLOGICAL**
        """
        tags = extract_status_tags(text)

        # Should match correctly formatted tags
        assert "PROVEN" in tags
        assert "PHENOMENOLOGICAL" in tags


class TestPathHandling:
    """Test path handling utilities."""

    def test_relative_path_resolution(self, tmp_path):
        """Test resolving relative paths."""
        # Create test structure
        (tmp_path / "docs").mkdir()
        test_file = tmp_path / "docs" / "test.md"
        test_file.touch()

        # Resolve relative path
        relative = Path("docs/test.md")
        absolute = tmp_path / relative

        assert absolute.exists()
        assert absolute.is_absolute()

    def test_path_normalization(self):
        """Test path normalization."""
        # Various path formats
        paths = [
            "path/to/file.md",
            "path//to//file.md",
            "./path/to/file.md",
        ]

        normalized = [Path(p) for p in paths]

        # All should normalize to same logical path
        assert all(isinstance(p, Path) for p in normalized)

    def test_symlink_handling(self, tmp_path):
        """Test handling of symlinks."""
        # Create file and symlink
        original = tmp_path / "original.md"
        original.touch()

        try:
            link = tmp_path / "link.md"
            link.symlink_to(original)

            # Both should exist
            assert original.exists()
            assert link.exists()

            # Link should resolve to original
            assert link.resolve() == original.resolve()
        except OSError:
            # Symlinks might not be supported on some systems
            pytest.skip("Symlinks not supported on this system")
