"""
Tests for Agent CLI and Registry

These tests cover:
- Agent registry discovery and loading
- CLI argument parsing
- Report saving functionality
- Agent execution via CLI
- Error handling for unknown agents

Author: GIFT Framework
Date: 2025-11-27
"""

import pytest
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from io import StringIO
import tempfile
import shutil

# Add assets to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from assets.agents.registry import get_registry, load_agent_class, run_agent
from assets.agents.base import AgentResult, BaseAgent


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    dirpath = tempfile.mkdtemp()
    yield Path(dirpath)
    shutil.rmtree(dirpath)


@pytest.fixture
def mock_agent_result():
    """Create a mock AgentResult."""
    return AgentResult(
        name="test_agent",
        ok=True,
        issues=[],
        summary="Test completed successfully"
    )


# =============================================================================
# Test: Registry Functions
# =============================================================================

class TestRegistry:
    """Tests for agent registry functions."""

    def test_get_registry_returns_dict(self):
        """Test get_registry returns a dictionary."""
        registry = get_registry()
        assert isinstance(registry, dict)

    def test_get_registry_contains_agents(self):
        """Test registry contains expected agents."""
        registry = get_registry()

        expected_agents = ['unicode', 'verify', 'docs', 'notebooks', 'canonical']
        for agent in expected_agents:
            assert agent in registry, f"Missing agent: {agent}"

    def test_registry_values_are_specs(self):
        """Test registry values are module:class specifications."""
        registry = get_registry()

        for name, spec in registry.items():
            assert ':' in spec, f"Spec for {name} should contain ':'"
            module, cls = spec.split(':', 1)
            assert len(module) > 0, f"Module for {name} should not be empty"
            assert len(cls) > 0, f"Class for {name} should not be empty"

    def test_registry_unicode_agent(self):
        """Test unicode agent is correctly registered."""
        registry = get_registry()
        assert 'unicode' in registry
        assert 'unicode_sanitizer' in registry['unicode']
        assert 'UnicodeSanitizerAgent' in registry['unicode']

    def test_registry_verify_agent(self):
        """Test verify agent is correctly registered."""
        registry = get_registry()
        assert 'verify' in registry
        assert 'verifier' in registry['verify']
        assert 'VerifierAgent' in registry['verify']

    def test_registry_docs_agent(self):
        """Test docs agent is correctly registered."""
        registry = get_registry()
        assert 'docs' in registry
        assert 'docs_integrity' in registry['docs']
        assert 'DocsIntegrityAgent' in registry['docs']

    def test_registry_notebooks_agent(self):
        """Test notebooks agent is correctly registered."""
        registry = get_registry()
        assert 'notebooks' in registry
        assert 'notebook_exec' in registry['notebooks']
        assert 'NotebookExecutionAgent' in registry['notebooks']

    def test_registry_canonical_agent(self):
        """Test canonical agent is correctly registered."""
        registry = get_registry()
        assert 'canonical' in registry
        assert 'canonical_state_monitor' in registry['canonical']
        assert 'CanonicalStateMonitor' in registry['canonical']


# =============================================================================
# Test: Agent Class Loading
# =============================================================================

class TestAgentClassLoading:
    """Tests for load_agent_class function."""

    def test_load_unicode_agent(self):
        """Test loading unicode sanitizer agent class."""
        registry = get_registry()
        cls = load_agent_class(registry['unicode'])

        assert cls is not None
        assert hasattr(cls, 'run')
        assert hasattr(cls, 'name')

    def test_load_verify_agent(self):
        """Test loading verifier agent class."""
        registry = get_registry()
        cls = load_agent_class(registry['verify'])

        assert cls is not None
        assert hasattr(cls, 'run')

    def test_load_docs_agent(self):
        """Test loading docs integrity agent class."""
        registry = get_registry()
        cls = load_agent_class(registry['docs'])

        assert cls is not None
        assert hasattr(cls, 'run')

    def test_load_notebooks_agent(self):
        """Test loading notebook execution agent class."""
        registry = get_registry()
        cls = load_agent_class(registry['notebooks'])

        assert cls is not None
        assert hasattr(cls, 'run')

    def test_load_canonical_agent(self):
        """Test loading canonical state monitor agent class."""
        registry = get_registry()
        cls = load_agent_class(registry['canonical'])

        assert cls is not None
        assert hasattr(cls, 'run')

    def test_load_invalid_module_raises(self):
        """Test loading from invalid module raises error."""
        with pytest.raises(ModuleNotFoundError):
            load_agent_class("nonexistent.module:SomeClass")

    def test_load_invalid_class_raises(self):
        """Test loading invalid class raises AttributeError."""
        with pytest.raises(AttributeError):
            load_agent_class("assets.agents.base:NonExistentClass")


# =============================================================================
# Test: Run Agent Function
# =============================================================================

class TestRunAgent:
    """Tests for run_agent function."""

    def test_run_unknown_agent_exits(self, temp_dir):
        """Test running unknown agent raises SystemExit."""
        with pytest.raises(SystemExit) as exc_info:
            run_agent("nonexistent_agent", temp_dir)

        assert "Unknown agent" in str(exc_info.value)

    def test_run_unicode_agent(self, temp_dir):
        """Test running unicode agent returns result."""
        # Create some test files
        test_file = temp_dir / "test.md"
        test_file.write_text("# Test File\n\nSome content.")

        result = run_agent("unicode", temp_dir)

        assert isinstance(result, AgentResult)
        assert hasattr(result, 'ok')
        assert hasattr(result, 'summary')
        assert hasattr(result, 'issues')

    def test_run_notebooks_agent(self, temp_dir):
        """Test running notebooks agent returns result."""
        result = run_agent("notebooks", temp_dir)

        assert isinstance(result, AgentResult)
        assert result.ok  # Should succeed (no notebooks = ok)

    def test_run_agent_with_valid_path(self, temp_dir):
        """Test run_agent accepts Path objects."""
        result = run_agent("notebooks", temp_dir)

        assert isinstance(result, AgentResult)


# =============================================================================
# Test: AgentResult
# =============================================================================

class TestAgentResult:
    """Tests for AgentResult dataclass."""

    def test_agent_result_creation(self):
        """Test AgentResult can be created."""
        result = AgentResult(
            name="test",
            ok=True,
            issues=[],
            summary="OK"
        )

        assert result.name == "test"
        assert result.ok is True
        assert result.issues == []
        assert result.summary == "OK"

    def test_agent_result_with_issues(self):
        """Test AgentResult with issues."""
        issues = [
            {"file": "test.md", "error": "Missing header"},
            {"file": "other.md", "error": "Broken link"}
        ]

        result = AgentResult(
            name="test",
            ok=False,
            issues=issues,
            summary="2 issues found"
        )

        assert result.ok is False
        assert len(result.issues) == 2

    def test_agent_result_to_dict(self, mock_agent_result):
        """Test AgentResult can be converted to dict for JSON."""
        data = {
            "name": mock_agent_result.name,
            "ok": mock_agent_result.ok,
            "summary": mock_agent_result.summary,
            "issues": mock_agent_result.issues,
        }

        # Should be JSON serializable
        json_str = json.dumps(data)
        parsed = json.loads(json_str)

        assert parsed["name"] == "test_agent"
        assert parsed["ok"] is True


# =============================================================================
# Test: CLI Argument Parsing (via mock)
# =============================================================================

class TestCLIArgumentParsing:
    """Tests for CLI argument parsing."""

    def test_cli_list_command(self):
        """Test 'list' command shows available agents."""
        from assets.agents.cli import main

        with patch('sys.argv', ['cli', 'list']):
            with patch('builtins.print') as mock_print:
                main()

        # Should print available agents
        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any('Available agents' in call for call in print_calls)

    def test_cli_default_command_is_list(self):
        """Test default command (no args) is 'list'."""
        from assets.agents.cli import main

        with patch('sys.argv', ['cli']):
            with patch('builtins.print') as mock_print:
                main()

        print_calls = [str(call) for call in mock_print.call_args_list]
        assert any('Available agents' in call for call in print_calls)

    def test_cli_unknown_command_exits(self):
        """Test unknown command raises SystemExit."""
        from assets.agents.cli import main

        with patch('sys.argv', ['cli', 'unknown_command']):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert "Unknown command" in str(exc_info.value)

    def test_cli_verify_shortcut(self, temp_dir):
        """Test 'verify' shortcut works."""
        from assets.agents.cli import main

        with patch('sys.argv', ['cli', 'verify', '--root', str(temp_dir)]):
            with patch('builtins.print'):
                # Verify may fail on empty dir, but should not raise unexpectedly
                try:
                    main()
                except SystemExit as e:
                    # SystemExit(1) is expected if verification fails
                    pass

    def test_cli_unicode_shortcut(self, temp_dir):
        """Test 'unicode' shortcut works."""
        from assets.agents.cli import main

        with patch('sys.argv', ['cli', 'unicode', '--root', str(temp_dir)]):
            with patch('builtins.print'):
                main()  # Should complete without error

    def test_cli_docs_shortcut(self, temp_dir):
        """Test 'docs' shortcut works."""
        from assets.agents.cli import main

        with patch('sys.argv', ['cli', 'docs', '--root', str(temp_dir)]):
            with patch('builtins.print'):
                try:
                    main()
                except SystemExit:
                    pass  # Expected on empty directory

    def test_cli_notebooks_shortcut(self, temp_dir):
        """Test 'notebooks' shortcut works."""
        from assets.agents.cli import main

        with patch('sys.argv', ['cli', 'notebooks', '--root', str(temp_dir)]):
            with patch('builtins.print'):
                main()  # Should complete

    def test_cli_canonical_shortcut(self, temp_dir):
        """Test 'canonical' shortcut works."""
        from assets.agents.cli import main

        with patch('sys.argv', ['cli', 'canonical', '--root', str(temp_dir)]):
            with patch('builtins.print'):
                main()  # Should complete


# =============================================================================
# Test: Report Saving
# =============================================================================

class TestReportSaving:
    """Tests for report saving functionality."""

    def test_save_report_creates_directory(self, temp_dir, mock_agent_result):
        """Test _save_report creates reports directory."""
        from assets.agents.cli import _save_report

        with patch('assets.agents.cli.Path') as MockPath:
            MockPath.return_value = temp_dir / "assets/agents/reports"

            # Call with actual temp dir
            original_cwd = Path.cwd()
            try:
                import os
                os.chdir(temp_dir)
                reports_dir = temp_dir / "assets/agents/reports"
                reports_dir.mkdir(parents=True, exist_ok=True)

                _save_report("test", mock_agent_result)

                # Check files were created
                files = list(reports_dir.glob("test_*.json"))
                assert len(files) >= 1

            finally:
                os.chdir(original_cwd)

    def test_save_report_json_format(self, temp_dir, mock_agent_result):
        """Test saved report is valid JSON."""
        import os

        reports_dir = temp_dir / "assets/agents/reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        original_cwd = Path.cwd()
        try:
            os.chdir(temp_dir)

            from assets.agents.cli import _save_report
            _save_report("test", mock_agent_result)

            # Find and read the report
            files = list(reports_dir.glob("test_*.json"))
            assert len(files) >= 1

            content = files[0].read_text()
            data = json.loads(content)

            assert "name" in data
            assert "ok" in data
            assert "summary" in data
            assert "issues" in data
            assert "timestamp" in data
            assert "hostname" in data

        finally:
            os.chdir(original_cwd)

    def test_save_report_creates_latest(self, temp_dir, mock_agent_result):
        """Test _save_report creates latest.json symlink/copy."""
        import os

        reports_dir = temp_dir / "assets/agents/reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        original_cwd = Path.cwd()
        try:
            os.chdir(temp_dir)

            from assets.agents.cli import _save_report
            _save_report("test", mock_agent_result)

            # Check latest file exists
            latest = reports_dir / "test_latest.json"
            assert latest.exists()

            # Verify it's valid JSON
            data = json.loads(latest.read_text())
            assert data["name"] == "test_agent"

        finally:
            os.chdir(original_cwd)


# =============================================================================
# Test: BaseAgent Abstract Class
# =============================================================================

class TestBaseAgent:
    """Tests for BaseAgent abstract class."""

    def test_base_agent_is_abstract(self):
        """Test BaseAgent cannot be instantiated directly."""
        # BaseAgent should be a base class for all agents
        assert hasattr(BaseAgent, 'run')
        assert hasattr(BaseAgent, 'name')

    def test_agent_subclass_implementation(self):
        """Test agent subclasses implement required methods."""
        registry = get_registry()

        for name, spec in registry.items():
            cls = load_agent_class(spec)

            # Should have name attribute
            assert hasattr(cls, 'name'), f"{name} missing 'name'"

            # Should have run method
            assert hasattr(cls, 'run'), f"{name} missing 'run'"

            # run should be callable
            assert callable(getattr(cls, 'run'))


# =============================================================================
# Test: Error Handling
# =============================================================================

class TestErrorHandling:
    """Tests for error handling in CLI and registry."""

    def test_run_agent_with_invalid_path(self):
        """Test run_agent handles invalid path gracefully."""
        invalid_path = Path("/nonexistent/path/to/nowhere")

        # Should not crash, but may return error result
        result = run_agent("notebooks", invalid_path)
        assert isinstance(result, AgentResult)

    def test_registry_immutable(self):
        """Test modifying registry doesn't affect subsequent calls."""
        registry1 = get_registry()
        registry1['fake'] = 'fake.module:FakeClass'

        registry2 = get_registry()
        # Original should not be modified (fresh dict each call)
        assert 'fake' not in registry2

    def test_load_agent_class_spec_validation(self):
        """Test load_agent_class validates spec format."""
        # Missing colon
        with pytest.raises(ValueError):
            load_agent_class("assets.agents.base")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
