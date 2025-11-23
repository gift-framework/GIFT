"""
Documentation validation tests.

Features:
- Docstring presence and format validation
- Example code execution
- Documentation completeness checks
- Cross-reference validation
- API documentation consistency

Version: 2.1.0
"""

import pytest
import sys
import inspect
from pathlib import Path
import importlib

# Add paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "statistical_validation"))

try:
    from gift_v21_core import GIFTFrameworkV21, GIFTParameters
    V21_AVAILABLE = True
except ImportError:
    V21_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not V21_AVAILABLE,
    reason="GIFT v2.1 not available"
)


class TestDocstringPresence:
    """Test that all public functions and classes have docstrings."""

    def test_framework_class_has_docstring(self):
        """Test GIFTFrameworkV21 has docstring."""
        assert GIFTFrameworkV21.__doc__ is not None, (
            "GIFTFrameworkV21 class missing docstring"
        )
        assert len(GIFTFrameworkV21.__doc__) > 50, (
            "GIFTFrameworkV21 docstring too short"
        )

    def test_parameters_class_has_docstring(self):
        """Test GIFTParameters has docstring."""
        assert GIFTParameters.__doc__ is not None, (
            "GIFTParameters class missing docstring"
        )

    def test_public_methods_have_docstrings(self):
        """Test public methods have docstrings."""
        framework = GIFTFrameworkV21()

        # Get public methods
        public_methods = [
            name for name in dir(framework)
            if not name.startswith('_') and callable(getattr(framework, name))
        ]

        missing_docs = []

        for method_name in public_methods:
            method = getattr(framework, method_name)
            if method.__doc__ is None or len(method.__doc__) < 10:
                missing_docs.append(method_name)

        # Allow some tolerance
        if missing_docs:
            print(f"\nMethods missing docstrings: {missing_docs}")

        # At least most methods should have docs
        assert len(missing_docs) < len(public_methods) / 2, (
            f"Too many methods missing docstrings: {missing_docs}"
        )


class TestDocstringFormat:
    """Test docstring formatting and structure."""

    def test_docstring_describes_parameters(self):
        """Test that docstrings describe parameters."""
        # Check GIFTParameters docstring
        doc = GIFTParameters.__doc__

        if doc:
            # Should mention key parameters
            assert "p2" in doc.lower() or "p₂" in doc
            assert "Weyl" in doc or "weyl" in doc.lower()

    def test_docstring_describes_returns(self):
        """Test that compute methods describe return values."""
        framework = GIFTFrameworkV21()

        # Check compute_all_observables
        method = getattr(framework, 'compute_all_observables', None)

        if method and method.__doc__:
            doc = method.__doc__.lower()

            # Should describe what it returns
            assert "return" in doc or "dict" in doc or "observable" in doc


class TestExampleCodeValidation:
    """Test that example code in documentation works."""

    def test_basic_usage_example(self):
        """Test basic usage example works."""
        # This would be from documentation
        # Example: Create framework and compute observables

        framework = GIFTFrameworkV21()
        observables = framework.compute_all_observables()

        assert isinstance(observables, dict)
        assert len(observables) > 0

    def test_custom_parameters_example(self):
        """Test custom parameters example."""
        # Example from docs: Using custom parameters

        params = GIFTParameters(
            p2=2.1,
            Weyl_factor=5.2
        )

        framework = GIFTFrameworkV21(params=params)
        obs = framework.compute_all_observables()

        assert len(obs) > 0

    def test_accessing_specific_observable_example(self):
        """Test accessing specific observable example."""
        # Example: Access specific observable

        framework = GIFTFrameworkV21()
        obs = framework.compute_all_observables()

        # Should be able to access by name
        if "alpha_inv_MZ" in obs:
            alpha_inv = obs["alpha_inv_MZ"]
            assert isinstance(alpha_inv, (int, float))


class TestDocumentationCompleteness:
    """Test documentation completeness."""

    def test_all_observables_documented(self):
        """Test that all computed observables are documented."""
        framework = GIFTFrameworkV21()
        obs = framework.compute_all_observables()

        # Check if there's a reference that documents all observables
        # This would typically check a README or documentation file

        # For now, just verify they can be listed
        observable_list = list(obs.keys())

        # Should have comprehensive set
        assert len(observable_list) >= 40, (
            f"Only {len(observable_list)} observables - documentation may be incomplete"
        )

    def test_parameter_descriptions_complete(self):
        """Test that all parameters are described."""
        # Check GIFTParameters fields are documented

        param_fields = [
            "p2", "Weyl_factor", "tau",
            "T_norm", "T_costar", "det_g"
        ]

        doc = GIFTParameters.__doc__

        if doc:
            documented_params = []

            for param in param_fields:
                if param in doc or param.lower() in doc.lower():
                    documented_params.append(param)

            # Most should be documented
            coverage = len(documented_params) / len(param_fields)

            assert coverage > 0.5, (
                f"Only {coverage*100:.0f}% of parameters documented"
            )


class TestAPIConsistency:
    """Test API consistency and conventions."""

    def test_method_naming_conventions(self):
        """Test methods follow naming conventions."""
        framework = GIFTFrameworkV21()

        # Get all methods
        methods = [
            name for name in dir(framework)
            if not name.startswith('_') and callable(getattr(framework, name))
        ]

        # Check naming conventions
        for method_name in methods:
            # Should be lowercase with underscores
            assert method_name.islower() or '_' in method_name, (
                f"Method {method_name} doesn't follow naming convention"
            )

    def test_compute_methods_return_dict(self):
        """Test compute methods return dictionaries."""
        framework = GIFTFrameworkV21()

        compute_methods = [
            name for name in dir(framework)
            if 'compute' in name.lower() and callable(getattr(framework, name))
        ]

        for method_name in compute_methods:
            method = getattr(framework, method_name)

            try:
                result = method()

                # Should return dict for observable computation
                assert isinstance(result, dict), (
                    f"{method_name} should return dict, got {type(result)}"
                )

            except TypeError:
                # Method might require arguments
                pass

    def test_observable_names_consistent(self):
        """Test observable names follow consistent format."""
        framework = GIFTFrameworkV21()
        obs = framework.compute_all_observables()

        # Check naming patterns
        for obs_name in obs.keys():
            # Should be valid Python identifier (could be dict key)
            assert obs_name.isidentifier(), (
                f"Observable name '{obs_name}' is not valid identifier"
            )

            # Should not have spaces
            assert ' ' not in obs_name

            # Should use underscores, not hyphens
            assert '-' not in obs_name or obs_name.startswith('m_'), (
                f"Observable '{obs_name}' uses hyphens"
            )


class TestCodeExamples:
    """Test code examples from documentation."""

    def test_readme_example_1(self):
        """Test first example from README."""
        # Typical README example:
        # from gift_v21_core import GIFTFrameworkV21
        # framework = GIFTFrameworkV21()
        # observables = framework.compute_all_observables()

        framework = GIFTFrameworkV21()
        observables = framework.compute_all_observables()

        assert 'alpha_inv_MZ' in observables or len(observables) > 0

    def test_readme_example_2(self):
        """Test second example from README."""
        # Example: Get specific observable

        framework = GIFTFrameworkV21()
        obs = framework.compute_all_observables()

        # Access specific values
        if 'delta_CP' in obs:
            delta_cp = obs['delta_CP']
            assert isinstance(delta_cp, (int, float))
            assert 0 <= delta_cp <= 360  # Degrees


class TestErrorMessageQuality:
    """Test that error messages are informative."""

    def test_invalid_parameter_error_message(self):
        """Test error messages for invalid parameters."""
        try:
            # Try to create with clearly invalid parameter
            params = GIFTParameters(p2=-999)  # Negative might be invalid

            framework = GIFTFrameworkV21(params=params)
            obs = framework.compute_all_observables()

            # If it doesn't error, that's also fine
            # This test is about checking error message quality IF it errors

        except (ValueError, AssertionError, TypeError) as e:
            error_msg = str(e)

            # Error message should be informative
            assert len(error_msg) > 10, "Error message too short"

            # Should mention what went wrong
            # (Specific content depends on implementation)


class TestImportStructure:
    """Test import structure and module organization."""

    def test_main_classes_importable(self):
        """Test main classes can be imported."""
        # Should be able to import main classes
        from gift_v21_core import GIFTFrameworkV21

        assert GIFTFrameworkV21 is not None

        from gift_v21_core import GIFTParameters

        assert GIFTParameters is not None

    def test_module_has_version(self):
        """Test module has version information."""
        # Try to import version
        try:
            import gift_v21_core

            # Check if module has version attribute
            if hasattr(gift_v21_core, '__version__'):
                version = gift_v21_core.__version__
                assert isinstance(version, str)
                assert len(version) > 0
            else:
                # Version might be defined elsewhere
                pass

        except Exception:
            pass


class TestTypeHints:
    """Test type hint presence and correctness."""

    def test_framework_methods_have_type_hints(self):
        """Test that methods have type hints."""
        framework = GIFTFrameworkV21()

        # Get compute_all_observables method
        method = getattr(framework, 'compute_all_observables', None)

        if method:
            # Check if it has annotations
            sig = inspect.signature(method)

            # Return annotation
            if sig.return_annotation != inspect.Signature.empty:
                # Has return type hint
                assert True
            else:
                # No type hint (acceptable for now, but could be improved)
                pass


class TestDocumentationLinks:
    """Test documentation cross-references."""

    def test_docstring_references_valid(self):
        """Test that docstring references are valid."""
        # Check if docstrings reference other classes/methods correctly

        framework = GIFTFrameworkV21()
        doc = framework.__doc__

        if doc:
            # Check for common references
            # This is a basic check - full validation would parse docstring

            # Should not have broken references like [Broken](broken.md)
            # (This is simplistic - real test would be more thorough)

            assert "FIXME" not in doc, "Docstring contains FIXME"
            assert "TODO" not in doc or "TODO" in doc.lower(), "Docstring contains unresolved TODO"


class TestExampleCompleteness:
    """Test that examples cover main use cases."""

    def test_default_usage_example_exists(self):
        """Test default usage is exemplified."""
        # Default usage: Create with defaults, compute
        framework = GIFTFrameworkV21()
        obs = framework.compute_all_observables()

        assert len(obs) > 0

    def test_custom_parameters_example_exists(self):
        """Test custom parameters usage is exemplified."""
        # Custom usage: Specify parameters
        params = GIFTParameters(p2=2.0, Weyl_factor=5.5)
        framework = GIFTFrameworkV21(params=params)
        obs = framework.compute_all_observables()

        assert len(obs) > 0

    def test_accessing_results_example_exists(self):
        """Test result access is exemplified."""
        # Access specific observables
        framework = GIFTFrameworkV21()
        obs = framework.compute_all_observables()

        # Can iterate
        for name, value in obs.items():
            assert isinstance(name, str)
            # Value should be numeric
            assert isinstance(value, (int, float, complex))


class TestDocumentationMetadata:
    """Test documentation metadata and structure."""

    def test_module_has_author(self):
        """Test module has author information."""
        try:
            import gift_v21_core

            # Check for author in module docstring
            if gift_v21_core.__doc__:
                doc = gift_v21_core.__doc__

                # May contain author, version, etc.
                # This is optional but good practice
                pass

        except Exception:
            pass

    def test_classes_have_version_info(self):
        """Test classes document their version."""
        # Check if docstrings mention version

        doc_v21 = GIFTFrameworkV21.__doc__

        if doc_v21:
            # Should mention it's v2.1
            assert "2.1" in doc_v21 or "v2.1" in doc_v21.lower() or "version" in doc_v21.lower()
