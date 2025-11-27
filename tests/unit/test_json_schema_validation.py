"""
Tests for JSON Schema Validation

These tests ensure JSON outputs from various components conform to expected schemas.
This prevents silent format changes that could break downstream consumers.

Covers:
- Agent report output format
- Observable validation output format
- Training history format
- Checkpoint metadata format
- Experimental data format

Author: GIFT Framework
Date: 2025-11-27
"""

import pytest
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import sys

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "statistical_validation"))


# =============================================================================
# Schema Definitions
# =============================================================================

@dataclass
class FieldSpec:
    """Specification for a JSON field."""
    name: str
    field_type: type
    required: bool = True
    nullable: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    min_length: Optional[int] = None
    allowed_values: Optional[List[Any]] = None


def validate_field(data: Dict, spec: FieldSpec) -> List[str]:
    """Validate a single field against its specification."""
    errors = []

    if spec.name not in data:
        if spec.required:
            errors.append(f"Missing required field: {spec.name}")
        return errors

    value = data[spec.name]

    if value is None:
        if not spec.nullable:
            errors.append(f"Field {spec.name} is null but not nullable")
        return errors

    if not isinstance(value, spec.field_type):
        # Handle numeric type flexibility
        if spec.field_type in (int, float) and isinstance(value, (int, float)):
            pass
        else:
            errors.append(f"Field {spec.name} has wrong type: expected {spec.field_type.__name__}, got {type(value).__name__}")
            return errors

    if spec.min_value is not None and isinstance(value, (int, float)):
        if value < spec.min_value:
            errors.append(f"Field {spec.name} value {value} below minimum {spec.min_value}")

    if spec.max_value is not None and isinstance(value, (int, float)):
        if value > spec.max_value:
            errors.append(f"Field {spec.name} value {value} above maximum {spec.max_value}")

    if spec.min_length is not None and hasattr(value, '__len__'):
        if len(value) < spec.min_length:
            errors.append(f"Field {spec.name} length {len(value)} below minimum {spec.min_length}")

    if spec.allowed_values is not None:
        if value not in spec.allowed_values:
            errors.append(f"Field {spec.name} value {value} not in allowed values {spec.allowed_values}")

    return errors


def validate_schema(data: Dict, specs: List[FieldSpec]) -> List[str]:
    """Validate data against a list of field specifications."""
    all_errors = []
    for spec in specs:
        errors = validate_field(data, spec)
        all_errors.extend(errors)
    return all_errors


# =============================================================================
# Agent Report Schema
# =============================================================================

AGENT_REPORT_SCHEMA = [
    FieldSpec("name", str, required=True),
    FieldSpec("ok", bool, required=True),
    FieldSpec("summary", str, required=True, min_length=1),
    FieldSpec("issues", list, required=True),
    FieldSpec("timestamp", str, required=True),
    FieldSpec("hostname", str, required=True),
]


class TestAgentReportSchema:
    """Tests for agent report JSON schema."""

    def test_valid_agent_report(self):
        """Test valid agent report passes validation."""
        report = {
            "name": "verify",
            "ok": True,
            "summary": "Verification completed successfully",
            "issues": [],
            "timestamp": "20251127_120000",
            "hostname": "test-host"
        }

        errors = validate_schema(report, AGENT_REPORT_SCHEMA)
        assert len(errors) == 0

    def test_agent_report_missing_name(self):
        """Test missing name field is caught."""
        report = {
            "ok": True,
            "summary": "Test",
            "issues": [],
            "timestamp": "20251127_120000",
            "hostname": "test-host"
        }

        errors = validate_schema(report, AGENT_REPORT_SCHEMA)
        assert any("name" in e for e in errors)

    def test_agent_report_wrong_type(self):
        """Test wrong type is caught."""
        report = {
            "name": "verify",
            "ok": "true",  # Should be bool
            "summary": "Test",
            "issues": [],
            "timestamp": "20251127_120000",
            "hostname": "test-host"
        }

        errors = validate_schema(report, AGENT_REPORT_SCHEMA)
        assert any("ok" in e and "type" in e for e in errors)

    def test_agent_report_with_issues(self):
        """Test agent report with issues is valid."""
        report = {
            "name": "verify",
            "ok": False,
            "summary": "2 issues found",
            "issues": [
                {"file": "test.md", "error": "Missing header"},
                {"file": "other.md", "error": "Broken link"}
            ],
            "timestamp": "20251127_120000",
            "hostname": "test-host"
        }

        errors = validate_schema(report, AGENT_REPORT_SCHEMA)
        assert len(errors) == 0


# =============================================================================
# Observable Data Schema
# =============================================================================

OBSERVABLE_SCHEMA = [
    FieldSpec("value", float, required=True),
    FieldSpec("uncertainty", float, required=False, min_value=0),
    FieldSpec("experimental", float, required=False),
    FieldSpec("deviation", float, required=False),
    FieldSpec("status", str, required=False, allowed_values=[
        "PROVEN", "TOPOLOGICAL", "DERIVED", "THEORETICAL", "PHENOMENOLOGICAL", "EXPLORATORY"
    ]),
]


class TestObservableSchema:
    """Tests for observable data JSON schema."""

    def test_valid_observable(self):
        """Test valid observable passes validation."""
        observable = {
            "value": 137.036,
            "uncertainty": 0.001,
            "experimental": 137.035999,
            "deviation": 0.0001,
            "status": "TOPOLOGICAL"
        }

        errors = validate_schema(observable, OBSERVABLE_SCHEMA)
        assert len(errors) == 0

    def test_observable_minimal(self):
        """Test minimal observable (only value) passes."""
        observable = {
            "value": 0.2312
        }

        errors = validate_schema(observable, OBSERVABLE_SCHEMA)
        assert len(errors) == 0

    def test_observable_negative_uncertainty(self):
        """Test negative uncertainty is caught."""
        observable = {
            "value": 137.036,
            "uncertainty": -0.001
        }

        errors = validate_schema(observable, OBSERVABLE_SCHEMA)
        assert any("uncertainty" in e for e in errors)

    def test_observable_invalid_status(self):
        """Test invalid status is caught."""
        observable = {
            "value": 137.036,
            "status": "INVALID_STATUS"
        }

        errors = validate_schema(observable, OBSERVABLE_SCHEMA)
        assert any("status" in e for e in errors)


# =============================================================================
# Validation Results Schema
# =============================================================================

VALIDATION_RESULT_SCHEMA = [
    FieldSpec("observable_name", str, required=True),
    FieldSpec("predicted", float, required=True),
    FieldSpec("experimental", float, required=True),
    FieldSpec("absolute_deviation", float, required=True, min_value=0),
    FieldSpec("relative_deviation", float, required=True, min_value=0),
    FieldSpec("within_uncertainty", bool, required=False),
]


class TestValidationResultSchema:
    """Tests for validation result JSON schema."""

    def test_valid_validation_result(self):
        """Test valid validation result passes."""
        result = {
            "observable_name": "alpha_inv",
            "predicted": 137.036,
            "experimental": 137.035999,
            "absolute_deviation": 0.000001,
            "relative_deviation": 0.0000001,
            "within_uncertainty": True
        }

        errors = validate_schema(result, VALIDATION_RESULT_SCHEMA)
        assert len(errors) == 0

    def test_validation_result_missing_required(self):
        """Test missing required fields are caught."""
        result = {
            "observable_name": "alpha_inv",
            "predicted": 137.036
            # Missing experimental and deviations
        }

        errors = validate_schema(result, VALIDATION_RESULT_SCHEMA)
        assert len(errors) > 0


# =============================================================================
# Training History Schema
# =============================================================================

TRAINING_HISTORY_SCHEMA = [
    FieldSpec("epoch", list, required=True, min_length=1),
    FieldSpec("loss", list, required=True, min_length=1),
]


class TestTrainingHistorySchema:
    """Tests for training history JSON schema."""

    def test_valid_training_history(self):
        """Test valid training history passes."""
        history = {
            "epoch": [0, 1, 2, 3, 4],
            "loss": [1.0, 0.5, 0.25, 0.1, 0.05],
            "torsion": [0.1, 0.05, 0.02, 0.01, 0.005],
            "volume": [0.5, 0.3, 0.2, 0.1, 0.05]
        }

        errors = validate_schema(history, TRAINING_HISTORY_SCHEMA)
        assert len(errors) == 0

    def test_training_history_empty_epoch(self):
        """Test empty epoch list is caught."""
        history = {
            "epoch": [],
            "loss": []
        }

        errors = validate_schema(history, TRAINING_HISTORY_SCHEMA)
        assert any("epoch" in e for e in errors)

    def test_training_history_json_serializable(self):
        """Test training history can be JSON serialized."""
        history = {
            "epoch": list(range(100)),
            "loss": [0.1 / (i + 1) for i in range(100)],
        }

        # Should not raise
        json_str = json.dumps(history)
        parsed = json.loads(json_str)

        assert parsed["epoch"] == history["epoch"]


# =============================================================================
# Experimental Data Schema
# =============================================================================

EXPERIMENTAL_DATA_ITEM_SCHEMA = [
    FieldSpec("central_value", float, required=True),
    FieldSpec("uncertainty", float, required=True, min_value=0),
]


class TestExperimentalDataSchema:
    """Tests for experimental data JSON schema."""

    def test_valid_experimental_data_item(self):
        """Test valid experimental data item passes."""
        item = {
            "central_value": 137.035999,
            "uncertainty": 0.000001
        }

        errors = validate_schema(item, EXPERIMENTAL_DATA_ITEM_SCHEMA)
        assert len(errors) == 0

    def test_experimental_data_from_framework(self):
        """Test experimental data from GIFTFrameworkV21 is valid."""
        from run_validation_v21 import GIFTFrameworkV21

        fw = GIFTFrameworkV21()

        for name, (value, uncertainty) in fw.experimental_data.items():
            item = {
                "central_value": value,
                "uncertainty": uncertainty
            }
            errors = validate_schema(item, EXPERIMENTAL_DATA_ITEM_SCHEMA)
            assert len(errors) == 0, f"Invalid experimental data for {name}: {errors}"


# =============================================================================
# Checkpoint Metadata Schema
# =============================================================================

CHECKPOINT_METADATA_SCHEMA = [
    FieldSpec("epoch", int, required=True, min_value=0),
    FieldSpec("loss", float, required=False),
]


class TestCheckpointMetadataSchema:
    """Tests for checkpoint metadata JSON schema."""

    def test_valid_checkpoint_metadata(self):
        """Test valid checkpoint metadata passes."""
        metadata = {
            "epoch": 1000,
            "loss": 0.001
        }

        errors = validate_schema(metadata, CHECKPOINT_METADATA_SCHEMA)
        assert len(errors) == 0

    def test_checkpoint_negative_epoch(self):
        """Test negative epoch is caught."""
        metadata = {
            "epoch": -1,
            "loss": 0.001
        }

        errors = validate_schema(metadata, CHECKPOINT_METADATA_SCHEMA)
        assert any("epoch" in e for e in errors)


# =============================================================================
# Full Validation Output Schema
# =============================================================================

class TestFullValidationOutput:
    """Tests for complete validation output structure."""

    def test_create_valid_output(self):
        """Test creating a complete valid output structure."""
        output = {
            "metadata": {
                "version": "2.1.0",
                "timestamp": "2025-11-27T12:00:00",
                "n_samples": 100000
            },
            "observables": {
                "alpha_inv": {
                    "predicted": 137.036,
                    "experimental": 137.035999,
                    "deviation_percent": 0.0001,
                    "status": "TOPOLOGICAL"
                },
                "sin2_theta_W": {
                    "predicted": 0.2312,
                    "experimental": 0.23122,
                    "deviation_percent": 0.01,
                    "status": "PROVEN"
                }
            },
            "summary": {
                "total_observables": 39,
                "mean_deviation_percent": 0.128,
                "max_deviation_percent": 0.98,
                "proven_exact": 13
            }
        }

        # Should be JSON serializable
        json_str = json.dumps(output)
        parsed = json.loads(json_str)

        assert parsed["metadata"]["version"] == "2.1.0"
        assert len(parsed["observables"]) == 2
        assert parsed["summary"]["proven_exact"] == 13

    def test_observable_values_finite(self):
        """Test all observable values are finite."""
        from run_validation_v21 import GIFTFrameworkV21

        fw = GIFTFrameworkV21()

        # Check computed observables
        alpha_inv = fw.compute_alpha_inverse()
        sin2_theta_W = fw.compute_sin2_theta_W()
        alpha_s = fw.compute_alpha_s()

        assert np.isfinite(alpha_inv)
        assert np.isfinite(sin2_theta_W)
        assert np.isfinite(alpha_s)


# =============================================================================
# JSON Serialization Edge Cases
# =============================================================================

class TestJSONSerializationEdgeCases:
    """Tests for JSON serialization edge cases."""

    def test_numpy_float_serialization(self):
        """Test numpy floats can be serialized."""
        data = {
            "value": np.float64(137.036),
            "uncertainty": np.float32(0.001)
        }

        # Convert numpy types to Python types
        serializable = {k: float(v) for k, v in data.items()}

        json_str = json.dumps(serializable)
        parsed = json.loads(json_str)

        assert abs(parsed["value"] - 137.036) < 0.001

    def test_numpy_array_serialization(self):
        """Test numpy arrays can be serialized."""
        data = {
            "values": np.array([1.0, 2.0, 3.0]),
            "counts": np.array([1, 2, 3])
        }

        # Convert arrays to lists
        serializable = {k: v.tolist() for k, v in data.items()}

        json_str = json.dumps(serializable)
        parsed = json.loads(json_str)

        assert parsed["values"] == [1.0, 2.0, 3.0]
        assert parsed["counts"] == [1, 2, 3]

    def test_nan_handling(self):
        """Test NaN values are handled appropriately."""
        data = {
            "value": float('nan'),
            "other": 1.0
        }

        # Python's json module allows NaN by default (non-standard JSON)
        # For strict JSON compliance, use allow_nan=False and handle separately
        try:
            # This will work in Python's json but produce non-standard JSON
            json_str = json.dumps(data)
            parsed = json.loads(json_str)
            # NaN is preserved as NaN (not None)
            assert np.isnan(parsed["value"])
            assert parsed["other"] == 1.0
        except ValueError:
            # If allow_nan=False were set, we'd get a ValueError
            pass

        # Test explicit NaN replacement for strict JSON
        clean_data = {k: (None if isinstance(v, float) and np.isnan(v) else v)
                      for k, v in data.items()}
        json_str = json.dumps(clean_data)
        parsed = json.loads(json_str)
        assert parsed["value"] is None

    def test_inf_handling(self):
        """Test Inf values are handled appropriately."""
        data = {
            "pos_inf": float('inf'),
            "neg_inf": float('-inf'),
            "normal": 1.0
        }

        # Python's json module allows Inf by default (non-standard JSON)
        # For strict JSON compliance, use allow_nan=False (also affects Inf)
        try:
            json_str = json.dumps(data)
            parsed = json.loads(json_str)
            # Inf is preserved as Inf
            assert np.isinf(parsed["pos_inf"]) and parsed["pos_inf"] > 0
            assert np.isinf(parsed["neg_inf"]) and parsed["neg_inf"] < 0
            assert parsed["normal"] == 1.0
        except ValueError:
            pass

        # Test explicit Inf replacement for strict JSON
        def replace_inf(v):
            if isinstance(v, float):
                if np.isinf(v):
                    return "Infinity" if v > 0 else "-Infinity"
            return v

        clean_data = {k: replace_inf(v) for k, v in data.items()}
        json_str = json.dumps(clean_data)
        parsed = json.loads(json_str)

        assert parsed["pos_inf"] == "Infinity"
        assert parsed["neg_inf"] == "-Infinity"

    def test_deep_nesting(self):
        """Test deeply nested structures serialize correctly."""
        data = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "value": 42
                        }
                    }
                }
            }
        }

        json_str = json.dumps(data)
        parsed = json.loads(json_str)

        assert parsed["level1"]["level2"]["level3"]["level4"]["value"] == 42


# =============================================================================
# Schema Version Compatibility
# =============================================================================

class TestSchemaVersionCompatibility:
    """Tests for schema version compatibility."""

    def test_v21_output_backward_compatible(self):
        """Test v2.1 output is backward compatible with v2.0 readers."""
        # v2.0 expected fields
        v20_required = ["observables", "summary"]

        # v2.1 output
        v21_output = {
            "metadata": {"version": "2.1.0"},
            "observables": {"alpha_inv": {"value": 137.036}},
            "summary": {"mean_deviation": 0.128},
            "new_v21_field": "some data"  # New field
        }

        # v2.0 reader should still find required fields
        for field in v20_required:
            assert field in v21_output

    def test_missing_optional_fields_handled(self):
        """Test missing optional fields don't break validation."""
        # Minimal valid output
        minimal = {
            "observables": {},
            "summary": {}
        }

        # Should be valid JSON
        json_str = json.dumps(minimal)
        parsed = json.loads(json_str)

        assert "observables" in parsed
        assert "summary" in parsed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
