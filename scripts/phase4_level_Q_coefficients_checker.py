#!/usr/bin/env python3
"""Independent lossless-projection checker for the Level Q coefficient package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PACKAGE_PATH = "certificates/phase4_level_Q_coefficients.json"
CHECK_PATH = "certificates/phase4_donaldson_coefficients_check.json"
VALUES_PATH = "certificates/phase4_donaldson_coefficients_values.json"
OUT_PATH = "certificates/phase4_level_Q_coefficients_check.json"


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def add(checks: list[dict[str, Any]], name: str, ok: bool, detail: str) -> None:
    checks.append({"name": name, "pass": bool(ok), "detail": detail})


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    package = read_json(repo_root / PACKAGE_PATH)
    check = read_json(repo_root / CHECK_PATH)
    values = read_json(repo_root / VALUES_PATH)
    intervals = check.get("outward_interval_recomputation", {})
    checks: list[dict[str, Any]] = []

    add(checks, "artifact_name", package.get("artifact") == "phase4_level_Q_coefficients", "artifact id matches")
    add(checks, "package_all_pass", package.get("all_pass") is True, "package all_pass is true")
    add(checks, "source_checker_matches", package.get("source_checker") == CHECK_PATH, "source checker path is canonical")
    add(checks, "source_values_matches", package.get("source_values") == VALUES_PATH, "source values path is canonical")
    add(checks, "source_checker_passes", check.get("all_pass") is True, "source checker passes")
    add(checks, "source_values_passes", values.get("all_pass") is True, "source values pass")
    add(
        checks,
        "checker_summary_copied",
        package.get("source_checker_summary") == check.get("summary"),
        "checker summary is copied exactly",
    )
    add(
        checks,
        "rounding_protocol_copied",
        package.get("rounding_protocol") == intervals.get("rounding_protocol"),
        "rounding protocol is copied exactly",
    )
    add(
        checks,
        "decimal_places_copied",
        package.get("decimal_places") == intervals.get("decimal_places"),
        "decimal precision is copied exactly",
    )

    for name, record in package.get("constants", {}).items():
        source_value = values["constants"].get(name)
        source_interval = intervals["constants"].get(name)
        add(checks, f"constant_{name}_exists_in_values", source_value is not None, f"{name} exists in values")
        add(checks, f"constant_{name}_exists_in_intervals", source_interval is not None, f"{name} exists in interval recomputation")
        if source_value is not None:
            add(checks, f"constant_{name}_exact_lossless", record.get("exact") == source_value.get("exact"), f"{name} exact copied")
            add(checks, f"constant_{name}_float_lossless", record.get("float_display") == source_value.get("value"), f"{name} display copied")
            add(checks, f"constant_{name}_status_lossless", record.get("status") == source_value.get("status"), f"{name} status copied")
            add(checks, f"constant_{name}_provenance_lossless", record.get("provenance") == source_value.get("provenance"), f"{name} provenance copied")
        if source_interval is not None:
            add(checks, f"constant_{name}_interval_lossless", record.get("interval") == source_interval, f"{name} interval copied")

    for name, record in package.get("coefficients", {}).items():
        source_value = values["evaluated_coefficients"].get(name)
        source_interval = intervals["evaluated_coefficients"].get(name)
        add(checks, f"coefficient_{name}_exists_in_values", source_value is not None, f"{name} exists in values")
        add(checks, f"coefficient_{name}_exists_in_intervals", source_interval is not None, f"{name} exists in interval recomputation")
        if source_value is not None:
            add(checks, f"coefficient_{name}_exact_lossless", record.get("exact") == source_value.get("exact"), f"{name} exact copied")
            add(checks, f"coefficient_{name}_float_lossless", record.get("float_display") == source_value.get("value"), f"{name} display copied")
            add(checks, f"coefficient_{name}_status_lossless", record.get("status") == source_value.get("status"), f"{name} status copied")
            add(checks, f"coefficient_{name}_provenance_lossless", record.get("provenance") == source_value.get("provenance"), f"{name} provenance copied")
        if source_interval is not None:
            add(checks, f"coefficient_{name}_interval_lossless", record.get("interval") == source_interval, f"{name} interval copied")

    r_pkg = package.get("R_threshold", {})
    r_value = values["evaluated_coefficients"].get("R_threshold", {})
    add(checks, "R_threshold_float_lossless", r_pkg.get("float_display") == r_value.get("value"), "R_threshold float display copied")
    add(checks, "R_threshold_interval_lossless", r_pkg.get("interval") == intervals.get("R_threshold"), "R_threshold interval copied")
    add(checks, "R_threshold_semantics_lossless", r_pkg.get("semantics") == r_value.get("semantics"), "R_threshold semantics copied")
    add(
        checks,
        "claim_boundary_level_Q",
        any("Level Q" in line for line in package.get("claim_boundary", [])),
        "claim boundary names Level Q",
    )

    all_pass = all(item["pass"] for item in checks)
    payload = {
        "artifact": "phase4_level_Q_coefficients_check",
        "generated_by": "scripts/phase4_level_Q_coefficients_checker.py",
        "scope": "Lossless projection checker for phase4_level_Q_coefficients.json",
        "checked_artifact": PACKAGE_PATH,
        "checks": checks,
        "summary": {
            "checks_total": len(checks),
            "checks_passed": sum(1 for item in checks if item["pass"]),
            "checks_failed": sum(1 for item in checks if not item["pass"]),
        },
        "all_pass": all_pass,
    }

    out_path = repo_root / OUT_PATH
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    if not all_pass:
        raise SystemExit(f"checker failed; wrote {OUT_PATH}")
    print(f"wrote {out_path.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
