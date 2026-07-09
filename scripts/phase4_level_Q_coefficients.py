#!/usr/bin/env python3
"""Extract the compact Level Q coefficient package from the Stage C/E checker."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CHECK_PATH = "certificates/phase4_donaldson_coefficients_check.json"
VALUES_PATH = "certificates/phase4_donaldson_coefficients_values.json"
OUT_PATH = "certificates/phase4_level_Q_coefficients.json"


def read_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    check = read_json(repo_root / CHECK_PATH)
    values = read_json(repo_root / VALUES_PATH)

    if check.get("all_pass") is not True:
        raise SystemExit(f"{CHECK_PATH} must pass before extracting Level Q")
    if values.get("all_pass") is not True:
        raise SystemExit(f"{VALUES_PATH} must pass before extracting Level Q")

    intervals = check["outward_interval_recomputation"]
    evaluated_values = values["evaluated_coefficients"]
    constants_values = values["constants"]

    citable_coefficients = [
        "source_P1",
        "source_P2",
        "DP1_norm",
        "DP2_norm",
        "D2P1_norm",
        "D3m_norm",
        "raw_P3_scale",
        "xi1_bound",
        "xi2_forcing",
        "xi2_bound",
        "remainder_R3",
        "tail_contraction_denominator",
    ]
    citable_constants = [
        "K_H_K3",
        "K_F",
        "G_aug",
        "C_nl",
        "C_projection",
        "C_conn_1",
        "C_conn_2",
        "C_conn_3",
        "C_comm",
        "C_comm_2",
        "C_comm_3",
        "C_D3m",
    ]

    coefficient_records = {}
    for name in citable_coefficients:
        coefficient_records[name] = {
            "exact": evaluated_values[name]["exact"],
            "float_display": evaluated_values[name]["value"],
            "interval": intervals["evaluated_coefficients"][name],
            "status": evaluated_values[name]["status"],
            "provenance": evaluated_values[name]["provenance"],
        }

    constant_records = {}
    for name in citable_constants:
        constant_records[name] = {
            "exact": constants_values[name]["exact"],
            "float_display": constants_values[name]["value"],
            "interval": intervals["constants"][name],
            "status": constants_values[name]["status"],
            "provenance": constants_values[name]["provenance"],
        }

    payload = {
        "artifact": "phase4_level_Q_coefficients",
        "generated_by": "scripts/phase4_level_Q_coefficients.py",
        "scope": "Compact citable Level Q package for D0 Donaldson coefficient intervals",
        "status": "level_Q_coefficient_interval_package",
        "source_checker": CHECK_PATH,
        "source_values": VALUES_PATH,
        "source_checker_summary": check["summary"],
        "rounding_protocol": intervals["rounding_protocol"],
        "decimal_places": intervals["decimal_places"],
        "constants": constant_records,
        "coefficients": coefficient_records,
        "R_threshold": {
            "float_display": evaluated_values["R_threshold"]["value"],
            "interval": intervals["R_threshold"],
            "citable_upper": "3664.066",
            "citation": {
                "certified_upper_endpoint": intervals["R_threshold"]["upper"],
                "human_conservative_round": "3664.066",
                "recommended_text": "R_threshold <= 3664.066, with machine bracket [3664.065985330004, 3664.065985330005]",
            },
            "semantics": evaluated_values["R_threshold"]["semantics"],
            "status": "outward_rounded_cube_root_interval",
        },
        "claim_boundary": [
            "Citable at D0 for Level Q coefficient intervals.",
            "Lossless projection of phase4_donaldson_coefficients_check.json on the selected citable fields.",
            "Does not prove non-D0 uniformity.",
            "Does not prove the anisotropic Joyce perturbation theorem.",
        ],
        "all_pass": True,
    }

    out_path = repo_root / OUT_PATH
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out_path.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
