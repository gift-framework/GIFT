#!/usr/bin/env python3
"""Independent checker for the Phase 4.2 AR product-space contract."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any


CONTRACT_PATH = "certificates/phase4_ar_product_space_contract.json"
TYPE_GATE_PATH = "certificates/phase4_bigraded_type_check.json"
VALUES_PATH = "certificates/phase4_donaldson_coefficients_values.json"
OUT_PATH = "certificates/phase4_ar_product_space_check.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def add(checks: list[dict[str, Any]], name: str, ok: bool, detail: str) -> None:
    checks.append({"name": name, "pass": bool(ok), "detail": detail})


def shift(degree: list[int], rule: list[int]) -> list[int]:
    return [degree[0] + rule[0], degree[1] + rule[1]]


def frac(obj: dict[str, Any]) -> Fraction:
    return Fraction(obj["exact"])


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    contract = read_json(repo_root / CONTRACT_PATH)
    type_gate = read_json(repo_root / TYPE_GATE_PATH)
    values = read_json(repo_root / VALUES_PATH)
    checks: list[dict[str, Any]] = []

    add(checks, "artifact_name", contract.get("artifact") == "phase4_ar_product_space_contract", "artifact id matches")
    add(
        checks,
        "status_keeps_obligations_visible",
        contract.get("status") == "definition_with_explicit_unproved_inverse_obligations",
        "contract status is definition with obligations",
    )
    add(checks, "type_gate_passes", type_gate.get("all_pass") is True, "bigraded type gate passes")
    add(checks, "values_pass", values.get("all_pass") is True, "P4.1 values artifact passes")
    add(checks, "depends_on_values", VALUES_PATH in contract.get("depends_on", []), "contract depends on values artifact")

    product = contract.get("product_space", {})
    components = product.get("components", {})
    expected_components = {
        "omega": [1, 2],
        "lambda": [3, 0],
        "mu": [0, 4],
        "Theta": [2, 2],
    }
    add(checks, "product_space_name", product.get("name") == "A_beta", "product space is named A_beta")
    add(checks, "components_present", set(components) == set(expected_components), "all four components are present")
    add(
        checks,
        "norm_no_hidden_O1",
        product.get("norm", {}).get("no_hidden_O1_constants") is True,
        "product norm explicitly forbids hidden O(1) constants",
    )
    for name, degree in expected_components.items():
        add(checks, f"{name}_bidegree", components.get(name, {}).get("bidegree") == degree, f"{name} bidegree matches")
        add(checks, f"{name}_has_gauge", bool(components.get(name, {}).get("gauge")), f"{name} has a gauge field")
        add(checks, f"{name}_has_norm", bool(components.get(name, {}).get("norm")), f"{name} has a norm field")

    operators = contract.get("operators", {})
    f_h = operators.get("F_H", {})
    g_f = operators.get("G_f", {})
    pi = operators.get("Pi_reduced", {})
    add(checks, "F_H_degree_rule", f_h.get("degree_rule") == [2, -1], "F_H has degree shift (2,-1)")
    add(checks, "G_f_degree_rule", g_f.get("degree_rule") == [0, -1], "G_f has degree shift (0,-1)")
    add(checks, "Pi_reduced_bound_one", frac(pi.get("bound", {})) == 1, "Pi_reduced bound is one")

    constants = values.get("constants", {})
    expected_kh = Fraction(constants["K_H_K3"]["exact"])
    expected_kf = Fraction(constants["K_F"]["exact"])
    add(checks, "K_H_K3_matches_values", frac(g_f["bound"]) == expected_kh, "G_f bound matches P4.1 values")
    add(checks, "K_F_matches_values", frac(f_h["bound"]) == expected_kf, "F_H bound matches P4.1 values")

    blocks = contract.get("block_inverse_architecture", {})
    lam_block = blocks.get("lambda_from_omega", {})
    theta_block = blocks.get("Theta_from_mu", {})
    reduced_block = blocks.get("reduced_projection", {})
    add(
        checks,
        "lambda_block_degree",
        shift(expected_components["omega"], f_h["degree_rule"]) == lam_block.get("input_bidegree")
        and shift(lam_block.get("input_bidegree", [0, 0]), g_f["degree_rule"]) == lam_block.get("output_bidegree"),
        "lambda block degrees compose as omega -> F_H omega -> G_f F_H omega",
    )
    add(
        checks,
        "theta_block_degree",
        shift(expected_components["mu"], f_h["degree_rule"]) == theta_block.get("input_bidegree")
        and shift(theta_block.get("input_bidegree", [0, 0]), g_f["degree_rule"]) == theta_block.get("output_bidegree"),
        "Theta block degrees compose as mu -> F_H mu -> G_f F_H mu",
    )
    add(
        checks,
        "reduced_block_degree",
        reduced_block.get("input_bidegree") == [3, 2] and reduced_block.get("output_bidegree") == [3, 2],
        "reduced projection stays in M_eps bidegree",
    )

    candidate = contract.get("candidate_product_inverse_bound", {}).get("K_AR_prod", {})
    add(checks, "K_AR_prod_candidate_matches_block", frac(candidate) == expected_kh * expected_kf, "K_AR_prod candidate equals K_H_K3*K_F")
    add(
        checks,
        "K_AR_prod_structural_status",
        candidate.get("status") == "structural_product_max_bound",
        "K_AR_prod is structural product-max bound",
    )

    obligations = contract.get("theorem_obligations", {})
    add(checks, "product_space_defined", obligations.get("product_space_defined") is True, "product space is marked defined")
    for name in [
        "uniform_fibrewise_inverse_proved",
        "commutators_bounded_in_product_norm",
        "reduced_projection_global_identity_proved",
        "closedness_preservation_proved",
    ]:
        add(checks, f"{name}_open", obligations.get(name) is False, f"{name} remains open")
    add(
        checks,
        "boundary_prevents_AR_discharge",
        any("does not discharge AR" in line for line in contract.get("acceptance_boundary", [])),
        "acceptance boundary prevents AR promotion",
    )

    payload = {
        "artifact": "phase4_ar_product_space_check",
        "generated_by": "scripts/phase4_ar_product_space_checker.py",
        "scope": "Independent checker for the Phase 4.2 product-space contract",
        "status": "checker_for_definition_not_AR_theorem",
        "checked_artifact": CONTRACT_PATH,
        "checks": checks,
        "summary": {
            "checks_total": len(checks),
            "checks_passed": sum(1 for c in checks if c["pass"]),
            "checks_failed": sum(1 for c in checks if not c["pass"]),
        },
        "all_pass": all(c["pass"] for c in checks),
    }

    out = repo_root / OUT_PATH
    out.write_text(json.dumps(payload, indent=2) + "\n")
    if not payload["all_pass"]:
        raise SystemExit(f"checker failed; wrote {OUT_PATH}")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
