#!/usr/bin/env python3
"""Checker for the remaining P4.2 AR Neumann-slot candidates."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any


CANDIDATE_PATH = "certificates/phase4_ar_remaining_slots_candidate.json"
PRODUCT_SPACE_PATH = "certificates/phase4_ar_product_space_contract.json"
NEUMANN_BUDGET_PATH = "certificates/phase4_ar_neumann_budget_candidate.json"
OUT_PATH = "certificates/phase4_ar_remaining_slots_check.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def add(checks: list[dict[str, Any]], name: str, ok: bool, detail: str) -> None:
    checks.append({"name": name, "pass": bool(ok), "detail": detail})


def frac(obj: dict[str, Any]) -> Fraction:
    return Fraction(obj["exact"])


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    candidate = read_json(repo_root / CANDIDATE_PATH)
    product = read_json(repo_root / PRODUCT_SPACE_PATH)
    budget = read_json(repo_root / NEUMANN_BUDGET_PATH)
    checks: list[dict[str, Any]] = []

    add(checks, "artifact_name", candidate.get("artifact") == "phase4_ar_remaining_slots_candidate", "artifact id matches")
    add(checks, "status_candidate", candidate.get("status") == "candidate_formulas_not_certificates", "status remains candidate")
    for dep in [PRODUCT_SPACE_PATH, NEUMANN_BUDGET_PATH]:
        add(checks, f"depends_on_{Path(dep).stem}", dep in candidate.get("depends_on", []), f"depends on {dep}")

    K_H_K3 = Fraction(product["operators"]["G_f"]["bound"]["exact"])
    K_AR_prod = Fraction(product["candidate_product_inverse_bound"]["K_AR_prod"]["exact"])
    K_projection = Fraction(product["operators"]["Pi_reduced"]["bound"]["exact"])
    eps_AR = Fraction(budget["known_inputs"]["eps_AR"]["exact"])

    inputs = candidate["inputs"]
    add(checks, "K_H_matches_product", frac(inputs["K_H_K3"]) == K_H_K3, "K_H_K3 matches product-space contract")
    add(checks, "K_AR_matches_product", frac(inputs["K_AR_prod"]) == K_AR_prod, "K_AR_prod matches product-space contract")
    add(checks, "K_projection_matches_product", frac(inputs["K_projection"]) == K_projection, "K_projection matches product-space contract")
    add(checks, "eps_matches_budget", frac(inputs["eps_AR"]) == eps_AR, "eps_AR matches Neumann budget")

    expected = {
        "q_hodge_uniform": 16 * (1 + K_H_K3),
        "q_projection_residual": 16 * (1 + K_projection) * (1 + K_AR_prod),
        "q_gauge_transfer": 12 * (1 + K_AR_prod),
    }
    candidates = candidate.get("candidates", {})
    add(checks, "candidates_present", set(candidates) == set(expected), "all remaining slot candidates are present")

    for slot_id, constant in expected.items():
        slot_budget = budget["neumann_slots"][slot_id]
        target = Fraction(slot_budget["target_upper_bound"]["exact"])
        item = candidates[slot_id]
        values = item["candidate_values"]
        value = constant * eps_AR
        add(checks, f"{slot_id}_status_candidate", item.get("status") == "candidate_formula_not_certificate", f"{slot_id} remains candidate")
        add(checks, f"{slot_id}_constant_recomputes", frac(values[f"C_{slot_id}"]) == constant, f"{slot_id} constant recomputes")
        add(checks, f"{slot_id}_value_recomputes", frac(values[slot_id]) == value, f"{slot_id} value recomputes")
        add(checks, f"{slot_id}_margin_recomputes", frac(values["target_margin"]) == target - value, f"{slot_id} margin recomputes")
        add(checks, f"{slot_id}_below_target", value <= target, f"{slot_id} candidate is below target")
        add(checks, f"{slot_id}_constant_below_250", constant <= 250, f"{slot_id} constant is below 250")
        add(checks, f"{slot_id}_budget_still_pending", slot_budget["status"] == "pending_exact_certificate", f"{slot_id} budget slot remains pending")

    add(checks, "producer_checks_all_true", all(candidate.get("checks", {}).values()), "producer checks are true")
    add(
        checks,
        "boundary_does_not_close_slots",
        any("do not close" in line for line in candidate.get("acceptance_boundary", [])),
        "acceptance boundary prevents slot closure",
    )

    payload = {
        "artifact": "phase4_ar_remaining_slots_check",
        "generated_by": "scripts/phase4_ar_remaining_slots_checker.py",
        "scope": "Independent checker for remaining P4.2 AR Neumann-slot candidates",
        "status": "checker_for_candidates_not_AR_theorem",
        "checked_artifact": CANDIDATE_PATH,
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
