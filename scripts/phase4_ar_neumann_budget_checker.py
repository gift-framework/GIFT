#!/usr/bin/env python3
"""Checker for the Phase 4.2 AR Neumann-budget skeleton."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any


BUDGET_PATH = "certificates/phase4_ar_neumann_budget_candidate.json"
PRODUCT_SPACE_PATH = "certificates/phase4_ar_product_space_contract.json"
MAJORANT_PATH = "certificates/phase4_ar_majorant_candidate.json"
INVERSE_AUDIT_PATH = "certificates/phase4_ar_inverse_budget_audit.json"
OUT_PATH = "certificates/phase4_ar_neumann_budget_check.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def add(checks: list[dict[str, Any]], name: str, ok: bool, detail: str) -> None:
    checks.append({"name": name, "pass": bool(ok), "detail": detail})


def frac_record(obj: dict[str, Any]) -> Fraction:
    return Fraction(obj["exact"])


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    budget = read_json(repo_root / BUDGET_PATH)
    product = read_json(repo_root / PRODUCT_SPACE_PATH)
    majorant = read_json(repo_root / MAJORANT_PATH)
    inverse_audit = read_json(repo_root / INVERSE_AUDIT_PATH)
    checks: list[dict[str, Any]] = []

    add(checks, "artifact_name", budget.get("artifact") == "phase4_ar_neumann_budget_candidate", "artifact id matches")
    add(
        checks,
        "status_pending",
        budget.get("status") == "budget_skeleton_pending_exact_certificates",
        "budget is marked pending certificates",
    )
    for dep in [PRODUCT_SPACE_PATH, MAJORANT_PATH, INVERSE_AUDIT_PATH]:
        add(checks, f"depends_on_{Path(dep).stem}", dep in budget.get("depends_on", []), f"depends on {dep}")

    known = budget.get("known_inputs", {})
    add(
        checks,
        "K_AR_matches_majorant",
        frac_record(known["K_AR_prod"]) == Fraction(majorant["conditional_assumptions"]["K_AR_prod"]["exact"]),
        "K_AR_prod matches majorant",
    )
    add(
        checks,
        "q_scalar_matches_majorant",
        frac_record(known["q_AR_scalar"]) == Fraction(majorant["evaluated_at_eps_AR"]["q_AR"]["exact"]),
        "q_AR_scalar matches majorant",
    )
    add(
        checks,
        "inverse_audit_status_matches",
        known["inverse_budget_status"] == inverse_audit["status"],
        "inverse audit status matches",
    )
    add(
        checks,
        "product_obligations_match",
        known["product_space_obligations"] == product["theorem_obligations"],
        "product-space obligations match contract",
    )

    expected_slots = {
        "q_hodge_uniform",
        "q_comm_FH_Gf",
        "q_projection_residual",
        "q_gauge_transfer",
    }
    slots = budget.get("neumann_slots", {})
    add(checks, "slots_present", set(slots) == expected_slots, "all Neumann slots are present")

    per_slot_target = frac_record(budget["budget_targets"]["per_slot_target"])
    total_target = frac_record(budget["budget_targets"]["q_neumann_total_target"])
    q_scalar = frac_record(known["q_AR_scalar"])
    scalar_plus_target = frac_record(budget["budget_targets"]["q_scalar_plus_neumann_target"])
    threshold = frac_record(budget["budget_targets"]["contraction_threshold"])

    for slot_id, slot in slots.items():
        add(checks, f"{slot_id}_status_pending", slot.get("status") == "pending_exact_certificate", f"{slot_id} is pending")
        add(checks, f"{slot_id}_no_value", slot.get("value") is None, f"{slot_id} has no fake numeric value")
        add(
            checks,
            f"{slot_id}_target_matches",
            frac_record(slot["target_upper_bound"]) == per_slot_target,
            f"{slot_id} uses the per-slot target",
        )
        add(checks, f"{slot_id}_has_required_artifact", bool(slot.get("required_artifact")), f"{slot_id} names required artifact")
        if "candidate_artifact" in slot:
            cand = slot["candidate_artifact"]
            add(
                checks,
                f"{slot_id}_candidate_not_certificate",
                cand.get("status") == "candidate_formula_not_certificate",
                f"{slot_id} candidate remains non-certificate",
            )
            add(
                checks,
                f"{slot_id}_candidate_does_not_close_slot",
                slot.get("status") == "pending_exact_certificate" and slot.get("value") is None,
                f"{slot_id} remains pending despite candidate",
            )

    add(checks, "total_target_is_slot_sum", total_target == per_slot_target * len(expected_slots), "total target is sum of slot caps")
    add(
        checks,
        "scalar_plus_target_matches",
        scalar_plus_target == q_scalar + total_target,
        "q_scalar_plus_neumann target recomputes",
    )
    add(checks, "target_would_contract", scalar_plus_target < threshold, "target budget would still be a contraction")

    current = budget.get("current_evaluation", {})
    add(checks, "all_slots_not_certified", current.get("all_slots_certified") is False, "slots are not certified")
    add(checks, "q_total_absent", current.get("q_neumann_total") is None and current.get("q_total_with_scalar") is None, "no total is faked")
    add(checks, "promotion_not_ready", current.get("promotion_ready") is False, "promotion is not ready")
    add(
        checks,
        "boundary_blocks_promotion",
        any("Promotion requires" in line for line in budget.get("acceptance_boundary", [])),
        "acceptance boundary requires future certificates",
    )

    payload = {
        "artifact": "phase4_ar_neumann_budget_check",
        "generated_by": "scripts/phase4_ar_neumann_budget_checker.py",
        "scope": "Independent checker for the P4.2 AR Neumann-budget skeleton",
        "status": "checker_for_budget_skeleton_not_AR_theorem",
        "checked_artifact": BUDGET_PATH,
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
