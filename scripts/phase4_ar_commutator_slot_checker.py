#!/usr/bin/env python3
"""Checker for the P4.2 q_comm_FH_Gf candidate bound."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any


CANDIDATE_PATH = "certificates/phase4_ar_commutator_slot_candidate.json"
PRODUCT_SPACE_PATH = "certificates/phase4_ar_product_space_contract.json"
NEUMANN_BUDGET_PATH = "certificates/phase4_ar_neumann_budget_candidate.json"
OUT_PATH = "certificates/phase4_ar_commutator_slot_check.json"


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

    add(checks, "artifact_name", candidate.get("artifact") == "phase4_ar_commutator_slot_candidate", "artifact id matches")
    add(checks, "status_candidate", candidate.get("status") == "candidate_formula_not_certificate", "status remains candidate")
    add(checks, "slot_name", candidate.get("slot") == "q_comm_FH_Gf", "candidate targets q_comm_FH_Gf")
    for dep in [PRODUCT_SPACE_PATH, NEUMANN_BUDGET_PATH]:
        add(checks, f"depends_on_{Path(dep).stem}", dep in candidate.get("depends_on", []), f"depends on {dep}")

    K_F = Fraction(product["operators"]["F_H"]["bound"]["exact"])
    K_AR_prod = Fraction(product["candidate_product_inverse_bound"]["K_AR_prod"]["exact"])
    eps_AR = Fraction(budget["known_inputs"]["eps_AR"]["exact"])
    slot = budget["neumann_slots"]["q_comm_FH_Gf"]
    slot_target = Fraction(slot["target_upper_bound"]["exact"])

    inputs = candidate["inputs"]
    add(checks, "K_F_matches_product", frac(inputs["K_F"]) == K_F, "K_F matches product-space contract")
    add(checks, "K_AR_matches_product", frac(inputs["K_AR_prod"]) == K_AR_prod, "K_AR_prod matches product-space contract")
    add(checks, "eps_matches_budget", frac(inputs["eps_AR"]) == eps_AR, "eps_AR matches Neumann budget")
    add(checks, "target_matches_budget", frac(inputs["slot_target"]) == slot_target, "slot target matches Neumann budget")

    C_overhead = frac(inputs["C_overhead"])
    C_comm_AR = C_overhead * (1 + K_F) * (1 + K_AR_prod)
    q_comm = C_comm_AR * eps_AR
    values = candidate["candidate_values"]
    add(checks, "C_comm_AR_recomputes", frac(values["C_comm_AR"]) == C_comm_AR, "C_comm_AR recomputes")
    add(checks, "q_comm_recomputes", frac(values["q_comm_FH_Gf"]) == q_comm, "q_comm_FH_Gf recomputes")
    add(checks, "margin_recomputes", frac(values["target_margin"]) == slot_target - q_comm, "target margin recomputes")
    add(checks, "candidate_below_slot_target", q_comm <= slot_target, "candidate q_comm <= slot target")
    add(checks, "candidate_C_comm_AR_below_250", C_comm_AR <= 250, "candidate C_comm_AR <= 250")
    add(checks, "budget_slot_still_pending", slot["status"] == "pending_exact_certificate", "Neumann budget slot remains pending")
    add(
        checks,
        "boundary_does_not_close_slot",
        any("does not close" in line for line in candidate.get("acceptance_boundary", [])),
        "acceptance boundary prevents slot closure",
    )

    payload = {
        "artifact": "phase4_ar_commutator_slot_check",
        "generated_by": "scripts/phase4_ar_commutator_slot_checker.py",
        "scope": "Independent checker for the P4.2 q_comm_FH_Gf candidate formula",
        "status": "checker_for_candidate_not_commutator_theorem",
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
