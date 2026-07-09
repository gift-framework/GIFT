#!/usr/bin/env python3
"""Independent checker for the conditional Phase 4.2 AR majorant."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any


CERT_PATH = "certificates/phase4_ar_majorant_candidate.json"
VALUES_PATH = "certificates/phase4_donaldson_coefficients_values.json"
PRODUCT_SPACE_PATH = "certificates/phase4_ar_product_space_contract.json"
OUT_PATH = "certificates/phase4_ar_majorant_check.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def frac(obj: dict[str, Any], key: str) -> Fraction:
    return Fraction(obj[key]["exact"])


def add(checks: list[dict[str, Any]], name: str, ok: bool, detail: str) -> None:
    checks.append({"name": name, "pass": bool(ok), "detail": detail})


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cert = read_json(repo_root / CERT_PATH)
    values = read_json(repo_root / VALUES_PATH)
    product_space = read_json(repo_root / PRODUCT_SPACE_PATH)
    checks: list[dict[str, Any]] = []

    add(checks, "artifact_name", cert.get("artifact") == "phase4_ar_majorant_candidate", "artifact id matches")
    add(checks, "status_conditional", cert.get("status") == "conditional_majorant_candidate_not_AR_theorem", "status remains conditional")
    add(checks, "values_dependency", VALUES_PATH in cert.get("depends_on", []), "depends on P4.1 values")
    add(checks, "product_space_dependency", PRODUCT_SPACE_PATH in cert.get("depends_on", []), "depends on product-space contract")

    ev = values["evaluated_coefficients"]
    source_P1 = Fraction(ev["source_P1"]["exact"])
    source_P2 = Fraction(ev["source_P2"]["exact"])
    DP1_norm = Fraction(ev["DP1_norm"]["exact"])
    DP2_norm = Fraction(ev["DP2_norm"]["exact"])
    remainder_R3 = Fraction(ev["remainder_R3"]["exact"])

    K_AR_prod = Fraction(cert["conditional_assumptions"]["K_AR_prod"]["exact"])
    contract_K_AR_prod = Fraction(product_space["candidate_product_inverse_bound"]["K_AR_prod"]["exact"])
    eps_AR = frac(cert["chosen_threshold"], "eps_AR")
    R_AR = frac(cert["chosen_threshold"], "R_AR")
    r_AR = frac(cert["chosen_threshold"], "r_AR")

    add(checks, "product_space_is_defined", cert["conditional_assumptions"]["product_space_defined"] is True, "product space is defined by the contract")
    add(checks, "inverse_is_conditional", cert["conditional_assumptions"]["uniform_fibrewise_inverse_proved"] is False, "inverse theorem is not marked proved")
    add(checks, "K_AR_prod_matches_contract", K_AR_prod == contract_K_AR_prod, "K_AR_prod is read from the product-space contract")
    for name in [
        "commutators_bounded_in_product_norm",
        "reduced_projection_global_identity_proved",
        "closedness_preservation_proved",
    ]:
        add(checks, f"{name}_open", cert["conditional_assumptions"].get(name) is False, f"{name} remains open")
    add(checks, "R_eps_inverse", eps_AR * R_AR == 1, "eps_AR = 1/R_AR")

    source_majorant = source_P1 * eps_AR + source_P2 * eps_AR**2 + remainder_R3 * eps_AR**3
    displacement = K_AR_prod * source_majorant
    q_AR = K_AR_prod * (DP1_norm * eps_AR + DP2_norm * eps_AR**2 + remainder_R3 * eps_AR**3)
    tail = remainder_R3 * eps_AR**3

    evaluated = cert["evaluated_at_eps_AR"]
    expected = {
        "source_majorant": source_majorant,
        "displacement_majorant": displacement,
        "q_AR": q_AR,
        "tail_at_eps": tail,
    }
    for name, value in expected.items():
        add(checks, f"{name}_exact", evaluated[name]["exact"] == str(value), f"{name} exact recomputes")

    add(checks, "q_AR_lt_half", q_AR < Fraction(1, 2), "q_AR < 1/2")
    add(checks, "displacement_inside_half_ball", displacement <= r_AR / 2, "displacement <= r_AR/2")
    add(checks, "tail_lt_linear_source", tail <= source_P1 * eps_AR, "tail <= first-order source contribution")
    add(checks, "cert_checks_match", all(cert.get("checks", {}).values()), "producer checks all true")
    add(
        checks,
        "boundary_not_AR_theorem",
        any("does not prove" in line for line in cert.get("acceptance_boundary", [])),
        "acceptance boundary prevents AR promotion",
    )

    payload = {
        "artifact": "phase4_ar_majorant_check",
        "generated_by": "scripts/phase4_ar_majorant_checker.py",
        "scope": "Independent scalar checker for conditional P4.2 AR majorant",
        "status": "checker_for_conditional_majorant_not_AR_theorem",
        "checked_artifact": CERT_PATH,
        "checks": checks,
        "summary": {
            "checks_total": len(checks),
            "checks_passed": sum(1 for c in checks if c["pass"]),
            "checks_failed": sum(1 for c in checks if not c["pass"]),
        },
        "all_pass": all(c["pass"] for c in checks),
    }

    (repo_root / OUT_PATH).write_text(json.dumps(payload, indent=2) + "\n")
    if not payload["all_pass"]:
        raise SystemExit("phase4 AR majorant checker failed")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
