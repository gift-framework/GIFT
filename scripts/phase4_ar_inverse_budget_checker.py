#!/usr/bin/env python3
"""Checker for the P4.2 inverse-budget audit."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any


AUDIT_PATH = "certificates/phase4_ar_inverse_budget_audit.json"
CONTRACT_PATH = "certificates/phase4_ar_product_space_contract.json"
MAJORANT_PATH = "certificates/phase4_ar_majorant_candidate.json"
OUT_PATH = "certificates/phase4_ar_inverse_budget_check.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def add(checks: list[dict[str, Any]], name: str, ok: bool, detail: str) -> None:
    checks.append({"name": name, "pass": bool(ok), "detail": detail})


def frac(obj: dict[str, Any]) -> Fraction:
    return Fraction(obj["exact"])


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    audit = read_json(repo_root / AUDIT_PATH)
    contract = read_json(repo_root / CONTRACT_PATH)
    majorant = read_json(repo_root / MAJORANT_PATH)
    checks: list[dict[str, Any]] = []

    add(checks, "artifact_name", audit.get("artifact") == "phase4_ar_inverse_budget_audit", "artifact id matches")
    add(
        checks,
        "status_structural_bound",
        audit.get("status") == "audit_confirms_structural_product_max_bound",
        "audit status confirms the structural product-max bound",
    )
    add(checks, "depends_on_contract", CONTRACT_PATH in audit.get("depends_on", []), "audit depends on product-space contract")
    add(checks, "depends_on_majorant", MAJORANT_PATH in audit.get("depends_on", []), "audit depends on AR majorant")

    k_h = Fraction(contract["operators"]["G_f"]["bound"]["exact"])
    k_f = Fraction(contract["operators"]["F_H"]["bound"]["exact"])
    k_proj = Fraction(contract["operators"]["Pi_reduced"]["bound"]["exact"])
    k_ar = Fraction(majorant["conditional_assumptions"]["K_AR_prod"]["exact"])
    block = k_h * k_f

    declared = audit.get("declared_constants", {})
    add(checks, "K_H_matches", frac(declared["K_H_K3"]) == k_h, "K_H_K3 matches contract")
    add(checks, "K_F_matches", frac(declared["K_F"]) == k_f, "K_F matches contract")
    add(checks, "K_projection_matches", frac(declared["K_projection"]) == k_proj, "projection bound matches contract")
    add(checks, "K_AR_matches", frac(declared["K_AR_prod"]) == k_ar, "K_AR_prod matches majorant")
    add(checks, "block_product_matches", frac(declared["K_H_K3_times_K_F"]) == block, "block bound recomputes")
    add(checks, "K_AR_is_block_product", k_ar == block, "K_AR_prod is K_H_K3*K_F")

    compat = audit.get("compatibility_checks", {})
    add(checks, "K_AR_covers_G_f_truth", compat.get("K_AR_prod_covers_G_f") == (k_ar >= k_h), "G_f coverage truth matches arithmetic")
    add(
        checks,
        "K_AR_covers_G_f_F_H",
        compat.get("K_AR_prod_covers_G_f_F_H_block") is True and k_ar >= block,
        "audit confirms K_AR_prod covers the -G_f F_H block under the declared norm",
    )
    add(
        checks,
        "K_AR_covers_projection",
        compat.get("K_AR_prod_covers_projection") is True and k_ar >= k_proj,
        "audit confirms K_AR_prod covers projection bound under the declared norm",
    )
    add(
        checks,
        "direct_unweighted_max_norm_compatible",
        compat.get("direct_unweighted_max_norm_compatible") is True,
        "audit accepts direct unweighted max-norm compatibility",
    )
    add(
        checks,
        "finding_mentions_product_max_repair",
        "product-max" in audit.get("finding", {}).get("interpretation", ""),
        "finding names product-max repair route",
    )
    add(
        checks,
        "boundary_keeps_AR_open",
        any("does not prove" in line for line in audit.get("acceptance_boundary", [])),
        "acceptance boundary keeps analytic AR obligations open",
    )

    payload = {
        "artifact": "phase4_ar_inverse_budget_check",
        "generated_by": "scripts/phase4_ar_inverse_budget_checker.py",
        "scope": "Independent checker for P4.2 inverse-budget normalization audit",
        "status": "checker_for_audit_not_AR_theorem",
        "checked_artifact": AUDIT_PATH,
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
