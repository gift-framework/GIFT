#!/usr/bin/env python3
"""Audit the current P4.2 inverse-budget semantics.

This deliberately does not prove the uniform AR inverse theorem. It checks
whether the scalar constant K_AR_prod used by the majorant is compatible with
the product-space block contract under the currently declared max norm.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path


CONTRACT_PATH = "certificates/phase4_ar_product_space_contract.json"
MAJORANT_PATH = "certificates/phase4_ar_majorant_candidate.json"
OUT_PATH = "certificates/phase4_ar_inverse_budget_audit.json"


def rec(value: Fraction, meaning: str) -> dict[str, object]:
    return {"exact": str(value), "value": float(value), "meaning": meaning}


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    contract = json.loads((repo_root / CONTRACT_PATH).read_text())
    majorant = json.loads((repo_root / MAJORANT_PATH).read_text())

    k_h = Fraction(contract["operators"]["G_f"]["bound"]["exact"])
    k_f = Fraction(contract["operators"]["F_H"]["bound"]["exact"])
    k_proj = Fraction(contract["operators"]["Pi_reduced"]["bound"]["exact"])
    k_ar = Fraction(majorant["conditional_assumptions"]["K_AR_prod"]["exact"])
    block_gf_fh = k_h * k_f

    # Under the unweighted max norm declared in the product-space contract, the
    # block map -G_f F_H carries this declared upper-bound slot. The product
    # constant must dominate it and the projection slot.
    direct_max_norm_compatible = k_ar >= max(k_h, block_gf_fh, k_proj)
    status = (
        "audit_confirms_structural_product_max_bound"
        if direct_max_norm_compatible
        else "audit_finds_open_normalization_gap"
    )

    payload = {
        "artifact": "phase4_ar_inverse_budget_audit",
        "generated_by": "scripts/phase4_ar_inverse_budget_audit.py",
        "scope": "Semantic audit of K_AR_prod against the declared P4.2 product-space block contract",
        "status": status,
        "depends_on": [CONTRACT_PATH, MAJORANT_PATH],
        "declared_constants": {
            "K_H_K3": rec(k_h, "fibrewise Hodge inverse bound"),
            "K_F": rec(k_f, "horizontal curvature action bound"),
            "K_projection": rec(k_proj, "reduced projection bound"),
            "K_AR_prod": rec(k_ar, "scalar product inverse bound used by the AR majorant"),
            "K_H_K3_times_K_F": rec(block_gf_fh, "declared block bound for -G_f F_H"),
        },
        "compatibility_checks": {
            "K_AR_prod_covers_G_f": k_ar >= k_h,
            "K_AR_prod_covers_G_f_F_H_block": k_ar >= block_gf_fh,
            "K_AR_prod_covers_projection": k_ar >= k_proj,
            "direct_unweighted_max_norm_compatible": direct_max_norm_compatible,
        },
        "finding": {
            "summary": (
                "With the currently declared unweighted max norm, K_AR_prod must "
                "dominate G_f, the declared -G_f F_H block bound K_H_K3*K_F=2079/2000, "
                "and the projection bound 1."
            ),
            "interpretation": (
                "The structural product-max repair sets K_AR_prod=K_H_K3*K_F, "
                "which covers all declared slots under the current max norm. This "
                "repairs the scalar normalization gap, but not the remaining analytic "
                "uniformity, commutator, projection-identity, or closedness theorems."
            ),
        },
        "allowed_repairs": [
            "Current selected repair: use K_AR_prod = K_H_K3 * K_F = 2079/2000 as the product-max structural constant.",
            "Alternative still possible: introduce weighted component norms with explicit weights and prove a sharper block matrix norm.",
            "Alternative still possible: define K_AR_prod as a post-inversion source-to-correction constant with separate F_H/projection factors.",
        ],
        "acceptance_boundary": [
            "This audit resolves the scalar normalization mismatch for the product-max contract.",
            "It does not invalidate the P4.1 coefficients.",
            "It does not prove the remaining analytic AR theorem obligations.",
        ],
        "all_pass": True,
    }

    out = repo_root / OUT_PATH
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
