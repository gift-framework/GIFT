#!/usr/bin/env python3
"""Conditional scalar majorant for Phase 4.2 AR convergence.

This is not the global adiabatic reconstruction theorem. It packages a first
scalar majorant using the checked P4.1 D0 coefficient values, conditional on a
uniform product-space fibrewise inverse bound.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path


VALUES_PATH = "certificates/phase4_donaldson_coefficients_values.json"
PRODUCT_SPACE_PATH = "certificates/phase4_ar_product_space_contract.json"
OUT_PATH = "certificates/phase4_ar_majorant_candidate.json"


def f(x: Fraction) -> float:
    return float(x)


def rec(x: Fraction, meaning: str) -> dict[str, object]:
    return {"exact": str(x), "value": f(x), "meaning": meaning}


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    values = json.loads((repo_root / VALUES_PATH).read_text())
    product_space = json.loads((repo_root / PRODUCT_SPACE_PATH).read_text())
    ev = values["evaluated_coefficients"]

    source_P1 = Fraction(ev["source_P1"]["exact"])
    source_P2 = Fraction(ev["source_P2"]["exact"])
    DP1_norm = Fraction(ev["DP1_norm"]["exact"])
    DP2_norm = Fraction(ev["DP2_norm"]["exact"])
    remainder_R3 = Fraction(ev["remainder_R3"]["exact"])
    R_threshold = values["evaluated_coefficients"]["R_threshold"]["value"]

    # Conditional product-space inverse bound. This is now read from the public
    # product-space contract, but the contract still marks the uniform inverse
    # theorem as an open obligation.
    K_AR_prod = Fraction(product_space["candidate_product_inverse_bound"]["K_AR_prod"]["exact"])
    obligations = product_space["theorem_obligations"]

    R_AR = Fraction(4000, 1)
    eps_AR = Fraction(1, 4000)
    r_AR = Fraction(1, 100)

    C_AR_source = source_P1
    C_AR_second = source_P2
    C_AR_lin = DP1_norm
    C_AR_quad = DP2_norm
    C_AR_tail = remainder_R3

    source_majorant = C_AR_source * eps_AR + C_AR_second * eps_AR**2 + C_AR_tail * eps_AR**3
    displacement_majorant = K_AR_prod * source_majorant
    q_AR = K_AR_prod * (C_AR_lin * eps_AR + C_AR_quad * eps_AR**2 + C_AR_tail * eps_AR**3)
    tail_at_eps = C_AR_tail * eps_AR**3

    payload = {
        "artifact": "phase4_ar_majorant_candidate",
        "generated_by": "scripts/phase4_ar_majorant_candidate.py",
        "scope": "Conditional scalar majorant for P4.2 global AR convergence",
        "status": "conditional_majorant_candidate_not_AR_theorem",
        "depends_on": [
            VALUES_PATH,
            PRODUCT_SPACE_PATH,
        ],
        "conditional_assumptions": {
            "K_AR_prod": {
                "exact": str(K_AR_prod),
                "value": f(K_AR_prod),
                "meaning": "Structural product-max bound from phase4_ar_product_space_contract. Uniformity is still not proved.",
            },
            "product_space_defined": obligations["product_space_defined"],
            "uniform_fibrewise_inverse_proved": obligations["uniform_fibrewise_inverse_proved"],
            "commutators_bounded_in_product_norm": obligations["commutators_bounded_in_product_norm"],
            "reduced_projection_global_identity_proved": obligations["reduced_projection_global_identity_proved"],
            "closedness_preservation_proved": obligations["closedness_preservation_proved"],
        },
        "inputs": {
            "source_P1": rec(source_P1, "P4.1 checked first-order source"),
            "source_P2": rec(source_P2, "P4.1 checked second-order source"),
            "DP1_norm": rec(DP1_norm, "P4.1 checked first derivative"),
            "DP2_norm": rec(DP2_norm, "P4.1 checked second derivative proxy"),
            "remainder_R3": rec(remainder_R3, "P4.1 checked K=2 remainder coefficient"),
            "R_threshold_P4_1": {
                "value": R_threshold,
                "meaning": "P4.1 finite-order tail threshold from checked coefficients",
            },
        },
        "chosen_threshold": {
            "R_AR": rec(R_AR, "conservative public AR candidate threshold"),
            "eps_AR": rec(eps_AR, "1/R_AR"),
            "r_AR": rec(r_AR, "candidate reconstruction ball radius"),
        },
        "majorant_constants": {
            "C_AR_source": rec(C_AR_source, "coefficient of eps in source majorant"),
            "C_AR_second": rec(C_AR_second, "coefficient of eps^2 in source majorant"),
            "C_AR_lin": rec(C_AR_lin, "coefficient of eps in Lipschitz majorant"),
            "C_AR_quad": rec(C_AR_quad, "coefficient of eps^2 in Lipschitz majorant"),
            "C_AR_tail": rec(C_AR_tail, "coefficient of eps^3 tail majorant"),
        },
        "evaluated_at_eps_AR": {
            "source_majorant": rec(source_majorant, "C_AR_source eps + C_AR_second eps^2 + C_AR_tail eps^3"),
            "displacement_majorant": rec(displacement_majorant, "K_AR_prod * source_majorant"),
            "q_AR": rec(q_AR, "K_AR_prod * (C_AR_lin eps + C_AR_quad eps^2 + C_AR_tail eps^3)"),
            "tail_at_eps": rec(tail_at_eps, "C_AR_tail eps^3"),
        },
        "checks": {
            "R_AR_exceeds_P4_1_threshold": f(R_AR) >= R_threshold,
            "q_AR_lt_half": q_AR < Fraction(1, 2),
            "displacement_inside_half_ball": displacement_majorant <= r_AR / 2,
            "tail_lt_linear_source": tail_at_eps <= source_P1 * eps_AR,
        },
        "acceptance_boundary": [
            "This is a scalar conditional majorant only.",
            "It does not prove the product Banach-space reconstruction theorem.",
            "It must not be cited as AR discharged until K_AR_prod and the product-space contraction are proved.",
        ],
    }
    payload["all_pass"] = all(payload["checks"].values())

    (repo_root / OUT_PATH).write_text(json.dumps(payload, indent=2) + "\n")
    if not payload["all_pass"]:
        raise SystemExit("phase4 AR majorant candidate failed")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
