#!/usr/bin/env python3
"""Product-space contract for Phase 4.2 adiabatic reconstruction.

This artifact is a definition/check target, not an AR theorem. It fixes the
Banach product space and the block inverse architecture that the conditional
scalar majorant must eventually use.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path


OUT_PATH = "certificates/phase4_ar_product_space_contract.json"


def rec(value: Fraction, meaning: str, status: str = "conditional_bound") -> dict[str, object]:
    return {
        "exact": str(value),
        "value": float(value),
        "meaning": meaning,
        "status": status,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    k_h_k3 = Fraction(9, 20)
    k_f = Fraction(231, 100)
    k_proj = Fraction(1, 1)
    k_ar_prod_candidate = k_h_k3 * k_f

    payload = {
        "artifact": "phase4_ar_product_space_contract",
        "generated_by": "scripts/phase4_ar_product_space_contract.py",
        "scope": "Phase 4.2 product Banach-space and block-inverse contract for AR",
        "status": "definition_with_explicit_unproved_inverse_obligations",
        "depends_on": [
            "certificates/phase4_bigraded_type_check.json",
            "certificates/phase4_donaldson_coefficients_values.json",
            "certificates/phase4_donaldson_coefficients_check.json",
        ],
        "product_space": {
            "name": "A_beta",
            "definition": "A_omega x A_lambda x A_mu x A_Theta",
            "norm": {
                "formula": (
                    "max(||omega||_{1,2;beta}, ||lambda||_{3,0;beta}, "
                    "||mu||_{0,4;beta}, ||Theta||_{2,2;beta})"
                ),
                "status": "definition",
                "no_hidden_O1_constants": True,
            },
            "components": {
                "omega": {
                    "bidegree": [1, 2],
                    "space": "A_omega",
                    "gauge": "d_f omega = 0 and d_H omega = 0",
                    "norm": "weighted Holder norm in the Phase 3 beta convention",
                },
                "lambda": {
                    "bidegree": [3, 0],
                    "space": "A_lambda",
                    "gauge": "fibrewise coexact representative",
                    "norm": "weighted Holder norm after fibrewise Hodge inversion",
                },
                "mu": {
                    "bidegree": [0, 4],
                    "space": "A_mu",
                    "gauge": "d_H mu = 0",
                    "norm": "weighted Holder fibre-volume norm",
                },
                "Theta": {
                    "bidegree": [2, 2],
                    "space": "A_Theta",
                    "gauge": "fibrewise coexact representative modulo reduced projection",
                    "norm": "weighted Holder norm after fibrewise Hodge inversion",
                },
            },
        },
        "operators": {
            "F_H": {
                "degree_rule": [2, -1],
                "bound": rec(k_f, "horizontal curvature action bound inherited from D0"),
                "status": "certified_upper_bound_at_D0",
            },
            "G_f": {
                "formula": "Delta_f^{-1} d_f^* on d_f-exact K3 sectors",
                "degree_rule": [0, -1],
                "bound": rec(k_h_k3, "fibrewise Hodge inverse candidate bound K_H_K3 <= 9/20", "certified_upper_bound_at_D0"),
                "uniformity_status": "not_yet_proved_as_product_space_theorem",
            },
            "Pi_reduced": {
                "formula": "projection of d_H Theta to the reduced M_eps equation",
                "bound": rec(k_proj, "orthogonal projection bound"),
                "status": "certified_on_reduced_model_only",
            },
        },
        "block_inverse_architecture": {
            "lambda_from_omega": {
                "equation": "d_f lambda = -F_H omega",
                "operator": "-G_f F_H",
                "input_bidegree": [3, 1],
                "output_bidegree": [3, 0],
                "bound_formula": "K_H_K3 * K_F",
            },
            "Theta_from_mu": {
                "equation": "d_f Theta = -F_H mu",
                "operator": "-G_f F_H",
                "input_bidegree": [2, 3],
                "output_bidegree": [2, 2],
                "bound_formula": "K_H_K3 * K_F",
            },
            "reduced_projection": {
                "equation": "d_H Theta = 0 reduced to M_eps(h)=0",
                "operator": "Pi_reduced d_H",
                "input_bidegree": [3, 2],
                "output_bidegree": [3, 2],
                "bound_formula": "K_projection",
            },
        },
        "candidate_product_inverse_bound": {
            "K_AR_prod": rec(
                k_ar_prod_candidate,
                "structural product-max bound for the -G_f F_H block: K_H_K3 * K_F",
                "structural_product_max_bound",
            ),
            "derivation_boundary": (
                "This structural product-max bound repairs the earlier normalization "
                "gap K_AR_prod=K_H_K3. It still does not prove the uniform fibrewise "
                "inverse theorem, commutator theorem, reduced projection identity, or "
                "closedness preservation."
            ),
        },
        "theorem_obligations": {
            "product_space_defined": True,
            "uniform_fibrewise_inverse_proved": False,
            "commutators_bounded_in_product_norm": False,
            "reduced_projection_global_identity_proved": False,
            "closedness_preservation_proved": False,
        },
        "acceptance_boundary": [
            "This contract defines A_beta and the block inverse architecture.",
            "It does not prove the uniform product-space inverse theorem.",
            "It does not discharge AR until all theorem_obligations are true.",
        ],
        "all_pass": True,
    }

    out = repo_root / OUT_PATH
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
