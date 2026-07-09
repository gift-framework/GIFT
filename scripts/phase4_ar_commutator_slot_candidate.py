#!/usr/bin/env python3
"""Candidate bound for the P4.2 Neumann slot q_comm_FH_Gf.

This is not an analytic commutator certificate. It gives a reproducible target
formula for the first AR Neumann slot, using only already serialized product
constants and an explicit overhead factor.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path


PRODUCT_SPACE_PATH = "certificates/phase4_ar_product_space_contract.json"
NEUMANN_BUDGET_PATH = "certificates/phase4_ar_neumann_budget_candidate.json"
OUT_PATH = "certificates/phase4_ar_commutator_slot_candidate.json"


def rec(value: Fraction, meaning: str, status: str = "candidate_not_theorem") -> dict[str, object]:
    return {
        "exact": str(value),
        "value": float(value),
        "meaning": meaning,
        "status": status,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    product = json.loads((repo_root / PRODUCT_SPACE_PATH).read_text())
    budget = json.loads((repo_root / NEUMANN_BUDGET_PATH).read_text())

    K_F = Fraction(product["operators"]["F_H"]["bound"]["exact"])
    K_AR_prod = Fraction(product["candidate_product_inverse_bound"]["K_AR_prod"]["exact"])
    eps_AR = Fraction(budget["known_inputs"]["eps_AR"]["exact"])
    slot_target = Fraction(budget["neumann_slots"]["q_comm_FH_Gf"]["target_upper_bound"]["exact"])

    # Candidate Leibniz/cutoff overhead. The factor 8 is intentionally explicit:
    # it packages two first-order commutator placements, two collar/base sides,
    # and a factor two for gauge-transfer bookkeeping. A theorem must replace
    # this candidate decomposition by an actual estimate.
    C_overhead = Fraction(8, 1)
    C_comm_AR = C_overhead * (1 + K_F) * (1 + K_AR_prod)
    q_comm_FH_Gf = C_comm_AR * eps_AR

    payload = {
        "artifact": "phase4_ar_commutator_slot_candidate",
        "generated_by": "scripts/phase4_ar_commutator_slot_candidate.py",
        "scope": "Candidate formula for the AR Neumann slot q_comm_FH_Gf",
        "status": "candidate_formula_not_certificate",
        "depends_on": [PRODUCT_SPACE_PATH, NEUMANN_BUDGET_PATH],
        "slot": "q_comm_FH_Gf",
        "formula": {
            "C_comm_AR": "8 * (1 + K_F) * (1 + K_AR_prod)",
            "q_comm_FH_Gf": "C_comm_AR * eps_AR",
        },
        "inputs": {
            "K_F": rec(K_F, "horizontal curvature action bound from product-space contract", "certified_upper_bound_at_D0"),
            "K_AR_prod": rec(K_AR_prod, "structural product-max bound from product-space contract", "structural_product_max_bound"),
            "eps_AR": rec(eps_AR, "current AR threshold inverse"),
            "C_overhead": rec(C_overhead, "explicit candidate Leibniz/cutoff overhead"),
            "slot_target": rec(slot_target, "Neumann budget target for q_comm_FH_Gf"),
        },
        "candidate_values": {
            "C_comm_AR": rec(C_comm_AR, "candidate commutator constant"),
            "q_comm_FH_Gf": rec(q_comm_FH_Gf, "candidate Neumann error for F_H/G_f commutators"),
            "target_margin": rec(slot_target - q_comm_FH_Gf, "slot_target - q_comm_FH_Gf candidate"),
        },
        "checks": {
            "candidate_below_slot_target": q_comm_FH_Gf <= slot_target,
            "candidate_C_comm_AR_below_250": C_comm_AR <= 250,
            "status_not_certificate": True,
        },
        "open_assumptions": [
            "A product-norm commutator theorem must justify the factor 8 overhead or replace it by a certified value.",
            "Collar/base transition commutators must be bounded in the same A_beta norm.",
            "No derivative loss beyond the beta-weighted product norm may be hidden in K_F or K_AR_prod.",
        ],
        "acceptance_boundary": [
            "This candidate may guide the q_comm_FH_Gf proof target.",
            "It does not close the q_comm_FH_Gf Neumann slot.",
            "The Neumann budget must keep q_comm_FH_Gf pending until a theorem-grade commutator certificate exists.",
        ],
        "all_pass": True,
    }

    out = repo_root / OUT_PATH
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
