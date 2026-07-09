#!/usr/bin/env python3
"""Neumann-budget skeleton for the Phase 4.2 AR theorem.

This is a planning/certification interface, not an analytic theorem. It fixes
the exact slots that must be bounded before the conditional scalar AR majorant
can be promoted to a genuine product-space reconstruction theorem.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path


PRODUCT_SPACE_PATH = "certificates/phase4_ar_product_space_contract.json"
MAJORANT_PATH = "certificates/phase4_ar_majorant_candidate.json"
INVERSE_AUDIT_PATH = "certificates/phase4_ar_inverse_budget_audit.json"
COMMUTATOR_CANDIDATE_PATH = "certificates/phase4_ar_commutator_slot_candidate.json"
REMAINING_CANDIDATE_PATH = "certificates/phase4_ar_remaining_slots_candidate.json"
OUT_PATH = "certificates/phase4_ar_neumann_budget_candidate.json"


def rec(value: Fraction, meaning: str) -> dict[str, object]:
    return {"exact": str(value), "value": float(value), "meaning": meaning}


def pending_slot(
    slot_id: str,
    role: str,
    formula: str,
    target: Fraction,
    required_artifact: str,
) -> dict[str, object]:
    return {
        "id": slot_id,
        "status": "pending_exact_certificate",
        "role": role,
        "formula": formula,
        "target_upper_bound": rec(target, "sufficient per-slot Neumann budget cap"),
        "value": None,
        "required_artifact": required_artifact,
        "acceptance_test": "Provide a certified upper bound value <= target_upper_bound.",
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    product = json.loads((repo_root / PRODUCT_SPACE_PATH).read_text())
    majorant = json.loads((repo_root / MAJORANT_PATH).read_text())
    inverse_audit = json.loads((repo_root / INVERSE_AUDIT_PATH).read_text())

    eps_AR = Fraction(majorant["chosen_threshold"]["eps_AR"]["exact"])
    R_AR = Fraction(majorant["chosen_threshold"]["R_AR"]["exact"])
    q_AR_scalar = Fraction(majorant["evaluated_at_eps_AR"]["q_AR"]["exact"])
    K_AR_prod = Fraction(majorant["conditional_assumptions"]["K_AR_prod"]["exact"])

    slot_cap = Fraction(1, 16)
    total_target = Fraction(1, 4)
    scalar_plus_target = q_AR_scalar + total_target

    slots = {
        "q_hodge_uniform": pending_slot(
            "q_hodge_uniform",
            "Defect between the fibrewise Hodge inverse bound and the beta-weighted product norm over the full fibration.",
            "sup ||G_f||_{A_beta source -> A_beta correction} / K_H_K3 - 1, positive part",
            slot_cap,
            "uniform fibrewise Hodge inverse theorem in the product norm",
        ),
        "q_comm_FH_Gf": pending_slot(
            "q_comm_FH_Gf",
            "Neumann error from commutators of F_H, G_f, d_H, cutoffs, and collar/base transitions.",
            "C_comm_AR * eps_AR",
            slot_cap,
            "product-norm commutator certificate for F_H/G_f and collars",
        ),
        "q_projection_residual": pending_slot(
            "q_projection_residual",
            "Residual after replacing d_H Theta=0 by the reduced equation M_eps(h)=0.",
            "C_projection_residual * eps_AR",
            slot_cap,
            "global reduced-projection identity certificate",
        ),
        "q_gauge_transfer": pending_slot(
            "q_gauge_transfer",
            "Error introduced by transferring between Donaldson gauge, fibrewise coexact gauge, and product-space coordinates.",
            "C_gauge_transfer * eps_AR",
            slot_cap,
            "gauge-transfer and closedness-preservation certificate",
        ),
    }
    commutator_candidate_path = repo_root / COMMUTATOR_CANDIDATE_PATH
    if commutator_candidate_path.exists():
        commutator_candidate = json.loads(commutator_candidate_path.read_text())
        slots["q_comm_FH_Gf"]["candidate_artifact"] = {
            "path": COMMUTATOR_CANDIDATE_PATH,
            "status": commutator_candidate["status"],
            "candidate_value": commutator_candidate["candidate_values"]["q_comm_FH_Gf"],
            "candidate_below_target": commutator_candidate["checks"]["candidate_below_slot_target"],
            "note": "Candidate only; the slot remains pending_exact_certificate.",
        }
    remaining_candidate_path = repo_root / REMAINING_CANDIDATE_PATH
    if remaining_candidate_path.exists():
        remaining_candidate = json.loads(remaining_candidate_path.read_text())
        for slot_id in ["q_hodge_uniform", "q_projection_residual", "q_gauge_transfer"]:
            candidate = remaining_candidate["candidates"][slot_id]
            slots[slot_id]["candidate_artifact"] = {
                "path": REMAINING_CANDIDATE_PATH,
                "status": candidate["status"],
                "candidate_value": candidate["candidate_values"][slot_id],
                "candidate_below_target": candidate["checks"]["candidate_below_slot_target"],
                "note": "Candidate only; the slot remains pending_exact_certificate.",
            }

    payload = {
        "artifact": "phase4_ar_neumann_budget_candidate",
        "generated_by": "scripts/phase4_ar_neumann_budget_candidate.py",
        "scope": "P4.2 Neumann-budget interface for upgrading the conditional AR majorant to a product-space theorem",
        "status": "budget_skeleton_pending_exact_certificates",
        "depends_on": [PRODUCT_SPACE_PATH, MAJORANT_PATH, INVERSE_AUDIT_PATH],
        "known_inputs": {
            "R_AR": rec(R_AR, "current AR threshold"),
            "eps_AR": rec(eps_AR, "1/R_AR"),
            "K_AR_prod": rec(K_AR_prod, "structural product-max bound from the product-space contract"),
            "q_AR_scalar": rec(q_AR_scalar, "checked scalar contraction majorant before Neumann-error slots"),
            "inverse_budget_status": inverse_audit["status"],
            "product_space_obligations": product["theorem_obligations"],
        },
        "neumann_slots": slots,
        "budget_targets": {
            "per_slot_target": rec(slot_cap, "initial sufficient target for each missing analytic slot"),
            "q_neumann_total_target": rec(total_target, "initial sufficient target for sum of Neumann-error slots"),
            "q_scalar_plus_neumann_target": rec(scalar_plus_target, "q_AR_scalar + q_neumann_total_target"),
            "contraction_threshold": rec(Fraction(1, 1), "strict contraction threshold"),
        },
        "current_evaluation": {
            "all_slots_certified": False,
            "q_neumann_total": None,
            "q_total_with_scalar": None,
            "promotion_ready": False,
            "reason": "All four Neumann-error slots are pending exact analytic certificates.",
        },
        "acceptance_boundary": [
            "This is a budget skeleton, not an AR theorem.",
            "It must not be cited as proving the product-space inverse or closedness.",
            "Promotion requires every slot to carry a certified upper bound and q_AR_scalar + q_neumann_total < 1.",
        ],
        "all_pass": True,
    }

    out = repo_root / OUT_PATH
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
