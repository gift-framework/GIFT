#!/usr/bin/env python3
"""Candidate bounds for the remaining P4.2 AR Neumann slots.

This produces candidate formulas for q_hodge_uniform, q_projection_residual,
and q_gauge_transfer. It is not an analytic theorem and does not close any slot.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path


PRODUCT_SPACE_PATH = "certificates/phase4_ar_product_space_contract.json"
NEUMANN_BUDGET_PATH = "certificates/phase4_ar_neumann_budget_candidate.json"
OUT_PATH = "certificates/phase4_ar_remaining_slots_candidate.json"


def rec(value: Fraction, meaning: str, status: str = "candidate_not_theorem") -> dict[str, object]:
    return {
        "exact": str(value),
        "value": float(value),
        "meaning": meaning,
        "status": status,
    }


def slot_record(slot_id: str, constant: Fraction, eps_AR: Fraction, target: Fraction, formula: str, meaning: str) -> dict[str, object]:
    value = constant * eps_AR
    return {
        "slot": slot_id,
        "formula": {
            f"C_{slot_id}": formula,
            slot_id: f"C_{slot_id} * eps_AR",
        },
        "candidate_values": {
            f"C_{slot_id}": rec(constant, meaning),
            slot_id: rec(value, f"candidate Neumann error for {slot_id}"),
            "target_margin": rec(target - value, f"slot_target - {slot_id} candidate"),
        },
        "checks": {
            "candidate_below_slot_target": value <= target,
            "candidate_constant_below_250": constant <= 250,
        },
        "status": "candidate_formula_not_certificate",
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    product = json.loads((repo_root / PRODUCT_SPACE_PATH).read_text())
    budget = json.loads((repo_root / NEUMANN_BUDGET_PATH).read_text())

    K_H_K3 = Fraction(product["operators"]["G_f"]["bound"]["exact"])
    K_AR_prod = Fraction(product["candidate_product_inverse_bound"]["K_AR_prod"]["exact"])
    K_projection = Fraction(product["operators"]["Pi_reduced"]["bound"]["exact"])
    eps_AR = Fraction(budget["known_inputs"]["eps_AR"]["exact"])

    target_hodge = Fraction(budget["neumann_slots"]["q_hodge_uniform"]["target_upper_bound"]["exact"])
    target_projection = Fraction(budget["neumann_slots"]["q_projection_residual"]["target_upper_bound"]["exact"])
    target_gauge = Fraction(budget["neumann_slots"]["q_gauge_transfer"]["target_upper_bound"]["exact"])

    C_hodge_uniform = 16 * (1 + K_H_K3)
    C_projection_residual = 16 * (1 + K_projection) * (1 + K_AR_prod)
    C_gauge_transfer = 12 * (1 + K_AR_prod)

    candidates = {
        "q_hodge_uniform": slot_record(
            "q_hodge_uniform",
            C_hodge_uniform,
            eps_AR,
            target_hodge,
            "16 * (1 + K_H_K3)",
            "candidate fibrewise-to-product norm uniformity constant",
        ),
        "q_projection_residual": slot_record(
            "q_projection_residual",
            C_projection_residual,
            eps_AR,
            target_projection,
            "16 * (1 + K_projection) * (1 + K_AR_prod)",
            "candidate reduced-projection residual constant",
        ),
        "q_gauge_transfer": slot_record(
            "q_gauge_transfer",
            C_gauge_transfer,
            eps_AR,
            target_gauge,
            "12 * (1 + K_AR_prod)",
            "candidate gauge-transfer constant",
        ),
    }

    payload = {
        "artifact": "phase4_ar_remaining_slots_candidate",
        "generated_by": "scripts/phase4_ar_remaining_slots_candidate.py",
        "scope": "Candidate formulas for the remaining P4.2 AR Neumann slots",
        "status": "candidate_formulas_not_certificates",
        "depends_on": [PRODUCT_SPACE_PATH, NEUMANN_BUDGET_PATH],
        "inputs": {
            "K_H_K3": rec(K_H_K3, "fibrewise Hodge inverse bound", "certified_upper_bound_at_D0"),
            "K_AR_prod": rec(K_AR_prod, "structural product-max bound", "structural_product_max_bound"),
            "K_projection": rec(K_projection, "reduced projection bound", "certified_on_reduced_model_only"),
            "eps_AR": rec(eps_AR, "current AR threshold inverse"),
        },
        "candidates": candidates,
        "checks": {
            "all_candidates_below_targets": all(item["checks"]["candidate_below_slot_target"] for item in candidates.values()),
            "all_candidate_constants_below_250": all(item["checks"]["candidate_constant_below_250"] for item in candidates.values()),
            "status_not_certificate": True,
        },
        "open_assumptions": [
            "q_hodge_uniform needs the uniform fibrewise inverse theorem in the beta-weighted product norm.",
            "q_projection_residual needs the global reduced-projection identity.",
            "q_gauge_transfer needs a gauge-transfer and closedness-preservation theorem.",
        ],
        "acceptance_boundary": [
            "These candidates may guide the remaining Neumann-slot proof targets.",
            "They do not close any Neumann slot.",
            "The Neumann budget must keep all three slots pending until theorem-grade certificates exist.",
        ],
        "all_pass": True,
    }

    out = repo_root / OUT_PATH
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
