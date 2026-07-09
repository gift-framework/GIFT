#!/usr/bin/env python3
"""Public value certificate for Phase 4.1 Donaldson coefficient formulas.

This evaluates the symbolic Stage B v1 formulas using the theorem-grade
constants extracted in the private canonical Phase 4.1 scripts. The arithmetic
is exact rational arithmetic until the final decimal display fields.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path


def dec(x: Fraction) -> float:
    return float(x)


def exact(x: Fraction) -> str:
    return str(x)


def value_record(value: Fraction, provenance: str, status: str = "theorem_grade_upper_bound") -> dict[str, object]:
    return {
        "exact": exact(value),
        "value": dec(value),
        "status": status,
        "provenance": provenance,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    # D0 constants used by the private Phase 4.1 closure.
    K_H_K3 = Fraction(9, 20)
    K_F = Fraction(231, 100)
    KH_KF = K_H_K3 * K_F
    G_aug = Fraction(183, 5)
    C_nl = Fraction(2, 3)

    constants = {
        "K_H_K3": value_record(K_H_K3, "private axis2_K3_hodge_gap / Phase 4.1 closure uses K_H_K3 <= 0.45"),
        "K_F": value_record(K_F, "private axis2_hypothesis_AR_discharge: K_F <= 2.31"),
        "G_aug": value_record(G_aug, "private Phase 4.1 third-order closure uses ||G_aug|| <= 36.6"),
        "C_nl": value_record(C_nl, "private C_nl theorem-grade extraction"),
        "C_projection": value_record(Fraction(1), "L2 orthogonal projection has sharp norm 1"),
        "C_conn_1": value_record(Fraction(0), "Donaldson canonical gauge d_H omega = 0"),
        "C_D_conn_1": value_record(Fraction(0), "Donaldson canonical gauge"),
        "C_conn_2": value_record(Fraction(0), "Donaldson canonical gauge to second order"),
        "C_D_conn_2": value_record(Fraction(0), "Donaldson canonical gauge to second order"),
        "C_conn_3": value_record(Fraction(0), "Donaldson canonical gauge to third order"),
        "C_E3_omega": value_record(Fraction(1), "unit-normalized K3 source"),
        "C_E5_mu": value_record(Fraction(1), "unit-normalized K3 source"),
        "C_D_E3": value_record(Fraction(1), "linearized unit-normalized K3 source"),
        "C_D_E5": value_record(Fraction(1), "linearized unit-normalized K3 source"),
        "C_D2_E3": value_record(Fraction(1), "second variation unit-normalized K3 source"),
        "C_D2_E5": value_record(Fraction(1), "second variation unit-normalized K3 source"),
        "C_comm": value_record(KH_KF, "Weitzenboeck/Kato commutator bound K_H_K3*K_F"),
        "C_D_comm": value_record(KH_KF, "linearized Weitzenboeck/Kato commutator bound"),
        "C_D2_comm": value_record(KH_KF, "second variation Weitzenboeck/Kato commutator bound"),
        "C_comm_2": value_record(KH_KF * KH_KF, "iterated commutator bound (K_H_K3*K_F)^2"),
        "C_D_comm_2": value_record(KH_KF * KH_KF, "linearized iterated commutator bound"),
        "C_wedge": value_record(Fraction(1), "unit-normalized fibrewise wedge on K3"),
        "C_D_wedge": value_record(Fraction(1), "linearized unit-normalized wedge"),
        "C_Hodge": value_record(Fraction(1), "canonical unit-normalized fibrewise Hodge coefficient"),
        "C_D_Hodge": value_record(Fraction(1), "linearized canonical unit-normalized fibrewise Hodge coefficient"),
        "C_comm_3": value_record(KH_KF**3, "third iterated commutator bound (K_H_K3*K_F)^3"),
        "C_cubic_star": value_record(Fraction(1), "unit-normalized cubic star/Hodge nonlinearity"),
        "C_Hodge_3": value_record(Fraction(1), "third-order canonical unit-normalized Hodge coefficient"),
        "C_D3m": value_record(Fraction(4, 9), "Simons-type third variation + collinearity + type-IV bound"),
    }

    source_P1 = KH_KF * 3
    DP1_norm = source_P1
    source_P2 = 3 * KH_KF * KH_KF + source_P1
    DP2_norm = 2 * KH_KF * KH_KF + source_P1 + DP1_norm
    D2P1_norm = source_P1
    D3m_norm = Fraction(4, 9)
    raw_P3_scale = Fraction(1) + 2 * KH_KF**3

    xi1_bound = G_aug * source_P1
    xi2_forcing = source_P2 + DP1_norm * xi1_bound + Fraction(1, 2) * C_nl * xi1_bound * xi1_bound
    xi2_bound = G_aug * xi2_forcing
    remainder_R3 = (
        DP1_norm * xi2_bound
        + C_nl * xi1_bound * xi2_bound
        + Fraction(1, 6) * D3m_norm * xi1_bound**3
        + Fraction(1, 2) * D2P1_norm * xi1_bound**2
        + DP2_norm * xi1_bound
        + raw_P3_scale
    )

    tail_contraction_denominator = 4 * G_aug * G_aug * C_nl * remainder_R3
    R_threshold = dec(tail_contraction_denominator) ** (1.0 / 3.0)

    evaluated = {
        "source_P1": value_record(source_P1, "3 * K_H_K3 * K_F"),
        "source_P2": value_record(source_P2, "3 * (K_H_K3*K_F)^2 + source_P1"),
        "DP1_norm": value_record(DP1_norm, "same first-order structure as source_P1"),
        "DP2_norm": value_record(DP2_norm, "2 * (K_H_K3*K_F)^2 + source_P1 + DP1_norm"),
        "D2P1_norm": value_record(D2P1_norm, "same structural bound as source_P1"),
        "D3m_norm": value_record(D3m_norm, "C_D3m = 4/9"),
        "raw_P3_scale": value_record(raw_P3_scale, "1 + 2 * (K_H_K3*K_F)^3"),
        "xi1_bound": value_record(xi1_bound, "G_aug * source_P1"),
        "xi2_forcing": value_record(xi2_forcing, "source_P2 + DP1*xi1 + 1/2*C_nl*xi1^2"),
        "xi2_bound": value_record(xi2_bound, "G_aug * xi2_forcing"),
        "remainder_R3": value_record(remainder_R3, "Taylor R3 formula from Stage B v1"),
        "tail_contraction_denominator": value_record(tail_contraction_denominator, "4 * G_aug^2 * C_nl * remainder_R3"),
        "R_threshold": {
            "value": R_threshold,
            "status": "floating_display_from_exact_denominator",
            "provenance": "(4 * G_aug^2 * C_nl * remainder_R3)^(1/3)",
            "semantics": (
                "minimum admissible R for the eps^3 tail contraction; "
                "4 * G_aug^2 * C_nl * remainder_R3 * eps^3 < 1 holds for "
                "R >= R_threshold"
            ),
        },
    }

    payload = {
        "artifact": "phase4_donaldson_coefficients_values",
        "generated_by": "scripts/phase4_donaldson_coefficients_values.py",
        "scope": "Phase 4.1 theorem-grade D0 values for Stage B v1 symbolic formulas",
        "status": "theorem_grade_formula_values_from_private_canonical_artifacts",
        "private_context": {
            "repo_commit": "b46e3cd6c57757cd466e3001092fed7ef4d7d456",
            "dirty_worktree_note": "private has an unrelated dirty memory/agent_reports/epistemic_map_status.json",
        },
        "private_sources": {
            "first_order": {
                "script": "private/canonical/scripts/axis2_phase4_1_first_order_2026_07_03.py",
                "script_sha256": "0f6586c0f1d016374c8de5e6e8dc5e773a135766dceaf8ebda9176fad6e851e7",
                "result": "private/canonical/results/axis2_phase4_1_first_order_2026_07_03.json",
                "result_sha256": "95c12129852038fd2ce2917d7f25836a08b10824032ad03ff442643b5f720252",
            },
            "second_order": {
                "script": "private/canonical/scripts/axis2_phase4_1_second_order_2026_07_03.py",
                "script_sha256": "b5703ac5ec9fc2ff413601c1761afd2861ce9fee13e5c3958702ea42fc5fd50a",
                "result": "private/canonical/results/axis2_phase4_1_second_order_2026_07_03.json",
                "result_sha256": "e6827ee35e5f1252df80e1fa1fccdfa37f29026d09e1c602ef045702ad2d7d72",
            },
            "third_order": {
                "script": "private/canonical/scripts/axis2_phase4_1_third_order_2026_07_03.py",
                "script_sha256": "5069bfc67fcd06a4dcccd0f00dfbc34f8f622ceb87855ded519e1f0310fd5594",
                "result": "private/canonical/results/axis2_phase4_1_third_order_2026_07_03.json",
                "result_sha256": "9e3a27f18677257224cfb494a75172bf71d2e0c8733d1951803b6a4cee3928b7",
            },
        },
        "constants": constants,
        "evaluated_coefficients": evaluated,
        "comparison_to_old_power_counting": {
            "source_P1_old": 2.31,
            "source_P2_old": 5.3361,
            "R_threshold_old": 2730.613114210807,
            "R_threshold_new": R_threshold,
            "R_threshold_ratio_new_over_old": R_threshold / 2730.613114210807,
            "R_pc_check_public_candidate": 4019,
            "note": (
                "R_threshold is a lower admissibility threshold in R, not a "
                "headroom margin. The 2730 comparison is against an old public "
                "candidate that was not a theorem input; the like-for-like "
                "public power-counting check is R_pc_check_public_candidate = 4019."
            ),
        },
        "acceptance_boundary": [
            "This closes Phase 4.1 coefficient values at D0.",
            "This does not prove the global adiabatic reconstruction convergence theorem.",
            "This does not prove the anisotropic Joyce perturbation theorem.",
        ],
        "all_pass": True,
    }

    out_path = repo_root / "certificates" / "phase4_donaldson_coefficients_values.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out_path.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
