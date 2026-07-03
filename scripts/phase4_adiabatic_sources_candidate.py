#!/usr/bin/env python3
"""
Phase 4.1 candidate certificate for the adiabatic operator source norms.

This script publicizes the formal structure already checked in the private
axis2_M_epsilon_adiabatic_2026_07_01 artifact:

    M_eps(h) = m(h) + eps P1(h) + eps^2 P2(h) + R_eps(h)

and serializes a first D0 candidate norm package for:

    ||P1(h0)||, ||P2(h0)||, ||R3||.

The numbers here are NOT theorem-grade. They are a conservative candidate
based on the current D0 horizontal-curvature scale ||F_H|| <= 2.31 and the
sharp-current nonlinear constant C_nl = 2/3. The purpose is to make P4.1
auditable and to expose exactly what must later be replaced by interval
certificates from the actual Donaldson (E1)-(E5) operator.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path


def f(x: Fraction) -> float:
    return float(x)


def main() -> None:
    # D0 public/current constants.
    G_aug = Fraction(36635, 1000)  # current working ||G_aug|| <= 36.635
    K_F = Fraction(231, 100)       # ||F_H|| <= cond(A_bulk) * K_A = 2.31
    C_nl = Fraction(2, 3)          # sharp-current C_nl from private P1/C_nl chain

    # Candidate source and derivative scales.
    S1 = K_F
    S2 = K_F * K_F
    L_P1 = K_F
    L_P2 = K_F * K_F
    L2_P1 = K_F * K_F
    L3_m = Fraction(1, 1)
    S3_raw = K_F * K_F * K_F

    # Finite-order ansatz bounds:
    # xi1 = -G_aug P1(h0)
    # xi2 = -G_aug [P2 + DP1 xi1 + 1/2 D2m[xi1,xi1]]
    xi1 = G_aug * S1
    xi2_forcing = S2 + L_P1 * xi1 + Fraction(1, 2) * C_nl * xi1 * xi1
    xi2 = G_aug * xi2_forcing

    # Candidate eps^3 coefficient. This is intentionally broad and structural:
    #   DP1 xi2 + D2m(xi1,xi2) + 1/6 D3m(xi1^3)
    #   + 1/2 D2P1(xi1^2) + DP2(xi1) + raw P3-scale.
    R3_coeff = (
        L_P1 * xi2
        + C_nl * xi1 * xi2
        + Fraction(1, 6) * L3_m * xi1 * xi1 * xi1
        + Fraction(1, 2) * L2_P1 * xi1 * xi1
        + L_P2 * xi1
        + S3_raw
    )

    # Tail contraction threshold from the same scalar IFT shape used in the
    # private M_epsilon artifact:
    #   4 G_aug^2 C_nl (R3_coeff eps^3) < 1.
    tail_denominator = 4 * G_aug * G_aug * C_nl * R3_coeff
    eps_tail_threshold = float(Fraction(1, 1) / tail_denominator) ** (1.0 / 3.0)

    out = {
        "artifact": "phase4_adiabatic_sources_candidate",
        "generated_by": "scripts/phase4_adiabatic_sources_candidate.py",
        "scope": "Phase 4.1 adiabatic operator source-norm candidate at D0",
        "status": "candidate_not_theorem",
        "operator_form": {
            "formula": "M_eps(h) = m(h) + eps * P1(h) + eps^2 * P2(h) + R_eps(h)",
            "domain": "X_beta^ext maximal-section Banach neighbourhood of h0",
            "codomain": "Y_{beta-2} Donaldson/maximal-section source space",
            "epsilon": "R^-1",
            "background": "m(h0)=0 from Phase 3 maximal-section theorem once proved",
        },
        "D0_input_constants": {
            "G_aug_working_bound": {"exact": str(G_aug), "value": f(G_aug), "status": "working_bound"},
            "K_F_norm_F_H": {"exact": str(K_F), "value": f(K_F), "status": "D0 datum envelope"},
            "C_nl": {"exact": str(C_nl), "value": f(C_nl), "status": "sharp-current at D0"},
        },
        "candidate_source_norms": {
            "source_P1": {
                "exact": str(S1),
                "value": f(S1),
                "model": "||P1(h0)|| <= ||F_H||",
                "status": "candidate_not_theorem",
            },
            "source_P2": {
                "exact": str(S2),
                "value": f(S2),
                "model": "||P2(h0)|| <= ||F_H||^2",
                "status": "candidate_not_theorem",
            },
            "DP1_norm": {"exact": str(L_P1), "value": f(L_P1), "status": "candidate_not_theorem"},
            "DP2_norm": {"exact": str(L_P2), "value": f(L_P2), "status": "candidate_not_theorem"},
            "D2P1_norm": {"exact": str(L2_P1), "value": f(L2_P1), "status": "candidate_not_theorem"},
            "D3m_norm": {"exact": str(L3_m), "value": f(L3_m), "status": "candidate_not_theorem"},
            "raw_P3_scale": {"exact": str(S3_raw), "value": f(S3_raw), "status": "candidate_not_theorem"},
        },
        "finite_order_ansatz_bounds": {
            "xi1_bound": {
                "formula": "||xi1|| <= ||G_aug|| * ||P1(h0)||",
                "exact": str(xi1),
                "value": f(xi1),
            },
            "xi2_forcing_bound": {
                "formula": "||P2|| + ||DP1|| ||xi1|| + 1/2 C_nl ||xi1||^2",
                "exact": str(xi2_forcing),
                "value": f(xi2_forcing),
            },
            "xi2_bound": {
                "formula": "||xi2|| <= ||G_aug|| * xi2_forcing",
                "exact": str(xi2),
                "value": f(xi2),
            },
        },
        "remainder_R3_candidate": {
            "formula": "DP1*xi2 + C_nl*xi1*xi2 + (1/6)D3m*xi1^3 + (1/2)D2P1*xi1^2 + DP2*xi1 + P3_raw",
            "coefficient_exact": str(R3_coeff),
            "coefficient_value": f(R3_coeff),
            "meaning": "Candidate upper bound for ||R_eps|| / eps^3 after the K=2 ansatz.",
            "status": "candidate_not_theorem",
        },
        "tail_contraction_candidate": {
            "condition": "4 * ||G_aug||^2 * C_nl * R3_coeff * eps^3 < 1",
            "eps_threshold_candidate": eps_tail_threshold,
            "R_threshold_candidate": 1.0 / eps_tail_threshold,
            "status": "candidate_not_theorem",
            "interpretation": "Only a scale diagnostic. The constants are intentionally broad and not yet from the actual Donaldson operator.",
        },
        "acceptance_requirements_to_promote": [
            "derive P1, P2, and R_eps from the actual Donaldson (E1)-(E5) equations in the chosen norms",
            "replace K_F power-counting by interval enclosures for ||P1(h0)|| and ||P2(h0)||",
            "bound DP1, DP2, D2P1, D3m, and raw P3 with outward-rounded intervals",
            "provide an independent checker distinct from this producer script",
            "only then update datum_D0 source_P1/source_P2/remainder_R3 to certified intervals",
        ],
        "all_pass": True,
    }

    repo_root = Path(__file__).resolve().parents[1]
    out_path = repo_root / "certificates" / "phase4_adiabatic_sources_candidate.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
