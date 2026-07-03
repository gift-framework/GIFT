#!/usr/bin/env python3
"""
Phase 3 public certificate: constant-mode Pi_obs and clean annular commutator.

This script certifies two local pieces needed by the public Phase-3 ledger:

1. Pi_obs on the already-reduced lower-root source model is the orthogonal
   half-angle projection followed by longitudinal averaging. On that model,
   the projection residual is zero and the projection norm is one.

2. The clean moving-coordinate annular commutator is a Neumann-small operator
   in the Mazzeo regime used by D0. This is the public, serializable version of
   the private collar matching N6 bookkeeping:

       q_comm = (4/3) * K_Sch^Maz * kappa_E * epsilon.

   For D0 we use K_Sch^Maz <= 17, kappa_E <= 3, epsilon = r0 = 1e-2, hence
   q_comm <= 0.68 < 1 and epsilon_0 >= 1 / ((4/3) K_Sch kappa_E).

The script deliberately does not claim the full global Fredholm theorem. It
only freezes these two checkable inputs for the next Phase-3 assembly step.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import sympy as sp


rho = sp.symbols("rho", positive=True, real=True)
varphi = sp.symbols("varphi", real=True)
s = sp.symbols("s", real=True)


def project_half_angle(expr: sp.Expr) -> dict:
    """Project onto cos(varphi/2), sin(varphi/2) on the sigma-odd 4pi cover."""
    basis_r = sp.cos(varphi / 2)
    basis_i = sp.sin(varphi / 2)
    period = 4 * sp.pi
    norm_r = sp.integrate(basis_r * basis_r, (varphi, 0, period))
    norm_i = sp.integrate(basis_i * basis_i, (varphi, 0, period))
    cross = sp.integrate(basis_r * basis_i, (varphi, 0, period))

    coeff_r = sp.simplify(sp.integrate(expr * basis_r, (varphi, 0, period)) / norm_r)
    coeff_i = sp.simplify(sp.integrate(expr * basis_i, (varphi, 0, period)) / norm_i)
    residual = sp.simplify(expr - coeff_r * basis_r - coeff_i * basis_i)
    return {
        "basis_norm_R": norm_r,
        "basis_norm_I": norm_i,
        "basis_cross": cross,
        "coeff_R": coeff_r,
        "coeff_I": coeff_i,
        "residual": residual,
    }


def longitudinal_average(expr: sp.Expr, period: sp.Expr = 2 * sp.pi) -> sp.Expr:
    return sp.simplify(sp.integrate(expr, (s, 0, period)) / period)


def certify_pi_obs() -> dict:
    """Certify Pi_obs on a symbolic lower-root source with oscillatory tails."""
    a0, b0, a1, b1 = sp.symbols("a0 b0 a1 b1", real=True)
    lower_root_trace = (
        (a0 + a1 * sp.cos(s)) * sp.cos(varphi / 2)
        + (b0 + b1 * sp.sin(s)) * sp.sin(varphi / 2)
    )

    angular = project_half_angle(lower_root_trace)
    avg_r = longitudinal_average(angular["coeff_R"])
    avg_i = longitudinal_average(angular["coeff_I"])

    # Boundedness check: both component projections are orthogonal projections
    # followed by averaging, hence each has operator norm <= 1 in L2. We verify
    # the angular projection is exactly idempotent on the lower-root model.
    projected_trace = sp.simplify(
        angular["coeff_R"] * sp.cos(varphi / 2)
        + angular["coeff_I"] * sp.sin(varphi / 2)
    )
    idempotent_residual = sp.simplify(lower_root_trace - projected_trace)

    return {
        "status": "certified_on_reduced_lower_root_model",
        "model_trace": str(lower_root_trace),
        "angular_orthogonality": {
            "integral_cos2": str(angular["basis_norm_R"]),
            "integral_sin2": str(angular["basis_norm_I"]),
            "integral_cross": str(angular["basis_cross"]),
            "pass": bool(
                sp.simplify(angular["basis_norm_R"] - angular["basis_norm_I"]) == 0
                and sp.simplify(angular["basis_cross"]) == 0
            ),
        },
        "extracted_coefficients": {
            "a_of_s": str(angular["coeff_R"]),
            "b_of_s": str(angular["coeff_I"]),
            "angular_projection_residual": str(angular["residual"]),
            "idempotent_residual": str(idempotent_residual),
            "pass": bool(angular["residual"] == 0 and idempotent_residual == 0),
        },
        "constant_mode": {
            "Pi_const_a": str(avg_r),
            "Pi_const_b": str(avg_i),
            "expected": {"e_R": "a0", "e_I": "b0"},
            "pass": bool(sp.simplify(avg_r - a0) == 0 and sp.simplify(avg_i - b0) == 0),
        },
        "operator_norm_statement": {
            "angular_projection_norm_L2": 1,
            "longitudinal_average_norm_L2": 1,
            "Pi_obs_proto_norm_on_model": 1,
            "meaning": "Bounded projection on the reduced lower-root constant-mode source model.",
        },
    }


def certify_commutator() -> dict:
    """Certify the clean annular commutator Neumann factor in D0 bookkeeping."""
    r0 = Fraction(1, 100)
    K_sch_maz = Fraction(17, 1)
    kappa_E = Fraction(3, 1)
    clean_factor = Fraction(4, 3) * K_sch_maz * kappa_E
    q_comm = clean_factor * r0
    eps0 = Fraction(1, 1) / clean_factor

    # Also expose the sharper kappa_E<=1 variant recorded in the sigma-min note.
    kappa_E_sharp = Fraction(1, 1)
    clean_factor_sharp = Fraction(4, 3) * K_sch_maz * kappa_E_sharp
    q_comm_sharp = clean_factor_sharp * r0
    eps0_sharp = Fraction(1, 1) / clean_factor_sharp

    return {
        "status": "neumann_small_in_clean_moving_coordinate_construction",
        "formula": "q_comm = (4/3) * K_Sch^Maz * kappa_E * epsilon",
        "interpretation": (
            "This is a parametrix/operator commutator bound, not an additional "
            "source coefficient in C_src. The clean construction avoids the "
            "internal bulk/collar split and its K_Sch^2 loss."
        ),
        "D0_conservative_regime": {
            "epsilon": float(r0),
            "K_Sch_Maz_upper_bound": float(K_sch_maz),
            "kappa_E_upper_bound": float(kappa_E),
            "clean_factor": float(clean_factor),
            "q_comm": float(q_comm),
            "epsilon_0_lower_bound": float(eps0),
            "q_comm_less_than_1": bool(q_comm < 1),
            "epsilon_0_above_1e_minus_2": bool(eps0 > r0),
        },
        "D0_sharp_kappa_E_variant": {
            "epsilon": float(r0),
            "K_Sch_Maz_upper_bound": float(K_sch_maz),
            "kappa_E_upper_bound": float(kappa_E_sharp),
            "clean_factor": float(clean_factor_sharp),
            "q_comm": float(q_comm_sharp),
            "epsilon_0_lower_bound": float(eps0_sharp),
            "q_comm_less_than_1": bool(q_comm_sharp < 1),
            "epsilon_0_above_1e_minus_2": bool(eps0_sharp > r0),
        },
        "superseded_bad_split": {
            "formula": "C_chi * ((4/3) K_Sch)^2 * kappa_E * epsilon",
            "status": "not used in the clean D0 public certificate",
            "reason": "The global moving-coordinate inversion inverts once on the whole tube, so the internal K_Sch^2 split is avoided.",
        },
    }


def main() -> None:
    pi_obs = certify_pi_obs()
    commutator = certify_commutator()
    all_pass = (
        pi_obs["angular_orthogonality"]["pass"]
        and pi_obs["extracted_coefficients"]["pass"]
        and pi_obs["constant_mode"]["pass"]
        and commutator["D0_conservative_regime"]["q_comm_less_than_1"]
        and commutator["D0_conservative_regime"]["epsilon_0_above_1e_minus_2"]
    )

    out = {
        "artifact": "phase3_projection_commutator_certificate",
        "generated_by": "scripts/phase3_projection_commutator_certificate.py",
        "scope": "public Phase-3 D0 local certificate",
        "status": "certified_local_inputs" if all_pass else "failed",
        "Pi_obs_constant_mode": pi_obs,
        "R_comm_clean_annular_commutator": commutator,
        "integration_consequence": {
            "C_src_current": "27/16 from private P1 chain",
            "R_aff_D0": 0,
            "R_comm_role": "operator/parametrix Neumann factor, not an additive source envelope in the clean construction",
            "remaining_public_phase3_tasks": [
                "write the full Pi_obs PDE identification theorem beyond the reduced source model",
                "assemble Phase 3.2/3.3 global right inverse theorem",
                "propagate this certificate into theorem_ledger and source residual ledger",
            ],
        },
        "all_pass": all_pass,
    }

    repo_root = Path(__file__).resolve().parents[1]
    out_path = repo_root / "certificates" / "phase3_projection_commutator_certificate.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
