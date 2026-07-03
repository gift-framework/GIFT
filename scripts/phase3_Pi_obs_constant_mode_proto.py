#!/usr/bin/env python3
"""
Prototype operational model for the constant-section obstruction projection.

Purpose:
  Implement the simplest honest version of the map
      Pi_obs : Y_{beta-2} -> R^{2N}
  in the regime where the source has already been reduced to its lower-root
  trace form on one collar:

      rho^(-1/2) (a(s) cos(varphi/2) + b(s) sin(varphi/2)) + higher terms.

The prototype does two things:
  1. extract the lower-root coefficient functions a(s), b(s),
  2. project them to the constant longitudinal mode by averaging over s.

This is not the full PDE obstruction projection. It is the operational model
used by the current Phase 3 branch before the theorem-level identification of
Pi_obs is written.
"""

from __future__ import annotations

import json
from pathlib import Path

import sympy as sp


rho = sp.symbols("rho", positive=True, real=True)
varphi = sp.symbols("varphi", real=True)
s = sp.symbols("s", real=True)


def extract_lower_root_coeff_functions(expr: sp.Expr) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """
    Extract a(s), b(s) from the lower-root sigma-odd channel

        rho^(-1/2) (a(s) cos(varphi/2) + b(s) sin(varphi/2)).

    Returns (a_of_s, b_of_s, residual_after_projection).
    """
    weighted = sp.simplify(sp.sqrt(rho) * expr)
    leading = sp.simplify(sp.limit(weighted, rho, 0, dir="+"))

    basis_R = sp.cos(varphi / 2)
    basis_I = sp.sin(varphi / 2)
    norm = sp.integrate(basis_R * basis_R, (varphi, 0, 4 * sp.pi))

    coeff_R = sp.simplify(sp.integrate(leading * basis_R, (varphi, 0, 4 * sp.pi)) / norm)
    coeff_I = sp.simplify(sp.integrate(leading * basis_I, (varphi, 0, 4 * sp.pi)) / norm)
    residual = sp.simplify(leading - coeff_R * basis_R - coeff_I * basis_I)
    return coeff_R, coeff_I, residual


def constant_mode_average(coeff: sp.Expr, period: sp.Expr) -> sp.Expr:
    """Average a longitudinal coefficient over one s-period."""
    return sp.simplify(sp.integrate(coeff, (s, 0, period)) / period)


def main() -> None:
    period = 2 * sp.pi

    # Demo source: lower-root obstruction part with one constant and one
    # oscillatory longitudinal component, plus a higher-root tail.
    a_of_s = 2 + sp.cos(s)
    b_of_s = 5 + 3 * sp.sin(s)
    demo_expr = (
        rho ** sp.Rational(-1, 2)
        * (a_of_s * sp.cos(varphi / 2) + b_of_s * sp.sin(varphi / 2))
        + rho ** sp.Rational(3, 2) * sp.cos(3 * varphi / 2)
    )

    coeff_R, coeff_I, residual = extract_lower_root_coeff_functions(demo_expr)
    avg_R = constant_mode_average(coeff_R, period)
    avg_I = constant_mode_average(coeff_I, period)

    out = {
        "artifact": "phase3_Pi_obs_constant_mode_proto",
        "status": "operational_constant_mode_model",
        "scope": "single_collar_rank_one_demo",
        "input_regime": {
            "assumed_source_form": (
                "rho^(-1/2) (a(s) cos(varphi/2) + b(s) sin(varphi/2)) + higher terms"
            ),
            "longitudinal_period_used_in_demo": str(period),
        },
        "demo_input": str(demo_expr),
        "extracted_lower_root_coeff_functions": {
            "a_of_s": str(coeff_R),
            "b_of_s": str(coeff_I),
            "projection_residual": str(residual),
        },
        "constant_mode_projection": {
            "Pi_const(a_of_s)": str(avg_R),
            "Pi_const(b_of_s)": str(avg_I),
            "R2_coordinate_output": {
                "e_R": str(avg_R),
                "e_I": str(avg_I),
            },
        },
        "interpretation": {
            "meaning": (
                "This is the prototype rule for Pi_obs once the source has been "
                "reduced to its lower-root trace representation."
            ),
            "not_a_claim": (
                "This does not construct the full PDE source projection or prove "
                "that every source reduces to this model without further work."
            ),
        },
    }

    repo_root = Path(__file__).resolve().parents[1]
    out_path = repo_root / "certificates" / "phase3_Pi_obs_constant_mode_proto.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
