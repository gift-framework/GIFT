#!/usr/bin/env python3
"""
Prototype projector onto the real mu = -1/2 sigma-odd basis.

Goal:
  Given a symbolic collar expression in (rho, varphi), extract the coefficients
  of the lower-root basis
      Phi_{-1/2,R} = rho^(-1/2) cos(varphi/2)
      Phi_{-1/2,I} = rho^(-1/2) sin(varphi/2)
  after multiplying by rho^(1/2).

This is the exact algebraic skeleton needed once a symbolic model for the local
source trace ev_{-1/2}(m(h_bar)) is available.
"""

from __future__ import annotations

import json
from pathlib import Path

import sympy as sp


rho = sp.symbols("rho", positive=True, real=True)
varphi = sp.symbols("varphi", real=True)


def extract_mu_minus_half_coefficients(expr: sp.Expr) -> dict[str, object]:
    """
    Extract the coefficients of rho^(-1/2) cos(varphi/2) and
    rho^(-1/2) sin(varphi/2) from a symbolic expression.

    Method:
      1. Multiply by rho^(1/2).
      2. Take the rho -> 0 limit to isolate the mu = -1/2 coefficient.
      3. Project onto cos(varphi/2), sin(varphi/2).
    """
    weighted = sp.simplify(sp.sqrt(rho) * expr)
    leading = sp.simplify(sp.limit(weighted, rho, 0, dir="+"))

    basis_R = sp.cos(varphi / 2)
    basis_I = sp.sin(varphi / 2)

    # Orthogonal projection on [0, 4pi] for the half-integer sigma-odd basis.
    norm = sp.integrate(basis_R * basis_R, (varphi, 0, 4 * sp.pi))
    coeff_R = sp.simplify(sp.integrate(leading * basis_R, (varphi, 0, 4 * sp.pi)) / norm)
    coeff_I = sp.simplify(sp.integrate(leading * basis_I, (varphi, 0, 4 * sp.pi)) / norm)

    residual = sp.simplify(leading - coeff_R * basis_R - coeff_I * basis_I)

    return {
        "weighted_expr": str(weighted),
        "leading_trace_after_rho_half_weight": str(leading),
        "coeff_Phi_minus_half_R": str(coeff_R),
        "coeff_Phi_minus_half_I": str(coeff_I),
        "projection_residual": str(residual),
    }


def main() -> None:
    a, b = sp.symbols("a b", real=True)

    # Demo ansatz in the exact target channel plus a higher-order tail.
    demo_expr = (
        rho ** sp.Rational(-1, 2) * (a * sp.cos(varphi / 2) + b * sp.sin(varphi / 2))
        + rho ** sp.Rational(3, 2) * sp.cos(3 * varphi / 2)
    )

    extracted = extract_mu_minus_half_coefficients(demo_expr)

    out = {
        "artifact": "phase3_mu_minus_half_projector_proto",
        "status": "symbolic_projection_prototype",
        "basis": {
            "Phi_minus_half_R": "rho^(-1/2) cos(varphi/2)",
            "Phi_minus_half_I": "rho^(-1/2) sin(varphi/2)",
            "integration_domain": "[0, 4pi]",
        },
        "demo_input": str(demo_expr),
        "demo_extraction": extracted,
        "intended_use": (
            "Replace demo_input by a symbolic model for the actual local residual "
            "m(h_loc,aff + h_loc,branch), then read off kappa_R and kappa_I."
        ),
    }

    repo_root = Path(__file__).resolve().parents[1]
    out_path = repo_root / "certificates" / "phase3_mu_minus_half_projector_proto.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
