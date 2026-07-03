#!/usr/bin/env python3
"""
Prototype comparing raw source profile coefficients with lower-root trace data.

Purpose:
  Make explicit that a raw source profile of the form
      rho^(1/2) * (a cos(varphi/2) + b sin(varphi/2))
  is not, by itself, a lower-root trace coefficient in the canonical
  mu = -1/2 basis.

This clarifies the exact gap that remains between the draft's pointwise source
formula and the theorem-grade coefficient kappa_src.
"""

from __future__ import annotations

import json
from pathlib import Path

import sympy as sp


rho = sp.symbols("rho", positive=True, real=True)
varphi = sp.symbols("varphi", real=True)


def project_half_angle(angular_expr: sp.Expr) -> dict[str, str]:
    basis_R = sp.cos(varphi / 2)
    basis_I = sp.sin(varphi / 2)
    norm = sp.integrate(basis_R * basis_R, (varphi, 0, 4 * sp.pi))
    coeff_R = sp.simplify(sp.integrate(angular_expr * basis_R, (varphi, 0, 4 * sp.pi)) / norm)
    coeff_I = sp.simplify(sp.integrate(angular_expr * basis_I, (varphi, 0, 4 * sp.pi)) / norm)
    residual = sp.simplify(angular_expr - coeff_R * basis_R - coeff_I * basis_I)
    return {
        "coeff_R": str(coeff_R),
        "coeff_I": str(coeff_I),
        "residual": str(residual),
    }


def main() -> None:
    a, b = sp.symbols("a b", real=True)

    raw_profile = rho ** sp.Rational(1, 2) * (
        a * sp.cos(varphi / 2) + b * sp.sin(varphi / 2)
    )

    raw_angular = sp.simplify(raw_profile / sp.sqrt(rho))
    raw_projection = project_half_angle(raw_angular)

    # Literal lower-root trace extraction in the canonical convention:
    # multiply by rho^(1/2) and take rho -> 0.
    lower_root_candidate = sp.simplify(sp.sqrt(rho) * raw_profile)
    lower_root_limit = sp.simplify(sp.limit(lower_root_candidate, rho, 0, dir="+"))
    lower_root_projection = project_half_angle(lower_root_limit)

    weighted_profile = sp.simplify(rho**3 * raw_profile)

    out = {
        "artifact": "phase3_profile_to_trace_proto",
        "status": "convention_comparison_prototype",
        "raw_profile_input": str(raw_profile),
        "raw_profile_half_angle_coefficients": raw_projection,
        "literal_lower_root_trace_test": {
            "rho_half_times_profile": str(lower_root_candidate),
            "rho_to_0_limit": str(lower_root_limit),
            "projected_coefficients": lower_root_projection,
            "interpretation": (
                "Under the canonical lower-root trace convention, a pure rho^(1/2) "
                "profile has zero mu=-1/2 trace coefficient."
            ),
        },
        "weighted_quantity": {
            "rho3_times_profile": str(weighted_profile),
            "interpretation": (
                "The same raw profile contributes nontrivially to the weighted "
                "Y_-3 bookkeeping even though its literal lower-root trace is zero."
            ),
        },
    }

    repo_root = Path(__file__).resolve().parents[1]
    out_path = repo_root / "certificates" / "phase3_profile_to_trace_proto.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
