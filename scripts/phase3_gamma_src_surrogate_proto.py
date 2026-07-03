#!/usr/bin/env python3
"""
Explicit local quadratic surrogate for gamma_src.

Purpose:
  Package the smallest concrete surrogate for the leading source coefficient in
  the rank-one constant-section channel:

      gamma_src,sur = 3/8

  with the corresponding real-axis obstruction pair and D0 comparison scale.

This is a modeling surrogate, not an extracted theorem-grade coefficient.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    datum = json.loads((repo_root / "certificates" / "datum_D0.json").read_text())

    a_bulk_lb = datum["fields"]["A_bulk_alpha1_alpha1"]["value"][0]
    r0 = datum["fields"]["r0"]["value"][0]

    gamma_sur = Fraction(3, 8)
    kappa_r_lb = float(gamma_sur) * float(a_bulk_lb)
    source_scale_lb = kappa_r_lb * (float(r0) ** 3.5)

    out = {
        "artifact": "phase3_gamma_src_surrogate_proto",
        "status": "explicit_local_surrogate",
        "scope": "rank_one_constant_mode",
        "surrogate_assumption": {
            "id": "S_qsur",
            "statement": (
                "The leading obstruction projection of the quadratic seed residual "
                "uses the same lower-root channel coefficient as the certified "
                "branch-motion normal form."
            ),
        },
        "surrogate_output": {
            "gamma_src_sur": str(gamma_sur),
            "real_axis_pair": {
                "kappa_src_R": "(3/8) * A_bulk(alpha_1, alpha_1)",
                "kappa_src_I": "0",
            },
            "covariant_pair_extension": (
                "(3/8) * A_bulk(alpha_1, alpha_1) * "
                "((c_R)^2 - (c_I)^2, 2 c_R c_I)"
            ),
        },
        "D0_specialization": {
            "A_bulk_alpha1_alpha1_lower_bound": a_bulk_lb,
            "r0": r0,
            "candidate_lower_bound_for_kappa_src_R": kappa_r_lb,
            "candidate_lower_bound_for_weighted_source_scale": source_scale_lb,
        },
        "not_certified": [
            "That the true nonlinear source coefficient equals 3/8.",
            "That the full PDE obstruction projection coincides with this surrogate on all inputs.",
        ],
    }

    out_path = repo_root / "certificates" / "phase3_gamma_src_surrogate_proto.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
