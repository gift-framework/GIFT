#!/usr/bin/env python3
"""
Prototype Phase 3 local coefficient projection.

Purpose:
  Convert a lower-root source coefficient in the fixed basis
      Phi_{-1/2,R}, Phi_{-1/2,I}
  into the corresponding cubic corrective datum in the upper-root basis
      psi_R, psi_I
  using the model identity
      J_h(psi_*) = 2 * A_bulk(alpha_1, alpha_1) * Phi_{-1/2,*}
  on the rank-one alpha_1 direction.

This is not yet the extraction of kappa_src itself. It is the exact algebraic
map that will be applied once the source coefficient has been computed.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path


def main() -> None:
    two = Fraction(2, 1)
    half = Fraction(1, 2)

    out = {
        "artifact": "phase3_kappa_src_projection_proto",
        "status": "algebraic_interface_prototype",
        "fixed_bases": {
            "lower_root": [
                "Phi_{-1/2,R} = rho^(-1/2) cos(varphi/2)",
                "Phi_{-1/2,I} = rho^(-1/2) sin(varphi/2)",
            ],
            "upper_root": [
                "psi_R = rho^(3/2) cos(varphi/2)",
                "psi_I = rho^(3/2) sin(varphi/2)",
            ],
        },
        "model_identity": {
            "J_h(psi_R)": "2 * A_bulk(alpha_1, alpha_1) * Phi_{-1/2,R}",
            "J_h(psi_I)": "2 * A_bulk(alpha_1, alpha_1) * Phi_{-1/2,I}",
            "scalar_factor": str(two),
        },
        "source_to_correction_map": {
            "source_input": (
                "ev_{-1/2}(m(h_bar)) = c0^2 * "
                "(kappa_R * Phi_{-1/2,R} + kappa_I * Phi_{-1/2,I})"
            ),
            "corrective_upper_root_datum": (
                "v0 = - c0^2 / (2 * A_bulk(alpha_1, alpha_1)) * "
                "(kappa_R * psi_R + kappa_I * psi_I)"
            ),
            "per_coordinate_multiplier": (
                "-1 / (2 * A_bulk(alpha_1, alpha_1))"
            ),
        },
        "normalized_A5_specialization": {
            "if_A_bulk_alpha1_alpha1_equals_1": {
                "coordinate_multiplier": str(-half),
                "meaning": (
                    "In the normalized model, one unit of lower-root source "
                    "coefficient is cancelled by minus one-half unit of cubic "
                    "upper-root datum in the matching coordinate."
                ),
            }
        },
        "what_this_does_not_do": [
            "It does not compute kappa_R or kappa_I.",
            "It does not bound the O(rho^(3/2)) tail.",
            "It does not certify the full nonlinear source term.",
        ],
    }

    repo_root = Path(__file__).resolve().parents[1]
    out_path = repo_root / "certificates" / "phase3_kappa_src_projection_proto.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
