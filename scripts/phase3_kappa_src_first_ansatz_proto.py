#!/usr/bin/env python3
"""
Prototype first real ansatz for the collar source coefficient kappa_src.

Purpose:
  Freeze the first channel-compatible candidate for the obstruction pair
  (kappa_src,R, kappa_src,I) using:
    - the real scalar seed coefficient c0 from draft Lemma 5.9,
    - the quadratic dependence on c0,
    - the exact lower-root comparison coefficient 3/8,
    - the datum-level lower bound for A_bulk(alpha_1, alpha_1).

This is a conditional ansatz, not a theorem-grade extraction.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    datum_path = repo_root / "certificates" / "datum_D0.json"
    datum = json.loads(datum_path.read_text())

    a_bulk_lb = datum["fields"]["A_bulk_alpha1_alpha1"]["value"][0]
    gamma_geom = Fraction(3, 8)
    kappa_r_candidate_lb = float(gamma_geom) * float(a_bulk_lb)

    out = {
        "artifact": "phase3_kappa_src_first_ansatz_proto",
        "status": "conditional_first_ansatz",
        "scope": "collar_rank_one_constant_mode",
        "inputs": {
            "draft_seed_shape": "h_loc,branch = c0^(i) * w^(3/2) * alpha_1 + ... with c0^(i) real and s-independent",
            "draft_source_shape": "Pi_obs^(i)(m(h_bar)) is quadratic in c0^(i) and s-uniform up to O(kappa_g r0)",
            "comparison_channel_coefficient": "3/8",
            "datum_A_bulk_alpha1_alpha1_lower_bound": a_bulk_lb,
        },
        "leading_real_axis_ansatz": {
            "formula": (
                "Pi_obs^(i)(m(h_bar)) ~= (c0^(i))^2 * "
                "(kappa_src,R^(i) * e_i^R + kappa_src,I^(i) * e_i^I)"
            ),
            "candidate": {
                "kappa_src,R^(i)": "(3/8) * A_bulk(alpha_1, alpha_1)|_{Sigma_i}",
                "kappa_src,I^(i)": "0",
            },
        },
        "covariant_pair_extension": {
            "formula": (
                "Q_quad(c_R, c_I) = gamma_src * A_bulk(alpha_1, alpha_1) * "
                "((c_R)^2 - (c_I)^2, 2 c_R c_I)"
            ),
            "real_axis_specialization": (
                "if c_I = 0, then Q_quad(c0, 0) = "
                "(gamma_src * A_bulk(alpha_1, alpha_1) * c0^2, 0)"
            ),
        },
        "D0_comparison_scale": {
            "gamma_geom": str(gamma_geom),
            "candidate_lower_bound_for_kappa_src_R": kappa_r_candidate_lb,
            "meaning": (
                "At D0, if the source coefficient matches the branch-motion "
                "comparison value 3/8, then the real obstruction coordinate has "
                "scale at least this size."
            ),
        },
        "not_certified": [
            "That gamma_src equals 3/8.",
            "That the true source coefficient has zero imaginary component beyond leading order.",
            "That Pi_obs has already been identified with this explicit quadratic map for all sources.",
        ],
    }

    out_path = repo_root / "certificates" / "phase3_kappa_src_first_ansatz_proto.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
