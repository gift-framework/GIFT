#!/usr/bin/env python3
"""
Legacy first ansatz for kappa_src, reconciled with the sharp source channel.

The old comparison ansatz used the lower-root normal-form coefficient 3/8 as a
quadratic source model. The 2026-07-02 parity reconciliation supersedes that
as the active fixed-Sigma sigma-odd source model: the relevant source is cubic
with theorem-grade coefficient C_src = 27/16. The old ansatz remains useful only
as a diagnostic for the alpha1-perp regular sigma-even sector.
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
    c_src = Fraction(27, 16)
    legacy_kappa_r_candidate_lb = float(gamma_geom) * float(a_bulk_lb)

    out = {
        "artifact": "phase3_kappa_src_first_ansatz_proto",
        "status": "legacy_comparison_superseded_for_sigma_odd_source",
        "scope": "collar_rank_one_constant_mode",
        "inputs": {
            "draft_seed_shape": "h_loc,branch = c0^(i) * w^(3/2) * alpha_1 + ... with c0^(i) real and s-independent",
            "legacy_draft_source_shape": "Pi_obs^(i)(m(h_bar)) was tested as quadratic in c0^(i) and s-uniform up to O(kappa_g r0)",
            "legacy_comparison_channel_coefficient": "3/8",
            "active_sigma_odd_source_coefficient": "27/16",
            "datum_A_bulk_alpha1_alpha1_lower_bound": a_bulk_lb,
        },
        "legacy_leading_real_axis_ansatz": {
            "formula": (
                "Pi_obs^(i)(m(h_bar)) ~= (c0^(i))^2 * "
                "(kappa_src,R^(i) * e_i^R + kappa_src,I^(i) * e_i^I)"
            ),
            "candidate": {
                "kappa_src,R^(i)": "(3/8) * A_bulk(alpha_1, alpha_1)|_{Sigma_i}",
                "kappa_src,I^(i)": "0",
            },
            "current_status": (
                "superseded for the fixed-Sigma sigma-odd source channel; retained "
                "only as a comparison for lower-root normal forms or regular-sector diagnostics"
            ),
        },
        "active_sigma_odd_channel": {
            "coefficient": {
                "name": "C_src",
                "exact": str(c_src),
                "value": float(c_src),
            },
            "degree_in_c0": 3,
            "source_channel": "fixed-Sigma sigma-odd mu=-1/2 obstruction channel",
            "provenance": "private/canonical/notes/axis2_codex_reconciliation_2026_07_02.md and axis2_gamma_src_Yneg3_norm_2026_07_02",
            "parity_rule": "quadratic-in-c0 contribution is alpha1-perp regular sigma-even, not sigma-odd mu=-1/2",
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
            "legacy_candidate_lower_bound_for_kappa_src_R": legacy_kappa_r_candidate_lb,
            "meaning": (
                "At D0, this is only the old lower-root comparison scale. It is "
                "not the active fixed-Sigma sigma-odd source coefficient."
            ),
        },
        "not_certified": [
            "That gamma_src equals 3/8 in the fixed-Sigma sigma-odd source channel.",
            "That the true fixed-Sigma source is quadratic in c0.",
            "That Pi_obs has already been identified with this explicit quadratic map for all sources.",
        ],
    }

    out_path = repo_root / "certificates" / "phase3_kappa_src_first_ansatz_proto.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
