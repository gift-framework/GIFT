#!/usr/bin/env python3
"""
Reconcile the old quadratic gamma_src surrogate with the sharp source channel.

Purpose:
  Keep the exact lower-root normal-form coefficient 3/8 on file as a legacy
  comparison value, while making the active fixed-Sigma sigma-odd source
  coefficient theorem-grade:

      C_src = 27/16.

The parity reconciliation says the quadratic-in-c0 contribution belongs to the
alpha1-perp regular sigma-even sector, not to the sigma-odd mu=-1/2 obstruction
channel. This file therefore supersedes the old gamma_src,sur certificate.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    datum = json.loads((repo_root / "certificates" / "datum_D0.json").read_text())

    r0 = datum["fields"]["r0"]["value"][0]

    c_src = Fraction(27, 16)
    legacy_gamma_sur = Fraction(3, 8)
    source_scale_bound = float(c_src) * (float(r0) ** 3.5)

    out = {
        "artifact": "phase3_gamma_src_surrogate_proto",
        "status": "superseded_by_sigma_odd_cubic_source",
        "scope": "fixed_Sigma_sigma_odd_mu_minus_half_source_channel_at_D0",
        "active_source_coefficient": {
            "name": "C_src",
            "exact": str(c_src),
            "value": float(c_src),
            "degree_in_c0": 3,
            "channel": "fixed-Sigma sigma-odd obstruction channel",
            "provenance": "private/canonical/notes/axis2_codex_reconciliation_2026_07_02.md and axis2_gamma_src_Yneg3_norm_2026_07_02",
            "status": "theorem_grade_current_public_ledger_value",
        },
        "legacy_quadratic_comparison": {
            "name": "gamma_src_sur",
            "exact": str(legacy_gamma_sur),
            "value": float(legacy_gamma_sur),
            "status": "legacy_comparison_not_active_source_coefficient",
            "valid_scope": "lower-root branch-motion normal-form comparison and possible alpha1-perp regular-sector diagnostic only",
        },
        "parity_reconciliation": {
            "statement": (
                "The quadratic-in-c0 contribution is assigned to the alpha1-perp "
                "regular sigma-even sector, while the fixed-Sigma sigma-odd "
                "obstruction source is cubic with C_src = 27/16."
            ),
            "confirmed_bridge_identity": "J_h(psi_R) = 2 * A_bulk * Phi_{-1/2,R}",
            "cover_side_conformal_factor": 4,
        },
        "D0_active_scale": {
            "r0": r0,
            "C_src": str(c_src),
            "weighted_source_scale_upper_envelope": source_scale_bound,
        },
        "do_not_use_as": [
            "a proof that gamma_src equals 3/8 in the fixed-Sigma sigma-odd source channel",
            "a replacement for the private theorem-grade extraction C_src = 27/16",
            "a global Phase-3 parametrix or maximal-section theorem",
        ],
    }

    out_path = repo_root / "certificates" / "phase3_gamma_src_surrogate_proto.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
