#!/usr/bin/env python3
"""
Phase 3 Task 4: effective Jacobi parametrix candidate for D0.

This is a reproducible numerical/symbolic experiment, not a theorem.

It freezes the finite-mode normal-operator calculation behind the working
Mazzeo/Schauder constant and separates:

  1. the exact scalar indicial inverse in the half-integer sigma-odd sector;
  2. the rank-19 twist transfer through cond(A_bulk);
  3. an explicit edge/Schauder overhead slot.

The output is a candidate bound for K_Sch^Maz(D0), plus the induced rank-19
Jacobi candidate. It must not be cited as a theorem until replaced by a
genuine edge-calculus Schauder proof with outward-rounded intervals.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path


def half_integer_modes(max_k: int) -> list[Fraction]:
    """Positive half-integer angular modes 1/2, 3/2, ... ."""
    return [Fraction(2 * k + 1, 2) for k in range(max_k + 1)]


def indicial_inverse_constant(beta: Fraction, max_k: int, max_q: int, r0: Fraction) -> dict:
    """
    Finite-mode scan for the conjugated scalar normal operator.

    Model denominator:

        |beta^2 - m^2| + (q r0)^2

    The q=0, m=1/2 mode is the known worst mode at beta=1, giving 4/3.
    Positive longitudinal q only improves the denominator in the D0 scan.
    """
    worst = None
    records = []
    for m in half_integer_modes(max_k):
        for q in range(max_q + 1):
            denom = abs(beta * beta - m * m) + (Fraction(q, 1) * r0) ** 2
            if denom == 0:
                inv = None
            else:
                inv = Fraction(1, 1) / denom
            record = {
                "m": str(m),
                "q": q,
                "denominator": str(denom),
                "inverse": None if inv is None else str(inv),
            }
            records.append(record)
            if inv is not None and (worst is None or inv > worst["inverse_fraction"]):
                worst = {
                    "m": str(m),
                    "q": q,
                    "denominator": str(denom),
                    "inverse": str(inv),
                    "inverse_float": float(inv),
                    "inverse_fraction": inv,
                }

    return {
        "beta": str(beta),
        "max_k": max_k,
        "max_q": max_q,
        "r0": str(r0),
        "worst_mode": {k: v for k, v in worst.items() if k != "inverse_fraction"},
        "expected_exact_worst": {
            "mode": "m=1/2, q=0",
            "denominator": "3/4",
            "inverse": "4/3",
            "pass": bool(worst["inverse_fraction"] == Fraction(4, 3)),
        },
        "sample_records_first_12": records[:12],
    }


def main() -> None:
    beta = Fraction(1, 1)
    r0 = Fraction(1, 100)
    max_k = 12
    max_q = 12
    cond_a_bulk = Fraction(231, 100)

    normal_scan = indicial_inverse_constant(beta=beta, max_k=max_k, max_q=max_q, r0=r0)
    k_ind = Fraction(4, 3)

    # This is deliberately exposed as a calibration slot, not hidden as a proof.
    # C_edge_overhead=4 says: candidate scalar Mazzeo/Schauder constant =
    # four times the exact indicial inverse. The current public envelope 17 is
    # retained as the theorem-safe value until an edge proof replaces this.
    c_edge_overhead = Fraction(4, 1)
    k_sch_candidate = k_ind * c_edge_overhead
    k_rank19_candidate = k_sch_candidate * cond_a_bulk

    current_public_envelope = Fraction(17, 1)
    q_comm_with_candidate = Fraction(4, 3) * k_sch_candidate * Fraction(3, 1) * r0
    eps0_with_candidate = Fraction(1, 1) / (Fraction(4, 3) * k_sch_candidate * Fraction(3, 1))

    out = {
        "artifact": "phase3_effective_jacobi_parametrix_candidate",
        "generated_by": "scripts/phase3_effective_jacobi_parametrix_candidate.py",
        "scope": "collar finite-mode normal-operator experiment plus D0 bookkeeping",
        "status": "candidate_not_theorem",
        "normal_operator_scan": normal_scan,
        "candidate_constants": {
            "K_ind_exact": {
                "value": float(k_ind),
                "exact": str(k_ind),
                "meaning": "Worst scalar indicial inverse in the sigma-odd half-integer sector at beta=1.",
            },
            "C_edge_overhead_candidate": {
                "value": float(c_edge_overhead),
                "exact": str(c_edge_overhead),
                "meaning": "Explicit calibration slot for local Schauder/Holder overhead; not theorem-grade.",
            },
            "K_Sch_Maz_candidate_scalar": {
                "value": float(k_sch_candidate),
                "exact": str(k_sch_candidate),
                "compared_to_current_public_envelope_17": float(current_public_envelope / k_sch_candidate),
            },
            "cond_A_bulk_D0": {
                "value": float(cond_a_bulk),
                "exact": str(cond_a_bulk),
            },
            "K_Jacobi_rank19_candidate": {
                "value": float(k_rank19_candidate),
                "exact": str(k_rank19_candidate),
                "meaning": "Candidate after the rank-19 twist transfer by cond(A_bulk).",
            },
        },
        "D0_implications_if_candidate_were_proved": {
            "q_comm_formula": "(4/3) * K_Sch_candidate * kappa_E * epsilon",
            "using_kappa_E": 3.0,
            "using_epsilon": float(r0),
            "q_comm": float(q_comm_with_candidate),
            "epsilon_0_lower_bound": float(eps0_with_candidate),
        },
        "current_safe_value": {
            "K_Sch_Maz_public_envelope": 17.0,
            "status": "still the theorem-safe public envelope",
            "reason": "The candidate has not been upgraded to a genuine edge Schauder proof.",
        },
        "acceptance_requirements_to_promote": [
            "replace C_edge_overhead_candidate by an edge-calculus Schauder proof",
            "prove coefficient perturbation bounds for the actual J_h on each collar",
            "show uniformity over all 77 collars with outward-rounded intervals",
            "independently check the serialized interval certificate",
        ],
        "all_pass": bool(normal_scan["expected_exact_worst"]["pass"]),
    }

    repo_root = Path(__file__).resolve().parents[1]
    out_path = repo_root / "certificates" / "phase3_effective_jacobi_parametrix_candidate.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
