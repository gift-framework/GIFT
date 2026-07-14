# Phase 4.2: remaining Neumann-slot candidates

> **2026-07-09 status.** Historical candidate slots. Stage D imports the
> theorem-grade L2 assembly for D0; this file remains a reproducibility trail
> for the public candidate formulas only.

## Status

Candidate formulas only. These are not theorem-grade analytic certificates.

Artifacts:

- [phase4_ar_remaining_slots_candidate.py](../../scripts/phase4_ar_remaining_slots_candidate.py)
- [phase4_ar_remaining_slots_candidate.json](../../certificates/phase4_ar_remaining_slots_candidate.json)
- [phase4_ar_remaining_slots_checker.py](../../scripts/phase4_ar_remaining_slots_checker.py)
- [phase4_ar_remaining_slots_check.json](../../certificates/phase4_ar_remaining_slots_check.json)

## Candidate Values

For `q_hodge_uniform`:

- `C_q_hodge_uniform = 116/5 = 23.2`;
- `q_hodge_uniform = 29/5000 = 0.0058`.

For `q_projection_residual`:

- `C_q_projection_residual = 8158/125 = 65.264`;
- `q_projection_residual = 4079/250000 = 0.016316`.

For `q_gauge_transfer`:

- `C_q_gauge_transfer = 12237/500 = 24.474`;
- `q_gauge_transfer = 12237/2000000 = 0.0061185`.

Each candidate is below the Neumann slot target `1/16 = 0.0625`.

## Boundary

These formulas are proof targets, not proofs. The corresponding Neumann slots
remain `pending_exact_certificate`.

Required theorem artifacts:

- uniform fibrewise inverse theorem in the product norm;
- global reduced-projection identity;
- gauge-transfer and closedness-preservation theorem.

Checker result:

`32/32` checks pass.
