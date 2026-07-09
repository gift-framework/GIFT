# Phase 4.2: `q_comm_FH_Gf` candidate

## Status

Candidate formula only. This is not a theorem-grade commutator certificate.

Artifacts:

- [phase4_ar_commutator_slot_candidate.py](../scripts/phase4_ar_commutator_slot_candidate.py)
- [phase4_ar_commutator_slot_candidate.json](../certificates/phase4_ar_commutator_slot_candidate.json)
- [phase4_ar_commutator_slot_checker.py](../scripts/phase4_ar_commutator_slot_checker.py)
- [phase4_ar_commutator_slot_check.json](../certificates/phase4_ar_commutator_slot_check.json)

## Formula

The candidate uses

`C_comm_AR = 8 * (1 + K_F) * (1 + K_AR_prod)`

and

`q_comm_FH_Gf = C_comm_AR * eps_AR`.

With

- `K_F = 231/100`;
- `K_AR_prod = 2079/2000`;
- `eps_AR = 1/4000`;

this gives

- `C_comm_AR = 1350149/25000 = 54.00596`;
- `q_comm_FH_Gf = 1350149/100000000 = 0.01350149`.

The Neumann slot target is `1/16 = 0.0625`, so the candidate has margin

`0.04899851`.

## Boundary

The factor `8` is an explicit candidate overhead for Leibniz, cutoff,
collar/base, and gauge-transfer bookkeeping. A real theorem must replace this
with a certified product-norm commutator estimate.

The Neumann slot remains `pending_exact_certificate`.

Checker result:

`16/16` checks pass.
