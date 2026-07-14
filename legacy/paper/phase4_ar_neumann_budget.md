# Phase 4.2: AR Neumann budget

> **2026-07-09 status.** Historical public Neumann-slot scaffold. Stage D
> imports the theorem-grade L2 assembly for D0; this note is retained as the
> public candidate trail and is no longer the active reconstruction theorem
> citation. Use `certificates/datum_D0.json`, `docs/stageD_recap_2026_07_09.md`,
> and `docs/stageE_recap_2026_07_09.md` for current D0 bookkeeping.

## Status

Budget skeleton only. This is not a product-space reconstruction theorem.

Artifacts:

- [phase4_ar_neumann_budget_candidate.py](../../scripts/phase4_ar_neumann_budget_candidate.py)
- [phase4_ar_neumann_budget_candidate.json](../../certificates/phase4_ar_neumann_budget_candidate.json)
- [phase4_ar_neumann_budget_checker.py](../../scripts/phase4_ar_neumann_budget_checker.py)
- [phase4_ar_neumann_budget_check.json](../../certificates/phase4_ar_neumann_budget_check.json)

## Purpose

The scalar AR majorant already gives

`q_AR_scalar = 0.001034633607583419`

at `R_AR = 4000` with the repaired product-max constant

`K_AR_prod = 2079/2000`.

The missing analytic work is now isolated into four Neumann-error slots:

- `q_hodge_uniform`;
- `q_comm_FH_Gf`;
- `q_projection_residual`;
- `q_gauge_transfer`.

## Budget

The initial sufficient budget assigns each slot the target

`1/16`.

Thus

`q_neumann_total_target = 1/4`

and

`q_AR_scalar + q_neumann_total_target < 1`.

This is intentionally conservative: it leaves a large contraction margin while
forcing every missing analytic estimate to be named and independently checked.

## Current State

All four slots are still

`pending_exact_certificate`.

No `q_neumann_total` is serialized yet, and `promotion_ready = false`.

The first slot, `q_comm_FH_Gf`, now has a checked candidate formula:

- [phase4_ar_commutator_slot_candidate.md](phase4_ar_commutator_slot_candidate.md)
- [phase4_ar_commutator_slot_candidate.json](../../certificates/phase4_ar_commutator_slot_candidate.json)

The candidate gives `q_comm_FH_Gf = 0.01350149 < 1/16`, but does not close the
slot.

The three remaining slots also have checked candidate formulas:

- [phase4_ar_remaining_slots_candidate.md](phase4_ar_remaining_slots_candidate.md)
- [phase4_ar_remaining_slots_candidate.json](../../certificates/phase4_ar_remaining_slots_candidate.json)

Candidate values:

- `q_hodge_uniform = 0.0058`;
- `q_projection_residual = 0.016316`;
- `q_gauge_transfer = 0.0061185`.

These candidates are all below `1/16`, but do not close their slots.

Checker result:

`41/41` checks pass.

The checker verifies:

- dependencies on the product-space contract, scalar majorant, and inverse audit;
- exact agreement of `K_AR_prod` and `q_AR_scalar`;
- the four expected Neumann slots are present;
- each slot is pending and has no fake numerical value;
- the target budget would still be a contraction;
- AR promotion is explicitly blocked until all slots are certified.

## Next Analytic Target

The first slot to attack should be `q_comm_FH_Gf`, because it is the most
concrete operator estimate:

`q_comm_FH_Gf = C_comm_AR * eps_AR`.

At `eps_AR = 1/4000`, the budget target `1/16` allows

`C_comm_AR <= 250`.

That is a generous first threshold; sharper constants can be substituted later.
