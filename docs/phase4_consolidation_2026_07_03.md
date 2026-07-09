# Phase 4 consolidation -- 2026-07-03

## Scope

This document consolidates the Phase 4 work added after
[phase4_progress_2026_07_03.md](phase4_progress_2026_07_03.md).

It is a status document, not a theorem. The authoritative claim boundary remains
[theorem_ledger.yaml](../audit/theorem_ledger.yaml).

## Executive Status

P4.1 coefficient extraction at `D0` is now closed at the public certificate
level.

P4.2 is not discharged. It now has a checked product-space contract, a repaired
product-max scalar normalization, a checked scalar majorant, and a full
Neumann-budget interface with candidate values for every missing slot.

The remaining P4.2 work is analytic theorem work:

- uniform fibrewise inverse in the product norm;
- product-norm commutator theorem;
- global reduced-projection identity;
- gauge-transfer and closedness-preservation theorem.

## P4.1 Stage D/E

Stage D patched `datum_D0.json` with the public Phase 4.1 values:

- `K_H_K3 = 9/20 = 0.45`;
- `source_P1 = 6237/2000 = 3.1185`;
- `source_P2 = 25440723/4000000 = 6.36018075`;
- `DP1_norm = 6237/2000 = 3.1185`;
- `DP2_norm = 16796241/2000000 = 8.3981205`;
- `D2P1_norm = 6237/2000 = 3.1185`;
- `D3m_norm = 4/9`;
- `raw_P3_scale = 12985939039/4000000000 = 3.24648475975`;
- `remainder_R3 = 13770793.1700815927923`;
- `R_threshold = 3664.0659853300026`.

`R_threshold` is a lower admissibility threshold: the tail condition holds for
`R >= R_threshold`.

Stage E upgraded the checker to outward-rounded interval recomputation.

Artifacts:

- [phase4_donaldson_coefficients_values.json](../certificates/phase4_donaldson_coefficients_values.json)
- [phase4_donaldson_coefficients_check.json](../certificates/phase4_donaldson_coefficients_check.json)
- [phase4_donaldson_coefficients_checker_stageC_v1.md](../paper/phase4_donaldson_coefficients_checker_stageC_v1.md)
- [phase4_donaldson_coefficients_values.md](../paper/phase4_donaldson_coefficients_values.md)

Current checker result:

`172/172` checks pass.

## Phase 3 Source Reconciliation

The old `3/8` source surrogate is no longer active for the fixed-Sigma
sigma-odd source channel.

Current public boundary:

- active fixed-Sigma sigma-odd coefficient: `C_src = 27/16`;
- legacy `3/8`: lower-root normal-form / regular-sector diagnostic only;
- no Phase 3 global maximal-section theorem is promoted by this bookkeeping.

Artifacts:

- [phase3_gamma_src_surrogate.md](../paper/phase3_gamma_src_surrogate.md)
- [phase3_kappa_src_first_ansatz.md](../paper/phase3_kappa_src_first_ansatz.md)
- [phase3_gamma_src_surrogate_proto.json](../certificates/phase3_gamma_src_surrogate_proto.json)
- [phase3_kappa_src_first_ansatz_proto.json](../certificates/phase3_kappa_src_first_ansatz_proto.json)

## P4.2 Product-Space Contract

The AR product-space contract fixes

`A_beta = A_omega x A_lambda x A_mu x A_Theta`

with component bidegrees:

- `omega in Omega^{1,2}`;
- `lambda in Omega^{3,0}`;
- `mu in Omega^{0,4}`;
- `Theta in Omega^{2,2}`.

It also fixes the block architecture:

- `d_f lambda = -F_H omega`, using `-G_f F_H`;
- `d_f Theta = -F_H mu`, using `-G_f F_H`;
- `d_H Theta = 0`, reduced through `Pi_reduced d_H`.

Artifacts:

- [phase4_ar_product_space_contract.json](../certificates/phase4_ar_product_space_contract.json)
- [phase4_ar_product_space_check.json](../certificates/phase4_ar_product_space_check.json)
- [phase4_ar_product_space_contract.md](../paper/phase4_ar_product_space_contract.md)

Checker result:

`36/36` checks pass.

This defines the product-space contract. It does not prove the uniform inverse
theorem.

## Product-Max Normalization

The earlier scalar choice `K_AR_prod = 9/20` was too small for the declared
product-max block architecture. The repaired structural value is

`K_AR_prod = K_H_K3 * K_F = 2079/2000 = 1.0395`.

This dominates:

- `G_f`;
- the `G_f F_H` block;
- the projection slot.

Artifacts:

- [phase4_ar_inverse_budget_audit.json](../certificates/phase4_ar_inverse_budget_audit.json)
- [phase4_ar_inverse_budget_check.json](../certificates/phase4_ar_inverse_budget_check.json)
- [phase4_ar_inverse_budget_audit.md](../paper/phase4_ar_inverse_budget_audit.md)

Checker result:

`16/16` checks pass.

This repairs the scalar normalization. It does not prove the analytic AR
theorem obligations.

## Scalar AR Majorant

With the repaired product-max constant, the scalar majorant at `R_AR = 4000`
is:

- `eps_AR = 1/4000`;
- `r_AR = 1/100`;
- `source_majorant = 0.0009951911545794`;
- `displacement_majorant = 0.0010345012051852862`;
- `q_AR = 0.001034633607583419`.

Artifacts:

- [phase4_ar_majorant_candidate.json](../certificates/phase4_ar_majorant_candidate.json)
- [phase4_ar_majorant_check.json](../certificates/phase4_ar_majorant_check.json)
- [phase4_ar_majorant_candidate.md](../paper/phase4_ar_majorant_candidate.md)

Checker result:

`20/20` checks pass.

This is still conditional on the missing product-space analytic theorem.

## Neumann Budget

The remaining AR theorem gap is decomposed into four Neumann-error slots:

- `q_hodge_uniform`;
- `q_comm_FH_Gf`;
- `q_projection_residual`;
- `q_gauge_transfer`.

Each slot has target `1/16`, hence

`q_neumann_total_target = 1/4`.

Together with the scalar majorant,

`q_AR_scalar + q_neumann_total_target = 0.25103463360758344 < 1`.

Artifacts:

- [phase4_ar_neumann_budget_candidate.json](../certificates/phase4_ar_neumann_budget_candidate.json)
- [phase4_ar_neumann_budget_check.json](../certificates/phase4_ar_neumann_budget_check.json)
- [phase4_ar_neumann_budget.md](../paper/phase4_ar_neumann_budget.md)

Checker result:

`41/41` checks pass.

Stage D repin (2026-07-09): the private 2026-07-04 V/W/X/Y theorems
discharge all four slots, and `axis2_L2_assembly_theorem_2026_07_05`
assembles them into `q_total = 26236977/3200000000 ~= 8.20e-3 < 1/2`
under the product-max norm. The original candidate values below remain
historical Codex audit witnesses, not the active status.

## Slot Candidates

Every Neumann slot now has a checked candidate value below `1/16`.

| Slot | Candidate value | Status |
|---|---:|---|
| `q_hodge_uniform` | `0.0058` | candidate only |
| `q_comm_FH_Gf` | `0.01350149` | candidate only |
| `q_projection_residual` | `0.016316` | candidate only |
| `q_gauge_transfer` | `0.0061185` | candidate only |

Artifacts:

- [phase4_ar_commutator_slot_candidate.json](../certificates/phase4_ar_commutator_slot_candidate.json)
- [phase4_ar_commutator_slot_check.json](../certificates/phase4_ar_commutator_slot_check.json)
- [phase4_ar_commutator_slot_candidate.md](../paper/phase4_ar_commutator_slot_candidate.md)
- [phase4_ar_remaining_slots_candidate.json](../certificates/phase4_ar_remaining_slots_candidate.json)
- [phase4_ar_remaining_slots_check.json](../certificates/phase4_ar_remaining_slots_check.json)
- [phase4_ar_remaining_slots_candidate.md](../paper/phase4_ar_remaining_slots_candidate.md)

Checker results:

- commutator slot: `16/16`;
- remaining slots: `32/32`.

These are proof targets, not theorem-grade estimates.

## Current Proof Boundary

The following statements are now public and checked:

- P4.1 coefficient values at `D0`;
- outward-rounded interval packaging for P4.1 scalar values;
- product-space bidegree and block-inverse contract;
- product-max normalization `K_AR_prod = 2079/2000`;
- scalar AR majorant at `R_AR = 4000`;
- Neumann-budget decomposition;
- candidate values for all four Neumann slots.

The following statements are not proved:

- uniform fibrewise inverse in the product norm;
- product-norm commutator theorem;
- global reduced-projection identity;
- gauge-transfer theorem;
- closedness preservation;
- convergent adiabatic reconstruction theorem;
- anisotropic Joyce theorem `(J)`;
- Level E compact torsion-free `G2` existence.

## Next Best Target

The most concrete next target is to turn the candidate

`q_comm_FH_Gf = C_comm_AR * eps_AR`

into a theorem-grade commutator certificate.

The current candidate has

`C_comm_AR = 54.00596`.

The slot budget allows any certified

`C_comm_AR <= 250`.

That gives a generous first threshold for a real product-norm commutator proof.
