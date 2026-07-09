# Phase 4.2: global adiabatic reconstruction theorem specification

## Status

Specification only. This is not a proof of adiabatic reconstruction.

P4.1 coefficient extraction at `D0` is closed by:

- [phase4_donaldson_coefficients_values.json](../certificates/phase4_donaldson_coefficients_values.json)
- [phase4_donaldson_coefficients_check.json](../certificates/phase4_donaldson_coefficients_check.json)

The product-space definition layer is now fixed by:

- [phase4_ar_product_space_contract.json](../certificates/phase4_ar_product_space_contract.json)
- [phase4_ar_product_space_check.json](../certificates/phase4_ar_product_space_check.json)

The Neumann-error budget layer is fixed by:

- [phase4_ar_neumann_budget_candidate.json](../certificates/phase4_ar_neumann_budget_candidate.json)
- [phase4_ar_neumann_budget_check.json](../certificates/phase4_ar_neumann_budget_check.json)

The first commutator-slot candidate is recorded by:

- [phase4_ar_commutator_slot_candidate.json](../certificates/phase4_ar_commutator_slot_candidate.json)
- [phase4_ar_commutator_slot_check.json](../certificates/phase4_ar_commutator_slot_check.json)

The remaining slot candidates are recorded by:

- [phase4_ar_remaining_slots_candidate.json](../certificates/phase4_ar_remaining_slots_candidate.json)
- [phase4_ar_remaining_slots_check.json](../certificates/phase4_ar_remaining_slots_check.json)

The next target is the global AR theorem: a convergent reconstruction map from
a solution of the reduced equation `M_eps(h)=0` to actual Donaldson data
`(omega_eps, lambda_eps, mu_eps, Theta_eps)`.

## Inputs

Fixed datum:

- `D0`, the rank-one 77-unlink datum.

Reduced solution:

- `h_eps in X_beta^ext`;
- `M_eps(h_eps) = 0`;
- `||h_eps - h0||_{X_beta^ext} <= r_AR`.

Checked coefficient inputs:

- `source_P1 = 3.1185`;
- `source_P2 = 6.36018075`;
- `DP1_norm = 3.1185`;
- `DP2_norm = 8.3981205`;
- `D2P1_norm = 3.1185`;
- `D3m_norm = 4/9`;
- `raw_P3_scale = 3.24648475975`;
- `remainder_R3 = 13770793.170081593`;
- `R_threshold = 3664.0659853300026`.

Fibrewise inverse input:

- `G_f = Delta_f^{-1} d_f^*` on the relevant `d_f`-exact K3 source sectors;
- uniform bound `||G_f|| <= K_H_K3`;
- compatibility of `G_f` with the weighted base norms used by `X_beta^ext`
  and `Y_{beta-2}`.

The product-space contract records this as an explicit open theorem obligation,
not as a hidden constant.

## Product Space

The reconstruction Banach space is now fixed by the public product-space
contract:

`A_beta = A_omega x A_lambda x A_mu x A_Theta`

with components:

- `omega in Omega^{1,2}`;
- `lambda in Omega^{3,0}`;
- `mu in Omega^{0,4}`;
- `Theta in Omega^{2,2}`.

The norm must be a weighted product norm compatible with:

- fibrewise Schauder estimates on K3;
- base weighted Holder estimates in the `beta` convention;
- the `F_H : Omega^{p,q} -> Omega^{p+2,q-1}` action;
- the projection `Pi_reduced(d_H Theta)`.

No unnamed `O(1)` constants are allowed in this norm. The current checker
verifies the component bidegrees and block degree shifts, but not the uniform
analytic inverse theorem.

## Reconstruction Map

The target map is

`A_eps : h -> (omega_eps(h), lambda_eps(h), mu_eps(h), Theta_eps(h))`.

It must solve Donaldson's bigraded equations:

`d_f omega = 0`

`d_H omega = 0`

`d_f lambda = -F_H omega`

`d_H mu = 0`

`d_f Theta = -F_H mu`

and reduce

`d_H Theta = 0`

to the scalar equation

`M_eps(h) = 0`.

## Majorant Constants

The proof must introduce explicit constants:

- `C_AR_source`;
- `C_AR_lin`;
- `C_AR_quad`;
- `C_AR_tail`;
- `r_AR`;
- `eps_AR`;
- `R_AR = eps_AR^{-1}`.

The P4.1 values above may enter `C_AR_source` and `C_AR_tail`, but they do not
by themselves prove convergence.

Required majorant shape:

`||A_eps(h) - A0(h)||_{A_beta} <= C_AR_source eps`

and for the nonlinear tail map `T_eps`,

`||T_eps(a)-T_eps(b)||_{A_beta}
 <= q_AR ||a-b||_{A_beta}`

with an explicit `q_AR < 1`.

The current budget skeleton splits the remaining Neumann error into:

- `q_hodge_uniform`;
- `q_comm_FH_Gf`;
- `q_projection_residual`;
- `q_gauge_transfer`.

Each slot has target `1/16`; all are still pending exact analytic certificates.
All four slots now have checked candidate values below target, but none is yet
theorem-grade.

## Output Theorem

Target statement:

For explicit `eps_AR > 0`, if `0 < eps <= eps_AR` and
`M_eps(h_eps)=0`, then there exists a unique reconstructed tuple

`A_eps(h_eps) = (omega_eps, lambda_eps, mu_eps, Theta_eps)`

in the chosen gauge such that:

- Donaldson equations `(E1)`--`(E5)` hold exactly;
- `d_H Theta_eps = 0` follows from `M_eps(h_eps)=0`;
- the reconstructed `Phi_eps(h_eps)` is closed;
- its residual coclosed defect is bounded by an explicit constant
  `C_tors eps` in the Phase 5 norm convention.

## Acceptance Gate

P4.2 is not closed until the repository contains:

- the product Banach-space definition; done at contract/checker level;
- a uniform fibrewise Hodge inverse theorem or explicit hypothesis; still open;
- a majorant/contraction certificate with `q_AR < 1`;
- certified Neumann-error slots with total error still below the contraction
  threshold;
- an independent checker for the scalar inequalities;
- a ledger update keeping P4.2 separate from P5 `(J)`.
