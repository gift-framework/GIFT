# Theorem Q: fully certified quantitative package

## Status

Current status: target quantitative theorem only. Partial ingredients exist.

## Goal

Every analytic constant entering the Level E proof must come with a reproducible certificate, either symbolic or interval-rigorous, with an independent checker.

## Minimal certificate inventory

The final package must expose, at minimum, explicit certified enclosures for:

- `r0`
- `d_min`
- `inj_base`
- `curv_base`
- `kappa_g`
- `cond(A_bulk)`
- `A_bulk(alpha1, alpha1)`
- `K_H_K3`
- `norm(F_H)`
- `source_P1`
- `source_P2`
- `remainder_R3`
- `D2m`
- `K_Sch_Maz`
- `sigma_min(A_red)`
- `q_parametrix`
- `C_G`
- `C_Q`
- `R_J`

## Certificate contract

Each constant must record:

- source script path,
- producer commit hash,
- input hash,
- precision,
- outward rounding protocol,
- serialized interval,
- independent checker command.

## Current partial status

- `K3ClosedFormWitness.lean` gives a real box-local certificate.
- `CollarResummationCertificate.lean` gives exact scalar constants.
- `axis2_certificate_sigma_min_NK_2026_06_30.json` gives a reproducible numerical scaffold for one part of the Jacobi reduction.
- `axis2_joyce_hypotheses_D0_2026_07_01.json` is an audit note, not yet a theorem-grade certificate for `(J)`.
- `phase3_projection_commutator_certificate.json` certifies the reduced-model
  `Pi_obs` projection and the clean D0 annular commutator factor `q_comm <= 0.68`.
- `phase3_effective_jacobi_parametrix_candidate.json` gives a Task 4 candidate
  for `K_Sch_Maz`: exact normal inverse `4/3`, scalar candidate `16/3`, rank-19
  candidate `12.32`. This is not theorem-grade; `K_Sch_Maz <= 17` remains the
  public safe envelope.
- `phase4_adiabatic_sources_candidate.json` is now legacy comparison material:
  it fixed the public operator form and gave power-counting candidate bounds
  `P1 <= 2.31`, `P2 <= 5.3361`, `R3 <= 5.688783e6`. These are superseded by
  `phase4_donaldson_coefficients_values.json`.
- `phase4_true_coefficients_derivation_plan.md` fixes the next P4.1 route:
  derive `P1`, `P2`, and `R_eps` from Donaldson's bigraded equations, produce
  `phase4_donaldson_coefficients.json`, and verify it with an independent
  checker before any Level Q promotion.
- `phase4_donaldson_E1_E5_operator.md` completes the symbolic Stage A ledger:
  it records bidegrees, domains/codomains, `F_H` action, `K_H_K3` usage, and
  the symbolic coefficient definitions. It contains no numerical certificate.
- `phase4_bigraded_type_check.json` is a pre-coefficient type-check only. It
  verifies the corrected Donaldson bidegrees
  `omega/lambda/Theta/mu = (1,2)/(3,0)/(2,2)/(0,4)` and must gate Stage B.
- `phase4_donaldson_coefficients.json` is Stage B v1: a type-gated symbolic
  formula tree for `P1`, `P2`, and `R3`, with no placeholder block.
- `phase4_donaldson_coefficients_check.json` is Stage C v1: an independent
  structural and arithmetic checker for that symbolic formula tree. With
  `phase4_donaldson_coefficients_values.json`, it passes `146/146` checks.
- `phase4_donaldson_coefficients_values.json` evaluates the D0 coefficient
  formulas: `source_P1 = 3.1185`, `source_P2 = 6.36018075`, and
  `remainder_R3 = 13770793.170081593`. This closes P4.1 coefficients at D0,
  not the global AR convergence theorem.

## Acceptance rule

No theorem may import a floating-point summary without a serialized outward-rounded interval and a checker distinct from the producer.
