# Mission Progress Summary

## Scope

This document summarizes the current state of the Phase 0-7 programme for the
rank-one 77-unlink Donaldson / branched adiabatic construction.

It is a progress summary, not a theorem ledger. For scope discipline and claim
strength, see:

- [analytic_status.md](/home/brieuc/gift-framework/GIFT/docs/analytic_status.md)
- [theorem_ledger.yaml](/home/brieuc/gift-framework/GIFT/audit/theorem_ledger.yaml)

## Global headline

Current honest status:

- Level `E`: not yet proved; theorem target fixed.
- Level `Q`: partially scaffolded with several reproducible certificates, with
  the `D0` sharp-constant layer substantially improved by the private
  2026-07-02 P1/P2/P3/C_nl chain, but not globally closed.
- Level `CF`: explicitly separated as a distinct research track; no compact closed form proved.

Operationally, the branch is deep in **Phase 3**. Phases 0 and 1 are largely
stabilized, Phase 2 is partial, Phase 3 is actively under construction, and
Phases 4-7 remain open.

Latest cross-repo update from `private` commits through `767727b0`:

- `(G)` source constants at `D0`: `C_src = 27/16` in the sigma-odd cubic
  channel and `C_nl = 2/3`, giving `r_G(D0) ~= 1.51e-4` and margin
  approximately `3318x` below the `1/2` contraction threshold.
- public projection/commutator certificate: `Pi_obs` is certified on the
  reduced lower-root constant-mode model and the clean annular commutator has
  `q_comm <= 0.68` at `D0` in the conservative Mazzeo regime.
- Task 4 candidate: finite-mode Jacobi normal-operator experiment gives
  `K_ind = 4/3`, candidate scalar `K_Sch^Maz,cand = 16/3`, and rank-19
  candidate `12.32`; the theorem-safe public envelope remains `K_Sch^Maz <= 17`.
- `(AR)` fibrewise Hodge constant: theorem-grade reduction to the scalar K3
  gap, with working certified bound `K_H^{K3} <= 0.45` and margin
  approximately `49x` at `D0`.
- `(E)` exterior constants: `C_FS <= 0.3`, inter-tube exponent `gamma = 2`,
  `C_link <= 1`, `C_outer <= 1`, giving `delta_E(D0) ~= 0.0206` and margin
  approximately `24x`.

These updates sharpen the `D0` numerical closure. They do not by themselves
prove the full global Phase 3 operator theorem, the adiabatic reconstruction
theorem, the anisotropic Joyce theorem `(J)`, or the compact `K7` result.

## Phase-by-phase status

### Phase 0 — Truth ledger and canonicalisation

Status: substantially completed.

Main artifacts:

- [theorem_ledger.yaml](/home/brieuc/gift-framework/GIFT/audit/theorem_ledger.yaml)
- [dependency_graph.json](/home/brieuc/gift-framework/GIFT/audit/dependency_graph.json)
- [claim_scope.md](/home/brieuc/gift-framework/GIFT/audit/claim_scope.md)
- [legacy_quarantine.md](/home/brieuc/gift-framework/GIFT/audit/legacy_quarantine.md)

Resolved points:

- `AnalyticalMetric.lean` is frozen as a local flat `R^7` artifact, not a compact `K_7` metric.
- `K3ClosedFormWitness.lean` is frozen as a box-local interval witness, not a global Ricci-flat K3 theorem.
- `G2DonaldsonLinkCohomology.lean` is separated into arithmetic/formal consequences versus non-formal geometric realization.
- `CollarResummationCertificate.lean` is frozen as a combinatorial certificate, not a PDE convergence theorem.
- `DonaldsonGlobalBaseAudit.lean` is quarantined as legacy / exploratory.

Remaining work:

- keep the ledger updated as stronger or sharper artifacts are added.
- continue preventing repo prose from over-promoting local or conditional results.

### Phase 1 — Precise theorem targets

Status: completed at the specification level.

Main artifacts:

- [theorem_E_exact.md](/home/brieuc/gift-framework/GIFT/paper/theorem_E_exact.md)
- [theorem_Q_certified.md](/home/brieuc/gift-framework/GIFT/paper/theorem_Q_certified.md)
- [conjecture_CF_finite_closed_form.md](/home/brieuc/gift-framework/GIFT/paper/conjecture_CF_finite_closed_form.md)
- [assumptions_map.tex](/home/brieuc/gift-framework/GIFT/paper/assumptions_map.tex)

Resolved points:

- Levels `E`, `Q`, and `CF` are separated explicitly.
- the main compact torsion-free target theorem is fixed without allowing shortcut phrases.
- the closed-form claim is isolated as a separate programme.

Remaining work:

- no structural rewrite needed; future work is theorem discharge, not target redefinition.

### Phase 2 — Explicit datum `D0`

Status: partial, with sharp constants updated by the 2026-07-02 private chain.

Main artifact:

- [datum_D0.json](/home/brieuc/gift-framework/GIFT/certificates/datum_D0.json)

Resolved or frozen fields:

- `r0 = 10^-2`
- `d_min = 1`
- `kappa_g <= 1`
- `cond(A_bulk) <= 2.31`
- lower bound `A_bulk(alpha1,alpha1) >= 0.43290043290043284`
- upper bounds for `K_H_K3`, `||F_H||`, `D2m`
- sharp-current `K_H_K3 <= 0.45` from the K3 Hodge-gap reduction
- sharp-current `C_src = 27/16` and `C_nl = 2/3` for the `D0` G-discharge
- sharp-current exterior package
  `(C_FS, gamma_link, C_link, C_outer) = (0.3, 2, 1, 1)`

Still open at theorem grade:

- `inj_base`
- `curv_base`
- exact interval for `A_bulk(alpha1,alpha1)` rather than only a lower bound
- `source_P1`
- `source_P2`
- some remainder and scale constants requested by the original plan

Next real Phase 2 task:

- continue turning `datum_D0.json` from `partial_certified` into a complete datum certificate.

### Phase 3 — Global maximal-section theorem

Status: active main workstream.

#### P3.1 Spaces

Status: completed at the definition level.

Main artifact:

- [phase3_spaces.md](/home/brieuc/gift-framework/GIFT/paper/phase3_spaces.md)

Resolved points:

- coordinates, sigma-odd sector, weighted spaces
- `X_beta^{(0)}`
- `X_beta^{ext}`
- `Y_{beta-2}`
- constant-section layer `R^{2N}`
- trace conventions and realization map

#### P3.2 Linear theorem

Status: completed at the theorem-specification level.

Main artifact:

- [phase3_linear_theorem.md](/home/brieuc/gift-framework/GIFT/paper/phase3_linear_theorem.md)

Resolved points:

- operator `J_h : X_beta^{ext} -> Y_{beta-2}`
- relevant indicial window
- reduced map `A_red = A_loc + E_geom + E_link`
- role of DtN positivity and the constant-section obstruction layer

Still open:

- theorem-level construction of `Pi_obs`
- theorem-level proof that no extra modes survive outside `R^{2N}`
- theorem-level right inverse `G_aug`

#### P3.3 Global parametrix

Status: completed at the architecture level.

Main artifact:

- [phase3_global_parametrix.md](/home/brieuc/gift-framework/GIFT/paper/phase3_global_parametrix.md)

Resolved points:

- decomposition `G_glob = sum_i chi_i G_i tilde_chi_i + G_bulk`
- separation of commutator, geometry, link, and DtN errors

Still open:

- theorem-level bound `||J_h G_glob - I|| <= q < 1`

#### P3.4 Nonlinear closure

Status: completed at the fixed-point scaffold level; the local `D0`
source/nonlinear constants have now been sharpened to theorem-grade in the
private chain, while the global operator theorem remains open.

Main artifact:

- [phase3_nonlinear_closure.md](/home/brieuc/gift-framework/GIFT/paper/phase3_nonlinear_closure.md)

Resolved points:

- fixed-point formulation for `m(h_bar_global + u) = 0`
- explicit scalar closure inequality shape

Still open:

- actual contraction theorem in `X_beta^{ext}`

#### P3 numerical scaffold

Status: partially frozen and very favorable numerically.

Main artifacts:

- [phase3_source_nonlinear_constants.md](/home/brieuc/gift-framework/GIFT/paper/phase3_source_nonlinear_constants.md)
- [phase3_G_closure_status.json](/home/brieuc/gift-framework/GIFT/certificates/phase3_G_closure_status.json)

Current notable values:

- `||J_ext^{-1}|| <= 36.635` as current scaffold
- old envelope bookkeeping: `r_G(D0) ~= 2.68424645e-4`
- sharp-current bookkeeping from private P1/C_nl:
  `C_src = 27/16`, `C_nl = 2/3`, `r_G(D0) ~= 1.51e-4`

Interpretation:

- the numerical smallness is not the bottleneck.
- the old source/nonlinear envelopes are no longer the bottleneck at `D0`.
- the remaining bottleneck is turning the full global Phase 3 operator package
  into an actual theorem artifact.

#### P3 source-term subprogramme

Status: strong progress in clarification, basis fixing, and intermediate tooling.

Main artifacts:

- [phase3_source_residual_certificate.md](/home/brieuc/gift-framework/GIFT/paper/phase3_source_residual_certificate.md)
- [phase3_source_residual_ledger.json](/home/brieuc/gift-framework/GIFT/certificates/phase3_source_residual_ledger.json)
- [phase3_projection_commutator_certificate.md](/home/brieuc/gift-framework/GIFT/paper/phase3_projection_commutator_certificate.md)
- [phase3_projection_commutator_certificate.json](/home/brieuc/gift-framework/GIFT/certificates/phase3_projection_commutator_certificate.json)
- [phase3_effective_jacobi_parametrix_candidate.md](/home/brieuc/gift-framework/GIFT/paper/phase3_effective_jacobi_parametrix_candidate.md)
- [phase3_effective_jacobi_parametrix_candidate.json](/home/brieuc/gift-framework/GIFT/certificates/phase3_effective_jacobi_parametrix_candidate.json)
- [phase3_Rquad_coefficient.md](/home/brieuc/gift-framework/GIFT/paper/phase3_Rquad_coefficient.md)
- [phase3_mu_minus_half_channel.md](/home/brieuc/gift-framework/GIFT/paper/phase3_mu_minus_half_channel.md)
- [phase3_mu_minus_half_basis.md](/home/brieuc/gift-framework/GIFT/paper/phase3_mu_minus_half_basis.md)
- [phase3_source_trace_resolution.md](/home/brieuc/gift-framework/GIFT/paper/phase3_source_trace_resolution.md)
- [phase3_source_to_obstruction_bridge.md](/home/brieuc/gift-framework/GIFT/paper/phase3_source_to_obstruction_bridge.md)
- [phase3_Pi_obs_constant_mode.md](/home/brieuc/gift-framework/GIFT/paper/phase3_Pi_obs_constant_mode.md)
- [phase3_kappa_src_first_ansatz.md](/home/brieuc/gift-framework/GIFT/paper/phase3_kappa_src_first_ansatz.md)
- [phase3_kappa_src_vs_G_discharge.md](/home/brieuc/gift-framework/GIFT/paper/phase3_kappa_src_vs_G_discharge.md)
- [phase3_gamma_src_surrogate.md](/home/brieuc/gift-framework/GIFT/paper/phase3_gamma_src_surrogate.md)

Resolved points:

- `R_aff(D0) = 0` is frozen exactly.
- the lower-root real basis is fixed.
- the source-to-cubic-correction algebra is fixed.
- raw profile, lower-root trace, and weighted source quantities are now separated cleanly.
- the correct bridge is identified as
  `profile_src -> source in Y_{beta-2} -> Pi_obs(source) in R^{2N}`.
- a constant-mode prototype for `Pi_obs` is fixed.
- first non-arbitrary ansatz for `kappa_src` is fixed.
- first explicit surrogate with `gamma_src,sur = 3/8` is fixed as a modeling layer, not a theorem.
- current global `G` discharge is shown to be compatible with that surrogate, but much coarser.
- `Pi_obs` is certified on the reduced lower-root constant-mode source model.
- `R_comm` is certified as a clean parametrix Neumann factor
  `q_comm <= 0.68` at `D0`, not as an additive source envelope.
- `K_Sch^Maz` has a reproducible candidate experiment (`16/3` scalar, `12.32`
  rank-19), but this is not theorem-grade.

Still open:

- full PDE identification of `Pi_obs` beyond the reduced lower-root model
- integration of `q_comm` into the full global parametrix theorem
- promotion or rejection of the `K_Sch^Maz,cand = 16/3` experiment via a real
  edge-Schauder proof

Current sharp Phase 3 bottleneck:

- reconcile or supersede the public surrogate / ansatz layer with the private
  theorem-grade cubic coefficient `C_src = 27/16`, then assemble the global
  operator/right-inverse theorem using the projection/commutator certificate.

### Phase 4 — Adiabatic reconstruction

Status: started at P4.1 operator-form / candidate-source level; not started as
a convergent theorem.

Main available artifact:

- [phase4_adiabatic_operator_form.md](/home/brieuc/gift-framework/GIFT/paper/phase4_adiabatic_operator_form.md)
- [phase4_true_coefficients_derivation_plan.md](/home/brieuc/gift-framework/GIFT/paper/phase4_true_coefficients_derivation_plan.md)
- [phase4_donaldson_E1_E5_operator.md](/home/brieuc/gift-framework/GIFT/paper/phase4_donaldson_E1_E5_operator.md)
- [phase4_bigraded_type_check.json](/home/brieuc/gift-framework/GIFT/certificates/phase4_bigraded_type_check.json)
- [phase4_donaldson_coefficients_stageB_v1.md](/home/brieuc/gift-framework/GIFT/paper/phase4_donaldson_coefficients_stageB_v1.md)
- [phase4_donaldson_coefficients.json](/home/brieuc/gift-framework/GIFT/certificates/phase4_donaldson_coefficients.json)
- [phase4_donaldson_coefficients_checker_stageC_v1.md](/home/brieuc/gift-framework/GIFT/paper/phase4_donaldson_coefficients_checker_stageC_v1.md)
- [phase4_donaldson_coefficients_check.json](/home/brieuc/gift-framework/GIFT/certificates/phase4_donaldson_coefficients_check.json)
- [phase4_donaldson_coefficients_values.md](/home/brieuc/gift-framework/GIFT/paper/phase4_donaldson_coefficients_values.md)
- [phase4_donaldson_coefficients_values.json](/home/brieuc/gift-framework/GIFT/certificates/phase4_donaldson_coefficients_values.json)
- [phase4_ar_convergence_theorem_spec.md](/home/brieuc/gift-framework/GIFT/paper/phase4_ar_convergence_theorem_spec.md)
- [phase4_ar_majorant_candidate.md](/home/brieuc/gift-framework/GIFT/paper/phase4_ar_majorant_candidate.md)
- [phase4_ar_majorant_candidate.json](/home/brieuc/gift-framework/GIFT/certificates/phase4_ar_majorant_candidate.json)
- [phase4_ar_majorant_check.json](/home/brieuc/gift-framework/GIFT/certificates/phase4_ar_majorant_check.json)
- [phase4_progress_2026_07_03.md](/home/brieuc/gift-framework/GIFT/docs/phase4_progress_2026_07_03.md)
- [phase4_adiabatic_sources_candidate.json](/home/brieuc/gift-framework/GIFT/certificates/phase4_adiabatic_sources_candidate.json)
- previous private formal matching:
  `private/canonical/results/axis2_M_epsilon_adiabatic_2026_07_01.json`

Resolved points:

- the objective and theorem target are clearly stated.
- the reconstruction problem is separated from Phase 3 and from the final perturbative torsion-free step.
- the public operator form is fixed as
  `M_eps(h) = m(h) + eps P1(h) + eps^2 P2(h) + R_eps(h)`.
- legacy candidate source norms are serialized for comparison only:
  `||P1|| <= 2.31`, `||P2|| <= 5.3361`,
  candidate `||R_eps||/eps^3 <= 5.688783e6`.
- the next coefficient-derivation route is planned: extract the actual
  Donaldson `E1`--`E5` operator coefficients, serialize a structured
  coefficient tree, then check it independently before touching `datum_D0`.
- Stage A of that route is written: the `E1`--`E5` operator ledger fixes
  bidegrees, domains/codomains, the `F_H` action, the `K_H_K3` Hodge inverse
  convention, and the symbolic definitions of `P1`, `P2`, and `R_eps`.
- the Stage A bidegrees have been corrected to Donaldson's package:
  `omega in Omega^{1,2}`, `lambda in Omega^{3,0}`,
  `Theta in Omega^{2,2}`, `mu in Omega^{0,4}`. A pre-coefficient type-check
  certificate now verifies `F_H omega in Omega^{3,1}` and
  `F_H mu in Omega^{2,3}`.
- Stage B v1 now serializes type-gated symbolic upper-bound formulas for
  `source_P1`, `source_P2`, `DP1_norm`, `DP2_norm`, `D2P1_norm`, `D3m_norm`,
  `raw_P3_scale`, and `remainder_R3`. The old power-counting candidate is
  retained only as a comparison block, not as a derivation.
- Stage C v1 independently checks those symbolic formulas and their evaluated
  D0 values: `146/146` checks pass, including bidegrees, `G_f` use, formula
  dependency resolution, absence of placeholders, arithmetic recomputation, and
  the comparison-only status of legacy power-counting numbers.
- Phase 4.1 D0 coefficient values are now evaluated and checked:
  `source_P1 = 3.1185`, `source_P2 = 6.36018075`,
  `DP1_norm = 3.1185`, `DP2_norm = 8.3981205`,
  `D2P1_norm = 3.1185`, `D3m_norm = 4/9`,
  `raw_P3_scale = 3.24648475975`,
  `remainder_R3 = 13770793.170081593`, and
  `R_threshold = 3664.0659853300026`. The checker now passes `146/146`.
- Phase 4.2 is now specified as the global AR convergence theorem target:
  product Banach space, fibrewise Hodge inverse uniformity, majorant constants,
  and the output `M_eps(h_eps)=0 -> Phi_eps(h_eps)` closed.
- A first conditional scalar AR majorant is serialized at `R_AR = 4000`,
  assuming `K_AR_prod <= 9/20`. It gives `q_AR ~= 4.48e-4` and passes `14/14`
  checks, but explicitly marks the product-space inverse theorem as unproved.

Still open:

- global fibrewise Hodge inverse proof
- convergent reconstruction theorem with certified constants
- private draft cleanup: remove or downgrade claims that the closed
  small-torsion construction has been reduced only to `(J)`. Current public
  status is `Stage B -> Stage C -> AR discharged`, not `AR discharged`.
- replacement of the P4.1 power-counting candidate by formula-level or
  interval enclosures

### Phase 5 — Hypothesis `(J)` / anisotropic Joyce theorem

Status: only specified, not proved.

Main artifact:

- [anisotropic_joyce_theorem.md](/home/brieuc/gift-framework/GIFT/paper/anisotropic_joyce_theorem.md)

Resolved points:

- exact theorem target fixed
- three-scale geometry stated explicitly
- required operator package and nonlinear estimate listed

Still open:

- anisotropic Banach spaces
- uniform parametrix
- linear inverse theorem
- nonlinear contraction theorem
- gauge and obstruction audit

This remains the decisive open theorem for Level `E`.

### Phase 6 — Compactness, positivity, holonomy

Status: not started as proof work.

Resolved points:

- final theorem wrapper target is specified.

Still open:

- compact manifold theorem chain
- positivity globally on `K_7`
- actual torsion-free theorem input from earlier phases
- full holonomy wrapper theorem

### Phase 7 — Closed-form programme

Status: separated conceptually, not advanced as an active proof line.

Resolved points:

- closed-form is explicitly treated as independent from Level `E`.

Still open:

- Branch CF-A: prove no correction is needed
- Branch CF-B: finite invariant ansatz
- Branch CF-C: prove the honest endpoint is only an implicit convergent correction

## What is actually accomplished

The branch has already accomplished four things that matter:

1. it now has strict scope control, so local or box-local certificates are no longer accidentally promoted to compact-global theorems;
2. it has a complete theorem scaffold for the whole programme through Phase 5;
3. it has turned Phase 3 from a vague “Hypothesis (G)” into a concrete operator-and-constant problem;
4. it has reduced the current main gap to a sharp Phase-3 integration problem:
   the private chain supplies the `D0` source constants, while the public
   operator/projection/commutator theorem package still has to catch up.

## What remains structurally decisive

There are still three major proof gaps before Level `E`:

1. close Phase 3 with a theorem-grade global maximal background;
2. prove Phase 4, the convergent adiabatic reconstruction;
3. prove Phase 5, the anisotropic perturbation theorem `(J)`.

Only after those can Phase 6 close compact torsion-free existence with full holonomy.

## Current next steps

The most useful immediate next tasks are:

1. integrate the private `C_src = 27/16` and `C_nl = 2/3` derivations into the
   public Phase-3 certificate layer;
2. write the full PDE identification theorem for `Pi_obs` beyond the reduced
   source model;
3. assemble the Phase 3.2/3.3 global right inverse theorem with `q_comm <= 0.68`;
4. then return to the theorem-level assembly of Phase 3.
