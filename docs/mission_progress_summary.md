# Mission Progress Summary

## Scope

This document summarizes the current state of the Phase 0-7 programme for the
rank-one 77-unlink Donaldson / branched adiabatic construction.

It is a progress summary, not a theorem ledger. For scope discipline and claim
strength, see:

- [analytic_status.md](analytic_status.md)
- [theorem_ledger.yaml](../audit/theorem_ledger.yaml)

## Global headline

Current honest status:

- Level `E`: not yet proved; theorem target fixed.
- Level `Q`: partially scaffolded with several reproducible certificates, with
  the `D0` sharp-constant layer substantially improved by the private
  2026-07-02 P1/P2/P3/C_nl chain, but not globally closed.
- Level `CF`: explicitly separated as a distinct research track; no compact closed form proved.

Operationally, the branch is in **Stage D reconciliation** after the private
2026-07-05 closures. Phases 0 and 1 are largely stabilized, Phase 2 is
partial, Phase 3/L1.6 and Phase 4/L2 assembly are now pinned to the private
theorem-grade closures at `D0`; Stage E remains the next interval-promotion
track.

Latest cross-repo update from `private` through the 2026-07-09 read window:

- `(G)` source constants at `D0`: `C_src = 27/16` in the sigma-odd cubic
  channel and `C_nl = 2/3`, giving `r_G(D0) ~= 1.51e-4` and margin
  approximately `3318x` below the `1/2` contraction threshold.
- `Pi_obs` PDE identification is promoted by private M-L1.d; the older
  reduced lower-root constant-mode model remains an implementation witness.
- L1.6 is closed: raw 77-collar outward-rounded intervals promote
  `K_Sch <= 16/3`; headline `q_coeff ~= 0.16811`, sharp `~= 0.10959`.
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

### Phase 0: Truth ledger and canonicalisation

Status: substantially completed.

Main artifacts:

- [theorem_ledger.yaml](../audit/theorem_ledger.yaml)
- [dependency_graph.json](../audit/dependency_graph.json)
- [claim_scope.md](../audit/claim_scope.md)
- [legacy_quarantine.md](../audit/legacy_quarantine.md)

Resolved points:

- `AnalyticalMetric.lean` is frozen as a local flat `R^7` artifact, not a compact `K_7` metric.
- `K3ClosedFormWitness.lean` is frozen as a box-local interval witness, not a global Ricci-flat K3 theorem.
- `G2DonaldsonLinkCohomology.lean` is separated into arithmetic/formal consequences versus non-formal geometric realization.
- `CollarResummationCertificate.lean` is frozen as a combinatorial certificate, not a PDE convergence theorem.
- `DonaldsonGlobalBaseAudit.lean` is quarantined as legacy / exploratory.

Remaining work:

- keep the ledger updated as stronger or sharper artifacts are added.
- continue preventing repo prose from over-promoting local or conditional results.

### Phase 1: Precise theorem targets

Status: completed at the specification level.

Main artifacts:

- [theorem_E_exact.md](../paper/theorem_E_exact.md)
- [theorem_Q_certified.md](../paper/theorem_Q_certified.md)
- [conjecture_CF_finite_closed_form.md](../paper/conjecture_CF_finite_closed_form.md)
- [assumptions_map.tex](../paper/assumptions_map.tex)

Resolved points:

- Levels `E`, `Q`, and `CF` are separated explicitly.
- the main compact torsion-free target theorem is fixed without allowing shortcut phrases.
- the closed-form claim is isolated as a separate programme.

Remaining work:

- no structural rewrite needed; future work is theorem discharge, not target redefinition.

### Phase 2: Explicit datum `D0`

Status: partial, with sharp constants updated by the 2026-07-02 private chain.

Main artifact:

- [datum_D0.json](../certificates/datum_D0.json)

Resolved or frozen fields:

- `r0 = 10^-2`
- `d_min = 1`
- `kappa_g <= 1`
- `cond(A_bulk) <= 2.31`
- lower bound `A_bulk(alpha1,alpha1) >= 0.43290043290043284`
- upper bounds for `K_H_K3`, `||F_H||`, `D2m`
- sharp-current `K_H_K3 <= 0.45` from the K3 Hodge-gap reduction
- sharp-current `C_src = 27/16` and `C_nl = 2/3` for the `D0` G-discharge
- Phase 4.1 coefficient fields `source_P1`, `source_P2`, `DP1_norm`,
  `DP2_norm`, `D2P1_norm`, `D3m_norm`, `raw_P3_scale`, `remainder_R3`, and
  the tail-contraction threshold `R_threshold = 3664.0659853300026`
- sharp-current exterior package
  `(C_FS, gamma_link, C_link, C_outer) = (0.3, 2, 1, 1)`

Still open at theorem grade:

- `inj_base`
- `curv_base`
- exact interval for `A_bulk(alpha1,alpha1)` rather than only a lower bound
- outward-rounded interval recomputation for the Phase 4.1 coefficient layer

Next real Phase 2 task:

- continue turning `datum_D0.json` from `partial_certified` into a complete datum certificate.

### Phase 3: Global maximal-section theorem

Status: active main workstream.

#### P3.1 Spaces

Status: completed at the definition level.

Main artifact:

- [phase3_spaces.md](../paper/phase3_spaces.md)

Resolved points:

- coordinates, sigma-odd sector, weighted spaces
- `X_beta^{(0)}`
- `X_beta^{ext}`
- `Y_{beta-2}`
- constant-section layer `R^{2N}`
- trace conventions and realization map

#### P3.2 Linear theorem

Status: completed at the theorem-specification level; the `D0` `Pi_obs`
identification is pinned to private M-L1.d.

Main artifact:

- [phase3_linear_theorem.md](../paper/phase3_linear_theorem.md)

Resolved points:

- operator `J_h : X_beta^{ext} -> Y_{beta-2}`
- relevant indicial window
- reduced map `A_red = A_loc + E_geom + E_link`
- role of DtN positivity and the constant-section obstruction layer

Still open:

- theorem-level right inverse `G_aug` beyond the already-closing crude
  scaffold; the exact non-crude composition is a sharpening item.

#### P3.3 Global parametrix

Status: completed at the architecture level.

Main artifact:

- [phase3_global_parametrix.md](../paper/phase3_global_parametrix.md)

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

- [phase3_nonlinear_closure.md](../paper/phase3_nonlinear_closure.md)

Resolved points:

- fixed-point formulation for `m(h_bar_global + u) = 0`
- explicit scalar closure inequality shape

Still open:

- actual contraction theorem in `X_beta^{ext}`

#### P3 numerical scaffold

Status: partially frozen and very favorable numerically.

Main artifacts:

- [phase3_source_nonlinear_constants.md](../paper/phase3_source_nonlinear_constants.md)
- [phase3_G_closure_status.json](../certificates/phase3_G_closure_status.json)

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

- [phase3_source_residual_certificate.md](../paper/phase3_source_residual_certificate.md)
- [phase3_source_residual_ledger.json](../certificates/phase3_source_residual_ledger.json)
- [phase3_projection_commutator_certificate.md](../legacy/paper/phase3_projection_commutator_certificate.md)
- [phase3_projection_commutator_certificate.json](../legacy/certificates/phase3_projection_commutator_certificate.json)
- [phase3_effective_jacobi_parametrix_candidate.md](../legacy/paper/phase3_effective_jacobi_parametrix_candidate.md)
- [phase3_effective_jacobi_parametrix_candidate.json](../legacy/certificates/phase3_effective_jacobi_parametrix_candidate.json)
- [phase3_Rquad_coefficient.md](../paper/phase3_Rquad_coefficient.md)
- [phase3_mu_minus_half_channel.md](../paper/phase3_mu_minus_half_channel.md)
- [phase3_mu_minus_half_basis.md](../paper/phase3_mu_minus_half_basis.md)
- [phase3_source_trace_resolution.md](../paper/phase3_source_trace_resolution.md)
- [phase3_source_to_obstruction_bridge.md](../paper/phase3_source_to_obstruction_bridge.md)
- [phase3_Pi_obs_constant_mode.md](../paper/phase3_Pi_obs_constant_mode.md)
- [phase3_kappa_src_first_ansatz.md](../paper/phase3_kappa_src_first_ansatz.md)
- [phase3_kappa_src_vs_G_discharge.md](../paper/phase3_kappa_src_vs_G_discharge.md)
- [phase3_gamma_src_surrogate.md](../legacy/paper/phase3_gamma_src_surrogate.md)

Resolved points:

- `R_aff(D0) = 0` is frozen exactly.
- the lower-root real basis is fixed.
- the source-to-cubic-correction algebra is fixed.
- raw profile, lower-root trace, and weighted source quantities are now separated cleanly.
- the correct bridge is identified as
  `profile_src -> source in Y_{beta-2} -> Pi_obs(source) in R^{2N}`.
- a constant-mode prototype for `Pi_obs` is fixed.
- the first non-arbitrary ansatz for `kappa_src` is retained only as a legacy
  comparison.
- the old explicit surrogate `gamma_src,sur = 3/8` is superseded for the
  fixed-Sigma sigma-odd source channel; it remains only a lower-root
  normal-form / regular-sector diagnostic.
- current global `G` discharge uses the sharp-current `C_src = 27/16` source
  coefficient, not the old `3/8` surrogate.
- `Pi_obs` is certified at PDE-identification level for `D0` by private
  `axis2_L1d_Pi_obs_identification_2026_07_04`; the reduced constant-mode
  source model is now only the proto witness.
- `R_comm` is certified as a clean parametrix Neumann factor
  `q_comm <= 0.68` at `D0`, not as an additive source envelope.
- `K_Sch^Maz <= 16/3` is theorem-grade at `D0` via private M-L1.n/o/p raw
  77-collar intervals and recollement.

Still open:

- integration of the pinned `Pi_obs` and `K_Sch` closures into a final
  full global parametrix theorem artifact.
- exact non-crude `G_aug` composition, now a sharpening item rather than a
  load-bearing blocker.

Current sharp Phase 3 bottleneck:

- assemble the global operator/right-inverse theorem using the
  projection/commutator certificate and the theorem-grade cubic coefficient
  `C_src = 27/16`.

### Phase 4: Adiabatic reconstruction

Status: P4.1 coefficients are checked at `D0`; P4.2 has a conditional scalar
majorant, but no convergent AR theorem yet.

Main available artifact:

- [phase4_adiabatic_operator_form.md](../legacy/paper/phase4_adiabatic_operator_form.md)
- [phase4_true_coefficients_derivation_plan.md](../paper/phase4_true_coefficients_derivation_plan.md)
- [phase4_donaldson_E1_E5_operator.md](../paper/phase4_donaldson_E1_E5_operator.md)
- [phase4_bigraded_type_check.json](../certificates/phase4_bigraded_type_check.json)
- [phase4_donaldson_coefficients_stageB_v1.md](../paper/phase4_donaldson_coefficients_stageB_v1.md)
- [phase4_donaldson_coefficients.json](../certificates/phase4_donaldson_coefficients.json)
- [phase4_donaldson_coefficients_checker_stageC_v1.md](../paper/phase4_donaldson_coefficients_checker_stageC_v1.md)
- [phase4_donaldson_coefficients_check.json](../certificates/phase4_donaldson_coefficients_check.json)
- [phase4_donaldson_coefficients_values.md](../paper/phase4_donaldson_coefficients_values.md)
- [phase4_donaldson_coefficients_values.json](../certificates/phase4_donaldson_coefficients_values.json)
- [phase4_ar_convergence_theorem_spec.md](../paper/phase4_ar_convergence_theorem_spec.md)
- [phase4_ar_product_space_contract.md](../paper/phase4_ar_product_space_contract.md)
- [phase4_ar_product_space_contract.json](../certificates/phase4_ar_product_space_contract.json)
- [phase4_ar_product_space_check.json](../certificates/phase4_ar_product_space_check.json)
- [phase4_ar_inverse_budget_audit.md](../paper/phase4_ar_inverse_budget_audit.md)
- [phase4_ar_inverse_budget_audit.json](../certificates/phase4_ar_inverse_budget_audit.json)
- [phase4_ar_inverse_budget_check.json](../certificates/phase4_ar_inverse_budget_check.json)
- [phase4_ar_neumann_budget.md](../legacy/paper/phase4_ar_neumann_budget.md)
- [phase4_ar_neumann_budget_candidate.json](../certificates/phase4_ar_neumann_budget_candidate.json)
- [phase4_ar_neumann_budget_check.json](../certificates/phase4_ar_neumann_budget_check.json)
- [phase4_ar_commutator_slot_candidate.md](../legacy/paper/phase4_ar_commutator_slot_candidate.md)
- [phase4_ar_commutator_slot_candidate.json](../certificates/phase4_ar_commutator_slot_candidate.json)
- [phase4_ar_commutator_slot_check.json](../certificates/phase4_ar_commutator_slot_check.json)
- [phase4_ar_remaining_slots_candidate.md](../legacy/paper/phase4_ar_remaining_slots_candidate.md)
- [phase4_ar_remaining_slots_candidate.json](../certificates/phase4_ar_remaining_slots_candidate.json)
- [phase4_ar_remaining_slots_check.json](../certificates/phase4_ar_remaining_slots_check.json)
- [phase4_consolidation_2026_07_03.md](../legacy/docs/phase4_consolidation_2026_07_03.md)
- [phase4_ar_majorant_candidate.md](../legacy/paper/phase4_ar_majorant_candidate.md)
- [phase4_ar_majorant_candidate.json](../certificates/phase4_ar_majorant_candidate.json)
- [phase4_ar_majorant_check.json](../certificates/phase4_ar_majorant_check.json)
- [phase4_progress_2026_07_03.md](../legacy/docs/phase4_progress_2026_07_03.md)
- [phase4_adiabatic_sources_candidate.json](../legacy/certificates/phase4_adiabatic_sources_candidate.json)
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
- Stage C/E independently checks those symbolic formulas and their evaluated
  D0 values: `172/172` checks pass, including bidegrees, `G_f` use, formula
  dependency resolution, absence of placeholders, arithmetic recomputation,
  outward-rounded interval packaging, and the comparison-only status of legacy
  power-counting numbers.
- Phase 4.1 D0 coefficient values are now evaluated and checked:
  `source_P1 = 3.1185`, `source_P2 = 6.36018075`,
  `DP1_norm = 3.1185`, `DP2_norm = 8.3981205`,
  `D2P1_norm = 3.1185`, `D3m_norm = 4/9`,
  `raw_P3_scale = 3.24648475975`,
  `remainder_R3 = 13770793.170081593`, and
  `R_threshold = 3664.0659853300026`. The checker now passes `172/172`.
- Phase 4.2 is now specified as the global AR convergence theorem target:
  product Banach space, fibrewise Hodge inverse uniformity, majorant constants,
  and the output `M_eps(h_eps)=0 -> Phi_eps(h_eps)` closed.
- The P4.2 product-space contract is now serialized and checked:
  `A_beta = A_omega x A_lambda x A_mu x A_Theta`, Donaldson bidegrees,
  `F_H/G_f` degree shifts, block inverse architecture, and conditional
  `K_AR_prod = 2079/2000`; `36/36` checks pass.
- The inverse-budget audit is now serialized and checked: under the current
  unweighted max norm, `K_AR_prod = 2079/2000` dominates `G_f`, the declared
  `G_f F_H` block bound `2079/2000`, and projection bound `1`; `15/15` checks
  pass. This repairs the scalar normalization issue.
- A first conditional scalar AR majorant is serialized at `R_AR = 4000`,
  using `K_AR_prod <= 2079/2000` from the product-space contract. It gives
  `q_AR ~= 1.035e-3` and passes `20/20` checks, but explicitly marks the
  uniform inverse, commutator, reduced-projection, and closedness obligations as
  unproved.
- The AR Neumann budget skeleton is serialized and checked. It splits the
  remaining analytic error into `q_hodge_uniform`, `q_comm_FH_Gf`,
  `q_projection_residual`, and `q_gauge_transfer`, each with target `1/16`;
  `41/41` checks pass. Stage D repin: private theorem-grade V/W/X/Y slots
  discharge these obligations and L2 assembly gives
  `q_total = 26236977/3200000000 ~= 8.20e-3 < 1/2`.
- The first commutator-slot candidate is serialized and checked:
  `C_comm_AR = 54.00596`, `q_comm_FH_Gf = 0.01350149 < 1/16`; `16/16`
  checks pass. It is a candidate formula only, not a theorem-grade commutator
  certificate.
- The remaining three Neumann slots also have checked candidates:
  `q_hodge_uniform = 0.0058`, `q_projection_residual = 0.016316`,
  `q_gauge_transfer = 0.0061185`; `32/32` checks pass. These are candidate
  formulas only, not theorem-grade analytic certificates.

Still open:

- global fibrewise Hodge inverse proof
- convergent reconstruction theorem with certified constants
- product-norm commutator bounds, reduced-projection identity, and closedness
  preservation.

### Phase 5: Hypothesis `(J)` / anisotropic Joyce theorem

Status: only specified, not proved.

Main artifact:

- [anisotropic_joyce_theorem.md](../paper/anisotropic_joyce_theorem.md)

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

### Phase 6: Compactness, positivity, holonomy

Status: not started as proof work.

Resolved points:

- final theorem wrapper target is specified.

Still open:

- compact manifold theorem chain
- positivity globally on `K_7`
- actual torsion-free theorem input from earlier phases
- full holonomy wrapper theorem

### Phase 7: Closed-form programme

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

1. write the full PDE identification theorem for `Pi_obs` beyond the reduced
   source model;
2. assemble the Phase 3.2/3.3 global right inverse theorem with `q_comm <= 0.68`;
3. prove the uniform product-space inverse, commutator, reduced-projection, and
   closedness obligations for AR;
4. then return to the theorem-level assembly of Phase 3 and AR.
