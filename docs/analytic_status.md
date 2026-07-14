# Analytic Status Ledger

This document freezes the current analytic status of the Donaldson/Kovalev-Lefschetz branch and prevents promotion of local or numerical artifacts to compact-global theorems.

## Scope labels

- `local`: model on `R^7`, Eguchi-Hanson, one fibre, or one formal chart.
- `box-local`: interval certificate on finitely many explicit boxes.
- `collar`: one discriminant collar or one branched local model.
- `bulk`: smooth region away from the discriminant.
- `global`: full non-compact or punctured base problem on `S^3 \ Sigma`.
- `compact-global`: statement on the compact total space `K_7`.

## Current headline

The repository currently contains a credible conditional programme for Level E,
a Stage E certified D0 coefficient package for Level Q, and no finite
closed-form compact torsion-free metric for Level CF.

The Level E theorem remains conditional because the anisotropic perturbation
theorem `(J)`, compact datum/topology closure, and final public theorem wrappers
are not all discharged in this repository.

The active Level Q citation path is:

- `certificates/datum_D0.json`
- `certificates/phase4_donaldson_coefficients_values.json`
- `certificates/phase4_donaldson_coefficients_check.json`
- `certificates/phase4_level_Q_coefficients.json`
- `certificates/phase4_level_Q_coefficients_check.json`
- `certificates/phase4_level_Q_table_check.json`
- `paper/theorem_Q_certified.md`

The citable Stage E headline is
`R_threshold <= 3664.066`, with machine bracket
`[3664.065985330004, 3664.065985330005]`.

## 2026-07-09 Stage D/E Cleanup

The following older public artifacts are retained as historical comparison
material only:

- `legacy/paper/phase3_effective_jacobi_parametrix_candidate.md`
- `legacy/certificates/phase3_effective_jacobi_parametrix_candidate.json`
- `legacy/paper/phase3_projection_commutator_certificate.md`
- `legacy/certificates/phase3_projection_commutator_certificate.json`
- `legacy/paper/phase4_adiabatic_operator_form.md`
- `legacy/certificates/phase4_adiabatic_sources_candidate.json`
- `legacy/paper/phase4_ar_neumann_budget.md`
- `legacy/paper/phase4_ar_commutator_slot_candidate.md`
- `legacy/paper/phase4_ar_remaining_slots_candidate.md`
- `legacy/paper/phase4_ar_majorant_candidate.md`

They must not be used to override the Stage D/E path. In particular:

- `K_Sch^Maz <= 17` is a stale public envelope; the active D0 scalar bound is
  `K_Sch^Maz <= 16/3`, imported through the Stage D `L1.6_K_Sch` block.
- `gamma_src,sur = 3/8` is a legacy surrogate, not the active source constant.
  The active source constant is `C_src = 27/16`.
- Phase 4 power-counting candidates are superseded by the Stage E coefficient
  values and independent checkers.

## 2026-07-02 D0 Sharp-Constant Update

The sibling `private` repository is ahead of this public Phase-3 scaffold on
the local `D0` constants. Through commit `767727b0`, it records:

- `C_src = 27/16` for the fixed-discriminant sigma-odd cubic source channel.
- `C_nl = 2/3` for the symmetric-space second-variation bound.
- `K_H^{K3} <= 0.45` after reducing the exact-2-form Hodge gap to the scalar
  K3 gap.
- `C_FS <= 0.3`, inter-tube `gamma = 2`, `C_link <= 1`, and `C_outer <= 1`
  for the exterior `(E)` package.

These are sharp-current `D0` constants, not a compact-global torsion-free
metric theorem. Stage D/E promote the D0 constant and coefficient bookkeeping,
while preserving the warnings about the missing public global Phase 3 theorem,
anisotropic Joyce theorem `(J)`, and Appendix C compact-topology proof.

## Strong-claim audit

### Claims that must be read narrowly

- `K7-Lean/GIFT/Foundations/AnalyticalMetric.lean`
  Scope: `local`.
  Status: exact constant flat `G_2` form on `R^7`, not a compact `K_7` metric.
  Reason: the file proves constancy, scaled identity metric, and zero torsion by `d phi = d * phi = 0` for a constant form. It does not build an atlas, descent, compact quotient, or compact torsion-free structure.

- `K7-Lean/GIFT/Foundations/K3ClosedFormWitness.lean`
  Scope: `box-local`.
  Status: rigorous interval certificate on 4000 frozen Krawczyk boxes for one explicit 667-parameter witness.
  Reason: the header explicitly says the result does not promote to a whole-K3 bound and isolates the remaining positivity/SOS problem.

- `K7-Lean/GIFT/Foundations/G2DonaldsonLinkCohomology.lean`
  Scope: arithmetic and combinatorial only.
  Status: formal arithmetic consequences from definitions plus a hand-off to an external Leray/branched-cover derivation.
  Reason: the header explicitly states the sheaf-theoretic and analytic derivation is not formalized.

- `K7-Lean/GIFT/Foundations/CollarResummationCertificate.lean`
  Scope: `collar`.
  Status: exact binomial and indicial-parity identities only.
  Reason: it proves `sum |binom(3/2,k)| = 3` and parity of `beta^2 - m^2`; it is not a convergence theorem for the collar PDE or a Fredholm theorem.

- `K7-Lean/GIFT/Foundations/DonaldsonGlobalBaseAudit.lean`
  Scope: exploratory / legacy.
  Status: inconsistent with the current rank-one 77-unlink branch if read literally.
  Reason: it records several obstructed/open fields, yet also sets `globalDonaldsonBaseGeometryStatusCertificate = .matches`. This should not be consumed as a discharge of the current global background problem.

### Public text that overstates current scope

- `README.md`
  The line advertising an "Explicit closed-form G2 ansatz on a K3-coassociative neck" is acceptable only as a neck-level statement, not as a compact torsion-free `K_7` theorem.

- `CITATION.md`
  Phrases such as "Donaldson direct chain closed" must be read as closure of a formal or certificate pipeline, not as Level E on compact `K_7`, unless paired with explicit anisotropic perturbation and compact descent artifacts.

- `docs/GIFT_FOR_EVERYONE.md`
  The sentence "Joyce's theorem guarantees we can have a torsion-free metric on K7" is too strong for the current branch. The natural compact Joyce theorem fails in the collapsing regime, and the required K3-fibered anisotropic refinement is not yet present as a proved theorem in the repo.

### Public Phase-3 text now stale in the other direction

- `paper/phase3_kappa_src_extraction.md`
  The public note still frames `gamma_src` as an open quadratic extraction task.
  It should be reconciled with the private theorem-grade result
  `C_src = 27/16`, cubic in `c_0`, and with the parity explanation in
  `private/canonical/notes/axis2_codex_reconciliation_2026_07_02.md`.

- `paper/phase3_nonlinear_closure.md` and
  `paper/phase3_source_nonlinear_constants.md`
  Any statements still treating `C_src <= 2` and `C_nl <= 1` as the current
  best constants should be read as superseded envelopes for `D0`, not as the
  latest sharp-current bookkeeping.

- `legacy/paper/phase3_effective_jacobi_parametrix_candidate.md`
  The public Task 4 candidate is superseded for the D0 scalar constant by the
  Stage D `K_Sch^Maz <= 16/3` import. Keep the file as the original experiment
  and as a reproducibility trail, not as the active citation target.

- `legacy/paper/phase3_projection_commutator_certificate.md`
  Any numerical example using `K_Sch^Maz <= 17` is historical. The current
  citable scalar bound is `16/3`; the certificate remains useful for the
  projection/commutator convention.

### Public Phase-4 candidate text now superseded

- `legacy/paper/phase4_adiabatic_operator_form.md`
  Keep as the operator-form/power-counting scaffold only. Active P4.1 values are
  in `phase4_donaldson_coefficients_values.json` and checked by
  `phase4_donaldson_coefficients_check.json`.

- `paper/phase4_ar_*candidate*.md`
  Keep as the public construction trail for the Neumann-slot candidates. Stage D
  imports the theorem-grade L2 assembly; Stage E packages the outward-rounded
  coefficient layer. These candidate notes are no longer load-bearing theorem
  inputs.

## Current branch classification

### Canonical analytic branch

- `private/canonical/papers/rank_one_branched_adiabatic/draft.md`
- `private/canonical/scripts/axis2_*`
- `private/canonical/results/axis2_*`
- `private/canonical/results/phase_iii3_*`
- `private/canonical/results/k3_closedform_witness_manifest.json`
- `K7-Lean/GIFT/Foundations/K3ClosedFormWitness.lean`
- `K7-Lean/GIFT/Foundations/K3ClosedFormBoxEnclosures.lean`
- `K7-Lean/GIFT/Foundations/G2DonaldsonLinkCohomology.lean`
- `K7-Lean/GIFT/Foundations/CollarResummationCertificate.lean`

These are the current rank-one / 77-unlink / branched adiabatic materials.

### Legacy or comparison branches

- `K7-Lean/GIFT/Foundations/AnalyticalMetric.lean`
  Classification: comparison only. Useful as a local flat reference, not a dependency for compact `K_7`.

- `K7-Lean/GIFT/Foundations/DonaldsonGlobalBaseAudit.lean`
  Classification: legacy exploratory branch tied to the older Fano-link and global-coframe audit.

- `K7-Lean/GIFT/Foundations/MetricGapClosure.lean`
  Classification: meta-ledger. Useful for honest separation of levels, but not a proof artifact for the current rank-one branch.

- `private/computation/metric_certification/run_g*.py`
  Classification: older TCS / Joyce / global-assembly track. Keep quarantined unless a result is re-exported into the current axis2 branch with explicit hypotheses.

## Immediate repository obligations

- No theorem, docstring, or summary may call a result an "exact metric on `K_7`" unless it proves compactness, global descent, positivity, and torsion-freeness on `K_7`.
- The anisotropic perturbation theorem `(J)` must be treated as open until represented by an actual operator theorem with spaces, parametrix, and nonlinear estimate.
- The current best honest compact statement is still conditional on `(J)`, and the current best closed-form statement is neck-level or box-local, not compact-global.
