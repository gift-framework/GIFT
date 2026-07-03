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

The repository currently contains a credible conditional programme for Level E, partial quantitative artifacts for Level Q, and no finite closed-form compact torsion-free metric for Level CF.

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
metric theorem. They should be used to update the Phase-3 numerical closure
ledger, while preserving the existing warnings about the missing global
parametrix theorem, adiabatic reconstruction theorem, anisotropic Joyce
theorem `(J)`, and Appendix C compact-topology proof.

## Strong-claim audit

### Claims that must be read narrowly

- `core/GIFT/Foundations/AnalyticalMetric.lean`
  Scope: `local`.
  Status: exact constant flat `G_2` form on `R^7`, not a compact `K_7` metric.
  Reason: the file proves constancy, scaled identity metric, and zero torsion by `d phi = d * phi = 0` for a constant form. It does not build an atlas, descent, compact quotient, or compact torsion-free structure.

- `core/GIFT/Foundations/K3ClosedFormWitness.lean`
  Scope: `box-local`.
  Status: rigorous interval certificate on 4000 frozen Krawczyk boxes for one explicit 667-parameter witness.
  Reason: the header explicitly says the result does not promote to a whole-K3 bound and isolates the remaining positivity/SOS problem.

- `core/GIFT/Foundations/G2DonaldsonLinkCohomology.lean`
  Scope: arithmetic and combinatorial only.
  Status: formal arithmetic consequences from definitions plus a hand-off to an external Leray/branched-cover derivation.
  Reason: the header explicitly states the sheaf-theoretic and analytic derivation is not formalized.

- `core/GIFT/Foundations/CollarResummationCertificate.lean`
  Scope: `collar`.
  Status: exact binomial and indicial-parity identities only.
  Reason: it proves `sum |binom(3/2,k)| = 3` and parity of `beta^2 - m^2`; it is not a convergence theorem for the collar PDE or a Fredholm theorem.

- `core/GIFT/Foundations/DonaldsonGlobalBaseAudit.lean`
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

- `GIFT/paper/phase3_kappa_src_extraction.md`
  The public note still frames `gamma_src` as an open quadratic extraction task.
  It should be reconciled with the private theorem-grade result
  `C_src = 27/16`, cubic in `c_0`, and with the parity explanation in
  `private/canonical/notes/axis2_codex_reconciliation_2026_07_02.md`.

- `GIFT/paper/phase3_nonlinear_closure.md` and
  `GIFT/paper/phase3_source_nonlinear_constants.md`
  Any statements still treating `C_src <= 2` and `C_nl <= 1` as the current
  best constants should be read as superseded envelopes for `D0`, not as the
  latest sharp-current bookkeeping.

## Current branch classification

### Canonical analytic branch

- `private/canonical/papers/rank_one_branched_adiabatic/draft.md`
- `private/canonical/scripts/axis2_*`
- `private/canonical/results/axis2_*`
- `private/canonical/results/phase_iii3_*`
- `private/canonical/results/k3_closedform_witness_manifest.json`
- `core/GIFT/Foundations/K3ClosedFormWitness.lean`
- `core/GIFT/Foundations/K3ClosedFormBoxEnclosures.lean`
- `core/GIFT/Foundations/G2DonaldsonLinkCohomology.lean`
- `core/GIFT/Foundations/CollarResummationCertificate.lean`

These are the current rank-one / 77-unlink / branched adiabatic materials.

### Legacy or comparison branches

- `core/GIFT/Foundations/AnalyticalMetric.lean`
  Classification: comparison only. Useful as a local flat reference, not a dependency for compact `K_7`.

- `core/GIFT/Foundations/DonaldsonGlobalBaseAudit.lean`
  Classification: legacy exploratory branch tied to the older Fano-link and global-coframe audit.

- `core/GIFT/Foundations/MetricGapClosure.lean`
  Classification: meta-ledger. Useful for honest separation of levels, but not a proof artifact for the current rank-one branch.

- `private/computation/metric_certification/run_g*.py`
  Classification: older TCS / Joyce / global-assembly track. Keep quarantined unless a result is re-exported into the current axis2 branch with explicit hypotheses.

## Immediate repository obligations

- No theorem, docstring, or summary may call a result an "exact metric on `K_7`" unless it proves compactness, global descent, positivity, and torsion-freeness on `K_7`.
- The anisotropic perturbation theorem `(J)` must be treated as open until represented by an actual operator theorem with spaces, parametrix, and nonlinear estimate.
- The current best honest compact statement is still conditional on `(J)`, and the current best closed-form statement is neck-level or box-local, not compact-global.
