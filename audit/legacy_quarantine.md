# Legacy Quarantine

This file separates the current rank-one 77-unlink adiabatic branch from older or orthogonal material.

## Canonical branch for current work

The current analytic branch is:

- `private/canonical/papers/rank_one_branched_adiabatic/draft.md`
- `private/canonical/scripts/axis2_*`
- `private/canonical/results/axis2_*`
- `private/canonical/results/phase_iii3_*`
- `K7-Lean/GIFT/Foundations/K3ClosedFormWitness.lean`
- `K7-Lean/GIFT/Foundations/K3ClosedFormBoxEnclosures.lean`
- `K7-Lean/GIFT/Foundations/G2DonaldsonLinkCohomology.lean`
- `K7-Lean/GIFT/Foundations/CollarResummationCertificate.lean`

These artifacts are allowed to feed the Phase 0-5 theorem chain.

## Quarantined as legacy or comparison-only

### Flat `R^7` branch

- `K7-Lean/GIFT/Foundations/AnalyticalMetric.lean`
- Role: comparison / local sanity check only.
- Not a dependency for compact `K_7`.

### Older Fano-link / global-coframe branch

- `K7-Lean/GIFT/Foundations/DonaldsonGlobalBaseAudit.lean`
- `K7-Lean/contrib/python/gift_core/geometry/wirtinger_symbolic.py`
- Role: historical exploration of Fano-link compatibility and symbolic Wirtinger layers.
- Status for current branch: orthogonal unless explicitly re-imported through a rank-one 77-unlink argument.

### TCS / generic Joyce metric-certification branch

- `private/computation/metric_certification/run_g1_tcs_atlas.py`
- `private/computation/metric_certification/run_g2_joyce_certified.py`
- `private/computation/metric_certification/run_g4_holonomy_proof.py`
- `private/computation/metric_certification/run_g8_torsion_free.py`
- Role: older TCS/global assembly route.
- Status for current branch: do not reuse its conclusions without an explicit bridge theorem to the present datum `D0`.

### Mixed meta-ledger files

- `K7-Lean/GIFT/Foundations/MetricGapClosure.lean`
- Role: honest separation of levels, useful as repository hygiene.
- Status for current branch: not itself a proof of any Phase 2-5 gate.

## Operational rule

If a future proof depends on one of the quarantined files, the new artifact must state one of:

- "comparison only"
- "reused constant only"
- "legacy theorem imported with unchanged hypotheses"

Nothing from the quarantined branch may silently discharge a gate in the current programme.
