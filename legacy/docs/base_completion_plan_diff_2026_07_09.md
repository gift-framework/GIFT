# Base Completion Plan Diff - 2026-07-09

This note compares the original completion plan against the current repository
state after the Stage D / Stage E work recorded on 2026-07-09.

Status labels:

- `DONE`: artifact exists and matches the requested scope.
- `PARTIAL`: useful artifact exists, but the acceptance gate is not fully met.
- `CONDITIONAL`: statement is isolated as an assumption/import, not discharged in
  the public proof chain.
- `OPEN`: no theorem-grade closure yet.
- `STALE`: older artifact remains true only as historical/candidate material and
  needs a banner, ledger update, or replacement.

## Headline Diff

Level E is still conditional. The file `paper/theorem_E_exact.md` still states
"target theorem only" and depends on named analytic artifacts, including
`anisotropic_joyce_theorem_J`.

Level Q has moved forward materially. Stage D fixed the D0 constant package
around `K_Sch = 16/3`; Stage E exposed the coefficient table, compact Level Q
package, and paper-facing checker:

- `scripts/phase4_donaldson_coefficients_checker.py`: `172/172 pass`
- `scripts/phase4_level_Q_coefficients_checker.py`: `185/185 pass`
- `scripts/phase4_level_Q_table_checker.py`: `43/43 pass`

Level CF remains separate and open. No compact-global finite closed expression
for the torsion-free `G_2` metric is present.

## Phase 0 - Truth Ledger And Canonicalisation

Status: `PARTIAL -> mostly DONE, needs Stage D/E refresh`.

Present artifacts:

- `audit/theorem_ledger.yaml`
- `audit/dependency_graph.json`
- `audit/claim_scope.md`
- `audit/legacy_quarantine.md`

The ledger already prevents the main false promotions:

- `AnalyticalMetric.lean` is local flat `R^7`, not compact `K_7`.
- `K3ClosedFormWitness.lean` is box-local interval certification, not a global
  Ricci-flat K3 metric.
- `CollarResummationCertificate.lean` is a binomial/indicial certificate, not a
  PDE convergence theorem.
- `DonaldsonGlobalBaseAudit.lean` is quarantined as legacy/exploratory.

Remaining work:

- Refresh ledger entries that still describe Phase 3/4 artifacts as candidates
  when Stage D/E now provide sharper imported D0 packages.
- Make the dependency graph reflect the current chain:
  `D0 constants -> K_Sch=16/3 -> L2 AR assembly -> Level Q coefficient table`.
- Add explicit stale banners or cross-links for older candidate files that still
  advertise weaker constants, especially `K_Sch <= 17` era artifacts.

## Phase 1 - Precise Mathematical Target

Status: `DONE for separation, CONDITIONAL for theorem`.

Present artifacts:

- `paper/theorem_E_exact.md`
- `paper/theorem_Q_certified.md`
- `paper/conjecture_CF_finite_closed_form.md`
- `paper/assumptions_map.tex`

The three levels are correctly separated:

- Level E: target compact torsion-free existence theorem, not discharged.
- Level Q: certified constants and interval packages.
- Level CF: separate closed-form research programme.

Remaining work:

- Update `docs/analytic_status.md`; it still says "partial quantitative
  artifacts" and does not yet cite the Stage E compact Level Q package.
- Keep `paper/theorem_E_exact.md` conditional until Phase 5 and the compact
  topology/global geometry wrappers are actually discharged.

## Phase 2 - Make Datum D0 Real

Status: `PARTIAL / CONDITIONAL`.

Present artifact:

- `certificates/datum_D0.json`

Current progress:

- D0 constants are centralized.
- Stage D repinned the certificate to the current private commit and added the
  `L1.6_K_Sch` and `L2_AR_assembly_theorem` blocks.
- The active source constant is `C_src = 27/16`; `gamma_src = 3/8` is demoted to
  a legacy diagnostic.

Remaining work:

- Smooth compact realization of the actual fibration is still not a theorem-grade
  public artifact.
- The topological chain `b2=21`, `b3=77`, `pi1=1` remains conditional on the
  documented external/Leray package.
- The fresh-machine acceptance gate still needs a stricter audit for every field:
  source script, commit hash, outward rounding, unit convention, and independent
  checker.
- The certificate still needs final treatment of base geometry slots such as
  `inj_base`, `curv_base`, and exact two-sided handling of
  `A_bulk(alpha1, alpha1)` if those are to be theorem inputs.

## Phase 3 - Global Maximal Section

Status: `PARTIAL / CONDITIONAL`.

Current progress:

- The rank-one 77-unlink branch has a much sharper D0 numerical closure than the
  old public candidate notes.
- Stage D imports the current private `K_Sch = 16/3` package and records the
  projected source/obstruction bookkeeping.

Remaining work:

- The public ledger still contains entries such as
  `global_maximal_section_scaffold: heuristic`.
- The acceptance gate asks for a global nonlinear theorem with explicit constants
  for the actual right inverse. That final theorem artifact is not yet visible as
  a clean public closure.
- Older candidate files around `K_Sch <= 17` should be marked historical or
  redirected to the Stage D closure.

Practical next step:

- Reconcile the public Phase 3 ledger with the private L1 closure. Either import
  a named theorem artifact into `GIFT`, or keep Phase 3 explicitly conditional.

## Phase 4 - Adiabatic Reconstruction

Status: `DONE at D0 coefficient-package level, CONDITIONAL for full public chain`.

Current progress:

- Stage D added the `L2_AR_assembly_theorem` block to `datum_D0.json`.
- Stage E regenerated and checked:
  - `certificates/phase4_donaldson_coefficients_values.json`
  - `certificates/phase4_donaldson_coefficients_check.json`
  - `certificates/phase4_level_Q_coefficients.json`
  - `certificates/phase4_level_Q_coefficients_check.json`
  - `certificates/phase4_level_Q_table_check.json`
- The paper-facing Level Q table now includes the citable threshold
  `R_threshold <= 3664.066`.

Remaining work:

- Public theorem stitching is still needed so the Stage D/E certificates become
  named dependencies of `theorem_E_exact.md`.
- Keep the distinction between D0-certified reconstruction and any broader
  uniform statement beyond the explicit datum.

## Phase 5 - Hypothesis (J)

Status: `OPEN`.

Present artifact:

- `paper/anisotropic_joyce_theorem.md`

This file is only a specification. It correctly records:

- anisotropic spaces `X_R`, `Y_R`;
- three-region parametrix architecture;
- uniform right inverse target;
- nonlinear quadratic estimate;
- harmonic/gauge/obstruction audit.

Remaining work:

- Prove the anisotropic elliptic package.
- Construct the three-region parametrix and bound the Neumann error.
- Certify `C_G`, `C_Q`, `q`, `R_J`.
- Close the contraction theorem for `eta_R`.

This is still the decisive Level E lock.

## Phase 6 - Holonomy And Global Geometry

Status: `OPEN / CONDITIONAL`.

Current progress:

- The target dependency chain is stated in `paper/theorem_E_exact.md`.

Remaining work:

- No final Lean export theorem of the requested shape was found.
- The wrapper depends on compact realization/topology from Phase 2 and the
  torsion-free perturbation theorem from Phase 5.
- Once those are discharged, the holonomy step should be comparatively small:
  compact positive torsion-free `G_2` + `pi1(K_7)=1` -> full holonomy `G_2`,
  with named assumptions exposed.

## Phase 7 - Closed-Form Programme

Status: `OPEN / separate`.

Present artifact:

- `paper/conjecture_CF_finite_closed_form.md`

Remaining work:

- No finite compact-global expression has been verified against both torsion
  equations.
- The honest endpoint remains one of:
  - structural cancellation,
  - finite invariant ansatz,
  - explicit datum plus certified analytic existence plus computable implicit
    correction.

## First Five Codex Tasks

Task 1 - Repository truth ledger: `PARTIAL/DONE`.
Initial ledger exists. It needs Stage D/E refresh and stale-candidate cleanup.

Task 2 - Canonical analytic branch: `PARTIAL/DONE`.
The current rank-one 77-unlink branch is identified; legacy branches are mostly
quarantined.

Task 3 - Hypothesis (J) specification: `DONE`.
`paper/anisotropic_joyce_theorem.md` is a specification only, as requested.

Task 4 - Effective Jacobi parametrix: `PARTIAL -> advanced by Stage D`.
The repo has historical candidate artifacts, while Stage D imports the sharper
`K_Sch = 16/3` D0 closure. The public ledger needs reconciliation.

Task 5 - Global K3 positivity route: `OPEN`.
The K3 witness remains box-local. The global SOS/Positivstellensatz route is
not yet closed.

## Remaining Work, Ordered

1. Refresh Phase 0 truth ledger and dependency graph for Stage D/E.
2. Mark stale Phase 3 candidate artifacts as historical or redirect them to the
   current `K_Sch = 16/3` package.
3. Finish or explicitly isolate the Phase 2 compact datum/topology theorem
   chain, including missing certificate fields.
4. Import or write the public Phase 3 global maximal-section theorem artifact.
5. Stitch the Stage D/E Phase 4 certificates into named public theorem
   dependencies.
6. Attack Phase 5 `(J)`: anisotropic spaces, right inverse, obstruction audit,
   nonlinear contraction.
7. Add the Phase 6 Lean/global-geometry wrapper after Phase 5 and the compact
   topology slots are theorem-grade.
8. Keep Phase 7 CF separate unless an exact finite ansatz is directly verified.

## Current Honest Summary

The next logical engineering tranche is not another coefficient squeeze. It is
repository consolidation:

`ledger refresh -> stale artifact quarantine -> public theorem stitching`.

The next mathematical hard tranche is Phase 5:

`anisotropic Joyce theorem (J)`.

After that, Level E can become a compact-global theorem. Until then, the main
existence theorem remains conditional, while Level Q for the D0 coefficient
package is substantially stronger than the original baseline plan.
