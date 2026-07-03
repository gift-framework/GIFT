# Claim Scope Audit

This note records where existing comments or summaries currently overstate the mathematical scope.

## File-level audit targets

### `core/GIFT/Foundations/AnalyticalMetric.lean`

- Actual content: a constant scaled `G_2` form on `R^7` with scaled identity metric and zero torsion by constancy.
- Safe scope label: `local`.
- Unsafe promotion: anything calling it an exact compact metric on `K_7`.
- Required quarantine sentence: "This module is a local flat model and does not construct a compact `K_7` metric."

### `core/GIFT/Foundations/K3ClosedFormWitness.lean`

- Actual content: Lean-checked box-local residual aggregate on 4000 Krawczyk boxes.
- Safe scope label: `box-local`.
- Unsafe promotion: "global Ricci-flat K3 metric", "whole-K3 positivity", or "global spectral control".
- Required carry-over gap: whole-K3 positivity and exact-form spectral lower bound remain separate tasks.

### `core/GIFT/Foundations/G2DonaldsonLinkCohomology.lean`

- Actual content: arithmetic consequences of a modeled branched-cover formula and reflection invariant rank.
- Safe scope label: `global` arithmetic interface.
- Unsafe promotion: "the smooth compact K3-fibration with these Betti numbers exists" unless the Leray/Wang/Van Kampen theorem chain is independently supplied.

### `core/GIFT/Foundations/CollarResummationCertificate.lean`

- Actual content: scalar resummation and indicial parity.
- Safe scope label: `collar`.
- Unsafe promotion: a statement that the collar PDE, normal operator inversion, or edge calculus has thereby converged.

### `core/GIFT/Foundations/DonaldsonGlobalBaseAudit.lean`

- Actual content: exploratory status flags from an older Fano-link / global-coframe route.
- Safe scope label: `legacy`.
- Unsafe promotion: treating `globalDonaldsonBaseGeometryStatusCertificate = .matches` as a discharge of the current rank-one 77-unlink global maximal background.

## Public prose requiring care

### `GIFT/docs/GIFT_FOR_EVERYONE.md`

- Current phrase: "Joyce's theorem guarantees we can have a torsion-free metric on K7."
- Required interpretation: false as written for the current collapsing branch.
- Correct replacement: a compact bounded-geometry Joyce theorem does not directly apply; a separate anisotropic K3-fibered perturbation theorem is needed.

### `GIFT/README.md` and `GIFT/CITATION.md`

- Current issue: neck-level or certificate-level closure language can be misread as compact-global Level E closure.
- Required interpretation: any "closed-form" phrase must be annotated by its actual scope: neck-level, local, box-local, or conditional on `(J)`.

## Canonical naming rule

No theorem, docstring, status string, or release note may use any of the following phrases without compact-global proof artifacts:

- "exact metric on K7"
- "torsion-free metric on K7"
- "closed-form metric on K7"
- "Joyce applies"
- "Donaldson converges"

Each such phrase must be replaced by one of:

- "local flat model on R^7"
- "box-local K3 residual certificate"
- "conditional on anisotropic perturbation theorem (J)"
- "formal finite-order adiabatic ansatz"
- "arithmetic/topological shadow"
