# Phase 3.5: source and nonlinear constants for the global maximal-section closure

## Status

Partial certification only, with the original placeholder layer superseded at
`D0` by the private 2026-07-02 P1/C_nl chain. This file separates:

- constants that are already numerically frozen at `D0`,
- constants whose *smallness consequence* is numerically clear,
- constants whose *analytic derivation* is still missing at theorem level.

It is the bridge between [phase3_nonlinear_closure.md](/home/brieuc/gift-framework/GIFT/paper/phase3_nonlinear_closure.md) and a future fully rigorous discharge of Proposition 3.1ter.

Latest sharp-current values from `private`:

- `C_src = 27/16`, cubic in the fixed-discriminant sigma-odd channel.
- `C_nl = 2/3`, from the symmetric-space second-variation bound.
- `r_G(D0) ~= 1.51e-4`, margin approximately `3318x` below `1/2`.

The older `C_src <= 2`, `C_nl <= 1` bookkeeping below should be read as the
conservative envelope layer that was used before these private derivations.

## Why this file is needed

Phase 3.4 reduced the global maximal-section closure to four constants:

- `C_lin = ||J_h^{{ext},-1}||`,
- `C_src = ||m(h_bar_global)||_{Y_{-3}}` after factoring out the `r0^(7/2)` scale,
- `C_nl`,
- the derived ball radius `rho = 2 C_lin C_src`.

At present the branch already has an excellent *numerical margin* at `D0`, but that is not the same thing as a theorem. The missing issue is not whether the contraction is small; it is whether the constants entering that contraction have been derived with explicit certified estimates rather than inserted as "universal O(1)" placeholders.

## Current numerical picture at `D0`

The current source for `(G-quant)` is:

- [axis2_hypothesis_G_discharge_2026_07_01.py](/home/brieuc/gift-framework/private/canonical/scripts/axis2_hypothesis_G_discharge_2026_07_01.py),
- [axis2_hypothesis_G_discharge_2026_07_01.json](/home/brieuc/gift-framework/private/canonical/results/axis2_hypothesis_G_discharge_2026_07_01.json),
- encoded locally in [phase3_G_closure_status.json](/home/brieuc/gift-framework/GIFT/certificates/phase3_G_closure_status.json).

Using the original placeholders:

- `C_lin <= 36.635`,
- `||D^2 m|| <= 1`,
- `C_src <= 2`,
- `r0 = 10^-2`,
- `eta_aff(D0) = 0`,

one gets:

- `||m(h_bar_global)||_{Y_-3} <= 2 * 10^-7`,
- `r_G(D0) <= 2.68424645 * 10^-4`,
- `rho = 2 C_lin C_src_actual <= 1.4654 * 10^-5`,
- `4 C_lin^2 C_nl C_src_actual <= 1.07369858 * 10^-3`.

So the contraction inequality is numerically nowhere near the threshold `1/2`.

Using the sharp-current private constants:

- `C_src = 27/16`,
- `C_nl = 2/3`,

one gets:

- `||m(h_bar_global)||_{Y_-3} <= 1.6875 * 10^-7`,
- `r_G(D0) <= 1.51 * 10^-4`,
- margin approximately `3318x` below the threshold `1/2`.

## Exact status of each constant

### `C_lin`

Current status: conditional theorem input.

Meaning:

- this is the operator norm of the augmented right inverse for
  `J_h : X_beta^{ext} -> Y_{beta-2}`.

What is already frozen:

- the reduced obstruction architecture,
- the `A_red` Neumann margin,
- the working value `36.635`.

What still blocks full theorem status:

- commutator bounds in the annuli,
- bulk transmission estimate,
- final assembly of `G_true` from the parametrix.

This belongs to the completion of Phase 3.2 and Phase 3.3.

### `C_src`

Current status: theorem-grade at `D0` in the private chain; public integration
still pending.

Meaning:

- the coefficient in
  `||m(h_bar_global)||_{Y_-3} <= C_src * r0^(7/2) * ||A_bulk||_{C^{0,alpha}} + C_aff * eta_aff`.

What is already numerically clear in the old public layer:

- the scale `r0^(7/2)`,
- the disappearance of the affine mismatch term at `D0`,
- the final smallness `2 * 10^-7` under the current choice `C_src <= 2`.

What the private 2026-07-02 chain changes:

- the quadratic `3/8` public surrogate is superseded for the fixed-Sigma
  Lemma 5.9 channel by a cubic sigma-odd coefficient;
- the theorem-grade value is `C_src = 27/16`;
- the `Y_-3` scale remains `r0^(7/2)`;
- the subleading tail is absorbed by a geometric factor at `r0 = 10^-2`.

What still needs public integration:

- reconcile this file and the public surrogate notes with the private parity
  and reconciliation notes;
- separate any remaining projection and annular commutator terms as public
  theorem artifacts.

So `C_src` is no longer the first concrete constant gap at `D0`; it is now a
public-ledger synchronization task plus a projection/commutator integration
task.

### `C_nl`

Current status: theorem-grade at `D0` in the private chain; public integration
still pending.

Meaning:

- the constant in the weighted quadratic estimate
  `||N_h(u) - N_h(v)|| <= C_nl (||u|| + ||v||) ||u-v||`.

What was already available:

- a frozen datum bound `||D^2 m|| <= 1`,
- the draft fixed-point inequalities in Proposition 3.1ter.

What the private 2026-07-02 chain changes:

- the symmetric-space normal-coordinate expansion and type-IV curvature bound
  give `C_nl = 2/3`;
- the pure alpha_1 sigma-odd contribution vanishes by collinearity;
- even the conservative fallback recovers the old envelope `C_nl <= 1`.

What remains open at the global theorem level:

- the actual weighted bilinear estimate in the full edge spaces,
- proof that the nonlinear map preserves `X_beta^{ext}` globally,
- proof that the finite-dimensional `R^{2N}` layer does not grow under iteration.

So `C_nl` is no longer an unexplained `O(1)` envelope at `D0`, but the full
global mapping theorem still needs to be written.

## What can already be promoted safely

The following statement is already safe:

For the normalized datum `D0`, the current sharp-source bookkeeping gives
`C_src = 27/16`, `C_nl = 2/3`, and the global maximal-section contraction
inequality is numerically closed with a very large margin, conditional on the
global linear/operator package being promoted to theorem form.

This is a legitimate *conditional numerical closure* statement.

## What cannot yet be promoted safely

The following statements are not yet safe:

- "Proposition 3.1ter is fully rigorous."
- "The global maximal background is discharged unconditionally."
- "Phase 3 is complete."

The reason is narrow and concrete:

- the global linear right inverse and parametrix still need theorem artifacts,
- the public Phase-3 projection/commutator bookkeeping still needs integration,
- the weighted nonlinear estimate still needs to be stated and proved in the
  full `X_beta^{ext}` mapping setting.

## Minimal path to theorem-grade closure

The next artifacts should be built separately.

### S1. Residual-source certificate

Integrate the private certificate for

`||m(h_bar_global)||_{Y_-3} <= C_src_exact`

and keep the public decomposition explicit:

- collar quadratic remainder,
- annular cutoff commutator,
- affine mismatch term.

At `D0`, the affine mismatch term should vanish exactly.

This decomposition is now frozen in:

- [phase3_source_residual_certificate.md](/home/brieuc/gift-framework/GIFT/paper/phase3_source_residual_certificate.md),
- [phase3_source_residual_ledger.json](/home/brieuc/gift-framework/GIFT/certificates/phase3_source_residual_ledger.json).

### S2. Weighted nonlinear estimate

Integrate the private `C_nl = 2/3` Simons-type estimate into a theorem note
for

`||N_h(u) - N_h(v)||_{Y_{beta-2}} <= C_nl_exact (||u|| + ||v||) ||u-v||`

on the chosen Banach ball.

This estimate must be phrased in the actual norms of
[phase3_spaces.md](/home/brieuc/gift-framework/GIFT/paper/phase3_spaces.md), not in a schematic unweighted norm.

## Relation to the current certificate

The machine-readable status freeze for this file is:

- [phase3_G_closure_status.json](/home/brieuc/gift-framework/GIFT/certificates/phase3_G_closure_status.json).

That certificate used to be deliberately conservative and predated the private
P1/C_nl updates. It has now been moved to the sharp-current constants
`C_src = 27/16` and `C_nl = 2/3`; any older copy with the `2/1` envelope
bookkeeping is superseded.
