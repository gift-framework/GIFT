# Phase 3.S1.o: confrontation of the first `kappa_src` ansatz with `G`-discharge

## Status

Compatibility check completed.

This file does not upgrade the source coefficient to theorem grade. It records
what the existing `axis2_hypothesis_G_discharge_2026_07_01` script does and
does not already constrain about the first `kappa_src` ansatz.

## What the current `G` discharge uses

The current script

`certificates/axis2/scripts/axis2_hypothesis_G_discharge_2026_07_01.py`

uses the source estimate

`||m(h_bar_global)||_{Y_-3} <= C_src * c0^2 * K_A * r0^(7/2)`

with the hard-coded conservative choice

`C_src = 2`,
`K_A = 1`,
`|c0| <= 1`.

At `D0`, this gives

`||m(h_bar_global)||_{Y_-3} <= 2e-7`.

This is an envelope estimate, not an extracted coefficient.

## What the first ansatz predicts

The first channel-compatible ansatz from
[phase3_kappa_src_first_ansatz.md](phase3_kappa_src_first_ansatz.md)
predicts the leading real obstruction coefficient

`kappa_src,R^(i) ?= (3/8) * A_bulk(alpha_1, alpha_1)|_{Sigma_i}`,

with `kappa_src,I^(i) ?= 0` at leading order.

Using the current certified lower comparison value

`A_bulk(alpha_1, alpha_1) >= 0.43290043290043284`,

this gives the `D0` comparison scale

`kappa_src,R^(i) >= 0.1623376623376623`.

If one inserts that scale directly into the weighted `r0^(7/2)` source profile,
the corresponding `D0` candidate source size is

`1.6233766233766234e-8`.

## Quantitative comparison

So at `D0`:

- current `G` discharge source envelope:
  `2.0e-7`,
- first `kappa_src` comparison scale:
  `1.6233766233766234e-8`.

The ratio is

`2.0e-7 / 1.6233766233766234e-8 = 12.32`.

The same factor propagates to the local contraction contribution:

- current `r_G` local term:
  `2.68424645e-4`,
- ansatz-level comparison term:
  `2.178771469155844e-5`.

Again the ratio is `12.32`.

## Interpretation

This means:

1. the existing `G`-discharge is fully compatible with the `3/8` ansatz;
2. the existing `C_src = 2` does not encode any sharp information about the
   source coefficient;
3. the present `G` discharge is using a conservative universal envelope about
   one order of magnitude looser than the first channel-based comparison scale.

So the confrontation does **not** falsify the ansatz. It shows instead that the
current script is too coarse to distinguish between:

- the candidate value `gamma_src = 3/8`,
- and a substantially larger but still `O(1)` universal constant.

## What this changes

The repository now has a concrete diagnostic:

- if a future local extraction gives `gamma_src` near `3/8`, then the present
  `G` discharge remains valid and simply becomes sharper by a factor `~12`;
- if a future extraction gives a very different coefficient, the current
  `G` discharge still likely survives numerically, but the theorem ledger must
  continue to classify `C_src = 2` as a conditional envelope rather than an
  extracted analytic constant.

## Next target

The next useful step is no longer to compare against `G` globally. That check
is done. The next useful step is to attack the true local coefficient:

- either identify the exact quadratic tensor coefficient in Lemma 5.9,
- or construct a local symbolic surrogate that outputs a concrete `gamma_src`
  candidate in the fixed lower-root basis.
