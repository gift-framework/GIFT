# Phase 3.S1.p: local quadratic surrogate for `gamma_src`

## Status

Explicit surrogate only. Superseded for the fixed-discriminant Lemma 5.9
source channel by the private 2026-07-02 theorem-grade extraction:

- the quadratic `gamma_src,sur = 3/8` lives only at the modeling/surrogate
  layer recorded here;
- the sharp-current fixed-Sigma obstruction-channel coefficient is cubic,
  `C_src = 27/16`;
- the parity reconciliation is recorded in
  `private/canonical/notes/axis2_codex_reconciliation_2026_07_02.md`.

This file does not prove the true quadratic tensor coefficient in Lemma 5.9.
It records the smallest local surrogate that outputs a concrete candidate
`gamma_src` while exposing exactly one additional modeling assumption.

## Purpose

At this point, the branch has:

- the exact curvature coefficient `3/4` of `d_w^2(w^(3/2))`,
- the exact lower-root branch-motion coefficient `3/8`,
- the fixed lower-root basis and constant-mode obstruction layer,
- the first real-axis ansatz for `kappa_src`.

What is still missing is a theorem-grade extraction of the true nonlinear source
coefficient. The present surrogate fills only that gap, and only conditionally.

## Surrogate assumption

Introduce the single modeling assumption:

`(S_qsur)`

In the rank-one constant-section channel, the leading obstruction projection of
the quadratic seed residual is carried by the same lower-root normal-form
coefficient as the already certified branch-motion scaffold.

In formulas:

`Pi_obs^(i)(m(h_bar))_lead`
`:=`
`(c0^(i))^2 * gamma_src,sur * A_bulk(alpha_1, alpha_1)|_{Sigma_i} * e_i^R`,

with

`gamma_src,sur = 3/8`,

and zero leading `e_i^I` component.

## Why this is the minimal surrogate

This surrogate does not invent a new number. It reuses the only exact lower-root
channel coefficient already certified in the branch:

- [phase3_mu_minus_half_channel.md](/home/brieuc/gift-framework/GIFT/paper/phase3_mu_minus_half_channel.md).

So the extra assumption is not

- "some unknown constant is about one",

but specifically

- "the nonlinear source projection uses the same channel coefficient as the
  branch-motion normal form."

That is a sharp, falsifiable modeling statement.

## Output of the surrogate

Under `(S_qsur)`, the constant-mode obstruction coefficient becomes

`kappa_src,R^(i) = (3/8) * A_bulk(alpha_1, alpha_1)|_{Sigma_i}`,

`kappa_src,I^(i) = 0`.

Equivalently, in the pair coordinates `(c_i^R, c_i^I)`:

`Q_sur(c_i^R, c_i^I)`
`=`
`(3/8) * A_bulk(alpha_1, alpha_1)`
`*`
`((c_i^R)^2 - (c_i^I)^2, 2 c_i^R c_i^I)`.

## `D0` specialization

Using the currently frozen lower bound

`A_bulk(alpha_1, alpha_1) >= 0.43290043290043284`,

the surrogate gives

`kappa_src,R^(i) >= 0.1623376623376623`,

and the corresponding weighted source comparison scale

`kappa_src,R^(i) * r0^(7/2) >= 1.6233766233766234e-8`.

## Proper interpretation

This surrogate is useful only for two things:

1. sharpening downstream quantitative checks,
2. defining a concrete coefficient target for future local extraction.

It must **not** be cited as the true value of `gamma_src`.

After the private P1 chain, it must also not be cited as the current best
coefficient for the fixed-Sigma sigma-odd source in Lemma 5.9. The current best
coefficient there is `27/16`, cubic in `c_0`. The `3/8` surrogate may still be
useful as a comparison value for branch-motion normal forms or for diagnosing
which channel a quadratic contribution belongs to.

## What would falsify the surrogate

Any of the following would falsify `(S_qsur)`:

1. a local symbolic calculation showing a different lower-root scalar
   coefficient;
2. a nonzero leading `e_i^I` component for the real-axis seed;
3. a theorem-level identification of `Pi_obs` with a trace formula producing a
   coefficient different from `3/8`.

## Why it is still worth keeping

The repository now has three tiers:

1. proven channel geometry:
   exact `3/8` in the lower-root normal form;
2. explicit surrogate:
   `gamma_src,sur = 3/8`;
3. theorem target:
   true extracted `gamma_src`.

That is cleaner than jumping directly from “unknown `O(1)`” to “exact theorem”.
