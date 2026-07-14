# Phase 3.S1.p: reconciled `gamma_src` surrogate record

## Status

Superseded record. The old quadratic surrogate is no longer an active source
coefficient for the fixed-discriminant Lemma 5.9 sigma-odd channel.

- the sharp-current fixed-Sigma sigma-odd obstruction-channel coefficient is
  cubic,
  `C_src = 27/16`;
- the old quadratic `gamma_src,sur = 3/8` lives only as a legacy comparison for
  the lower-root normal form or for diagnosing a regular sigma-even sector;
- the parity reconciliation is recorded by
  `private/canonical/notes/axis2_codex_reconciliation_2026_07_02.md`.

This file must not be used to cite `3/8` as the fixed-Sigma sigma-odd source
coefficient.

## Purpose

At this point, the branch has:

- the exact curvature coefficient `3/4` of `d_w^2(w^(3/2))`,
- the exact lower-root branch-motion coefficient `3/8`,
- the fixed lower-root basis and constant-mode obstruction layer,
- the first real-axis ansatz for `kappa_src`.

The theorem-grade extraction now available for the fixed-Sigma sigma-odd source
coefficient is `C_src = 27/16`. What remains missing is the full global
Phase-3 Banach-space parametrix and maximal-section theorem.

## Surrogate assumption

The old file introduced the modeling assumption:

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

Stage D supersedes this as the active source model: the quadratic-in-`c0`
contribution belongs to the `alpha_1`-perp regular sigma-even sector, while the
fixed-Sigma sigma-odd obstruction source is cubic with coefficient `27/16`.

## Why the legacy value is kept

The legacy comparison does not invent a new number. It reuses the exact
lower-root normal-form coefficient already certified in the branch:

- [phase3_mu_minus_half_channel.md](../../paper/phase3_mu_minus_half_channel.md).

It remains useful only as a diagnostic comparison. It is not allowed to compete
with the active `C_src = 27/16` source coefficient.

## Active output

The public certificate now records:

`C_src = 27/16`

for the fixed-Sigma sigma-odd `mu=-1/2` obstruction channel at `D0`.

It also records the confirmed bridge identity:

`J_h(psi_R) = 2 * A_bulk * Phi_{-1/2,R}`.

## `D0` specialization

With `r0 = 10^-2`, the active scalar envelope records

`C_src * r0^(7/2) = 1.6875000000000003e-7`.

## Proper interpretation

The legacy `3/8` comparison is useful only for two things:

1. sharpening downstream quantitative checks,
2. defining a concrete coefficient target for future local extraction.

It must **not** be cited as the true value of `gamma_src` in the fixed-Sigma
sigma-odd channel. The current public ledger value there is `C_src = 27/16`,
cubic in `c_0`.

## What would falsify the surrogate

Any of the following would falsify the old `(S_qsur)` interpretation:

1. a local symbolic calculation showing a different lower-root scalar
   coefficient;
2. a nonzero leading `e_i^I` component for the real-axis seed;
3. a theorem-level identification of `Pi_obs` with a trace formula producing a
   coefficient different from `3/8`.

## Why the file is still worth keeping

The repository now has three tiers:

1. proven channel geometry:
   exact `3/8` in the lower-root normal form;
2. legacy surrogate:
   `gamma_src,sur = 3/8`, not active for the sigma-odd source;
3. sharp current source:
   `C_src = 27/16`.

That keeps the historical diagnostic without polluting the current source
ledger.
