# Phase 3.S1.n: first real ansatz for `kappa_src`

## Status

Conditional ansatz only. Superseded as the current source coefficient model
for the fixed-Sigma Lemma 5.9 channel by the private 2026-07-02 P1 chain:
the sharp-current obstruction contribution is cubic in `c_0` with coefficient
`27/16`, not the quadratic comparison ansatz `3/8`.

This file does not claim that `kappa_src` has been computed. It records the
first non-arbitrary candidate compatible with the present draft and with the
already frozen lower-root channel data.

## Purpose

After:

- [phase3_source_to_obstruction_bridge.md](phase3_source_to_obstruction_bridge.md),
- [phase3_Pi_obs_constant_mode.md](phase3_Pi_obs_constant_mode.md),
- [phase3_mu_minus_half_channel.md](phase3_mu_minus_half_channel.md),

the next missing step is not yet a theorem-grade coefficient extraction. The
next honest step is to record the first serious candidate for the constant-mode
pair `(kappa_src,R^(i), kappa_src,I^(i))`.

## Input from the current draft

Lemma 5.9 in the current draft states:

1. the seed branch datum on each collar is

   `h_loc,branch = c0^(i) w^(3/2) alpha_1 + ...`

   with `c0^(i) in R`, `s`-independent;

2. the leading source is quadratic in `c0^(i)`;

3. the leading lower-root source is `s`-uniform up to `O(kappa_g r0)`.

These three facts already force a strong shape for the first ansatz.

## Real-direction consequence

Because the leading coefficient is quadratic in one real scalar `c0^(i)`, the
dominant constant-mode obstruction pair should lie in one fixed real direction
of the basis `{e_i^R, e_i^I}`.

So the first admissible ansatz is:

`Pi_obs^{(i)}(m(h_bar))`
`~`
`(c0^(i))^2 * (kappa_src,R^(i) e_i^R + kappa_src,I^(i) e_i^I)`

with

- `kappa_src,I^(i) = 0` at leading order,
- `kappa_src,R^(i)` independent of `s` at leading order.

This is the simplest shape compatible with the present normalization where the
seed datum itself is written using one real scalar `c0^(i)`.

## Geometric coefficient candidate

Lemma 5.9 also says that the source coefficient is polynomial in

`A_bulk|_{Sigma_i}`.

In the rank-one reduced channel, the first scalar candidate is therefore

`kappa_src,R^(i) = gamma_src * A_bulk(alpha_1, alpha_1)|_{Sigma_i}`

with one unresolved scalar `gamma_src`.

At this stage:

- the dependence on `A_bulk(alpha_1, alpha_1)` is structurally justified,
- the scalar `gamma_src` is still unknown.

## First comparison value for `gamma_src`

The closest exact coefficient already frozen in the same lower-root channel is
the branch-motion normal-form value

`gamma_geom = 3/8`.

So the first comparison ansatz is

`gamma_src ?= 3/8`,

hence

`kappa_src,R^(i) ?= (3/8) * A_bulk(alpha_1, alpha_1)|_{Sigma_i}`,

`kappa_src,I^(i) ?= 0`.

This is not a theorem. It is the first channel-compatible coefficient guess.

## Why this ansatz is honest

This ansatz uses only information already present in the branch:

1. real scalar seed coefficient `c0^(i)` from the draft;
2. quadratic dependence on that scalar from Lemma 5.9;
3. the fixed real obstruction basis from
   [phase3_mu_minus_half_basis.md](phase3_mu_minus_half_basis.md);
4. the exact lower-root coefficient `3/8` from
   [phase3_mu_minus_half_channel.md](phase3_mu_minus_half_channel.md).

What it does **not** do is identify the nonlinear source coefficient with the
branch-motion coefficient. That equality remains unproved.

## Covariant quadratic form for later use

The current draft writes only one real scalar `c0^(i)`, but the extended-domain
layer already uses real pairs `(c_i^R, c_i^I)`.

So the natural quadratic extension to the full real pair is:

`Q_quad(c_i^R, c_i^I)`
`:=`
`gamma_src * A_bulk(alpha_1, alpha_1)`
`*`
`((c_i^R)^2 - (c_i^I)^2, 2 c_i^R c_i^I)`.

This is the real form of the complex square map.

In the present real-seed regime `c_i^I = 0`, it reduces to

`(gamma_src * A_bulk(alpha_1, alpha_1) * (c0^(i))^2, 0)`.

This is the correct shape to keep on file for later, even though only the
real-axis specialization is currently supported by the draft.

## `D0` consequence

Using the currently frozen lower bound

`A_bulk(alpha_1, alpha_1) >= 0.43290043290043284`,

the comparison ansatz `gamma_src = 3/8` yields the explicit lower comparison
value

`kappa_src,R^(i) >= 0.1623376623376623`

at the normalized datum `D0`, again only as a candidate comparison scale.

## Current operational use

This file is now historical/diagnostic. It should test two distinct statements
without promoting the old comparison ansatz:

1. a regular-sector diagnostic:

   if a quadratic fixed-Sigma term appears, check whether it belongs to the
   `alpha_1`-perp regular sigma-even sector rather than the sigma-odd
   obstruction channel;

2. a lower-root comparison:

   compare any branch-motion normal-form coefficient against
   `(3/8) * A_bulk(alpha_1, alpha_1)`.

Neither statement competes with the active fixed-Sigma sigma-odd source
coefficient `C_src = 27/16`.

Post-2026-07-02 reading: this remains a useful historical and diagnostic
ansatz, but the public Phase-3 ledger should now compare it against the private
parity result:

- quadratic fixed-Sigma terms land in the sigma-even regular sector;
- the sigma-odd obstruction source is cubic;
- the sharp-current `D0` source constant is `C_src = 27/16`.
