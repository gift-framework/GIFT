# Phase 3.S1.d: matching target between Lemma 5.9 and the lower-root normal form

## Status

Target fixed; coefficient not yet extracted.

This note states the exact quantity that still has to be computed in order to turn the present `mu = -1/2` structural information into a theorem-grade collar source coefficient.

The machine-readable freeze is:

- [phase3_mu_minus_half_matching_target.json](/home/brieuc/gift-framework/GIFT/certificates/phase3_mu_minus_half_matching_target.json).

## The problem in one line

We already know two things:

1. Lemma 5.9 says that the leading source trace `ev_{-1/2}(m(h_bar))` exists, is `s`-uniform at leading order, and is quadratic in `c0`.
2. The `rho^2` scaffold gives an exact nonzero coefficient `3/8` in the same lower-root channel `mu = -1/2`.

What is still missing is the scalar that writes the first object in the basis of the second.

## Target scalar

For each component `Sigma_i`, define the scalar `kappa_src^(i)` by

`ev_{-1/2}^{(i)}(m(h_bar)) = (c0^(i))^2 * kappa_src^(i) * Phi_{-1/2}^{std} + higher corrections`,

where `Phi_{-1/2}^{std}` is the fixed standard basis element of the `mu=-1/2`, `sigma`-odd trace space used by the extended-domain obstruction theory.

This is the coefficient that actually matters for Phase 3.

## Why this is the right target

Once `kappa_src^(i)` is known, the collar source bound becomes explicit:

`||R_quad^(i)||_{Y_-3} <= |kappa_src^(i)| * |c0^(i)|^2 * r0^(7/2) + tail`.

So the remaining problem is no longer “understand the source qualitatively”, but simply:

- compute `kappa_src^(i)`,
- bound the `O(rho^(3/2))` tail,
- sum over components.

## What this uses from the current branch

From Lemma 5.9 and its current write-up:

- the linearized contribution does not produce the leading lower-root source,
- the leading lower-root source comes from the quadratic remainder,
- the coefficient is `s`-uniform at leading order.

From the `rho^2` scaffold:

- the repository already contains an exact lower-root normal form with coefficient `3/8`.

So there is no ambiguity anymore about the channel. Only the source coefficient in that channel remains to be extracted.

## Immediate next step

The basis note is now frozen in:

- [phase3_mu_minus_half_basis.md](/home/brieuc/gift-framework/GIFT/paper/phase3_mu_minus_half_basis.md),
- [phase3_mu_minus_half_basis.json](/home/brieuc/gift-framework/GIFT/certificates/phase3_mu_minus_half_basis.json).

So the remaining next step is no longer to choose coordinates, but to compute the actual two real coefficients of `ev_{-1/2}^{(i)}(m(h_bar))` in that fixed basis.
