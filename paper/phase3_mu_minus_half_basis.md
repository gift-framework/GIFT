# Phase 3.S1.e: standard basis for the `mu = -1/2` obstruction channel

## Status

Definition fixed.

This note fixes the standard real basis of the lower-root `sigma`-odd trace bundle and aligns it with the `R^{2N}` coordinates already used in Phase 3.

The machine-readable freeze is:

- [phase3_mu_minus_half_basis.json](/home/brieuc/gift-framework/GIFT/certificates/phase3_mu_minus_half_basis.json).

## Per-component basis

On each collar `U_i`, define:

- `Phi_{-1/2,R}^{(i)} = rho^(-1/2) cos(varphi/2)` in the `alpha_1` direction,
- `Phi_{-1/2,I}^{(i)} = rho^(-1/2) sin(varphi/2)` in the `alpha_1` direction.

These span the real lower-root trace space

`T_{-1/2}|_{Sigma_i}`.

## Relation to the upper-root data

The extended-domain layer already uses the upper-root basis

- `psi_i^R = rho^(3/2) cos(varphi/2)`,
- `psi_i^I = rho^(3/2) sin(varphi/2)`.

The model identity is:

- `Delta_w psi_i^R = 2 Phi_{-1/2,R}^{(i)}`,
- `Delta_w psi_i^I = 2 Phi_{-1/2,I}^{(i)}`.

So in this normalization:

- one unit in the upper-root coordinate `e_i^R` produces two units in the lower-root basis `Phi_{-1/2,R}^{(i)}`,
- and similarly for the `I` coordinate.

This is exactly the normalization behind the diagonal block

`A_loc = 2 A_bulk(alpha_1, alpha_1) I`.

## Serialization rule

From now on, the scalar `kappa_src^(i)` from the matching note must always be recorded in the lower-root real basis

`{Phi_{-1/2,R}^{(i)}, Phi_{-1/2,I}^{(i)}}`.

If an intermediate computation uses complex coefficients in

`e^{± i varphi/2}`,

they must be converted to the real cosine/sine basis before being stored.

## Why this closes the ambiguity

Before this note, the phrase

`ev_{-1/2}^{(i)}(m(h_bar))`

still left a basis ambiguity.

After this note, the remaining extraction target is concrete:

- compute the two real coefficients of `ev_{-1/2}^{(i)}(m(h_bar))` in
  `Phi_{-1/2,R}^{(i)}, Phi_{-1/2,I}^{(i)}`,
- then exploit `s`-uniformity to reduce them to the constant-section `R^{2N}` layer.
