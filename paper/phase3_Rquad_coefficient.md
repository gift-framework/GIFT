# Phase 3.S1.a: explicit coefficient scaffold for the collar quadratic source

## Status

Symbolic coefficient extracted; nonlinear tensor coefficient still open.

This note records the first explicit numerical piece behind `R_quad`: the branch seed

`w^(3/2)`

has second derivative

`(3/4) w^(-1/2)`,

so its weighted `Y_-3` bookkeeping contribution is exactly of order `r0^(7/2)`.

The machine-readable freeze is:

- [phase3_Rquad_coefficient_scaffold.json](/home/brieuc/gift-framework/GIFT/certificates/phase3_Rquad_coefficient_scaffold.json).

## What is now explicit

From the local model script:

- `d/dw (w^(3/2)) = (3/2) w^(1/2)`,
- `d^2/dw^2 (w^(3/2)) = (3/4) w^(-1/2)`.

Therefore a quadratic source built from one bounded factor and one curvature-type factor inherits the model size

`(3/4) rho^(1/2)`,

and after multiplying by the `Y_-3` weight `rho^3` one gets

`(3/4) rho^(7/2)`.

At `r0 = 10^-2`, this model contribution is

- pointwise: `7.5 * 10^-2`,
- weighted: `7.5 * 10^-8`.

## Why this matters

This is the first place where `C_src` stops being a pure `O(1)` slogan and starts carrying a visible coefficient.

It does not close `R_quad` yet, because the full quadratic source still includes:

- the actual tensor contractions in the maximal operator,
- the `A_bulk(alpha_1, alpha_1)` factor,
- and the passage from the model scalar branch to the full rank-19 normal-bundle geometry.

But it does certify that the `r0^(7/2)` exponent is not an artifact of loose dimensional analysis: it comes directly from the explicit `3/4` curvature coefficient of the `w^(3/2)` seed.

## Remaining gap for `R_quad`

To promote `R_quad` itself to theorem grade, the next step is still:

- compute the full leading coefficient of the nonlinear source term in the `m = +/- 1/2` channel,
- isolate the `A_bulk(alpha_1, alpha_1)` dependence explicitly,
- bound the `O(rho^(3/2))` tail on `rho <= r0`.
