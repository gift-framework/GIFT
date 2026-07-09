# Phase 3.S1.m: operational constant-mode model for `Pi_obs`

## Status

Operational prototype only.

This file does not claim that the full PDE obstruction projection

`Pi_obs : Y_{beta-2} -> R^{2N}`

has already been constructed. It isolates the minimal constant-section model
that the current branch can use honestly.

## Purpose

After [phase3_source_to_obstruction_bridge.md](phase3_source_to_obstruction_bridge.md),
the next missing piece is not a new asymptotic basis. It is an operational rule
for how a source in the lower-root channel contributes to the finite-dimensional
layer `R^{2N}`.

The present file fixes that rule in the simplest admissible regime:

- one collar at a time,
- rank-one `alpha_1` direction,
- longitudinal constant mode only.

## Provisional model class

Fix one collar `U_i`, with longitudinal parameter `s` on `Sigma_i`.

Consider a source `f` whose lower-root obstruction part is represented as

`f_obs^(i)(s, rho, varphi)`
`=`
`rho^(-1/2) (a_i(s) cos(varphi/2) + b_i(s) sin(varphi/2))`
`+`
`higher modes / higher roots`.

Here:

- `a_i(s)` and `b_i(s)` are real longitudinal coefficient functions,
- `higher modes` means non-constant Fourier modes in `s`,
- `higher roots` means terms whose radial order is strictly above `mu = -1/2`.

This is the first regime in which the current branch can write an honest
candidate formula for the obstruction projection.

## Constant-mode projection

Let `Pi_const^{(i)}` denote longitudinal averaging on the component `Sigma_i`:

`Pi_const^{(i)} a_i = (1 / |Sigma_i|) integral_{Sigma_i} a_i(s) ds`,

and similarly for `b_i`.

Then define the operational constant-mode obstruction map

`Pi_obs,proto^{(i)}(f)`
`:=`
`((1 / |Sigma_i|) integral a_i(s) ds) e_i^R`
`+`
`((1 / |Sigma_i|) integral b_i(s) ds) e_i^I`.

Globally:

`Pi_obs,proto(f) = sum_i Pi_obs,proto^{(i)}(f)`.

This is the first concrete model for the map

`Y_{beta-2} -> R^{2N}`.

## Why this is the right provisional rule

This rule matches the existing Phase 3 architecture:

1. [phase3_mu_minus_half_basis.md](phase3_mu_minus_half_basis.md)
   fixes the real lower-root basis on each collar.

2. [phase3_spaces.md](phase3_spaces.md)
   defines `R^{2N}` as the constant longitudinal layer.

3. [phase3_linear_theorem.md](phase3_linear_theorem.md)
   already states that non-constant longitudinal modes are absorbed by DtN
   positivity, leaving only the constant-section obstruction.

So before the full theorem is written, the only honest prototype is:

- extract the lower-root coefficient functions,
- average them in `s`,
- represent the result in the fixed `R^{2N}` basis.

## What this prototype does not claim

This file does not claim that, for every source in `Y_{beta-2}`,

`Pi_obs = Pi_obs,proto`.

That equality still needs theorem-level justification from:

- the true definition of the cokernel/source projection,
- the DtN elimination of all non-constant longitudinal modes,
- the identification of the model lower-root trace with the obstruction
  coordinates used by the reduced matrix.

So the correct safe statement is:

`Pi_obs,proto`

is the operational constant-mode model that the branch should use whenever the
source has already been reduced to its lower-root trace representation.

## Consequence for `kappa_src`

Under the prototype rule, the source coefficient pair on `U_i` becomes

`kappa_src,R^(i) = (1 / |Sigma_i|) integral a_i(s) ds`,

`kappa_src,I^(i) = (1 / |Sigma_i|) integral b_i(s) ds`,

provided the source has already been put in the form

`rho^(-1/2) (a_i(s) cos(varphi/2) + b_i(s) sin(varphi/2)) + ...`.

This does not remove the need for the earlier bridge:

`profile_src -> source in Y_{beta-2} -> lower-root obstruction part`.

It only fixes the last step:

`lower-root obstruction part -> R^{2N}`.

## Immediate next use

The next practical computation should therefore follow this chain:

1. derive a symbolic or certified model for the lower-root coefficient
   functions `a_i(s), b_i(s)`,
2. apply constant-mode averaging,
3. feed the resulting pair into
   [phase3_kappa_src_projection.md](phase3_kappa_src_projection.md).

That is the first route from the local source term to a serializable candidate
for `kappa_src`.
