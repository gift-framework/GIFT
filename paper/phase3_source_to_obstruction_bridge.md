# Phase 3.S1.l: the correct bridge is `profile_src -> Pi_obs(profile_src)`

## Status

Canonical analytical bridge fixed.

This note records the corrected viewpoint for the local source term:

the object that enters the finite-dimensional obstruction theory is not the raw
profile itself, and not even a naive literal trace taken from that profile, but
the bounded source projection

`Pi_obs : Y_{beta-2} -> R^{2N}`.

## Why this fixes the earlier confusion

The branch had been oscillating between two pictures:

- a raw source profile written at radial order `rho^(1/2)`,
- a lower-root obstruction space written in the basis
  `rho^(-1/2) cos(varphi/2), rho^(-1/2) sin(varphi/2)`.

Those are not the same object.

The correct bridge is:

`profile_src`
`->`
`source element in Y_{beta-2}`
`->`
`Pi_obs(profile_src) in R^{2N}`.

In other words, the lower-root basis is a representation of the *obstruction
projection*, not a literal replacement of the raw pointwise source profile.

## Repository consequence

From now on, the symbol

`kappa_src`

should be understood as the coordinate of

`Pi_obs(m(h_bar))`

in the fixed `R^{2N}` basis, not as the coefficient of the raw source profile
itself.

This is the analytically correct quantity for Phase 3.

## How this fits the existing artifacts

The current chain is now:

1. [phase3_source_trace_resolution.md](/home/brieuc/gift-framework/GIFT/paper/phase3_source_trace_resolution.md)
   separates raw profile, trace coefficient, and weighted quantity.

2. [phase3_mu_minus_half_basis.md](/home/brieuc/gift-framework/GIFT/paper/phase3_mu_minus_half_basis.md)
   fixes the lower-root basis used to represent the obstruction layer.

3. [phase3_kappa_src_projection.md](/home/brieuc/gift-framework/GIFT/paper/phase3_kappa_src_projection.md)
   gives the algebraic map from an obstruction coefficient to the cubic
   correcting datum.

What was missing was the statement that the relevant local coefficient is really
the result of `Pi_obs`, not a naive raw-profile coefficient.

## Revised extraction target

The next coefficient extraction should therefore target:

`Pi_obs(m(h_loc,aff + h_loc,branch))`

componentwise on each collar, then read its coordinates in the fixed
`R^{2N}` basis.

This is slightly different from saying

`compute ev_{-1/2}(m(h_bar))`

unless one has already proved that `Pi_obs` is literally given by that trace in
the constant-section mode with the chosen normalization.

## Safe practical interpretation

Until a theorem identifies `Pi_obs` with an explicit trace extraction formula,
the branch should use the following language:

- `profile_src` for the pointwise local leading term,
- `ev_{-1/2}` for the lower-root trace representation when appropriate,
- `Pi_obs(profile_src)` or `Pi_obs(m(h_bar))` for the actual obstruction datum.

This is the least misleading formulation currently available.
