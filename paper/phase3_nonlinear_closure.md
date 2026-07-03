# Phase 3.4: nonlinear closure for the global maximal section

## Status

Closure theorem specification only. This file does not claim that the nonlinear fixed-point argument has already been discharged; it freezes the exact form that argument must take once the linear package of Phase 3.2 and the parametrix package of Phase 3.3 are in theorem form.

2026-07-02 update from the sibling `private` repo: the `D0` source and
nonlinear constants have been sharpened to `C_src = 27/16` and `C_nl = 2/3`,
giving `4 C_lin^2 C_nl C_src_actual ~= 6.03e-4` with `C_lin <= 36.6` and
margin approximately `3318x` below the `1/2` threshold. This improves the
scalar closure inequality, but it does not replace the missing global linear
right-inverse theorem or the full weighted mapping theorem.

## Purpose

After:

- [phase3_spaces.md](/home/brieuc/gift-framework/GIFT/paper/phase3_spaces.md),
- [phase3_linear_theorem.md](/home/brieuc/gift-framework/GIFT/paper/phase3_linear_theorem.md),
- [phase3_global_parametrix.md](/home/brieuc/gift-framework/GIFT/paper/phase3_global_parametrix.md),

the remaining task in Phase 3 is to solve the actual maximal-section equation

`m(h_0) = 0`

on `S^3 \ Sigma`, for the rank-one 77-unlink datum, by a contraction argument in the extended space `X_beta^{ext}`.

This file fixes the nonlinear map, the quadratic estimate, and the exact constant inequalities that must be checked.

## Background splitting

Let `h_bar_global` be the explicit collar-plus-bulk approximate maximal section assembled from the source-free affine bulk background and the branch-collar models.

Write the true section as

`h = h_bar_global + u`

with

`u in X_beta^{ext}`.

The nonlinear maximal-section equation becomes

`m(h_bar_global + u) = 0`.

## Taylor expansion and source term

Expand at `h_bar_global`:

`m(h_bar_global + u) = m(h_bar_global) + J_h u + N_h(u)`,

where:

- `J_h` is the Jacobi operator already fixed in [phase3_linear_theorem.md](/home/brieuc/gift-framework/GIFT/paper/phase3_linear_theorem.md),
- `N_h(u)` is the nonlinear remainder,
- `N_h(0) = 0`,
- `D N_h(0) = 0`.

The Phase 3 closure problem is therefore:

`J_h u = - m(h_bar_global) - N_h(u)`.

## Right inverse input

Assume the theorem-level output of Phase 3.2 and Phase 3.3:

- a bounded right inverse `G_true : Y_{beta-2} -> X_beta^{ext}`,
- an explicit norm bound `||G_true|| <= C_lin`,
- the residual estimate `||m(h_bar_global)||_{Y_{beta-2}} <= C_src`.

Then the equation is rewritten as the fixed-point problem

`u = T(u) := - G_true m(h_bar_global) - G_true N_h(u)`.

## Nonlinear estimate to prove

The nonlinear remainder must satisfy a theorem-grade quadratic estimate on a ball in `X_beta^{ext}`:

`||N_h(u) - N_h(v)||_{Y_{beta-2}} <= C_nl (||u||_{X_beta^{ext}} + ||v||_{X_beta^{ext}}) ||u - v||_{X_beta^{ext}}`.

In particular,

`||N_h(u)||_{Y_{beta-2}} <= C_nl ||u||_{X_beta^{ext}}^2`

for `u` in the chosen small ball around the origin.

This estimate must be derived from:

- the `C^2` control of the maximal-section operator,
- the datum constant `D2m <= 1`,
- ordinary product estimates in the annuli,
- weighted edge product estimates in the collars,
- boundedness of the realization and trace maps defining `X_beta^{ext}`.

## Radius choice

Choose a radius `rho > 0` and let

`B_rho = { u in X_beta^{ext} : ||u|| <= rho }`.

The contraction proof must verify two inequalities:

1. self-mapping:

   `C_lin C_src + C_lin C_nl rho^2 <= rho`;

2. contraction:

   `2 C_lin C_nl rho < 1`.

The standard convenient target is

- `rho = 2 C_lin C_src`,
- `4 C_lin^2 C_nl C_src <= 1/2`,

which implies both conditions.

## Canonical theorem shape

The target theorem for Phase 3.4 should read:

Fix `beta in (1/2, 3/2)`. Assume:

1. the datum constants from [datum_D0.json](/home/brieuc/gift-framework/GIFT/certificates/datum_D0.json),
2. the linear right-inverse theorem of [phase3_linear_theorem.md](/home/brieuc/gift-framework/GIFT/paper/phase3_linear_theorem.md),
3. the global parametrix theorem of [phase3_global_parametrix.md](/home/brieuc/gift-framework/GIFT/paper/phase3_global_parametrix.md),
4. the quadratic estimate above with explicit constant `C_nl`,
5. the source bound `||m(h_bar_global)|| <= C_src`.

If

`4 C_lin^2 C_nl C_src <= 1/2`,

then there exists a unique

`u_* in B_{2 C_lin C_src} subset X_beta^{ext}`

such that

`m(h_bar_global + u_*) = 0`.

Setting

`h_0 = h_bar_global + u_*`,

one obtains the actual global maximal section.

## Quantities that must become explicit

This theorem is not allowed to use the phrases:

- "for the source sufficiently small",
- "for the nonlinear term sufficiently tame",
- "by a standard contraction theorem".

Instead it must expose the exact constants:

- `C_lin = ||G_true||`,
- `C_src = ||m(h_bar_global)||_{Y_{beta-2}}`,
- `C_nl`,
- `rho = 2 C_lin C_src`.

## Scope of uniqueness

The uniqueness statement must be local to the chosen Banach ball:

`u_*` is unique in `B_rho`.

This phase does not yet claim a moduli-space classification outside that ball, and it does not upgrade local uniqueness to any global rigidity statement.

## Interaction with the finite-dimensional obstruction layer

The fixed-point problem takes place in `X_beta^{ext}`, not in `X_beta^{(0)}`.

This matters for two reasons:

1. the constant-section obstruction has already been absorbed by the reduced finite-dimensional correction in Phase 3.2 and Phase 3.3;
2. the nonlinear term must be shown not to regenerate an uncontrolled mode outside the already chosen `R^{2N}` extension.

At theorem level this should appear as:

- `T(B_rho) subset X_beta^{ext}`,
- `Pi_obs (m(h_bar_global + u_*)) = 0`,
- no extra asymptotic layer is created by the nonlinear iteration.

## What is already available in the current branch

The following inputs are already frozen numerically or structurally:

- the datum bound `D2m <= 1`,
- the sharp-current private bound `C_nl = 2/3` at `D0`,
- the sharp-current private source coefficient `C_src = 27/16` at `D0`,
- the working linear candidate `C_lin <= 36.635`,
- the Banach spaces and extension layer,
- the reduced obstruction architecture.

These are not yet enough to close Phase 3, because the global right inverse,
parametrix assembly, projection theorem, commutator control, and weighted
mapping theorem still have to be promoted from scaffold to theorem artifacts.

## What remains open after this specification

To turn this file into a theorem, the branch still needs:

1. public integration of the private source bound for `m(h_bar_global)`,
2. a theorem-grade weighted quadratic estimate for `N_h` in the full mapping setting,
3. a complete proof that `T` preserves the extended domain,
4. a final check of the scalar inequality `4 C_lin^2 C_nl C_src <= 1/2`.

## Output of Phase 3

Once the above is discharged, the Phase 3 conclusion should be recorded as:

There exists a global sigma-odd maximal section

`h_0 : S^3 \ Sigma -> M_K3`

with:

- `m(h_0) = 0`,
- `||h_0 - h_bar_global||_{X_beta^{ext}} <= 2 C_lin C_src`,
- all constants depending only on the certified datum and the explicit linear and nonlinear estimates.

This is the exact endpoint needed before entering the adiabatic reconstruction phase.
