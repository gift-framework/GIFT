# Phase 3.1: Banach spaces for the global maximal-section problem

## Status

This file fixes definitions and notation only. It does not prove the Fredholm theorem or the existence of a right inverse.

## Purpose

The current branch already has:

- the operator identification `J_h` as the Jacobi operator on the rank-19 normal bundle,
- the sigma-odd branched indicial picture,
- the Pacard-Mazzeo weight window,
- the reduced `R^{2N}` obstruction scaffold.

What was still missing was one canonical definition of the spaces on which all later Phase 3 statements live. This file supplies that definition.

## Global datum and local collars

Let

- `B = S^3`,
- `Sigma = Sigma_1 union ... union Sigma_N` with `N = 77`,
- `N_h -> B \ Sigma` the rank-19 normal bundle of the spacelike maximal-section background,
- `U_i` a fixed tubular collar around `Sigma_i`,
- `r0 > 0` the collar radius from the datum certificate.

On each `U_i`, choose coordinates

- `s` along `Sigma_i`,
- `w = u + i v` downstairs in the transverse base plane,
- `rho = |w|`,
- `varphi = arg(w)`,
- `z` upstairs on the branched double cover, with `z^2 = w`.

The involution is

- `sigma(z) = -z`,
- equivalently `varphi -> varphi + 2 pi` on the cover.

All spaces below are defined for sections of `N_h` which are sigma-odd in the transverse variable.

## Sigma-odd sector

A local section `xi(s, z)` belongs to the sigma-odd sector if

`xi(s, -z) = - xi(s, z)`.

Equivalently, in the angular Fourier expansion on the cover, only half-integer angular modes appear:

`e^{i m varphi}` with `m in 1/2 + Z`.

In downstairs language, the model odd modes are

- `rho^{1/2} cos(varphi/2)`,
- `rho^{1/2} sin(varphi/2)`,
- `rho^{3/2} cos(varphi/2)`,
- `rho^{3/2} sin(varphi/2)`,

and higher odd half-integer analogues.

## Edge metric and derivatives

Fix once and for all the local model edge metric

`g_edge = ds^2 + d rho^2 + rho^2 d varphi^2`

on each collar. This metric is used only to define weighted norms and Hölder seminorms. It is not the full nonlinear metric whose existence is being proved later.

Write `nabla_edge^j` for the `j`-th covariant derivative with respect to `g_edge` and any fixed background bundle connection on `N_h`. Different smooth choices give equivalent norms on the collars; all later theorems should be stated with respect to one fixed choice.

## Local weighted Holder spaces

Let `alpha in (0, 1)` and `beta in R`.

### Target space

Define the local source space `Y_{beta-2}(U_i; N_h)` to be the sigma-odd sections `f` on `U_i \ Sigma_i` such that

`||f||_{Y_{beta-2}(U_i)} := sup rho^{-(beta-2)} |f| + [rho^{-(beta-2)} f]_{C^{0,alpha}_{edge}}`

is finite.

### Domain space

Define the local domain space `X_beta(U_i; N_h)` to be the sigma-odd sections `xi` on `U_i \ Sigma_i` such that

`||xi||_{X_beta(U_i)} := sum_{j=0}^2 sup rho^{-beta + j} |nabla_edge^j xi| + [rho^{-beta + 2} nabla_edge^2 xi]_{C^{0,alpha}_{edge}}`

is finite.

These are the standard weighted `C^{2,alpha}` and `C^{0,alpha}` spaces in the edge coordinates.

## Fredholm window used in Phase 3

The operative window is

`beta in (1/2, 3/2)`.

This is the clean Pacard-Mazzeo window for the sigma-odd branch problem:

- the upper boundary corresponds to the cubic `rho^{3/2}` data,
- the lower boundary corresponds to the cokernel root `rho^{-1/2}`,
- no sigma-odd indicial root lies in the open interval.

No later theorem in Phase 3 may switch to a different window without stating it explicitly.

## Polyhomogeneous trace conventions

The spaces themselves do not imply polyhomogeneity. However, all actual solutions of the model edge equation that appear in the argument are expected to admit a polyhomogeneous expansion.

For such a solution `xi`, define:

- `ev_{3/2}^{(i)}(xi)` = the coefficient of the `rho^{3/2}` sigma-odd mode on `U_i`,
- `ev_{-1/2}^{(i)}(xi)` = the coefficient of the `rho^{-1/2}` sigma-odd mode on `U_i`.

Each of these has two real components, corresponding to

- `cos(varphi/2)`,
- `sin(varphi/2)`.

Thus `ev_{3/2}^{(i)}` and `ev_{-1/2}^{(i)}` take values in sections of a rank-2 real trace bundle over `Sigma_i`.

## The reduced constant-section layer

The current rank-one branch uses only the constant longitudinal Fourier mode on each component. Therefore define

`R^{2N} = direct sum_{i=1}^N R^2`

with one copy of `R^2` per component `Sigma_i`, corresponding to the constant coefficients of

- `rho^{3/2} cos(varphi/2)`,
- `rho^{3/2} sin(varphi/2)`.

Write an element as

`c = (c_1^R, c_1^I, ..., c_N^R, c_N^I)`.

This is a finite-dimensional datum layer, not a function space on the whole collar.

## The closed subspace `X_beta^{(0)}`

Define `X_beta^{(0)}` to be the closed subspace of `X_beta` consisting of sigma-odd sections whose `rho^{3/2}` constant-section asymptotic datum vanishes on every component:

`ev_{3/2,const}^{(i)}(xi) = 0` for all `i = 1, ..., N`.

Here `ev_{3/2,const}^{(i)}` means:

- first take the `rho^{3/2}` sigma-odd trace,
- then project to the longitudinal Fourier mode `q = 0` along `Sigma_i`.

This is the infinite-dimensional complement on which the linear edge problem is to be inverted after the finite-dimensional obstruction has been split off.

## The extended domain `X_beta^{ext}`

Define

`X_beta^{ext} = X_beta^{(0)} (+) R^{2N}`.

An element of `X_beta^{ext}` is written as

`(xi, c)` with `xi in X_beta^{(0)}` and `c in R^{2N}`.

To interpret `(xi, c)` as an actual section on the collars, fix once and for all a family of local model functions

- `psi_i^R`,
- `psi_i^I`,

supported in `U_i`, each carrying the corresponding `rho^{3/2}` odd asymptotic and zero on the other components. Then the realization map is

`iota_ext(xi, c) = xi + sum_i (c_i^R psi_i^R + c_i^I psi_i^I)`.

The later PDE theorems should be stated on `X_beta^{ext}` through this realization map.

## Norm on the extended domain

Fix the product norm

`||(xi, c)||_{X_beta^{ext}} = ||xi||_{X_beta^{(0)}} + |c|_{R^{2N}}`

where `|.|_{R^{2N}}` is the Euclidean norm.

If a later theorem uses an equivalent norm involving the realized section `iota_ext(xi, c)`, that equivalence must be stated explicitly and the constants must be tracked.

## Global spaces

Let `U = union_i U_i` and `B_bulk = B \ U`.

### Global source space

Define `Y_{beta-2}` globally to consist of sections `f` on `B \ Sigma` such that:

- on each collar `U_i`, `f` belongs to the local sigma-odd weighted space above,
- on the bulk region `B_bulk`, `f` is ordinary `C^{0,alpha}`,
- the partition-of-unity sum of these local norms is finite.

### Global domain space

Define `X_beta^{(0)}` globally to consist of sections `xi` on `B \ Sigma` such that:

- on each collar `U_i`, `xi` belongs to the local `X_beta^{(0)}(U_i)`,
- on the bulk region `B_bulk`, `xi` is ordinary `C^{2,alpha}`,
- the local pieces agree on overlaps in the ordinary sense,
- the partition-of-unity sum of norms is finite.

Then define globally

`X_beta^{ext} = X_beta^{(0)} (+) R^{2N}`

using the fixed realization map built from the local model functions.

## Transmission conditions

No extra jump condition is inserted by hand into the definition of the spaces. Transmission is encoded as follows:

- the collar and bulk representatives define one genuine section on the overlap,
- the local model functions `psi_i^R`, `psi_i^I` are cut off smoothly before the outer collar boundary,
- the DtN matching appears later as a theorem about the operator, not as a built-in space constraint.

This separation matters: spaces are definitions, DtN positivity is part of the linear analysis.

## Trace bundle and obstruction projection

Define the global constant-section trace map

`T_const : X_beta^{ext} -> R^{2N}`

by

- restricting to each collar,
- extracting the `rho^{3/2}` sigma-odd coefficient,
- projecting to the longitudinal constant mode.

By construction:

- `T_const` vanishes on `X_beta^{(0)}`,
- `T_const` is the identity on the adjoined `R^{2N}` layer.

This is the canonical projection used by the reduced obstruction problem.

## Operator domain convention

All later Phase 3 theorems should read the Jacobi operator as

`J_h : X_beta^{ext} -> Y_{beta-2}`

through the realization map `iota_ext`.

This means:

- `J_h` acts on the realized section on `B \ Sigma`,
- the finite-dimensional coordinates `c` enter only through the chosen model representatives,
- reduced matrices such as `A_red` are extracted by composing `J_h` with the trace map and the corresponding source projection.

## What is definition and what is theorem

The following are definitions fixed here:

- coordinates and sigma-odd sector,
- local weighted spaces,
- global weighted spaces,
- `X_beta^{(0)}`,
- `X_beta^{ext}`,
- product norm,
- realization map,
- constant-section trace map.

The following are not proved here:

- Fredholmness of `J_h`,
- vanishing of the kernel,
- identification of the cokernel,
- boundedness of the right inverse,
- invertibility of the reduced obstruction matrix.

Those belong to P3.2-P3.4.

## Immediate use in the next phase

With this file fixed, the next theorem can be stated without ambiguity:

- there exists a bounded projection `Pi_coker : Y_{beta-2} -> R^{2N}`,
- the reduced map on `R^{2N}` is `A_red = A_loc + E_geom + E_link`,
- the complement is inverted on `X_beta^{(0)}` by a bounded parametrix.
