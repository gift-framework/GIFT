# Phase 3.3: global parametrix for the Jacobi problem

## Status

Parametrix specification only. This file does not yet claim a complete proof that the Neumann error is globally below one, but it fixes the exact architecture and the constants that must enter that proof.

## Purpose

After:

- [phase3_spaces.md](phase3_spaces.md),
- [phase3_linear_theorem.md](phase3_linear_theorem.md),

the next missing step is to define the actual global parametrix

`G_glob = sum_i chi_i G_i tilde_chi_i + G_bulk`

and to isolate the exact error operator

`E_glob = J_h G_glob - I`.

This file freezes that structure.

## Geometric decomposition

Let:

- `U_i` be the branch collars around `Sigma_i`,
- `U_i'` be a slightly smaller collar contained in `U_i`,
- `B_bulk = B \ union_i U_i'`,
- `Ann_i = U_i \ U_i'` the transition annulus.

Choose smooth cutoffs:

- `tilde_chi_i` supported in `U_i`,
- `chi_i` supported in `U_i`,
- `chi_i = 1` on `supp(tilde_chi_i)`,
- `sum_i chi_i + chi_bulk = 1`.

All derivatives of the cutoffs are supported in the annuli `Ann_i`.

## Local collar inverse

For each component `Sigma_i`, let

`G_i : Y_{beta-2}(U_i) -> X_beta^{(0)}(U_i)`

be the collar inverse on the infinite-dimensional complement, as supplied by the future linear theorem.

Its role is:

- invert the sigma-odd edge equation on the collar,
- leave the constant-section obstruction to the reduced `R^{2N}` layer,
- use the local branch model and the DtN reduction.

## Bulk solver

Let

`G_bulk : Y_{beta-2}(B_bulk) -> C^{2,alpha}(B_bulk; N_h)`

be the source-free affine bulk solver with the boundary conditions induced by the collar matching problem.

At the level of architecture, `G_bulk` is not a second edge solver. It is:

- an ordinary elliptic solver on the smooth region away from the discriminant,
- coupled to the collars only through boundary traces and DtN matching,
- the place where the source-free affine background is used.

## Extended obstruction layer

The infinite-dimensional parametrix above is not enough on its own. The global parametrix must also incorporate the finite-dimensional asymptotic-data correction.

Write:

- `iota_fd : R^{2N} -> X_beta^{ext}`,
- `A_red = A_loc + E_geom + E_link`,
- `A_red^{-1}` the reduced inverse once the Neumann estimate is available.

The global parametrix therefore has two coupled pieces:

1. an infinite-dimensional collar-plus-bulk inverse on `X_beta^{(0)}`,
2. a finite-dimensional correction through `R^{2N}`.

## Canonical formula

The parametrix should be written in two stages.

### Stage 1: raw geometric parametrix

Define

`G_raw f = sum_i chi_i G_i (tilde_chi_i f) + chi_bulk G_bulk (chi_bulk f)`.

This solves the equation modulo:

- commutators `[J_h, chi_i] G_i (tilde_chi_i f)`,
- mismatch between collar and bulk traces,
- the finite-dimensional obstruction in the constant-section layer.

### Stage 2: augmented correction

Let `Pi_obs : Y_{beta-2} -> R^{2N}` be the obstruction projection.

Then the augmented parametrix is obtained by correcting `G_raw` with the reduced inverse:

`G_glob f = G_raw f + iota_fd A_red^{-1} Pi_obs (f - J_h G_raw f)`.

This is the first formula that should later be proved to be a true right inverse.

## Error decomposition

The error operator

`E_glob = J_h G_glob - I`

must be decomposed as a sum of four explicit terms:

1. `E_comm`
   Cutoff commutators from `[J_h, chi_i]` and `[J_h, chi_bulk]`.

2. `E_geom`
   Geometric perturbation from the Frenet-Serret deviation and the variation of `A_bulk`.

3. `E_link`
   Inter-component coupling via the Newton kernel.

4. `E_DtN`
   Collar/bulk mismatch removed by the DtN interface and reduced correction.

The proof task is to show that these four terms combine into a strict contraction in the operator norm relevant to `Y_{beta-2}`.

## Quantities already available at `D0`

The current branch already has:

- `||E_geom|| <= 0.06282151815625661`,
- `||E_link|| <= 8.93933327491038e-06`,
- `delta_total <= 0.07256917840040894` conservatively for the reduced matrix,
- `Lambda_{q,1/2} >= 50`,
- matching coercivity `T_{q,1/2} >= 100`,
- reduced inverse bound `||(A_red)^{-1}|| <= 1.2453759063214098` conservatively.

These do not yet prove the full parametrix theorem, but they are already the correct constants for the reduced sector.

## What still has to be proved for the parametrix theorem

### P3.3.a Commutator estimate

One must bound

`[J_h, chi_i] G_i (tilde_chi_i f)`

in `Y_{beta-2}`.

Because the derivatives of the cutoffs live in the annuli `Ann_i`, the estimate should be ordinary elliptic there, not edge-singular.

### P3.3.b Bulk transmission estimate

One must show that the source-free affine bulk solver and the collar solutions match with only one Schauder loss, not a squared one.

### P3.3.c Reduced projection compatibility

One must prove that `Pi_obs (J_h G_raw f)` extracts exactly the residual constant-section obstruction and no hidden Fourier mode.

### P3.3.d Neumann closure

Finally one must prove

`||E_glob|| <= q < 1`

for a concrete `q`.

This is the exact point where the global theorem stops being architecture and becomes a proof.

## Target theorem statement

The Phase 3.3 theorem should eventually read:

There exists a bounded linear operator

`G_glob : Y_{beta-2} -> X_beta^{ext}`

such that

- `||J_h G_glob - I|| <= q < 1`,
- `||G_glob|| <= C_param`,
- both constants depend only on the frozen datum constants and the chosen cutoffs.

Then the actual right inverse is recovered by Neumann series:

`G_true = G_glob (I + (I - J_h G_glob) + (I - J_h G_glob)^2 + ...)`.

## Relation to the current numerical certificates

The present axis2 certificates mostly control the reduced sector:

- `axis2_DtN_interface_2026_06_30`
  gives positivity of the inner DtN map;
- `axis2_DtN_matching_2026_07_01`
  gives matching coercivity and kills hidden outer kernel modes;
- `axis2_E_geom_interval_2026_06_30`
  gives the geometric perturbation bound;
- `axis2_E_link_multipole_2026_06_30`
  gives the dipole `1/d^2` coupling bound;
- `axis2_certificate_sigma_min_NK_2026_06_30`
  assembles these into the reduced Neumann certificate.

The missing theorem-level work is to wrap these reduced facts into the full operator `G_glob`.

## Interaction with the next phase

Once this parametrix theorem is fixed, Phase 3.4 can state the nonlinear closure as:

- build `h_0 = h_bar_global + G_true(nonlin source)`,
- solve by contraction in `X_beta^{ext}`,
- obtain the actual maximal background.
