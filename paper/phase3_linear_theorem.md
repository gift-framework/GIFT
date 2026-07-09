# Phase 3.2: linear theorem for the global Jacobi operator

## Status

Theorem specification only. This file does not claim that the Fredholm theorem or the right inverse has already been proved in full detail.

## Purpose

This file fixes the exact linear theorem needed after [phase3_spaces.md](phase3_spaces.md). It isolates:

- the operator,
- the Fredholm window,
- the trace and obstruction maps,
- the reduced finite-dimensional matrix,
- the explicit constants already available from the current axis2 certificates.

## Operator and spaces

Fix:

- `alpha in (0,1)`,
- `beta in (1/2, 3/2)`,
- the global spaces `X_beta^{(0)}`, `X_beta^{ext}`, `Y_{beta-2}` from [phase3_spaces.md](phase3_spaces.md).

Let

`J_h : X_beta^{ext} -> Y_{beta-2}`

be the Jacobi operator of the maximal-section background, acting through the realization map of the extended domain.

## Hypotheses that belong to the linear theorem

The theorem is allowed to assume only datum-level hypotheses, not later nonlinear closure:

1. `A1`
   Rank-one monodromy, so the leading branched direction is the distinguished line `R alpha_1`.

2. `A2`
   Collar-affine local model on each discriminant component.

3. `A4`
   Uniform local collar geometry, including the `kappa_g r0` perturbative regime.

4. `A5`
   Positive-definite bulk metric `A_bulk` on the rank-19 normal bundle, with controlled condition number.

5. `A6`
   Split-component regime with coupling controlled by `r0 / d_min`.

6. `A7-A8`
   Geometric and analytic scales needed for the branch tubes.

Nothing in this theorem may depend on adiabatic reconstruction or on the anisotropic Joyce theorem `(J)`.

## Model operator statement

On each collar `U_i`, the leading operator is

`J_h^model = A_bulk (partial_s^2 + Delta_w)`

with sigma-odd indicial roots

`mu in { +/- 1/2, +/- 3/2, +/- 5/2, ... }`.

The lower relevant root for the obstruction theory is `mu = -1/2`, and the upper cubic-data root is `mu = 3/2`.

## Theorem L1: boundedness and Fredholm package

The first theorem to prove should read:

For `beta in (1/2, 3/2)`, the operator

`J_h : X_beta^{ext} -> Y_{beta-2}`

is bounded and Fredholm. Moreover:

1. the infinite-dimensional complement `X_beta^{(0)}` carries the ordinary edge inversion problem;
2. the only finite-dimensional extension needed in this window is the constant-section layer `R^{2N}`;
3. the obstruction is read by a bounded source projection

   `Pi_obs : Y_{beta-2} -> R^{2N}`;

4. the corresponding reduced operator

   `A_red = Pi_obs o J_h o iota_ext |_{R^{2N}}`

   is of the form

   `A_red = A_loc + E_geom + E_link`.

This is the linear heart of Phase 3.

## Theorem L2: explicit reduced matrix

The reduced matrix must be stated exactly as:

- `A_loc = 2 A_bulk(alpha_1, alpha_1) I_{2N}`,
- `||E_geom|| <= C_g ||kappa_g|| r0`,
- `||E_link|| <= C_link (N-1) (r0 / d_min)^2`.

### Current numerical scaffold at `D0`

The branch already has the following conservative values:

- `sigma_min(A_loc) >= 0.8658008658008657`,
- `||E_geom|| <= 0.06282151815625661`,
- `||E_link|| <= 8.93933327491038e-06`,
- `delta_total <= 0.07256917840040894`,
- `||(A_red)^{-1}|| <= 1.2453759063214098`.

These come from:

- [axis2_certificate_sigma_min_NK_2026_06_30.json](../../private/canonical/results/axis2_certificate_sigma_min_NK_2026_06_30.json),
- [axis2_DtN_interface_2026_06_30.json](../../private/canonical/results/axis2_DtN_interface_2026_06_30.json),
- [axis2_E_link_interval_2026_06_30.json](../../private/canonical/results/axis2_E_link_interval_2026_06_30.json),
- [axis2_E_geom_interval_2026_06_30.json](../../private/canonical/results/axis2_E_geom_interval_2026_06_30.json).

## Theorem L3: DtN positivity and non-constant mode absorption

The theorem also needs the mode-separation statement:

- the constant longitudinal mode contributes to the reduced `R^{2N}` obstruction,
- every non-constant longitudinal Fourier mode is absorbed into the infinite-dimensional complement via positive DtN matching.

The precise quantitative statement should be:

For every longitudinal Fourier mode `q`,

`Lambda_{q,1/2} >= 1 / (2 r0)`.

At `D0`, this gives

`Lambda_{q,1/2} >= 50`.

This is what justifies reducing the obstruction from the full trace bundle to the constant-section layer `R^{2N}`.

## Theorem L4: explicit right inverse

After the reduced matrix is inverted, the theorem should provide a bounded right inverse

`G_aug : Y_{beta-2} -> X_beta^{ext}`

with an explicit bound

`||G_aug|| <= C_aug`.

### Current candidate bound

The current branch uses

`||G_aug|| <= 36.635`

as the working bound in the `G`-quant discharge.

This should be promoted into a theorem-grade constant only after the proof explicitly combines:

1. the edge Schauder bound on `X_beta^{(0)}`,
2. the DtN coercivity,
3. the inverse of `A_red`,
4. the collar-bulk commutator control.

## Dependencies between the linear sub-results

The proof graph should be read in this order:

1. normal operator and indicial spectrum,
2. definition of trace and extended domain,
3. DtN positivity on the inner model,
4. reduction of the obstruction to the constant-section mode,
5. reduced matrix decomposition `A_loc + E_geom + E_link`,
6. Neumann inversion of `A_red`,
7. assembly of the augmented right inverse `G_aug`.

## What the current branch has already settled

The following are already effectively fixed:

- the operator is `J_h`, not the fibrewise Donaldson operator;
- the relevant window is `beta in (1/2,3/2)`;
- the obstruction coefficient is `2 A_bulk(alpha_1, alpha_1)`, not the old cover-Euclidean artifact `8`;
- the reduced coupling is small at `D0`;
- the DtN positivity is explicit and strong at `D0`.

## What is still missing at theorem level

The following still need to be written as actual proofs rather than certificate summaries:

1. a full boundedness proof of `J_h` on the global spaces;
2. the exact construction of `Pi_obs`;
3. the proof that the non-constant longitudinal modes lie entirely in the invertible complement;
4. the proof that the extended domain adds exactly the needed `R^{2N}` layer and no more;
5. the derivation of the global bound for `G_aug`.

## Canonical theorem statement to target next

The next write-up step should produce a formal theorem statement of the following shape:

There exist bounded linear maps

- `Pi_obs : Y_{beta-2} -> R^{2N}`,
- `G_0 : Y_{beta-2} -> X_beta^{(0)}`,
- `G_aug : Y_{beta-2} -> X_beta^{ext}`,

such that

- `A_red = Pi_obs o J_h o iota_ext |_{R^{2N}}` is invertible,
- `J_h o G_aug = Id` on `Y_{beta-2}`,
- `||G_aug|| <= C_aug(D0)`,
- `C_aug(D0)` is expressed explicitly in terms of the frozen constants from [datum_D0.json](../certificates/datum_D0.json).

## Interaction with later phases

Phase 3.3 will use this theorem to build the actual global parametrix.

Phase 3.4 will use it to close the nonlinear maximal-section problem.

Phase 4 will then plug the resulting maximal background into the adiabatic reconstruction map.
