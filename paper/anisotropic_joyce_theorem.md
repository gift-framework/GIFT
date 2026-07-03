# Anisotropic Joyce Theorem `(J)`

## Status

Specification only. No proof attempt in this file.

## Purpose

This file fixes the exact theorem needed in Phase 5 and prevents the current project from treating a scale audit or a citation to Joyce/FHN as if it were already the perturbation theorem.

## Input family

Let `widetilde phi_R` be a closed positive `G_2`-form on compact `K_7` obtained from the adiabatic reconstruction pipeline for `R >= R_0`, with three geometric scales:

- base scale `ell_base ~ 1`,
- fibre scale `ell_fibre ~ R^{-1}`,
- core scale `ell_core ~ R^{-2}`.

Let the manifold be covered by charts of four types:

- base charts away from the discriminant and away from the collapsing cores,
- fibre charts in smooth K3-fibred regions,
- Eguchi-Hanson core charts near each ODP smoothing,
- neck charts interpolating between fibre and core scales.

## Unknown

Find a 2-form `eta_R` such that

`phi_R = widetilde phi_R + d eta_R`

and

`d (*_{phi_R} phi_R) = 0`.

## Gauge-fixed operator

Choose a gauge condition `mathcal G_R(eta_R) = 0` and let

`L_R : mathcal X_R -> mathcal Y_R`

be the gauge-fixed linearization of the torsion map at `widetilde phi_R`.

The theorem must identify:

- the precise gauge,
- the harmonic 2-form obstruction space,
- the cohomology-class preservation statement,
- uniqueness modulo gauge / diffeomorphism,
- the treatment of collapse-induced small eigenvalues.

## Weighted spaces

The spaces must be anisotropic. A valid template is:

- `mathcal X_R`: weighted `C^{2,alpha}` or Sobolev space of 2-forms with region-dependent scaling built from `ell_base`, `ell_fibre`, `ell_core`,
- `mathcal Y_R`: weighted `C^{0,alpha}` or Sobolev space for the torsion residual,
- norms defined by a partition of unity subordinate to the three-region cover,
- transition weights compatible with the neck overlap and bounded multiplicity.

The theorem must define:

- regionwise derivatives measured in scale-normalized frames,
- regionwise weights,
- overlap constants,
- anisotropic Sobolev and Poincare constants.

## Linear estimate required

There must exist a right inverse `G_R : mathcal Y_R -> mathcal X_R` such that

`||G_R|| <= C_G`

uniformly for all `R >= R_J`.

This requires:

1. base parametrix,
2. K3-fibre parametrix,
3. Eguchi-Hanson core parametrix,
4. neck correction,
5. commutator estimate,
6. Neumann closure

with explicit bound

`||L_R G_R - I|| <= q < 1`.

## Nonlinear estimate required

For the nonlinear remainder `Q_R`, prove

`||Q_R(eta) - Q_R(xi)||_{mathcal Y_R} <= C_Q (||eta||_{mathcal X_R} + ||xi||_{mathcal X_R}) ||eta - xi||_{mathcal X_R}`

with `C_Q` independent of `R`.

## Smallness hypothesis to certify

The theorem must use the actual anisotropic residual of `widetilde phi_R`, not a bounded-geometry surrogate, and certify

`C_G C_Q ||d (*_{widetilde phi_R} widetilde phi_R)||_{mathcal Y_R} < 1/4`

for every `R >= R_J`.

## Output

For every `R >= R_J`, there exists a unique small solution `eta_R in mathcal X_R` such that

- `mathcal G_R(eta_R) = 0`,
- `phi_R = widetilde phi_R + d eta_R` is positive,
- `d phi_R = 0`,
- `d (*_{phi_R} phi_R) = 0`,
- `[phi_R] = [widetilde phi_R]`.

## Finite-dimensional obstructions that must be explicit

The theorem must not suppress the obstruction analysis. It must identify:

- harmonic 2-form directions,
- collapsing small modes,
- neck matching parameters,
- any gluing parameters,
- exact matrix whose invertibility removes the obstruction.

## Proof dependency graph

The proof of `(J)` is allowed to depend on:

- certified datum constants from `datum_D0.json`,
- Phase 3 global Jacobi theorem,
- Phase 4 adiabatic reconstruction theorem,
- explicit K3 fibre spectral control,
- explicit Eguchi-Hanson model constants.

It may not depend only on:

- natural bounded-geometry Joyce constants,
- a formal series,
- an unpublished appeal to "FHN-type refinement",
- box-local K3 residuals without whole-fibre control.

## Deliverables expected from the proof phase

- theorem statement in prose,
- operator namespace and exact spaces,
- parametrix architecture diagram,
- explicit obstruction matrix definition,
- certified `C_G`, `C_Q`, `q`, `R_J`,
- companion checker for all interval constants.
