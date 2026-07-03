# Theorem E: compact torsion-free `G_2` structure on `K_7`

## Status

Current status: target theorem only. Not yet discharged.

## Statement

Let `D0 = (S^3, pi: K_7 -> S^3, Sigma_77, h, Gamma)` be the explicit rank-one Kovalev-Lefschetz datum with fixed 77-component unlink discriminant and fixed rank-one Picard-Lefschetz monodromy.

The first plausible exact theorem is:

For some explicit `R_* < infinity`, for every `R >= R_*`, there exists a positive `G_2`-form `phi_R` on the compact manifold `K_7` such that

- `d phi_R = 0`,
- `d (*_{phi_R} phi_R) = 0`,
- `[phi_R] = [phi_0] + R [B]`,
- `pi_1(K_7) = 1`,
- therefore `Hol(g_{phi_R}) = G_2`.

## Dependencies

This theorem depends on named artifacts only:

1. `datum_D0_theorem`
   Smooth compact realization of `K_7 -> S^3` with fixed discriminant and explicit constants.

2. `global_maximal_section_theorem`
   A true nonlinear theorem solving `m(h_0) = 0` on `S^3 \ Sigma`.

3. `adiabatic_reconstruction_theorem`
   A convergent theorem for the Donaldson equations `(E1)-(E5)`.

4. `closed_small_torsion_family_theorem`
   Construction of closed positive `widetilde phi_R` with explicit anisotropic small torsion.

5. `anisotropic_joyce_theorem_J`
   Three-scale perturbation theorem producing `eta_R` with `phi_R = widetilde phi_R + d eta_R`.

6. `full_holonomy_wrapper`
   Converts compactness, positivity, torsion-free, and `pi_1(K_7) = 1` into `Hol = G_2`.

## Non-allowed shortcuts

The theorem may not use any of the following unnamed shortcuts:

- "small enough"
- "standard weighted estimate"
- "`O(1)`"
- "Joyce applies"
- "closed-form correction"

Each must be replaced by either an explicit constant, a named theorem, or an isolated hypothesis.
