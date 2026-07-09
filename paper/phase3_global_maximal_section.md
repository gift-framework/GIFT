# Phase 3: global maximal-section theorem

## Status

Specification and theorem scaffold only. The numerical ingredients exist, but the global theorem is not yet discharged.

## Objective

Promote the present axis2 Jacobi / DtN / Neumann scaffolds into a genuine theorem solving

`m(h_0) = 0`

on `S^3 \ Sigma` for the actual rank-one 77-unlink datum.

## Certified datum inputs used

This phase may use only the currently frozen datum certificate fields:

- `r0 = 10^-2`,
- `d_min = 1`,
- `kappa_g <= 1`,
- `cond(A_bulk) <= 2.31`,
- `A_bulk(alpha1,alpha1) >= 0.43290043290043284`,
- `D2m <= 1`.

See [datum_D0.json](../certificates/datum_D0.json).

## Operator identity already fixed

The edge operator belongs to the Jacobi operator of the maximal section, not to the fibrewise Donaldson operator:

- base operator: `J_h : Gamma(N_h) -> Gamma(N_h)`,
- rank of `N_h`: `19`,
- branch-collar indicial structure: half-integer modes, with the relevant lower root at downstairs weight `-1/2`.

Current source: `axis2_jacobi_maximal_section_2026_06_25`.

## P3.1 Banach spaces to define explicitly

The canonical definitions are now frozen in [phase3_spaces.md](phase3_spaces.md). That file should be treated as the source of notation for all later Phase 3 statements.

The theorem must define the following spaces without relying on shorthand:

- downstairs base coordinate `w = u + i v`,
- branched cover coordinate `z` with `z^2 = w`,
- sigma-odd sector,
- weighted Hölder spaces `X_beta^{(0)}` and `Y_{beta-2}`,
- extended domain `X_beta^{ext}`,
- asymptotic trace bundle at the lower root,
- constant-section layer `R^{2N}`,
- transmission conditions across collar / bulk interfaces.

### Proposed window

Use the Pacard-Mazzeo window

`beta in (1/2, 3/2)`

with:

- `X_beta^{(0)}` = sigma-odd weighted `C^{2,alpha}` sections whose `mu = 3/2` trace is killed,
- `X_beta^{ext} = X_beta^{(0)} (+) R^{2N}`,
- `Y_{beta-2}` = sigma-odd weighted `C^{0,alpha}` target space.

## P3.2 Linear theorem to prove

The canonical statement is now frozen in [phase3_linear_theorem.md](phase3_linear_theorem.md).

For the actual Jacobi operator

`J_h : X_beta^{ext} -> Y_{beta-2}`

prove:

1. normal operator computation,
2. indicial spectrum,
3. kernel and cokernel description in the chosen window,
4. uniform collar model across all 77 components,
5. coefficient perturbation estimate from the smooth base geometry,
6. explicit right-inverse norm.

## P3.3 Global parametrix architecture

The canonical architecture is now frozen in [phase3_global_parametrix.md](phase3_global_parametrix.md).

The proof should build

`G_glob = sum_i chi_i G_i \tilde chi_i + G_bulk`

with:

- `G_i` the collar inverse on each branch tube,
- `G_bulk` a source-free affine bulk solver,
- DtN matching on the tube boundaries,
- commutator control,
- Neumann closure.

## P3.4 Nonlinear closure

The fixed-point closure is now frozen in [phase3_nonlinear_closure.md](phase3_nonlinear_closure.md).

This is the first place where the Phase 3 chain becomes an actual PDE solution rather than a parametrix architecture. The theorem must solve

`m(h_bar_global + u) = 0`

for `u in X_beta^{ext}` by combining:

- the right inverse `G_true`,
- the source bound for `m(h_bar_global)`,
- the weighted quadratic estimate for the nonlinear remainder `N_h`,
- and one explicit scalar inequality guaranteeing contraction.

## P3.5 Source and nonlinear constants

The current status of `C_src` and `C_nl` is now frozen in:

- [phase3_source_nonlinear_constants.md](phase3_source_nonlinear_constants.md),
- [phase3_G_closure_status.json](../certificates/phase3_G_closure_status.json).

These artifacts separate:

- what is already numerically tiny at `D0`,
- what is theorem-usable conditionally,
- what remains an analytic placeholder still requiring an explicit proof.

## Explicit numerical scaffold already available

The current branch has the following partial constants:

- local block:
  `sigma_min(A_loc) >= 0.8658008658008657` conservatively,
  from `A_loc = 2 A_bulk(alpha1,alpha1) I_{2N}`;
- geometric perturbation:
  `||E_geom|| <= 0.06282151815625661`;
- link coupling:
  `||E_link|| <= 8.93933327491038e-06`;
- total reduced perturbation:
  `delta_total <= 0.07256917840040894` conservatively;
- DtN positivity:
  `c_DtN >= 50`;
- augmented inverse scaffold:
  `||J_ext^{-1}|| <= 36.635`.

These are not yet the theorem, but they are the right numerical targets for the proof.

## P3.4 Cokernel reduction theorem to prove

The theorem must state that the finite-dimensional obstruction map on the constant-section layer is

`A_red = A_loc + E_geom + E_link`

and must prove invertibility from explicit bounds, not just from a qualitative argument.

### Current partial closure

The present scaffold supports:

- conservative `delta_total < 1`,
- Neumann inversion of the reduced matrix,
- a candidate explicit bound for the inverse.

### Missing theorem-level pieces

- exact map from the PDE trace to the `R^{2N}` obstruction layer,
- proof that no hidden mode outside `R^{2N}` survives in the chosen window,
- theorem-level statement of how non-constant Fourier modes are absorbed into `X_beta^{(0)}`.

## Target theorem statement

There exist explicit constants `C_max` and `C_src` depending only on the certified datum constants such that the global background `h_0` satisfies

- `m(h_0) = 0` on `S^3 \ Sigma`,
- `||h_0 - h_global_bar||_{X_beta^{ext}} <= C_max`,
- `||m(h_global_bar)||_{Y_{beta-2}} <= C_src`.

## Immediate next implementation tasks

1. Freeze the formal definition of `X_beta^{(0)}`, `X_beta^{ext}`, and `Y_{beta-2}` in repository prose.
2. Promote the current reduced constants into one explicit theorem statement for `A_red`.
3. Split the current numerical scaffold into:
   - theorem-usable constants,
   - still-heuristic constants,
   - pending exact certificates.
4. Add a dedicated certificate for `||J_ext^{-1}||` with exact provenance and checker.
5. Extract a theorem-grade source bound `C_src` and nonlinear constant `C_nl` for the contraction step.

## Non-allowed shortcuts

This phase may not claim "Hypothesis (G) discharged" unless the actual theorem contains:

- spaces,
- operator,
- reduced obstruction map,
- right-inverse norm,
- nonlinear closure statement,
- exact dependence on the certified datum constants.
