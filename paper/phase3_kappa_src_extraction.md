# Phase 3.S1.f: extraction protocol for `kappa_src^(i)`

## Status

Protocol fixed; the original public computation path is superseded at `D0` by
the private 2026-07-02 P1 chain. The old quadratic extraction task should now
be read as a reconciliation/integration task:

- private theorem-grade result: fixed-Sigma sigma-odd source is cubic in
  `c_0`, with `C_src = 27/16`;
- public surrogate layer: quadratic `gamma_src,sur = 3/8`, explicitly
  uncertified;
- next public task: reconcile the projection and commutator bookkeeping with
  the private parity/cubic derivation.

This note turns the remaining collar-source gap into a concrete extraction task.

The machine-readable freeze is:

- [phase3_kappa_src_extraction_plan.json](../certificates/phase3_kappa_src_extraction_plan.json).

## Target formula

Original public target formula:

`Pi_obs^{(i)}(m(h_bar)) = (c0^(i))^2 * (kappa_src,R^(i) e_i^R + kappa_src,I^(i) e_i^I) + remainder`.

Private P1 changes the fixed-Sigma obstruction-channel target to a cubic
source. The quadratic formula above should therefore be treated as a surrogate
or as a possible regular-sector channel, not as the current obstruction-channel
target.

In the old formula, everything was fixed except the two coefficients

- `kappa_src,R^(i)`,
- `kappa_src,I^(i)`.

## Extraction protocol

1. Expand the local residual `m(h_loc,aff + h_loc,branch)` near `rho = 0`.
2. Represent the resulting source in `Y_{beta-2}` near the obstruction channel.
3. Apply or identify the source projection `Pi_obs`.
4. Factor out `(c0^(i))^2`.
5. Bound the remaining `O(rho^(3/2))` tail separately in `Y_-3`.

This is the exact path from the current Lemma 5.9 statement to a theorem-grade collar constant.

The coefficient-extraction skeleton is now implemented in:

- [phase3_mu_minus_half_projector.md](phase3_mu_minus_half_projector.md),
- [phase3_mu_minus_half_projector_proto.json](../certificates/phase3_mu_minus_half_projector_proto.json).

However, the branch currently still needs the convention-resolution note:

- [phase3_source_trace_convention.md](phase3_source_trace_convention.md),
- [phase3_source_trace_convention_status.json](../certificates/phase3_source_trace_convention_status.json).

The canonical resolution is now fixed in:

- [phase3_source_trace_resolution.md](phase3_source_trace_resolution.md),
- [phase3_source_trace_resolution.json](../certificates/phase3_source_trace_resolution.json).

The corrected analytical bridge is now fixed in:

- [phase3_source_to_obstruction_bridge.md](phase3_source_to_obstruction_bridge.md),
- [phase3_source_to_obstruction_bridge.json](../certificates/phase3_source_to_obstruction_bridge.json).

The operational constant-mode model for the last projection step is now fixed in:

- [phase3_Pi_obs_constant_mode.md](phase3_Pi_obs_constant_mode.md),
- [phase3_Pi_obs_constant_mode_proto.json](../certificates/phase3_Pi_obs_constant_mode_proto.json).

The first channel-compatible coefficient ansatz is now frozen in:

- [phase3_kappa_src_first_ansatz.md](phase3_kappa_src_first_ansatz.md),
- [phase3_kappa_src_first_ansatz_proto.json](../certificates/phase3_kappa_src_first_ansatz_proto.json).

Its confrontation with the current `G`-discharge envelope is now frozen in:

- [phase3_kappa_src_vs_G_discharge.md](phase3_kappa_src_vs_G_discharge.md),
- [phase3_kappa_src_vs_G_discharge.json](../certificates/phase3_kappa_src_vs_G_discharge.json).

The first explicit local quadratic surrogate is now frozen in:

- [phase3_gamma_src_surrogate.md](phase3_gamma_src_surrogate.md),
- [phase3_gamma_src_surrogate_proto.json](../certificates/phase3_gamma_src_surrogate_proto.json).

## Why this is now well-posed

The previous ambiguity is gone:

- the channel is fixed,
- the real basis is fixed,
- the relation to `R^{2N}` is fixed,
- the normal-form comparison channel is fixed,
- the constant longitudinal projection rule is fixed at prototype level.

So the next missing artifact is not conceptual. It is a coefficient computation.

## Expected `D0` simplification

At the normalized datum `D0`, the 77 identical-copies construction strongly suggests that the component-wise pair

`(kappa_src,R^(i), kappa_src,I^(i))`

should collapse to a common pair independent of `i`.

That symmetry reduction is not yet a theorem here, but it is the right shape to expect from the certificate.

## What remains after this

Once the private cubic coefficient is integrated into this public protocol, the
collar contribution to `C_src` is reduced to a plain explicit inequality. Then
only the projection theorem and annular commutator term `R_comm` remain to be
certified separately before `C_src_exact` can be assembled in the public
ledger.

The algebraic conversion from the extracted coefficient pair to the cubic corrective
datum is now frozen separately in:

- [phase3_kappa_src_projection.md](phase3_kappa_src_projection.md),
- [phase3_kappa_src_projection_proto.json](../certificates/phase3_kappa_src_projection_proto.json).
