# Phase 4.1: Donaldson E1--E5 operator ledger

## Status

Symbolic operator ledger for P4.1. This is not a coefficient certificate, not a
convergent reconstruction theorem, and not a torsion-free existence theorem.

This file is the Stage A input for replacing the current power-counting
candidate

`M_eps(h) = m(h) + eps P1(h) + eps^2 P2(h) + R_eps(h)`

by coefficients derived from Donaldson's bigraded equations.

Current related artifacts:

- [phase4_true_coefficients_derivation_plan.md](phase4_true_coefficients_derivation_plan.md)
- [phase4_adiabatic_operator_form.md](phase4_adiabatic_operator_form.md)
- [phase4_bigraded_type_check.json](../certificates/phase4_bigraded_type_check.json)
- [phase4_adiabatic_sources_candidate.json](../certificates/phase4_adiabatic_sources_candidate.json)

## Bigraded Convention

Let `pi : M^7 -> B^3` be the K3 fibration on the current rank-one 77-unlink
branch, with fibres `X_b`. For a chosen Ehresmann connection `H`, write

`Omega^{p,q} = C^\infty(B, Lambda^p T^*B tensor Omega^q(X_b))`.

The exterior derivative decomposes as

`d = d_f + d_H + F_H`,

with bidegrees:

- `d_f : Omega^{p,q} -> Omega^{p,q+1}`;
- `d_H : Omega^{p,q} -> Omega^{p+1,q}`;
- `F_H : Omega^{p,q} -> Omega^{p+2,q-1}`.

Here `F_H` is the curvature of `H`, acting by contraction of the vertical
curvature vector field into the fibrewise form component.

The P4 norms are the same public norms as Phase 3:

- reduced variable: `h in X_beta^ext`;
- reduced source: `M_eps(h) in Y_{beta-2}`;
- all projected coefficients `P1`, `P2`, `R_eps / eps^3` are measured as
  `X_beta^ext -> Y_{beta-2}` objects, or at `h0` as elements of `Y_{beta-2}`.

## Unknowns And Bidegrees

The reconstruction unknowns attached to a period section `h` are:

- `omega(h) in Omega^{1,2}`:
  horizontal one-form valued in fibrewise closed 2-forms. This is the
  mixed-degree component of Donaldson's `phi = omega + lambda`.
- `lambda(h) in Omega^{3,0}`:
  the purely horizontal volume component of Donaldson's `phi`. This is the
  global ledger component previously denoted locally by `lambda_3`.
- `Theta(h) in Omega^{2,2}`:
  the mixed component of `*_{phi} phi = Theta + mu`.
- `mu(h) in Omega^{0,4}`:
  the fibrewise volume-form component of `*_{phi} phi`.
- `Phi_eps(h)`:
  reconstructed closed positive 3-form candidate obtained after solving the
  fibrewise equations and inserting the base scale `eps = R^-1`.

The local notation `omega_F wedge lambda_1` may still be useful inside a
trivialisation for the `(1,2)` component `omega`, but it is not the global
ledger component called `lambda`. In Donaldson's global bigraded bookkeeping,
`lambda` is already the `(3,0)` component.

## Donaldson Equations

The local numbering used by the current private draft is:

`(E1) d omega = 0`

`(E2) d mu = 0`

`(E3) d_f lambda = -F_H omega`

`(E4) d_H mu = 0`

`(E5) d_f Theta = -F_H mu`

The remaining reduced equation is:

`(E6) d_H Theta = 0`

The coefficient extractor uses the bigraded components of these equations.
The components used before Stage B are:

`d_f omega = 0`

`d_H omega = 0`

`d_f lambda = -F_H omega`

`d_H mu = 0`

`d_f Theta = -F_H mu`

`d_H Theta = 0`

These are type-consistent with the bidegrees above:

`F_H omega in Omega^{3,1}`,

and

`F_H mu in Omega^{2,3}`.

The horizontal closure component `d_H omega = 0` is the connection-choice
equation. If the chosen gauge rewrites part of it through an auxiliary
correction, the term must be serialized as a `d_H omega` source with its gauge
convention recorded.

In the adiabatic reduction, `(E6)` is projected to the maximal-section source:

`Pi_reduced(E6) = M_eps(h)`.

At `eps = 0`, this gives the Phase 3 maximal equation:

`M_0(h) = m(h)`.

For `eps > 0`, the eliminated solutions of `(E1)`--`(E5)` feed correction
terms into `(E6)`:

`M_eps(h) = m(h) + eps P1(h) + eps^2 P2(h) + R_eps(h)`.

## Fibrewise Hodge Inversion

The equations with `d_f` on the left are solved fibre-by-fibre on K3. The
operator that must be named in every estimate is:

`G_f := Delta_f^{-1} d_f^*`

on the relevant `d_f`-exact source sector.

The public constant name is:

`K_H_K3 = ||G_f||`.

No coefficient may use an unnamed `O(1)` elliptic inverse. The two direct
uses in the Stage B producer are:

`lambda = -G_f(F_H omega) + harmonic gauge term in Omega^{3,0}`,

and

`Theta = -G_f(F_H mu) + harmonic gauge term in Omega^{2,2}`.

The harmonic gauge terms must be either fixed by the Donaldson connection
choice, projected out by `Pi_reduced`, or serialized as finite-dimensional
obstruction variables. They are not allowed to be hidden inside `K_H_K3`.

## Connection Choice

In a local trivialisation, the connection is fixed by the equation

`partial_j omega_i - partial_i omega_j + L_{v_j} omega_i - L_{v_i} omega_j = 0`.

Stage B must encode this as a source of `d_H omega` and commutator terms, not
as a free cancellation. The associated coefficient names are:

- `C_conn_1`: first-order connection-choice contribution to `P1`;
- `C_conn_2`: second-order connection-choice contribution to `P2`;
- `C_comm`: commutators between `d_H`, `G_f`, and the varying fibre metric.

These names are symbolic constant slots for formula-level bounds. They are not
certified constants until formula-level or interval values are supplied and
checked independently.

## Epsilon Bookkeeping

Use `eps = R^-1`.

The coefficient extraction convention is:

- order `eps^0`:
  `Pi_reduced(E6)` equals `m(h)`;
- order `eps^1`:
  all single insertions of `F_H`, one `G_f`, one connection-choice correction,
  or one base/fibre commutator project to `P1(h)`;
- order `eps^2`:
  two curvature insertions, one curvature insertion applied to the first
  reconstructed correction, second connection-choice terms, and quadratic
  Hodge-star / hypersymplectic nonlinearities project to `P2(h)`;
- order `eps^3` and higher:
  the remaining terms form `R_eps(h)`.

Stage B must record the source of each power explicitly. A term may not be
assigned to `P1` or `P2` merely because its bound has the right size.

## Symbolic Definitions For Stage B

Let `A0(h)` be the `eps = 0` reconstruction solving `(E1)`--`(E5)` and let
`Ak(h)` be the order `eps^k` correction determined by the fibrewise Hodge
iteration and connection choice.

The reduced coefficients are defined symbolically by:

`P1(h) = Pi_reduced(Source_E6_order_1(A0(h), A1(h), H, F_H))`.

`P2(h) = Pi_reduced(Source_E6_order_2(A0(h), A1(h), A2(h), H, F_H))`.

`R_eps(h) = M_eps(h) - m(h) - eps P1(h) - eps^2 P2(h)`.

The candidate script must expose the following tree:

```text
F_H
  -> E3 source: F_H omega in Omega^{3,1}
  -> G_f inverse
  -> lambda in Omega^{3,0}
  -> E5 source: F_H mu in Omega^{2,3}
  -> G_f inverse
  -> Theta in Omega^{2,2}
d_H omega
  -> connection-choice correction
  -> commutator terms
d_H Theta
  -> E6 projection
  -> m
  -> P1
  -> P2
  -> R3
```

## Required Coefficient Names

The first producer must output at least:

- `source_P1 = ||P1(h0)||_{Y_{beta-2}}`;
- `source_P2 = ||P2(h0)||_{Y_{beta-2}}`;
- `DP1_norm = ||DP1(h0)||_{X_beta^ext -> Y_{beta-2}}`;
- `DP2_norm = ||DP2(h0)||_{X_beta^ext -> Y_{beta-2}}`;
- `D2P1_norm`;
- `D3m_norm`;
- `raw_P3_scale`;
- `remainder_R3 = ||R_eps|| / eps^3`.

Each item must record:

- formula source;
- domain and codomain;
- `eps` order;
- constants used, including `K_H_K3` for every fibrewise inverse;
- status;
- checker rule.

## Promotion Boundary

This document closes only Stage A of the P4.1 true-coefficient route. It does
not prove:

- the global fibrewise Hodge inverse estimate;
- convergence of the adiabatic reconstruction map;
- closedness of `Phi_eps(h)` for actual `eps > 0`;
- any Joyce perturbation statement;
- any compact `K7` torsion-free theorem.

The next artifact is `scripts/phase4_donaldson_coefficients.py`, whose first
version should serialize this operator tree and keep all new coefficient
statuses as `candidate_not_theorem`. It must first read or reproduce
`phase4_bigraded_type_check.json` and refuse coefficient production unless
`all_pass = true`.
