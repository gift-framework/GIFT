# Phase 3.S1.i: source-trace convention mismatch at the lower root

## Status

Open consistency issue identified.

This note records a genuine mismatch in the current analytic write-up that must be resolved before any theorem-grade extraction of `kappa_src` can be trusted.

## The mismatch

Two statements currently coexist in the branch:

1. The lower obstruction trace is repeatedly described as the `mu = -1/2` sigma-odd trace, i.e. in the basis

   `rho^(-1/2) cos(varphi/2), rho^(-1/2) sin(varphi/2)`.

2. The quadratic remainder in Lemma 5.9 and in the global source discussion is written as

   `|c0|^2 * A_bulk(alpha_1, alpha_1) * rho^(1/2) * (sigma-odd m = +/- 1/2) + O(rho^(3/2))`.

Taken literally, these are not the same radial order.

## Why this matters

All recent extraction scaffolds for `kappa_src` assumed that the leading source trace is to be read in the lower-root basis

`rho^(-1/2) cos(varphi/2), rho^(-1/2) sin(varphi/2)`.

If the actual leading source is instead `rho^(1/2)` in the same angular channel, then one of the following must be true:

- the draft has a sign/order typo in the radial exponent;
- the source is being quoted after an implicit weight renormalization;
- the evaluation map `ev_{-1/2}` is being used in a shifted convention not yet documented.

Until this is resolved, any extracted scalar `kappa_src` would risk being attached to the wrong trace space.

## Evidence currently on disk

The `mu = -1/2` convention is explicit in:

- [phase3_spaces.md](phase3_spaces.md),
- the draft §4.3 / §5.5 discussion of the lower-root cokernel,
- [phase3_mu_minus_half_basis.md](phase3_mu_minus_half_basis.md).

The `rho^(1/2)` source statement is explicit in:

- draft §3.1 global source discussion,
- draft Lemma 5.9 sketch,
- `axis2_hypothesis_G_discharge_2026_07_01.py`.

## Safe consequence

The coefficient-extraction machinery implemented so far remains useful, but it must be treated as a projector onto a **candidate** lower-root basis until this convention mismatch is resolved.

## Required next step

The convention-resolution artifact is now frozen in:

- [phase3_source_trace_resolution.md](phase3_source_trace_resolution.md),
- [phase3_source_trace_resolution.json](../certificates/phase3_source_trace_resolution.json).

Repository convention from now on:

- `rho^(1/2)` statements are read as raw source-profile statements,
- `ev_{-1/2}` is reserved for coefficients in the fixed lower-root basis,
- weighted `Y_-3` quantities are named separately.
