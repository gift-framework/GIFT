# Phase 3.S1.c: exact coefficient already available in the `mu = -1/2` channel

## Status

Channel coefficient extracted; source coefficient still open.

This note records a distinction that matters for the rest of Phase 3:

- we still do **not** know the full coefficient of the source term `R_quad`,
- but we **do** already know one exact coefficient in the correct lower-root channel `mu = -1/2`.

The machine-readable freeze is:

- [phase3_mu_minus_half_channel.json](../certificates/phase3_mu_minus_half_channel.json).

## Exact lower-root coefficient

From the `rho^2` scaffold:

`(w-rho)^(3/2) = ... + (3/8) rho^2 w^(-1/2) + ...`

So the coefficient in the `mu = -1/2` sigma-odd channel is exactly

`3/8`.

This is the first exact coefficient in the actual obstruction channel, not merely in the curvature bookkeeping of `w^(3/2)`.

## What it means

This coefficient belongs to the geometric branch-motion lever:

- it shows that second-order branch motion hits the `mu = -1/2` channel nontrivially,
- and does so with a nonzero explicit coefficient.

That is different from saying it is the coefficient of the source residual `R_quad`.

## Why this is useful

This narrows the real target.

The next coefficient extraction should not be framed as:

- "compute something in a toy flat graph,"

but rather as:

- "compare the actual source trace of Lemma 5.9 against the already known geometric normal form in the same `mu = -1/2` channel."

In other words, the repository already contains a nontrivial exact normal-form coefficient in the correct channel, and the remaining problem is to connect the source term to that channel quantitatively.
