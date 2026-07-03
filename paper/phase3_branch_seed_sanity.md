# Phase 3.S1.b: sanity check on pure branch seeds

## Status

Local sanity check only.

This note records a negative result that is useful for scope control: the lower-root collar source `R_quad` is not reproduced by the simplest flat branch-graph probes.

The machine-readable freeze is:

- [phase3_branch_seed_sanity.json](/home/brieuc/gift-framework/GIFT/certificates/phase3_branch_seed_sanity.json).

## Scalar flat probe

For the codimension-one flat maximal graph with seed

`u = c Re(z^3)`,

the symbolic expansion of the maximal operator starts at order `c^3`, with leading term in the same angular channel as `Re(z^3)`.

So this probe does **not** generate the lower obstruction channel `m = 1/2`.

## Vector flat probe

For the codimension-two flat seed

`(u1, u2) = (c Re(z^3), c Im(z^3))`,

the symbolic probe gives zero through the tested order.

So this probe also does **not** produce the desired lower-root source.

## Why this matters

This rules out a tempting but wrong shortcut:

one cannot identify the actual `R_quad` of the maximal-section problem with the naive nonlinear residual of a pure flat branch graph.

Therefore the coefficient still missing in Phase 3 must come from the full geometric setup:

- the actual maximal-section operator rather than the simplest flat graph surrogate,
- the normal-bundle metric `A_bulk`,
- the true tensor contractions of the nonlinear remainder,
- and possibly the interaction between the branch channel and the period-domain geometry.

This is a useful narrowing of the gap: the next coefficient extraction has to be done in the real operator, not in the toy scalar model.
