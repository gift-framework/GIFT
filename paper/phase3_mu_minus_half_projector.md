# Phase 3.S1.h: prototype projector onto `Phi_{-1/2,R/I}`

## Status

Executed symbolic prototype.

This note records the first working projector that extracts the two real coefficients in the fixed lower-root basis.

The executed prototype is:

- [phase3_mu_minus_half_projector_proto.py](/home/brieuc/gift-framework/GIFT/scripts/phase3_mu_minus_half_projector_proto.py),
- with output [phase3_mu_minus_half_projector_proto.json](/home/brieuc/gift-framework/GIFT/certificates/phase3_mu_minus_half_projector_proto.json).

## What it does

Given a symbolic collar expression `expr(rho, varphi)`, the prototype:

1. multiplies by `rho^(1/2)`,
2. takes the limit `rho -> 0`,
3. projects the resulting angular function onto
   `cos(varphi/2)` and `sin(varphi/2)`
   by orthogonal integration on `[0, 4pi]`.

This is exactly the algebraic skeleton needed for extracting

- `kappa_R`,
- `kappa_I`

once a symbolic model for the actual source trace is available.

## Test result

On the demo ansatz

`rho^(-1/2) (a cos(varphi/2) + b sin(varphi/2)) + rho^(3/2) cos(3 varphi/2)`,

the prototype returns

- `kappa_R = a`,
- `kappa_I = b`,
- projection residual `0`.

So the basis extraction itself is no longer hypothetical; it is implemented and checked on a nontrivial test with a higher-order tail.

## What remains

The only missing ingredient is the actual symbolic expression to feed into this projector:

`expr = ev_{-1/2}(m(h_loc,aff + h_loc,branch))`.

Once such an expression or sufficiently explicit ansatz is available, the projector can be reused directly.
