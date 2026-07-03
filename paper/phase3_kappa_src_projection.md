# Phase 3.S1.g: algebraic map from source coefficient to cubic corrective datum

## Status

Prototype executed.

This note records the exact algebraic interface between the lower-root source coefficient and the cubic datum in the extended-domain layer.

The executed prototype is:

- [phase3_kappa_src_projection_proto.py](/home/brieuc/gift-framework/GIFT/scripts/phase3_kappa_src_projection_proto.py),
- with output [phase3_kappa_src_projection_proto.json](/home/brieuc/gift-framework/GIFT/certificates/phase3_kappa_src_projection_proto.json).

## Main formula

If the lower-root source trace on one component is

`ev_{-1/2}(m(h_bar)) = c0^2 * (kappa_R Phi_{-1/2,R} + kappa_I Phi_{-1/2,I})`,

then the cubic corrective datum that cancels it through the model diagonal block is

`v0 = - c0^2 / (2 * A_bulk(alpha_1, alpha_1)) * (kappa_R psi_R + kappa_I psi_I)`.

This is just the model identity

`J_h(psi_*) = 2 * A_bulk(alpha_1, alpha_1) * Phi_{-1/2,*}`

written in coordinates.

## Normalized specialization

In the normalized `A5` model where

`A_bulk(alpha_1, alpha_1) = 1`,

the coefficient multiplier becomes simply

`-1/2`.

So one unit of lower-root source coefficient is cancelled by minus one-half unit of cubic upper-root datum in the matching coordinate.

## Why this matters

This prototype does not compute `kappa_R` or `kappa_I`, but it removes one more layer of ambiguity:

- once the source coefficient is extracted,
- the corresponding `R^{2N}` correction is immediate.

So the remaining local analytic task is now exactly one thing:

- compute the coefficient pair `(kappa_R, kappa_I)` and its tail bound.
