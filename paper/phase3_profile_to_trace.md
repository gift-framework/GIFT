# Phase 3.S1.k: prototype comparison between raw source profile and lower-root trace

## Status

Executed symbolic prototype.

This note records the first explicit test of the convention resolution.

The executed prototype is:

- [phase3_profile_to_trace_proto.py](/home/brieuc/gift-framework/GIFT/scripts/phase3_profile_to_trace_proto.py),
- with output [phase3_profile_to_trace_proto.json](/home/brieuc/gift-framework/GIFT/certificates/phase3_profile_to_trace_proto.json).

## Test statement

Take a pure raw source profile

`profile_src = rho^(1/2) (a cos(varphi/2) + b sin(varphi/2))`.

Then:

- as a raw half-angle profile, it carries coefficients `(a, b)`,
- but as a literal lower-root trace in the canonical `mu=-1/2` sense, it gives zero.

The prototype verifies exactly that.

## Why this matters

This confirms that the convention refactoring was necessary.

A formula written at radial order `rho^(1/2)`:

- can still control the weighted `Y_-3` norm,
- but cannot be fed directly into the `ev_{-1/2}` projector as though it were already a lower-root trace coefficient.

So the remaining local analytic gap is now sharply identified:

we still need the actual mechanism that converts the raw source profile used in the draft into the lower-root trace data used by the obstruction theory.

## Consequence for `kappa_src`

At the moment, `kappa_src` should be understood as a coefficient of the **trace object**, not of the raw profile.

So any future computation claiming to extract `kappa_src` must explicitly document the step that passes from

`profile_src`

to

`ev_{-1/2}`.
