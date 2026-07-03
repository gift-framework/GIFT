# Phase 4.1: adiabatic operator form and source candidates

## Status

Legacy candidate P4.1 artifact only. Not a convergent reconstruction theorem.
The power-counting bounds in this file are superseded by
[phase4_donaldson_coefficients_values.json](/home/brieuc/gift-framework/GIFT/certificates/phase4_donaldson_coefficients_values.json).

Machine artifact:

- [phase4_adiabatic_sources_candidate.json](/home/brieuc/gift-framework/GIFT/certificates/phase4_adiabatic_sources_candidate.json)

Producer:

- [phase4_adiabatic_sources_candidate.py](/home/brieuc/gift-framework/GIFT/scripts/phase4_adiabatic_sources_candidate.py)

This file freezes the public Phase 4.1 operator shape and serializes first
candidate norms for `source_P1`, `source_P2`, and `remainder_R3`.

## Operator Form

The adiabatic reconstruction is organized around

`M_eps(h) = m(h) + eps P1(h) + eps^2 P2(h) + R_eps(h)`,

with:

- `eps = R^-1`;
- domain: a neighbourhood of `h0` in `X_beta^ext`;
- codomain: `Y_{beta-2}`;
- background equation: `m(h0) = 0`, supplied only after Phase 3 is proved.

The finite-order ansatz is

`h_eps = h0 + eps xi1 + eps^2 xi2 + h_tail`.

Order matching gives:

`xi1 = -G_aug P1(h0)`,

and

`xi2 = -G_aug [ P2(h0) + DP1(h0)[xi1] + (1/2) D2m_{h0}[xi1,xi1] ]`.

The residual after the `K=2` ansatz is `O(eps^3)`.

## Candidate D0 Norms

The current candidate uses only the available D0 scale

`||F_H|| <= 2.31`

and the sharp-current nonlinear constant

`C_nl = 2/3`.

Candidate source norms:

- `||P1(h0)|| <= 2.31`;
- `||P2(h0)|| <= 2.31^2 = 5.3361`;
- `||DP1|| <= 2.31`;
- `||DP2|| <= 5.3361`;
- `||D2P1|| <= 5.3361`;
- `||D3m|| <= 1`;
- raw `P3` scale `<= 2.31^3 = 12.326391`.

These are power-counting candidates, not theorem inputs.

## Candidate Consequences

Using the working `||G_aug|| <= 36.635`, the script obtains:

- `||xi1|| <= 84.62685`;
- `xi2` forcing bound `<= 2588.0587038075`;
- `||xi2|| <= 94813.53061398776`;
- candidate `||R_eps|| / eps^3 <= 5688783.303307713`.

The scalar tail test

`4 ||G_aug||^2 C_nl R3_coeff eps^3 < 1`

then gives:

- `eps <= 0.00036621811958484525`;
- equivalently `R >= 2730.613114210807`.

This is a diagnostic threshold only. It is expected to be pessimistic because
the current candidate uses crude powers of `||F_H||` rather than the actual
Donaldson `(E1)-(E5)` operator coefficients.

## Promotion Requirements

To promote this P4.1 artifact into Level Q:

1. derive `P1`, `P2`, and `R_eps` from the actual Donaldson equations in the
   selected norms;
2. replace the `||F_H||` power-counting by interval enclosures for
   `||P1(h0)||` and `||P2(h0)||`;
3. certify `DP1`, `DP2`, `D2P1`, `D3m`, and the raw `P3` scale;
4. provide an independent checker;
5. only then update `datum_D0.json` from candidate values to certified
   intervals.

Until then, Phase 4 remains at operator-form/candidate-source status.
