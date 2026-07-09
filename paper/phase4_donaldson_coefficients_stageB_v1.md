# Phase 4.1 Stage B v1: typed Donaldson coefficient formulas

## Status

Candidate symbolic formula artifact only. This is not a numerical coefficient
certificate, not an interval certificate, and not a convergence theorem for
adiabatic reconstruction.

Producer:

- [phase4_donaldson_coefficients.py](../scripts/phase4_donaldson_coefficients.py)

Machine output:

- [phase4_donaldson_coefficients.json](../certificates/phase4_donaldson_coefficients.json)

Gate:

- [phase4_bigraded_type_check.json](../certificates/phase4_bigraded_type_check.json)

The producer refuses coefficient production unless the bigraded type gate has
`all_pass: true`.

## Output

The artifact serializes:

- Donaldson source equations:
  `d_H omega = 0`, `d_f lambda = -F_H omega`,
  `d_H mu = 0`, `d_f Theta = -F_H mu`, `d_H Theta = 0`;
- the typed edges
  `F_H omega -> G_f -> lambda`,
  `F_H mu -> G_f -> Theta`,
  `d_H omega -> connection choice`,
  `d_H Theta -> Pi_reduced -> M_eps`;
- symbolic coefficient objects `P1`, `P2`, and `R3`;
- symbolic upper-bound formulas for `source_P1`, `source_P2`, `DP1_norm`,
  `DP2_norm`, `D2P1_norm`, `D3m_norm`, `raw_P3_scale`, and `remainder_R3`.

The old power-counting candidate is retained only under
`comparison_to_power_counting_candidate`. It is not used as a derivation.

## Boundary

This closes the consolidated symbolic-formula pass for Stage B. The D0 values
for these formulas are supplied separately by
[phase4_donaldson_coefficients_values.json](../certificates/phase4_donaldson_coefficients_values.json).
This file itself does not provide:

- interval enclosures;
- a fibrewise Hodge inverse theorem;
- convergence of the reconstruction map.

The current independent checker verifies both the symbolic structure and the
D0 values once the values artifact is present.
