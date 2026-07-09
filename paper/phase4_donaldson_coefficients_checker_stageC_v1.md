# Phase 4.1 Stage C/E: coefficient-formula and interval checker

## Status

Independent checker for the Stage B v1 symbolic coefficient formulas and their
`D0` evaluated values. It now includes outward-rounded interval recomputation
for the rational coefficient layer.

Checker:

- [phase4_donaldson_coefficients_checker.py](../scripts/phase4_donaldson_coefficients_checker.py)

Machine output:

- [phase4_donaldson_coefficients_check.json](../certificates/phase4_donaldson_coefficients_check.json)

Checked artifact:

- [phase4_donaldson_coefficients.json](../certificates/phase4_donaldson_coefficients.json)

## Checks

The checker independently verifies:

- the coefficient JSON points to the canonical bigraded type gate;
- the type gate has `all_pass: true`;
- the Donaldson bidegrees are
  `omega/lambda/Theta/mu/M_eps = (1,2)/(3,0)/(2,2)/(0,4)/(3,2)`;
- `F_H omega` has bidegree `(3,1)` and `G_f` returns `lambda`;
- `F_H mu` has bidegree `(2,3)` and `G_f` returns `Theta`;
- `d_H omega` and `d_H Theta` have the expected source bidegrees;
- `P1`, `P2`, and `R3` are present, symbolic, and use
  `X_beta^ext -> Y_{beta-2}`;
- the legacy `coefficient_placeholders` block is absent;
- coefficient formulas have symbolic upper-bound expressions, constants,
  source edges, and checker rules;
- internal dependencies such as `source_P1`, `DP1_norm`, and `raw_P3_scale`
  resolve to formula objects;
- legacy power-counting numbers remain comparison-only.
- every exact rational `D0` coefficient value is reserialized as an
  outward-rounded decimal interval;
- `R_threshold` is bracketed by integer bisection on the cube, avoiding any
  floating-point input to the interval acceptance test.

Current result with the D0 values artifact present:

`172/172` checks pass.

## Boundary

This checker confirms symbolic formula consistency and exact rational
evaluation of the D0 coefficient values, with outward-rounded interval
packaging for the scalar coefficient layer. It does not check:

- global Taylor/majorant convergence for the full reconstruction map;
- convergence of the reconstruction map.

The next upgrade is the global AR product-space theorem, not another P4.1
coefficient placeholder pass.
