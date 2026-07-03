# Phase 4.1: evaluated Donaldson coefficient values at D0

## Status

Theorem-grade coefficient-value artifact for the public Stage B v1 symbolic
formulas at the datum `D0`.

This closes the P4.1 coefficient extraction at `D0`. It does not prove the
global adiabatic reconstruction convergence theorem.

Producer:

- [phase4_donaldson_coefficients_values.py](/home/brieuc/gift-framework/GIFT/scripts/phase4_donaldson_coefficients_values.py)

Machine output:

- [phase4_donaldson_coefficients_values.json](/home/brieuc/gift-framework/GIFT/certificates/phase4_donaldson_coefficients_values.json)

Checker:

- [phase4_donaldson_coefficients_checker.py](/home/brieuc/gift-framework/GIFT/scripts/phase4_donaldson_coefficients_checker.py)
- [phase4_donaldson_coefficients_check.json](/home/brieuc/gift-framework/GIFT/certificates/phase4_donaldson_coefficients_check.json)

## Values

The checked D0 values are:

- `source_P1 = 6237/2000 = 3.1185`;
- `source_P2 = 25440723/4000000 = 6.36018075`;
- `DP1_norm = 6237/2000 = 3.1185`;
- `DP2_norm = 16796241/2000000 = 8.3981205`;
- `D2P1_norm = 6237/2000 = 3.1185`;
- `D3m_norm = 4/9`;
- `raw_P3_scale = 12985939039/4000000000 = 3.24648475975`;
- `remainder_R3 = 137707931700815927923/10000000000000`;
- `R_threshold = 3664.0659853300026`.

The checker recomputes the exact rational values from:

- `K_H_K3 = 9/20`;
- `K_F = 231/100`;
- `G_aug = 183/5`;
- `C_nl = 2/3`;
- Donaldson gauge constants `C_conn_1 = C_conn_2 = C_conn_3 = 0`;
- unit-normalized wedge/Hodge/source constants equal to `1`;
- `C_D3m = 4/9`.

Current checker result:

`146/146` checks pass.

## Boundary

The old public power-counting values remain only a comparison. The values above
replace them for `source_P1`, `source_P2`, and `remainder_R3` in
`datum_D0.json`.

This still does not discharge:

- global fibrewise Hodge inverse theorem beyond the constants used here;
- product-space convergence of the adiabatic reconstruction map;
- anisotropic Joyce perturbation.
