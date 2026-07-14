# Phase 4 progress note -- 2026-07-03

## Completed

P4.1 coefficient extraction at `D0` is now public, reproducible, and checked.

Artifacts:

- [phase4_donaldson_E1_E5_operator.md](../../paper/phase4_donaldson_E1_E5_operator.md)
- [phase4_bigraded_type_check.json](../../certificates/phase4_bigraded_type_check.json)
- [phase4_donaldson_coefficients.json](../../certificates/phase4_donaldson_coefficients.json)
- [phase4_donaldson_coefficients_values.json](../../certificates/phase4_donaldson_coefficients_values.json)
- [phase4_donaldson_coefficients_check.json](../../certificates/phase4_donaldson_coefficients_check.json)
- [phase4_level_Q_coefficients.json](../../certificates/phase4_level_Q_coefficients.json)
- [phase4_level_Q_coefficients_check.json](../../certificates/phase4_level_Q_coefficients_check.json)
- [datum_D0.json](../../certificates/datum_D0.json)

Checked D0 values:

- `source_P1 = 3.1185`;
- `source_P2 = 6.36018075`;
- `DP1_norm = 3.1185`;
- `DP2_norm = 8.3981205`;
- `D2P1_norm = 3.1185`;
- `D3m_norm = 4/9`;
- `raw_P3_scale = 3.24648475975`;
- `remainder_R3 = 13770793.170081593`;
- `R_threshold = 3664.0659853300026`.

`R_threshold` is the minimum admissible `R` for the `eps^3` tail contraction:
the certified condition holds for `R >= R_threshold`. It is not an upper
headroom margin.

Checker result:

`172/172` checks pass, including outward-rounded interval recomputation for the
scalar coefficient layer.

Compact Level Q extraction:

`185/185` checks pass. The compact package exposes the citable intervals,
including
`R_threshold in [3664.065985330004, 3664.065985330005]`. Prose may cite the
conservative human round `R_threshold <= 3664.066`.

Paper-facing table check:

`43/43` checks pass for the Level Q coefficient table in
`paper/theorem_Q_certified.md`.

## Scope

This closes P4.1 coefficient values at `D0`.

It does not close:

- global fibrewise Hodge inverse uniformity;
- product-space convergence of the adiabatic reconstruction map;
- preservation of closedness for the actual reconstructed `Phi_eps`;
- anisotropic Joyce perturbation.

## Product-space contract

The Phase 4.2 product-space contract is now serialized:

- [phase4_ar_product_space_contract.json](../../certificates/phase4_ar_product_space_contract.json)
- [phase4_ar_product_space_check.json](../../certificates/phase4_ar_product_space_check.json)

It fixes `A_beta = A_omega x A_lambda x A_mu x A_Theta`, the component
bidegrees, the `F_H/G_f` block architecture, and the conditional
`K_AR_prod = 2079/2000` structural product-max value used by the scalar
majorant.

Checker result:

`36/36` checks pass.

Stage D update (2026-07-09): these obligations are now imported from the
private V/W/X/Y slot theorems and `axis2_L2_assembly_theorem_2026_07_05`.
The historical Codex contract remains the public audit witness for the
product-max normalization.

Previously open theorem obligations:

- uniform fibrewise inverse in the product norm;
- commutator bounds in the product norm;
- global reduced-projection identity;
- closedness preservation.

## Continued

The first conditional scalar AR majorant is now serialized:

- [phase4_ar_majorant_candidate.json](../../certificates/phase4_ar_majorant_candidate.json)
- [phase4_ar_majorant_check.json](../../certificates/phase4_ar_majorant_check.json)

It uses the checked P4.1 values and chooses:

- `R_AR = 4000`;
- `eps_AR = 1/4000`;
- `r_AR = 1/100`.

At `eps_AR`, it gives:

- `source_majorant = 0.0009951911545794`;
- `displacement_majorant = 0.0010345012051852862`;
- `q_AR = 0.001034633607583419`.

Checker result:

`20/20` checks pass.

This uses `K_AR_prod <= 2079/2000` from the product-space contract. Stage D
imports the private L2 assembly closure; Stage E now focuses on outward-rounded
interval packaging of the coefficient layer for Level Q.
