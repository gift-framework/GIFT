# Phase 4 progress note -- 2026-07-03

## Completed

P4.1 coefficient extraction at `D0` is now public, reproducible, and checked.

Artifacts:

- [phase4_donaldson_E1_E5_operator.md](/home/brieuc/gift-framework/GIFT/paper/phase4_donaldson_E1_E5_operator.md)
- [phase4_bigraded_type_check.json](/home/brieuc/gift-framework/GIFT/certificates/phase4_bigraded_type_check.json)
- [phase4_donaldson_coefficients.json](/home/brieuc/gift-framework/GIFT/certificates/phase4_donaldson_coefficients.json)
- [phase4_donaldson_coefficients_values.json](/home/brieuc/gift-framework/GIFT/certificates/phase4_donaldson_coefficients_values.json)
- [phase4_donaldson_coefficients_check.json](/home/brieuc/gift-framework/GIFT/certificates/phase4_donaldson_coefficients_check.json)
- [datum_D0.json](/home/brieuc/gift-framework/GIFT/certificates/datum_D0.json)

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

Checker result:

`146/146` checks pass.

## Scope

This closes P4.1 coefficient values at `D0`.

It does not close:

- global fibrewise Hodge inverse uniformity;
- product-space convergence of the adiabatic reconstruction map;
- preservation of closedness for the actual reconstructed `Phi_eps`;
- anisotropic Joyce perturbation.

## Next

The next Phase 4 artifact is the global AR theorem specification:

- define the product Banach space for `(omega, lambda, mu, Theta)`;
- name the fibrewise inverse and its uniformity assumptions;
- build the majorant/contraction constants using the now-checked P4.1 values;
- state the exact output: `M_eps(h_eps)=0` implies the reconstructed
  `Phi_eps(h_eps)` is closed and has controlled coclosed defect.

## Continued

The first conditional scalar AR majorant is now serialized:

- [phase4_ar_majorant_candidate.json](/home/brieuc/gift-framework/GIFT/certificates/phase4_ar_majorant_candidate.json)
- [phase4_ar_majorant_check.json](/home/brieuc/gift-framework/GIFT/certificates/phase4_ar_majorant_check.json)

It uses the checked P4.1 values and chooses:

- `R_AR = 4000`;
- `eps_AR = 1/4000`;
- `r_AR = 1/100`.

At `eps_AR`, it gives:

- `source_majorant = 0.0009951911545794`;
- `displacement_majorant = 0.00044783601956072995`;
- `q_AR = 0.0004478933366161987`.

Checker result:

`14/14` checks pass.

This is conditional on `K_AR_prod <= 9/20`; the product-space inverse theorem
is still open.
