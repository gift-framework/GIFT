# Phase 4.2: conditional scalar AR majorant

## Status

Conditional scalar majorant only. This is not the global adiabatic
reconstruction theorem.

Producer:

- [phase4_ar_majorant_candidate.py](/home/brieuc/gift-framework/GIFT/scripts/phase4_ar_majorant_candidate.py)

Machine output:

- [phase4_ar_majorant_candidate.json](/home/brieuc/gift-framework/GIFT/certificates/phase4_ar_majorant_candidate.json)

Checker:

- [phase4_ar_majorant_checker.py](/home/brieuc/gift-framework/GIFT/scripts/phase4_ar_majorant_checker.py)
- [phase4_ar_majorant_check.json](/home/brieuc/gift-framework/GIFT/certificates/phase4_ar_majorant_check.json)

## Conditional Input

The majorant assumes:

`K_AR_prod <= 9/20`.

This is the product-space fibrewise reconstruction inverse bound. It is not
yet proved. The certificate explicitly records:

- `product_space_defined = false`;
- `uniform_fibrewise_inverse_proved = false`.

## Candidate Threshold

The certificate chooses:

- `R_AR = 4000`;
- `eps_AR = 1/4000`;
- `r_AR = 1/100`.

At `eps_AR`, using the checked P4.1 values:

- `source_majorant = 0.0009951911545794`;
- `displacement_majorant = 0.00044783601956072995`;
- `q_AR = 0.0004478933366161987`;
- `tail_at_eps = 0.0002151686432825249`.

Checker result:

`14/14` checks pass.

## Boundary

This scalar certificate shows that once the product-space inverse bound and the
actual reconstruction map estimates are proved, the D0 coefficient values leave
a large numerical margin at `R_AR = 4000`.

It does not prove:

- the product Banach-space definition;
- the uniform fibrewise Hodge inverse theorem;
- the nonlinear reconstruction contraction;
- closedness of the actual reconstructed `Phi_eps`;
- any part of the anisotropic Joyce theorem.
