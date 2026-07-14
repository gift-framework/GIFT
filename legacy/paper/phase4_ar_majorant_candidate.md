# Phase 4.2: conditional scalar AR majorant

> **2026-07-09 status.** Historical conditional majorant. Stage D imports the
> theorem-grade L2 assembly for D0; this note remains useful for the public
> product-space audit trail but is not the active reconstruction theorem
> citation.

## Status

Conditional scalar majorant only. This is not the global adiabatic
reconstruction theorem.

Producer:

- [phase4_ar_majorant_candidate.py](../../scripts/phase4_ar_majorant_candidate.py)

Machine output:

- [phase4_ar_majorant_candidate.json](../../certificates/phase4_ar_majorant_candidate.json)

Checker:

- [phase4_ar_majorant_checker.py](../../scripts/phase4_ar_majorant_checker.py)
- [phase4_ar_majorant_check.json](../../certificates/phase4_ar_majorant_check.json)

Product-space contract:

- [phase4_ar_product_space_contract.json](../../certificates/phase4_ar_product_space_contract.json)
- [phase4_ar_product_space_check.json](../../certificates/phase4_ar_product_space_check.json)
- [phase4_ar_inverse_budget_audit.json](../../certificates/phase4_ar_inverse_budget_audit.json)
- [phase4_ar_inverse_budget_check.json](../../certificates/phase4_ar_inverse_budget_check.json)

## Conditional Input

The majorant assumes:

`K_AR_prod <= 2079/2000 = 1.0395`.

This value now comes from the product-space contract. The product space itself
is defined and checked, but the uniform inverse theorem is not yet proved. The
certificate explicitly records:

- `product_space_defined = true`;
- `uniform_fibrewise_inverse_proved = false`.
- `commutators_bounded_in_product_norm = false`;
- `reduced_projection_global_identity_proved = false`;
- `closedness_preservation_proved = false`.

The inverse-budget audit confirms that this product-max value covers the
declared `G_f F_H` block and projection slot under the current max norm.

## Candidate Threshold

The certificate chooses:

- `R_AR = 4000`;
- `eps_AR = 1/4000`;
- `r_AR = 1/100`.

At `eps_AR`, using the checked P4.1 values:

- `source_majorant = 0.0009951911545794`;
- `displacement_majorant = 0.0010345012051852862`;
- `q_AR = 0.001034633607583419`;
- `tail_at_eps = 0.0002151686432825249`.

Checker result:

`20/20` checks pass.

## Boundary

This scalar certificate shows that once the product-space inverse bound and the
actual reconstruction map estimates are proved, the D0 coefficient values leave
a large numerical margin at `R_AR = 4000`.

It does not prove:

- the uniform fibrewise Hodge inverse theorem;
- the nonlinear reconstruction contraction;
- closedness of the actual reconstructed `Phi_eps`;
- any part of the anisotropic Joyce theorem.
