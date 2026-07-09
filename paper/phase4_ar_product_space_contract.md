# Phase 4.2: AR product-space contract

## Status

Definition and checker target only. This is not the global adiabatic
reconstruction theorem.

Artifacts:

- [phase4_ar_product_space_contract.py](../scripts/phase4_ar_product_space_contract.py)
- [phase4_ar_product_space_contract.json](../certificates/phase4_ar_product_space_contract.json)
- [phase4_ar_product_space_checker.py](../scripts/phase4_ar_product_space_checker.py)
- [phase4_ar_product_space_check.json](../certificates/phase4_ar_product_space_check.json)
- [phase4_ar_inverse_budget_audit.md](phase4_ar_inverse_budget_audit.md)

## Contract

The product space is

`A_beta = A_omega x A_lambda x A_mu x A_Theta`

with max norm

`max(||omega||, ||lambda||, ||mu||, ||Theta||)`

in the Phase 3 weighted `beta` convention.

Components:

- `omega in Omega^{1,2}`;
- `lambda in Omega^{3,0}`;
- `mu in Omega^{0,4}`;
- `Theta in Omega^{2,2}`.

The block inverse architecture is:

- `d_f lambda = -F_H omega`, implemented by `-G_f F_H`;
- `d_f Theta = -F_H mu`, implemented by `-G_f F_H`;
- `d_H Theta = 0`, reduced through `Pi_reduced d_H` to `M_eps(h)=0`.

## Checked Constants

The contract imports:

- `K_H_K3 = 9/20`;
- `K_F = 231/100`;
- `K_projection = 1`.

It records the scalar majorant input

`K_AR_prod = K_H_K3 * K_F = 2079/2000 = 1.0395`

as the structural product-max bound for the declared block architecture.

The inverse-budget audit confirms that this repairs the previous normalization
gap: the product-max constant dominates `G_f`, the `G_f F_H` block, and the
projection slot under the declared max norm. The analytic uniformity theorem is
still open.

## Checker Result

`36/36` checks pass.

The checker verifies:

- Donaldson bidegrees;
- `F_H` and `G_f` degree shifts;
- block inverse source and target bidegrees;
- agreement with the P4.1 values certificate;
- product-space definition status;
- all analytic theorem obligations still marked open.

## Open Obligations

The contract explicitly does not prove:

- uniform fibrewise inverse in the product norm;
- commutator bounds in the product norm;
- global reduced-projection identity;
- closedness preservation of the reconstructed tuple.

These are the remaining P4.2 theorem obligations before AR can be discharged.
