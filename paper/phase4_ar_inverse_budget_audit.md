# Phase 4.2: inverse-budget audit

## Status

Audit only. This is not an AR theorem.

Artifacts:

- [phase4_ar_inverse_budget_audit.py](../scripts/phase4_ar_inverse_budget_audit.py)
- [phase4_ar_inverse_budget_audit.json](../certificates/phase4_ar_inverse_budget_audit.json)
- [phase4_ar_inverse_budget_checker.py](../scripts/phase4_ar_inverse_budget_checker.py)
- [phase4_ar_inverse_budget_check.json](../certificates/phase4_ar_inverse_budget_check.json)

## Finding

The repaired scalar majorant uses

`K_AR_prod = K_H_K3 * K_F = 2079/2000 = 1.0395`.

Under the unweighted max norm declared in the current product-space contract,
the block `-G_f F_H` carries the declared bound

`K_H_K3 * K_F = (9/20) * (231/100) = 2079/2000 = 1.0395`.

The reduced projection slot carries bound `1`.

This value covers `G_f`, the `G_f F_H` block, and the projection slot under the
current unweighted product-max interpretation.

## Checked Result

`15/15` checks pass.

The checker verifies:

- `K_AR_prod` covers `G_f` alone;
- `K_AR_prod` covers `G_f F_H`;
- `K_AR_prod` covers the projection slot;
- the unweighted max-norm normalization mismatch is repaired.

## Consequence

The scalar normalization issue is repaired by option 3. Before P4.2 can become
a theorem, the remaining work is analytic: uniform fibrewise inverse in the
product norm, product commutator bounds, reduced-projection identity, and
closedness preservation.
