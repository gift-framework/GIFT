# Phase 4.1 Plan: deriving the true adiabatic coefficients

## Status

Execution plan for replacing the P4.1 power-counting candidates by actual
Donaldson-operator coefficients.

Current candidate artifact:

- [phase4_adiabatic_operator_form.md](/home/brieuc/gift-framework/GIFT/paper/phase4_adiabatic_operator_form.md)
- [phase4_adiabatic_sources_candidate.json](/home/brieuc/gift-framework/GIFT/certificates/phase4_adiabatic_sources_candidate.json)

Target replacement artifacts:

- `paper/phase4_donaldson_E1_E5_operator.md`
- `scripts/phase4_bigraded_type_checker.py`
- `certificates/phase4_bigraded_type_check.json`
- `scripts/phase4_donaldson_coefficients.py`
- `certificates/phase4_donaldson_coefficients.json`
- `paper/phase4_donaldson_coefficients_stageB_v1.md`
- `scripts/phase4_donaldson_coefficients_values.py`
- `certificates/phase4_donaldson_coefficients_values.json`
- `paper/phase4_donaldson_coefficients_values.md`
- `scripts/phase4_donaldson_coefficients_checker.py`
- `certificates/phase4_donaldson_coefficients_check.json`
- `paper/phase4_donaldson_coefficients_checker_stageC_v1.md`

## Source equations to extract

Use Donaldson's bigraded decomposition of forms for a K3 fibration:

`d = d_f + d_H + F_H`.

The closed positive form data are represented by:

- connection `H`;
- hypersymplectic fibrewise form `omega in Omega^{1,2}`;
- horizontal volume component `lambda in Omega^{3,0}`;
- period section `h : B -> H^2(K3)`.

The equations entering P4.1 are:

1. fibrewise closure:
   `d_f omega = 0`;
2. horizontal closure:
   `d_H omega = 0`;
3. horizontal volume equation:
   `d_f lambda = -F_H omega`;
4. connection-choice equation in a local trivialisation:
   `partial_j omega_i - partial_i omega_j + L_{v_j} omega_i - L_{v_i} omega_j = 0`;
5. the remaining coclosed/torsion-free reduction, after eliminating the
   fibrewise unknowns, becomes
   `M_eps(h) = m(h) + eps P1(h) + eps^2 P2(h) + R_eps(h)`.

The plan below derives `P1`, `P2`, and `R_eps` as operator coefficients in the
same `X_beta^ext -> Y_{beta-2}` norm convention used by Phase 3.

## Coefficient Inventory

The final P4.1 coefficient certificate must expose:

- `source_P1 = ||P1(h0)||`;
- `source_P2 = ||P2(h0)||`;
- `DP1_norm = ||DP1(h0)||`;
- `DP2_norm = ||DP2(h0)||`;
- `D2P1_norm = ||D^2P1(h0)||`;
- `D3m_norm = ||D^3m(h0)||`;
- `raw_P3_scale`;
- `remainder_R3 = ||R_eps|| / eps^3` after the `K=2` ansatz.

Each item must have a formula, norm, status, provenance, and checker.

## Stage A: symbolic operator ledger

Deliverable:

- `paper/phase4_donaldson_E1_E5_operator.md`
- `scripts/phase4_bigraded_type_checker.py`
- `certificates/phase4_bigraded_type_check.json`

Tasks:

1. Write the bigraded domains/codomains for each term:
   `omega`, `H`, `lambda`, `Phi`, `P1`, `P2`, `R_eps`.
2. State how `F_H` acts:
   `F_H : Omega^{p,q} -> Omega^{p+2,q-1}`.
3. State how fibrewise Hodge inversion solves
   `d_f lambda = -F_H omega`.
4. Track the exact powers of `eps` coming from the base/fibre scaling.
5. Produce symbolic definitions:
   `P1 = Pi_reduced(source at order eps)`;
   `P2 = Pi_reduced(source at order eps^2)`;
   `R_eps = remaining order >= eps^3`.
6. Type-check every operator-tree edge before numerical coefficient work.

Acceptance:

- no numeric constants yet;
- every term has a domain/codomain and eps power;
- every use of Hodge inversion names `K_H_K3`.
- type-check certificate has `all_pass: true`.

## Stage B: first coefficient producer

Deliverable:

- `scripts/phase4_donaldson_coefficients.py`
- `certificates/phase4_donaldson_coefficients.json`
- `paper/phase4_donaldson_coefficients_stageB_v1.md`

Precondition:

- `certificates/phase4_bigraded_type_check.json` has `all_pass: true`.

Tasks:

1. Implement a structured coefficient tree:
   `F_H`, `d_H omega`, `d_f^{-1}`, `lambda`, `P1`, `P2`, `R3`.
2. Start with symbolic constants:
   `K_F`, `K_H`, `C_nl`, `G_aug`, `C_wedge`, `C_Hodge`.
3. Replace the crude candidates:
   `P1 <= K_F`;
   `P2 <= K_F^2`;
   with formula-level bounds:
   `P1 <= C_P1 * K_H * K_F * source0`;
   `P2 <= C_P2 * (K_H*K_F)^2 * source0 + lower-order commutators`.
4. Serialize exact rational arithmetic where coefficients are combinatorial.
5. Keep status `candidate_not_theorem` until an independent checker exists.

Acceptance:

- producer refuses to emit coefficient bounds if the bigraded type-check fails;
- output includes the old crude bounds and new formula-level bounds side by
  side;
- output records whether the new bounds improve `R_threshold_candidate`.

Stage B v1 status:

- the producer and symbolic JSON exist;
- the current output is a typed coefficient tree with symbolic upper-bound
  formulas and no placeholder block;
- D0 numeric/rational values for the symbolic constants now exist in
  `phase4_donaldson_coefficients_values.json`.

## Stage C: independent checker

Deliverable:

- `scripts/phase4_donaldson_coefficients_checker.py`
- `certificates/phase4_donaldson_coefficients_check.json`
- `paper/phase4_donaldson_coefficients_checker_stageC_v1.md`

Tasks:

1. Read `phase4_donaldson_coefficients.json`.
2. Recompute every scalar inequality from serialized rational inputs.
3. Reject missing source paths, missing norm labels, or non-rational constants.
4. Check the `K=2` Taylor algebra:
   `xi1 = -G_aug P1`;
   `xi2 = -G_aug(P2 + DP1 xi1 + 1/2 D2m[xi1,xi1])`;
   `R3` formula includes all listed third-order terms.

Acceptance:

- checker uses no shared helper functions from the producer;
- checker emits `all_pass: true` only if all inequalities reproduce exactly.

Stage C v1 status:

- the checker exists and independently verifies the symbolic Stage B formulas;
- current result is `146/146` structural and arithmetic checks passing;
- interval packaging for broader AR convergence remains open, but P4.1 D0
  coefficient values are evaluated and checked.

## Stage D: replace datum candidates

After Stages B-C pass, update:

- [datum_D0.json](/home/brieuc/gift-framework/GIFT/certificates/datum_D0.json)

Fields to update:

- `source_P1`;
- `source_P2`;
- `remainder_R3`.

Allowed statuses:

- `candidate_not_theorem` if still formula-level only;
- `certified_upper_bound` only after outward-rounded intervals and independent
  checker exist.

## Stage E: promotion gate

The P4.1 coefficients are promotable to Level Q only when:

1. `P1`, `P2`, and `R_eps` are derived from the actual Donaldson equations, not
   power-counting proxies;
2. every norm is in the public `X_beta^ext -> Y_{beta-2}` convention;
3. all constants are rational or outward-rounded intervals;
4. a checker distinct from the producer passes;
5. the theorem ledger still marks P4.3 convergence as open.

This closes P4.1, not P4.3.

## Current Next Step

P4.1 coefficient extraction at `D0` is now represented by:

- symbolic formulas in `phase4_donaldson_coefficients.json`;
- theorem-grade D0 values in `phase4_donaldson_coefficients_values.json`;
- structural/arithmetic checking in `phase4_donaldson_coefficients_check.json`.

The next Phase 4 task is no longer coefficient extraction. It is the global
adiabatic reconstruction theorem: fibrewise Hodge inverse uniformity,
product-space contraction/majorant, and preservation of closedness for the
actual reconstructed `Phi_eps`.
