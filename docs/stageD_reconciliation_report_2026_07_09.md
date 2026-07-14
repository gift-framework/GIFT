# Stage D reconciliation report, 2026-07-09

Scope: Codex branch `codex/gift-work`, reconciled against the private
mission brief `private/docs/codex_stageD_mission_2026_07_09.md`.

## Checklist status

- `certificates/datum_D0.json`: patched/re-pinned for Stage D.
- `gamma_src,sur = 3/8`: superseded for the fixed-Sigma sigma-odd source
  channel; retained only as a lower-root normal-form / alpha1-perp
  regular-sector diagnostic.
- Neumann-budget 16/16 + 41/41 audit: cross-checked against private V/W/X/Y
  theorem-grade slot values.
- `phase3_Pi_obs` protos: pinned against private M-L1.d PDE identification.
- H_global claim scope: current Stage D import records two global slots in the
  private 07-09 brief, `{Construction C.1, L4}`, with `(J)` separate.
- Markdown absolute links: relativized across the public `paper/*.md` files.
- Stage E plan: acknowledged as the next outward-rounded interval
  recomputation track for `phase4_donaldson_coefficients_checker`.

## datum_D0 import

`certificates/datum_D0.json` now carries:

- `K_H_K3 <= 9/20`.
- Phase 4.1 fields `DP1_norm`, `DP2_norm`, `D2P1_norm`, `D3m_norm`,
  `raw_P3_scale`, and `R_threshold`.
- `R_threshold` as a lower admissibility threshold:
  `R >= 3664.0659853300026`.
- `L1.6_K_Sch` block:
  `K_Sch <= 16/3`, headline `q_coeff ~= 0.1681084133`, sharp
  `q_coeff ~= 0.1095943575`, and crude `G_aug < 36.6`.
- `L2_AR_assembly_theorem` block:
  `q_total = 26236977/3200000000 ~= 8.199e-3 < 1/2` on
  `X_AR = X_omega x X_lambda x X_mu x X_Theta` with product-max norm.

The deliberately pending fields remain pending:

- `inj_base`
- `curv_base`
- exact two-sided interval for `A_bulk(alpha_1, alpha_1)`

## Neumann-budget cross-check

Codex pre-Stage-D audit:

- product-max normalization repaired to `K_AR_prod = K_H_K3 * K_F =
  2079/2000`;
- 16/16 inverse-budget checks pass;
- 41/41 Neumann skeleton checks pass;
- all four slots were previously marked candidate/pending.

Private theorem-grade import:

| Slot | Private theorem-grade value | q slot at R_AR=4000 | Margin vs 1/16 |
|---|---:|---:|---:|
| `q_comm` | `222453/40000 ~= 5.561325` | `0.00139033125` | `44.95x` |
| `q_proj` | `65133/20000 ~= 3.25665` | `0.0008141625` | `76.77x` |
| `q_hodge` | `1253637/800000 ~= 1.56704625` | `0.0003917615625` | `159.54x` |
| `q_gauge` | `22839/1250 ~= 18.2712` | `0.0045678` | `13.68x` |

Structural divergence: none found. The expected normalization difference is
exactly the product-max convention; no sign flip or factor-placement mismatch
was detected from the imported formulas.

## Pi_obs pin

The older Codex artifacts remain valid as reduced-model prototypes, but the
active status is now private M-L1.d:

- `Pi_obs^PDE = Pi_{R^{2N}}`;
- `||Pi|| = 1`;
- coker dimension is exactly `2N = 154` at `D0`;
- the old "full PDE identification remains open" wording has been removed
  from the ledger and summary docs.

## Claim scope

For Stage D, use the 2026-07-09 brief as the active claim boundary:

- L1.6 is closed at `D0`.
- L2 assembly is closed at `D0`.
- H_global has two slots: `Construction C.1` and `L4`.
- `(J)` remains separate as the torsion-free perturbation hypothesis.

This report intentionally does not merge or push the branch.

## Stage E

Next planned track: extend `phase4_donaldson_coefficients_checker` from a
structural checker to outward-rounded interval recomputation, targeting Level Q
promotion of the Corollary B adiabatic-source norms.
