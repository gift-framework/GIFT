# Stale Artifact Quarantine - 2026-07-09

This file records public artifacts that remain useful as provenance or
comparison material but are no longer the active theorem/certificate path.

## Active Citation Path

Use these for current D0 Level Q bookkeeping:

- `certificates/datum_D0.json`
- `certificates/phase4_donaldson_coefficients_values.json`
- `certificates/phase4_donaldson_coefficients_check.json`
- `certificates/phase4_level_Q_coefficients.json`
- `certificates/phase4_level_Q_coefficients_check.json`
- `certificates/phase4_level_Q_table_check.json`
- `paper/theorem_Q_certified.md`
- `docs/stageD_reconciliation_report_2026_07_09.md`
- `docs/stageD_recap_2026_07_09.md`
- `docs/stageE_recap_2026_07_09.md`

## Quarantined As Historical Or Comparison Only

| Artifact | Reason | Replacement / active path |
| --- | --- | --- |
| `paper/phase3_effective_jacobi_parametrix_candidate.md` | Original Task 4 finite-mode candidate; still not itself the edge-Schauder proof. | Stage D `L1.6_K_Sch` block in `datum_D0.json`, `K_Sch^Maz <= 16/3`. |
| `certificates/phase3_effective_jacobi_parametrix_candidate.json` | Machine output for the historical candidate. | `datum_D0.json`. |
| `paper/phase3_projection_commutator_certificate.md` | Projection convention remains useful, but examples with `K_Sch^Maz <= 17` are stale. | Same formula with Stage D `K_Sch^Maz <= 16/3`. |
| `certificates/phase3_projection_commutator_certificate.json` | Historical constants include the old public envelope. | `datum_D0.json` plus Stage D recaps. |
| `paper/phase3_gamma_src_surrogate.md` | `gamma_src,sur = 3/8` is a legacy surrogate. | Active source constant `C_src = 27/16`. |
| `certificates/phase3_gamma_src_surrogate_proto.json` | Machine output for the surrogate comparison. | Stage D source bookkeeping in `datum_D0.json`. |
| `paper/phase4_adiabatic_operator_form.md` | Power-counting candidate scaffold. | Stage E coefficient values and checks. |
| `certificates/phase4_adiabatic_sources_candidate.json` | Candidate norms, not theorem inputs. | `phase4_donaldson_coefficients_values.json` and Level Q package. |
| `paper/phase4_ar_neumann_budget.md` | Public candidate trail for AR slots. | Private L2 assembly import plus Stage D/E public packaging. |
| `paper/phase4_ar_commutator_slot_candidate.md` | Candidate slot note. | Private theorem-grade slot import recorded by Stage D. |
| `paper/phase4_ar_remaining_slots_candidate.md` | Candidate slot note. | Private theorem-grade slot import recorded by Stage D. |
| `paper/phase4_ar_majorant_candidate.md` | Candidate majorant note. | Stage D L2 assembly and Stage E coefficient package. |

## Rules

- Do not delete these artifacts unless a reproducibility-preserving archive move
  is agreed.
- Do not cite quarantined artifacts as active theorem inputs.
- If a paper or theorem needs a numerical constant now, route the citation
  through the active path above.
- Keep Level E conditional until Phase 5 `(J)` and the compact datum/topology
  wrappers are discharged.
