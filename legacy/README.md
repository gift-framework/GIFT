# Legacy — archived artifacts

Artifacts moved here on 2026-07-14 as the reproducibility-preserving archive
move agreed under the rules of
[`docs/stale_artifact_quarantine_2026_07_09.md`](../docs/stale_artifact_quarantine_2026_07_09.md).
Nothing here is part of the active theorem/certificate path; everything remains
in git history at its original location.

## Conventions

- Directory structure mirrors the original layout: a file that lived at
  `paper/X.md` now lives at `legacy/paper/X.md`.
- **Frozen certificate JSONs are not rewritten.** Provenance strings inside
  archived or active certificates may refer to pre-move paths
  (e.g. `certificates/phase3_gamma_src_surrogate_proto.json`); resolve them
  through the mapping below.
- Generator scripts remain in `scripts/` (they are part of the analytic trail
  and regenerate at the original paths by design).
- Do not cite anything under `legacy/` as an active theorem input. The active
  citation path is listed in the quarantine file.

## Mapping (moved 2026-07-14)

| Original path | Archived path |
| --- | --- |
| `paper/phase3_effective_jacobi_parametrix_candidate.md` | `legacy/paper/phase3_effective_jacobi_parametrix_candidate.md` |
| `paper/phase3_projection_commutator_certificate.md` | `legacy/paper/phase3_projection_commutator_certificate.md` |
| `paper/phase3_gamma_src_surrogate.md` | `legacy/paper/phase3_gamma_src_surrogate.md` |
| `paper/phase4_adiabatic_operator_form.md` | `legacy/paper/phase4_adiabatic_operator_form.md` |
| `paper/phase4_ar_neumann_budget.md` | `legacy/paper/phase4_ar_neumann_budget.md` |
| `paper/phase4_ar_commutator_slot_candidate.md` | `legacy/paper/phase4_ar_commutator_slot_candidate.md` |
| `paper/phase4_ar_remaining_slots_candidate.md` | `legacy/paper/phase4_ar_remaining_slots_candidate.md` |
| `paper/phase4_ar_majorant_candidate.md` | `legacy/paper/phase4_ar_majorant_candidate.md` |
| `certificates/phase3_effective_jacobi_parametrix_candidate.json` | `legacy/certificates/phase3_effective_jacobi_parametrix_candidate.json` |
| `certificates/phase3_projection_commutator_certificate.json` | `legacy/certificates/phase3_projection_commutator_certificate.json` |
| `certificates/phase3_gamma_src_surrogate_proto.json` | `legacy/certificates/phase3_gamma_src_surrogate_proto.json` |
| `certificates/phase4_adiabatic_sources_candidate.json` | `legacy/certificates/phase4_adiabatic_sources_candidate.json` |
| `docs/phase4_progress_2026_07_03.md` | `legacy/docs/phase4_progress_2026_07_03.md` |
| `docs/phase4_consolidation_2026_07_03.md` | `legacy/docs/phase4_consolidation_2026_07_03.md` |
| `docs/base_completion_plan_diff_2026_07_09.md` | `legacy/docs/base_completion_plan_diff_2026_07_09.md` |
