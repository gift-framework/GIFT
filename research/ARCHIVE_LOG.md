# Archive Log — Research Folder Cleanup

**Date**: 2026-02-08
**Purpose**: Organize the research folder by removing duplicates, archiving dead ends, and consolidating council session transcripts. All moved files are preserved in `archive/` subdirectories.

---

## Methodology

1. **Dead ends**: Research directions explicitly falsified or abandoned, moved to `archive/dead-ends/`
2. **Council sessions**: Raw AI review transcripts (GPT, Grok, Claude), moved to `archive/council-sessions/`
3. **Superseded files**: Earlier versions replaced by newer syntheses, moved to `archive/superseded/`
4. **Reference papers**: External PDFs/papers, moved to `archive/reference-papers/`

Nothing was deleted. All files remain accessible in their archive locations.

---

## Root-Level Moves

### Dead Ends (`archive/dead-ends/`)

| File | Reason | Key Finding |
|------|--------|-------------|
| `PHASE2B_HONEST_RESULTS.md` | h_G2^2 = 36 hypothesis **falsified** (2026-02-03). Optimal P ~ 20, not 36. | Banded operator works (R^2 > 99%) but G2 structure not confirmed |
| `PROMISING_DIRECTIONS.md` | Post-Phase2B brainstorm (2026-02-03). Most directions were subsequently explored in riemann/ | Documents 5 promising directions after h_G2 falsification |
| `REVERSE_ENGINEERING_LAGS.md` | Lag reverse engineering attempt. Approach was inconclusive | Explored whether GIFT lags {5,8,13,27} emerge naturally |
| `COUNCIL8_TEST_RESULTS.md` | Test results from council phase 8, superseded by PHASE2B findings | Pre-falsification test data |
| `AXIOM_REDUCTION_PLAN.md` | Plan to reduce ~47 axioms; work moved to gift-core repo | Axiom reduction is now tracked in the core Lean repository |

### Council Sessions (`archive/council-sessions/`)

| File | Size | Content |
|------|------|---------|
| `council-7.md` | 2.5 MB | Raw HTML dump of Claude conversation on Riemann/Millennium conditions |
| `council-8.md` | 20 KB | GPT review of operator H results, "order of mission" for tests |
| `council-9.md` | 22 KB | GPT analysis of Fibonacci recurrence formula, critical review |
| `council-20.md` | 17 KB | Grok review of prime-spectral paper for arXiv submission |

### Superseded (`archive/superseded/`)

| File | Superseded By | Reason |
|------|--------------|--------|
| `GIFT_Complete_Analytical_Framework.md` | `K7_EXPLICIT_METRIC_SYNTHESIS.md` | Earlier draft; synthesis doc is more complete (Feb 7) |
| `GIFT_K7_Analytical_Structure.md` | `K7_EXPLICIT_METRIC_SYNTHESIS.md` | Earlier K7 structural analysis, absorbed into synthesis |
| `K7_EXPLICIT_METRIC_ANALYTICAL.md` | `K7_EXPLICIT_METRIC_SYNTHESIS.md` | Analytical metric work, consolidated into synthesis |
| `OPERATOR_H_ROADMAP.md` | `OPERATOR_H_RESULTS_2026-02-02.md` | Roadmap superseded by actual results |

### Reference Papers (`archive/reference-papers/`)

| File | Size | Description |
|------|------|-------------|
| `2404.12114v1.pdf` | 679 KB | arXiv reference paper (PDF) |
| `2404.12114v1.txt` | 74 KB | arXiv reference paper (text extraction) |
| `2505.21192v5.pdf` | 7.5 MB | Large arXiv reference paper |
| `eta.tex` | 153 KB | LaTeX source for eta function analysis (reference material) |

---

## riemann/ Subdirectory Moves

### Council Sessions (`riemann/archive/council-sessions/`)

19 raw council session files, representing AI-assisted review transcripts from the Riemann hypothesis exploration:

| Files | Content |
|-------|---------|
| `council-2.md` through `council-6.md` | Early exploration sessions (5 files) |
| `council-10.md` through `council-15.md` | Mid-phase sessions (6 files) |
| `council-17.md`, `council-18.md`, `council-19.md` | Late sessions (3 files) |
| `council-17-synthesis.md` | Session 17 synthesis |
| `COUNCIL_REPORT_G2_FIBONACCI_RIEMANN.md` | G2/Fibonacci/Riemann council report |
| `COUNCIL_SYNTHESIS.md` | General synthesis |
| `COUNCIL_SYNTHESIS_11.md` | Session 11 synthesis |
| `COUNCIL_SYNTHESIS_12.md` | Session 12 synthesis |

### Superseded (`riemann/archive/superseded/`)

| File | Superseded By | Reason |
|------|--------------|--------|
| `PHASE2_FINDINGS.md` | `PHASE2_COMPLETE_REPORT.md` | Preliminary findings absorbed into complete report |
| `PHASE2_RG_FLOW_DISCOVERY.md` | `PHASE2_COMPLETE_REPORT.md` | RG flow discovery documented in complete report |
| `LI_CONVERGENCE_NOTE.md` | `LI_CRITERION_EXPLORATION.md` | Short note absorbed into full exploration |
| `LMFDB_ACCESS_GUIDE.md` | N/A (operational guide) | One-time setup guide, no longer needed |
| `GIFT_Riemann_ML_Exploration.ipynb` | `GIFT_Riemann_ML_Exploration_results.ipynb` | Results notebook supersedes exploration notebook |

---

## tcs/ Subdirectory Cleanup

### Superseded metric data (`tcs/metric/` → kept in place, documented)

| File | Status | Note |
|------|--------|------|
| `k7_metric_data.json` | Superseded | First version of metric data |
| `k7_metric_v2.json` | Superseded | Second version |
| `k7_metric_final.json` | **CURRENT** | Final version, kept |

These are small JSON files; kept in place for reproducibility but documented as superseded.

---

## heegner-riemann/ Cleanup

### Data versioning (kept in place, documented)

| File | Status | Note |
|------|--------|------|
| `gift_zeta_matches.csv` | Superseded by v2 | Original match data |
| `gift_zeta_matches_v2.csv` | **CURRENT** | Updated match data |
| `training_matches.csv` | Supporting data | Train/test split for validation |
| `holdout_matches.csv` | Supporting data | Holdout set |

---

## Files KEPT at Root Level

These files represent **current, active research**:

| File | Topic | Date/Status |
|------|-------|-------------|
| `K7_EXPLICIT_METRIC_SYNTHESIS.md` | Complete K7 metric synthesis | Feb 7, 2026 — Latest |
| `PRIME_SPECTRAL_K7_METRIC.md` | Prime spectral K7 analysis | Feb 7, 2026 — Latest |
| `OPERATOR_H_RESULTS_2026-02-02.md` | Operator H computation results | Feb 2, 2026 — Current |
| `LEAN_IMPLEMENTATION_PLAN.md` | Lean 4 formalization plan | Current |
| `PHASE3_WEIL_ROADMAP.md` | Phase 3 Weil trace formula roadmap | Current |
| `RIEMANN_GIFT_CORRESPONDENCES.md` | Riemann-GIFT correspondences | Current |
| `SPECTRAL_ANALYSIS.md` | Spectral methodology overview | Current |
| `TORSION_FREE_CONDITION_ANALYSIS.md` | Torsion-free metric analysis | Current |
| `UNIFIED_SPECTRAL_HYPOTHESIS.md` | Unified spectral hypothesis | Current |
| `YM-RH-latest.md` | Yang-Mills / Riemann connection | Current summary |

---

## Summary Statistics

| Category | Files Moved | Size Freed from Root |
|----------|-------------|---------------------|
| Dead ends | 5 | ~25 KB |
| Council sessions (root) | 4 | ~2.6 MB |
| Superseded (root) | 4 | ~44 KB |
| Reference papers | 4 | ~8.4 MB |
| Council sessions (riemann/) | 19 | ~250 KB |
| Superseded (riemann/) | 5 | ~280 KB |
| **Total** | **41** | **~11.6 MB** |
