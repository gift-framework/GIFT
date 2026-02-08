# Research

This folder contains **exploratory research** extending the GIFT framework to open problems in theoretical physics, with particular focus on the **Clay Millennium Problems**.

Unlike `publications/` (peer-reviewed quality) and `docs/` (stable documentation), content here represents **work in progress** with varying levels of validation.

For the full chronological history of this research, see [TIMELINE.md](./TIMELINE.md).
For details on the 2026-02-08 cleanup, see [ARCHIVE_LOG.md](./ARCHIVE_LOG.md).

---

## Current Research Areas

### 1. [yang-mills/](./yang-mills/) — Mass Gap Problem

Investigation of the Yang-Mills mass gap through GIFT's geometric framework.

**Key findings**:
- Spectral gap formula: **lambda_1 x H\* = 13** (where H\* = b2 + b3 + 1 = 99)
- K7 manifold achieves **1.48% deviation** from target (other manifolds: 15-30%)
- Universal constant emerges from G2 holonomy structure

**Status**: VALIDATED — Blind testing confirmed, awaiting formal proof.

**Key documents**:
- `PROGRESS.md` — Summary of current state
- `STATUS.md` — Detailed progression log
- `UNIVERSALITY_CONJECTURE.md` — The main conjecture

**Limitations**: Lean formalization is axiom-heavy; mass gap argument has circular elements.

---

### 2. [riemann/](./riemann/) — Deep Riemann Investigation

Comprehensive investigation of Riemann zeta zeros and GIFT topology, including L-functions, Selberg trace formula, and fractal encoding.

**Key findings**:
- Fibonacci-Riemann recurrence captures trend but not fine structure
- L-function compositional hierarchy validated on real LMFDB data
- Selberg-Fibonacci derivation as analytical bridge
- h_G2^2 = 36 hypothesis **falsified** (documented in `archive/dead-ends/`)

**Status**: EXPLORATORY — Active research with mixed results.

**Key documents**:
- `EXPLORATION_SUMMARY_FEB2026.md` — Executive summary
- `SELBERG_FIBONACCI_DERIVATION.md` — Key analytical result
- `FALSIFICATION_VERDICT.md` — Fibonacci recurrence test battery
- `RIEMANN_FIRST_DERIVATION.md` — Original derivation path

---

### 3. [heegner-riemann/](./heegner-riemann/) — Riemann Hypothesis Connection

Investigation of deep connections between GIFT topology and the Riemann zeta function.

**Key findings**:
- Zeta zeros correspond to topological constants (gamma_1 ~ 14, gamma_2 ~ 21, gamma_20 ~ 77)
- Heegner number 163 = 248 - 8 - 77 = |Roots(E8)| - b3 (**Lean-verified**)
- 100,000 zeros analyzed with 204 matches to GIFT expressions

**Status**: NUMERICALLY VALIDATED — Awaiting theoretical explanation.

**Key documents**:
- `PROGRESS.md` — Summary of current state
- `EXPLORATION_NOTES.md` — Main findings and methodology
- `SELBERG_TRACE_SYNTHESIS.md` — Trace formula connection

---

### 4. [tcs/](./tcs/) — TCS K7 Metric Construction

Complete documentation of the Twisted Connected Sum construction for K7.

**Key findings**:
- 8-phase pathway from ACyl CY3 to spectral bounds
- Explicit G2 metric code (`metric/g2_metric_final.py`)
- Selection constant candidate: kappa = pi^2/14

**Status**: DOCUMENTATION COMPLETE — kappa is candidate, not validated.

**Key documents**:
- `SYNTHESIS.md` — Complete derivation chain
- `GIFT_CONNECTIONS.md` — Link to physical predictions

---

### 5. [spectral/](./spectral/) — Pell Equation Bridge

Analytical derivation connecting number theory to spectral geometry.

**Key findings**:
- **Pell equation**: 99^2 - 50 x 14^2 = 1
- **Continued fraction**: sqrt(50) = [7; 14, 14, ...] = [dim(K7); dim(G2), ...]
- Conjecture: lambda_1 = dim(G2)/H\* is the **unique** Pell-derived solution

**Status**: CONJECTURED — Elegant theoretical argument, awaiting verification.

---

## Active Root Documents

| Document | Description |
|----------|-------------|
| `K7_EXPLICIT_METRIC_SYNTHESIS.md` | Complete K7 metric synthesis (latest, Feb 7) |
| `PRIME_SPECTRAL_K7_METRIC.md` | Prime spectral K7 analysis (latest, Feb 7) |
| `OPERATOR_H_RESULTS_2026-02-02.md` | Banded operator computation results |
| `LEAN_IMPLEMENTATION_PLAN.md` | Lean 4 formalization strategy |
| `PHASE3_WEIL_ROADMAP.md` | Phase 3 Weil trace formula roadmap |
| `RIEMANN_GIFT_CORRESPONDENCES.md` | Riemann-GIFT mathematical connections |
| `SPECTRAL_ANALYSIS.md` | Spectral methodology overview |
| `TORSION_FREE_CONDITION_ANALYSIS.md` | Torsion-free metric conditions |
| `UNIFIED_SPECTRAL_HYPOTHESIS.md` | Unified spectral hypothesis |
| `YM-RH-latest.md` | Yang-Mills / Riemann connection summary |

---

## Supporting Folders

### [notebooks/](./notebooks/)
Computational notebooks and scripts. Includes GPU validation runs (A100), spectral analysis, convergence studies.

### [pattern_recognition/](./pattern_recognition/)
Machine learning for pattern discovery in GIFT relations.

### [tests/](./tests/)
Validation test suite for spectral computations.

---

## Archive

### [archive/](./archive/)
All archived material, organized by category:
- `dead-ends/` — Falsified hypotheses and abandoned approaches (2026-02-08)
- `council-sessions/` — Raw AI review transcripts (2026-02-08)
- `superseded/` — Earlier versions replaced by newer syntheses (2026-02-08)
- `reference-papers/` — External reference PDFs and papers (2026-02-08)
- `notebooks/` — Old notebook versions v1-v9 (2026-01-29)
- `metrics/` — Superseded metric implementations (2026-01-29)
- `spectral-exploratory/` — 32 exploratory docs from spectral/ (2026-01-29)

### [legacy/](./legacy/)
Archived planning documents, sprint reports, and superseded analyses from pre-v3.3.

---

## Status Classifications

| Status | Meaning |
|--------|---------|
| **VALIDATED** | Independently verified (blind testing, multiple methods) |
| **NUMERICALLY VALIDATED** | Computationally verified, awaiting theoretical proof |
| **CONJECTURED** | Proposed formula, strong evidence |
| **EXPLORATORY** | Early investigation, results may change |
| **FALSIFIED** | Hypothesis tested and refuted |
| **ARCHIVED** | Historical record, superseded by newer work |

---

## Clay Millennium Relevance

This research potentially contributes to **two** Clay Millennium Problems:

1. **Yang-Mills Mass Gap**: The spectral gap formula lambda_1 x H\* = 13 provides a candidate for the mass gap on G2-holonomy manifolds.

2. **Riemann Hypothesis**: The correspondence between zeta zeros and topological constants suggests a deep connection between number theory and quantum geometry.

---

## Contributing

Research in this folder should:
1. Clearly state validation status
2. Distinguish proven results from conjectures
3. Document methodology to enable reproduction
4. Reference relevant literature
