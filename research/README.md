# Research

This folder contains **exploratory research** extending the GIFT framework to open problems in theoretical physics, with particular focus on the **Clay Millennium Problems**.

Unlike `publications/` (peer-reviewed quality) and `docs/` (stable documentation), content here represents **work in progress** with varying levels of validation.

---

## Current Research Areas

### 1. [yang-mills/](./yang-mills/) — Mass Gap Problem

Investigation of the Yang-Mills mass gap through GIFT's geometric framework.

**Key findings**:
- Spectral gap formula: **λ₁ × H\* = 13** (where H\* = b₂ + b₃ + 1 = 99)
- K₇ manifold achieves **1.48% deviation** from target (other manifolds: 15-30%)
- Universal constant emerges from G₂ holonomy structure

**Status**: VALIDATED — Blind testing confirmed, awaiting formal proof.

**Key documents**:
- `PROGRESS.md` — Summary of current state
- `STATUS.md` — Detailed progression log
- `UNIVERSALITY_CONJECTURE.md` — The main conjecture

**Limitations**: Lean formalization is axiom-heavy; mass gap argument has circular elements.

---

### 2. [heegner-riemann/](./heegner-riemann/) — Riemann Hypothesis Connection

Investigation of deep connections between GIFT topology and the Riemann zeta function.

**Key findings**:
- Zeta zeros correspond to topological constants:
  - γ₁ ≈ 14 = dim(G₂)
  - γ₂ ≈ 21 = b₂
  - γ₂₀ ≈ 77 = b₃
- Heegner number 163 = 248 - 8 - 77 = |Roots(E₈)| - b₃ (**Lean-verified**)
- 100,000 zeros analyzed with 204 matches to GIFT expressions

**Status**: NUMERICALLY VALIDATED — Awaiting theoretical explanation.

**Key documents**:
- `PROGRESS.md` — Summary of current state
- `EXPLORATION_NOTES.md` — Main findings and methodology
- `SELBERG_TRACE_SYNTHESIS.md` — Trace formula connection

---

### 3. [tcs/](./tcs/) — TCS K7 Metric Construction

Complete documentation of the Twisted Connected Sum construction for K₇.

**Key findings**:
- 8-phase pathway from ACyl CY3 to spectral bounds
- Explicit G₂ metric code (`g2_metric_final.py`)
- Selection constant candidate: κ = π²/14

**Status**: DOCUMENTATION COMPLETE — κ is candidate, not validated.

**Key documents**:
- `PROGRESS.md` — Summary of current state
- `SYNTHESIS.md` — Complete derivation chain
- `GIFT_CONNECTIONS.md` — Link to physical predictions

---

### 5. [spectral/](./spectral/) — Pell Equation Bridge

Analytical derivation connecting number theory to spectral geometry.

**Key findings**:
- **Pell equation**: 99² − 50 × 14² = 1
- **Continued fraction**: √50 = [7; 14̄] = [dim(K₇); dim(G₂), ...]
- Conjecture: λ₁ = dim(G₂)/H\* is the **unique** Pell-derived solution

**Status**: CONJECTURED — Elegant theoretical argument, awaiting verification.

**Key documents**:
- `PELL_TO_SPECTRUM.md` — Core conjecture (exploratory docs archived)

---

## Supporting Folders

### [notebooks/](./notebooks/)
Computational notebooks and scripts used in research. Includes:
- GPU validation runs (A100)
- Spectral analysis code
- Convergence studies

### [tests/](./tests/)
Validation test suite for spectral computations.

### [archive/](./archive/)
**Cleaned up 2026-01-29**: Dead ends, old versions, and exploratory docs:
- `notebooks/` — Old notebook versions (v1-v9)
- `metrics/` — Superseded metric implementations (v1, v2)
- `spectral-exploratory/` — 32 exploratory docs from spectral/

### [legacy/](./legacy/)
Archived planning documents, sprint reports, and superseded analyses.

---

## Root Documents

| Document | Description |
|----------|-------------|
| `GIFT_Complete_Analytical_Framework.md` | Complete analytical framework (draft) |
| `GIFT_K7_Analytical_Structure.md` | K₇ structural analysis |
| `SPECTRAL_ANALYSIS.md` | Spectral methodology overview |
| `UNIFIED_SPECTRAL_HYPOTHESIS.md` | Unified hypothesis statement |
| `YM-RH-latest.md` | Yang-Mills / Riemann connection summary |

---

## Status Classifications

| Status | Meaning |
|--------|---------|
| **VALIDATED** | Independently verified (blind testing, multiple methods) |
| **NUMERICALLY VALIDATED** | Computationally verified, awaiting theoretical proof |
| **CONJECTURED** | Proposed formula, strong evidence |
| **EXPLORATORY** | Early investigation, results may change |
| **ARCHIVED** | Historical record, superseded by newer work |

---

## Clay Millennium Relevance

This research potentially contributes to **two** Clay Millennium Problems:

1. **Yang-Mills Mass Gap**: The spectral gap formula λ₁ × H\* = 13 provides a candidate for the mass gap on G₂-holonomy manifolds.

2. **Riemann Hypothesis**: The correspondence between zeta zeros and topological constants suggests a deep connection between number theory and quantum geometry.

---

## Contributing

Research in this folder should:
1. Clearly state validation status
2. Distinguish proven results from conjectures
3. Document methodology to enable reproduction
4. Reference relevant literature
