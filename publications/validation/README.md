# Statistical Validation for GIFT Framework

This directory archives the v3.3 validation pipeline. The v3.4 statistics refresh was run in core/private (Phase 2 of v3.4 release plan, 2026-04-30):

- `algebraic_montecarlo.py` (3M configs) → log₁₀ p_algebraic = −138, rank 0/3M
- `sensitivity_analysis.py` → r_eff = 15.53, overdetermination 2.13×, 53 strong pairs
- `observable_dataset.py` → 95 observables (35 Type I + 19 II + 21 III + 22 IV)

These newer scripts and results live in the canonical workspace (private repo). For the v3.3 archival pipeline, see [`legacy/v3.3/`](legacy/v3.3/).

## v3.4 Headline Results (2026-04-29)

| Metric | Value |
|--------|-------|
| Type I observables | 35 (exact-target relations) |
| Mean deviation (Type I) | **0.39%** (PDG 2024 / NuFIT 6.0) |
| Total observables | 95 (35 I + 19 II + 21 III + 22 IV) |
| Algebraic null model | log₁₀ p = −138 over 3M+ formulas |
| Lean certificate | 213 conjuncts, 4 main-chain axioms, 0 sorry |

## v3.3 Legacy Pipeline

Archived in [`legacy/v3.3/`](legacy/v3.3/) as the validation snapshot accompanying the v3.3.24 framework release (2026-03-02). 3,070,396-config exhaustive search, 7-component bullet-proof analysis, Westfall-Young maxT, Bayesian comparison across 4 priors. Headline number was 0.24% mean deviation across 32 well-measured observables.

## Full Documentation

See [STATISTICAL_EVIDENCE.md](../references/STATISTICAL_EVIDENCE.md) for the canonical methodology and per-observable breakdown.

---

**Version**: 3.4.13 (2026-04-29)
