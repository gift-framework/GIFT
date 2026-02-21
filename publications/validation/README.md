# Statistical Validation for GIFT Framework v3.3

This module implements comprehensive Monte Carlo validation to assess whether the GIFT framework's agreement with experimental data results from genuine topological constraints.

## v3.3 Validation Results

### Exhaustive Search (3,070,396 configs)

| Metric | Value |
|--------|-------|
| Configurations tested | 3,070,396 |
| Better than GIFT | 0 |
| 95% CI (Clopper-Pearson) | [0, 3.7×10⁻⁶] |

### Bullet-Proof Analysis (7 components)

| Component | Result |
|-----------|--------|
| Three null families | All p < 2×10⁻⁵ (σ > 4.2) |
| Westfall-Young maxT | 11/33 significant (global p = 0.008) |
| Pre-registered test split | p = 6.7×10⁻⁵ (σ = 4.0) |
| Bayes factor (4 priors) | 304–4,738 (decisive) |
| ΔWAIC | 550 (GIFT preferred) |
| Robustness | Weight-invariant, no dominating observable |
| Multi-seed replication | 10 seeds, cross-metric consistent |

## Quick Start

```bash
# Bullet-proof validation (7 components, ~15 seconds)
python3 bulletproof_validation_v33.py

# Exhaustive search (3M+ configs, ~5 minutes)
python3 exhaustive_validation_v33.py
```

## Key Findings

1. **GIFT Mean Deviation**: 0.21% total / 0.22% dimensionless only (33 observables)
2. **Zero configurations** out of 3,070,396 beat GIFT
3. **E₈×E₈ uniqueness**: Outperforms all gauge groups by 21×
4. **G₂ necessity**: Calabi-Yau (SU(3)) fails by 11×
5. **Westfall-Young maxT**: 11/33 individually significant after correlation-aware FWER

## Observables Tested (33)

### Core 18
- Structural: N_gen
- Electroweak: sin²θ_W, α_s, λ_H, α⁻¹
- Leptons: Q_Koide, m_τ/m_e, m_μ/m_e
- Quarks: m_s/m_d, m_c/m_s, m_b/m_t, m_u/m_d
- PMNS: δ_CP, θ₁₃, θ₂₃, θ₁₂
- Cosmology: Ω_DE, n_s

### Extended 15
- PMNS sin² form: sin²θ₁₂, sin²θ₂₃, sin²θ₁₃
- CKM: sin²θ₁₂, A_Wolfenstein, sin²θ₂₃
- Bosons: m_H/m_t, m_H/m_W, m_W/m_Z
- Cosmology: Ω_DM/Ω_b, h, Ω_b/Ω_m, σ₈, Y_p
- Leptons: m_μ/m_τ

## Files

```
publications/validation/
├── bulletproof_validation_v33.py        # 7-component bullet-proof validation
├── exhaustive_validation_v33.py         # Exhaustive search (3M+ configs)
├── validation_v33.py                    # Core module (formulas, experimental data)
├── comprehensive_statistics_v33.py      # Advanced statistical tests
├── riemann_rigorous_validation.py       # Riemann-GIFT connection tests
├── bulletproof_validation_v33_results.json  # Bullet-proof results
├── validation_v33_results.json          # Core results
├── GIFT_Statistical_Validation_Report_v33.md # v3.3 report
├── VALIDATION_SUMMARY_v33.md            # v3.3 summary
├── results/                             # Paper-specific results
└── README.md                            # This file
```

## Riemann-GIFT Connection Validation

A separate rigorous validation tests the claimed Riemann zeta zero connection.

```bash
# Run Riemann validation (~15 minutes)
python3 riemann_rigorous_validation.py
```

**8 independent tests** were conducted:

| Test | Result | Finding |
|------|--------|---------|
| Sobol Coefficient Search | PASS | 0/10000 beat GIFT |
| Rational Uniqueness | FAIL | 625 rationals beat 31/21 |
| Lag Space Search | FAIL | GIFT (8,21) ranks #213/595 |
| Fluctuation Analysis | PASS | R²=0.67 on detrended |
| Permutation Test | PASS | Original distinct (14σ) |
| Null Distribution | FAIL | p=0.5 (typical) |
| Bootstrap Stability | FAIL | CV=46% (unstable) |
| R² Decomposition | PASS | 99.9% from trend |

**Verdict**: 4 PASS / 4 FAIL — **WEAK EVIDENCE**

The Riemann connection is documented in [SPECULATIVE_PHYSICS.md](../references/SPECULATIVE_PHYSICS.md) with honest caveats.
The 33 dimensionless predictions do NOT depend on Riemann.

## Full Documentation

See [STATISTICAL_EVIDENCE.md](../references/STATISTICAL_EVIDENCE.md) for:
- Complete methodology
- Per-observable breakdown
- Theoretical selection principles
- Honest caveats

## Requirements

- Python 3.8+
- No external dependencies (uses only stdlib)
- Runtime: ~10 minutes on modern CPU

---

**Version**: 3.3.18 (2026-02-21)
