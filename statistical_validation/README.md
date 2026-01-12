# Statistical Validation for GIFT Framework v3.3

This module implements comprehensive Monte Carlo validation to assess whether the GIFT framework's agreement with experimental data results from genuine topological constraints.

## v3.3 Validation Results

| Test | Configurations | Better than GIFT |
|------|----------------|------------------|
| Betti variations (b₂, b₃) | 100,000 | 0 |
| Gauge group comparison | 8 | 0 |
| Holonomy comparison | 4 | 0 |
| Full combinatorial | 91,896 | 0 |
| Local sensitivity | 441 | 0 |
| **Total** | **192,349** | **0** |

**Result**: p-value < 5×10⁻⁶, significance > 4.5σ

## Quick Start

```bash
# Run v3.3 validation (takes ~10 minutes)
python3 validation_v33.py

# Results saved to validation_v33_results.json
```

## Key Findings

1. **GIFT Mean Deviation**: 0.21% across 33 observables
2. **Alternative Mean Deviation**: 32.9%
3. **Zero configurations** out of 192,349 beat GIFT
4. **E₈×E₈ uniqueness**: Outperforms all gauge groups by 10x
5. **G₂ necessity**: Calabi-Yau (SU(3)) fails by 5x

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
statistical_validation/
├── validation_v33.py           # Main validation script
├── validation_v33_results.json # Complete results
├── README.md                   # This file
└── [legacy scripts]            # Previous versions
```

## Full Documentation

See [docs/STATISTICAL_EVIDENCE.md](../docs/STATISTICAL_EVIDENCE.md) for:
- Complete methodology
- Per-observable breakdown
- Theoretical selection principles
- Honest caveats

## Requirements

- Python 3.8+
- No external dependencies (uses only stdlib)
- Runtime: ~10 minutes on modern CPU

---

**Version**: 3.3.0 (2026-01-12)
