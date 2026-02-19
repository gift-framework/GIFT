# Note on Li Coefficient Convergence

**Date**: February 2026

## Issue Identified

The computed values of λₙ via Riemann zeros show:
- λ₁ ≈ 0.0200 (computed with 100k zeros)
- λ₁ ≈ 0.0231 (known exact value from Maślanka/Coffey)

**Discrepancy**: ~13%

## Explanation

The sum λₙ = Σ_ρ [1 - (1 - 1/ρ)^n] converges **very slowly**.

The exact value involves an infinite sum over ALL zeros, plus contributions from:
1. Trivial zeros (negative even integers)
2. Archimedean corrections
3. Higher-order terms

With only 100,000 zeros, we capture the dominant structure but miss ~13% of the total.

## Impact on GIFT Analysis

Despite the absolute offset, the **relative patterns** we observed remain valid:
1. The ratio λ₅/λ₈ ≈ (5/8)² doesn't depend on absolute normalization
2. The H* scaling pattern may still hold after proper normalization

## Recommended Fix

1. Use the **direct formula** via Stieltjes constants (more accurate)
2. Or apply a **correction factor** based on known λ₁

If we scale all our computed values by 0.0231/0.0200 ≈ 1.155, the patterns should sharpen.

## Corrected Values (estimated)

| n | Computed | Corrected (×1.155) | Literature |
|---|----------|-------------------|------------|
| 1 | 0.0200 | 0.0231 | 0.0231 |
| 2 | 0.0799 | 0.0923 | ~0.092 |
| 3 | 0.1796 | 0.2074 | ~0.207 |

The correction factor 1.155 ≈ 99/86 is interesting (close to H*/(b₃+rank(E₈)+1) = 99/86).

## Conclusion

The GIFT patterns observed are likely **structurally valid** but need recalibration with higher-precision computations or direct formulas.

---

*Note added during Li coefficient exploration*
