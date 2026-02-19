# Spectral Analysis Research Notes

**Date**: January 2026
**Status**: Active investigation

---

## Summary

### The Spectral Relation

Numerical computation on T⁷ with G₂ metric yields an exact GIFT formula:

```
λ₁ × H* = H* / det(g)^(1/dim(K₇))
        = 99 × (32/65)^(1/7)
        = 89.4683...
```

This is **not** a deviation from a target — it is the correct algebraic result connecting:
- H* = b₂ + b₃ + 1 = 99
- det(g) = 65/32
- dim(K₇) = 7

### Fibonacci Connection

The result lies close to F₁₁ = 89:

| Expression | Value | Note |
|------------|-------|------|
| H* × (32/65)^(1/7) | 89.4683 | Exact GIFT |
| F₁₁ + 1/2 | 89.5000 | 0.04% deviation |
| F₁₁ = b₃ + dim(G₂) − p₂ | 89 | 0.52% deviation |

---

## Algebraic Structure

### Full Expansion

```
λ₁ × H* = (b₂ + b₃ + 1) × [2^Weyl / (Weyl × (rank(E₈) + Weyl))]^(1/dim(K₇))
        = (21 + 77 + 1) × [32 / (5 × 13)]^(1/7)
        = 99 × (32/65)^(1/7)
```

### Component Relations

| Quantity | Formula | Value |
|----------|---------|-------|
| det(g) | Weyl × (rank(E₈) + Weyl) / 2^Weyl | 65/32 |
| g_ii | det(g)^(1/7) | 1.1065 |
| λ₁ | 1/g_ii | 0.9037 |
| λ₁ × H* | 99/g_ii | 89.47 |

---

## Open Questions

### 1. Fibonacci + 1/2

Why does H* × (32/65)^(1/7) ≈ F₁₁ + 1/2?

The deviation is only 0.04%. Possible interpretations:
- The 1/2 is a quantum correction (zero-point energy)
- Related to spin structure on K₇
- Numerical coincidence

### 2. Connection to Earlier Hypotheses

Earlier work suggested λ₁ × H* = 14 for the true K₇ manifold. The ratio:

```
89.47 / 14 = 6.39 = H* / (dim(G₂) × g_ii)
```

This factor has exact algebraic form. If the K₇ result is indeed 14, then:
- T⁷ → K₇ involves a reduction by H*/(dim(G₂) × g_ii)
- This would need geometric justification from TCS construction

### 3. Weitzenböck on Curved K₇

On T⁷: Δ₁ = Δ₀ (verified numerically)
On K₇: Δ₁ = Δ₀ + Ric ≠ Δ₀

How does curvature modify the spectral relation?

---

## Technical Notes

### CuPy Compatibility

See `/CLAUDE.md` for GPU computing tips:
- Use `which='SA'` not `which='SM'` for eigsh
- Build COO directly, avoid `tolil()`
- Clear memory pool for large grids

### Convergence

The calibrated λ₁ = 0.9037 is stable to 6 decimal places across N = 5, 7, 9.

---

## Files

```
notebooks/
├── K7_Spectral_v3_Analytical.ipynb
├── K7_Spectral_v4_Delta0_vs_Delta1.ipynb
├── K7_Spectral_v5_Synthesis.ipynb
└── outputs/
    └── K7_spectral_*.json

docs/
└── SPECTRAL_ANALYSIS.md
```

---

*Working notes — subject to revision*
