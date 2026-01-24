# Spectral Analysis Research Notes

**Date**: January 2026
**Status**: Active investigation

---

## Summary of Findings

### Established Results

| Manifold | Metric | λ₁ × H* | Interpretation |
|----------|--------|---------|----------------|
| T⁷ | Euclidean | 99.0 | Baseline (H*) |
| T⁷ | G₂ constant | 89.47 | ≈ F₁₁ (Fibonacci) |
| K₇ (predicted) | G₂ | 12.65 | 9.6% below target 14 |

### Key Formula

```
λ₁(K₇) = λ₁(T⁷) × dim(G₂) / H*
       = (1/g_ii) × (14/99)
       = (32/65)^(1/7) × (14/99)
```

### Algebraic Identities Verified

- H* = 7 × 14 + 1 = 99
- b₂ + b₃ = 7 × 14 = 98
- F₁₁ = b₃ + dim(G₂) − p₂ = 77 + 14 − 2 = 89
- det(g) = 65/32 (from Betti structure)

---

## Open Questions

### 1. The 10% Gap

**Observation**: Predicted K₇ product = 12.65, target = 14

**Possible explanations**:
- Non-constant metric on K₇ (holonomy requires variation)
- TCS gluing introduces spectral shifts
- Target 14 = dim(G₂) is for gauge Laplacian, not scalar
- Additional geometric factors in K₇ construction

### 2. Why Fibonacci?

**Observation**: T⁷ with G₂ metric gives λ₁ × H* ≈ 89 = F₁₁

**Questions**:
- Is this coincidence or structure?
- Does F₁₁ = b₃ + dim(G₂) − p₂ have deeper meaning?
- Connection to modular forms / τ function?

### 3. Δ₀ vs Δ₁ on True K₇

**Observation**: On T⁷, Δ₀ = Δ₁ (Weitzenböck with Ric=0)

**Question**: On K₇ with non-zero curvature, does Δ₁ ≠ Δ₀?

---

## Next Steps

### Short Term

1. [ ] Literature search: spectral geometry of G₂ manifolds
2. [ ] Check if 14/12.65 ≈ 1.107 has algebraic meaning
3. [ ] Investigate torsion effects on spectrum

### Medium Term

1. [ ] Implement TCS mesh generation
2. [ ] Compute spectrum on actual K₇ geometry
3. [ ] Compare Δ₀, Δ₁, Δ₂ on curved background

### Long Term

1. [ ] Analytic approximation of TCS spectrum
2. [ ] Connection to Yang-Mills mass gap
3. [ ] Formalize spectral results in Lean

---

## Technical Notes

### CuPy Compatibility

Documented in `/CLAUDE.md`:
- Use `which='SA'` not `which='SM'` for eigsh
- Build COO directly, no `tolil()`
- Clear memory pool for large grids

### Grid Convergence

Calibrated λ₁ is stable across N=5,7,9 to 6 decimal places.
Grid N=7 (823,543 points) is sufficient for most purposes.

---

## Files

```
notebooks/
├── K7_Spectral_v3_Analytical.ipynb   # Core computation
├── K7_Spectral_v4_Delta0_vs_Delta1.ipynb
├── K7_Spectral_v5_Synthesis.ipynb    # GPU + all identities
└── outputs/
    ├── K7_spectral_v3_results.json
    ├── K7_spectral_v4_results.json
    └── K7_spectral_synthesis_results.json

docs/
└── SPECTRAL_ANALYSIS.md              # Academic summary
```

---

*Working notes - subject to revision*
