# Yang-Mills Mass Gap: Progress Summary

**Last Updated**: 2026-01-29
**Status**: NUMERICALLY VALIDATED (single manifold) | LEAN: PARTIAL (axiom-heavy)

---

## Core Result

```
λ₁ × H* = 13 = dim(G₂) - 1
```

For K₇ (H*=99): λ₁×H* = **13.19** (1.48% deviation)

---

## What's Proven vs Conjectured

| Claim | Status | Notes |
|-------|--------|-------|
| λ₁×H* ≈ 13 for K₇ | **VALIDATED** | Graph Laplacian, N=5000, blind testing |
| Betti independence | **VALIDATED** | Spread < 2.3×10⁻¹³% |
| det(g) = 65/32 | **VALIDATED** | PINN achieves exact |
| Lean theorem `MassGapRatio` | **AXIOM-HEAVY** | Many prerequisites unproven |
| Universality across G₂ manifolds | **OPEN** | Other manifolds 15-30% off |
| Physical mass scale | **THEORETICAL** | Depends on unvalidated κ |

---

## Known Limitations

1. **Circular argument risk**: The Lean formalization relies on axioms that encode the desired result
2. **Single-manifold validation**: Only K₇ achieves close match; Joyce/Kovalev manifolds far off
3. **Graph ≠ Continuous**: Graph Laplacian doesn't converge to Laplace-Beltrami without true metric
4. **No explicit metric**: Joyce metrics are existence results, not closed forms

---

## Key Files (Post-Cleanup)

| File | Purpose |
|------|---------|
| `notebooks/G2_Universality_v11_Test13.ipynb` | Latest validation (13 vs 14) |
| `notebooks/GIFT_Direct_Method.ipynb` | Direct spectral method |
| `notebooks/Spectral_YangMills_Complete.ipynb` | Complete analysis |
| `STATUS.md` | Detailed log with full history |
| `UNIVERSALITY_CONJECTURE.md` | Open conjecture statement |
| `DEEP_STRUCTURE.md` | Why H*=99 is special |

**Archived**: v1-v9 notebooks, exploratory scripts → `archive/notebooks/`

---

## Open Questions

1. **Why 13, not 14?** Graph vs continuous artifact, or genuine feature?
2. **K₇ uniqueness**: Why does H*=14×7+1=99 achieve the best match?
3. **Universality**: Can we test on other G₂ manifolds with known metrics?

---

## Next Steps

1. **Reduce Lean axioms**: Prove prerequisites instead of assuming them
2. **Alternative validation**: Analytical proof via Cheeger inequality
3. **Physical interpretation**: What is the actual mass scale if λ₁=14/99?

---

*The mass gap formula has strong numerical support for K₇, but the Lean formalization needs significant work to become a true proof rather than an axiom-encoded claim.*
