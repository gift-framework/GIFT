# Em Dash Cleanup Report

## Summary

Cleaned em dashes (—) from main documentation files. Formulas verified to have proper geometric anchoring.

## Files Cleaned

### publications/v2.1/gift_main.md
- Removed 5 em dashes (lines 79-82, 973)
- Replaced with simple hyphens with spaces

### docs/INTERNAL_RELATIONS_ANALYSIS.md  
- Removed 1 em dash (line 18)

## Formula Consistency: ✓ VERIFIED

All key formulas in publications/v2.1/ have proper geometric anchoring:

| Formula | Status | Example |
|---------|--------|---------|
| δ_CP | ✓ | 7(dim(K₇)) × 14(dim(G₂)) + 99(H*) = 197° |
| m_τ/m_e | ✓ | 7(dim(K₇)) + 2480(10×dim(E₈)) + 990(10×H*) = 3477 |
| Q_Koide | ✓ | dim(G₂)/b₂(K₇) = 14/21 = 2/3 |
| m_s/m_d | ✓ | p₂²(=4) × Weyl_factor(=5) = 20 |
| β₀ | ✓ | π/rank(E₈) = π/8 |
| ξ | ✓ | (Weyl_factor/p₂) × β₀ = 5π/16 |
| H* | ✓ | b₂(K₇) + b₃(K₇) + 1 = 21 + 77 + 1 = 99 |
| Ω_DE | ✓ | ln(2) × (b₂+b₃)/H* = ln(2) × 98/99 |

## Known Issues (Not Fixed)

### gift_technical.md (v2.0 and v2.1)
- 174 em dashes in each version
- **Root cause**: UTF-8 encoding corruption
- Affects multiplication symbols (×), subscripts, mathematical notation
- **Recommendation**: Complete file re-encoding or regeneration needed

### assets/pattern_explorer/reports/*.md
- ~100 em dashes total across reports
- Non-critical exploratory documents
- Can be cleaned if needed

### G2_ML/1.0/PRESENTATION_SUMMARY.md
- 10 em dashes
- Archived/historical document
- Low priority

## Files Verified Clean

✓ README.md
✓ QUICK_START.md  
✓ STRUCTURE.md
✓ CONTRIBUTING.md
✓ CLAUDE.md
✓ All publications/v2.1/supplements/*.md
✓ publications/v2.1/gift_extensions.md
✓ publications/v2.1/GIFT_v21_*.md

## Next Steps

1. Commit cleaned files
2. Consider re-encoding or regenerating gift_technical.md
3. Optional: Clean pattern_explorer reports if needed

---
Date: 2025-11-21
