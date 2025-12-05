# Archived: G2 Machine Learning Module

**Status**: ARCHIVED - December 2025

This directory contains the original neural network implementations for G₂ metric extraction on K₇ manifolds. This work has been **integrated into the production library**.

## Current Location

The K₇ metric pipeline is now maintained in:

**[gift-framework/core](https://github.com/gift-framework/core)** (giftpy v1.2.0+)

```bash
pip install giftpy
```

```python
import gift_core as gc

# Run the full K7 metric pipeline
config = gc.PipelineConfig(neck_length=15.0, resolution=32, use_pinn=True)
result = gc.run_pipeline(config)

print(f"det(g) = {result.det_g}")      # 65/32
print(f"b2 = {result.betti[2]}")       # 21
print(f"b3 = {result.betti[3]}")       # 77
```

## Why This Archive?

The G2_ML development work was moved to a production library to:

1. **Production quality**: Clean API, proper packaging, PyPI distribution
2. **Unified pipeline**: Geometry → G₂ → Harmonic forms → Physics → Verification
3. **Lean/Coq integration**: Certified bounds exportable to proof assistants
4. **Maintainability**: Single source of truth in `giftpy`

## Contents (Historical)

```
G2_ML/
├── G2_Lean/                    # Lean 4 certificates (dev versions)
│   ├── *.lean                  # G2 verification certificates
│   ├── *.ipynb                 # Training notebooks
│   └── *.json                  # Validation results
├── VERSIONS.md                 # Complete version history (v0.1-v2.1)
├── K7_Hodge_b3_77_Validation_Colab.ipynb
└── archived.zip                # All historical versions
```

## Key Results (Validated)

These results are now reproduced by `giftpy`:

| Metric | Value | Status |
|--------|-------|--------|
| det(g) | 65/32 = 2.03125 | Verified |
| κ_T | 1/61 | Verified |
| b₂ | 21 | Verified |
| b₃ | 77 | Verified |
| Banach K | < 0.9 | 35× safety margin |

## Do Not Modify

This archive is preserved for historical reference. All updates should be made to `gift-framework/core`.

## Migration Guide

| Old (G2_ML) | New (giftpy) |
|-------------|--------------|
| `G2_Lean/*.ipynb` | `gc.run_pipeline()` |
| Manual Lean export | `result.certificate.to_lean()` |
| Scattered notebooks | Unified `gc.GIFTPipeline` |

---

*Archived: December 2025*
*Production code: [giftpy](https://pypi.org/project/giftpy/) v1.2.0+*
