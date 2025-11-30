# GIFT K7 Lean Certificates

Formal verification of GIFT framework G₂ holonomy constraints.

## Quick Start

```bash
# Install elan (Lean version manager)
curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh

# Build project (downloads Mathlib, takes ~10-20 min first time)
cd lean_project
lake update
lake build
```

## Verified Theorems

### Level 3: Sample Verification (50 Sobol points)
- `det_g_near_target`: det(g) ∈ [2.0305, 2.0318] ≈ 65/32
- `torsion_samples_below_joyce`: max torsion 0.00055 < 0.1

### Level 4b: Global Bound (Effective Lipschitz)
- `global_torsion_below_joyce`: **∀x ∈ [-1,1]⁷, ||T(x)|| < 0.1**
- Method: L_eff = 0.0009, δ = 1.28, bound = 0.00177

### Combined
- `gift_k7_verified`: All constraints satisfied
- `k7_has_torsion_free_g2`: Joyce theorem → torsion-free G₂ exists

## Results Summary

| Constraint | Target | Verified | Status |
|------------|--------|----------|--------|
| det(g) | 65/32 | [2.0305, 2.0318] | ✓ |
| ||T|| samples | < 0.1 | [0.00037, 0.00055] | ✓ |
| ||T|| global | < 0.1 | 0.00177 | ✓ |

## Files

- `lakefile.lean` - Lake build configuration
- `lean-toolchain` - Lean 4 version
- `GIFT/Level4bCertificate.lean` - Main certificate with all theorems

## Colab Alternative

```python
!curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y
!source ~/.elan/env && cd lean_project && lake update && lake build
```
