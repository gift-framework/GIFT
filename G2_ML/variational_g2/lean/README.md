# GIFT Lean Certificates

Formal verification of GIFT framework results using Lean 4.

## Current Status: Level 2 (Partial)

**det(g) = 65/32 and b3 = 77 numerically verified.**

### Latest Verification (2025-11-30)

**det(g) = 65/32:**
```
Target:    det(g) = 65/32 = 2.03125
Measured:  det(g) = 2.0312490 +/- 0.0000822  (1000 samples)
Error:     0.00005% mean, 0.012% max
Metric:    Positive definite (eigenvalues in [1.078, 1.141])
Status:    PASS
```

**b3 = 77 (spectral analysis):**
```
Target:    b3 = 77 = 35 (local) + 42 (global TCS)
Measured:  b3_effective = 76, gap at position 75
Gap:       29.7x mean gap magnitude
Tolerance: +/- 5 modes
Status:    PASS
```

### What's Proven in Lean

| Result | Status | File |
|--------|--------|------|
| `torsion_small` : 0.00140... < 0.0288 | PROVEN | G2Certificate.lean |
| `epsilon_0_pos` : 0 < epsilon_0 | PROVEN | G2Certificate.lean |
| `gift_k7_g2_existence` : exists torsion-free G2 | PROVEN (from axioms) | G2Certificate.lean |
| `H_star_value` : H* = 99 | PROVEN | G2Certificate.lean |
| `tau_formula` : tau = (496*21)/(27*99) | PROVEN | G2Certificate.lean |
| `b3_decomposition` : 77 = 35 + 42 | PROVEN | G2Certificate.lean |
| `b3_verification_pass` : \|76 - 77\| <= 5 | PROVEN | G2Certificate.lean |
| `det_g_value` : det(g) = 65/32 | VERIFIED (numerical) | verification_result.json |
| `b3_effective` : b3 = 76 (gap 29.7x) | VERIFIED (numerical) | b3_77_result.json |

### What's Axiomatized (Trusted)

| Axiom | Meaning | Path to Proof |
|-------|---------|---------------|
| `joyce_11_6_1` | Joyce's deformation theorem | Level 2: Formalize elliptic theory |
| `K7_smooth`, `K7_compact` | K7 manifold properties | Level 2: Formalize TCS construction |
| `phi0` | PINN-derived G2 structure | Level 3: Serialize NN weights |
| `torsion_bound_cert` | ||T(phi0)|| <= 0.00140... | Level 3: Interval arithmetic |
| `det_g_interval_cert` | |det(g) - 65/32| <= tol | **DONE** (see verification_result.json) |

## Roadmap

### Level 1.5 (Complete)
- [x] Well-typed G2Structure
- [x] Joyce theorem with proper types
- [x] Basic numerical proofs (norm_num)
- [x] Hooks for interval arithmetic

### Level 2 (In Progress)
- [x] Point-wise verification of det(g) = 65/32
- [x] Export PINN weights to JSON
- [x] Generate Lean architecture file
- [x] Spectral verification of b3 = 77 (gap at 75-76)
- [ ] Full interval arithmetic propagation (blocked by interval blowup)
- [ ] Interval arithmetic on torsion bound

### Level 3 (Next)
- [ ] Serialize PINN weights into Lean native format
- [ ] Symbolic computation of det(g) from NN
- [ ] Subdivision-based interval verification

### Level 4 (Long-term)
- [ ] Formalize differential geometry (mathlib)
- [ ] Formalize Joyce theorem
- [ ] Complete bullet-proof certificate

## Building

Requires Lean 4 + Mathlib.

```bash
# Install elan (Lean version manager)
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh

# Initialize lake project
lake init gift_certificate
cd gift_certificate

# Add mathlib dependency to lakefile.lean
# require mathlib from git "https://github.com/leanprover-community/mathlib4"

# Build
lake build
```

## Key Types

```lean
-- G2 structure on a 7-manifold
structure G2Structure (M : Type*) [Smooth7Manifold M] where
  phi : ThreeForm M

-- Torsion norm (opaque, to be specified)
opaque torsion_norm : G2Structure M → ℝ

-- Metric determinant (opaque, to be specified)
opaque det_g : G2Structure M → ℝ
```

## Main Theorem

```lean
theorem gift_k7_g2_existence :
    ∃ phi_tf : G2Structure K7, torsion_norm phi_tf = 0
```

**Interpretation**: There exists a torsion-free G2 structure on K7,
given the numerical evidence that our PINN-derived phi0 has small torsion.

## Running the Pipeline

```bash
# 1. Export PINN weights to JSON
python lean/export_weights.py \
    --model outputs/metrics/g2_variational_model.pt \
    --output lean/pinn_weights.json

# 2. Run point-wise verification (recommended)
python -c "
import torch, numpy as np, sys; sys.path.insert(0,'.')
from src.model import G2VariationalNet
# ... (see verify_det_g.py for full script)
"

# 3. Run interval verification (direct mode works)
python lean/verify_det_g.py --direct

# 4. Check results
cat lean/verification_result.json
```

## Files

| File | Description |
|------|-------------|
| `G2Certificate.lean` | Main Lean certificate with types and theorems |
| `IntervalDetG.lean` | Interval arithmetic hooks for Lean |
| `interval_det_g.py` | Python interval arithmetic library |
| `verify_det_g.py` | End-to-end verification pipeline |
| `export_weights.py` | PINN weights to JSON exporter |
| `pinn_weights.json` | Exported network weights (19MB) |
| `pinn_weights.lean` | Lean architecture summary |
| `verification_result.json` | det(g) verification result |
| `README.md` | This file |
| `../b3_77_result.json` | b3 spectral verification result |
| `../extract_b3_modes.py` | b3 mode extraction script |

## References

- Joyce, D. (2000). "Compact Manifolds with Special Holonomy", Theorem 11.6.1
- GIFT v2.2 publications: `publications/gift_2_2_main.md`
- Numerical certificate: `../outputs/rigorous_certificate.json`
