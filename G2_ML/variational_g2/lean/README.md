# GIFT Lean Certificates

Formal verification of GIFT framework results using Lean 4.

## Current Status: Level 1.5

**Well-typed skeleton with axiomatized numerics.**

### What's Proven in Lean

| Result | Status | File |
|--------|--------|------|
| `torsion_small` : 0.00140... < 0.0288 | PROVEN | G2Certificate.lean |
| `epsilon_0_pos` : 0 < epsilon_0 | PROVEN | G2Certificate.lean |
| `gift_k7_g2_existence` : exists torsion-free G2 | PROVEN (from axioms) | G2Certificate.lean |
| `H_star_value` : H* = 99 | PROVEN | G2Certificate.lean |
| `tau_formula` : tau = (496*21)/(27*99) | PROVEN | G2Certificate.lean |

### What's Axiomatized (Trusted)

| Axiom | Meaning | Path to Proof |
|-------|---------|---------------|
| `joyce_11_6_1` | Joyce's deformation theorem | Level 2: Formalize elliptic theory |
| `K7_smooth`, `K7_compact` | K7 manifold properties | Level 2: Formalize TCS construction |
| `phi0` | PINN-derived G2 structure | Level 3: Serialize NN weights |
| `torsion_bound_cert` | ||T(phi0)|| <= 0.00140... | Level 3: Interval arithmetic |
| `det_g_interval_cert` | |det(g) - 65/32| <= tol | Level 3: Interval arithmetic |

## Roadmap

### Level 1.5 (Current)
- [x] Well-typed G2Structure
- [x] Joyce theorem with proper types
- [x] Basic numerical proofs (norm_num)
- [x] Hooks for interval arithmetic

### Level 2 (Next)
- [ ] Interval arithmetic on det(g) = 65/32
- [ ] Interval arithmetic on torsion bound
- [ ] Replace axioms with theorems

### Level 3 (Future)
- [ ] Serialize PINN weights into Lean
- [ ] Symbolic computation of det(g) from NN
- [ ] Full interval verification

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

## Files

- `G2Certificate.lean` - Main certificate with types and theorems
- `README.md` - This file

## References

- Joyce, D. (2000). "Compact Manifolds with Special Holonomy", Theorem 11.6.1
- GIFT v2.2 publications: `publications/gift_2_2_main.md`
- Numerical certificate: `../outputs/rigorous_certificate.json`
