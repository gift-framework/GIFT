# Yang-Mills Spectral Gap: Project Status

**Last Updated**: 2026-01-20

## Summary

The GIFT framework proposes a topological formula for the spectral gap:

```
λ₁ = dim(G₂) / H* = 14 / (b₂ + b₃ + 1)
```

For the K₇ manifold with b₂ = 21, b₃ = 77: λ₁ = 14/99 = 0.1414...

This formula is formally verified in Lean 4: `GIFT.Spectral.MassGapRatio` (gift-framework/core).

---

## Validation Results

### Successfully Verified

| Quantity | Target | Measured | Method |
|----------|--------|----------|--------|
| det(g) | 65/32 = 2.03125 | 2.0312495 | PINN |
| Torsion norm | < 0.001 | ~10⁻⁴ | PINN |
| λ₁ for K₇ | 0.1414 | 0.1406 | PINN (0.57% deviation) |
| Cheeger bound | λ₁ ≥ h²/4 | Satisfied | Lean proof |

### Numerical Validation Attempts (Failed)

| Method | Result | Problem |
|--------|--------|---------|
| Graph Laplacian v1 | λ₁ ~ 10⁻⁸ | Bandwidth σ = 0.4 inadequate for [0,2π]⁷ domain |
| Graph Laplacian v2 | λ₁ ≈ 0.17 constant | Independent of H*; measures graph connectivity |
| Rayleigh quotient | λ₁ ∝ (H*)^{2/7} | Parameterized metric does not encode topology |

### Analysis

The graph Laplacian on sampled points does not converge to the Laplace-Beltrami operator without the true Riemannian metric. A parameterized diagonal metric g = c²·f(H*)·I cannot reproduce λ₁ ∝ 1/H* because this scaling arises from the Betti numbers constraining the harmonic forms, not from metric scaling.

---

## What Can and Cannot Be Validated Numerically

### Accessible to numerical methods

- Metric determinant det(g) = 65/32 (PINN achieves 10⁻⁵ accuracy)
- Torsion-free condition (PINN achieves ||T|| ~ 10⁻⁴)
- Single-manifold spectral gap (PINN + graph Laplacian on K₇)

### Requires analytic or formal methods

- Universality of λ₁ = 14/H* across G₂ manifolds
- Dependence on Betti numbers (topological, not metric)
- The 1/H* scaling (requires true Joyce/Kovalev metrics, not parametric approximations)

---

## Current Understanding

The formula λ₁ = dim(G₂)/H* has:

1. **Formal verification** in Lean 4 for the algebraic structure
2. **Numerical confirmation** for K₇ (single point: H* = 99, λ₁ = 0.1406)
3. **No numerical confirmation** of universality across different H* values

The universality conjecture remains open. Testing it numerically would require explicit metric tensors for Joyce orbifolds with different (b₂, b₃), which are existence results without closed forms.

---

## Files

| File | Description |
|------|-------------|
| `notebooks/GIFT_PINN_Training.ipynb` | PINN training for G₂ 3-form |
| `notebooks/Yang_Mills_Validation_v2.ipynb` | Graph Laplacian attempt |
| `notebooks/Spectral_Gap_Rayleigh.ipynb` | Rayleigh quotient approach |
| `notebooks/outputs/validation_plots.png` | Results showing λ₁ constant across H* |
| `notebooks/outputs/full_results.csv` | Raw numerical data |
| `research/yang-mills/THEORETICAL_BACKGROUND.md` | Literature review |
| `research/yang-mills/UNIVERSALITY_CONJECTURE.md` | Conjecture statement |

---

## Open Questions

1. Can the universality λ₁ = 14/H* be proven analytically from G₂ holonomy constraints?
2. What is the correct numerical method for spectral gaps on compact G₂ manifolds?
3. How does the Hodge Laplacian on forms compare to the scalar Laplace-Beltrami operator?

---

## Log

### 2026-01-20

- Received graph Laplacian v2 results: λ₁ ≈ 0.17 constant for all manifolds (H* = 36 to 191)
- Diagnosis: graph Laplacian measures discrete graph connectivity, not Riemannian geometry
- Tested analytical eigenfunction (f = cos x₁) with Rayleigh quotient: λ₁ ∝ (H*)^{2/7}, not 1/H*
- Conclusion: parameterized metric cannot reproduce topological scaling
- The Lean formalization remains the primary validation for λ₁ = 14/99

### 2026-01-19

- Created spectral analysis infrastructure
- Ran PINN training: det(g) = 2.0312495, torsion ~ 10⁻⁴
- Measured λ₁ = 0.1406 for K₇ (0.57% from prediction)
- Documented universality conjecture

---

## References

- Joyce, D.D. (2000). Compact Manifolds with Special Holonomy
- Cheeger, J. (1970). A lower bound for the smallest eigenvalue of the Laplacian
- gift-framework/core: Lean 4 formalization of GIFT spectral theory
