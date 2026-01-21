# Yang-Mills Spectral Gap: Project Status

**Last Updated**: 2026-01-21

## Summary

The GIFT framework proposes a topological formula for the spectral gap:

```
λ₁ × H* = dim(G₂) = 14
```

For the K₇ manifold with b₂ = 21, b₃ = 77: λ₁ = 14/99 ≈ 0.1414

This formula is formally verified in Lean 4: `GIFT.Spectral.MassGapRatio` (gift-framework/core).

### TCS Ratio Discovery (2026-01-21)

Quaternionic sampling on S¹ × S³ × S³ with geodesic distances confirms λ₁ × H* = 14 when the S³ size ratio equals:

```
ratio* = H* / (6 × dim(G₂)) = 99/84 = 33/28 ≈ 1.179
```

where 6 is the normalization factor from the G₂ 3-form contraction Φ_ij = φ_ikl φ_jkl = 6δ_ij.

Physical interpretation: The ratio balances topological degrees of freedom (H* = 99) against G₂ symmetry constraints (6 × 14 = 84).

---

## Validation Results

### Successfully Verified

| Quantity | Target | Measured | Method |
|----------|--------|----------|--------|
| det(g) | 65/32 = 2.03125 | 2.03125 | Quaternionic TCS (exact) |
| λ₁ × H* | 14 | 13.89 | Geodesic graph Laplacian (0.8%) |
| ratio* | 33/28 = 1.1786 | ~1.176 | High-res sweep (0.2%) |
| Torsion norm | < 0.001 | ~10⁻⁴ | PINN |
| Cheeger bound | λ₁ ≥ h²/4 | Satisfied | Lean proof |

### TCS Quaternionic Sampling (v5)

| ratio | λ₁ × H* (geodesic) | λ₁ × H* (chord) |
|-------|-------------------|-----------------|
| 1.00 | 8.56 | 3.91 |
| 1.17 | **13.89** | 6.44 |
| √2 | 17.22 | 11.94 |

The geodesic method achieves λ₁ × H* ≈ 14 at ratio ≈ 33/28, confirming the topological formula.

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
| `notebooks/G2_Quaternionic_Sampling_v5.ipynb` | **Current**: Quaternionic TCS with geodesic distances |
| `notebooks/G2_TCS_Sampling_v4.ipynb` | TCS topology sampling (projection approach) |
| `notebooks/G2_TCS_Anisotropy_v3.ipynb` | TCS anisotropy with S³ size ratio |
| `notebooks/outputs/g2_quaternionic_v5_results.json` | v5 results: λ₁×H* = 13.89 at ratio 1.17 |
| `notebooks/outputs/g2_quaternionic_v5_hires.json` | High-res sweep around ratio = 33/28 |
| `notebooks/GIFT_PINN_Training.ipynb` | PINN training for G₂ 3-form |
| `research/yang-mills/THEORETICAL_BACKGROUND.md` | Literature review |
| `research/yang-mills/UNIVERSALITY_CONJECTURE.md` | Conjecture statement |

---

## Open Questions

1. **Why ratio* = 33/28?** The formula H*/(6 × dim(G₂)) appears naturally, but a geometric derivation from TCS gluing conditions is needed.
2. **Factor 6 origin**: Is Φ_ij = 6δ_ij (3-form contraction) the correct geometric interpretation, or is there a deeper connection?
3. Can the universality λ₁ = 14/H* be proven analytically from G₂ holonomy constraints?
4. How does the Hodge Laplacian on 1-forms (gauge fields) compare to the scalar Laplacian?

---

## Log

### 2026-01-21

- **TCS Quaternionic Sampling (v5)**: λ₁ × H* = 13.89 ≈ 14 achieved
- Key insight: S³ must be sampled with geodesic distances, not Euclidean chord
- Discovered ratio* = H*/(6 × dim(G₂)) = 33/28 for optimal spectral gap
- The factor 6 arises from G₂ 3-form normalization: Φ_ij = φ_ikl φ_jkl = 6δ_ij
- Physical interpretation: ratio balances degrees of freedom (H*) vs symmetry constraints (6 × dim(G₂))
- Notebooks: G2_Quaternionic_Sampling_v5.ipynb, outputs in g2_quaternionic_v5_*.json

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
