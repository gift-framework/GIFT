# Analysis of λ₁ Sweep Results

## Summary

The 1D Laplacian model FAILS to reproduce λ₁ ~ 1/L²:

- Measured slope: α = -0.715 (expected: -2)
- Measured coefficient: c = 1.51 (expected: π² ≈ 9.87)
- Variance: 100% (should be < 10%)

## Root Causes

### 1. Oversimplified Metric

Our metric is essentially:
```
g = dt² + I₄ + dθ² + dψ²
```

This is nearly FLAT - it doesn't capture:
- K3 curvature (Ricci-flat but curved)
- ACyl corrections on the ends
- True TCS gluing structure

### 2. 1D Laplacian Inadequate

We computed:
```
Δf = (1/√g) ∂_t(√g g^{tt} ∂_t f)
```

But the true Laplacian on K7 is 7-dimensional:
```
Δf = Σᵢⱼ (1/√g) ∂_i(√g g^{ij} ∂_j f)
```

The transverse modes on K3 × T² contribute significantly.

### 3. Missing Cross-Section Gap

For λ₁ ~ π²/L² to hold, we need:
```
λ₁(K3 × T²) >> π²/L²
```

With our flat approximation, λ₁(K3 × T²) = 0 (constant mode on torus), which violates this condition.

## What Would Work

### Option A: Full 7D Eigenvalue Problem

Discretize K7 on a 7D mesh and solve the full Laplacian.
- Pros: Correct physics
- Cons: Memory O(N⁷), computationally expensive

### Option B: 3D Reduction (t, θ, ψ)

Keep K3 as fixed background, solve on I × T²:
```
Δ₃D = ∂²/∂t² + ∂²/∂θ² + ∂²/∂ψ²
```

With periodic BC on θ, ψ and Neumann on t.

### Option C: Use Literature Values

The neck-stretching result λ₁ ~ π²/L² is KNOWN to hold for proper TCS (Langlais 2024).
We should trust the theoretical result rather than our numerical toy model.

## Conclusion

Our numerical test is **inconclusive** due to model inadequacy, NOT because the theoretical prediction is wrong.

The formula λ₁ = π²/L² + O(e^{-δL}) remains **plausible** based on:
1. Analytic surgery literature (Langlais, Mazzeo-Melrose)
2. Cheeger inequality bounds
3. Separation of variables heuristic

But it requires a proper TCS metric and full 7D Laplacian to validate numerically.

## Status Update

| Claim | Previous | Now |
|-------|----------|-----|
| λ₁ ~ 1/L² | Heuristic | **Still heuristic** (numerical test inconclusive) |
| c = π² | Assumed | **Unverified** (test failed due to model issues) |
| κ = π²/14 | Candidate | **Still candidate** |

The selection principle κ = π²/14 remains a **conjecture** requiring either:
1. Better numerical computation (proper TCS metric)
2. Analytical proof (neck-stretching theorem)
