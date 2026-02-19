# Canonical Spectral Estimator Strategy

**Date**: January 2026
**Status**: Research Complete — Path to Rigor Identified

---

## The Problem

Previous spectral gap estimates used **tunable k-scaling**:
```
k = 0.366 × √N  → gives λ₁×H* = 14 (Pell)
k = 0.74 × √N   → gives λ₁×H* = 13 (spinor)
```

This is **not rigorous** — we can "dial" k to get any value we want.

**Goal**: Find a **canonical estimator** where N→∞ limit is independent of tuning.

---

## The Solution: Belkin-Niyogi Canonical Scaling

### Theoretical Framework (2008)

For k-NN graph Laplacian on d-dimensional manifold:

| Quantity | Formula | d=7 (K₇) |
|----------|---------|----------|
| Optimal k exponent | 6/(d+6) | **0.462** |
| Convergence rate | N^(-2/(d+6)) | **N^(-0.154)** |
| Bias-variance balance | ε ~ N^(-1/(d+6)) | N^(-0.077) |

### Critical Insight

The **coefficient c** in `k = c × N^0.462` affects finite-N values but **NOT the N→∞ limit**.

```
┌─────────────────────────────────────────────────────────────┐
│  CANONICAL k-SCALING TEST                                   │
│                                                             │
│  k = c × N^0.462   for c ∈ {1, 2, 4, 8}                    │
│                                                             │
│  If theory holds:                                           │
│    lim_{N→∞} λ₁×H* is INDEPENDENT of c                     │
│                                                             │
│  The limit is the TRUE geometric invariant.                 │
└─────────────────────────────────────────────────────────────┘
```

### Comparison with Empirical Scaling

| Method | Exponent | k at N=20k | Note |
|--------|----------|------------|------|
| **Belkin-Niyogi** | 0.462 | 2×N^0.462 = 93 | Canonical |
| Empirical √N | 0.500 | 0.74×141 = 104 | Tuned to get 13 |
| Pell √N | 0.500 | 0.366×141 = 52 | Tuned to get 14 |

**Key**: Theoretical exponent (0.462) is LOWER than √N (0.5).

---

## Four Independent Paths to Rigorous Validation

### Path 1: c-Independence Test (Notebook Ready)

**Method**: `K7_Canonical_Estimator.ipynb`
- Test c ∈ {1, 2, 4, 8} with k = c × N^0.462
- Richardson extrapolation with O(N^-0.154) convergence
- Verify all c values give same limit

**Pass criterion**: Spread of limits < 1 unit

### Path 2: Self-Tuned k-NN Kernels (Cheng & Wu 2022)

**Method**: Local bandwidth σᵢ = distance to k-th neighbor
- NO manual σ parameter
- Proven convergence to weighted Laplace-Beltrami operator
- Automatic density adaptation

**Reference**: "Convergence of Graph Laplacian with kNN Self-tuned Kernels"

### Path 3: Heat Kernel Spectral Method

**Method**: Use heat kernel e^{-tΔ} as t→0
- t-scaling automatically determined
- Connects to manifold Laplacian via spectral theorem
- No ε-tuning

**For d=7**:
```
ε ~ (log N / N)^0.182
λ convergence: O(N^-0.091)
```

### Path 4: Formal Lean Proof (Long-term)

**Required components**:
1. Weyl law formalization (eigenvalue asymptotics)
2. Cheeger inequality bounds (isoperimetric)
3. Graph Laplacian convergence theorem
4. G₂-specific spectral constraints

**Status**: Foundation exists (Joyce theorem, 165+ identities), spectral infrastructure missing.

---

## Expected Results

### If c-Independent Limit Exists

The limit λ₁×H* will be one of:

| Value | Interpretation | Evidence |
|-------|----------------|----------|
| **14** | dim(G₂) | Pell equation: 99²−50×14²=1 |
| **13.5** | (13+14)/2 = dim(J₃(O))/2 | Numerical midpoint |
| **13** | dim(G₂)−h | Parallel spinor correction |

The **numerical finding** λ₁×H* = 13.56 ± 0.04 suggests the true value lies between 13 and 14.

### If Limit Depends on c

This would indicate:
- Graph Laplacian has systematic bias
- Need heat kernel or other advanced methods
- Or: the "spectral gap" is moduli-dependent (metric-dependent)

---

## Implementation Checklist

### Phase 1: Numerical Validation (Current)

- [x] Create canonical estimator notebook (`K7_Canonical_Estimator.ipynb`)
- [ ] Run c-independence test on Colab A100
- [ ] Richardson extrapolation with O(N^-0.154) rate
- [ ] Document limit and spread

### Phase 2: Self-Tuned Validation

- [ ] Implement Cheng-Wu self-tuned k-NN kernel
- [ ] Compare with Belkin-Niyogi results
- [ ] Verify convergence without ANY tuning

### Phase 3: Formal Documentation

- [ ] Update K7_SPECTRAL_GAP_SYNTHESIS.md with canonical results
- [ ] Write rigorous methodology section
- [ ] State theorem with precise conditions

### Phase 4: Lean Formalization (Future)

- [ ] Formalize Weyl law statement
- [ ] Cheeger inequality for K₇
- [ ] Graph → manifold Laplacian convergence
- [ ] Full spectral gap theorem

---

## Key References

### Theory
1. Belkin & Niyogi (2008) — "Towards a theoretical foundation for Laplacian-based manifold methods"
2. Hein, Audibert, von Luxburg (2007) — "Graph Laplacians and their convergence"
3. Cheng & Wu (2022) — "Convergence of Graph Laplacian with kNN Self-tuned Kernels"

### Spectral Geometry
4. Calder & Garcia Trillos (2019) — "Improved spectral convergence rates"
5. Langlais (2024) — "Analysis and Spectral Theory of Neck-Stretching Problems"

### GIFT Internal
6. `K7_SPECTRAL_GAP_SYNTHESIS.md` — Current synthesis
7. `SPECTRAL_RESEARCH_SYNTHESIS.md` — Full research trajectory
8. `PELL_TO_SPECTRUM.md` — Pell equation connection

---

## Conclusion

**The path to rigorous validation is clear**:

1. Use **canonical k-scaling** (k = c × N^0.462)
2. Test **c-independence** of N→∞ limit
3. Apply **Richardson extrapolation** with correct rate O(N^-0.154)
4. Verify with **self-tuned k-NN** (no tuning at all)

If the limit is c-independent and equals a GIFT topological constant (13, 14, or 13.5), then we have **rigorous numerical validation** independent of parameter tuning.

The next step is to **run the canonical estimator notebook** and check if limits converge.

---

*GIFT Framework — Canonical Spectral Estimator Research*
