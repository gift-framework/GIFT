# Next Steps: From Numerical Observation to Yang-Mills Bridge

**Version**: 1.0
**Date**: January 2026
**Status**: ACTIVE ROADMAP

---

## Executive Summary

The spectral research has established a striking numerical observation:
- **λ₁ × H* ≈ 13** on the canonical TCS metric (ratio ≈ 1.18)
- **λ₁ × H* ≈ 21 = b₂** at ratio ≈ 1.4
- The landscape shows the **ratio acts as an invariant selector**

To progress toward Yang-Mills, we need to:
1. **Calibrate units** (make 13/21 physically meaningful)
2. **Explain the mechanism** (why does ratio select invariants?)
3. **Derive a selection principle** (why would Nature choose a specific ratio?)
4. **Connect to gauge theory** (move from scalar to 1-form Laplacian)

---

## Move #1: Unit Calibration (CRITICAL BLOCKER)

### Problem Statement

Current setup:
- Graph Laplacian eigenvalues μ₁ are in **graph units** (spectrum in [0, 2] for normalized)
- σ (kernel bandwidth) varies with N, k, and sampling
- The numbers 13, 21 are **not in physical units**

Without calibration, we cannot claim λ₁ × H* = dim(Hol) - h in any meaningful sense.

### Solution: σ²-Rescaling Protocol

**For every run, log and report:**

```python
@dataclass
class CalibratedResult:
    # Sampling
    N: int
    k: int
    seed: int

    # Kernel definition
    sigma: float                    # Actual σ used
    sigma_definition: str           # "median_knn", "sqrt_dim_k", etc.
    kernel_type: str                # "gaussian", "adaptive", etc.

    # Laplacian
    laplacian_type: str             # "symmetric", "random_walk", "unnormalized"

    # Raw eigenvalue
    mu1_graph: float                # Raw graph eigenvalue

    # Calibrated eigenvalue
    lambda1_hat: float              # := μ₁ / σ²  (ε-rescaling)

    # Product
    product_raw: float              # μ₁ × H*
    product_calibrated: float       # λ̂₁ × H*
```

### Calibration via Known Manifolds

Run the **exact same pipeline** on manifolds with known spectra:

| Manifold | Exact λ₁ | What to measure |
|----------|----------|-----------------|
| S³ | 3 | C_S3 := λ̂₁(graph) / 3 |
| S⁷ | 7 | C_S7 := λ̂₁(graph) / 7 |
| T⁷ (period 2π) | 1 | C_T7 := λ̂₁(graph) / 1 |

**Calibration factor**: C = average(C_S3, C_S7, C_T7)

Then for K₇: **λ₁(physical) = λ̂₁(graph) / C**

### Implementation

See `spectral_calibrated.py` for the implementation.

**Validation**: After calibration, S³ should give λ₁ ≈ 3, S⁷ should give λ₁ ≈ 7.

---

## Move #2: Mode Localization Analysis

### Problem Statement

The landscape shows λ₁ × H* varies with ratio. **Why?**

Hypothesis: The first eigenmode v₁ **localizes on different factors** depending on ratio:
- ratio ≈ 1.18: mode is "neck-dominant" → sees dim(G₂) - 1 = 13
- ratio ≈ 1.4: mode is "S³₂-dominant" → sees b₂ = 21

### Diagnostic: Participation Ratio

```python
def participation_ratio(v: np.ndarray) -> float:
    """
    PR = 1 / (N × Σᵢ vᵢ⁴)

    - PR → 1/N: fully localized (one site)
    - PR → 1: fully delocalized (uniform)
    """
    v_normalized = v / np.linalg.norm(v)
    return 1.0 / (len(v) * np.sum(v_normalized**4))
```

### Diagnostic: Factor Correlation

For TCS = S¹ × S³₁ × S³₂, decompose v₁:

```python
def factor_correlation(v: np.ndarray, coords: dict) -> dict:
    """
    Measure how much v₁ correlates with each factor.

    coords = {
        's1': array of S¹ coordinates (θ),
        's3_1': array of S³₁ coordinates (q1),
        's3_2': array of S³₂ coordinates (q2)
    }
    """
    v_sq = v**2  # probability distribution

    # Variance on each factor
    var_s1 = weighted_variance(coords['s1'], v_sq)
    var_s3_1 = weighted_variance(coords['s3_1'], v_sq)
    var_s3_2 = weighted_variance(coords['s3_2'], v_sq)

    total = var_s1 + var_s3_1 + var_s3_2

    return {
        's1_fraction': var_s1 / total,
        's3_1_fraction': var_s3_1 / total,
        's3_2_fraction': var_s3_2 / total
    }
```

### Test Protocol

For ratios in [1.0, 1.18, 1.4, 2.0]:
1. Compute (μ₁, v₁)
2. Calculate participation_ratio(v₁)
3. Calculate factor_correlation(v₁, coords)
4. Plot PR vs ratio, factor fractions vs ratio

**Expected outcome**: Clear transition in mode structure around ratio ≈ 1.2-1.4.

---

## Move #3: Canonical Selection Principle

### Problem Statement

Currently, ratio = H*/84 is **ad hoc**. To connect to Yang-Mills, we need:

> **A variational principle that selects the canonical ratio from geometry alone.**

### Candidate Principles

#### 3.1 Torsion Minimization

The TCS construction has residual torsion at the neck. Define:

```
T(ratio) = ∫_M |T|² dvol
```

where T is the torsion tensor of the G₂ structure.

**Test**: Compute T(ratio) numerically. Does ratio ≈ 1.18 minimize torsion?

#### 3.2 Geometric Normalization

Fix Vol(M) = 1 and minimize diameter:

```
ratio* = argmin_r { diam(M) | Vol(M) = 1 }
```

Or the converse: fix diam = 1 and maximize volume.

**Test**: For TCS, compute Vol(ratio) and diam(ratio). Find extrema.

#### 3.3 Spectral Stationarity

The canonical ratio is where λ₁ × H* is **stationary and maximally robust**:

```
ratio* = argmin_r { |∂(λ₁H*)/∂r| } ∩ argmax_r { 1/σ_seeds(λ₁H*) }
```

This is a "fixed point" argument: the physics selects the ratio where the spectral product is insensitive to perturbations.

**Test**: From landscape data, find where ∂(product)/∂(ratio) ≈ 0 AND variance across seeds is minimal.

### Implementation Priority

1. **Spectral stationarity** (can do now with existing landscape data)
2. **Torsion minimization** (requires implementing torsion computation)
3. **Geometric normalization** (requires volume/diameter integrals)

---

## Move #4: From Scalar to 1-Form Laplacian

### Problem Statement

The Yang-Mills mass gap concerns the **gauge field**, not a scalar field. The relevant operator is closer to:

- **Hodge Laplacian on 1-forms**: Δ₁ = dd* + d*d
- **Twisted adjoint Laplacian**: acting on adjoint-valued forms

Currently, all our work uses Δ₀ (scalar Laplacian). This is a **gap in the YM bridge**.

### Hodge Laplacian on 1-Forms

For G₂ manifolds with Ric = 0:
```
Δ₁ = ∇*∇  (Weitzenböck identity simplifies)
```

The spectrum of Δ₁ is related to but distinct from Δ₀.

### Graph Implementation

For a 1-form α on a graph:
- α assigns a value to each **edge** (not vertex)
- The discrete exterior derivative d₀: functions → 1-forms is the incidence matrix
- The Hodge Laplacian is: Δ₁ = d₀ᵀ d₀ + d₁ d₁ᵀ (if we have 2-forms)

For k-NN graph on TCS:

```python
def hodge_laplacian_1forms(W: np.ndarray) -> np.ndarray:
    """
    Construct Hodge Laplacian on 1-forms for weighted graph.

    For a graph with adjacency W:
    - d₀ = incidence matrix (vertices → edges)
    - Δ₁ = d₀ᵀ D⁻¹ d₀  (weighted)

    Returns sparse matrix acting on edge space.
    """
    n = W.shape[0]

    # Build edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if W[i,j] > 0:
                edges.append((i, j, W[i,j]))

    m = len(edges)  # number of edges

    # Incidence matrix d₀: n vertices → m edges
    d0 = np.zeros((m, n))
    edge_weights = np.zeros(m)

    for e, (i, j, w) in enumerate(edges):
        d0[e, i] = +1
        d0[e, j] = -1
        edge_weights[e] = w

    # Weighted Hodge Laplacian
    # Δ₁ = d₀ᵀ d₀ (up-Laplacian, no 2-forms in this simple version)
    L1 = d0.T @ np.diag(edge_weights) @ d0

    return L1, edges
```

### Test Protocol

For several ratios, compute λ₁(Δ₁) and check:
- Does λ₁(Δ₁) × H* show the same pattern as λ₁(Δ₀) × H*?
- Is the "13 regime" more natural/stable for Δ₁?

**If yes**: This strengthens the YM connection significantly.

---

## Implementation Roadmap

```
Week 1: Move #1 (Calibration)
├── [ ] Implement CalibratedResult dataclass
├── [ ] Add σ logging to compute_lambda1()
├── [ ] Run calibration on S³, S⁷, T⁷
├── [ ] Re-run landscape with calibrated units
└── [ ] Verify: S³ → λ₁ ≈ 3, S⁷ → λ₁ ≈ 7

Week 2: Move #2 (Mode Localization)
├── [ ] Implement participation_ratio()
├── [ ] Implement factor_correlation()
├── [ ] Run diagnostic on ratio = {1.0, 1.18, 1.4, 2.0}
├── [ ] Visualize: PR vs ratio, factor fractions
└── [ ] Document mechanism if found

Week 3: Move #3 (Selection Principle)
├── [ ] Analyze existing landscape for stationarity
├── [ ] Implement torsion computation (if feasible)
├── [ ] Test all three candidate principles
└── [ ] Document which principle (if any) selects ratio ≈ 1.18

Week 4: Move #4 (1-Form Laplacian)
├── [ ] Implement hodge_laplacian_1forms()
├── [ ] Run spectral analysis on 1-forms
├── [ ] Compare λ₁(Δ₁) vs λ₁(Δ₀) patterns
└── [ ] Document implications for YM bridge
```

---

## Success Criteria

### Minimum Viable Progress
- [ ] Calibrated units: S³ gives λ₁ ≈ 3 ± 5%
- [ ] Mode localization: Clear PR transition documented
- [ ] Selection principle: At least one principle identifies ratio ≈ 1.18

### Stretch Goals
- [ ] 1-form Laplacian shows cleaner "13 regime"
- [ ] Analytical argument linking selection principle to G₂ geometry
- [ ] Draft section for Yang-Mills paper

---

## Files Created/Modified

| File | Purpose |
|------|---------|
| `NEXT_STEPS.md` | This roadmap |
| `spectral_calibrated.py` | Move #1 implementation |
| `mode_localization.py` | Move #2 implementation |
| `selection_principles.py` | Move #3 implementation |
| `hodge_1forms.py` | Move #4 implementation |

---

## References

1. Coifman & Lafon (2006) - Diffusion maps and ε-scaling
2. Belkin & Niyogi (2003) - Laplacian eigenmaps convergence
3. Singer (2006) - From graph Laplacian to Laplace-Beltrami
4. Joyce (1996) - Compact G₂ manifolds
5. Donaldson (2009) - Gauge theory and G₂ manifolds

---

*This roadmap bridges numerical observation to rigorous Yang-Mills connection.*
