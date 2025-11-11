# QUICK REFERENCE: Complete_G2_Metric_Training_v0_8.ipynb

## Architecture at a Glance
| Component | Configuration | Status |
|-----------|---|---|
| **Manifold** | TCS Neck: [−T,T] × (S¹)² × T⁴ | 7D |
| **Neck length** | T = 24.48 (τ ≈ 3.897) | Per GIFT params |
| **Fiber** | (S¹)² with R = 2π | Gluing circles |
| **Base** | K3-like T⁴ with φ-hierarchy | Hints at complex struct |
| **Boundaries** | ACyl zones ±3.0 units | C² continuity |

## Differential Operators Summary
| Operator | Status | Method | Quality |
|----------|--------|--------|---------|
| **d** (exterior) | Implemented | Finite diff + autodiff | Approximate |
| **δ** (codiff) | NOT explicit | Loss-based enforcement | Implicit |
| **Δ** (Laplacian) | NOT implemented | — | Missing |
| **Hodge star** | Implemented | Metric-weighted scaling | Reasonable |
| **Christoffel** | Implemented | Finite diff ε=1e-4 | Stable |
| **Ricci** | Implemented (v0.8) | From Christoffel | Validation only |

## Neural Networks
| Network | Params | Output | Role |
|---------|--------|--------|------|
| **G2PhiNetwork** | 180k | 35 3-form components | Primary structure |
| **MetricNetwork** | 800k | 28 metric coeffs → 7×7 SPD | Direct metric learning |
| **BoundaryNetwork** | 2 | Exponential decay factors | ACyl transitions |
| **Harmonic2Forms** | 1.05M | 21 × 21-dim forms | b₂=21 extraction |
| **TOTAL** | ~2M | — | Full system |

## Loss Function (8 Components)
| Loss | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Purpose |
|------|--------|--------|--------|---------|---------|
| Torsion | 0.1 | 2.0 | 5.0 | 10.0 | dφ = 0 enforcement |
| Volume | 0.6 | 0.4 | 0.2 | 0.15 | V = (2π)⁷ target |
| Harmonic Ortho | 6.0 | 3.0 | 1.5 | 1.0 | Gram → I |
| Harmonic Det | 3.0 | 1.5 | 0.8 | 0.5 | det(G) ≈ 1 |
| Separation | 2.0 | 1.0 | 0.5 | 0.3 | Diagonal >> off-diag |
| Boundary | 0.05 | 0.5 | 1.5 | 2.0 | φ → 0 at ±T |
| Decay | 0.05 | 0.3 | 0.5 | 0.5 | exp(-γ\|t\|/T) |
| ACyl | 0.0 | 0.1 | 0.6 | 0.8 | C² boundary continuity |

## Cohomology Extraction
### b₂ = 21 (Harmonic 2-Forms)
```
Method:    Network → Gram matrix → Eigenanalysis
Steps:     21 distinct networks → 21×21 inner products
Validation: det(G) ≈ 0.995, ||G - I|| < 0.2
Quality:   18/21 eigenvalues in [0.85, 1.15]
Output:    Eigenvector basis + spectrum visualization
```

### b₃ = 77 (Harmonic 3-Forms)
```
Method:    FFT spectral extraction (grid = 12⁷)
Steps:     1. Sample 12⁷ grid
           2. Compute φ (chunked by t-slices)
           3. FFT each of 35 components
           4. Score 250 candidates by energy
           5. Gram-Schmidt orthogonalization → 77 forms
Time:      ~30-35 min on A100 80GB
Output:    77 orthonormal 3-forms + Gram matrix
```

## Validation Checklist (v0.8)
| Check | Method | Threshold | New in v0.8 |
|-------|--------|-----------|------------|
| Torsion | ||∇φ|| | → 0 | Enhanced loss ramp |
| det(Gram₂₁) | Eigenanalysis | 0.995 ± 0.3 | — |
| Gram error | ||G-I|| | < 0.2 | — |
| **Ricci-flat** | FD Christoffel | \|\|R_ij\|\| < 1e-3 | **NEW** |
| b₂ rank | Eigenvalues | 21 | **NEW validation** |
| b₃ rank | Eigenvalues | 75-77 | **NEW validation** |
| Euler χ | Σ(-1)^k b_k | 0 | **NEW check** |
| Mesh convergence | Multi-resolution | CV < 5% | **NEW** |
| Reproducibility | Multi-seed | torsion CV < 5% | **NEW** |

## Training Curriculum
```
PHASE 1 [0-2k]:    Establish structure (harmonic-heavy)
PHASE 2 [2k-5k]:   Impose torsion + mixed precision
PHASE 3 [5k-8k]:   Refine ACyl + b₃ extraction
PHASE 4 [8k-10k]:  Final polish + convergence
```

## GIFT Parameters Embedded
```
τ = 10416/2673 ≈ 3.897    (neck modulus)
ξ = 5π/16 ≈ 0.982 rad      (HyperKähler rotation)
γ = 511/884 ≈ 0.578        (exponential decay)
φ = (1+√5)/2 ≈ 1.618       (golden ratio)

Topological:
b₂ = 21, b₃ = 77 (K₇ Betti numbers)

Derived:
E₈×E₈ structure: 496 dimensions
J₃(O): 27 dimensions (Jordan algebra)
```

## Critical v0.8 Fixes
1. **Boundary decay**: Fixed U-shape bug (was from boundary distance, now from center)
2. **Torsion computation**: Removed clone() in validation (preserves gradient flow)
3. **Ricci validation**: Finite difference computation with 100-point sampling
4. **Grid b₃=12**: Verified 77-form extraction
5. **Gradient tracking**: coords now requires_grad in Ricci computation

## Key Limitations for "Legit TCS G₂"
- **No true wedge product**: Uses norms only
- **No Hodge decomposition**: Only explicit harmonic forms
- **No codifferential**: Loss-based torsion instead of PDE
- **No Laplacian**: Missing harmonic analysis machinery
- **Metric mapping**: φ → g is ad-hoc (0.1 scaling + regularization)
- **Finite differences**: ε=1e-4 fixed, no adaptivity
- **Validation post-hoc**: Should enforce geometry during training

## Files Generated
```
b2_extraction.png              (Gram matrix, eigenvalues, error)
b2_extraction_results.json     (Numerical metrics)
b2_gram_matrix.npy             (Gram matrix array)

b3_extraction_results.json     (77 forms status)
ricci_curvature.json           (Ricci tensor analysis)
cohomology_validation.json     (Topology checks)
complete_summary.json          (Full results)
```

---

## For Refactoring to "Legit TCS G₂"
### Priority 1: Differential Geometry
- [ ] Implement true ∧ (wedge product)
- [ ] Implement δ (codifferential)
- [ ] Implement Δ = dδ + δd (Laplacian)
- [ ] Hodge decomposition: V = H ⊕ dΩ^{k-1} ⊕ δΩ^{k+1}

### Priority 2: Curvature
- [ ] Full Riemann tensor (not just Ricci)
- [ ] Sectional curvature
- [ ] Ricci-flatness as hard constraint (not just validation)

### Priority 3: Optimization
- [ ] Constrained optimization on metric manifold
- [ ] PDE solver for torsion-free condition
- [ ] Galerkin projection on harmonic basis

### Priority 4: Numerics
- [ ] Adaptive finite difference stencils
- [ ] Higher-order methods (not just forward difference)
- [ ] Spectral methods for smoother manifolds

