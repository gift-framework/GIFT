# Analytical Structure of the G2 3-form phi(x) on K7

**GIFT Framework v2.2** - Documentation of the G2 structure extracted from meta-Hodge analysis.

## 1. Decomposition

The G2 3-form on K7 decomposes as:

```
phi(x) = phi_local + phi_global(x)
```

where:
- **phi_local**: Constant canonical G2 form (Bryant-Salamon)
- **phi_global(x)**: Position-dependent TCS (Twisted Connected Sum) modulation

## 2. phi_local: Canonical G2 Structure

The local (fiber) component is the standard G2 3-form on R^7:

```
phi_local = e^{012} + e^{034} + e^{056} + e^{135} - e^{146} - e^{236} - e^{245}
```

### Numerical Coefficients (from v1.6 data)

| Component | (i,j,k) | Mean Value | Sign |
|-----------|---------|------------|------|
| e^{012}   | (0,1,2) | +0.1628    | +    |
| e^{034}   | (0,3,4) | +0.1570    | +    |
| e^{056}   | (0,5,6) | +0.1544    | +    |
| e^{135}   | (1,3,5) | +0.1537    | +    |
| e^{146}   | (1,4,6) | +0.1530    | -    |
| e^{236}   | (2,3,6) | +0.1547    | -    |
| e^{245}   | (2,4,5) | +0.1589    | -    |

**Note**: All coefficients are approximately equal (~0.155), confirming the canonical structure.
The small variance (~10^-5) indicates phi_local is essentially constant over the manifold.

## 3. phi_global(x): TCS Modulation

The global component encodes the TCS gluing geometry and varies with position x = (x0, x1, ..., x6).

### Principal Component Structure

SVD of phi_global reveals:
- **PC0** (sigma=29.4): Dominated by (0,1,2) - (0,1,3) coupling
- **PC1** (sigma=19.5): Dominated by (0,1,6) - (0,1,5) coupling
- **PC2** (sigma=17.3): Dominated by (0,3,6) with (0,1,3) admixture

**Interpretation**: The first coordinate x0 plays a special role, appearing in all dominant components.
This is consistent with x0 being the "base" coordinate in the TCS fibration.

### Polynomial Approximation

phi_global can be approximated as a polynomial in x:

```
phi_global^I(x) = sum_J c^I_J * p_J(x)
```

where p_J are polynomial basis functions:
- Constant: 1
- Linear: x0, x1, ..., x6
- Quadratic: x0^2, ..., x6^2, x0*x1, x0*x2, ...

**Fit Quality**:
- Mean R^2: 0.38 (38% variance explained)
- Best fit: component (0,3,4) with R^2 = 0.78
- Worst fit: some components with R^2 < 0.1

**Dominant Polynomial Terms** (by importance):
1. x0 (coefficient importance: 0.50)
2. x0^2 (importance: 0.28)
3. x0*x1 (importance: 0.19)
4. x0*x6 (importance: 0.18)
5. x0*x5 (importance: 0.17)

**Observation**: x0 dominates both linear and quadratic terms, confirming its special role.

## 4. Effective Analytical Form

Combining the above, an effective analytical ansatz is:

```
phi(x) = phi_0 * (1 + alpha * x0 + beta * x0^2 + ...)
       + cross_terms(x0 * x_j)
```

where phi_0 is the canonical G2 form and alpha, beta are O(1) coefficients.

### Constraints

For a valid G2 structure, phi must satisfy:
1. **Antisymmetry**: phi_{ijk} = -phi_{ikj} etc. (automatically satisfied)
2. **Metric positivity**: g = (1/6) phi^2 gives positive definite metric
3. **Torsion-free**: d(phi) = 0, d(*phi) = 0 (approximately satisfied)

### det(g) Calibration

To achieve det(g) = 65/32, the overall scale of phi must be calibrated:
- For g ~ phi^2, det(g) ~ phi^14
- Scale factor: (65/32)^{1/14} ~ 1.052

## 5. Interpretation for TCS G2 Manifolds

The decomposition phi = phi_local + phi_global has a natural TCS interpretation:

**TCS Structure**: K7 = (S^1 x CY3_L) cup (S^1 x CY3_R)

- **phi_local** (35 modes): Fiber direction Λ^3(R^7) - constant G2 backbone
- **phi_global** (42 modes): Base direction - TCS gluing/profile functions

The 42 global modes arise from:
- b2(CY3_L) + b2(CY3_R) contributions
- Gluing region corrections
- Profile functions varying along the base

## 6. Future Directions

To fully determine phi(x) analytically:

1. **Impose torsion-free conditions**: Minimize ||d(phi)||, ||d(*phi)||
2. **Match TCS asymptotics**: phi should approach standard forms near the gluing region
3. **Verify G2 identities**: Check ||phi||^2_g = 7 throughout K7
4. **Harmonic refinement**: Project onto true harmonic representatives

## 7. Code Reference

The numerical extraction is implemented in:
- `G2_ML/meta_hodge/candidate_library.py`: `collect_b3_77()`, `_extract_global_modes()`
- `G2_ML/meta_hodge/geometry_loader.py`: `phi_to_metric_exact()`, `canonical_g2_phi()`

---

**Version**: 1.0 (2024)
**Data Source**: v1.6 GIFT-calibrated samples (1024 points)
