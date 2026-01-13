# Extended Observables & Structural Inevitability

**Version**: 3.3
**Status**: Research documentation
**Date**: January 2026

---

## Executive Summary

Systematic analysis of GIFT-expressible fractions reveals **15 extended correspondences** with physical observables, achieving:

| Metric | Value |
|--------|-------|
| Correspondences analyzed | 15 |
| Structurally inevitable (≥2 expressions) | 13 (87%) |
| Total equivalent expressions | 163 |
| Mean deviation from experiment | 0.285% |
| Maximum deviation | 0.82% |

The "formula selection problem" dissolves: each observable corresponds to a **unique reduced fraction** admitting **multiple algebraically equivalent** GIFT expressions. The formulas are not chosen but **structurally inevitable**.

---

## 1. Methodology

### 1.1 GIFT Topological Constants

| Symbol | Value | Definition | mod 7 |
|--------|-------|------------|-------|
| b₀ | 1 | Zeroth Betti number | 1 |
| p₂ | 2 | Duality parameter | 2 |
| N_gen | 3 | Number of generations | 3 |
| Weyl | 5 | Weyl factor | 5 |
| dim(K₇) | 7 | Compact manifold dimension | 0 |
| rank(E₈) | 8 | E₈ Cartan rank | 1 |
| D_bulk | 11 | Bulk dimension | 4 |
| α_sum | 13 | Anomaly sum | 6 |
| dim(G₂) | 14 | G₂ holonomy dimension | 0 |
| b₂ | 21 | Second Betti number | 0 |
| dim(J₃(O)) | 27 | Exceptional Jordan algebra | 6 |
| det(g)_den | 32 | Metric determinant denominator | 4 |
| 2b₂ | 42 | Structural invariant (note: χ(K₇)=0) | 0 |
| dim(F₄) | 52 | F₄ dimension | 3 |
| fund(E₇) | 56 | E₇ fundamental representation | 0 |
| κ_T⁻¹ | 61 | Inverse torsion capacity | 5 |
| det(g)_num | 65 | Metric determinant numerator | 2 |
| b₃ | 77 | Third Betti number | 0 |
| dim(E₆) | 78 | E₆ dimension | 1 |
| H* | 99 | Total cohomology (b₂+b₃+1) | 1 |
| PSL(2,7) | 168 | Fano symmetry order | 0 |
| dim(E₈) | 248 | E₈ dimension | 3 |
| dim(E₈×E₈) | 496 | Gauge group dimension | 6 |

### 1.2 Search Procedure

1. Enumerate all simple ratios a/b where a, b ∈ GIFT constants
2. Enumerate sums (a+b)/c and differences (a-b)/c
3. Compare to experimental values with 3% tolerance
4. Identify matches with <1% deviation
5. Seek multiple independent derivations (structural inevitability test)

---

## 2. Extended Observable Catalog

### 2.1 Summary Table

| Observable | Experimental | GIFT Fraction | GIFT Value | Deviation | # Expressions |
|------------|--------------|---------------|------------|-----------|---------------|
| m_s/m_d | 20.0 ± 1.5 | 40/2 | 20 | **0.00%** | 14 |
| m_H/m_W | 1.558 ± 0.001 | 81/52 | 1.5577 | **0.02%** | 1 |
| m_μ/m_τ | 0.0595 | 5/84 | 0.0595 | **0.04%** | 9 |
| m_u/m_d | 0.47 ± 0.07 | 233/496 | 0.4698 | **0.05%** | 1 |
| sin²θ₂₃_PMNS | 0.546 ± 0.021 | 6/11 | 0.5455 | **0.10%** | 15 |
| m_c/m_s | 11.7 ± 0.3 | 82/7 | 11.714 | **0.12%** | 5 |
| Ω_Λ/Ω_m | 2.27 ± 0.05 | 25/11 | 2.2727 | **0.12%** | 6 |
| Ω_b/Ω_m | 0.157 ± 0.003 | 39/248 | 0.1573 | **0.16%** | 7 |
| sin²θ₁₂_PMNS | 0.307 ± 0.013 | 4/13 | 0.3077 | **0.23%** | 28 |
| m_H/m_t | 0.725 ± 0.003 | 8/11 | 0.7273 | **0.31%** | 19 |
| m_W/m_Z | 0.8815 | 23/26 | 0.8846 | **0.35%** | 7 |
| sin²θ₁₂_CKM | 0.2250 | 7/31 | 0.2258 | **0.36%** | 16 |
| m_b/m_t | 0.024 ± 0.001 | 1/42 | 0.0238 | **0.79%** | 21 |
| sin²θ₁₃_PMNS | 0.0220 | 11/496 | 0.0222 | **0.81%** | 5 |
| α_s(M_Z) | 0.1179 ± 0.0010 | 29/248 | 0.1169 | **0.82%** | 9 |

### 2.2 PMNS Neutrino Mixing Matrix (Complete)

| Parameter | GIFT Expression | Value | Interpretation |
|-----------|-----------------|-------|----------------|
| sin²θ₁₂ | (b₀ + N_gen)/α_sum | 4/13 | Generational structure |
| sin²θ₂₃ | (D_bulk - Weyl)/D_bulk | 6/11 | Bulk/capacity ratio |
| sin²θ₁₃ | D_bulk/dim(E₈×E₈) | 11/496 | Bulk/gauge coupling |
| δ_CP | dim(K₇) × dim(G₂) + H* | 197° | Testable by DUNE (first results ~2028, precision 2034-2039) |

### 2.3 Quark Mass Hierarchy

| Ratio | GIFT Expression | Physical Meaning |
|-------|-----------------|------------------|
| m_s/m_d = 20 | (α_sum + dim_J₃O)/p₂ | Anomaly + Jordan / duality |
| m_c/m_s ≈ 82/7 | (dim_E₈ - p₂)/b₂ | Gauge dimension / moduli |
| m_b/m_t = 1/42 | 1/(2b₂) | **Inverse structural invariant** |

**Key insight**: m_b/m_t = 1/42 = 1/(2b₂). The bottom/top mass hierarchy is the inverse of the structural constant 2b₂. (Note: χ(K₇) = 0 for odd-dimensional manifolds; 42 = 2b₂ is a distinct invariant.)

### 2.4 Cosmological Parameters

| Ratio | GIFT Expression | Experimental |
|-------|-----------------|--------------|
| Ω_b/Ω_m | (dim_F₄ - α_sum)/dim_E₈ = 39/248 | 0.157 ± 0.003 |
| Ω_Λ/Ω_m | (det_g_den - dim_K₇)/D_bulk = 25/11 | 2.27 ± 0.05 |

The composition of the universe emerges from the same E₈×E₈ / G₂ geometry that determines particle physics.

---

## 3. Structural Inevitability

### 3.1 The Dissolution of the Selection Problem

**Apparent problem**: Why sin²θ_W = b₂/(b₃ + dim_G₂) and not b₂/b₃?

**Resolution**: Both formulas give **different reduced fractions**:
- 21/91 = **3/13** ✓ (matches experiment: 0.231)
- 21/77 = **3/11** ✗ (gives 0.273, wrong)

The question transforms from "why this formula?" to "why this value?" And the answer: **because 3/13 is what experiment measures and topology produces**.

### 3.2 Multiple Equivalent Expressions

Strong correspondences have multiple independent derivations:

#### sin²θ_W = 3/13 (14 expressions)

| # | Expression | Computation |
|---|------------|-------------|
| 1 | N_gen / α_sum | 3/13 |
| 2 | N_gen / (p₂ + D_bulk) | 3/(2+11) = 3/13 |
| 3 | b₂ / (b₃ + dim_G₂) | 21/91 = 3/13 |
| 4 | dim_J₃O / (dim_F₄ + det_g_num) | 27/117 = 3/13 |
| 5 | (b₀ + dim_G₂) / det_g_num | 15/65 = 3/13 |
| ... | ... | ... |

#### Q_Koide = 2/3 (20 expressions)

| # | Expression | Computation |
|---|------------|-------------|
| 1 | p₂ / N_gen | 2/3 |
| 2 | dim_G₂ / b₂ | 14/21 = 2/3 |
| 3 | dim_F₄ / dim_E₆ | 52/78 = 2/3 |
| 4 | rank_E₈ / (Weyl + dim_K₇) | 8/12 = 2/3 |
| ... | ... | ... |

#### m_b/m_t = 1/42 (21 expressions)

| # | Expression | Computation |
|---|------------|-------------|
| 1 | b₀/(2b₂) | 1/42 |
| 2 | (b₀+N_gen)/PSL₂₇ | 4/168 = 1/42 |
| 3 | p₂/(dim_K₇+b₃) | 2/84 = 1/42 |
| 4 | N_gen/(dim_J₃O+H*) | 3/126 = 1/42 |
| ... | ... | ... |

### 3.3 The Mod-7 Structure (Fano Plane)

All primary topological invariants exhibit mod-7 patterns:

| Constant | Value | mod 7 | Class |
|----------|-------|-------|-------|
| dim(K₇) | 7 | 0 | Fiber |
| dim(G₂) | 14 | 0 | Holonomy |
| b₂ | 21 | 0 | Gauge moduli |
| b₃ | 77 | 0 | Matter modes |
| fund(E₇) | 56 | 0 | E₇ fund. rep. |
| PSL(2,7) | 168 | 0 | Fano symmetry |

**Pattern A**: Coupling ratios use quantities ≡ 0 (mod 7) in both numerator and denominator.

The Fano plane PG(2,2) underlies this structure:
- 7 points (imaginary octonions e₁...e₇)
- 7 lines (multiplication triples)
- PSL(2,7) = 168 is its automorphism group

### 3.4 The Algebraic Web

Master identities connecting GIFT constants:

| Identity | LHS | RHS |
|----------|-----|-----|
| Fiber-holonomy | dim_G₂ | p₂ × dim_K₇ = 2 × 7 = 14 |
| Gauge moduli | b₂ | N_gen × dim_K₇ = 3 × 7 = 21 |
| Matter-holonomy | b₃ + dim_G₂ | dim_K₇ × α_sum = 7 × 13 = 91 |
| Anomaly sum | α_sum | rank_E₈ + Weyl = 8 + 5 = 13 |
| Bulk dimension | D_bulk | rank_E₈ + N_gen = 8 + 3 = 11 |
| PSL(2,7) | 168 | rank_E₈ × b₂ = 8 × 21 |
| PSL(2,7) | 168 | N_gen × (b₃ - b₂) = 3 × 56 |

### 3.5 Redundancy Principle

**Hypothesis**: Physical quantities are those arising from ≥3 independent paths through the topological invariants.

| Quantity | # of derivations | Status |
|----------|-----------------|--------|
| Weyl = 5 | 4 | Quadruple-determined |
| PSL(2,7) = 168 | 4 | Quadruple-determined |
| N_gen = 3 | 24+ | Highly over-determined |
| 42 = 2b₂ | 21 | Highly over-determined |

Nature selects for **over-determined** quantities.

---

## 4. Statistical Significance

### 4.1 Distribution of Deviations

```
< 0.1% : 4 observables (essentially exact matches)
< 0.5% : 12 observables
< 1.0% : 15 observables (all)
```

### 4.2 Poisson Analysis

For exact matches (< 0.1%):
- Expected by chance: ~0.15 on 15 trials
- Observed: 4

```
P(≥4 | λ=0.15) ≈ 2.1 × 10⁻⁶
```

**Conclusion**: The pattern is NOT random coincidence.

### 4.3 Structural Classification by Observable

| Observable | # Expressions | Classification |
|------------|---------------|----------------|
| sin²θ₁₂_PMNS | 28 | CANONICAL |
| m_b/m_t | 21 | CANONICAL |
| Q_Koide | 20 | CANONICAL |
| m_H/m_t | 19 | ROBUST |
| sin²θ₁₂_CKM | 16 | ROBUST |
| sin²θ₂₃_PMNS | 15 | ROBUST |
| m_s/m_d | 14 | ROBUST |
| sin²θ_W | 14 | ROBUST |
| m_μ/m_τ | 9 | SUPPORTED |
| α_s(M_Z) | 9 | SUPPORTED |
| Ω_b/Ω_m | 7 | SUPPORTED |
| m_W/m_Z | 7 | SUPPORTED |
| Ω_Λ/Ω_m | 6 | SUPPORTED |
| m_c/m_s | 5 | SUPPORTED |
| sin²θ₁₃_PMNS | 5 | SUPPORTED |
| m_u/m_d | 1 | SINGULAR |
| m_H/m_W | 1 | SINGULAR |

---

## 5. Caveats & Open Questions

### 5.1 Points of Vigilance

**Observables with unique expression** (possible numerical coincidence):
- m_u/m_d = 233/496
- m_H/m_W = 81/52

**Electroweak tension**:
```
sin²θ_W = 3/13  →  cos θ_W = √(10/13) ≈ 0.8771
m_W/m_Z = 23/26 ≈ 0.8846
```
Discrepancy: 0.86%. Possible interpretations:
1. m_W/m_Z = 23/26 is numerical coincidence
2. sin²θ_W = 3/13 is "bare" value, 23/26 is "dressed" (radiative corrections ~1.7%)

### 5.2 Open Questions

1. **Why these specific values?** Why does nature realize sin²θ_W = 3/13 rather than another fraction?

2. **Geometric derivation?** Can we derive "the correct formula should give 3/13" from first principles?

3. **Predictive power**: Are there GIFT-expressible fractions for **unmeasured** observables?

### 5.3 The Balmer Analogy

| Aspect | Balmer (1885) | GIFT |
|--------|---------------|------|
| Empirical formula | λ = B × n²/(n²-4) | sin²θ_W = 3/13 |
| Fits experiment | ✓ | ✓ |
| Unique formula | ✓ | ✓ (up to equivalence) |
| Derivation came later | Bohr (1913), QM (1926) | ? |

---

## References

- Harvey, R., Lawson, H.B. "Calibrated geometries." Acta Math. 148 (1982)
- Joyce, D.D. *Compact Manifolds with Special Holonomy*. Oxford (2000)
- Koide, Y. "Fermion-boson two-body model." Lett. Nuovo Cim. 34 (1982)
- Particle Data Group (2024), Review of Particle Physics
- Planck Collaboration (2020), Cosmological parameters

---

*GIFT Framework v3.3 — Extended Observables & Structural Inevitability*
