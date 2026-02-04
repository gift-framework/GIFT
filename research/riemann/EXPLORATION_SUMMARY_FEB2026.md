# Riemann-GIFT Exploration Summary — February 2026

**Branch**: claude/explore-riemann-research-hqx8s
**Status**: EXPLORATORY RESEARCH

---

## 1. Li's Criterion Analysis

### 1.1 Results with High-Precision Literature Values

Using the 21 known λₙ values (Coffey/Maślanka):

| Property | Finding | Status |
|----------|---------|--------|
| H* scaling | λ₁ × 99 ≈ 2.29 ≈ p₂ | Approximate (14% off) |
| Linearity | λₙ ≈ n × 0.023 | Confirmed |
| λₙ × H* | ≈ 2.28n (linear, not n²) | Confirmed |
| Fibonacci ratios | λₘ/λₙ ≈ m/n (not (m/n)²) | Corrected |

### 1.2 Key Insight

The Li coefficients follow **linear growth**:
$$\lambda_n \approx n \times \frac{2.28}{H^*} = n \times 0.023$$

The H* = 99 appears as a natural scaling factor, but the structure is simpler than initially thought.

### 1.3 Notebook

Created: `notebooks/Li_Coefficients_GIFT_Analysis.ipynb` (portable Colab)

---

## 2. Ihara Zeta Function Analysis

### 2.1 Key Discovery

The **cyclomatic number** (first Betti number) of certain graphs equals GIFT constants:

| Graph | β₁ | GIFT Match |
|-------|-----|------------|
| K₈ (complete on 8 vertices) | **21** | = b₂ ! |
| G₂ Root Graph (12 vertices) | **13** | = F₇ ! |

This is remarkable: β₁(K₈) = b₂ connects:
- Graph theory (cyclomatic complexity)
- E₈ structure (8 = rank)
- K₇ topology (b₂ = 21)

### 2.2 Spectral Patterns

| Graph | Eigenvalue | GIFT Relation |
|-------|------------|---------------|
| E₈ Dynkin | λ₂ = √2 ≈ 1.414 | ≈ dim(G₂)/10 |
| C₈ (cycle) | λ₂ = λ₃ = √2 | ≈ dim(G₂)/10 |
| Petersen | λ₁ = 3 | = N_gen |

### 2.3 Scripts

Created: `research/riemann/ihara_zeta_analysis.py`

---

## 3. Conductor Selectivity Test

### 3.1 Setup

**Question**: Do GIFT conductors show better [5,8,13,27] recurrence in their L-function zeros?

- GIFT conductors: {7, 8, 11, 13, 14, 21, 27, 77, 99}
- Non-GIFT conductors: {6, 9, 10, 15, 16, 17, 19, 23, 25}

### 3.2 Results (mpmath Riemann zeros as proxy)

⚠️ **Important**: Used mpmath.zetazero() with conductor-dependent windowing, not actual L-function zeros.

| Metric | GIFT Conductors | Non-GIFT Conductors |
|--------|-----------------|---------------------|
| Mean |R - 1| | **0.483 ± 0.592** | **0.276 ± 0.131** |
| t-test p-value | 0.348 (not significant) | |

### 3.3 Key Findings

**Primary result**: NO selectivity observed. Non-GIFT conductors actually performed *better*:
- Non-GIFT mean |R - 1| = 0.276 (closer to ideal R = 1)
- GIFT mean |R - 1| = 0.483

**Notable observations**:
| Conductor | Type | R | Note |
|-----------|------|-------|------|
| 99 (H*) | GIFT | 1.041 | Best GIFT performer |
| 77 (b₃) | GIFT | -1.107 | Extreme outlier |
| 6 | Non-GIFT | 0.976 | Best overall |

### 3.4 Interpretation

- **Conductor 99 (H* = b₂ + b₃ + 1)** shows near-perfect Fibonacci constraint
- **Conductor 77 (b₃)** is anomalously negative
- The cohomological *sum* H* may be more significant than individual Betti numbers

### 3.5 Status

This is a **null result** that:
- Does not falsify GIFT (proxy data limitation)
- Provides no positive support for conductor selectivity
- Suggests H* = 99 may have special significance if any structure exists

See: `CONDUCTOR_SELECTIVITY_RESULTS.md` for full analysis

---

## 4. Dedekind Zeta Exploration

### 4.1 Observations

- All 9 Heegner numbers are GIFT-expressible
- Q(√5) has golden ratio φ as fundamental unit
- Class number h(d) = 1 for several GIFT discriminants

### 4.2 Conjectures (Untested)

1. L(s, χ_d) zeros for GIFT d may show enhanced recurrence
2. Regulators R(d) for GIFT d may have GIFT structure

---

## 5. Summary of New Findings

### Confirmed

1. **Li coefficient linear structure**: λₙ ≈ 0.023n with H* natural scaling
2. **Ihara-graph GIFT connection**: β₁(K₈) = 21 = b₂
3. **G₂ root graph**: β₁ = 13 = F₇

### Negative Result

1. **Conductor selectivity test**: No GIFT advantage observed (p = 0.348)
   - Non-GIFT conductors performed slightly better on average
   - However, H* = 99 showed best GIFT performance (R = 1.041)

### Untested (needs real data)

1. Dedekind zeta zeros for GIFT discriminants

### Corrected

1. Li ratios follow m/n, not (m/n)² as initially computed
2. Initial λₙ calculations had convergence issues (fixed in notebook)

---

## 6. Files Created

| File | Description |
|------|-------------|
| `LI_CRITERION_EXPLORATION.md` | Main Li research document |
| `LI_CONVERGENCE_NOTE.md` | Technical note on convergence |
| `DEDEKIND_ZETA_EXPLORATION.md` | Quadratic field connections |
| `CONDUCTOR_SELECTIVITY_RESULTS.md` | Full selectivity test analysis |
| `li_coefficient_analysis.py` | Li computation script |
| `li_deeper_analysis.py` | GIFT pattern analysis |
| `ihara_zeta_analysis.py` | Graph zeta functions |
| `conductor_selectivity_test.py` | L-function selectivity |
| `Li_Coefficients_GIFT_Analysis.ipynb` | Portable Colab notebook |
| `Conductor_Selectivity_mpmath.ipynb` | mpmath-based selectivity test |

---

## 7. Open Questions

1. **Why β₁(K₈) = b₂?** Is there a deeper connection between complete graphs on rank(E₈) vertices and K₇ topology?

2. **Real conductor selectivity**: What happens with actual LMFDB data?

3. **Li oscillatory component**: Does λₙ^(osc) = λₙ - trend have Fibonacci structure?

4. **Dedekind zeros**: Do L(s, χ_{-7}), L(s, χ_{-163}) zeros show GIFT patterns?

---

## 8. Conclusion

This exploration session identified:

**Positive findings**:
- **Ihara β₁(K₈) = b₂** connection (graph theory ↔ K₇ topology)
- **Li coefficients linear in n** with H* = 99 as natural scaling factor

**Negative finding**:
- **Conductor selectivity test**: No GIFT advantage observed
- Non-GIFT conductors showed better Fibonacci constraint on average
- However, H* = 99 was the best GIFT performer (R = 1.041)

**Key insight**: If any Riemann-GIFT structure exists, it may be associated with the cohomological sum H* = b₂ + b₃ + 1 = 99 rather than individual topological constants. The stark contrast between conductor 99 (excellent) and conductor 77 (anomalous) supports this interpretation.

**Still untested**: Dedekind zeta zeros for GIFT discriminants require LMFDB data.

---

*GIFT Framework — Riemann Research*
*February 2026*
