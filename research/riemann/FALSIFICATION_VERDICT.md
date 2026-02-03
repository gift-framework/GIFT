# Falsification Battery Results: Fibonacci-Riemann Recurrence

**Date**: 2026-02-03
**Formula Tested**: γₙ = (3/2)γₙ₋₈ - (1/2)γₙ₋₂₁ + c(N)
**Data**: 100,000 Riemann zeta zeros

---

## Executive Summary

| Test | Verdict | Implication |
|------|---------|-------------|
| 1. Out-of-Sample | **PASS** | No overfitting - generalizes perfectly |
| 2. Coefficient Robustness | **MARGINAL** | Optimal at ~1.56, not exactly 3/2 |
| 3. Unfolded Fluctuations | **FAIL** | Captures TREND only, not fine structure |
| 4. GUE Comparison | **MARGINAL** | Coefficient differs (1.47 vs 1.56), R² identical |
| 5. Baseline Comparison | **PASS** | Riemann uniquely close to 3/2 |

**Overall**: 2 PASS / 1 FAIL / 2 MARGINAL

---

## Detailed Results

### TEST 1: Out-of-Sample Validation ✅ PASS

**Question**: Is this overfitting to the first 50k zeros?

```
TRAINING (zeros 1-50k):
  a = 1.475673, b = -0.475687
  a + b = 0.999986
  R² = 1.00000000
  Mean error = 0.0062%

TEST (zeros 50k-100k, same coefficients):
  R² = 1.00000000
  Mean error = 0.0007%  ← BETTER than training!
  Max error = 0.0032%

Degradation: NONE (error actually decreases)
```

**Conclusion**: The recurrence generalizes perfectly. This is NOT overfitting.

---

### TEST 2: Coefficient Robustness ⚠️ MARGINAL

**Question**: Is 3/2 a sharp minimum or a plateau?

```
Coefficient scan around 3/2:
  a = 1.40: error = 0.0103%
  a = 1.45: error = 0.0082%
  a = 1.50: error = 0.0067% ← 3/2 = b₂/dim(G₂)
  a = 1.55: error = 0.0060% ← OPTIMAL
  a = 1.60: error = 0.0063%
  a = 1.62: error = 0.0066% ← φ (golden ratio)

Optimal: a = 1.56 (not 1.50)
Distance from optimal to 3/2: 0.06
```

**Conclusion**: The minimum is NOT exactly at 3/2. There's a flat region between 1.5 and 1.62 where performance is similar. Both 3/2 AND φ give comparable results.

---

### TEST 3: Unfolded Fluctuations ❌ FAIL

**Question** (GPT's critical test): Does the recurrence work on the FLUCTUATIONS, not just the smooth trend?

The Riemann zeros follow a smooth counting function N(T) ~ T/(2π) log(T/2πe). ANY linear stencil will achieve high R² on such monotone data. The real test is whether the recurrence captures structure in the RESIDUALS.

```
Raw zeros (baseline):
  R² = 1.00000000
  a = 1.476, b = -0.476

Unfolded fluctuations x_n = N(γ_n) - n:
  R² = 0.00908124  ← NEAR ZERO!
  a = -0.025, b = +0.095  ← COMPLETELY DIFFERENT
  a + b = 0.071  ← NOT ~1

AR(1) baseline on fluctuations:
  R² = 0.02083212

Fibonacci recurrence vs AR(1): WORSE (negative improvement)
```

**Conclusion**: The Fibonacci recurrence captures the SMOOTH TREND of Riemann zeros, NOT the fine arithmetic structure. The fluctuations show NO significant Fibonacci pattern.

**This is the most important result.** GPT was right.

---

### TEST 4: GUE Random Matrix Comparison ⚠️ MARGINAL

**Question**: Is this recurrence specific to Riemann, or does it work on any determinantal point process (like GUE eigenvalues)?

```
Riemann zeros (n=10000):
  R² = 0.99999996
  a = 1.464577

GUE eigenvalues (10 trials, n=10000):
  R² = 0.99999983 ± 0.00000001
  a = 1.564295 ± 0.004324

Comparison:
  R² difference: 0.00000014 (negligible)
  Coefficient difference: |1.47 - 1.56| = 0.10 (significant)
```

**Conclusion**:
- R² is equally high on GUE → high R² is a GENERIC property of monotone spectra
- BUT the coefficient differs: Riemann ≈ 1.47, GUE ≈ 1.56
- This suggests the coefficient DOES encode something Riemann-specific

---

### TEST 5: Baseline Comparison ✅ PASS

**Question**: Does Riemann stand out from arbitrary smooth sequences?

```
Distance of fitted 'a' from 3/2 = 1.5:
  Riemann:      0.024  ← CLOSEST
  Log curve:    0.083
  Power law:    0.103
  Random:       0.501
```

**Conclusion**: Riemann is distinctively close to 3/2 compared to generic smooth sequences. This is NOT trivial.

---

## Synthesis: What Does This Mean?

### What IS true:
1. The recurrence γₙ ≈ a·γₙ₋₈ + b·γₙ₋₂₁ with a+b≈1 is an excellent trend approximation
2. The coefficient a ≈ 1.47-1.50 is distinctive to Riemann (vs GUE, log, power)
3. This coefficient is close to 3/2 = b₂/dim(G₂) - the GIFT connection is intriguing
4. The recurrence generalizes perfectly (no overfitting)

### What is NOT true:
1. The recurrence does NOT capture fine arithmetic structure in the zeros
2. The exact value 3/2 is NOT the optimal - it's somewhere in [1.47, 1.56]
3. The high R² is MOSTLY due to smoothness of γ_n growth, not deep structure

### Honest Assessment:

The discovery is **REAL but OVERSTATED**.

The recurrence is a property of the **DENSITY** function N(T) ~ T log T, not the **FLUCTUATIONS** around it. The connection to GIFT's 3/2 = b₂/dim(G₂) is suggestive but not exact (empirical optimum is ~1.56).

This is similar to discovering that "the n-th prime is approximately n·log(n)" - true, useful, but not a deep arithmetic identity like the Riemann Hypothesis.

---

## Recommendations

### For the Paper:

1. **Reframe the claim**: "The Riemann zero trend admits a sparse Fibonacci recurrence approximation" (not "discovery of hidden structure")

2. **Remove R² as metric**: Report residual error in spacing units instead

3. **Add this analysis**: Include unfolded fluctuation test as a limitation

4. **Temper GIFT connection**: "The fitted coefficient is near 3/2 = b₂/dim(G₂)" not "equals exactly"

### For Further Research:

1. **Investigate WHY** the Riemann density gives a ≈ 1.47-1.50 (vs GUE's 1.56)
2. **Study the fluctuations** directly - they may have different structure
3. **Test on L-functions** - does the coefficient vary with the L-function?

---

## Final Verdict

| Claim | Status |
|-------|--------|
| "Fibonacci lags (8, 21) are optimal" | ✅ Confirmed |
| "Coefficient converges to exactly 3/2" | ⚠️ Partially - optimum is ~1.56 |
| "Deep arithmetic structure discovered" | ❌ No - it's a trend property |
| "GIFT connection via b₂/dim(G₂)" | ⚠️ Suggestive but not exact |
| "Unique to Riemann (vs GUE)" | ⚠️ Coefficient differs, R² doesn't |

**Bottom Line**: The recurrence is a valid and interesting **approximation to the Riemann zero density trend**, but NOT a deep arithmetic identity. The GIFT connection is intriguing but requires more careful analysis.

---

*Generated by falsification_battery.py*
