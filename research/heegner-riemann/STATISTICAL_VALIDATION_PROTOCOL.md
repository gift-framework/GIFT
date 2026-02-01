# Statistical Validation Protocol for GIFT-Zeta Correspondences

## Pre-Registration Document

**Date**: 2026-01-24
**Status**: PROTOCOL — To be executed before further analysis
**Purpose**: Establish rigorous standards to avoid confirmation bias

---

## 1. Problem Statement

We have observed 204 correspondences between Riemann zeta zeros (γₙ) and GIFT topological constants (C) with precision < 0.5%. The question is:

**Is this statistically significant, or could it arise by chance?**

---

## 2. Pre-Registered Hypotheses

### 2.1 Primary Hypothesis (H1)

**GIFT constants appear in the zeta zero sequence more often than expected by chance.**

Operationalization:
- Define the set of GIFT constants G = {14, 21, 77, 99, 163, 240, 248, ...}
- For each C ∈ G, find the closest zeta zero γₙ
- Measure precision: p_C = |γₙ - C| / C
- Compare the distribution of {p_C} to a null model

### 2.2 Secondary Hypothesis (H2)

**Multiples of dim(K₇) = 7 are preferentially matched.**

Operationalization:
- Count matches for targets 7k (k = 3, 4, ..., 200) with precision < 0.2%
- Compare to expected rate under null model

### 2.3 Tertiary Hypothesis (H3)

**The correspondence improves with "tier" (topological importance).**

Operationalization:
- Tier 1: Direct topological constants (dim(G₂), b₂, b₃, H*, dim(E₈))
- Tier 2: Derived constants (Heegner numbers, b₃-b₂, etc.)
- Tier 3-4: Combinations
- Test if precision(Tier 1) < precision(Tier 2) < ...

---

## 3. Null Model

### 3.1 Definition

Under the null hypothesis, the zeta zeros are "random" in the sense that their fractional parts {γₙ} are uniformly distributed modulo 1.

**Null Model**: For a target C, the closest zero γₙ satisfies:
```
|γₙ - C| ~ Uniform(0, Δ/2)

where Δ ≈ 2π/log(C) is the average gap at height C
```

### 3.2 Expected Precision Under Null

For target C with average gap Δ:
```
E[precision] = E[|γₙ - C|/C] ≈ Δ/(4C) = π/(2C × log(C))
```

For C = 100: E[precision] ≈ 0.34%
For C = 248: E[precision] ≈ 0.23%

**Critical observation**: Small precision values are EXPECTED for large targets!

### 3.3 Multiple Testing Correction

With k = 81 GIFT constants tested:
- Bonferroni threshold: α/k = 0.05/81 ≈ 0.0006
- Benjamini-Hochberg FDR: sort p-values, accept if p_i < (i/k) × α

---

## 4. Test Design

### 4.1 Test 1: Permutation Test

**Procedure**:
1. For each GIFT constant C, compute precision p_C to nearest zero
2. Generate N = 10,000 random permutations of the zeros (preserving gaps)
3. Compute p_C for each permutation
4. p-value = proportion of permutations with p_C ≤ observed

**Output**: Individual p-values for each GIFT constant

### 4.2 Test 2: Combined Fisher Test

**Procedure**:
1. Compute individual p-values {p₁, ..., p_k} from Test 1
2. Fisher's combined statistic: χ² = -2 Σ log(pᵢ)
3. Under null: χ² ~ χ²(2k)
4. Combined p-value from χ² distribution

**Already computed**: χ² = 32.7, df = 18, p ≈ 0.018

### 4.3 Test 3: Holdout Validation

**Design**:
- Training set: zeros 1-100,000 (already analyzed)
- Holdout set: zeros 100,001-200,000 (NOT YET ANALYZED)

**Procedure**:
1. Use training set to identify GIFT constants with precision < 0.1%
2. Pre-register these as "confirmed correspondences"
3. Predict that the same pattern will appear in holdout set
4. Analyze holdout set and compare

**Key targets for holdout** (from training):
- γ₆₀ ≈ 163 (0.019%) — predict continuation
- γ₁₀₇ ≈ 248 (0.041%) — predict continuation
- Multiples of 7 — predict 80%+ match rate continues

### 4.4 Test 4: Random Baseline Comparison

**Procedure**:
1. Generate 1000 random sequences of 100,000 "pseudo-zeros"
   - Same density as actual zeros: N(T) ~ (T/2π) log(T/2π)
   - Same local statistics (gap distribution)
2. Run the full GIFT matching on each random sequence
3. Compare number of matches to actual data

**Threshold**: If actual matches exceed 95th percentile of random, reject null.

---

## 5. Pre-Registered Predictions

### 5.1 Specific Numerical Predictions

Before analyzing zeros 100,001-200,000, we predict:

| Target | Predicted Index (approx) | Precision |
|--------|--------------------------|-----------|
| 496 | γ₂₆₈ | < 0.1% (already verified) |
| 504 = 72 × 7 | γ₂₇₄ (±5) | < 0.5% |
| 749 = 107 × 7 | ~γ₄₆₀ | < 0.3% |
| 1001 = 143 × 7 | ~γ₆₅₀ | < 0.3% |
| 1379 = 197 × 7 | ~γ₉₄₀ | < 0.3% |

### 5.2 Pattern Predictions

1. **Multiples of 7**: At least 80% of n×7 (for n = 200-300) will have a match with precision < 0.3%

2. **Exceptional Lie dimensions**: dim(E₆)=78, dim(E₇)=133, dim(E₈)=248 will all have matches < 0.5%

3. **Heegner numbers**: All 9 Heegner numbers {1,2,3,7,11,19,43,67,163} will have matches < 1%

---

## 6. Success Criteria

### 6.1 Strong Evidence (publish-worthy)

- Combined Fisher p-value < 0.01 (currently p ≈ 0.018)
- Holdout prediction accuracy > 80%
- At least 3 individual GIFT constants with p < 0.001

### 6.2 Moderate Evidence (continue research)

- Combined p-value < 0.05
- Holdout prediction accuracy > 60%
- Clear pattern in multiples of 7

### 6.3 Weak Evidence (reconsider hypothesis)

- Combined p-value > 0.1
- Holdout prediction accuracy < 50%
- Random baseline produces similar match rate

---

## 7. Protocol Timeline

### Phase 1: Finalize Protocol (Today)
- [x] Define hypotheses
- [x] Specify null model
- [x] Pre-register predictions
- [ ] Get external review (optional but recommended)

### Phase 2: Holdout Analysis (Next)
- [ ] Obtain zeros 100,001-200,000 from Odlyzko tables
- [ ] Run matching algorithm (same code as training)
- [ ] Compare to pre-registered predictions
- [ ] Compute holdout p-values

### Phase 3: Random Baseline (Parallel)
- [ ] Generate 1000 random pseudo-zero sequences
- [ ] Run matching on each
- [ ] Compute match distribution
- [ ] Compare actual data to distribution

### Phase 4: Report
- [ ] Document all results honestly
- [ ] Include negative findings
- [ ] Discuss limitations
- [ ] Suggest follow-up experiments

---

## 8. Potential Confounds

### 8.1 Look-Elsewhere Effect

We searched for patterns and found matches. This inflates apparent significance.

**Mitigation**: Pre-registration, holdout test, multiple testing correction.

### 8.2 Floating Point Issues

Precision calculations might have numerical errors.

**Mitigation**: Use high-precision arithmetic (128-bit floats), verify against published tables.

### 8.3 Target Selection Bias

GIFT constants were chosen because they have nice mathematical properties. Small integers naturally cluster near early zeros.

**Mitigation**: Compare to random integer targets with similar distribution.

### 8.4 Gap Distribution

Zeta zeros have structured gaps (GUE statistics), not uniform. This affects expected precision.

**Mitigation**: Use actual gap distribution in null model, not uniform.

---

## 9. Code for Validation

```python
import numpy as np
from scipy import stats

def permutation_test(zeros: np.ndarray, target: float, n_perms: int = 10000) -> float:
    """Permutation test for target-zero correspondence."""
    # Find closest zero to target
    diffs = np.abs(zeros - target)
    observed_precision = np.min(diffs) / target

    # Generate permutations (shift zeros preserving gaps)
    gaps = np.diff(zeros)
    null_precisions = []

    for _ in range(n_perms):
        # Random circular shift
        shift = np.random.randint(len(zeros))
        shifted_zeros = zeros + zeros[shift] - zeros[0]

        # Find closest to target
        diffs = np.abs(shifted_zeros - target)
        null_precisions.append(np.min(diffs) / target)

    # p-value
    p_value = np.mean(np.array(null_precisions) <= observed_precision)
    return p_value

def fisher_combined(p_values: list) -> float:
    """Fisher's combined probability test."""
    chi2 = -2 * np.sum(np.log(p_values))
    df = 2 * len(p_values)
    return 1 - stats.chi2.cdf(chi2, df)

def holdout_validation(training_matches: dict, holdout_zeros: np.ndarray) -> dict:
    """Validate training predictions on holdout set."""
    results = {}
    for target, predicted_precision in training_matches.items():
        # Find actual precision in holdout
        diffs = np.abs(holdout_zeros - target)
        actual_precision = np.min(diffs) / target

        results[target] = {
            'predicted': predicted_precision,
            'actual': actual_precision,
            'passed': actual_precision < 2 * predicted_precision  # Allow 2x margin
        }

    return results
```

---

## 10. Conclusion

This protocol establishes a rigorous framework for validating the GIFT-Zeta correspondences. Key features:

1. **Pre-registration**: Hypotheses and predictions specified before further analysis
2. **Holdout test**: Independent validation on unseen data
3. **Multiple testing**: Bonferroni and FDR corrections
4. **Random baseline**: Comparison to null distribution
5. **Honest reporting**: Include negative findings

The current evidence (p ≈ 0.018 combined) is suggestive but not conclusive. The holdout test will provide stronger evidence either way.

---

*"The first principle is that you must not fool yourself — and you are the easiest person to fool."*
— Richard Feynman

---
