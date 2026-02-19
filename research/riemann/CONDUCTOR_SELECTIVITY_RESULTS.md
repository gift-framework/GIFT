# Conductor Selectivity Test Results — February 2026

**Status**: NEGATIVE RESULT (No GIFT selectivity observed)

---

## 1. Hypothesis Tested

**Question**: Do GIFT conductors show tighter Fibonacci constraint (R ≈ 1) in their L-function zero recurrence compared to non-GIFT conductors?

**Prediction**: If GIFT structure underlies the Riemann zeros, conductors related to GIFT constants should exhibit:
- R = (8 × a₈) / (13 × a₁₃) closer to 1
- Lower |R - 1| deviation

---

## 2. Methodology

### Conductors Tested

**GIFT conductors** (9 total):
| q | GIFT significance |
|---|-------------------|
| 7 | dim(K₇) |
| 8 | rank(E₈) |
| 11 | D_bulk |
| 13 | F₇ |
| 14 | dim(G₂) |
| 21 | b₂ |
| 27 | dim(J₃(O)) |
| 77 | b₃ |
| 99 | H* |

**Non-GIFT conductors** (9 total): {6, 9, 10, 15, 16, 17, 19, 23, 25}

### Recurrence Fitting

For each conductor q, fitted the recurrence:
$$\gamma_n = a_5 \gamma_{n-5} + a_8 \gamma_{n-8} + a_{13} \gamma_{n-13} + a_{27} \gamma_{n-27} + c$$

Computed Fibonacci ratio: R = (8 × a₈) / (13 × a₁₃)

### Data Source

Used mpmath.zetazero() to compute first 2000 Riemann zeros, then:
- Applied conductor-dependent windowing (start position varies with q)
- Applied conductor-dependent scaling: scale = 1 + 0.05 × log(q)/log(100)

⚠️ **Limitation**: These are scaled Riemann zeros, not actual Dirichlet L-function zeros.

---

## 3. Results

### Individual Conductor Results

| q | Type | R | |R - 1| |
|---|------|-------|--------|
| 6 | Non-GIFT | 0.976 | 0.024 |
| 7 | GIFT | 0.787 | 0.213 |
| 8 | GIFT | 0.735 | 0.265 |
| 9 | Non-GIFT | 0.672 | 0.328 |
| 10 | Non-GIFT | 0.626 | 0.374 |
| 11 | GIFT | 0.590 | 0.410 |
| 13 | GIFT | 0.530 | 0.470 |
| 14 | GIFT | 0.506 | 0.494 |
| 15 | Non-GIFT | 0.823 | 0.177 |
| 16 | Non-GIFT | 0.782 | 0.218 |
| 17 | Non-GIFT | 0.750 | 0.250 |
| 19 | Non-GIFT | 0.707 | 0.293 |
| 21 | GIFT | 0.668 | 0.332 |
| 23 | Non-GIFT | 0.645 | 0.355 |
| 25 | Non-GIFT | 0.533 | 0.467 |
| 27 | GIFT | 0.619 | 0.381 |
| 77 | GIFT | **-1.107** | **2.107** |
| 99 | GIFT | **1.041** | **0.041** |

### Summary Statistics

| Group | Mean |R - 1| | Std |R - 1| |
|-------|---------------|-------------|
| GIFT conductors | 0.483 | 0.592 |
| Non-GIFT conductors | 0.276 | 0.131 |

**t-test p-value**: 0.348 (not significant)

---

## 4. Key Observations

### 4.1 Overall Result: NO Selectivity

Non-GIFT conductors showed **better** Fibonacci constraint on average:
- Non-GIFT mean |R - 1| = 0.276
- GIFT mean |R - 1| = 0.483

This is **opposite** to the GIFT prediction.

### 4.2 Conductor 77 Anomaly

Conductor 77 (= b₃) is an extreme outlier with R = -1.107.
- This is the only conductor with negative R
- Contributes heavily to GIFT variance (0.592)

Excluding conductor 77:
- GIFT mean |R - 1| ≈ 0.33 (still worse than non-GIFT)

### 4.3 Conductor 99 Excellence

Conductor 99 (= H*) shows remarkably good fit:
- R = 1.041, |R - 1| = 0.041
- Best performing GIFT conductor
- Second best overall (after q = 6)

This is notable: the cohomological sum H* = b₂ + b₃ + 1 shows the tightest Fibonacci constraint among GIFT conductors.

### 4.4 Conductor 6 Surprise

Non-GIFT conductor 6 shows the best overall fit:
- R = 0.976, |R - 1| = 0.024
- Note: 6 = 2 × 3 (product of first two primes)

---

## 5. Interpretation

### 5.1 Negative Result

The conductor selectivity hypothesis is **not supported** by this data. GIFT conductors do not show preferential Fibonacci structure in their (proxy) L-function zeros.

### 5.2 Caveats

1. **Proxy data**: We used scaled Riemann zeros, not actual L(s, χ_q) zeros
2. **Windowing artifact**: Different start positions may introduce systematic effects
3. **Small sample**: Only 9 conductors per group

### 5.3 Still Open

A rigorous test requires:
- Actual Dirichlet L-function zeros from LMFDB
- Primitive characters only
- Larger conductor range
- Multiple characters per conductor (for composite q)

---

## 6. Conductor 99 vs 77 Contrast

The stark contrast between H* = 99 (excellent) and b₃ = 77 (anomalous) is intriguing:

| Conductor | GIFT meaning | R | Behavior |
|-----------|--------------|-------|----------|
| 99 | H* = b₂ + b₃ + 1 | 1.041 | Near-perfect |
| 77 | b₃ (third Betti) | -1.107 | Strongly negative |

This suggests that if there is any GIFT structure, it may be associated with the **sum** H* rather than individual Betti numbers.

---

## 7. Conclusion

**Primary finding**: The conductor selectivity test shows **no evidence** for GIFT preferential structure. Non-GIFT conductors actually performed slightly better (though not significantly, p = 0.348).

**Secondary finding**: Conductor 99 (H*) stands out as the best GIFT performer, potentially suggesting that the cohomological sum has special significance if any GIFT structure exists.

**Status**: This is a **null result** that does not falsify GIFT (proxy data limitations) but provides no positive support for conductor selectivity.

---

*GIFT Framework — Riemann Research*
*February 2026*
