# Spectral Gap Calibration Study

**Date**: January 2026
**Status**: In Progress

---

## Objective

Determine whether the observed λ₁ × H* = 13 on K₇ is:
1. **Structural**: A real physical value (dim(G₂) - h = 14 - 1 = 13)
2. **Artifact**: Discretization bias (true value = 14 from Pell equation)

---

## Methodology

Apply the **identical pipeline** used for K₇ to spaces with analytically known λ₁:

| Space | Dimension | λ₁ exact | Purpose |
|-------|-----------|----------|---------|
| S³ | 3 | 3 | General calibration |
| T⁷ | 7 | 1 | **Same dimension as K₇** |

### Key Question

If the T⁷ pipeline shows a systematic bias of ~7% (which would turn 14 → 13), then the K₇ result is an artifact.

If the T⁷ pipeline is accurate (< 3% bias), then 13 is structural.

---

## Notebook

**File**: `/research/notebooks/Spectral_Calibration_S3_T7.ipynb`

**Requirements**:
- Colab Pro+ with A100 recommended for N > 30,000
- CuPy for GPU acceleration
- ~15 minutes runtime for full study

**Output**:
- `Spectral_Calibration_S3_T7.png` — Convergence plots
- `Spectral_Calibration_S3_T7_results.json` — Full data

---

## Expected Results

### Scenario A: Bias matches correction needed
```
T⁷ calibration factor ≈ 0.93 (7% underestimate)
K₇ corrected: 13.07 / 0.93 ≈ 14.0
Verdict: ARTIFACT — true value is 14 (Pell)
```

### Scenario B: Bias is small
```
T⁷ calibration factor ≈ 0.98 (2% bias)
K₇ corrected: 13.07 / 0.98 ≈ 13.3
Verdict: STRUCTURAL — 13 is real (dim(G₂) - h)
```

### Scenario C: Inconclusive
```
T⁷ shows unstable results or intermediate bias
Action: Increase N, try different methods
```

---

## Next Steps Based on Results

### If STRUCTURAL (13 is real):
1. Focus on analytical proof via Cheeger bounds
2. Investigate why λ₁ × H* = dim(G₂) - h instead of dim(G₂)
3. The "−1" may come from parallel spinor (G₂ has 1 parallel spinor)

### If ARTIFACT (14 is real):
1. Apply calibration factor to all future K₇ measurements
2. Pell equation 99² - 50×14² = 1 is confirmed
3. Focus on improving numerical convergence

### If INCONCLUSIVE:
1. Run Belkin-Niyogi study (Rail A1)
2. Compute Cheeger constant directly (Rail C1)
3. Try larger N (75k, 100k)

---

## Theoretical Background

### S³ Spectrum
The Laplacian on S³ (unit 3-sphere) has eigenvalues:
```
λₙ = n(n+2), n = 0, 1, 2, ...
λ₀ = 0, λ₁ = 3, λ₂ = 8, ...
```

### T⁷ Spectrum
The Laplacian on T⁷ = (S¹)⁷ with unit radii has eigenvalues:
```
λ = Σᵢ nᵢ², nᵢ ∈ ℤ
λ₀ = 0, λ₁ = 1 (multiplicity 14)
```

### K₇ Conjecture
For K₇ with G₂ holonomy:
```
λ₁ × H* = 14  (Pell equation: 99² - 50×14² = 1)
or
λ₁ × H* = 13  (dim(G₂) - h = 14 - 1, parallel spinor correction)
```

---

*GIFT Spectral Gap Research Program*
