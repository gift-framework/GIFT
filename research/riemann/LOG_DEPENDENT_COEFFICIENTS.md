# Log-Dependent Coefficients in the GIFT-Riemann Recurrence

## Mathematical Framework for n-Dependent Coefficients

**Document Version**: 1.0
**Date**: 2026-01-31
**Status**: RESEARCH PROPOSAL
**Motivation**: AI Council recommendation (Gemini) to address coefficient instability

---

## 1. The Problem: Coefficient Instability

### 1.1 Observed Evidence

From the GIFT-Riemann research, we have:

| Observation | Implication |
|-------------|-------------|
| Coefficients vary ~50% across fitting ranges | Not truly constant |
| Error decreases: 0.51% (n<100) to 0.05% (n>10k) | Asymptotic improvement |
| Density of zeros ~ log(T)/2pi | Logarithmic scale dependence |

The current recurrence:
$$\gamma_n = a_5 \gamma_{n-5} + a_8 \gamma_{n-8} + a_{13} \gamma_{n-13} + a_{27} \gamma_{n-27} + c$$

assumes constant coefficients, which appears to be an approximation.

### 1.2 Theoretical Motivation

The zeros gamma_n have known asymptotic behavior:
$$\gamma_n \sim \frac{2\pi n}{\ln n} \quad (n \to \infty)$$

This logarithmic scaling suggests that any recurrence coefficients should incorporate log(n) terms to properly capture the changing "scale" of the zeros.

---

## 2. Proposed Functional Forms

### 2.1 Form A: Pure Log-Correction

**Model**:
$$a_i(n) = \frac{a_i^0}{\ln(n)}$$

**Motivation**: As n increases, log(n) grows slowly, so coefficients decrease. This could explain why the recurrence "tightens" for large n.

**Recurrence**:
$$\gamma_n = \frac{1}{\ln(n)} \left[ a_5^0 \gamma_{n-5} + a_8^0 \gamma_{n-8} + a_{13}^0 \gamma_{n-13} + a_{27}^0 \gamma_{n-27} \right] + c(n)$$

**Issue**: Dimensionally, dividing gamma by log(n) changes units. Need careful normalization.

### 2.2 Form B: Asymptotic-Plus-Correction

**Model**:
$$a_i(n) = a_i^\infty + \frac{b_i}{\ln(n)}$$

**Motivation**: Coefficients approach asymptotic values a_i^infinity as n grows, with O(1/log n) corrections.

**Recurrence**:
$$\gamma_n = \sum_i \left( a_i^\infty + \frac{b_i}{\ln(n)} \right) \gamma_{n-\ell_i} + c^\infty + \frac{d}{\ln(n)}$$

**Advantage**:
- Natural limit as n -> infinity
- GIFT constants could appear as a_i^infinity
- b_i encode finite-n corrections

### 2.3 Form C: Log-Weighted Normalized Recurrence

**Model**: Normalize by the asymptotic scale

Let $\tilde{\gamma}_n = \gamma_n \cdot \frac{\ln(n)}{2\pi n}$ (normalized zero)

Then:
$$\tilde{\gamma}_n \approx \sum_i a_i \cdot \tilde{\gamma}_{n-\ell_i} + c$$

**Motivation**: Remove the asymptotic growth, work with "detrended" zeros.

**Advantage**: Coefficients might be truly constant on normalized data.

### 2.4 Form D: Riemann-Siegel Inspired

From the Riemann-Siegel formula, corrections to the zero density involve terms like:

$$N(T) = \frac{T}{2\pi} \ln\frac{T}{2\pi} - \frac{T}{2\pi} + \frac{7}{8} + S(T) + O(1/T)$$

where S(T) is the argument of zeta on the critical line.

**Model**:
$$a_i(n) = a_i^0 \left(1 + \frac{c_1}{\ln(\gamma_n)} + \frac{c_2}{\ln^2(\gamma_n)} + \ldots \right)$$

**Advantage**: Matches known asymptotic expansion structure.

### 2.5 Form E: Power-Law Correction

**Model**:
$$a_i(n) = a_i^0 \cdot n^{\alpha_i}$$

where alpha_i are small exponents (e.g., |alpha| < 0.1).

**Recurrence**:
$$\gamma_n = \sum_i a_i^0 \cdot n^{\alpha_i} \cdot \gamma_{n-\ell_i} + c$$

**Issue**: Less motivated physically; more parameters.

---

## 3. Mathematical Analysis

### 3.1 Dimensional Consistency

For the recurrence to be dimensionally consistent:
- LHS: gamma_n has units of [height on critical line]
- RHS: sum of gamma terms must have same units

Key constraint: Sum of coefficients must be bounded:
$$\sum_i a_i(n) \lesssim 1 + O(1/\ln n)$$

Otherwise, the recurrence diverges.

### 3.2 Stability Analysis

For a linear recurrence with n-dependent coefficients:
$$x_n = \sum_i a_i(n) x_{n-\ell_i}$$

the characteristic equation becomes position-dependent. Stability requires:

$$\left| \sum_i a_i(n) z^{\ell_i} \right| < 1 \quad \text{for } |z| = 1$$

### 3.3 Connection to Difference Equations

A recurrence with log-dependent coefficients is a **non-autonomous difference equation**. These are well-studied in:
- Discrete dynamical systems
- Perturbation theory for difference equations
- Adiabatic approximation (when coefficients change slowly)

### 3.4 The "Slowly Varying" Assumption

If a_i(n) = a_i(1 + O(1/log n)), then:
- Coefficients change by O(1%) per decade in n
- Justifies fitting in "local windows" and interpolating

---

## 4. Fitting Methodology

### 4.1 Nonlinear Least Squares for Form B

**Parameters**: {a_i^infinity, b_i, c^infinity, d} = 10 parameters for 4 lags

**Objective**:
$$\min \sum_{n=28}^{N} \left| \gamma_n - \sum_i \left(a_i^\infty + \frac{b_i}{\ln n}\right) \gamma_{n-\ell_i} - c^\infty - \frac{d}{\ln n} \right|^2$$

**Algorithm**: Use scipy.optimize.curve_fit or lmfit

```python
from scipy.optimize import curve_fit
import numpy as np

def model_form_B(n_array, gamma_lagged,
                 a5_inf, b5, a8_inf, b8,
                 a13_inf, b13, a27_inf, b27,
                 c_inf, d):
    """
    gamma_lagged: dict with keys 5, 8, 13, 27 containing lagged gamma arrays
    """
    result = np.zeros_like(n_array, dtype=float)
    log_n = np.log(n_array)

    for lag, (a_inf, b) in [(5, (a5_inf, b5)),
                             (8, (a8_inf, b8)),
                             (13, (a13_inf, b13)),
                             (27, (a27_inf, b27))]:
        a_n = a_inf + b / log_n
        result += a_n * gamma_lagged[lag]

    result += c_inf + d / log_n
    return result
```

### 4.2 Windowed Fitting to Extract a_i(n)

**Method**:
1. Fit standard recurrence on sliding windows: [n, n+W] for W ~ 500
2. Record coefficients as function of window center n_c
3. Fit a_i(n_c) to proposed functional forms
4. Test which form has lowest residual

```python
def windowed_fit(gamma, lags, window_size=500, step=100):
    """
    Fit recurrence in sliding windows to observe coefficient evolution.
    """
    results = []
    max_lag = max(lags)

    for start in range(max_lag, len(gamma) - window_size, step):
        end = start + window_size
        n_center = (start + end) // 2

        coeffs = fit_recurrence(gamma, lags, start, end)

        results.append({
            'n_center': n_center,
            'log_n': np.log(n_center),
            'coeffs': dict(zip(lags + ['c'], coeffs))
        })

    return results
```

### 4.3 Cross-Validation for Model Selection

Compare models using:
- AIC (Akaike Information Criterion): penalize extra parameters
- Leave-one-out cross-validation
- Train on n < 50k, test on n > 50k

**Model comparison table**:
| Model | Parameters | Expected Behavior |
|-------|------------|-------------------|
| Constant | 5 | Baseline |
| Form A | 5 | Pure scaling |
| Form B | 10 | Asymptotic + correction |
| Form C | 5 (on normalized) | Scale-free |
| Form D | 5+k | RS expansion |

---

## 5. GIFT Constants in Asymptotic Limits

### 5.1 Hypothesis: GIFT Ratios as Asymptotic Values

If we use Form B, the asymptotic coefficients a_i^infinity might be exactly the GIFT ratios:

| Coefficient | GIFT Prediction |
|-------------|-----------------|
| a_5^infinity | N_gen/h_G2 = 3/6 = 0.5 |
| a_8^infinity | dim(E7_fund)/H* = 56/99 |
| a_13^infinity | -dim(G2)/H* = -14/99 |
| a_27^infinity | 1/dim(J3O) = 1/27 |
| c^infinity | H*/Weyl = 99/5 = 19.8 |

**Test**: Fit Form B with GIFT values **fixed** for a_i^infinity, optimize only the b_i corrections.

### 5.2 Error Correction Interpretation

The b_i/log(n) terms could represent:
- Finite-size corrections from K7 compactification
- Contributions from torsion (which vanishes asymptotically in GIFT)
- Discretization effects from integer zero indices

### 5.3 Scaling Argument

If K7 spectral eigenvalues lambda_n scale as:
$$\lambda_n \sim \gamma_n / H^*$$

then log-corrections might arise from:
- Weyl law corrections on K7
- Spectral zeta function regularization
- Heat kernel short-time expansion

---

## 6. Riemann-Siegel Formula Insights

### 6.1 The Riemann-Siegel Formula

For zeta on the critical line:
$$\zeta(1/2 + it) = \sum_{n \le \sqrt{t/2\pi}} n^{-1/2-it} + \chi(1/2+it) \sum_{n \le \sqrt{t/2\pi}} n^{-1/2+it} + R(t)$$

where R(t) has an asymptotic expansion in inverse powers of t.

### 6.2 Implications for Zero Locations

The zeros gamma_n are approximately where:
$$\zeta(1/2 + i\gamma_n) = 0$$

Corrections to the main term involve:
- O(1/gamma) terms
- O(1/gamma^2) terms
- Oscillatory corrections from S(T)

### 6.3 Connection to Recurrence Coefficients

If gamma_n satisfies a recurrence, and Riemann-Siegel provides:
$$\gamma_n = g(n) + \frac{c_1}{\ln(\gamma_n)} + \frac{c_2}{\ln^2(\gamma_n)} + \ldots$$

then recurrence coefficients should have similar expansions.

**Key insight**: The log-corrections in Riemann-Siegel are in log(gamma_n), not log(n). But since gamma_n ~ 2*pi*n/log(n), we have:
$$\ln(\gamma_n) \approx \ln(2\pi n) - \ln\ln(n) \approx \ln(n) + \ln(2\pi) - \frac{\ln\ln n}{\ln n}$$

So log(gamma_n) ~ log(n) to leading order.

---

## 7. Experimental Design

### 7.1 Experiment 1: Windowed Coefficient Extraction

**Goal**: Map how coefficients vary with n

**Protocol**:
1. Use 100,000 Riemann zeros (Odlyzko tables)
2. Fit recurrence in windows [n, n+1000] with step 200
3. Record {a_5(n), a_8(n), a_13(n), a_27(n), c(n)} for each window center n
4. Plot coefficients vs log(n)

**Success criterion**: Coefficients show systematic trend with log(n)

### 7.2 Experiment 2: Fit Functional Forms

**Goal**: Determine which functional form best describes a_i(n)

**Protocol**:
1. From Experiment 1, have a_i(n_k) for many n_k
2. Fit each model:
   - Form A: a_i(n) = a_i^0 / log(n)
   - Form B: a_i(n) = a_i^inf + b_i / log(n)
   - Power law: a_i(n) = a_i^0 * n^alpha
3. Compare R^2 and AIC for each form

**Success criterion**: One form clearly dominates

### 7.3 Experiment 3: Test Stabilization

**Goal**: Do coefficients stabilize when normalized?

**Protocol**:
1. Define a_i*(n) = a_i(n) * log(n) (for Form A) or a_i(n) - a_i^inf (for Form B)
2. Check if a_i*(n) is approximately constant
3. Compute coefficient of variation of a_i*(n) across windows

**Success criterion**: CV(a_i*) < 10%

### 7.4 Experiment 4: GIFT Asymptotic Test

**Goal**: Test if GIFT ratios emerge as n -> infinity

**Protocol**:
1. Fit Form B with a_i^inf as free parameters
2. Compare fitted a_i^inf with GIFT predictions
3. Compute: delta_i = |a_i^inf(fitted) - a_i^inf(GIFT)| / |a_i^inf(GIFT)|

**Success criterion**: delta_i < 5% for all i

### 7.5 Experiment 5: Out-of-Sample with Log-Correction

**Goal**: Does log-corrected model generalize better?

**Protocol**:
1. Train on n in [28, 50000] using:
   - Constant coefficient model
   - Form B model
2. Test on n in [50001, 100000]
3. Compare prediction errors

**Success criterion**: Form B test error significantly lower than constant model

---

## 8. Implementation Plan

### 8.1 Phase 1: Data Preparation (1 day)

- Load 100k+ zeros from LMFDB or Odlyzko
- Verify data integrity
- Compute log(n) and log(gamma_n) arrays

### 8.2 Phase 2: Windowed Fitting (1-2 days)

- Implement sliding window fitter
- Run on full dataset
- Visualize coefficient evolution
- Identify trends

### 8.3 Phase 3: Model Fitting (2-3 days)

- Implement all functional forms (A, B, C, D, E)
- Fit each to extracted a_i(n)
- Compute goodness-of-fit metrics
- Select best model

### 8.4 Phase 4: Validation (2-3 days)

- Cross-validation experiments
- Out-of-sample testing
- GIFT asymptotic limit check
- Statistical significance tests

### 8.5 Phase 5: Documentation (1-2 days)

- Write up results
- Update GIFT-Riemann summary
- Prepare for peer review

---

## 9. Expected Outcomes

### 9.1 Optimistic Scenario

- Form B works with GIFT ratios as asymptotic limits
- b_i corrections are small and physically interpretable
- Coefficient instability explained
- Error < 0.01% for large n

**Implication**: Strong evidence for GIFT-Riemann connection

### 9.2 Moderate Scenario

- Log-correction helps but doesn't stabilize completely
- Asymptotic limits differ from GIFT by ~10-20%
- Some unexplained variance remains

**Implication**: Suggestive but not conclusive

### 9.3 Pessimistic Scenario

- No functional form works significantly better
- Coefficients don't stabilize
- Log-correction doesn't help

**Implication**: Recurrence is approximate; coefficient variation is fundamental noise

---

## 10. Theoretical Questions for Further Research

1. **Can we derive the recurrence from trace formulas?**
   - Weil explicit formula connects zeros to primes
   - Could Fibonacci structure emerge from prime distribution?

2. **What operator has this recurrence as its eigenvalue equation?**
   - Hilbert-Polya approach
   - K7 Laplacian connection?

3. **Are there other recurrences with different lag structures?**
   - 5-term recurrence with lag 14 (dim G2)?
   - Connection to modular forms?

4. **Does the recurrence hold for other L-functions?**
   - Dirichlet L-functions
   - Modular L-functions
   - Universal structure?

---

## Appendix A: Python Implementation Sketch

```python
import numpy as np
from scipy.optimize import curve_fit, minimize
from typing import List, Dict, Tuple

class LogDependentRecurrence:
    """
    Fit and test log-dependent recurrence models for Riemann zeros.
    """

    def __init__(self, gamma: np.ndarray, lags: List[int] = [5, 8, 13, 27]):
        self.gamma = gamma
        self.lags = lags
        self.max_lag = max(lags)

    def fit_form_B(self, start: int = None, end: int = None,
                   fix_asymptotic: Dict[int, float] = None) -> Dict:
        """
        Fit Form B: a_i(n) = a_i^inf + b_i / log(n)

        Parameters:
            start, end: fitting range
            fix_asymptotic: dict of fixed a_i^inf values (GIFT test)
        """
        if start is None:
            start = self.max_lag
        if end is None:
            end = len(self.gamma)

        # Build design matrix
        n_array = np.arange(start, end)
        log_n = np.log(n_array)

        # Target
        y = self.gamma[start:end]

        # Features: for each lag, we have two terms: gamma_{n-lag} and gamma_{n-lag}/log(n)
        X = []
        for n in range(start, end):
            row = []
            for lag in self.lags:
                g_lag = self.gamma[n - lag]
                row.append(g_lag)  # a_i^inf term
                row.append(g_lag / np.log(n))  # b_i term
            row.append(1.0)  # c^inf
            row.append(1.0 / np.log(n))  # d term
            X.append(row)

        X = np.array(X)

        # If fixing asymptotic values, modify the problem
        if fix_asymptotic:
            # Subtract fixed contribution from y
            y_mod = y.copy()
            for i, lag in enumerate(self.lags):
                if lag in fix_asymptotic:
                    for j, n in enumerate(range(start, end)):
                        y_mod[j] -= fix_asymptotic[lag] * self.gamma[n - lag]

            # Only fit b_i terms (odd columns except last two)
            X_reduced = X[:, 1::2]  # b terms only (indices 1, 3, 5, 7, 9)
            coeffs_reduced, _, _, _ = np.linalg.lstsq(X_reduced, y_mod, rcond=None)

            # Reconstruct full coefficient vector
            coeffs = []
            for i, lag in enumerate(self.lags):
                coeffs.append(fix_asymptotic.get(lag, 0))
                coeffs.append(coeffs_reduced[i])
            coeffs.extend(coeffs_reduced[-2:])  # c^inf, d
            coeffs = np.array(coeffs)
        else:
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

        # Parse coefficients
        result = {'form': 'B', 'lags': self.lags}
        for i, lag in enumerate(self.lags):
            result[f'a{lag}_inf'] = coeffs[2*i]
            result[f'b{lag}'] = coeffs[2*i + 1]
        result['c_inf'] = coeffs[-2]
        result['d'] = coeffs[-1]

        # Compute predictions and error
        y_pred = X @ coeffs
        errors = np.abs(y_pred - y) / y * 100
        result['mean_error_pct'] = np.mean(errors)
        result['n_points'] = len(y)

        return result

    def windowed_analysis(self, window_size: int = 1000, step: int = 200) -> List[Dict]:
        """
        Fit constant-coefficient recurrence in sliding windows.
        """
        results = []

        for start in range(self.max_lag, len(self.gamma) - window_size, step):
            end = start + window_size
            n_center = (start + end) // 2

            # Standard least squares fit
            X = []
            y = []
            for n in range(start, end):
                row = [self.gamma[n - lag] for lag in self.lags] + [1.0]
                X.append(row)
                y.append(self.gamma[n])

            X = np.array(X)
            y = np.array(y)
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

            results.append({
                'n_center': n_center,
                'log_n_center': np.log(n_center),
                'a5': coeffs[0],
                'a8': coeffs[1],
                'a13': coeffs[2],
                'a27': coeffs[3],
                'c': coeffs[4]
            })

        return results

    def test_coefficient_stabilization(self, windowed_results: List[Dict]) -> Dict:
        """
        Test if a_i(n) * log(n) is more stable than a_i(n).
        """
        report = {}

        for coef_name in ['a5', 'a8', 'a13', 'a27', 'c']:
            values = np.array([r[coef_name] for r in windowed_results])
            log_n = np.array([r['log_n_center'] for r in windowed_results])

            # CV for raw coefficient
            cv_raw = np.std(values) / np.abs(np.mean(values)) * 100

            # CV for log-normalized coefficient
            values_normalized = values * log_n
            cv_normalized = np.std(values_normalized) / np.abs(np.mean(values_normalized)) * 100

            report[coef_name] = {
                'cv_raw': cv_raw,
                'cv_normalized': cv_normalized,
                'stabilization_ratio': cv_raw / cv_normalized if cv_normalized > 0 else np.inf
            }

        return report


# GIFT asymptotic predictions
GIFT_ASYMPTOTIC = {
    5: 0.5,        # N_gen / h_G2 = 3/6
    8: 56/99,      # dim(E7_fund) / H*
    13: -14/99,    # -dim(G2) / H*
    27: 1/27       # 1 / dim(J3O)
}
```

---

## Appendix B: Riemann-Siegel Expansion Terms

For reference, the first few terms of the Riemann-Siegel remainder:

$$R(t) = (-1)^{N-1} \left(\frac{t}{2\pi}\right)^{-1/4} \left[ C_0(\rho) + C_1(\rho) \left(\frac{t}{2\pi}\right)^{-1/2} + \ldots \right]$$

where rho = sqrt(t/2*pi) - floor(sqrt(t/2*pi)) and C_k are known functions.

This suggests correction terms of order:
- O(t^{-1/4})
- O(t^{-3/4})
- ...

Which translate to corrections in log(n) when n ~ t/(2*pi*log(t)).

---

**Document prepared for GIFT Research Group**

*Next step: Implement experiments in Jupyter notebook*
