# Weil Explicit Formula: Research on Deriving Recurrence Relations

**Date**: 2026-01-31
**Purpose**: Investigate the trace formula approach for GIFT-Riemann connection
**Status**: RESEARCH / EXPLORATORY

---

## Executive Summary

The Weil explicit formula offers a *theoretically elegant* path to derive recurrence relations for Riemann zeros, but the practical challenges are formidable. This document provides a comprehensive analysis of the approach, its feasibility, and concrete next steps.

**Key Finding**: The trace formula approach is mathematically sophisticated but may explain *why* any recurrence for zeros would exist, rather than *discovering* new recurrences. The lags [5,8,13,27] found empirically likely reflect asymptotic behavior rather than deep arithmetic structure.

---

## 1. The Weil Explicit Formula in Detail

### 1.1 Classical Statement

The Riemann-Weil explicit formula connects **zeros of zeta** to **prime powers**:

$$\sum_{\rho} h(\gamma_\rho) = h\left(\frac{i}{2}\right) + h\left(-\frac{i}{2}\right) - \sum_{n=1}^{\infty} \frac{\Lambda(n)}{\sqrt{n}} \left[ \hat{h}(\log n) + \hat{h}(-\log n) \right] + \int_{-\infty}^{\infty} h(t) \, d\Omega(t)$$

Where:
- **h(t)**: Test function (analytic in a strip, decaying sufficiently at infinity)
- **ρ = 1/2 + iγ**: Non-trivial zeros (assuming RH: γ ∈ ℝ)
- **Λ(n)**: Von Mangoldt function (= log p if n = p^k, else 0)
- **ĥ**: Fourier transform of h
- **Ω(t)**: Archimedean contribution (involving log Γ terms)

### 1.2 Explicit Form (Weil, 1952)

For a test function g(x) with suitable properties:

$$\sum_{\gamma} g(\gamma) = g(i/2) + g(-i/2) - \frac{1}{2\pi} \int_{-\infty}^{\infty} g(t) \cdot \frac{\Gamma'}{\Gamma}\left(\frac{1}{4} + \frac{it}{2}\right) dt - \sum_{p} \sum_{k=1}^{\infty} \frac{\log p}{p^{k/2}} \cdot \hat{g}(k \log p)$$

### 1.3 Dual Interpretation

The formula has two dual readings:

| **Spectral Side** | **Arithmetic Side** |
|-------------------|---------------------|
| Sum over zeros γ | Sum over primes p |
| Eigenvalues of hypothetical operator | Periodic orbits (in dynamical interpretation) |
| ∑ h(γ) | ∑ (log p / p^{k/2}) ĥ(k log p) |

This duality is the key: **if zeros satisfy a recurrence, the Fourier transforms of test functions must encode this structure through the prime side**.

---

## 2. How Could a Recurrence Emerge?

### 2.1 The Conceptual Path

Suppose zeros satisfy:
$$\gamma_n = a_1 \gamma_{n-\ell_1} + a_2 \gamma_{n-\ell_2} + \cdots + a_k \gamma_{n-\ell_k} + c$$

This implies a **linear constraint** on the spectral side. For the explicit formula to produce this:

1. **Choose h such that** ∑ h(γₙ) picks out the recurrence coefficients
2. **The prime side must encode** the recurrence through interference patterns

### 2.2 Generating Function Approach

Define the formal series:
$$G(x) = \sum_{n=1}^{\infty} \gamma_n x^n$$

If γₙ satisfies a recurrence with lags [ℓ₁, ..., ℓₖ], then:
$$G(x) \cdot P(x) = \text{polynomial (initial conditions)}$$

where P(x) = 1 - a₁x^{ℓ₁} - a₂x^{ℓ₂} - ... - aₖx^{ℓₖ}

**For lags [5, 8, 13, 27]:**
$$P(x) = 1 - a_5 x^5 - a_8 x^8 - a_{13} x^{13} - a_{27} x^{27}$$

The roots of P(x) would determine the asymptotic behavior of γₙ.

### 2.3 The Trace Formula Connection

The Selberg trace formula for hyperbolic surfaces provides intuition:
$$\sum_{\lambda_n} h(\sqrt{\lambda_n - 1/4}) = \frac{\text{Area}}{4\pi} \int h(r) \cdot r \tanh(\pi r) \, dr + \sum_{\{P\}} \frac{\ell(P_0)}{2\sinh(\ell(P)/2)} \hat{h}(\ell(P))$$

Here **closed geodesics** play the role of primes. A recurrence for eigenvalues λₙ would require the length spectrum {ℓ(P)} to have special arithmetic structure.

**For Riemann zeros**: Primes replace geodesics. A recurrence emerges if:
$$\sum_p \frac{\log p}{p^{k/2}} e^{-ik t \log p}$$
has specific resonance frequencies at the lags [5, 8, 13, 27].

---

## 3. Test Functions for Revealing Recurrence Structure

### 3.1 Exponential Test Functions

To probe a recurrence at lag ℓ, use:
$$h_\ell(t) = e^{2\pi i \ell t / T}$$

where T is a characteristic scale. The sum ∑ h_ℓ(γ) becomes:
$$S_\ell(T) = \sum_{n=1}^{N} e^{2\pi i \ell \gamma_n / T}$$

If zeros have a recurrence component at lag ℓ, then |S_ℓ(T)| will show resonance peaks.

### 3.2 Gaussian Test Functions

More practical is:
$$h_\sigma(t) = e^{-t^2 / 2\sigma^2}$$

With Fourier transform:
$$\hat{h}_\sigma(u) = \sigma \sqrt{2\pi} e^{-\sigma^2 u^2 / 2}$$

**Strategy**:
- For large σ, the test function is broad → captures global structure
- For small σ, it's localized → probes fine spacing

### 3.3 Band-Limited Test Functions

For recurrence analysis, use:
$$h(t) = \text{sinc}\left(\frac{\pi t}{L}\right) = \frac{\sin(\pi t / L)}{\pi t / L}$$

This has compact Fourier support: ĥ(u) = 0 for |u| > L.

**Key Insight**: If the prime sum truncates at log p < L, the formula becomes:
$$\sum_\gamma h(\gamma) = \text{(finite sum over small primes)}$$

This could reveal if lags [5, 8, 13, 27] correspond to primes p with log p ≈ 5, 8, 13, 27.

### 3.4 Proposed Test Function for GIFT Lags

To specifically probe the recurrence structure:
$$h_{\text{GIFT}}(t) = \sum_{k \in \{5,8,13,27\}} c_k \cdot e^{-|t|/k}$$

The Fourier transform is:
$$\hat{h}_{\text{GIFT}}(u) = \sum_k \frac{2 c_k k}{1 + k^2 u^2}$$

Apply this to the explicit formula and examine if the prime side shows special behavior at these scales.

---

## 4. Existing Literature

### 4.1 Closest Work: Prolate Wave Operators (Connes et al., 2024)

**Reference**: [arXiv:2310.18423v2](https://arxiv.org/pdf/2310.18423) - "Zeta Zeros and Prolate Wave Operators"

This work advances the spectral realization by:
- Introducing a semilocal prolate wave operator
- Relating Weil's positivity criterion to operator traces
- Providing a Hilbert space framework for explicit formulas

**Key Quote**: "The semilocal trace formula gives, for each n, a Hilbert space theoretic framework in which the Weil quadratic form Qₙ becomes the trace of a simple operator theoretic expression."

### 4.2 Connes' Noncommutative Geometry

**Reference**: [Trace Formula in NCG](https://alainconnes.org/wp-content/uploads/selecta.ps-2.pdf)

Connes interprets zeros as:
- **Absorption spectrum** of an operator on adele classes
- **Trace formula** on a noncommutative space

This provides a geometric interpretation of explicit formulas but does not directly yield recurrence relations.

### 4.3 Berry-Keating Conjecture

**Reference**: [Riemann Zeros and Eigenvalue Asymptotics](https://empslocal.ex.ac.uk/people/staff/mrwatkin/zeta/berry-keating1.pdf)

Proposes the Hamiltonian H = xp (position times momentum) as underlying the zeros. The "Riemann dynamics" should be chaotic with periodic orbits at multiples of log p.

**Implication for Recurrence**: If H has a particular band structure, the eigenvalue equation could induce recurrences. However, no explicit recurrence has been derived.

### 4.4 LeClair-Mussardo Approach

**Reference**: [JHEP 2024](https://link.springer.com/article/10.1007/JHEP04(2024)062)

Uses integrable systems (Bethe ansatz) to model zeros as "quantized energies" of a scattering system. This could potentially yield recurrence-like structures through the Bethe equations.

### 4.5 Gap in Literature

**No existing work directly derives recurrence relations from trace formulas.**

The approaches above provide spectral interpretations but do not:
- Identify specific lags
- Connect to Fibonacci structures
- Produce coefficients aℓ

This is the **open problem** that the GIFT approach attempts to address.

---

## 5. Could Lags [5,8,13,27] Have Prime Distribution Meaning?

### 5.1 Direct Prime Correspondences

| Lag | log₂ p candidates | log p candidates | Prime interpretation |
|-----|-------------------|------------------|---------------------|
| 5 | p ≈ 32 | p ≈ 148 | Not prime (32 = 2⁵) |
| 8 | p ≈ 256 | p ≈ 2981 | 2981 is prime |
| 13 | p ≈ 8192 | p ≈ 442413 | Composite |
| 27 | p ≈ 134M | p ≈ 5.3×10¹¹ | N/A |

**Conclusion**: No simple "log p = lag" correspondence.

### 5.2 Prime Class Distribution

Consider primes in residue classes mod 5, 8, 13, 27:

**Mod 5 classes** (quadratic residues):
- Primes ≡ 1, 4 (mod 5): Fibonacci index divisibility (Fₚ₋₁ ≡ 0)
- Primes ≡ 2, 3 (mod 5): Different structure (Fₚ₊₁ ≡ 0)

**Mod 8 classes**:
- Primes ≡ 1, 3, 5, 7 (mod 8): Related to quadratic character of 2

**Mod 13 and 27**:
- 13 is a Fibonacci prime (F₇ = 13)
- 27 = 3³ relates to cubic residues

### 5.3 Fibonacci/Pisano Connection

The Pisano period π(n) gives the period of Fibonacci mod n:

| n | π(n) | Relevance |
|---|------|-----------|
| 5 | 20 | F₅ = 5 (Fibonacci prime) |
| 8 | 12 | 8 = rank(E₈) |
| 13 | 28 | F₇ = 13 (Fibonacci prime) |
| 27 | 72 | 27 = dim(J₃(O)) |

**Observation**: The lags satisfy:
- 5 + 8 = 13 (Fibonacci addition)
- 5 × 8 - 13 = 27 (bilinear relation)

This Fibonacci structure in the lags is *exact*, not approximate.

### 5.4 Speculative: Dirichlet L-function Connection

For a character χ mod q, the explicit formula becomes:
$$\sum_{\gamma_\chi} h(\gamma_\chi) = \ldots - \sum_p \frac{\chi(p) \log p}{p^{1/2}} \hat{h}(\log p) - \ldots$$

If the recurrence arises from interference between L-functions with conductors 5, 8, 13, 27, the Fibonacci structure might emerge from:
- Quadratic characters mod 5 and 13 (Fibonacci primes)
- 8 = 2³ (power of 2)
- 27 = 3³ (power of 3)

This remains highly speculative without explicit computation.

---

## 6. Alternative Interpretation: AR(k) vs Structural Recurrence

### 6.1 The AR Model Warning

The validation showed the recurrence behaves like an autoregressive model:
- Coefficients drift with n
- High correlation with smoothness
- Not "locked" to GIFT constants

**Critical Observation** (from GPT-4 council):
> "A 0.05% relative error at γₙ ≈ 70,000 corresponds to ~35-40 absolute units, i.e., ~50-60 average spacings. This is a trend predictor, not a fine structure predictor."

### 6.2 The "Unfolded" Metric

Define normalized spacings:
$$\tilde{\gamma}_n = N(\gamma_n) = \frac{\gamma_n}{2\pi} \log\left(\frac{\gamma_n}{2\pi}\right)$$

A *true* recurrence should hold for unfolded zeros, with error < 1 spacing.

### 6.3 What Would a Genuine Trace-Derived Recurrence Look Like?

If the explicit formula truly implied a recurrence, we would expect:
1. **Exact rational coefficients** (or at least algebraic)
2. **Stability across all n** (no drift)
3. **Derivable from prime distribution** via Fourier analysis
4. **Connection to known spectral operators**

The empirical [5,8,13,27] recurrence fails criteria 2-4.

---

## 7. Feasibility Assessment

### 7.1 What Would Be Required

To derive a recurrence from the Weil formula:

1. **Construct an explicit operator** H whose spectrum is {γₙ}
2. **Identify its band structure** (why lags [5,8,13,27]?)
3. **Prove the characteristic polynomial** of a truncated H matches the recurrence

This is essentially **solving the Hilbert-Pólya conjecture** plus additional structural constraints.

### 7.2 Probability of Success

| Approach | Feasibility | Potential Payoff |
|----------|-------------|------------------|
| Numerical verification | HIGH | LOW (already done) |
| Spectral operator search | MEDIUM | HIGH |
| Trace formula derivation | LOW | EXTREME |
| K₇ connection proof | VERY LOW | REVOLUTIONARY |

### 7.3 Recommended Path

**Phase 1 (Weeks 1-2)**: Rigorous unfolded metric analysis
- Compute error in units of local spacing
- If error >> 1 spacing, the recurrence is merely asymptotic

**Phase 2 (Weeks 2-4)**: Test function analysis
- Apply h_GIFT(t) to explicit formula numerically
- Check if prime side shows resonance at lags

**Phase 3 (Months 2-3)**: Operator search
- Use PINN to learn H with eigenvalues {γₙ}
- Check if H has structure compatible with K₇

**Phase 4 (If Phase 3 succeeds)**: Trace formula derivation
- Connect H to K₇ Laplacian
- Derive recurrence from geometry

---

## 8. Concrete Next Steps

### 8.1 Immediate Computations

```python
# Test: Fourier analysis of explicit formula at GIFT lags
import numpy as np
from scipy.fft import fft

def test_function_gift(t, lags=[5, 8, 13, 27]):
    """Test function sensitive to GIFT lags."""
    return sum(np.exp(-np.abs(t)/ell) for ell in lags)

# Apply to explicit formula (conceptual)
# Sum h(gamma_n) for n = 1 to N
# Compare with prime sum side
```

### 8.2 Questions to Answer

1. Does |∑ exp(2πi ℓ γₙ / T)| show peaks at ℓ ∈ {5,8,13,27}?
2. What is the error in unfolded units (spacings)?
3. Does the recurrence hold for other L-functions?

### 8.3 Literature to Consult

- [Connes-Consani 2024](https://arxiv.org/pdf/2310.18423) on prolate operators
- [Montgomery 1973](https://www.cambridge.org/core/journals/proceedings-of-the-london-mathematical-society/article/abs/on-the-pair-correlation-of-zeros-of-the-riemann-zetafunction/29A91CBF07210A94E8EEFEAFFBB9FDEE) on pair correlation
- [Goldston-Montgomery](https://www.semanticscholar.org/paper/532d303e855d71c7e6d129190956095b11766b34) on primes-zeros equivalence

---

## 9. Conclusions

### 9.1 The Honest Assessment

The Weil explicit formula provides the *framework* for understanding why zeros and primes are connected, but:

1. **No existing work** derives recurrence relations from it
2. **The [5,8,13,27] structure** may be asymptotic, not fundamental
3. **Coefficient instability** suggests we have an effective approximation, not an exact law

### 9.2 The Optimistic View

The Fibonacci structure (5+8=13, 5×8-13=27) is **too exact to be coincidence**. If this structure has meaning, it must emerge from:
- The Lie algebra dimension 8 = rank(E₈)
- The Jordan algebra dimension 27 = dim(J₃(O))
- The Fibonacci sequence (5, 8, 13 are F₅, F₆, F₇)

A deep connection to exceptional geometry remains *possible* but unproven.

### 9.3 The Path Forward

Before investing in trace formula derivations:
1. **Validate the unfolded metric** (is error < 1 spacing?)
2. **Test on Dirichlet L-functions** (universality check)
3. **Search for the operator H** numerically

If these succeed, the trace formula approach becomes justified. If they fail, the recurrence is an interesting numerical phenomenon without deep structural meaning.

---

## Appendix A: Key Formulas

### Riemann-von Mangoldt Formula
$$N(T) = \frac{T}{2\pi} \log\left(\frac{T}{2\pi}\right) - \frac{T}{2\pi} + O(\log T)$$

### Local Spacing
$$\Delta(T) \approx \frac{2\pi}{\log(T/2\pi)}$$

### Montgomery Pair Correlation
$$R_2(\alpha) = 1 - \left(\frac{\sin \pi \alpha}{\pi \alpha}\right)^2$$

### Weil Positivity Criterion
$$\sum_\gamma h(\gamma) \geq 0 \quad \text{for all } h = g * \tilde{g}$$

---

## References

1. [Weil Explicit Formula Overview](https://empslocal.ex.ac.uk/people/staff/mrwatkin//zeta/weilexplicitformula.htm)
2. [Selberg-Weil Connection](https://empslocal.ex.ac.uk/people/staff/mrwatkin/zeta/STF-WEF.htm)
3. [Connes: Trace Formula in NCG](https://alainconnes.org/wp-content/uploads/selecta.ps-2.pdf)
4. [Berry-Keating Eigenvalue Paper](https://empslocal.ex.ac.uk/people/staff/mrwatkin/zeta/berry-keating1.pdf)
5. [Hilbert-Polya Conjecture](https://en.wikipedia.org/wiki/Hilbert%E2%80%93P%C3%B3lya_conjecture)
6. [Montgomery Pair Correlation](https://en.wikipedia.org/wiki/Montgomery's_pair_correlation_conjecture)
7. [arXiv:2310.18423 - Prolate Wave Operators](https://arxiv.org/pdf/2310.18423)
8. [Pisano Periods](https://en.wikipedia.org/wiki/Pisano_period)
9. [Fibonacci and Modular Arithmetic](https://sumo.stanford.edu/pdfs/speakers/fibonacci.pdf)
10. [LeClair-Mussardo on Bethe Ansatz](https://link.springer.com/article/10.1007/JHEP04(2024)062)

---

*Document prepared for GIFT research collaboration*
*Status: Research-grade, not peer-reviewed*
