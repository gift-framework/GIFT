# Dimensional Observables (EXPLORATORY)

> **STATUS: EXPLORATORY**
>
> This document contains **heuristic formulas** for absolute masses in GeV/MeV. These formulas work numerically but **lack complete topological justification**. Use with caution.
>
> **Key Limitations:**
> - Electron mass m_e is an **INPUT** (not predicted)
> - Several formulas mix topological constants with fitting
> - The transition from dimensionless → dimensional is **not rigorous**
> - This content is **not included in Zenodo publication**

---

## Absolute Masses, Scale Bridge, and Cosmological Parameters

*This supplement extends the dimensionless predictions of the main document to absolute mass scales.*

**Version**: 3.0
**Date**: December 2025

---

## Table of Contents

- [Part I: The Scale Bridge](#part-i-the-scale-bridge)
- [Part II: Absolute Fermion Masses](#part-ii-absolute-fermion-masses)
- [Part III: Boson Masses](#part-iii-boson-masses)
- [Part IV: Limitations](#part-iv-limitations)

---

# Part I: The Scale Bridge

## 1. Dimensional Transmutation Problem

### 1.1 The Challenge

**Problem**: How do dimensionless topological numbers acquire dimensions (GeV)?

The GIFT framework predicts many dimensionless ratios exactly (e.g., m_s/m_d = 20), but connecting these to absolute masses requires a dimensional scale.

### 1.2 Natural Scales

The framework contains several natural scales:
- Planck mass: M_Pl ~ 10¹⁹ GeV
- String scale: M_s ~ M_Pl / e⁸ ~ 10¹⁶ GeV
- Electroweak scale: v ~ 246 GeV

---

## 2. The Λ_GIFT Structure

### 2.1 Formula

$$\Lambda_{GIFT} = \frac{21 \cdot e^8 \cdot 248}{7 \cdot \pi^4}$$

### 2.2 Components

- **21** = b₂(K₇): Gauge cohomology
- **e⁸** = exp(rank(E₈)): Exponential hierarchy factor
- **248** = dim(E₈): Gauge dimension
- **7** = dim(K₇): Manifold dimension

### 2.3 Reference Scale

The electron mass m_e serves as reference:
- Most precisely measured fermion mass
- Stable particle
- All other masses expressed as ratios × m_e

**Important**: m_e = 0.511 MeV is an **INPUT**, not predicted.

---

# Part II: Absolute Fermion Masses

## 3. Lepton Masses

### 3.1 Electron Mass (Reference - INPUT)

$$m_e = 0.51099895 \text{ MeV}$$

### 3.2 Muon Mass

**From ratio**: m_μ/m_e = 27^φ = 207.012

$$m_\mu = 207.012 \times m_e = 105.78 \text{ MeV}$$

**Experimental**: 105.658 MeV (deviation 0.12%)

### 3.3 Tau Mass

**From ratio**: m_τ/m_e = 3477 (PROVEN)

$$m_\tau = 3477 \times m_e = 1776.87 \text{ MeV}$$

**Experimental**: 1776.86 MeV (deviation 0.004%)

---

## 4. Quark Masses (HEURISTIC)

> **Warning**: These formulas are heuristic and should be treated as exploratory.

### 4.1 Light Quarks

| Quark | Formula | GIFT (MeV) | PDG (MeV) | Deviation |
|-------|---------|------------|-----------|-----------|
| u | √(14/3) × MeV | 2.16 | 2.16 ± 0.07 | 0.0% |
| d | log(107) × MeV | 4.67 | 4.67 ± 0.09 | 0.0% |
| s | 24×τ × MeV | 93.5 | 93.4 ± 0.8 | 0.1% |

### 4.2 Heavy Quarks

| Quark | Formula | GIFT (GeV) | PDG (GeV) | Deviation |
|-------|---------|------------|-----------|-----------|
| c | (14-π)³ × 0.1 | 1.280 | 1.27 ± 0.02 | 0.8% |
| b | 42×99 × MeV | 4.158 | 4.18 ± 0.03 | 0.5% |
| t | (496/3)^ξ | 173.1 | 173.1 ± 0.6 | 0.0% |

---

## 5. Neutrino Masses

### 5.1 Mass Sum (EXPLORATORY)

$$\Sigma m_\nu = 0.0587 \text{ eV}$$

**Current bound**: Σm_ν < 0.12 eV (consistent)

### 5.2 Individual Masses (EXPLORATORY)

| Neutrino | Mass (eV) |
|----------|-----------|
| m₁ | ~0.001 |
| m₂ | ~0.009 |
| m₃ | ~0.05 |

---

# Part III: Boson Masses

## 6. W and Z Masses

### 6.1 W Boson Mass

$$M_W = \frac{v}{2} \cdot g_2 = 80.38 \text{ GeV}$$

**Experimental**: 80.377 ± 0.012 GeV (deviation 0.004%)

### 6.2 Z Boson Mass

Using sin²θ_W = 3/13:
$$M_Z = M_W \cdot \sqrt{\frac{13}{10}} = 91.19 \text{ GeV}$$

**Experimental**: 91.188 GeV (deviation 0.002%)

---

## 7. Higgs Mass

### 7.1 From λ_H = √17/32 (PROVEN)

$$m_H = \sqrt{2\lambda_H} \cdot v = 125.09 \text{ GeV}$$

**Experimental**: 125.25 ± 0.17 GeV (deviation 0.13%)

---

# Part IV: Limitations

## 8. What GIFT Predicts vs. Assumes

### 8.1 Predicted (Dimensionless)

- All mass ratios
- Gauge couplings at M_Z
- Mixing angles and phases
- Cosmological ratios

### 8.2 Assumed (Dimensional)

- Reference scale (m_e or v)
- Fundamental constants (c, ℏ, G)

---

## 9. Theoretical Uncertainties

### 9.1 Higher-Order Corrections

- QCD corrections to quark masses
- Electroweak radiative corrections
- Threshold effects at mass scales

### 9.2 Non-Perturbative Effects

- Confinement corrections to light quarks
- Instanton contributions

---

## 10. Summary

**This supplement should NOT be used for:**
- Rigorous predictions
- Zenodo publication claims
- Falsification criteria

**This supplement IS useful for:**
- Phenomenological exploration
- Pattern recognition
- Future research directions

---

## References

1. Particle Data Group (2024). *Review of Particle Physics*
2. Planck Collaboration (2020). Cosmological parameters
3. Lattice QCD FLAG review (2024). Quark masses

---

*GIFT Framework v3.0 - Exploratory Content*
*Status: HEURISTIC - Not part of Zenodo publication*
