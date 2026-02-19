# Analytical Proof Strategy: λ₁ ~ π²/L² for TCS G₂ Manifolds

## Goal

**Theorem (Target):** Let M_L be a TCS G₂ manifold with neck length L. Then as L → ∞:

$$\lambda_1(M_L) = \frac{\pi^2}{L^2} + O(e^{-\delta L})$$

for some δ > 0 depending on the spectral gap of the cross-section.

---

## Proof Strategy Overview

### Phase 1: Problem Setup
- Define TCS structure precisely
- Identify the Laplace-Beltrami operator
- Decompose M_L into regions

### Phase 2: Cylindrical Analysis
- Fourier decomposition on S¹ × K3 cylinder
- Separation of variables
- Identify the "neck mode"

### Phase 3: Surgery Calculus
- Mazzeo-Melrose b-calculus framework
- Parametrix construction
- Resolvent estimates

### Phase 4: Eigenvalue Asymptotics
- Variational characterization
- Upper bound via test functions
- Lower bound via unique continuation

### Phase 5: Error Control
- Exponential decay of corrections
- Uniformity in geometric parameters

---

## Key Mathematical Ingredients

### 1. Spectral Theory on Manifolds with Cylindrical Ends

For a manifold M with cylindrical end [0,∞) × Y:
- The continuous spectrum starts at λ₀(Y) (first eigenvalue of Y)
- Discrete spectrum lies in [0, λ₀(Y))
- If λ₀(Y) > 0, there may be no discrete spectrum

### 2. TCS Specific Structure

M_L = M₊ ∪ ([-L,L] × S¹ × K3) ∪ M₋

where:
- M₊, M₋ are ACyl CY3 manifolds (asymptotic to cylinder)
- K3 is hyper-Kähler 4-manifold
- S¹ has period 2π

### 3. Cross-Section Spectrum

Y = S¹ × K3

Spectrum of Δ_Y:
- λ = 0 (constant mode on K3, constant on S¹)
- λ = 1 (first S¹ mode: sin(θ), cos(θ), with constant K3)
- λ = λ₁(K3) (first K3 mode, constant on S¹)
- etc.

**Key observation:** λ₁(Y) = min(1, λ₁(K3))

For K3 with standard metric, λ₁(K3) > 1, so λ₁(Y) = 1.

### 4. The Critical Mode

On the cylinder [-L, L] × Y, with Neumann BC at ends:

Δ(f(t)·1_Y) = -f''(t)·1_Y

The lowest Neumann eigenfunction on [-L, L] is:
- f₀(t) = 1 (eigenvalue 0)
- f₁(t) = cos(πt/L) (eigenvalue π²/L²)

This "neck mode" gives the candidate for λ₁(M_L).

---

## Literature Foundation

### Mazzeo-Melrose (1987-1998)
- b-calculus for manifolds with boundary
- Surgery calculus for degenerating families
- Resolvent and heat kernel asymptotics

### Cheeger (1983)
- Spectral convergence under collapse
- Eigenvalue bounds via isoperimetric constants

### Langlais (2024) [if available]
- Specific to TCS G₂ manifolds
- Neck-stretching eigenvalue behavior

### Colding-Minicozzi, Hein-Naber
- Spectral theory on special holonomy manifolds
- Regularity and decay estimates

---

## Detailed Phase Breakdown

See subsequent documents:
- `PHASE1_SETUP.md` - Problem formulation
- `PHASE2_CYLINDER.md` - Cylindrical analysis
- `PHASE3_SURGERY.md` - Surgery calculus
- `PHASE4_ASYMPTOTICS.md` - Main theorem proof
- `PHASE5_ERRORS.md` - Error estimates

---

## Success Criteria

The proof is complete when we establish:

1. **Upper bound:** ∃ test function ψ_L with Rayleigh quotient ≤ π²/L² + O(e^{-δL})

2. **Lower bound:** Any eigenfunction with λ < λ₁(Y) - ε concentrates on neck with eigenvalue ≥ π²/L² - O(e^{-δL})

3. **Gap:** Second eigenvalue λ₂(M_L) ≥ c > 0 independent of L (or λ₂ → λ₁(Y) as L → ∞)

---

## Connection to κ = π²/14

Once we prove λ₁ ~ π²/L², the selection principle follows:

1. GIFT requires λ₁ = 14/H* = 14/99 (topological constraint)

2. Setting π²/L² = 14/99:
   L² = 99π²/14
   L* = π√(99/14) ≈ 8.354

3. The selection functional κ = L*²/H* = π²/14

This connects dim(G₂) = 14 to the spectral geometry of K7.
