# Phase 1: Problem Setup

## 1.1 The TCS Construction

### Definition (Twisted Connected Sum)

A **TCS G₂ manifold** M_L is constructed from:

**Data:**
- (Z₊, ω₊, Ω₊): ACyl Calabi-Yau 3-fold asymptotic to ℝ₊ × S¹ × K3
- (Z₋, ω₋, Ω₋): ACyl Calabi-Yau 3-fold asymptotic to ℝ₊ × S¹ × K3
- r: K3 → K3 hyper-Kähler rotation (Donaldson matching)

**Construction:**

1. **Truncate:** For T > 0, let Z₊^T = {z ∈ Z₊ : t(z) ≤ T} where t is the cylindrical coordinate

2. **Glue:** Form
   $$M_L = Z_+^T \cup_\Phi ([-L, L] \times S^1 \times K3) \cup_\Psi Z_-^T$$

   where L = T - T₀ for fixed T₀, and Φ, Ψ are the asymptotic identifications

3. **G₂ structure:** The approximate G₂ form is
   $$\varphi_L = dt \wedge \omega_{K3} + \text{Re}(\Omega_{K3}) \wedge d\theta + \text{corrections}$$

### Metric Structure

On the neck region [-L, L] × S¹ × K3:

$$g_L = dt^2 + d\theta^2 + g_{K3}$$

where:
- t ∈ [-L, L] is the neck coordinate
- θ ∈ S¹ = ℝ/2πℤ
- g_{K3} is the Ricci-flat Kähler metric on K3

**Total dimension:** 1 + 1 + 4 + 1 (from CY3 structure) = 7 ✓

---

## 1.2 The Laplace-Beltrami Operator

### Definition

On a Riemannian manifold (M, g), the Laplace-Beltrami operator on functions is:

$$\Delta_g f = -\frac{1}{\sqrt{\det g}} \partial_i \left( \sqrt{\det g} \, g^{ij} \partial_j f \right)$$

**Sign convention:** We use the positive (geometer's) Laplacian, so Δ ≥ 0.

### On the Product Neck

For g_L = dt² + dθ² + g_{K3}:

$$\Delta_{g_L} = -\partial_t^2 - \partial_\theta^2 + \Delta_{K3}$$

This is a sum of commuting operators.

---

## 1.3 Function Spaces

### L² Space

$$L^2(M_L) = \left\{ f : M_L \to \mathbb{R} \, \Big| \, \int_{M_L} |f|^2 \, dV_{g_L} < \infty \right\}$$

### Sobolev Spaces

$$H^1(M_L) = \left\{ f \in L^2 \, \Big| \, \nabla f \in L^2 \right\}$$

$$H^2(M_L) = \left\{ f \in H^1 \, \Big| \, \Delta f \in L^2 \right\}$$

### Domain of Δ

$$\text{Dom}(\Delta) = H^2(M_L)$$

The Laplacian is self-adjoint on this domain (M_L is complete without boundary).

---

## 1.4 Eigenvalue Problem

### Spectral Problem

Find λ ∈ ℝ and f ∈ H²(M_L), f ≠ 0, such that:

$$\Delta_{g_L} f = \lambda f$$

### Properties

**Theorem (Standard spectral theory):**
1. Spectrum is discrete: 0 = λ₀ < λ₁ ≤ λ₂ ≤ ... → ∞
2. Eigenfunctions form orthonormal basis of L²
3. λ₀ = 0 has eigenspace = constants (M_L connected)
4. Min-max characterization:
   $$\lambda_k = \min_{\substack{V \subset H^1 \\ \dim V = k+1}} \max_{f \in V, f \neq 0} \frac{\int |\nabla f|^2}{\int |f|^2}$$

### Our Target

**Prove:** λ₁(M_L) = π²/L² + O(e^{-δL}) as L → ∞.

---

## 1.5 Decomposition of M_L

We decompose M_L into overlapping regions:

### Region A (Left End)
$$A = Z_+^T \cup ([-L, -L+1] \times S^1 \times K3)$$

### Region B (Neck)
$$B = [-L+1/2, L-1/2] \times S^1 \times K3$$

### Region C (Right End)
$$C = ([L-1, L] \times S^1 \times K3) \cup Z_-^T$$

### Overlap Regions
- A ∩ B: collar of width 1/2 on left
- B ∩ C: collar of width 1/2 on right

### Partition of Unity

Let {χ_A, χ_B, χ_C} be a smooth partition of unity:
- χ_A + χ_B + χ_C = 1
- supp(χ_A) ⊂ A, supp(χ_B) ⊂ B, supp(χ_C) ⊂ C
- |∇χ_i| ≤ C (uniform in L)
- |Δχ_i| ≤ C (uniform in L)

---

## 1.6 Cross-Section Analysis

### The Cross-Section Y = S¹ × K3

**Metric:** g_Y = dθ² + g_{K3}

**Laplacian:** Δ_Y = -∂_θ² + Δ_{K3}

### Spectrum of Δ_Y

**Eigenfunctions:** Products ψ(θ) ⊗ φ(x) where:
- ψ_n(θ) = e^{inθ}/√(2π) with eigenvalue n² on S¹
- φ_k(x) = eigenfunction of Δ_{K3} with eigenvalue μ_k

**Combined eigenvalue:** n² + μ_k

### Ordering

| Mode | n | μ_k | λ = n² + μ_k |
|------|---|-----|--------------|
| (0,0) | 0 | 0 | 0 |
| (±1,0) | ±1 | 0 | 1 |
| (0,1) | 0 | μ₁(K3) | μ₁(K3) |
| (±2,0) | ±2 | 0 | 4 |
| ... | ... | ... | ... |

**Key fact:** For K3 with standard normalization, μ₁(K3) > 1.

Therefore: **λ₁(Y) = 1** (from the S¹ modes ±1).

---

## 1.7 The Spectral Gap Condition

### Definition

The **cross-section spectral gap** is:

$$\gamma = \lambda_1(Y) = 1$$

### Significance

- Continuous spectrum of Δ on Z₊ (or Z₋) starts at γ
- On the finite neck, modes with transverse eigenvalue < γ can exist as discrete
- The "neck mode" we seek has transverse eigenvalue 0 (constant on Y)

### Decay Parameter

For exponential estimates, we use:

$$\delta = \sqrt{\gamma} = 1$$

Functions with Δf = λf, λ < γ, decay exponentially into the cylindrical ends with rate √(γ - λ).

---

## 1.8 Summary of Setup

We have established:

1. **Geometry:** M_L is a 7-manifold with neck of length 2L
2. **Operator:** Δ_{g_L} is positive self-adjoint with discrete spectrum
3. **Target:** Prove λ₁(M_L) ~ π²/L² as L → ∞
4. **Tool:** Cross-section gap γ = 1 controls decay

**Next:** Phase 2 analyzes the Laplacian on the neck region using Fourier decomposition.
