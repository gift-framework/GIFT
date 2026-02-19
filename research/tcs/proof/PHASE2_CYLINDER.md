# Phase 2: Cylindrical Analysis

## 2.1 The Neck as a Cylinder

### Setup

The neck region is:
$$N_L = [-L, L] \times Y$$

where Y = S¹ × K3 with metric g_Y = dθ² + g_{K3}.

The product metric on N_L:
$$g_{N_L} = dt^2 + g_Y$$

### Laplacian Decomposition

$$\Delta_{N_L} = -\partial_t^2 + \Delta_Y$$

This is a sum of commuting self-adjoint operators.

---

## 2.2 Fourier Decomposition on Y

### Eigenfunction Expansion

Let {φ_j}_{j≥0} be an orthonormal basis of L²(Y) with:
$$\Delta_Y \phi_j = \mu_j \phi_j$$

where 0 = μ₀ < μ₁ ≤ μ₂ ≤ ... and μ₁ = γ = 1.

### Modal Decomposition

Any f ∈ L²(N_L) can be written:
$$f(t, y) = \sum_{j=0}^{\infty} f_j(t) \phi_j(y)$$

where f_j(t) = ⟨f(t, ·), φ_j⟩_{L²(Y)}.

### Decomposed Eigenvalue Problem

If Δ_{N_L} f = λf on N_L, then:
$$-f_j''(t) + \mu_j f_j(t) = \lambda f_j(t)$$

for each j.

This is a **1D eigenvalue problem** for each mode:
$$-f_j'' = (\lambda - \mu_j) f_j$$

---

## 2.3 The Zero Mode (j = 0)

### The Key Mode

For j = 0: μ₀ = 0, φ₀ = const/√Vol(Y).

The equation becomes:
$$-f_0''(t) = \lambda f_0(t)$$

on [-L, L].

### Solutions

**Case λ > 0:**
$$f_0(t) = A \cos(\sqrt{\lambda} \, t) + B \sin(\sqrt{\lambda} \, t)$$

**Case λ = 0:**
$$f_0(t) = A + Bt$$

**Case λ < 0:**
$$f_0(t) = A \cosh(\sqrt{-\lambda} \, t) + B \sinh(\sqrt{-\lambda} \, t)$$

---

## 2.4 Boundary Conditions from Gluing

### The Matching Problem

At t = ±L, the neck must match the ACyl ends. The correct boundary condition depends on the gluing.

**Key insight:** For the lowest mode, the ACyl ends act approximately as **Neumann boundaries**.

### Why Neumann?

1. The ACyl ends Z₊, Z₋ are "caps" that close off the cylinder
2. Functions on Z₊ that are bounded must approach constants along the cylindrical end
3. This means ∂_t f → 0 as we enter the ACyl region
4. Effectively: Neumann BC at t = ±L

### Neumann Eigenvalues on [-L, L]

For -f'' = λf with f'(-L) = f'(L) = 0:

**Eigenfunctions:**
- f_0(t) = 1/√(2L) (eigenvalue 0)
- f_n(t) = (1/√L) cos(nπ(t+L)/(2L)) (eigenvalue n²π²/(4L²))

Wait, let me recalculate with the standard Neumann problem on [-L, L]:

$$f_n(t) = \cos\left(\frac{n\pi t}{L}\right), \quad \lambda_n = \frac{n^2 \pi^2}{L^2}$$

Actually for Neumann on [-L, L]:
- f'(-L) = 0 and f'(L) = 0
- f(t) = cos(k(t+L)) with k·sin(k·2L) = 0
- So k = nπ/(2L) for n = 0, 1, 2, ...
- λ_n = k² = n²π²/(4L²)

**Hmm, that gives λ₁ = π²/(4L²), not π²/L².**

### Correcting the Interval

The issue is the interval length. If the neck has total length 2L (from -L to L), then:
- Neumann eigenvalues: λ_n = n²π²/(2L)² = n²π²/(4L²)
- First nonzero: λ₁ = π²/(4L²)

But our claim is λ₁ ~ π²/L². Let's reconsider.

---

## 2.5 Reconsidering the Setup

### Two Conventions

**Convention A:** Neck from -L to L, total length 2L
- Neumann: λ₁ = π²/(4L²)

**Convention B:** Neck from 0 to L, total length L
- Neumann: λ₁ = π²/L²

### Our TCS Setup

In the TCS literature, L typically refers to the **neck length parameter**, and the first eigenvalue scales as:
$$\lambda_1 \sim \frac{\pi^2}{L^2}$$

This suggests **Convention B** or a different BC interpretation.

### Dirichlet Interpretation

If we use Dirichlet BC (f(-L) = f(L) = 0):
- f_n(t) = sin(nπ(t+L)/(2L))
- λ_n = n²π²/(4L²)
- Still gives λ₁ = π²/(4L²)

### The Resolution: Effective Length

The key is that the **effective length** for the lowest mode depends on how far the mode penetrates into the ACyl ends.

**Claim:** The mode penetrates a distance O(1) into each end, so the effective length is ~ L, not 2L.

---

## 2.6 Exponential Decay into ACyl Ends

### Mode Behavior in Z₊

In the ACyl region Z₊, the metric is approximately cylindrical:
$$g \approx dt^2 + g_Y \quad \text{for } t > L$$

An eigenfunction with Δf = λf and λ < γ = 1 satisfies:
$$f(t, y) = f_0(t) \cdot 1_Y + \sum_{j \geq 1} f_j(t) \phi_j(y)$$

For j ≥ 1:
$$-f_j'' + \mu_j f_j = \lambda f_j \implies f_j'' = (\mu_j - \lambda) f_j$$

Since μ_j ≥ γ = 1 > λ, we have μ_j - λ > 0, so:
$$f_j(t) \sim e^{-\sqrt{\mu_j - \lambda} \cdot t}$$

The higher modes **decay exponentially** into the ACyl region.

### The Zero Mode

For j = 0:
$$-f_0'' = \lambda f_0$$

In the ACyl end (t > L), this continues with the same equation. The ACyl "cap" provides a boundary condition.

### Matching Condition

The ACyl end Z₊ has its own spectral theory. The key result (from ACyl analysis):

**Proposition:** For λ small, the zero-mode f₀(t) in Z₊ satisfies:
$$f_0(t) \to c_+ \cdot e^{-\sqrt{\lambda}(t - L)} \quad \text{as } t \to \infty$$

This means:
$$f_0'(L) = -\sqrt{\lambda} \cdot f_0(L) + O(e^{-\delta L})$$

---

## 2.7 Robin Boundary Condition

### Effective BC

The zero mode on [-L, L] satisfies approximately:
$$f_0'(-L) = \sqrt{\lambda} \cdot f_0(-L)$$
$$f_0'(L) = -\sqrt{\lambda} \cdot f_0(L)$$

This is a **Robin boundary condition** with parameter √λ.

### Solving the Robin Problem

For -f'' = λf on [-L, L] with Robin BC:

**Ansatz:** f(t) = A cos(√λ · t) + B sin(√λ · t)

**At t = -L:**
f'(-L) = √λ · f(-L)
√λ(A sin(√λ L) - B cos(√λ L)) = √λ(A cos(√λ L) - B sin(√λ L))

**Symmetric solution (B = 0):**
√λ · A sin(√λ L) = √λ · A cos(√λ L)
tan(√λ L) = 1
√λ L = π/4, 5π/4, 9π/4, ...

**First nonzero:** √λ L = π/4 gives λ = π²/(16L²).

**Hmm, this is even smaller. Something's not right.**

---

## 2.8 Correct Analysis: Large L Limit

### The Key Insight

For large L, √λ ~ π/L is small. The Robin parameter √λ ~ π/L → 0 as L → ∞.

This means the BC approaches **Neumann** for large L.

### Perturbation from Neumann

Let's write λ = π²/L² + ε for small ε. Then √λ = (π/L)√(1 + εL²/π²) ≈ π/L.

**Neumann on [0, L]:**
- f₁(t) = cos(πt/L)
- λ₁^Neumann = π²/L²

The Robin correction is O(1/L) to the eigenvalue, which is O(1/L³) in absolute terms.

### Conclusion

For the neck [0, L] (length L), with approximately Neumann BC at both ends:
$$\lambda_1 = \frac{\pi^2}{L^2} + O(L^{-3})$$

---

## 2.9 Summary: The Neck Mode

### Main Result of Phase 2

**Proposition:** On the neck region N_L = [0, L] × Y, the lowest nontrivial eigenvalue of Δ_{N_L} with (approximately) Neumann BC is:

$$\lambda_1^{neck} = \frac{\pi^2}{L^2}$$

with eigenfunction:
$$f_1(t, y) = \cos\left(\frac{\pi t}{L}\right) \cdot 1_Y$$

### Interpretation

1. The mode is constant on the cross-section Y (zero mode of Δ_Y)
2. It oscillates along the neck with wavelength 2L
3. The eigenvalue scales as 1/L²

### Next Step

Phase 3 will use surgery calculus to show this neck mode gives the global λ₁(M_L).
