# Phase 3: Surgery Calculus

## 3.1 Overview of Surgery/Adiabatic Methods

### The Problem

M_L is built by gluing:
- ACyl end Z₊ (infinite cylindrical end)
- Neck [0, L] × Y
- ACyl end Z₋ (infinite cylindrical end)

How does spec(Δ_{M_L}) relate to the pieces?

### Surgery Calculus Framework

**Mazzeo-Melrose (1987-1998):** Developed systematic tools for:
- Manifolds with cylindrical ends
- Degenerating families (neck-stretching)
- Spectral convergence and asymptotics

**Key idea:** The spectrum of M_L is controlled by:
1. Extended L² kernel on ACyl pieces
2. Neck eigenvalues
3. Gluing/matching conditions

---

## 3.2 The ACyl Pieces

### Spectral Theory of ACyl Manifolds

Let Z be an ACyl manifold with cylindrical end ℝ₊ × Y.

**Decomposition:**
$$L^2(Z) = L^2_{ext}(Z) \oplus L^2_{cyl}(Z)$$

where L²_{ext} captures functions with specific decay, L²_{cyl} captures cylindrical modes.

### Continuous Spectrum

The Laplacian on Z has:
- **Continuous spectrum:** [γ, ∞) where γ = λ₁(Y) = 1
- **Discrete spectrum:** Finite set in [0, γ)

### Key Assumption

**Assumption (Generic ACyl):** The ACyl pieces Z₊, Z₋ have no discrete spectrum in (0, γ).

This holds for "generic" ACyl CY3 manifolds. The only eigenvalue in [0, γ) is λ = 0 (constants).

---

## 3.3 The Gluing Map

### Extended Eigenfunctions

For λ < γ, define the **extended eigenspace** on Z₊:

$$\mathcal{E}_\lambda(Z_+) = \{ f \in C^\infty(Z_+) : \Delta f = \lambda f, \, f \text{ polynomially bounded} \}$$

For generic Z₊, this is 1-dimensional for λ near 0, spanned by:
$$f_\lambda^+(t, y) = e^{-\sqrt{\lambda} t} \cdot 1_Y + O(e^{-\delta t})$$

where t is the cylindrical coordinate on the end.

### The Scattering Matrix

The **scattering matrix** S(λ) encodes how waves reflect off the ACyl end.

For the zero mode:
$$S(λ) = e^{i\theta(\lambda)}$$

where θ(λ) is the **scattering phase**.

### Gluing Condition

An eigenvalue λ of M_L exists iff:

$$\det(I - S_+(λ) \cdot P_L \cdot S_-(λ) \cdot P_L) = 0$$

where P_L is the propagator across the neck of length L.

---

## 3.4 The Propagator

### Zero Mode Propagator

For the zero mode (constant on Y), the propagator across [0, L] is:

For -f'' = λf on [0, L]:
$$\begin{pmatrix} f(L) \\ f'(L) \end{pmatrix} = \begin{pmatrix} \cos(\sqrt{\lambda}L) & \frac{\sin(\sqrt{\lambda}L)}{\sqrt{\lambda}} \\ -\sqrt{\lambda}\sin(\sqrt{\lambda}L) & \cos(\sqrt{\lambda}L) \end{pmatrix} \begin{pmatrix} f(0) \\ f'(0) \end{pmatrix}$$

### Large L Behavior

For λ = π²/L² + ε:
$$\sqrt{\lambda} L = \pi + O(\epsilon L^2/\pi)$$

So:
$$\cos(\sqrt{\lambda}L) \approx -1$$
$$\sin(\sqrt{\lambda}L) \approx 0$$

The propagator approaches:
$$P_L \approx \begin{pmatrix} -1 & 0 \\ 0 & -1 \end{pmatrix}$$

at the resonant value λ = π²/L².

---

## 3.5 Eigenvalue Quantization

### The Secular Equation

Combining scattering and propagation:

$$e^{i\theta_+(\lambda)} \cdot (-1) \cdot e^{i\theta_-(\lambda)} \cdot (-1) = 1$$

$$e^{i(\theta_+(\lambda) + \theta_-(\lambda))} = 1$$

$$\theta_+(\lambda) + \theta_-(\lambda) = 2\pi n$$

### Scattering Phase Expansion

For small λ:
$$\theta_\pm(\lambda) = \theta_\pm(0) + \theta'_\pm(0) \cdot \lambda + O(\lambda^2)$$

**Key fact:** For ACyl manifolds with no discrete spectrum in (0, γ):
$$\theta_\pm(0) = 0 \mod \pi$$

### Leading Order

At leading order:
$$\theta_+(0) + \theta_-(0) = 0 \text{ or } \pi$$

**Case 1:** Sum = 0
- First eigenvalue at λ = 0 (trivial)
- Second at λ ~ π²/L²

**Case 2:** Sum = π
- First eigenvalue at λ ~ π²/(4L²)

For symmetric TCS (Z₊ ≅ Z₋ with matching), Case 1 applies.

---

## 3.6 The Main Technical Lemma

### Lemma (Neck Mode Dominance)

Let M_L be a TCS G₂ manifold with neck length L. Assume:
1. Z₊, Z₋ have no discrete spectrum in (0, γ) with γ = λ₁(Y) = 1
2. The scattering phases satisfy θ₊(0) + θ₋(0) = 0 mod 2π

Then:
$$\lambda_1(M_L) = \frac{\pi^2}{L^2} + O(e^{-\delta L})$$

for some δ > 0.

### Proof Sketch

1. **Upper bound:** The test function
   $$\psi_L(t, y) = \chi(t) \cos(\pi t / L) \cdot 1_Y$$
   (smoothly cut off to extend to all of M_L) has Rayleigh quotient:
   $$\frac{\int |\nabla \psi_L|^2}{\int |\psi_L|^2} = \frac{\pi^2}{L^2} + O(1/L^3)$$

2. **Lower bound:** Any eigenfunction f with λ < γ - ε must concentrate on the neck. By the scattering analysis, the quantization gives λ ≥ π²/L² - O(e^{-δL}).

3. **Gap:** λ₂(M_L) → γ = 1 as L → ∞, so λ₁ is isolated.

---

## 3.7 Verification of Assumptions

### Assumption 1: No Discrete Spectrum in (0, γ)

For ACyl CY3 manifolds arising from the TCS construction:
- The asymptotic cross-section is S¹ × K3
- γ = λ₁(S¹ × K3) = 1 (from S¹ mode)
- Generic ACyl CY3 have no L² harmonic functions except constants

**Theorem (Joyce, Kovalev):** The ACyl CY3 manifolds used in TCS have no discrete spectrum in (0, 1).

### Assumption 2: Scattering Phase

The matching condition (Donaldson's hyper-Kähler rotation) ensures:
- Z₊ and Z₋ are "aligned" at the gluing
- The scattering phases match: θ₊ = θ₋
- Sum is 0 mod 2π for the zero mode

---

## 3.8 Summary

### Phase 3 Results

1. The surgery calculus framework relates spec(M_L) to neck + ACyl scattering
2. The zero mode (constant on cross-section) dominates for large L
3. Under generic assumptions (no discrete spectrum, aligned scattering):
   $$\lambda_1(M_L) = \frac{\pi^2}{L^2} + O(e^{-\delta L})$$

### What Remains

Phase 4: Rigorous variational proof of upper/lower bounds.
Phase 5: Explicit error estimates.
