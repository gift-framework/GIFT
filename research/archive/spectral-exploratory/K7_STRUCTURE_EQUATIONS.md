# K₇ Structure Equations

**Date**: 2026-01-26
**Status**: Step 1 of Rigorous Construction

---

## 1. Base Manifold

**Ansatz**: M⁷ = S³ × S³ × S¹ (before quotient)

- First S³: parametrized by Euler angles (ψ₁, θ₁, φ₁)
- Second S³: parametrized by Euler angles (ψ₂, θ₂, φ₂)
- S¹: parametrized by θ ∈ [0, 2π)

---

## 2. Left-Invariant Forms on S³ ≃ SU(2)

### First S³: Forms {σⁱ}

The left-invariant Maurer-Cartan forms on SU(2):

$$\sigma^1 = \cos\psi_1 \, d\theta_1 + \sin\psi_1 \sin\theta_1 \, d\phi_1$$

$$\sigma^2 = \sin\psi_1 \, d\theta_1 - \cos\psi_1 \sin\theta_1 \, d\phi_1$$

$$\sigma^3 = d\psi_1 + \cos\theta_1 \, d\phi_1$$

### Structure Equations (First S³)

$$d\sigma^1 = -\sigma^2 \wedge \sigma^3$$

$$d\sigma^2 = -\sigma^3 \wedge \sigma^1$$

$$d\sigma^3 = -\sigma^1 \wedge \sigma^2$$

**Compact form**: $d\sigma^i = -\frac{1}{2} \varepsilon^{ijk} \sigma^j \wedge \sigma^k$

### Second S³: Forms {Σⁱ}

Identical structure with coordinates (ψ₂, θ₂, φ₂):

$$\Sigma^1 = \cos\psi_2 \, d\theta_2 + \sin\psi_2 \sin\theta_2 \, d\phi_2$$

$$\Sigma^2 = \sin\psi_2 \, d\theta_2 - \cos\psi_2 \sin\theta_2 \, d\phi_2$$

$$\Sigma^3 = d\psi_2 + \cos\theta_2 \, d\phi_2$$

### Structure Equations (Second S³)

$$d\Sigma^1 = -\Sigma^2 \wedge \Sigma^3$$

$$d\Sigma^2 = -\Sigma^3 \wedge \Sigma^1$$

$$d\Sigma^3 = -\Sigma^1 \wedge \Sigma^2$$

---

## 3. The Coframe

### Definition

We define an orthonormal coframe {e¹, e², e³, e⁴, e⁵, e⁶, e⁷}:

$$e^1 = a \, \sigma^1, \quad e^2 = a \, \sigma^2, \quad e^3 = a \, \sigma^3$$

$$e^4 = b \, \Sigma^1, \quad e^5 = b \, \Sigma^2, \quad e^6 = b \, \Sigma^3$$

$$e^7 = c \, (d\theta + A)$$

where:
- **a** = radius of first S³
- **b** = radius of second S³
- **c** = radius of S¹ (with twist)
- **A** = connection 1-form on S³ × S³

### The Metric

$$g = (e^1)^2 + (e^2)^2 + (e^3)^2 + (e^4)^2 + (e^5)^2 + (e^6)^2 + (e^7)^2$$

$$= a^2 \left[(\sigma^1)^2 + (\sigma^2)^2 + (\sigma^3)^2\right] + b^2 \left[(\Sigma^1)^2 + (\Sigma^2)^2 + (\Sigma^3)^2\right] + c^2 (d\theta + A)^2$$

### Determinant

$$\det(g) = a^6 \cdot b^6 \cdot c^2$$

**GIFT constraint**: $\det(g) = 65/32$

---

## 4. Connection Ansatz

### General Form

The most general bi-invariant 1-form A on S³ × S³:

$$A = \alpha_1 \sigma^1 + \alpha_2 \sigma^2 + \alpha_3 \sigma^3 + \beta_1 \Sigma^1 + \beta_2 \Sigma^2 + \beta_3 \Sigma^3$$

### Symmetric Ansatz

For G₂ symmetry, we impose:
- Equal coupling to both S³ factors
- Alignment with Hopf fibers (σ³, Σ³)

$$A = \alpha \, (\sigma^3 + \Sigma^3)$$

This is the **diagonal Hopf connection**.

### Alternative: Anti-diagonal

$$A = \alpha \, (\sigma^3 - \Sigma^3)$$

We'll compute both cases.

---

## 5. Structure Equations for Coframe

### Forms e¹, e², e³ (First S³)

From $e^i = a \, \sigma^i$ and $d\sigma^i = -\frac{1}{2}\varepsilon^{ijk}\sigma^j \wedge \sigma^k$:

$$de^1 = a \, d\sigma^1 = -a \, \sigma^2 \wedge \sigma^3 = -\frac{1}{a} e^2 \wedge e^3$$

$$de^2 = -\frac{1}{a} e^3 \wedge e^1$$

$$de^3 = -\frac{1}{a} e^1 \wedge e^2$$

**Compact**: $de^i = -\frac{1}{a} \varepsilon^{ijk} e^j \wedge e^k$ for $i \in \{1,2,3\}$

### Forms e⁴, e⁵, e⁶ (Second S³)

Similarly:

$$de^4 = -\frac{1}{b} e^5 \wedge e^6$$

$$de^5 = -\frac{1}{b} e^6 \wedge e^4$$

$$de^6 = -\frac{1}{b} e^4 \wedge e^5$$

### Form e⁷ (Twisted S¹)

$$de^7 = c \, dA$$

With $A = \alpha(\sigma^3 + \Sigma^3)$:

$$dA = \alpha \, (d\sigma^3 + d\Sigma^3) = -\alpha \, (\sigma^1 \wedge \sigma^2 + \Sigma^1 \wedge \Sigma^2)$$

$$= -\frac{\alpha}{a^2} e^1 \wedge e^2 - \frac{\alpha}{b^2} e^4 \wedge e^5$$

Therefore:

$$de^7 = -\frac{c\alpha}{a^2} e^1 \wedge e^2 - \frac{c\alpha}{b^2} e^4 \wedge e^5$$

---

## 6. Summary: Structure Equations

$$\boxed{
\begin{aligned}
de^1 &= -\frac{1}{a} e^2 \wedge e^3 \\
de^2 &= -\frac{1}{a} e^3 \wedge e^1 \\
de^3 &= -\frac{1}{a} e^1 \wedge e^2 \\
de^4 &= -\frac{1}{b} e^5 \wedge e^6 \\
de^5 &= -\frac{1}{b} e^6 \wedge e^4 \\
de^6 &= -\frac{1}{b} e^4 \wedge e^5 \\
de^7 &= -\frac{c\alpha}{a^2} e^1 \wedge e^2 - \frac{c\alpha}{b^2} e^4 \wedge e^5
\end{aligned}
}$$

### Parameters

| Parameter | Meaning | GIFT Constraint |
|-----------|---------|-----------------|
| a | First S³ radius | From det(g) and λ₁ |
| b | Second S³ radius | From det(g) and λ₁ |
| c | S¹ radius | c² = H*/rank(E₈) = 99/8 |
| α | Connection strength | From dφ = 0 |

---

## 7. Alternative: Anti-Diagonal Connection

With $A = \alpha(\sigma^3 - \Sigma^3)$:

$$dA = -\alpha(\sigma^1 \wedge \sigma^2 - \Sigma^1 \wedge \Sigma^2)$$

$$de^7 = -\frac{c\alpha}{a^2} e^1 \wedge e^2 + \frac{c\alpha}{b^2} e^4 \wedge e^5$$

This gives different torsion constraints.

---

## 8. Notation for Wedge Products

For brevity, we use:
- $e^{ij} = e^i \wedge e^j$
- $e^{ijk} = e^i \wedge e^j \wedge e^k$

The structure equations become:

$$de^1 = -\frac{1}{a} e^{23}, \quad de^2 = -\frac{1}{a} e^{31}, \quad de^3 = -\frac{1}{a} e^{12}$$

$$de^4 = -\frac{1}{b} e^{56}, \quad de^5 = -\frac{1}{b} e^{64}, \quad de^6 = -\frac{1}{b} e^{45}$$

$$de^7 = -\frac{c\alpha}{a^2} e^{12} - \frac{c\alpha}{b^2} e^{45}$$

---

## 9. Useful Identities

### Volume Form

$$\text{vol}_7 = e^1 \wedge e^2 \wedge e^3 \wedge e^4 \wedge e^5 \wedge e^6 \wedge e^7 = e^{1234567}$$

### Hodge Star

For 3-forms in 7D with standard orientation:
$$*(e^{ijk}) = \pm e^{lmnp}$$
where {i,j,k,l,m,n,p} is a permutation of {1,2,3,4,5,6,7}.

---

## 10. Next Step

With structure equations established, proceed to:

**Step 2**: Write the G₂ 3-form φ explicitly in this coframe.

---

*GIFT Framework — Structure Equations*
*Step 1 Complete*
