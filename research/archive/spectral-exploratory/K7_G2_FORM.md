# K₇ G₂ 3-Form φ

**Date**: 2026-01-26
**Status**: Step 2 of Rigorous Construction

---

## 1. The Standard G₂ 3-Form

### Definition

In an orthonormal coframe {e¹,...,e⁷}, the canonical G₂ 3-form is:

$$\phi = e^{127} + e^{347} + e^{567} + e^{135} - e^{146} - e^{236} - e^{245}$$

This form is characterized by:
- Stabilizer in GL(7,ℝ) is exactly G₂
- Determines a metric and orientation
- The 7 terms correspond to the 7 lines of the Fano plane

### Expanded Form

$$\phi = e^1 \wedge e^2 \wedge e^7 + e^3 \wedge e^4 \wedge e^7 + e^5 \wedge e^6 \wedge e^7$$
$$+ e^1 \wedge e^3 \wedge e^5 - e^1 \wedge e^4 \wedge e^6 - e^2 \wedge e^3 \wedge e^6 - e^2 \wedge e^4 \wedge e^5$$

---

## 2. Substituting the Coframe

### From Structure Equations

Recall:
- $e^i = a \, \sigma^i$ for i = 1,2,3
- $e^{i+3} = b \, \Sigma^i$ for i = 1,2,3
- $e^7 = c \, (d\theta + A)$

### Explicit φ

$$\phi = a^2 c \, \sigma^1 \wedge \sigma^2 \wedge (d\theta + A)$$
$$+ ab c \, \sigma^3 \wedge \Sigma^1 \wedge (d\theta + A)$$
$$+ b^2 c \, \Sigma^2 \wedge \Sigma^3 \wedge (d\theta + A)$$
$$+ a^2 b \, \sigma^1 \wedge \sigma^3 \wedge \Sigma^2$$
$$- ab^2 \, \sigma^1 \wedge \Sigma^1 \wedge \Sigma^3$$
$$- a^2 b \, \sigma^2 \wedge \sigma^3 \wedge \Sigma^3$$
$$- ab^2 \, \sigma^2 \wedge \Sigma^1 \wedge \Sigma^2$$

### Grouping by Structure

**Terms with dθ** (S¹ direction):

$$\phi_{dθ} = \left( a^2 \sigma^{12} + ab \sigma^3 \Sigma^1 + b^2 \Sigma^{23} \right) \wedge c \, d\theta$$

**Terms with A** (connection contribution):

$$\phi_A = \left( a^2 \sigma^{12} + ab \sigma^3 \Sigma^1 + b^2 \Sigma^{23} \right) \wedge c \, A$$

**Pure S³ × S³ terms**:

$$\phi_0 = a^2 b \, \sigma^{13} \wedge \Sigma^2 - ab^2 \, \sigma^1 \wedge \Sigma^{13} - a^2 b \, \sigma^{23} \wedge \Sigma^3 - ab^2 \, \sigma^2 \wedge \Sigma^{12}$$

---

## 3. The Coassociative 4-Form *φ

### Definition

The Hodge dual of φ with respect to the G₂ metric:

$$*\phi = e^{3456} + e^{1256} + e^{1234} + e^{2467} - e^{2357} - e^{1457} - e^{1367}$$

### Expanded Form

$$*\phi = e^3 \wedge e^4 \wedge e^5 \wedge e^6 + e^1 \wedge e^2 \wedge e^5 \wedge e^6 + e^1 \wedge e^2 \wedge e^3 \wedge e^4$$
$$+ e^2 \wedge e^4 \wedge e^6 \wedge e^7 - e^2 \wedge e^3 \wedge e^5 \wedge e^7 - e^1 \wedge e^4 \wedge e^5 \wedge e^7 - e^1 \wedge e^3 \wedge e^6 \wedge e^7$$

### Substituting Coframe

$$*\phi = ab^3 \, \sigma^3 \wedge \Sigma^{123}$$
$$+ a^2 b^2 \, \sigma^{12} \wedge \Sigma^{23}$$
$$+ a^3 b \, \sigma^{123} \wedge \Sigma^1$$
$$+ ab^2 c \, \sigma^2 \wedge \Sigma^{13} \wedge (d\theta + A)$$
$$- a^2 bc \, \sigma^{23} \wedge \Sigma^2 \wedge (d\theta + A)$$
$$- ab^2 c \, \sigma^1 \wedge \Sigma^{12} \wedge (d\theta + A)$$
$$- a^2 bc \, \sigma^{13} \wedge \Sigma^3 \wedge (d\theta + A)$$

---

## 4. Key Observation: Matching Coefficients

### For G₂ Holonomy

The metric g is determined by φ via:

$$g_{ij} = \frac{1}{6} \iota_{e_i} \phi \wedge \iota_{e_j} \phi \wedge \phi$$

For our ansatz to give a consistent G₂ structure, the coefficients must satisfy certain relations.

### The Equal Radii Case

**Special case**: a = b (equal S³ radii)

This simplifies many expressions and is natural for symmetric TCS.

With a = b:

$$\phi = a^2 c \, (\sigma^{12} + \sigma^3 \Sigma^1 + \Sigma^{23}) \wedge (d\theta + A)$$
$$+ a^3 \, (\sigma^{13} \Sigma^2 - \sigma^1 \Sigma^{13} - \sigma^{23} \Sigma^3 - \sigma^2 \Sigma^{12})$$

---

## 5. The 2-Forms ω

### Define Auxiliary 2-Forms

Let:
$$\omega_1 = e^{23} = a^2 \sigma^{23}$$
$$\omega_2 = e^{56} = b^2 \Sigma^{23}$$
$$\omega_3 = e^{12} = a^2 \sigma^{12}$$
$$\omega_4 = e^{45} = b^2 \Sigma^{12}$$

### φ in Terms of ω

The terms involving e⁷ can be written:

$$\phi_7 = e^7 \wedge (\omega_3 / a^2 \cdot a^2 + ...) = e^7 \wedge \Omega$$

where Ω is a specific 2-form on S³ × S³.

---

## 6. Summary

### The G₂ 3-Form

$$\boxed{\phi = e^{127} + e^{347} + e^{567} + e^{135} - e^{146} - e^{236} - e^{245}}$$

### In Original Coordinates

$$\phi = a^2 c \, \sigma^{12} \wedge (d\theta + A) + abc \, \sigma^3 \wedge \Sigma^1 \wedge (d\theta + A) + b^2 c \, \Sigma^{23} \wedge (d\theta + A)$$
$$+ a^2 b \, \sigma^{13} \wedge \Sigma^2 - ab^2 \, \sigma^1 \wedge \Sigma^{13} - a^2 b \, \sigma^{23} \wedge \Sigma^3 - ab^2 \, \sigma^2 \wedge \Sigma^{12}$$

### The Coassociative 4-Form

$$\boxed{*\phi = e^{3456} + e^{1256} + e^{1234} + e^{2467} - e^{2357} - e^{1457} - e^{1367}}$$

---

## 7. Next Step

With φ and *φ explicit, proceed to:

**Step 3**: Compute dφ and d*φ (torsion calculation).

---

*GIFT Framework — G₂ 3-Form*
*Step 2 Complete*
