# K₇ Torsion Calculation

**Date**: 2026-01-26
**Status**: Step 3 of Rigorous Construction

---

## 1. Setup

### Structure Equations (from Step 1)

$$de^1 = -\frac{1}{a} e^{23}, \quad de^2 = -\frac{1}{a} e^{31}, \quad de^3 = -\frac{1}{a} e^{12}$$

$$de^4 = -\frac{1}{b} e^{56}, \quad de^5 = -\frac{1}{b} e^{64}, \quad de^6 = -\frac{1}{b} e^{45}$$

$$de^7 = -\frac{c\alpha}{a^2} e^{12} - \frac{c\alpha}{b^2} e^{45}$$

### G₂ 3-Form (from Step 2)

$$\phi = e^{127} + e^{347} + e^{567} + e^{135} - e^{146} - e^{236} - e^{245}$$

---

## 2. Computing dφ

Using the Leibniz rule: $d(e^{ijk}) = de^i \wedge e^{jk} - e^i \wedge de^j \wedge e^k + e^{ij} \wedge de^k$

### Term 1: d(e¹²⁷)

$$d(e^{127}) = de^1 \wedge e^{27} - e^1 \wedge de^2 \wedge e^7 + e^{12} \wedge de^7$$

- $de^1 \wedge e^{27} = -\frac{1}{a} e^{23} \wedge e^{27} = 0$ (repeated index)
- $e^1 \wedge de^2 \wedge e^7 = e^1 \wedge (-\frac{1}{a} e^{31}) \wedge e^7 = 0$ (repeated index)
- $e^{12} \wedge de^7 = e^{12} \wedge (-\frac{c\alpha}{a^2} e^{12} - \frac{c\alpha}{b^2} e^{45}) = -\frac{c\alpha}{b^2} e^{1245}$

$$\boxed{d(e^{127}) = -\frac{c\alpha}{b^2} e^{1245}}$$

### Term 2: d(e³⁴⁷)

$$d(e^{347}) = de^3 \wedge e^{47} - e^3 \wedge de^4 \wedge e^7 + e^{34} \wedge de^7$$

- $de^3 \wedge e^{47} = -\frac{1}{a} e^{12} \wedge e^{47} = -\frac{1}{a} e^{1247}$
- $e^3 \wedge de^4 \wedge e^7 = e^3 \wedge (-\frac{1}{b} e^{56}) \wedge e^7 = -\frac{1}{b} e^{3567}$
- $e^{34} \wedge de^7 = e^{34} \wedge (-\frac{c\alpha}{a^2} e^{12} - \frac{c\alpha}{b^2} e^{45}) = -\frac{c\alpha}{a^2} e^{3412} = \frac{c\alpha}{a^2} e^{1234}$

$$\boxed{d(e^{347}) = -\frac{1}{a} e^{1247} + \frac{1}{b} e^{3567} + \frac{c\alpha}{a^2} e^{1234}}$$

### Term 3: d(e⁵⁶⁷)

$$d(e^{567}) = de^5 \wedge e^{67} - e^5 \wedge de^6 \wedge e^7 + e^{56} \wedge de^7$$

- $de^5 \wedge e^{67} = -\frac{1}{b} e^{64} \wedge e^{67} = 0$
- $e^5 \wedge de^6 \wedge e^7 = e^5 \wedge (-\frac{1}{b} e^{45}) \wedge e^7 = 0$
- $e^{56} \wedge de^7 = e^{56} \wedge (-\frac{c\alpha}{a^2} e^{12}) = -\frac{c\alpha}{a^2} e^{5612} = -\frac{c\alpha}{a^2} e^{1256}$

$$\boxed{d(e^{567}) = -\frac{c\alpha}{a^2} e^{1256}}$$

### Term 4: d(e¹³⁵)

$$d(e^{135}) = de^1 \wedge e^{35} - e^1 \wedge de^3 \wedge e^5 + e^{13} \wedge de^5$$

- $de^1 \wedge e^{35} = -\frac{1}{a} e^{23} \wedge e^{35} = 0$
- $e^1 \wedge de^3 \wedge e^5 = e^1 \wedge (-\frac{1}{a} e^{12}) \wedge e^5 = 0$
- $e^{13} \wedge de^5 = e^{13} \wedge (-\frac{1}{b} e^{64}) = -\frac{1}{b} e^{1364} = \frac{1}{b} e^{1346}$

$$\boxed{d(e^{135}) = \frac{1}{b} e^{1346}}$$

### Term 5: d(-e¹⁴⁶)

$$d(-e^{146}) = -de^1 \wedge e^{46} + e^1 \wedge de^4 \wedge e^6 - e^{14} \wedge de^6$$

- $-de^1 \wedge e^{46} = \frac{1}{a} e^{23} \wedge e^{46} = \frac{1}{a} e^{2346}$
- $e^1 \wedge de^4 \wedge e^6 = e^1 \wedge (-\frac{1}{b} e^{56}) \wedge e^6 = 0$
- $-e^{14} \wedge de^6 = -e^{14} \wedge (-\frac{1}{b} e^{45}) = 0$

$$\boxed{d(-e^{146}) = \frac{1}{a} e^{2346}}$$

### Term 6: d(-e²³⁶)

$$d(-e^{236}) = -de^2 \wedge e^{36} + e^2 \wedge de^3 \wedge e^6 - e^{23} \wedge de^6$$

- $-de^2 \wedge e^{36} = \frac{1}{a} e^{31} \wedge e^{36} = 0$
- $e^2 \wedge de^3 \wedge e^6 = e^2 \wedge (-\frac{1}{a} e^{12}) \wedge e^6 = 0$
- $-e^{23} \wedge de^6 = -e^{23} \wedge (-\frac{1}{b} e^{45}) = \frac{1}{b} e^{2345}$

$$\boxed{d(-e^{236}) = \frac{1}{b} e^{2345}}$$

### Term 7: d(-e²⁴⁵)

$$d(-e^{245}) = -de^2 \wedge e^{45} + e^2 \wedge de^4 \wedge e^5 - e^{24} \wedge de^5$$

- $-de^2 \wedge e^{45} = \frac{1}{a} e^{31} \wedge e^{45} = \frac{1}{a} e^{1345}$
- $e^2 \wedge de^4 \wedge e^5 = e^2 \wedge (-\frac{1}{b} e^{56}) \wedge e^5 = 0$
- $-e^{24} \wedge de^5 = -e^{24} \wedge (-\frac{1}{b} e^{64}) = 0$

$$\boxed{d(-e^{245}) = \frac{1}{a} e^{1345}}$$

---

## 3. Summing All Terms

$$d\phi = -\frac{c\alpha}{b^2} e^{1245} - \frac{1}{a} e^{1247} + \frac{1}{b} e^{3567} + \frac{c\alpha}{a^2} e^{1234}$$
$$- \frac{c\alpha}{a^2} e^{1256} + \frac{1}{b} e^{1346} + \frac{1}{a} e^{2346} + \frac{1}{b} e^{2345} + \frac{1}{a} e^{1345}$$

### Organizing by 4-Form

| 4-Form | Coefficient |
|--------|-------------|
| $e^{1234}$ | $+\frac{c\alpha}{a^2}$ |
| $e^{1245}$ | $-\frac{c\alpha}{b^2}$ |
| $e^{1247}$ | $-\frac{1}{a}$ |
| $e^{1256}$ | $-\frac{c\alpha}{a^2}$ |
| $e^{1345}$ | $+\frac{1}{a}$ |
| $e^{1346}$ | $+\frac{1}{b}$ |
| $e^{2345}$ | $+\frac{1}{b}$ |
| $e^{2346}$ | $+\frac{1}{a}$ |
| $e^{3567}$ | $+\frac{1}{b}$ |

---

## 4. Torsion Analysis

### For dφ = 0

Each coefficient must vanish:

1. $\frac{c\alpha}{a^2} = 0$ → **α = 0** (or c = 0, impossible)
2. $\frac{c\alpha}{b^2} = 0$ → **α = 0** (same)
3. $\frac{1}{a} = 0$ → **Impossible**
4. $\frac{1}{b} = 0$ → **Impossible**

### Conclusion

**With the diagonal connection ansatz A = α(σ³ + Σ³):**

$$\boxed{d\phi \neq 0 \text{ for any choice of } (a, b, c, \alpha)}$$

The non-vanishing terms are:
- $-\frac{1}{a} e^{1247}$ (from S³ structure)
- $+\frac{1}{b} e^{3567}$ (from S³ structure)
- $+\frac{1}{a} e^{1345}$, $+\frac{1}{a} e^{2346}$
- $+\frac{1}{b} e^{1346}$, $+\frac{1}{b} e^{2345}$

These come from the **intrinsic curvature of the S³ factors**, not the connection.

---

## 5. Torsion Classes

The G₂ torsion decomposes as:

$$d\phi = \tau_0 \, *\phi + 3\tau_1 \wedge \phi + *\tau_3$$

$$d*\phi = 4\tau_1 \wedge *\phi + \tau_2 \wedge \phi$$

For our ansatz:
- $\tau_0 \neq 0$ (scalar torsion)
- $\tau_1, \tau_2, \tau_3$ may be non-zero

This is a **G₂ structure with torsion**, not torsion-free.

---

## 6. What This Means

### S³ × S³ × S¹ Does NOT Admit Torsion-Free G₂

The round S³'s have intrinsic curvature that prevents dφ = 0.

This is a well-known result:
- S³ × S³ × S¹ has a **nearly parallel G₂ structure** (weak G₂ holonomy)
- NOT a torsion-free G₂ structure

### Possible Resolutions

**Option A: Squashed spheres**

Replace round S³ with squashed S³ (Berger spheres):
$$ds^2_{S^3} = a^2 (\sigma_1^2 + \sigma_2^2) + a'^2 \sigma_3^2$$

This introduces more parameters that might allow dφ = 0.

**Option B: TCS Gluing**

Use S³ × S³ × S¹ as building blocks for a TCS construction:
1. Take two copies with cylindrical ends
2. Glue along K3 × S¹
3. Apply correction theorem

**Option C: Joyce Orbifold**

Consider (S³ × S³ × S¹)/Γ with specific discrete group Γ that resolves the torsion.

**Option D: Different Base**

Use a different 6-manifold (not S³ × S³) as the base for the S¹ fibration.

---

## 7. The Nearly Parallel G₂ Case

### Definition

A G₂ structure is **nearly parallel** if:
$$d\phi = \lambda \, *\phi$$

for some constant λ.

### For Our Ansatz

Let's check if dφ is proportional to *φ.

Recall *φ = e³⁴⁵⁶ + e¹²⁵⁶ + e¹²³⁴ + e²⁴⁶⁷ - e²³⁵⁷ - e¹⁴⁵⁷ - e¹³⁶⁷

Comparing with dφ:
- dφ has e¹²⁴⁷, e³⁵⁶⁷ which don't appear in *φ
- Therefore **not nearly parallel** either

---

## 8. Summary

$$\boxed{\text{The ansatz } S^3 \times S^3 \times S^1 \text{ does NOT give torsion-free } G_2}$$

### Residual Torsion

| 4-Form | Coefficient | Origin |
|--------|-------------|--------|
| $e^{1247}$ | $-1/a$ | S³ curvature |
| $e^{3567}$ | $+1/b$ | S³ curvature |
| $e^{1345}$, $e^{2346}$ | $+1/a$ | S³ × S³ cross |
| $e^{1346}$, $e^{2345}$ | $+1/b$ | S³ × S³ cross |

### Next Steps

→ **Step 4**: Either:
- (A) Try more general ansatz (squashed spheres, different connection)
- (B) Use TCS gluing theorem with controlled correction

---

*GIFT Framework — Torsion Calculation*
*Step 3 Complete: dφ ≠ 0*
