# K₇ Complete Torsion Analysis

**Date**: 2026-01-26
**Status**: COMPLETE — Both dφ AND d*φ computed

---

## 1. Structure Equations (Recap)

### Coframe

$$e^1 = a\sigma^1, \quad e^2 = a\sigma^2, \quad e^3 = a\sigma^3$$
$$e^4 = b\Sigma^1, \quad e^5 = b\Sigma^2, \quad e^6 = b\Sigma^3$$
$$e^7 = c(d\theta + A)$$

### Connection Ansatz

$$A = \alpha(\sigma^3 + \Sigma^3) = \frac{\alpha}{a}e^3 + \frac{\alpha}{b}e^6$$

### Structure Equations

$$de^1 = -\frac{1}{a}e^{23}, \quad de^2 = -\frac{1}{a}e^{31}, \quad de^3 = -\frac{1}{a}e^{12}$$

$$de^4 = -\frac{1}{b}e^{56}, \quad de^5 = -\frac{1}{b}e^{64}, \quad de^6 = -\frac{1}{b}e^{45}$$

$$de^7 = -\frac{c\alpha}{a^2}e^{12} - \frac{c\alpha}{b^2}e^{45}$$

---

## 2. The G₂ 3-Form

$$\phi = e^{127} + e^{347} + e^{567} + e^{135} - e^{146} - e^{236} - e^{245}$$

---

## 3. Computing dφ (Done Previously)

$$d\phi = -\frac{c\alpha}{b^2}e^{1245} - \frac{1}{a}e^{1247} + \frac{1}{b}e^{3567} + \frac{c\alpha}{a^2}e^{1234}$$
$$- \frac{c\alpha}{a^2}e^{1256} + \frac{1}{b}e^{1346} + \frac{1}{a}e^{2346} + \frac{1}{b}e^{2345} + \frac{1}{a}e^{1345}$$

---

## 4. The Coassociative 4-Form *φ

$$*\phi = e^{3456} + e^{1256} + e^{1234} + e^{2467} - e^{2357} - e^{1457} - e^{1367}$$

---

## 5. Computing d*φ

Using $d(e^{ijkl}) = de^i \wedge e^{jkl} - e^i \wedge de^j \wedge e^{kl} + e^{ij} \wedge de^k \wedge e^l - e^{ijk} \wedge de^l$

### Term 1: d(e³⁴⁵⁶)

$$d(e^{3456}) = de^3 \wedge e^{456} - e^3 \wedge de^4 \wedge e^{56} + e^{34} \wedge de^5 \wedge e^6 - e^{345} \wedge de^6$$

- $de^3 \wedge e^{456} = -\frac{1}{a}e^{12} \wedge e^{456} = -\frac{1}{a}e^{12456}$
- $e^3 \wedge de^4 \wedge e^{56} = e^3 \wedge (-\frac{1}{b}e^{56}) \wedge e^{56} = 0$
- $e^{34} \wedge de^5 \wedge e^6 = e^{34} \wedge (-\frac{1}{b}e^{64}) \wedge e^6 = 0$
- $e^{345} \wedge de^6 = e^{345} \wedge (-\frac{1}{b}e^{45}) = 0$

$$\boxed{d(e^{3456}) = -\frac{1}{a}e^{12456}}$$

### Term 2: d(e¹²⁵⁶)

$$d(e^{1256}) = de^1 \wedge e^{256} - e^1 \wedge de^2 \wedge e^{56} + e^{12} \wedge de^5 \wedge e^6 - e^{125} \wedge de^6$$

- $de^1 \wedge e^{256} = -\frac{1}{a}e^{23} \wedge e^{256} = 0$
- $e^1 \wedge de^2 \wedge e^{56} = e^1 \wedge (-\frac{1}{a}e^{31}) \wedge e^{56} = 0$
- $e^{12} \wedge de^5 \wedge e^6 = e^{12} \wedge (-\frac{1}{b}e^{64}) \wedge e^6 = 0$
- $e^{125} \wedge de^6 = e^{125} \wedge (-\frac{1}{b}e^{45}) = -\frac{1}{b}e^{12545} = 0$

$$\boxed{d(e^{1256}) = 0}$$

### Term 3: d(e¹²³⁴)

$$d(e^{1234}) = de^1 \wedge e^{234} - e^1 \wedge de^2 \wedge e^{34} + e^{12} \wedge de^3 \wedge e^4 - e^{123} \wedge de^4$$

- $de^1 \wedge e^{234} = -\frac{1}{a}e^{23} \wedge e^{234} = 0$
- $e^1 \wedge de^2 \wedge e^{34} = e^1 \wedge (-\frac{1}{a}e^{31}) \wedge e^{34} = 0$
- $e^{12} \wedge de^3 \wedge e^4 = e^{12} \wedge (-\frac{1}{a}e^{12}) \wedge e^4 = 0$
- $e^{123} \wedge de^4 = e^{123} \wedge (-\frac{1}{b}e^{56}) = -\frac{1}{b}e^{12356}$

$$\boxed{d(e^{1234}) = -\frac{1}{b}e^{12356}}$$

### Term 4: d(e²⁴⁶⁷)

$$d(e^{2467}) = de^2 \wedge e^{467} - e^2 \wedge de^4 \wedge e^{67} + e^{24} \wedge de^6 \wedge e^7 - e^{246} \wedge de^7$$

- $de^2 \wedge e^{467} = -\frac{1}{a}e^{31} \wedge e^{467} = -\frac{1}{a}e^{13467}$
- $e^2 \wedge de^4 \wedge e^{67} = e^2 \wedge (-\frac{1}{b}e^{56}) \wedge e^{67} = 0$
- $e^{24} \wedge de^6 \wedge e^7 = e^{24} \wedge (-\frac{1}{b}e^{45}) \wedge e^7 = 0$
- $e^{246} \wedge de^7 = e^{246} \wedge (-\frac{c\alpha}{a^2}e^{12} - \frac{c\alpha}{b^2}e^{45}) = -\frac{c\alpha}{a^2}e^{24612} - \frac{c\alpha}{b^2}e^{24645}$
  $= \frac{c\alpha}{a^2}e^{12246} + 0 = 0$

$$\boxed{d(e^{2467}) = -\frac{1}{a}e^{13467} = \frac{1}{a}e^{13467}}$$

Wait, let me recalculate the sign: $e^{31467} = -e^{13467}$, so $-\frac{1}{a}e^{31467} = \frac{1}{a}e^{13467}$

$$\boxed{d(e^{2467}) = \frac{1}{a}e^{13467}}$$

### Term 5: d(-e²³⁵⁷)

$$d(-e^{2357}) = -de^2 \wedge e^{357} + e^2 \wedge de^3 \wedge e^{57} - e^{23} \wedge de^5 \wedge e^7 + e^{235} \wedge de^7$$

- $-de^2 \wedge e^{357} = \frac{1}{a}e^{31} \wedge e^{357} = 0$
- $e^2 \wedge de^3 \wedge e^{57} = e^2 \wedge (-\frac{1}{a}e^{12}) \wedge e^{57} = 0$
- $-e^{23} \wedge de^5 \wedge e^7 = -e^{23} \wedge (-\frac{1}{b}e^{64}) \wedge e^7 = \frac{1}{b}e^{23647} = -\frac{1}{b}e^{23467}$
- $e^{235} \wedge de^7 = e^{235} \wedge (-\frac{c\alpha}{a^2}e^{12} - \frac{c\alpha}{b^2}e^{45})$
  $= -\frac{c\alpha}{a^2}e^{23512} - \frac{c\alpha}{b^2}e^{23545} = \frac{c\alpha}{a^2}e^{12235} + 0 = 0$

$$\boxed{d(-e^{2357}) = -\frac{1}{b}e^{23467}}$$

### Term 6: d(-e¹⁴⁵⁷)

$$d(-e^{1457}) = -de^1 \wedge e^{457} + e^1 \wedge de^4 \wedge e^{57} - e^{14} \wedge de^5 \wedge e^7 + e^{145} \wedge de^7$$

- $-de^1 \wedge e^{457} = \frac{1}{a}e^{23} \wedge e^{457} = \frac{1}{a}e^{23457}$
- $e^1 \wedge de^4 \wedge e^{57} = e^1 \wedge (-\frac{1}{b}e^{56}) \wedge e^{57} = 0$
- $-e^{14} \wedge de^5 \wedge e^7 = -e^{14} \wedge (-\frac{1}{b}e^{64}) \wedge e^7 = \frac{1}{b}e^{14647} = 0$
- $e^{145} \wedge de^7 = e^{145} \wedge (-\frac{c\alpha}{a^2}e^{12} - \frac{c\alpha}{b^2}e^{45})$
  $= -\frac{c\alpha}{a^2}e^{14512} - 0 = \frac{c\alpha}{a^2}e^{12145} = 0$

$$\boxed{d(-e^{1457}) = \frac{1}{a}e^{23457}}$$

### Term 7: d(-e¹³⁶⁷)

$$d(-e^{1367}) = -de^1 \wedge e^{367} + e^1 \wedge de^3 \wedge e^{67} - e^{13} \wedge de^6 \wedge e^7 + e^{136} \wedge de^7$$

- $-de^1 \wedge e^{367} = \frac{1}{a}e^{23} \wedge e^{367} = 0$
- $e^1 \wedge de^3 \wedge e^{67} = e^1 \wedge (-\frac{1}{a}e^{12}) \wedge e^{67} = 0$
- $-e^{13} \wedge de^6 \wedge e^7 = -e^{13} \wedge (-\frac{1}{b}e^{45}) \wedge e^7 = \frac{1}{b}e^{13457}$
- $e^{136} \wedge de^7 = e^{136} \wedge (-\frac{c\alpha}{a^2}e^{12} - \frac{c\alpha}{b^2}e^{45})$
  $= -\frac{c\alpha}{a^2}e^{13612} - \frac{c\alpha}{b^2}e^{13645} = 0 + \frac{c\alpha}{b^2}e^{13456}$

$$\boxed{d(-e^{1367}) = \frac{1}{b}e^{13457} + \frac{c\alpha}{b^2}e^{13456}}$$

---

## 6. Summing d*φ

$$d*\phi = -\frac{1}{a}e^{12456} + 0 - \frac{1}{b}e^{12356} + \frac{1}{a}e^{13467}$$
$$- \frac{1}{b}e^{23467} + \frac{1}{a}e^{23457} + \frac{1}{b}e^{13457} + \frac{c\alpha}{b^2}e^{13456}$$

### Organized by 5-Form

| 5-Form | Coefficient |
|--------|-------------|
| $e^{12356}$ | $-\frac{1}{b}$ |
| $e^{12456}$ | $-\frac{1}{a}$ |
| $e^{13456}$ | $+\frac{c\alpha}{b^2}$ |
| $e^{13457}$ | $+\frac{1}{b}$ |
| $e^{13467}$ | $+\frac{1}{a}$ |
| $e^{23457}$ | $+\frac{1}{a}$ |
| $e^{23467}$ | $-\frac{1}{b}$ |

---

## 7. Constraints for Torsion-Free G₂

### From dφ = 0

| 4-Form | Coefficient | Constraint |
|--------|-------------|------------|
| $e^{1234}$ | $\frac{c\alpha}{a^2}$ | $\alpha = 0$ |
| $e^{1245}$ | $-\frac{c\alpha}{b^2}$ | $\alpha = 0$ |
| $e^{1247}$ | $-\frac{1}{a}$ | **Impossible** |
| $e^{1256}$ | $-\frac{c\alpha}{a^2}$ | $\alpha = 0$ |
| $e^{1345}$ | $+\frac{1}{a}$ | **Impossible** |
| $e^{1346}$ | $+\frac{1}{b}$ | **Impossible** |
| $e^{2345}$ | $+\frac{1}{b}$ | **Impossible** |
| $e^{2346}$ | $+\frac{1}{a}$ | **Impossible** |
| $e^{3567}$ | $+\frac{1}{b}$ | **Impossible** |

### From d*φ = 0

| 5-Form | Coefficient | Constraint |
|--------|-------------|------------|
| $e^{12356}$ | $-\frac{1}{b}$ | **Impossible** |
| $e^{12456}$ | $-\frac{1}{a}$ | **Impossible** |
| $e^{13456}$ | $\frac{c\alpha}{b^2}$ | $\alpha = 0$ |
| $e^{13457}$ | $+\frac{1}{b}$ | **Impossible** |
| $e^{13467}$ | $+\frac{1}{a}$ | **Impossible** |
| $e^{23457}$ | $+\frac{1}{a}$ | **Impossible** |
| $e^{23467}$ | $-\frac{1}{b}$ | **Impossible** |

---

## 8. EXPLICIT CONSTRAINT LIST

### Result

$$\boxed{\text{NO SOLUTION EXISTS for } (a, b, c, \alpha) \text{ with } a, b, c > 0}$$

### The Obstruction

Both dφ and d*φ contain terms proportional to:
- $1/a$ (from first S³ curvature)
- $1/b$ (from second S³ curvature)

These **cannot be eliminated** for any finite a, b.

### Physical Meaning

The **intrinsic curvature of round S³** is incompatible with torsion-free G₂.

No choice of:
- Radii (a, b, c)
- Connection A
- Twist α

can make S³ × S³ × S¹ into a torsion-free G₂ manifold.

---

## 9. Alternative Ansätze

### Option A: Squashed S³ (Berger spheres)

Replace round S³ with:
$$ds^2_{S^3} = a_1^2(\sigma^1)^2 + a_2^2(\sigma^2)^2 + a_3^2(\sigma^3)^2$$

This adds parameters but the curvature obstruction persists.

### Option B: Nearly-Kähler S³ × S³

S³ × S³ admits a **nearly-Kähler structure** with:
$$d\omega = 3\text{Re}(\Omega), \quad d\text{Im}(\Omega) = -2\omega^2$$

This gives **nearly-parallel G₂** (τ₀ ≠ 0), not torsion-free.

### Option C: TCS Gluing (REQUIRED)

The only way to get torsion-free G₂ is:
1. Use K3 (not S³) as cross-section
2. Glue ACyl building blocks
3. Apply Kovalev correction theorem

---

## 10. Summary

### Complete Constraint Analysis

| Source | Constraints | Solvable? |
|--------|-------------|-----------|
| dφ = 0 | α = 0 AND 1/a = 0 AND 1/b = 0 | ❌ NO |
| d*φ = 0 | α = 0 AND 1/a = 0 AND 1/b = 0 | ❌ NO |

### The Verdict

$$\boxed{S^3 \times S^3 \times S^1 \text{ does NOT admit torsion-free } G_2 \text{ for any } (a,b,c,\alpha)}$$

**Proof complete.**

### Next Step

→ Use **TCS construction** with K3 cross-section (as in K7_TCS_GLUING.md)

---

*GIFT Framework — Complete Torsion Analysis*
*Both dφ AND d*φ computed explicitly*
