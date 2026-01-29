# GIFT K₇: TCS vs Variational Characterization

**Date**: Janvier 2026
**Status**: CRITICAL CLARIFICATION
**Impact**: Reframes the entire Tier-2 question

---

## 1. The Discovery

Standard TCS (Twisted Connected Sum) constructions satisfy:
```
0 ≤ b₂(M) ≤ 9
```
(Source: Corti-Haskins-Nordström-Pacini 2015)

But GIFT K₇ has:
```
b₂(K₇) = 21
```

**This is incompatible with standard TCS!**

---

## 2. What This Means

### Option A: Non-Standard Building Blocks

K₇ could be a TCS with unusual ACyl Calabi-Yau 3-folds that produce larger b₂. This would require:
- Building blocks with large H²
- Non-typical polarizing lattices
- Possibly Joyce orbifold resolution data

**Status**: No known explicit construction achieving b₂ = 21 via TCS.

### Option B: Variational Characterization

K₇ is defined by its **topological properties** (b₂ = 21, b₃ = 77, H* = 99) rather than an explicit TCS gluing. This is a:
- **Canonical representative** of a cohomology class
- **Minimizer** of some functional (volume, action, ...)
- **Fixed point** of some flow (Ricci, RG, ...)

**Status**: This is the working assumption in GIFT.

### Option C: Joyce Orbifold Resolution

Joyce's 1996 construction (T⁷/Γ resolution) typically gives:
- b₂ > 0 (unlike TCS which often has b₂ = 0)
- Various values depending on the orbifold group Γ

**Status**: Need to check Joyce's tables for b₂ = 21 examples.

---

## 3. Impact on Spectral Bounds

### Model Theorem (Tier 1): Still Valid

The Model Theorem:
```
c₁/L² ≤ λ₁(K) ≤ c₂/L²
```
is proven for **any** TCS with hypotheses (H1)-(H6). It doesn't require b₂ ≤ 9.

If K₇ is a TCS (even non-standard), the theorem applies.

### L² ~ H* (Tier 2): Needs Reframing

The conjecture L² ~ H* assumed TCS geometry. If K₇ is **not** a TCS:
- The "neck length" L is not defined
- The spectral gap formula needs different derivation
- H* = 99 enters via a different mechanism

**New interpretation**:
```
λ₁ = 14/99 is a TOPOLOGICAL prediction
     independent of whether K₇ has TCS structure
```

The value 14/99 = dim(G₂)/H* could arise from:
1. Representation theory of G₂ on H*(K₇)
2. Index theorem relating spectral and topological data
3. Heat kernel asymptotics

---

## 4. Literature Cross-Check

### TCS Betti Ranges

| Source | b₂ range | b₃ range |
|--------|----------|----------|
| Kovalev 2003 | ≤ 9 | 71-155 |
| Corti et al. 2015 | ≤ 9 | 55-239 |
| Extra-twisted (CGN) | ≤ 9 | varies |
| **GIFT K₇** | **21** | **77** |

### Joyce Orbifold Examples

Joyce's book contains tables of examples. Need to check if any has b₂ = 21.

Known: Joyce examples typically have b₂ > 0, unlike TCS.

---

## 5. Revised Tier Structure

### Tier 1 (THEOREM)
```
For TCS manifolds: λ₁ = Θ(1/L²)
```
**Status**: Proven (Model Theorem + Literature)

### Tier 2 (REFORMULATED)
```
For G₂ manifolds with H* = b₂ + b₃ + 1:
    λ₁ = c/H* for some c > 0
```
**Status**: CONJECTURE (no longer assumes TCS)

### Tier 2' (NEW QUESTION)
```
What determines c = dim(G₂) = 14?
- Holonomy constraint?
- Representation theory?
- Index theorem?
```
**Status**: OPEN

### Tier 3 (UNCHANGED)
```
Why K₇ specifically? Why (b₂, b₃) = (21, 77)?
```
**Status**: PHILOSOPHICAL (model selection)

---

## 6. Next Steps

1. **Check Joyce tables** for b₂ = 21 examples
2. **Index theorem approach**: Does Atiyah-Singer give λ₁ ~ 1/H*?
3. **Heat kernel**: Asymptotic expansion on G₂ manifolds
4. **Abandon TCS framing** for Tier-2, work with general G₂

---

## 7. Key References

- GIFTPY_FOR_GEOMETERS.md (line 126): "K₇ with b₂ = 21 either employs non-standard building blocks or should be understood via the variational characterization"
- Corti-Haskins-Nordström-Pacini 2015: b₂ ≤ 9 for standard TCS
- Joyce 2000: Orbifold resolution examples

---

*Critical clarification for GIFT Spectral Theory*
*Janvier 2026*
