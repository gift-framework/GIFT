# GIFT Framework v2.2 - Glossary and Terminology

**Version**: 2.2.0
**Date**: 2025-11-26

---

## Terminology: The Zero-Parameter Paradigm

### Definition

The GIFT framework contains **zero continuous adjustable parameters**. This means:

1. **No fitting**: No quantity is adjusted to match experimental data
2. **No optimization**: No minimization procedure determines framework constants
3. **Discrete choices only**: The only "freedom" consists in selecting mathematical structures (E₈×E₈, K₇ topology, G₂ holonomy)

### What this does NOT mean

- "No assumptions" - The framework assumes specific discrete mathematical structures
- "Unique theory" - Other gauge groups or manifolds could be explored
- "Complete derivation" - Some quantities (THEORETICAL status) require scale input

### Comparison with Standard Model

| Aspect | Standard Model | GIFT Framework |
|--------|----------------|----------------|
| Free continuous parameters | 19+ | **0** |
| Discrete structure choices | SU(3)×SU(2)×U(1), 3 generations | E₈×E₈, K₇, G₂ holonomy |
| Parameter origin | Experimental measurement | Topological derivation |
| Adjustability | Continuous tuning possible | Fixed by mathematical structure |

---

## Status Classifications

Throughout the framework documents, the following classifications indicate epistemic status:

| Status | Meaning | Example |
|--------|---------|---------|
| **PROVEN** | Exact topological identity with rigorous mathematical proof | sin²θ_W = 3/13, δ_CP = 197° |
| **TOPOLOGICAL** | Direct consequence of manifold structure without empirical input | κ_T = 1/61, det(g) = 65/32 |
| **DERIVED** | Calculated from proven/topological relations | m_c/m_s, CKM elements |
| **THEORETICAL** | Has theoretical justification, proof incomplete | v_EW, absolute quark masses |
| **PHENOMENOLOGICAL** | Empirically accurate, theoretical derivation in progress | m_μ/m_e |

---

## Core Mathematical Structures

### E₈ Exceptional Lie Algebra

| Property | Value | Significance |
|----------|-------|--------------|
| dim(E₈) | 248 | Total gauge degrees of freedom |
| rank(E₈) | 8 | Cartan subalgebra dimension |
| \|W(E₈)\| | 696,729,600 = 2¹⁴ × 3⁵ × 5² × 7 | Weyl group order |
| Root count | 240 | All roots have length √2 |

### K₇ Manifold

| Property | Value | Significance |
|----------|-------|--------------|
| dim(K₇) | 7 | Internal manifold dimension |
| b₂(K₇) | 21 | Gauge field multiplicity |
| b₃(K₇) | 77 | Matter field multiplicity |
| H* | 99 | Effective cohomological dimension |
| Holonomy | G₂ | Preserves N=1 supersymmetry |

### G₂ Holonomy Group

| Property | Value | Significance |
|----------|-------|--------------|
| dim(G₂) | 14 | Holonomy group dimension |
| rank(G₂) | 2 | Cartan subalgebra |
| Definition | Aut(O) | Automorphisms of octonions |

---

## Framework Constants

### Structural Constants (not free parameters)

| Symbol | Value | Origin | Status |
|--------|-------|--------|--------|
| p₂ | 2 | dim(G₂)/dim(K₇) = 14/7 | Fixed by geometry |
| β₀ | π/8 | π/rank(E₈) | Fixed by algebra |
| Weyl_factor | 5 | From \|W(E₈)\| = ...×5²×... | Fixed by group theory |
| det(g) | 65/32 | p₂ + 1/(b₂ + dim(G₂) - N_gen) | Fixed by topology |

### Derived Constants

| Symbol | Value | Formula | Status |
|--------|-------|---------|--------|
| ξ | 5π/16 | (Weyl/p₂) × β₀ | Derived exactly |
| τ | 3472/891 | 496×21/(27×99) | Proven rational |
| κ_T | 1/61 | 1/(b₃ - dim(G₂) - p₂) | Topological |

---

## Notation Conventions

### Subscripts and Superscripts

- **Topological**: b₂, b₃ (Betti numbers)
- **Algebraic**: dim(), rank(), |W()| (dimension, rank, Weyl group order)
- **Physical**: M_Z, m_e, α_s (masses, couplings)

### Special Numbers

| Number | Role in Framework |
|--------|-------------------|
| 61 | Torsion denominator: κ_T = 1/61 |
| 91 | Weinberg angle denominator: sin²θ_W = 21/91 = 3/13 |
| 221 | Structural constant: 248 - 27 = 13 × 17 |
| 3477 | Tau-electron ratio: m_τ/m_e = 3477 |

---

## Canonical Formulations

When discussing the zero-parameter nature, use these standard phrasings:

**Full form**:
> "The framework contains zero continuous adjustable parameters. Predictions derive uniquely from discrete structural choices: E₈×E₈ gauge group and K₇ manifold with G₂ holonomy."

**Short form** (for abstracts):
> "Zero continuous parameters; all predictions from fixed topological structure."

**Technical form**:
> "All quantities appearing in observable predictions derive from fixed mathematical structures of E₈×E₈ and K₇, with no adjustable continuous parameters."

---

## Document Version

- **Version**: 2.2.0
- **Last Updated**: 2025-11-26
- **Repository**: https://github.com/gift-framework/GIFT
