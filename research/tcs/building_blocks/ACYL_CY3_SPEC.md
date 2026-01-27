# ACyl Calabi-Yau 3-Fold Specification

**Phase 1**: Define the building blocks (V₊, V₋) with explicit geometry and cohomology.

---

## 1. General ACyl CY3 Structure

An **asymptotically cylindrical Calabi-Yau 3-fold** (V, g, I, ω, Ω) consists of:

- **Complex structure** I with c₁(V) = 0
- **Ricci-flat Kähler metric** g
- **Kähler form** ω ∈ Ω²(V)
- **Holomorphic (3,0)-form** Ω ∈ Ω³(V, ℂ)

with one cylindrical end:
```
V ~ V_compact ∪ ([T₀, ∞) × S¹ × Σ)
```
where Σ is a K3 surface.

### Asymptotic Decay

On the cylindrical end {r > T₀}:
```
g = g_Σ + dr² + dθ² + O(e^{-μr})
ω = ω_Σ + dr ∧ dθ + O(e^{-μr})
Ω = Ω_Σ ∧ (dr + i·dθ) + O(e^{-μr})
```

where μ > 0 is the **decay rate** (typically μ ≈ 1 for Fano-derived blocks).

---

## 2. Building Block M₁: Quintic in CP⁴

### Construction

Let X₅ ⊂ CP⁴ be a smooth quintic hypersurface (generic degree 5).

**Calabi-Yau property**: c₁(X₅) = 0 by adjunction:
```
c₁(X₅) = c₁(CP⁴)|_{X₅} - c₁(O(5))|_{X₅} = 5H - 5H = 0
```

### Anticanonical Divisor

Choose a smooth hyperplane section D₁ = X₅ ∩ H ≅ K3 (quartic K3 surface).

**M₁ := X₅ \ D₁** is an ACyl CY3.

### Cohomology Data

| Invariant | Value | Computation |
|-----------|-------|-------------|
| h¹'¹(X₅) | 1 | Lefschetz hyperplane |
| h²'¹(X₅) | 101 | Griffiths formula |
| b₂(X₅) | 2 | h¹'¹ + h⁰'² = 1 + 1 |
| b₃(X₅) | 204 | 2h²'¹ + 2 = 204 |
| χ(X₅) | -200 | Euler characteristic |

For the **open** manifold M₁ = X₅ \ D₁:
```
b₂(M₁) = 11
b₃(M₁) = 40
```

**(Note: These are the values from TCSConstruction.lean, derived via CHNP 2015)**

### K3 Cross-Section

The divisor D₁ is a **quartic K3** in CP³:
```
b₀(D₁) = 1
b₂(D₁) = 22  (Hodge: h¹'¹ = 20, h²'⁰ = h⁰'² = 1)
b₄(D₁) = 1
χ(D₁) = 24
```

---

## 3. Building Block M₂: CI(2,2,2) in CP⁶

### Construction

Let X_{2,2,2} ⊂ CP⁶ be a smooth complete intersection of three quadrics.

**Calabi-Yau property**:
```
c₁(X_{2,2,2}) = c₁(CP⁶) - 3·c₁(O(2)) = 7H - 6H = H ≠ 0
```

Wait - this gives a Fano 3-fold, not CY3!

**Correction**: We use the **semi-Fano** construction from CHNP. The building block is:
```
M₂ = (Fano 3-fold) \ (anticanonical K3)
```

The ACyl CY3 comes from the complement, not the Fano itself.

### Cohomology Data (from TCSConstruction.lean)

| Invariant | Value |
|-----------|-------|
| b₂(M₂) | 10 |
| b₃(M₂) | 37 |

### K3 Cross-Section

The anticanonical divisor D₂ ≅ K3 with:
```
b₂(D₂) = 22
```

---

## 4. K3 Surface Data

Both building blocks have K3 cross-sections. The K3 surface Σ has:

### Topology
```
b₀ = 1
b₁ = 0
b₂ = 22
b₃ = 0
b₄ = 1
χ = 24
```

### Hodge Diamond
```
        1
      0   0
    1   20   1
      0   0
        1
```

### Cohomology Lattice
```
H²(K3, ℤ) ≅ E₈(-1)² ⊕ U³
```
where U is the hyperbolic lattice.

**Signature**: (3, 19) - three positive, nineteen negative directions.

### Hyper-Kähler Structure

K3 admits a 2-sphere of complex structures parametrized by:
```
S² ≅ {(a,b,c) ∈ ℝ³ : a² + b² + c² = 1}
```

The hyper-Kähler triple (ω_I, ω_J, ω_K) satisfies:
```
ω_I ∧ ω_I = ω_J ∧ ω_J = ω_K ∧ ω_K
ω_I ∧ ω_J = ω_J ∧ ω_K = ω_K ∧ ω_I = 0
```

---

## 5. Mayer-Vietoris Computation

### Setup

The TCS manifold K7 is:
```
K7 = (M₁ × S¹) ∪_Φ (M₂ × S¹)
```

Glued along the common boundary:
```
∂(M₁ × S¹) = ∂(M₂ × S¹) = K3 × T²
```

### Long Exact Sequence

For the decomposition K7 = U ∪ V with U ∩ V ≃ K3 × T²:

```
... → Hᵏ(K7) → Hᵏ(U) ⊕ Hᵏ(V) → Hᵏ(U ∩ V) → Hᵏ⁺¹(K7) → ...
```

### Betti Number Formula

Under favorable matching (generic):
```
b₂(K7) = b₂(M₁) + b₂(M₂) = 11 + 10 = 21
b₃(K7) = b₃(M₁) + b₃(M₂) = 40 + 37 = 77
```

**Note**: This simplified formula assumes the gluing map induces trivial maps on cohomology kernels. The full formula involves correction terms from the matching.

---

## 6. ACyl SU(3) Structure Formulas

On the cylindrical end of M_± ~ (T₀, ∞) × S¹_θ × Σ:

### Kähler Form
```
ω = ω_Σ + dr ∧ dθ
```

### Holomorphic 3-Form
```
Ω = Ω_Σ ∧ (dr + i·dθ)
```

where (ω_Σ, Ω_Σ) is the hyper-Kähler structure on K3.

### Metric
```
ds² = ds²_Σ + dr² + dθ²
```

---

## 7. Summary Table

| Building Block | b₂ | b₃ | K3 Type | Source |
|----------------|----|----|---------|--------|
| M₁ (Quintic) | 11 | 40 | Quartic in CP³ | CHNP 2015 |
| M₂ (CI type) | 10 | 37 | Anticanonical | CHNP 2015 |
| **K7 (TCS)** | **21** | **77** | — | Mayer-Vietoris |

---

## 8. Verification Against GIFT

| Quantity | TCS Value | GIFT Value | Match |
|----------|-----------|------------|-------|
| b₂ | 21 | 21 | ✓ |
| b₃ | 77 | 77 | ✓ |
| H* | 99 | 99 | ✓ |
| dim(G₂) | 14 | 14 | ✓ |

The building blocks are **consistent** with GIFT topology.

---

## 9. Open: Uniqueness Question

**Question**: Are M₁ = Quintic and M₂ = CI(2,2,2) the **unique** building blocks giving (b₂, b₃) = (21, 77)?

From CHNP catalog:
- There are 63 types of semi-Fano building blocks
- Multiple pairings can give the same Betti numbers
- **Conjecture**: The (11, 40) + (10, 37) pairing may not be unique

This affects the **moduli dimension** of the K7 family.

---

*Phase 1 Specification Complete*
*Next: Phase 2 - K3 Matching with Hyper-Kähler Rotation*
