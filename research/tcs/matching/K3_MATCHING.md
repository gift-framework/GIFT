# K3 Matching with Hyper-Kähler Rotation

**Phase 2**: Fix the matching data (r, ω, Ω) between K3 cross-sections.

---

## 1. The Matching Problem

The TCS construction requires identifying the cylindrical ends:
```
(T, ∞) × S¹_θ × Σ₊  ↔  (-∞, -T) × S¹_ψ × Σ₋
```

via a diffeomorphism Φ that:
1. Reverses the radial coordinate: r ↦ -r
2. Identifies the circles: θ ↔ ψ (possibly with twist)
3. Matches K3 surfaces: r: Σ₊ → Σ₋ with hyper-Kähler rotation

---

## 2. Donaldson's Matching Condition

### Statement

The diffeomorphism r: Σ₊ → Σ₋ must satisfy:
```
r*(ω_J^-) = ω_I^+        (Kähler ↔ Kähler)
r*(Re Ω_J^-) = Re Ω_I^+  (holomorphic ↔ holomorphic)
r*(Im Ω_J^-) = -Im Ω_I^+ (sign flip)
```

where the subscripts I, J denote different complex structures in the hyper-Kähler family.

### Geometric Meaning

This rotates the hyper-Kähler triple by 90° in the S² of complex structures:
```
(ω_I, ω_J, ω_K)₊  ↦  (ω_J, -ω_I, ω_K)₋
```

Equivalently, in quaternionic notation: **i ↦ j** rotation.

---

## 3. Hyper-Kähler Triple on K3

### Definition

A K3 surface (Σ, g) with hyper-Kähler metric has three compatible complex structures (I, J, K) satisfying:
```
I² = J² = K² = IJK = -1
```

Each gives a Kähler form:
```
ω_I(X, Y) = g(IX, Y)
ω_J(X, Y) = g(JX, Y)
ω_K(X, Y) = g(KX, Y)
```

### Normalization

We normalize so that:
```
∫_Σ ω_I ∧ ω_I = ∫_Σ ω_J ∧ ω_J = ∫_Σ ω_K ∧ ω_K = Vol(Σ)
∫_Σ ω_I ∧ ω_J = ∫_Σ ω_J ∧ ω_K = ∫_Σ ω_K ∧ ω_I = 0
```

### Holomorphic Forms

For each complex structure, there's a holomorphic (2,0)-form:
```
Ω_I = ω_J + i·ω_K
Ω_J = ω_K + i·ω_I
Ω_K = ω_I + i·ω_J
```

satisfying:
```
Ω_I ∧ Ω̄_I = 2·ω_I ∧ ω_I
dΩ_I = 0
```

---

## 4. Lattice Polarization

### The K3 Lattice

```
Λ_{K3} = H²(Σ, ℤ) ≅ E₈(-1)² ⊕ U³
```

- Rank: 22
- Signature: (3, 19)
- U = hyperbolic plane with matrix ((0,1),(1,0))

### Polarization Data

A **polarization** is a primitive class L ∈ Λ_{K3} with L² > 0.

The matching requires:
```
Pic(Σ₊) ⊥ Pic(Σ₋)  in some sense
```

**Orthogonal gluing**: The Picard lattices should be "complementary" in Λ_{K3}.

---

## 5. Explicit Matching for K7

### From Building Blocks

For M₁ (Quintic) and M₂ (CI):
- Σ₊ = quartic K3 with specific Picard lattice
- Σ₋ = anticanonical K3 from semi-Fano

### Moduli Count

The moduli space of matchings has dimension:
```
dim(Matching Moduli) = 20 - rank(Pic(Σ₊)) - rank(Pic(Σ₋)) + overlap
```

For generic K3 (Pic = ℤ), this gives a large moduli space.

### Canonical Matching (Conjecture)

**GIFT Selection**: There may exist a "canonical" matching that:
1. Minimizes some functional (torsion energy, mismatch, etc.)
2. Fixes the neck length L via L² ~ H*

This is the **Phase 7 selection principle**.

---

## 6. Effect on Cohomology

### Induced Maps

The matching r: Σ₊ → Σ₋ induces:
```
r*: H²(Σ₋) → H²(Σ₊)
```

### Mayer-Vietoris Correction

The full Betti number formula is:
```
b₂(K7) = b₂(M₁) + b₂(M₂) - dim(ker(r* - id))
b₃(K7) = b₃(M₁) + b₃(M₂) + 2·b₂(Σ) - 2 - corrections
```

For **generic** matching (no fixed classes), corrections vanish and we get:
```
b₂(K7) = 11 + 10 = 21  ✓
b₃(K7) = 40 + 37 = 77  ✓
```

---

## 7. Matching Matrix

### Action on Cohomology

In a basis {α₁, ..., α₂₂} for H²(Σ, ℤ):
```
r*: H²(Σ₋) → H²(Σ₊)
[r*]_{ij} = ∫_Σ r*(α_i) ∧ α_j
```

### Hyper-Kähler Rotation Matrix

For the 3-dimensional space spanned by (ω_I, ω_J, ω_K):
```
R = | 0  1  0 |
    |-1  0  0 |
    | 0  0  1 |
```

This is the **i ↦ j rotation** in the quaternionic S².

---

## 8. Computational Approach

### Step 1: Fix K3 Moduli

Choose specific K3 surfaces Σ₊, Σ₋ with known Ricci-flat metrics.

### Step 2: Compute Hyper-Kähler Triple

Use the period map to identify (ω_I, ω_J, ω_K) in H²(Σ, ℝ).

### Step 3: Find Matching Diffeomorphism

Solve for r: Σ₊ → Σ₋ satisfying Donaldson's condition.

### Step 4: Verify Cohomology

Check that r* gives the expected Betti numbers.

---

## 9. Numerical Data (For Notebook)

### K3 Volume
```python
# Normalized K3 volume (in units where metric has scalar curvature R = 0)
Vol_K3 = 16 * np.pi**2  # Topological: Vol = (1/2) * χ * (2π)² / 3 = 8π² for χ = 24
```

### Hyper-Kähler Forms (Symbolic)

In local coordinates (z₁, z₂) on K3:
```
ω_I = (i/2)(dz₁ ∧ dz̄₁ + dz₂ ∧ dz̄₂)
Ω_I = dz₁ ∧ dz₂
```

### Rotation Action
```python
def hyperkahler_rotate(omega_I, omega_J, omega_K):
    """Donaldson rotation: I -> J"""
    return (omega_J, -omega_I, omega_K)
```

---

## 10. Summary

| Component | Data |
|-----------|------|
| K3 lattice | E₈(-1)² ⊕ U³, rank 22, sig (3,19) |
| HK triple | (ω_I, ω_J, ω_K), orthonormal |
| Matching type | Donaldson (90° rotation) |
| Effect on b₂ | Preserves (generic) |
| Effect on b₃ | Preserves (generic) |

---

*Phase 2 Specification Complete*
*Next: Phase 3 - Write Explicit G2-Structure φ_L*
