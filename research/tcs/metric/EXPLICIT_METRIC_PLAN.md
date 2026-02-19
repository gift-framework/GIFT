# Complete Explicit G₂ Metric on K7

**Goal**: Write down g_ij(x) for all x ∈ K7.

---

## 1. Global Strategy

K7 is built from:
```
K7 = (M₊ × S¹) ∪_Φ (M₋ × S¹)
```

We need metrics on:
1. **M₊ × S¹**: Product of ACyl CY3 with circle
2. **M₋ × S¹**: Product of ACyl CY3 with circle
3. **Neck region**: Gluing via cutoff functions

---

## 2. Coordinate Patches

### Patch 1: Core of M₊ (compact region)

Coordinates: (z₁, z₂, z₃, θ) where:
- (z₁, z₂, z₃) ∈ M₊ (complex coordinates on CY3)
- θ ∈ S¹ = [0, 2π)

### Patch 2: Core of M₋ (compact region)

Coordinates: (w₁, w₂, w₃, ψ) where:
- (w₁, w₂, w₃) ∈ M₋
- ψ ∈ S¹

### Patch 3: Cylindrical end of M₊

Coordinates: (r, x, θ) where:
- r ∈ [R₀, ∞) is radial (neck direction)
- x ∈ K3 (4 real coordinates)
- θ ∈ S¹

### Patch 4: Cylindrical end of M₋

Coordinates: (s, y, ψ) where:
- s ∈ (-∞, -R₀]
- y ∈ K3
- ψ ∈ S¹

### Patch 5: Neck (overlap region)

Coordinates: (t, x, θ, ψ) where:
- t ∈ [-L, L] (neck parameter)
- x ∈ K3
- θ, ψ ∈ S¹ (identified via hyper-Kähler rotation)

---

## 3. Metric on Each Patch

### On ACyl CY3 (M₊ cylindrical end)

The ACyl metric approaches:
```
g_CY = g_K3 + dr² + r²dφ²
```

More precisely (Haskins-Hein-Nordström):
```
g_CY = g_K3(x) + dr² + e^{-2μr}·h(r,x)
```

where h is the correction term decaying exponentially.

### On S¹ × CY3 (G₂ metric)

The product G₂ metric:
```
g_G2 = g_CY + dθ²
```

In components:
```
g_G2 = g_K3 + dr² + e^{-2μr}·h + dθ²
```

This is 7-dimensional: 4 (K3) + 1 (r) + 1 (φ from CY) + 1 (θ from S¹).

Wait - that's 7D but we need to be careful about the circle in CY3.

### Correct Structure

Actually, M₊ is asymptotic to (0,∞) × K3 × S¹_φ.

So S¹_θ × M₊ is asymptotic to (0,∞) × K3 × S¹_φ × S¹_θ.

The neck is: I × K3 × T² where T² = S¹_φ × S¹_θ.

Total: 1 + 4 + 2 = 7 ✓

---

## 4. The G₂ 3-Form

### On S¹ × CY3

```
φ = dθ ∧ ω_CY + Re(Ω_CY)
```

where:
- ω_CY = Kähler form of CY3
- Ω_CY = holomorphic (3,0)-form

### On the Neck (I × K3 × T²)

```
φ = dt ∧ ω_K3 + dθ ∧ dφ ∧ dt + Re(Ω_K3) ∧ dθ + Im(Ω_K3) ∧ dφ
```

Hmm, this needs more care. Let me use the standard formulas.

---

## 5. Standard G₂ Structure from SU(3)

Given SU(3)-structure (ω, Ω) on a 6-manifold X, the G₂-structure on S¹ × X is:

```
φ = dθ ∧ ω + Re(Ω)
ψ = (1/2)ω ∧ ω + Im(Ω) ∧ dθ
```

The metric is:
```
g_7 = dθ² + g_6
```

where g_6 is the SU(3) metric on X.

### Torsion-Free Condition

φ torsion-free ⟺ dφ = 0 AND d*φ = 0

For product structure:
- dφ = dθ ∧ dω + d(Re Ω) = 0 ⟺ dω = 0 and d(Re Ω) = 0
- d*φ = d((1/2)ω² + Im(Ω)∧dθ) = ω∧dω + d(Im Ω)∧dθ = 0

For CY3: dω = 0, dΩ = 0, so torsion-free ✓

---

## 6. Explicit K3 Metric (Needed for Neck)

### Kummer Surface Model

A concrete K3 can be built as resolution of T⁴/ℤ₂.

Coordinates on T⁴: (x₁, x₂, x₃, x₄) ∈ (ℝ/ℤ)⁴

The ℤ₂ action: xᵢ → -xᵢ

Fixed points: 16 points (2⁴ choices of 0 or 1/2)

### Eguchi-Hanson Resolution

Near each fixed point, replace the singularity with Eguchi-Hanson space:
```
g_EH = (1 - (a/r)⁴)⁻¹ dr² + r²(1 - (a/r)⁴)σ₁² + r²(σ₂² + σ₃²)
```

where σᵢ are left-invariant 1-forms on S³.

### Kähler Form

```
ω_K3 = (1/2)(dx₁∧dx₂ + dx₃∧dx₄) + (EH corrections)
```

### Holomorphic 2-Form

```
Ω_K3 = (dx₁ + i·dx₂) ∧ (dx₃ + i·dx₄) + (corrections)
```

---

## 7. Putting It Together

### Neck Region Metric (t ∈ [-L, L])

```
g_neck = dt² + g_K3(x) + dθ² + dψ²
```

This is flat T² fibered over K3 × I.

In coordinates (t, x₁, x₂, x₃, x₄, θ, ψ):

```
g_neck = diag(1, g_K3, 1, 1)
```

where g_K3 is the 4×4 K3 metric.

### Compact Region Metric (M₊)

For the Quintic CY3 in ℂℙ⁴:
```
g_Quintic = Fubini-Study restricted to {P₅(z) = 0}
```

where P₅ is a degree-5 polynomial.

### Gluing

Use cutoff function χ(t):
- χ(t) = 1 for t < -L + δ
- χ(t) = 0 for t > L - δ
- Smooth interpolation in between

```
g_global = χ(t)·g₊ + (1-χ(t))·g₋ + (transition terms)
```

---

## 8. Numerical Implementation

### Step 1: Discretize K3

Create a mesh on K3 (e.g., 32³ grid on T⁴/ℤ₂).

### Step 2: Compute K3 Metric

Use known formulas or solve Monge-Ampère numerically.

### Step 3: Extend to 7D

Add the t, θ, ψ coordinates.

### Step 4: Apply Cutoff

Blend the two sides using χ(t).

### Step 5: Output

Store g_ij(x) as a tensor field on the 7D mesh.

---

## 9. Simplification: Flat K3 Approximation

For a first version, approximate K3 as flat T⁴:
```
g_K3 ≈ dx₁² + dx₂² + dx₃² + dx₄²
```

This is WRONG for Ricci-flat, but gives a tractable starting point.

The G₂ metric becomes:
```
g_7 = dt² + dx₁² + dx₂² + dx₃² + dx₄² + dθ² + dψ²
```

This is flat T⁷! Not G₂ holonomy, but useful for testing code.

---

## 10. Next Steps

1. **Implement flat T⁷ test case**
2. **Add K3 Eguchi-Hanson corrections**
3. **Add ACyl corrections on ends**
4. **Verify G₂ constraints numerically**
5. **Export to JSON/HDF5 format**

---

*Metric Construction Plan: 2026-01-26*
