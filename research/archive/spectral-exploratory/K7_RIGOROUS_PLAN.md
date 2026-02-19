# K₇ Explicit Metric: Rigorous Construction Plan

**Date**: 2026-01-26
**Goal**: Construct a "solidly explicit" G₂ metric on K₇
**Status**: In Progress

---

## Overview

Following the rigorous methodology, we need to deliver:

1. **Coframe + structure equations** (deⁱ explicit)
2. **φ explicit** in this coframe
3. **Proof dφ = 0 and d*φ = 0** (or gluing theorem + e^{-δL} bound)
4. **Spectrum** computed on real operator with convergence control

---

## Step 1: Structure Equations

### The Ansatz

Base space: S³ × S³ × S¹ (or quotient by Γ)

Each S³ ≃ SU(2) has left-invariant Maurer-Cartan forms.

### SU(2) Left-Invariant Forms

For the first S³, use coordinates (ψ₁, θ₁, φ₁) (Euler angles):

```
σ¹ = cos(ψ₁)dθ₁ + sin(ψ₁)sin(θ₁)dφ₁
σ² = sin(ψ₁)dθ₁ - cos(ψ₁)sin(θ₁)dφ₁
σ³ = dψ₁ + cos(θ₁)dφ₁
```

Structure equations:
```
dσ¹ = -σ² ∧ σ³
dσ² = -σ³ ∧ σ¹
dσ³ = -σ¹ ∧ σ²
```

Or compactly: **dσⁱ = -½ εⁱⱼₖ σʲ ∧ σᵏ**

Similarly for second S³ with forms {Σⁱ}.

### The Coframe

```
e¹ = a σ¹       e⁴ = b Σ¹
e² = a σ²       e⁵ = b Σ²
e³ = a σ³       e⁶ = b Σ³
e⁷ = c (dθ + A)
```

where:
- a, b, c are constants (radii)
- A is a 1-form on S³ × S³ (connection)
- θ ∈ [0, 2π) is the S¹ coordinate

### Structure Equations for eⁱ

From dσⁱ = -½ εⁱⱼₖ σʲ ∧ σᵏ and eⁱ = a σⁱ:

```
de¹ = a dσ¹ = -a σ² ∧ σ³ = -(1/a) e² ∧ e³
de² = -(1/a) e³ ∧ e¹
de³ = -(1/a) e¹ ∧ e²
```

Similarly:
```
de⁴ = -(1/b) e⁵ ∧ e⁶
de⁵ = -(1/b) e⁶ ∧ e⁴
de⁶ = -(1/b) e⁴ ∧ e⁵
```

For e⁷:
```
de⁷ = c dA = c F
```
where F = dA is the curvature 2-form.

### Connection Ansatz

The most general left-invariant A on S³ × S³:
```
A = α₁ σ¹ + α₂ σ² + α₃ σ³ + β₁ Σ¹ + β₂ Σ² + β₃ Σ³
```

By symmetry, take:
```
A = α (σ³ + Σ³)
```
(diagonal ansatz along Hopf fibers)

Then:
```
F = dA = α (dσ³ + dΣ³) = -α (σ¹∧σ² + Σ¹∧Σ²)
       = -(α/a²)(e¹∧e² ) - (α/b²)(e⁴∧e⁵)
```

---

## Step 2: The G₂ 3-Form φ

### Standard G₂ Form

In an orthonormal coframe (e¹,...,e⁷), the standard G₂ 3-form is:

```
φ = e¹²⁷ + e³⁴⁷ + e⁵⁶⁷ + e¹³⁵ - e¹⁴⁶ - e²³⁶ - e²⁴⁵
```

where eⁱʲᵏ = eⁱ ∧ eʲ ∧ eᵏ.

### Explicit Form

```
φ = e¹ ∧ e² ∧ e⁷ + e³ ∧ e⁴ ∧ e⁷ + e⁵ ∧ e⁶ ∧ e⁷
  + e¹ ∧ e³ ∧ e⁵ - e¹ ∧ e⁴ ∧ e⁶ - e² ∧ e³ ∧ e⁶ - e² ∧ e⁴ ∧ e⁵
```

---

## Step 3: Torsion Calculation

### Compute dφ

Using structure equations and Leibniz rule:

```
d(e¹ ∧ e² ∧ e⁷) = de¹ ∧ e² ∧ e⁷ - e¹ ∧ de² ∧ e⁷ + e¹ ∧ e² ∧ de⁷
```

This requires systematic computation of all 7 terms.

### The Torsion Classes

For a G₂ structure, the torsion decomposes as:
```
dφ = τ₀ *φ + 3τ₁ ∧ φ + *τ₃
d*φ = 4τ₁ ∧ *φ + τ₂ ∧ φ
```

For **torsion-free** G₂: τ₀ = τ₁ = τ₂ = τ₃ = 0, i.e., dφ = d*φ = 0.

### Expected Constraints

The calculation will yield constraints on (a, b, c, α) of the form:
- Relations between a and b
- Conditions on α (connection strength)
- Possibly c determined by other parameters

---

## Step 4: Solving Constraints

### Case A: Exact Solution

If we find (a, b, c, α) such that dφ = d*φ = 0 exactly, we have a torsion-free G₂ metric.

### Case B: TCS Gluing (More Likely)

If no exact solution exists in this ansatz:

1. Find a **closed** G₂ structure (dφ = 0) with small torsion
2. Use gluing theorem: there exists φ̃_L torsion-free with
   ```
   ‖φ̃_L - φ_L‖_{C^k} ≤ C e^{-δL}
   ```

This is the standard approach in modern G₂ geometry (Kovalev, Corti-Haskins-Nordström-Pacini).

---

## Step 5: Topology Verification

### Target: K₇ with b₂ = 21, b₃ = 77

Need to verify:
- b₁ = 0 (simply connected)
- b₂ = 21
- b₃ = 77
- H* = b₂ + b₃ + 1 = 99

### Method: Mayer-Vietoris

For TCS: K₇ = M₁ ∪_Σ M₂

Long exact sequence gives Betti numbers from:
- Betti numbers of building blocks M₁, M₂
- Betti numbers of cross-section Σ = K3 × S¹

### For S³ × S³ × S¹ Quotient

If K₇ = (S³ × S³ × S¹)/Γ:
- Compute H*(S³ × S³ × S¹) = ℤ in degrees 0,1,3,3,4,6,7
- Apply Γ action
- Use transfer/spectral sequence

---

## Step 6: Spectrum Computation

### The Real Laplacian

Once g is explicit, compute:
```
Δ_g f = (1/√|g|) ∂_i (√|g| g^{ij} ∂_j f)
```

### Methods

**A. Symbolic (SageMath/SymPy)**:
- For product/warped product: separation of variables
- Get explicit eigenvalue equations

**B. Numerical (FEM/DEC)**:
- Discretize on mesh
- CuPy + sparse eigensolver
- Convergence study

**C. Lean Verification**:
- Formalize bounds
- Verify key identities

### Convergence Check

- Vary mesh resolution
- Compare different discretizations
- Error bounds on λ₁

---

## Step 7: Final Deliverables

### "Solidly Explicit" Pack

1. **Document**: `K7_STRUCTURE_EQUATIONS.md`
   - Coframe {eⁱ}
   - Structure equations {deⁱ}

2. **Document**: `K7_G2_FORM.md`
   - φ explicit
   - *φ explicit

3. **Document**: `K7_TORSION_CALCULATION.md`
   - dφ computation
   - d*φ computation
   - Constraint equations

4. **Document**: `K7_CONSTRAINTS_SOLUTION.md`
   - Exact solution OR
   - TCS gluing with bounds

5. **Document**: `K7_TOPOLOGY.md`
   - Betti number calculation
   - Verification b₂=21, b₃=77

6. **Notebook**: `K7_Spectrum_FEM.ipynb`
   - Discretization
   - Eigenvalue computation
   - Convergence analysis

7. **Lean**: Formalization of key results

---

## Timeline & Resources

| Step | Complexity | Tools |
|------|------------|-------|
| 1. Structure equations | Medium | Pen & paper / SymPy |
| 2. φ explicit | Easy | Direct substitution |
| 3. Torsion | Hard | SageMath / symbolic |
| 4. Constraints | Hard | Algebraic solving |
| 5. Topology | Medium | Mayer-Vietoris |
| 6. Spectrum | Hard | CuPy A100 / FEM |
| 7. Assembly | Easy | Documentation |

---

## Next Action

Start with **Step 1**: Write explicit structure equations.

---

*GIFT Framework — Rigorous K₇ Construction*
*2026-01-26*
