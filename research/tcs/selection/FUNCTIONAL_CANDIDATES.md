# Selection Functional Candidates

**Goal**: Find F(L) such that ∂F/∂L = 0 implies L² ∝ H*.

---

## 1. Torsion Energy Functional

### Definition

```
F_torsion(L) = ∫_{K_L} ||T(φ_L)||² dvol
```

where T(φ_L) is the torsion of the approximate G₂-structure before IFT correction.

### Scaling Analysis

On TCS with neck length L:
- Neck volume: Vol_neck ~ L
- Torsion on neck: ||T|| ~ e^{-δL} (exponential decay from ACyl)
- Compact pieces: ||T|| ~ O(1) but volume fixed

```
F_torsion(L) ~ const + L · e^{-2δL}
```

### Critical Point

```
∂F/∂L = e^{-2δL} - 2δL · e^{-2δL} = e^{-2δL}(1 - 2δL) = 0
```

Gives: L* = 1/(2δ)

**Problem**: This doesn't involve H* at all! The minimum is set by decay rate δ, not topology.

**Verdict**: ❌ F_torsion doesn't give L² ∝ H*.

---

## 2. IFT Correction Size

### Definition

```
F_IFT(L) = ||φ̃_L - φ_L||_{C^k}
```

where φ̃_L is the torsion-free correction.

### Scaling

From Kovalev/IFT theory:
```
||φ̃_L - φ_L|| ~ C · ||T(φ_L)|| ~ C · e^{-δL}
```

### Critical Point

F_IFT is monotonically decreasing in L (larger neck = smaller correction).

**No minimum** - F_IFT → 0 as L → ∞.

**Verdict**: ❌ F_IFT doesn't select finite L.

---

## 3. Gluing Mismatch Functional

### Definition

At the gluing region, we match two ACyl structures. Define:
```
F_mismatch(L) = ∫_{Σ_L} ||φ_+ - Φ*(φ_-)||² dA
```

where Σ_L is the gluing hypersurface at neck parameter L.

### Scaling

The mismatch comes from:
- Cut-off function derivatives: ~ 1/L
- Exponential tails from ACyl: ~ e^{-μL}

```
F_mismatch(L) ~ A/L² + B·e^{-2μL}
```

### Critical Point

```
∂F/∂L = -2A/L³ - 2μB·e^{-2μL} = 0
```

Gives: L³ · e^{-2μL} = -A/(μB)

**Problem**: Negative RHS if A, B > 0. No real solution.

**Verdict**: ❌ F_mismatch doesn't have a minimum (monotonic).

---

## 4. Spectral Functional (Self-Referential)

### Definition

```
F_spectral(L) = |λ₁(L) - λ_target|²
```

where λ_target = dim(G₂)/H*.

### Problem

This **presupposes** the target value - it's circular!

**Verdict**: ❌ Circular definition.

---

## 5. Volume-Normalized Eigenvalue

### Definition

```
F_VN(L) = λ₁(L) · Vol(K_L)^{2/7}
```

This is scale-invariant (dimensionless).

### Scaling

For TCS:
- Vol(K_L) ~ V₀ + V₁·L (linear in L for neck contribution)
- λ₁(L) ~ π²/L²

```
F_VN(L) ~ (π²/L²) · (V₀ + V₁L)^{2/7}
```

For large L:
```
F_VN(L) ~ π² · V₁^{2/7} · L^{-12/7}
```

### Critical Point

F_VN is monotonically decreasing for large L.

**Verdict**: ❌ No finite minimum.

---

## 6. Holonomy Deficit Functional ⭐

### Idea

The holonomy of g_L is "almost G₂" but not exactly until L is large enough.
Define a functional measuring "how far from exact G₂":

```
F_hol(L) = ||Hol(g_L) - G₂||
```

where the norm measures the gap in the holonomy group.

### Heuristic

- For small L: gluing is rough, holonomy is larger than G₂
- For large L: holonomy approaches G₂

But there's a **cost** to large L: the manifold becomes "thin" (large diameter).

### Combined Functional

```
F_combined(L) = ||Hol - G₂|| + μ · diam(K_L)²
```

The second term penalizes large L (large diameter).

### Scaling

- ||Hol - G₂|| ~ e^{-δL} (holonomy error decays)
- diam(K_L) ~ L (diameter grows with neck)

```
F_combined(L) ~ e^{-δL} + μL²
```

### Critical Point

```
∂F/∂L = -δe^{-δL} + 2μL = 0
δe^{-δL} = 2μL
```

This gives L* implicitly. For small μ: L* ~ (1/δ)log(δ/(2μ)).

**Problem**: Still doesn't involve H* directly.

**Verdict**: ⚠️ Promising structure but needs H* connection.

---

## 7. Topological Constraint Functional ⭐⭐

### Key Insight

What if the selection comes from a **constraint**, not a minimum?

The TCS construction must produce a manifold with:
- Fixed b₂, b₃ (from building blocks)
- Finite diameter
- Torsion-free G₂ structure

### Constraint Approach

The neck length L is **not free** - it's determined by requiring:
```
The IFT correction exists (small torsion condition)
```

Joyce's theorem requires ||T(φ_L)|| < ε₀ for some threshold ε₀.

If ||T(φ_L)|| ~ Ce^{-δL}, then:
```
Ce^{-δL} < ε₀ ⟹ L > (1/δ)log(C/ε₀)
```

This gives a **minimum L**, not a specific L*.

**Verdict**: ⚠️ Gives lower bound, not exact value.

---

## 8. Moduli Space Boundary ⭐⭐⭐

### Idea

The moduli space of TCS structures might have a **boundary** at L = L*.

For L < L*: no torsion-free solution exists (obstruction).
For L ≥ L*: solutions exist.

The physical manifold sits at **the boundary** L = L*.

### Why the Boundary?

Possible reasons:
1. **Index jump**: The linearized operator changes index at L*
2. **Kernel creation**: A harmonic form appears at L*
3. **Topological transition**: Something changes at L*

### Connection to H*

If the boundary is set by a topological condition:
```
L* = f(b₂, b₃, dim(G₂)) = f(H*, 14)
```

Then L*² ∝ H* could emerge from the topology of the moduli space.

**Verdict**: ⭐⭐⭐ Most promising conceptually, but needs proof.

---

## 9. Entropy Maximization

### Microcanonical Ensemble

Define:
```
S(L) = log(# of G₂ structures on K_L with energy < E)
```

The physical L* maximizes entropy:
```
∂S/∂L = 0
```

### Connection to Spectral Theory

The number of states below energy E is related to spectral density:
```
N(E) ~ Vol · E^{7/2} (Weyl law in 7D)
```

If Vol ~ L and we fix some scale via H*:
```
S(L) ~ log(L · E^{7/2})
```

**Problem**: This is too crude - need more precise spectral information.

**Verdict**: ⚠️ Interesting but incomplete.

---

## 10. Summary: Which F(L)?

| Functional | Has Minimum? | Involves H*? | Testable? |
|------------|--------------|--------------|-----------|
| F_torsion | Yes | No | Yes |
| F_IFT | No | No | Yes |
| F_mismatch | No | No | Yes |
| F_spectral | Yes | Circular | No |
| F_VN | No | No | Yes |
| F_hol + penalty | Yes | No | Hard |
| Constraint | Lower bound | Maybe | Yes |
| Moduli boundary | Sharp | Maybe | Hard |
| Entropy | Maybe | Maybe | Hard |

### Recommendation

**Primary path**: Investigate the **moduli space boundary** hypothesis.

**Secondary path**: Test whether numerical torsion minimization gives L² ∝ H*.

**Tertiary path**: Look for an index-theoretic constraint involving H*.

---

## 11. A Concrete Proposal

### The "Harmonic Threshold" Functional

Define:
```
F_harm(L) = dim(ker(Δ_L^(2))) - b₂
```

This measures whether the kernel of the Laplacian on 2-forms matches the topological b₂.

For torsion-free G₂:
- Hodge theorem: dim(ker(Δ^(2))) = b₂
- So F_harm = 0

For approximate φ_L (not torsion-free):
- F_harm might be nonzero
- F_harm(L*) = 0 at the threshold where Hodge theorem applies

### Testable Prediction

At L = L*, the harmonic 2-forms match the Betti number exactly.

For L < L*: dim(ker) ≠ b₂ (not enough neck to resolve topology).
For L ≥ L*: dim(ker) = b₂ (Hodge theorem holds).

**This gives a sharp transition at L***!

And if L* is set by requiring b₂ harmonic forms to exist:
```
L*² ~ (some function of b₂, b₃, and G₂ constraints)
```

This could naturally involve H*.

---

*Next: Numerical test of F_harm transition*
