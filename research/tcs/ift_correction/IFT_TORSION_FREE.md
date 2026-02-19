# Torsion-Free Correction via Implicit Function Theorem

**Phase 4**: The analytic core - correct φ_L to exact torsion-free φ̃_L.

---

## 1. The Correction Problem

### Given
- Approximate G2-structure φ_L with small torsion: ||T(φ_L)|| ≤ Ce^{-δL}
- Want exact solution: T(φ̃_L) = 0

### Strategy
Perturb φ_L by a small exact form dη:
```
φ̃_L = φ_L + dη
```
and solve for η using the Implicit Function Theorem.

---

## 2. The Torsion Operator

### Definition

The torsion of a G2-structure φ is encoded in:
```
T(φ) = (dφ, d*φ)
```

For a **closed** G2-structure (dφ = 0), torsion reduces to:
```
T(φ) = d*φ ∈ Ω⁴(M)
```

### Linearization

At a torsion-free φ₀, the linearization is:
```
DT|_{φ₀}: Ω³(M) → Ω⁴(M)
DT|_{φ₀}(α) = d*α + (nonlinear terms)
```

For perturbations of the form φ = φ₀ + dη:
```
DT|_{φ₀}(dη) = d(*dη) + O(||dη||²)
```

---

## 3. Gauge Fixing

### Problem
The operator DT is not elliptic without gauge fixing.

### Bianchi Gauge
Require η to be **coclosed**:
```
d*η = 0
```

Then the full system becomes:
```
d*dη + dd*η = -T(φ_L) + O(||η||²)
```

which is the **Hodge Laplacian** on 2-forms:
```
Δη = d*dη + dd*η
```

### Elliptic System
The gauge-fixed equation:
```
Δη = F(η, φ_L)
```
where F encodes the torsion and nonlinear corrections.

---

## 4. Function Spaces

### Sobolev Spaces

For k ≥ 4, α ∈ (0, 1):
```
W^{k,2}(M) = {u : ||u||_{W^{k,2}} < ∞}
```

with weighted norms on the neck:
```
||u||_{W^{k,2}_δ} = ||e^{δr} u||_{W^{k,2}}
```

### Mapping Properties

The Laplacian:
```
Δ: W^{k+2,2}(M) → W^{k,2}(M)
```
is Fredholm with index 0 (for appropriate weights).

---

## 5. The Implicit Function Theorem Setup

### Abstract Form

Let:
- X = W^{k+2,2}(Ω²) (space of 2-forms)
- Y = W^{k,2}(Ω⁴) (space of 4-forms)
- F: X × ℝ₊ → Y defined by F(η, L) = T(φ_L + dη)

We want to solve F(η, L) = 0 for η near 0.

### IFT Conditions

1. **F(0, L) small**: ||T(φ_L)|| ≤ Ce^{-δL} ✓
2. **D_η F(0, L) invertible**: Δ on 2-forms is invertible (no harmonic 2-forms on K7 matching kernel)
3. **Lipschitz continuity**: ||F(η₁, L) - F(η₂, L)|| ≤ C||η₁ - η₂||

### Conclusion

For L ≥ L₀ sufficiently large, there exists unique η_L with:
```
||η_L||_{W^{k+2,2}} ≤ C·||T(φ_L)||_{W^{k,2}} ≤ C'e^{-δL}
```

---

## 6. Uniform Invertibility on Long Neck

### The Key Estimate

For TCS manifolds, the linearized operator:
```
Δ: W^{k+2,2}(Ω²) → W^{k,2}(Ω²)
```
has **uniform inverse** for L large:
```
||Δ^{-1}|| ≤ C (independent of L)
```

### Why This Works

1. On compact pieces: Standard elliptic theory
2. On neck: Separation of variables → spectral gap from K3 × T²
3. Matching: Patching estimates via finite overlap

**Reference**: Kovalev (2003), Section 4; CHNP (2015), Appendix.

---

## 7. The Nonlinear Terms

### Expansion

```
T(φ_L + dη) = T(φ_L) + DT|_{φ_L}(dη) + Q(dη, dη) + O(||dη||³)
```

where Q is the quadratic part.

### Quadratic Estimate

```
||Q(dη, dη)||_{W^{k,2}} ≤ C||dη||²_{W^{k+1,2}}
```

### Contraction

For ||η|| ≤ ε small:
```
||N(η)||_{W^{k,2}} ≤ C||η||²_{W^{k+2,2}}
```

This gives contraction for η in a ball of radius O(e^{-δL}).

---

## 8. Main Theorem

### Statement

**Theorem (Kovalev-type existence)**: For L ≥ L₀, there exists a unique torsion-free G2-structure φ̃_L on K7 such that:

1. **Torsion-free**: dφ̃_L = 0 and d*φ̃_L = 0
2. **Close to approximate**: ||φ̃_L - φ_L||_{C^k} ≤ C_k e^{-δL}
3. **Same cohomology class**: [φ̃_L] = [φ_L] ∈ H³(K7)

### Proof Sketch

1. Set up F(η, L) = T(φ_L + dη)
2. Verify F(0, L) = O(e^{-δL}) (torsion estimate from Phase 3)
3. Verify D_η F invertible with uniform bound
4. Apply Banach fixed point / IFT
5. Bootstrap to C^k estimates

---

## 9. Holonomy Consequence

### From Torsion-Free to G2 Holonomy

**Theorem (Joyce)**: If (M⁷, φ, g) is compact with torsion-free G2-structure and π₁(M) finite, then Hol(g) = G₂ exactly.

### For K7

- K7 is simply connected: π₁(K7) = 0 (from TCS construction)
- φ̃_L is torsion-free
- Therefore: **Hol(g̃_L) = G₂**

---

## 10. Quantitative Bounds

### Torsion Decay

```
||T(φ_L)||_{C^k} ≤ C_k · e^{-δL}
```

where δ depends on:
- ACyl decay rate μ of building blocks
- Cut-off function derivatives

Typically δ ≈ μ/2.

### Correction Size

```
||φ̃_L - φ_L||_{C^k} ≤ C_k · e^{-δL}
||g̃_L - g_L||_{C^k} ≤ C'_k · e^{-δL}
```

### Exponential Closeness

This is the key for Phase 5-6: the exact metric g̃_L is **exponentially close** to the explicit approximate metric g_L.

---

## 11. Schematic of IFT Argument

```
                    ┌─────────────────────┐
                    │    φ_L (approx)     │
                    │  T(φ_L) ~ e^{-δL}   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Linearize: Δη = f  │
                    │  Solve: η = Δ⁻¹f    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Iterate: Fixed pt  │
                    │  ||η|| ~ e^{-δL}    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   φ̃_L = φ_L + dη    │
                    │   T(φ̃_L) = 0        │
                    └─────────────────────┘
```

---

## 12. Literature Axioms (for Lean)

From `LiteratureAxioms.lean`:

```lean
/-- CGN Proposition: Torsion-free correction is exponentially close -/
axiom cgn_torsion_free_correction (K : TCSManifold) (L : ℝ) (hL : L > K.L₀) :
  ∃ φ̃ : TorsionFreeG2 K.toManifold,
    ‖φ̃.form - K.approxG2(L).form‖_{C^k} ≤ K.C_k * Real.exp (-K.δ * L)
```

---

*Phase 4 Complete*
*Next: Phase 5 - Extract Metric and Normalize*
