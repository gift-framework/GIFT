# TCS Blueprint for K7 Metric Construction

**Status**: Phase 0 Complete
**Convention**: Vol(K7) = 1 (normalized volume)
**Branch**: claude/explore-k7-metric-xMzH0

---

## 1. TCS Type Selection

We use **Kovalev's classical TCS construction** (2003):
- Two ACyl Calabi-Yau 3-folds glued along K3 × S¹
- Hyper-Kähler rotation matching on K3 cross-sections
- Exponentially decaying torsion in neck length L

**Why not extra-twisted**: Extra-TCS (Nordström 2023) gives disconnected moduli, useful for
distinguishing manifolds but adds complexity we don't need for explicit metric construction.

---

## 2. Building Blocks

### M₁: Quintic 3-fold in CP⁴
```
b₂(M₁) = 11
b₃(M₁) = 40
```
- Anticanonical divisor D₁ ≅ K3
- ACyl end: M₁ \ D₁ ~ (0,∞) × S¹ × K3

### M₂: Complete Intersection CI(2,2,2) in CP⁶
```
b₂(M₂) = 10
b₃(M₂) = 37
```
- Anticanonical divisor D₂ ≅ K3
- ACyl end: M₂ \ D₂ ~ (0,∞) × S¹ × K3

### Derived K7 Topology (Mayer-Vietoris)
```
b₂(K7) = b₂(M₁) + b₂(M₂) = 11 + 10 = 21
b₃(K7) = b₃(M₁) + b₃(M₂) = 40 + 37 = 77
H* = 1 + b₂ + b₃ = 1 + 21 + 77 = 99
```

**Reference**: Corti-Haskins-Nordström-Pacini (2015), Duke Math. J.

---

## 3. Normalization Convention

**CHOICE**: Volume normalization
```
Vol(K7, g̃_L) = 1
```

This implies:
- Metric scaled by λ = Vol(K7, g_L)^{-2/7}
- Spectral gap scales: λ₁(g̃_L) = λ₁(g_L) · Vol^{2/7}
- Diameter scales: diam(g̃_L) = diam(g_L) · Vol^{-1/7}

**Alternative** (not chosen): diam = 1 normalization complicates volume integrals.

---

## 4. Key References

| Result | Reference | Year |
|--------|-----------|------|
| TCS construction | Kovalev, J. reine angew. Math. 565 | 2003 |
| ACyl CY3 theory | Haskins-Hein-Nordström, JDG 101 | 2015 |
| Semi-Fano building blocks | CHNP, Duke Math. J. 164 | 2015 |
| Spectral density | Langlais, Comm. Math. Phys. 406 | 2024 |
| ν-invariant/moduli | Crowley-Goette-Nordström, Invent. Math. 239 | 2024 |
| Joyce existence | Joyce, Oxford UP | 2000 |

---

## 5. Hypotheses Assumed

### H1: ACyl Decay Rate
The ACyl CY3 metrics approach cylinders at rate μ > 0:
```
|g_± - g_cyl|_{C^k} ≤ C_k · e^{-μr}
```

### H2: K3 Matching Exists
There exists hyper-Kähler rotation r: Σ₊ → Σ₋ satisfying Donaldson's condition:
```
r*(ω_-) = Re(Ω_+)
r*(Re Ω_-) = ω_+
r*(Im Ω_-) = -Im(Ω_+)
```

### H3: Neck Length Regime
We work with L > L₀ where:
```
L₀ = 2v₀/h₀  (threshold from Model Theorem)
```

### H4: Torsion Smallness
The approximate G2-structure φ_L has torsion:
```
||T(φ_L)||_{C^k} ≤ C_k · e^{-δL}
```
which satisfies Joyce's threshold for L sufficiently large.

### H5: Spectral Gap Scaling
```
c₁/L² ≤ λ₁(g_L) ≤ c₂/L²
```
with explicit constants from Model Theorem.

### H6: Cheeger Inequality
The Cheeger constant satisfies:
```
h(K7) ≥ Vol(X)/Vol(K7) ~ 1/L
```
giving λ₁ ≥ h²/4 ~ 1/L².

---

## 6. Construction Roadmap

```
Phase 1: Define (V_±) ACyl CY3
    ↓
Phase 2: K3 matching (r, Ω, ω)
    ↓
Phase 3: Write φ_L = dθ ∧ ω + Re(Ω) on pieces
         Glue with cutoff → global φ_L
    ↓
Phase 4: IFT correction φ_L → φ̃_L (torsion-free)
         Bound: ||φ̃_L - φ_L||_{C^k} ≤ Ce^{-δL}
    ↓
Phase 5: Extract metric g̃_L, normalize Vol = 1
    ↓
Phase 6: Spectral bounds λ₁(g̃_L) ~ 1/L²
    ↓
Phase 7: Selection principle L² ~ κ·H* (conjecture)
```

---

## 7. Expected Outputs

| Phase | Deliverable | Format |
|-------|-------------|--------|
| 0 | This blueprint | Markdown |
| 1 | Building block spec | Lean + Markdown |
| 2 | K3 matching data | Markdown + Python |
| 3 | Explicit φ_L formulas | Notebook (GPU) |
| 4 | IFT operator + bounds | Lean + Notebook |
| 5 | Metric g̃_L components | JSON + Notebook |
| 6 | Spectral certificates | Lean + Notebook |
| 7 | Selection conjecture | Markdown |

---

## 8. Connection to GIFT Predictions

The TCS construction must recover:

| Quantity | GIFT Value | TCS Derivation |
|----------|------------|----------------|
| b₂ | 21 | 11 + 10 (Mayer-Vietoris) |
| b₃ | 77 | 40 + 37 (Mayer-Vietoris) |
| H* | 99 | 1 + 21 + 77 |
| det(g) | 65/32 | From G2 structure (Phase 5) |
| λ₁ | 14/99 | Via L² ~ H* selection (Phase 7) |
| sin²θ_W | 3/13 | b₂/(b₃ + dim G₂) = 21/91 ≠ 3/13 ❓ |

**Note**: The Weinberg angle formula needs verification against TCS cohomology.

---

## 9. Open Questions

1. **Building block uniqueness**: Are Quintic + CI(2,2,2) the unique blocks giving (21, 77)?
2. **Matching moduli**: How many distinct matchings exist? (affects moduli dimension)
3. **Selection mechanism**: What functional F selects L² ~ H*?
4. **ν-invariant**: What is ν̃(K7)? Does it distinguish from other G2 manifolds?

---

*Blueprint created: 2026-01-26*
*Next: Phase 1 - Building block ACyl CY3 specification*
