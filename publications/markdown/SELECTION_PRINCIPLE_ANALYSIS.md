# Selection Principle Analysis (v3.3 Research Note)

*Working document exploring the selection principle for GIFT formulas*

## 1. The Problem

GIFT uses specific combinations of topological invariants. Why these and not others?

Example: sin²θ_W = b₂/(b₃ + dim_G₂) = 21/91 = 3/13

Why not b₂/b₃ = 21/77 = 3/11 ≈ 0.273 (wrong)?

---

## 2. Catalog of GIFT Constants (mod 7)

| Constant | Value | mod 7 | Class |
|----------|-------|-------|-------|
| dim(K₇) | 7 | 0 | Fiber |
| b₂ | 21 = 3×7 | 0 | Gauge moduli |
| b₃ | 77 = 11×7 | 0 | Matter modes |
| dim(G₂) | 14 = 2×7 | 0 | Holonomy |
| b₃ + dim_G₂ | 91 = 13×7 | 0 | — |
| b₂ + b₃ | 98 = 14×7 | 0 | — |
| PSL(2,7) | 168 = 24×7 | 0 | Fano symmetry |
| fund(E₇) | 56 = 8×7 | 0 | — |
| H* | 99 = 14×7 + 1 | **1** | Cohomology |
| rank(E₈) | 8 = 7 + 1 | **1** | Cartan |
| δ_CP mod 7 | 197 = 28×7 + 1 | **1** | — |
| p₂ | 2 | **2** | Binary |
| det(g)_num | 65 = 9×7 + 2 | **2** | — |
| N_gen | 3 | **3** | Generations |
| dim(E₈) | 248 = 35×7 + 3 | **3** | Gauge DOF |
| D_bulk | 11 = 7 + 4 | **4** | Bulk dim |
| Weyl | 5 | **5** | — |
| κ_T⁻¹ | 61 = 8×7 + 5 | **5** | — |
| dim(E₆) | 78 = 11×7 + 1 | **1** | — |

---

## 3. Observed Pattern: Formulas Use "Closed" Combinations

### Pattern A: 0/0 → reduced fraction

| Formula | Numerator | Denominator | Both mod 7 | Result |
|---------|-----------|-------------|------------|--------|
| sin²θ_W | b₂ = 21 | b₃ + dim_G₂ = 91 | 0/0 | 3/13 |
| Q_Koide | dim_G₂ = 14 | b₂ = 21 | 0/0 | 2/3 |
| N_gen | b₂ = 21 | dim_K₇ = 7 | 0/0 | 3 |
| b₂/fund(E₇) | 21 | 56 | 0/0 | 3/8 |

**Hypothesis A**: Coupling ratios use quantities ≡ 0 (mod 7) in both numerator and denominator.

### Pattern B: Differences yield specific residues

| Formula | Expression | Value | mod 7 |
|---------|------------|-------|-------|
| Weyl | dim_G₂ - rank_E₈ - 1 | 14 - 8 - 1 = 5 | 5 |
| κ_T⁻¹ | b₃ - dim_G₂ - p₂ | 77 - 14 - 2 = 61 | 5 |
| det(g)_num | H* - b₂ - 13 | 99 - 21 - 13 = 65 | 2 |

**Hypothesis B**: Differences that yield Weyl-class (≡ 5 mod 7) appear in "capacity" quantities.

### Pattern C: Sums reduce to other GIFT constants

| Sum | Value | Reduces to |
|-----|-------|------------|
| b₃ + dim_G₂ | 91 | 7 × 13 = dim_K₇ × α_sum |
| b₂ + b₃ | 98 | 7 × 14 = dim_K₇ × dim_G₂ |
| rank_E₈ + N_gen | 11 | D_bulk |
| b₂ + dim_G₂ - N_gen | 32 | 2^Weyl |

**Hypothesis C**: Only combinations that reduce to products of other GIFT constants are "admissible."

---

## 4. The Fano Plane Connection

PSL(2,7) = 168 has **4 independent derivations**:
1. (b₃ + dim_G₂) + b₃ = 91 + 77 = 168
2. rank(E₈) × b₂ = 8 × 21 = 168
3. N_gen × (b₃ - b₂) = 3 × 56 = 168
4. 7 × 6 × 4 = 168 (Fano combinatorics)

The Fano plane PG(2,2) has:
- 7 points (imaginary octonions e₁...e₇)
- 7 lines (multiplication triples)
- Each line contains 3 points
- Each point lies on 3 lines

**Conjecture**: The "admissible" combinations respect Fano plane incidence structure.

---

## 5. Testing the Closure Hypothesis

### What if we used "wrong" combinations?

| Wrong formula | Value | Experimental | Why excluded? |
|---------------|-------|--------------|---------------|
| b₂/b₃ = 21/77 | 0.273 | sin²θ_W = 0.231 | Too high |
| dim_G₂/(b₂+1) = 14/22 | 0.636 | Q_Koide = 0.667 | Too low |
| b₂/(b₃+1) = 21/78 | 0.269 | — | — |
| (dim_G₂+1)/b₂ = 15/21 | 0.714 | — | — |

### Checking closure under mod 7:

**b₂/(b₃ + dim_G₂) = 21/91**:
- GCD(21, 91) = 7
- 21/7 = 3, 91/7 = 13
- Result: 3/13 (both coprime to 7) ✓

**b₂/b₃ = 21/77**:
- GCD(21, 77) = 7
- 21/7 = 3, 77/7 = 11
- Result: 3/11 (both coprime to 7) ✓

Both satisfy mod-7 closure... so that's not the distinguishing criterion.

---

## 6. Alternative: Calibrated Geometry Selection

On a G₂ manifold, there are two types of calibrated submanifolds:
- **Associative** (3-dimensional): calibrated by φ
- **Coassociative** (4-dimensional): calibrated by *φ

Physical observables might come from **integrals over calibrated cycles**:

| Observable | Cycle type | Dimension |
|------------|------------|-----------|
| Gauge couplings | Associative | 3 |
| Matter masses | Coassociative | 4 |
| Mixing angles | Both | 3+4=7 |

**Hypothesis D**: sin²θ_W = b₂/(b₃ + dim_G₂) involves adding dim_G₂ because it represents integration over G₂-invariant forms, not just Betti numbers.

---

## 7. The "Complete Graph K₇" Interpretation

b₂ = 21 = C(7,2) = edges in complete graph K₇
b₃ = 77 = C(7,3) + 2×21 = triangles + 2×edges

The formulas might select combinations that have graph-theoretic meaning:

| Formula | Graph interpretation |
|---------|---------------------|
| b₂/dim_K₇ = 21/7 = 3 | Edges per vertex |
| dim_G₂/b₂ = 14/21 = 2/3 | ??? |
| b₂/(b₃+dim_G₂) = 21/91 = 3/13 | ??? |

91 = b₃ + dim_G₂ = 77 + 14 = C(7,3) + 2×21 + 14 = 35 + 42 + 14 = 35 + 56

Hmm, 56 = fund(E₇) = b₃ - b₂ = 8×7

---

## 8. Emerging Selection Rule (Draft)

**Candidate Rule**: A formula f(x₁, ..., xₙ) is admissible if:

1. **Fano closure**: All xᵢ are ≡ 0 (mod 7) OR the combination reduces via GCD
2. **Graph interpretation**: The result has meaning in K₇ graph theory
3. **Exceptional chain**: The result connects to exceptional algebra dimensions

**Test**: Apply to sin²θ_W = b₂/(b₃ + dim_G₂)
- Fano closure: 21 ≡ 0, 91 ≡ 0, GCD = 7 → 3/13 ✓
- Graph: 3 = edges per vertex; 13 = ???
- Exceptional: 13 = α_sum = anomaly coefficient ✓

The "+dim_G₂" in the denominator might encode "holonomy contribution to gauge-matter balance."

---

## 9. Next Steps

1. **Formalize in Lean**: Define "admissible combination" predicate
2. **Enumerate**: List all admissible combinations under proposed rules
3. **Match**: Check which ones correspond to known physical observables
4. **Predict**: Do any admissible combinations predict unknown observables?

---

## 10. Key Observation: Multiple Derivations

The strongest patterns have **multiple independent derivations**:

| Quantity | # of derivations |
|----------|-----------------|
| Weyl = 5 | 4 (quadruple) |
| PSL(2,7) = 168 | 4 |
| N_gen = 3 | 3+ |
| 42 | 3 |

**Hypothesis E**: Physical quantities are those that arise from ≥3 independent paths through the topological invariants.

This would be a **redundancy principle**: nature selects for over-determined quantities.

---

*Status: Working draft. Not for publication.*
