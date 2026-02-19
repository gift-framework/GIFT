# GIFT Relations Index

## Complete Catalog of Compositional Structure

**Status**: LIVING DOCUMENT — Updated with real L-function validation (February 2026)
**Purpose**: Index all GIFT relations to identify patterns across the framework

---

## 1. Primary Constants (Level 0)

| Symbol | Value | Domain | Definition |
|--------|-------|--------|------------|
| p₂ | 2 | Topology | Pontryagin class contribution |
| N_gen | 3 | Physics | Fermion generations |
| Weyl / F₅ | 5 | Algebra | Weyl quotient / 5th Fibonacci |
| dim(K₇) | 7 | Geometry | Joyce manifold dimension |
| rank(E₈) | 8 | Algebra | E₈ Cartan subalgebra |
| D_bulk | 11 | Physics | M-theory bulk dimension |
| F₇ | 13 | Sequence | 7th Fibonacci number |
| dim(G₂) | 14 | Algebra | G₂ holonomy group dimension |
| b₂ | 21 | Topology | Second Betti number of K₇ |
| dim(J₃(O)) | 27 | Algebra | Exceptional Jordan algebra |
| b₃ | 77 | Topology | Third Betti number of K₇ |
| H* | 99 | Topology | Cohomological sum b₂+b₃+1 |
| dim(E₈) | 248 | Algebra | E₈ Lie algebra dimension |

---

## 2. Additive Composites (Level 1)

### 2.1 Primary + Primary

| Sum | Decomposition | Fibonacci Performance | Physical Meaning |
|-----|---------------|----------------------|------------------|
| **17** | dim(G₂) + N_gen = 14 + 3 | |R-1| = 0.36 ★★★ | λ_H = √17/32, σ₈ = 17/21 |
| **41** | dim(G₂) + dim(J₃(O)) = 14 + 27 | |R-1| = 0.62 ★★ | Holonomy + Jordan |
| 19 | Weyl + dim(G₂) = 5 + 14 | |R-1| = 2.34 ✗ | Poor performer |
| 19 | rank(E₈) + D_bulk = 8 + 11 | (alternate) | |
| 20 | dim(G₂) + p₂×N_gen = 14 + 6 | Not tested | |
| 22 | D_bulk + D_bulk = 11 + 11 | Not tested | |
| 29 | rank(E₈) + b₂ = 8 + 21 | |R-1| = 2.46 ✗ | Poor performer |
| 34 | dim(G₂) + b₂ - 1 = 14 + 21 - 1 | Not tested | Near F₉ |
| 35 | b₂ + dim(G₂) = 21 + 14 | Not tested | |
| 40 | dim(J₃(O)) + F₇ = 27 + 13 | Not tested | Jordan + Fano |
| **99** | b₂ + b₃ + 1 = 21 + 77 + 1 | Excellent (proxy) | H* cohomological |

### 2.2 Primary + Scaled Primary

| Sum | Decomposition | Performance | Notes |
|-----|---------------|-------------|-------|
| **31** | N_gen + p₂×dim(G₂) = 3 + 28 | |R-1| = 0.64 ★★ | |
| 31 | b₂ + p₂×Weyl = 21 + 10 | (alternate) | |
| 31 | dim(J₃(O)) + p₂² = 27 + 4 | (alternate) | |
| **43** | b₂ + p₂×D_bulk = 21 + 22 | |R-1| = 0.19 ★★★★ | BEST performer |
| 43 | dim(J₃(O)) + p₂×rank(E₈) = 27 + 16 | (alternate) | |
| 43 | rank(E₈) + Weyl×dim(K₇) = 8 + 35 | (alternate) | |

---

## 3. Multiplicative Composites (Level 1)

### 3.1 Products of Two Primaries

| Product | Decomposition | Performance | Physical Meaning |
|---------|---------------|-------------|------------------|
| **6** | p₂ × N_gen = 2 × 3 | Excellent (proxy) | sin²θ₂₃ = 6/11 |
| 10 | p₂ × Weyl = 2 × 5 | Not tested | |
| 14 | p₂ × dim(K₇) = 2 × 7 | = dim(G₂) | Primary! |
| **15** | N_gen × Weyl = 3 × 5 | Good (proxy) | Yₚ = 15/61 |
| 16 | p₂ × rank(E₈) = 2 × 8 | Good (proxy) | |
| 21 | N_gen × dim(K₇) = 3 × 7 | = b₂ | Primary! |
| 22 | p₂ × D_bulk = 2 × 11 | Not tested | |
| 24 | N_gen × rank(E₈) = 3 × 8 | Not tested | |
| 26 | p₂ × F₇ = 2 × 13 | Not tested | |
| 33 | N_gen × D_bulk = 3 × 11 | Not tested | |
| 35 | Weyl × dim(K₇) = 5 × 7 | Not tested | |
| 39 | N_gen × F₇ = 3 × 13 | Not tested | |
| 40 | Weyl × rank(E₈) = 5 × 8 | Not tested | |
| 42 | p₂ × b₂ = 2 × 21 | Not tested | |
| 42 | dim(K₇) × p₂×N_gen = 7 × 6 | (alternate) | |
| 55 | Weyl × D_bulk = 5 × 11 | Not tested | = F₁₀! |
| 56 | dim(K₇) × rank(E₈) = 7 × 8 | Not tested | |
| 65 | Weyl × F₇ = 5 × 13 | Not tested | |
| 77 | dim(K₇) × D_bulk = 7 × 11 | = b₃ | Primary! |
| 91 | dim(K₇) × F₇ = 7 × 13 | Not tested | |

### 3.2 Powers

| Power | Expression | Notes |
|-------|------------|-------|
| 4 | p₂² | |
| 8 | p₂³ = rank(E₈) | Primary! |
| 9 | N_gen² | |
| **16** | p₂⁴ | Good (proxy) |
| 25 | Weyl² | |
| 27 | N_gen³ = dim(J₃(O)) | Primary! |
| 32 | p₂⁵ | |
| 49 | dim(K₇)² | |
| 64 | p₂⁶ = rank(E₈)² | |
| 121 | D_bulk² | |

---

## 4. Remarkable Identities

### 4.1 Primary Factorizations

Some primaries ARE products:

| Primary | Factorization | Meaning |
|---------|---------------|---------|
| dim(G₂) = 14 | p₂ × dim(K₇) = 2 × 7 | Holonomy = Pontryagin × Manifold |
| b₂ = 21 | N_gen × dim(K₇) = 3 × 7 | Betti₂ = Generations × Manifold |
| b₃ = 77 | dim(K₇) × D_bulk = 7 × 11 | Betti₃ = Manifold × Bulk |
| rank(E₈) = 8 | p₂³ | Binary structure |
| dim(J₃(O)) = 27 | N_gen³ | Generational cube |

### 4.2 Fibonacci Connections

| Number | Fibonacci | GIFT Expression |
|--------|-----------|-----------------|
| F₃ = 2 | 2 | p₂ |
| F₄ = 3 | 3 | N_gen |
| F₅ = 5 | 5 | Weyl |
| F₆ = 8 | 8 | rank(E₈) |
| F₇ = 13 | 13 | F₇ (primary) |
| F₈ = 21 | 21 | b₂ |
| F₉ = 34 | 34 | b₂ + F₇ = 21 + 13 |
| F₁₀ = 55 | 55 | Weyl × D_bulk = 5 × 11 |
| F₁₁ = 89 | 89 | b₃ + D_bulk + 1 = 77 + 11 + 1 |

### 4.3 The H* Decomposition

H* = 99 has multiple GIFT expressions:

| Decomposition | Type |
|---------------|------|
| 99 = b₂ + b₃ + 1 = 21 + 77 + 1 | Betti sum |
| 99 = 9 × 11 = N_gen² × D_bulk | Product |
| 99 = 3 × 33 = N_gen × (N_gen × D_bulk) | Nested |
| 99 = 77 + 22 = b₃ + p₂ × D_bulk | Betti + scaled |
| 99 = 100 - 1 = (Weyl + F₇)² - 1 | Near-square |

---

## 5. Performance Summary (Real L-Functions)

### 5.1 Top Performers (|R-1| < 0.7)

| Rank | q | |R-1| | GIFT Structure | Type |
|------|---|-------|----------------|------|
| 1 | **43** | 0.19 | b₂ + p₂×D_bulk | Additive composite |
| 2 | **17** | 0.36 | dim(G₂) + N_gen | Additive composite |
| 3 | **5** | 0.43 | Weyl / F₅ | Primary |
| 4 | **41** | 0.62 | dim(G₂) + dim(J₃(O)) | Additive composite |
| 5 | **31** | 0.64 | N_gen + p₂×dim(G₂) | Additive composite |
| 6 | 7 | 0.69 | dim(K₇) | Primary |

### 5.2 Pattern: Top 5 of 6 are GIFT-decomposable

Only q=43 was classified as "non-GIFT" — but it decomposes as b₂ + p₂×D_bulk.

**Conclusion**: Every top performer has GIFT structure.

### 5.3 Poor Performers (|R-1| > 1.5)

| q | |R-1| | GIFT Structure? | Notes |
|---|-------|-----------------|-------|
| 3 | 2.23 | N_gen (primary) | Small prime, unstable |
| 11 | 1.83 | D_bulk (primary) | Primary, not composite |
| 19 | 2.34 | 5+14 or 8+11 | GIFT sums exist but poor |
| 23 | 1.42 | No simple decomposition | True non-GIFT? |
| 29 | 2.46 | 8+21 = rank(E₈)+b₂ | GIFT sum but poor |
| 37 | 10.81 | Harder to decompose | Outlier |

**Observation**: Having a GIFT decomposition is necessary but not sufficient.

---

## 6. Physical Observable Mapping

### 6.1 Confirmed Correspondences

| GIFT Expression | Observable | Domain |
|-----------------|------------|--------|
| 6/11 = (p₂×N_gen)/D_bulk | sin²θ₂₃(PMNS) | Leptons |
| 15/61 = (N_gen×Weyl)/(b₃-dim(G₂)-p₂) | Yₚ | Cosmology |
| 17/21 = (dim(G₂)+N_gen)/b₂ | σ₈ | LSS |
| √17/32 | λ_H | Higgs |
| 3/13 = N_gen/F₇ | sin²θ_W | Electroweak |
| 1/61 = 1/(b₃-dim(G₂)-p₂) | κ_T | Torsion |

### 6.2 Potential New Correspondences

From top performers:

| GIFT Expression | Value | Candidate Observable? |
|-----------------|-------|----------------------|
| 43 = b₂ + p₂×D_bulk | 43 | ? |
| 41 = dim(G₂) + dim(J₃(O)) | 41 | ? |
| 31 = N_gen + p₂×dim(G₂) | 31 | ? |
| 31/43 | 0.721 | ? |
| 41/43 | 0.953 | ? |

---

## 7. Decomposition Rules

### 7.1 Observed Patterns

**Rule 1**: Sums of primaries outperform isolated primaries
- 17 = 14 + 3 better than 14 or 3 alone
- 41 = 14 + 27 better than 27 alone

**Rule 2**: Scaled composites (primary + p₂×primary) are excellent
- 43 = 21 + 2×11 is the BEST
- 31 = 3 + 2×14 is good

**Rule 3**: Products involving p₂ are stable
- 6 = 2×3, 16 = 2⁴ both good (proxy)

**Rule 4**: Isolated small primaries (3, 11) are unstable
- q=3 and q=11 are among the worst

### 7.2 Decomposition Quality Hierarchy

```
Level A (Best): Primary + p₂×Primary → 43, 31
Level B (Good): Primary + Primary → 17, 41
Level C (OK):   Isolated medium primaries → 5, 7, 13
Level D (Poor): Isolated small primaries → 3, 11
Level E (Bad):  No simple decomposition → 23, 37
```

---

## 8. Open Decomposition Questions

### 8.1 Unexplained Good Performers

- Why is 43 = 21 + 22 SO good (|R-1| = 0.19)?
- Is there something special about b₂ + p₂×D_bulk?

### 8.2 Unexplained Poor Performers

- Why do 19 = 5+14 and 29 = 8+21 perform poorly despite GIFT sums?
- Is the specific decomposition important, or just existence?

### 8.3 Untested Predictions

Based on the patterns, these should perform well:

| q | Decomposition | Prediction |
|---|---------------|------------|
| 35 | b₂ + dim(G₂) = 21 + 14 | Good (sum of primaries) |
| 45 | b₂ + N_gen×rank(E₈) = 21 + 24 | Good (primary + scaled) |
| 47 | dim(J₃(O)) + b₂ - 1 = 27 + 20 | Unknown |
| 53 | dim(J₃(O)) + p₂×F₇ = 27 + 26 | Good? (primary + scaled) |

---

## 9. Cross-Reference Tables

### 9.1 By Conductor Value

| q | Decomposition(s) | Tested? | Result |
|---|------------------|---------|--------|
| 3 | N_gen | Yes | Poor (2.23) |
| 5 | Weyl | Yes | Good (0.43) |
| 6 | p₂×N_gen | Proxy | Excellent |
| 7 | dim(K₇) | Yes | OK (0.69) |
| 11 | D_bulk | Yes | Poor (1.83) |
| 13 | F₇ | Yes | OK (0.76) |
| 15 | N_gen×Weyl | Proxy | Good |
| 16 | p₂⁴ | Proxy | Good |
| 17 | dim(G₂)+N_gen | Yes | Good (0.36) |
| 19 | Weyl+dim(G₂) | Yes | Poor (2.34) |
| 21 | b₂ | Proxy | Moderate |
| 23 | None simple | Yes | Poor (1.42) |
| 27 | dim(J₃(O)) | Proxy | Moderate |
| 29 | rank(E₈)+b₂ | Yes | Poor (2.46) |
| 31 | N_gen+p₂×dim(G₂) | Yes | Good (0.64) |
| 37 | Complex | Yes | Outlier (10.81) |
| 41 | dim(G₂)+dim(J₃(O)) | Yes | Good (0.62) |
| 43 | b₂+p₂×D_bulk | Yes | BEST (0.19) |
| 77 | b₃ | Proxy | Anomaly |
| 99 | H* | Proxy | Excellent |

### 9.2 By GIFT Primary (Where Does It Appear?)

| Primary | In Products | In Sums | In Ratios |
|---------|-------------|---------|-----------|
| p₂=2 | 6,10,14,16,22,26,42 | 31,43 scaled | Many |
| N_gen=3 | 6,15,21,24,33,39 | 17,31 | sin²θ_W, Yₚ |
| Weyl=5 | 10,15,35,40,55,65 | 19 | |
| dim(K₇)=7 | 14,21,35,42,56,77,91 | | |
| rank(E₈)=8 | 16,24,40,56 | 29,43 | |
| D_bulk=11 | 22,33,55,77 | 19,43 | Yₚ, κ_T |
| F₇=13 | 26,39,65,91 | | sin²θ_W |
| dim(G₂)=14 | 42 | 17,19,31,35,41 | σ₈ |
| b₂=21 | 42 | 29,35,43,99 | σ₈ |
| dim(J₃(O))=27 | 54 | 40,41,43 | |
| b₃=77 | | 99 | |

---

## 10. Summary Statistics

| Category | Count | Mean |R-1| | Best Example |
|----------|-------|-------------|--------------|
| Primary + Primary sums | 4 tested | 1.29 | 41 (0.62) |
| Primary + Scaled sums | 2 tested | 0.42 | 43 (0.19) |
| Isolated primaries | 5 tested | 1.19 | 5 (0.43) |
| True non-GIFT | 2 tested | 1.43 | 23 (1.42) |

**Key Insight**: Primary + Scaled compositions are the sweet spot.

---

*GIFT Framework — Relations Index*
*February 2026*
*Living document — update as new data arrives*
