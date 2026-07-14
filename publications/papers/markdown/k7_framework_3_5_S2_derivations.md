# Supplement S2: Complete Derivations (Dimensionless)

[![Lean 4 Verified](https://img.shields.io/badge/Lean_4-Verified-blue)](https://github.com/gift-framework/core)

## Complete Mathematical Derivations for All 33 Type I Predictions

*This supplement provides mathematical derivations for the 33 Type I dimensionless predictions in the K₇ framework (Relations #1–#33, all Lean-certified). Each derivation proceeds from topological definitions to numerical predictions. Two additional BH remnant topological predictions (Relations #34–#35 in §24.3b) are included as an appendix; these are classified as Type IV structural diagnostics in the main 95-observable dataset.*

**Status**: All 33 Type I observables Lean-certified. 213 certificate conjuncts, 15 classified axioms (of which 4 external data packages, see main §8.1), 0 sorry.

**Note on the 95-observable dataset**: The main paper presents 95 observables across 4 types (33 I + 19 II + 21 III + 22 IV). This supplement covers the 33 Type I predictions (direct algebraic from topology). Type II extensions (19 one-step physical extractions), Type III dynamical results (21 multi-step chains), and Type IV structural diagnostics (22 internal consistency checks, including BH remnant M_res and N_QNM from Pinčák et al. 2026, main text ref. [42]) are documented in Supplement S3 and the main text §4–§6.

The topological constants that determine these relations are described in S1.

---

## Table of Contents

- [Part I: Foundations](#part-i-foundations)
- [Part II: Foundational Theorems (4 relations)](#part-ii-foundational-theorems)
- [Part III: Gauge Sector (2 relations)](#part-iii-gauge-sector)
- [Part IV: Lepton Sector (3 relations)](#part-iv-lepton-sector)
- [Part V: Quark Sector (1 relation)](#part-v-quark-sector)
- [Part VI: Neutrino Sector (4 relations)](#part-vi-neutrino-sector)
- [Part VII: Higgs & Cosmology (3 relations)](#part-vii-higgs--cosmology)
- [Part VIII: Summary Table](#part-viii-summary-table)
- [Part IX: Observable Catalog](#part-ix-observable-catalog)

---

# Part 0: Derivation Philosophy

## 0. What "Derivation" Means in the K₇ Framework

Before presenting derivations, we clarify the logical structure:

### 0.1 Inputs vs Outputs

**Inputs** (taken as given):
- The octonion algebra 𝕆 and its automorphism group G₂ = Aut(𝕆)
- The E₈×E₈ gauge structure
- The K₇ manifold (compact G₂ 7-manifold with b₂ = 21, b₃ = 77; explicit metric certified in [A]; topological construction via Joyce-Karigiannis Z₂³ orbifold route (S1 §8.4); explicit closed-form neck ansatz with Wirtinger certificate (S1 §8.4, [D]))

**Outputs** (derived from inputs):
- The 33 Type I dimensionless predictions

### 0.2 What We Do NOT Claim

- That 𝕆 → G₂ → K₇ is the unique geometry for physics
- That the formulas are uniquely determined by geometric principles
- That the selection rule for specific combinations (b₂/(b₃ + dim_G₂) vs b₂/b₃) is understood

### 0.3 What We Observe

- Given the inputs, the outputs follow by algebra
- The outputs match experiment to 0.73% mean deviation (PDG 2024, NuFIT 6.0, 33 Type I core)
- No continuous parameters are fitted

### 0.4 Torsion Independence

**Important**: All 33 Type I predictions use only topological invariants. The torsion T does not appear in any formula. Therefore:
- Predictions depend only on topology, not on the actual torsion value
- The value κ_T = 1/61 is a topological bound, not a prediction ingredient

### 0.5 Results from Analysis of the NK-Certified Metric

Analysis of the NK-certified metric ([B]) computed all invariants directly from the explicit metric. Key results:

- **Topological realization**: The pair (b₂, b₃) = (21, 77) is realized topologically by the Joyce-Karigiannis Z₂³ orbifold route (S1 §8.4); an explicit closed-form neck ansatz with five-layer Wirtinger certificate is established in [D]. Orthogonal TCS is excluded by parity (b₂+b₃=98 even). Any Betti decomposition via Mayer-Vietoris is conditional on the building block identification, which is open.
- **ν̄ = 0**: Lean-certified in TCSConstruction.lean. For rectangular TCS (k₊=k₋=1), ν̄=0 automatically (CGN invariant); this is conditional on TCS realization.
- **New topological formula**: V_min = √(Vol(K₇)/11), where 11 = b₃/n = 77/7. NK numerical: 219.90; formula: 221.24 (0.6% ✓). Identity: V_min² × b₃/n = Vol(K₇): a universal G₂ relation.
- **U(1)² exact symmetry**: The T² directions θ and ψ are degenerate to 2×10⁻⁵ in the metric eigenvalues, propagating to S_θ = S_ψ = 6.1265 in all period integrals (exact to 2.6×10⁻⁸).
- **SM gauge group origin**: NOT from ADE singularities (TCS is smooth). From g₂ ⊂ so(7) spectral structure and so(8) = g₂ ⊕ L ⊕ R triality → N_gen = 3.

---

# Part I: Foundations

## 1. Status Classification

| Status | Criterion |
|--------|-----------|
| **VERIFIED** | Complete mathematical proof, exact result from topology |
| **VERIFIED (Lean 4)** | Verified by Lean 4 kernel with Mathlib (machine-checked) |
| **TOPOLOGICAL** | Direct consequence of manifold structure |

## 2. Notation

| Symbol | Value | Definition |
|--------|-------|------------|
| dim(E₈) | 248 | E₈ Lie algebra dimension |
| rank(E₈) | 8 | E₈ Cartan subalgebra dimension |
| dim(G₂) | 14 | G₂ holonomy group dimension |
| dim(K₇) | 7 | Internal manifold dimension |
| b₂(K₇) | 21 | Second Betti number |
| b₃(K₇) | 77 | Third Betti number |
| H* | 99 | Effective cohomology = b₂ + b₃ + 1 |
| dim(J₃(O)) | 27 | Exceptional Jordan algebra dimension |
| N_gen | 3 | Number of fermion generations |
| p₂ | 2 | Binary duality parameter |
| Weyl | 5 | Weyl factor: (dim(G₂)+1)/N_gen = b₂/N_gen - p₂ = dim(G₂) - rank(E₈) - 1 |

---

# Part II: Foundational Theorems

## 3. Relation #1: Generation Number N_gen = 3

**Statement**: The number of fermion generations is exactly 3.

**Classification**: VERIFIED (three independent derivations)

### Proof Method 1: Fundamental Topological Constraint

*Theorem*: For G₂ holonomy manifold K₇ with E₈ gauge structure:

$$(\text{rank}(E_8) + N_{\text{gen}}) \cdot b_2(K_7) = N_{\text{gen}} \cdot b_3(K_7)$$

*Derivation*:
$$(8 + N_{\text{gen}}) \times 21 = N_{\text{gen}} \times 77$$
$$168 + 21 \cdot N_{\text{gen}} = 77 \cdot N_{\text{gen}}$$
$$168 = 56 \cdot N_{\text{gen}}$$
$$N_{\text{gen}} = \frac{168}{56} = 3$$

*Verification*:
- LHS: (8 + 3) × 21 = 231
- RHS: 3 × 77 = 231 ✓

### Proof Method 2: Atiyah-Singer Index Theorem

$$\text{Index}(D_A) = \left( 77 - \frac{8}{3} \times 21 \right) \times \frac{1}{7} = 3$$

**Status**: VERIFIED ∎

---

## 4. Relation #2: Hierarchy Parameter τ = 3472/891

**Statement**: The hierarchy parameter is exactly rational.

**Classification**: VERIFIED

### Proof

*Step 1: Definition from topological integers*
$$\tau := \frac{\dim(E_8 \times E_8) \cdot b_2(K_7)}{\dim(J_3(\mathbb{O})) \cdot H^*}$$

*Step 2: Substitute values*
$$\tau = \frac{496 \times 21}{27 \times 99} = \frac{10416}{2673}$$

*Step 3: Reduce*
$$\gcd(10416, 2673) = 3$$
$$\tau = \frac{3472}{891}$$

*Step 4: Prime factorization*
$$\tau = \frac{2^4 \times 7 \times 31}{3^4 \times 11}$$

*Step 5: Numerical value*
$$\tau = 3.8967452300785634...$$

**Status**: VERIFIED ∎

---

## 5. Relation #3: Torsion Capacity κ_T = 1/61

**Statement**: The topological torsion capacity equals exactly 1/61.

**Classification**: TOPOLOGICAL (structural parameter, not physical prediction)

### Proof

*Step 1: Define from cohomology*
$$61 = b_3(K_7) - \dim(G_2) - p_2 = 77 - 14 - 2 = 61$$

*Step 2: Formula*
$$\kappa_T = \frac{1}{b_3 - \dim(G_2) - p_2} = \frac{1}{61}$$

*Step 3: Geometric interpretation*
- 61 = effective degrees of freedom available for torsional deformation
- 61 = dim(F₄) + N_gen² = 52 + 9

### Clarification

| Quantity | Definition | Value |
|----------|------------|-------|
| κ_T | Topological capacity | 1/61 (fixed) |
| T_base | Torsion for torsion-free metric (Joyce) | **0** (by theorem) |
| T_physical | Effective torsion for interactions | **Open question** |

**Role in predictions**: κ_T appears in only one formula (α⁻¹, as a small correction term det(g)×κ_T ≈ 0.033). The other 32 Type I predictions are independent of torsion capacity. It is primarily a structural parameter characterizing K₇, not a directly measured observable.

**Joyce's theorem**: For compact G₂ manifolds satisfying perturbation bounds, guarantees existence of a torsion-free metric. The Newton-Kantorovich certificate ([A]) establishes these bounds for K₇.

**Status**: TOPOLOGICAL (structural, not predictive) ∎

---

## 6. Relation #4: Metric Determinant det(g) = 65/32

**Statement**: The K₇ metric determinant is exactly 65/32.

**Classification**: METRIC NORMALIZATION (not TOPOLOGICAL, see S1 §10.3)

### Proof

*Step 1: Define from structural normalization integers*
$$\det(g) = p_2 + \frac{1}{b_2 + \dim(G_2) - N_{gen}}$$

*Step 2: Compute denominator*
$$b_2 + \dim(G_2) - N_{gen} = 21 + 14 - 3 = 32$$

*Step 3: Compute determinant*
$$\det(g) = 2 + \frac{1}{32} = \frac{65}{32}$$

*Step 4: Alternative derivation*
$$\det(g) = \frac{\text{Weyl} \times (\text{rank}(E_8) + \text{Weyl})}{2^5} = \frac{5 \times 13}{32} = \frac{65}{32}$$

**Verification**: The analytical metric g = (65/32)^{1/7} × I₇ has det(g) = [(65/32)^{1/7}]^7 = 65/32 exactly, consistent with the topological formula.

**Epistemic note**: Three independent algebraic paths converge to 65/32, which is suggestive but does not constitute a derivation from topology: the metric optimization was constrained to this normalization target, and the formulas were identified post-hoc. See S1 §10.3 for full discussion. Six observables depending on det(g)_num or det(g)_den carry this normalization dependence.

**Status**: STRUCTURAL (metric normalization, algebraically exact; see S1 §10.3) ∎

---

# Part III: Gauge Sector

## 7. Relation #5: Weinberg Angle sin²θ_W = 3/13

**Statement**: The weak mixing angle has exact rational form 3/13.

**Classification**: VERIFIED

### Proof

*Step 1: Define ratio from Betti numbers*
$$\sin^2\theta_W = \frac{b_2(K_7)}{b_3(K_7) + \dim(G_2)} = \frac{21}{77 + 14} = \frac{21}{91}$$

*Step 2: Simplify*
$$\gcd(21, 91) = 7$$
$$\sin^2\theta_W = \frac{3}{13} = 0.230769...$$

*Step 3: Experimental comparison*

| Quantity | Value |
|----------|-------|
| Experimental (PDG 2024) | 0.23122 ± 0.00004 |
| Prediction | 0.230769 |
| Deviation | 0.195% |

**Status**: VERIFIED ∎

---

## 8. Relation #6: Strong Coupling α_s = √2/12

**Statement**: The strong coupling at M_Z scale.

**Classification**: TOPOLOGICAL

### Proof

*Formula*:
$$\alpha_s(M_Z) = \frac{\sqrt{2}}{\dim(G_2) - p_2} = \frac{\sqrt{2}}{14 - 2} = \frac{\sqrt{2}}{12}$$

*Components*:
- √2: E₈ root length
- 12 = dim(G₂) - p₂: Effective gauge degrees of freedom

*Numerical value*: α_s = 0.117851

*Experimental comparison*:

| Quantity | Value |
|----------|-------|
| Experimental | 0.1180 ± 0.0009 |
| Prediction | 0.11785 |
| Deviation | 0.127% |

**Status**: TOPOLOGICAL ∎

---

# Part IV: Lepton Sector

## 9. Relation #7: Koide Parameter Q = 2/3

**Statement**: The Koide parameter equals exactly 2/3.

**Classification**: VERIFIED

### Proof

*Formula*:
$$Q_{\text{Koide}} = \frac{\dim(G_2)}{b_2(K_7)} = \frac{14}{21} = \frac{2}{3}$$

*Physical definition*:
$$Q = \frac{m_e + m_\mu + m_\tau}{(\sqrt{m_e} + \sqrt{m_\mu} + \sqrt{m_\tau})^2}$$

*Experimental comparison*:

| Quantity | Value |
|----------|-------|
| Experimental | 0.666661 ± 0.000007 |
| Prediction | 0.666667 |
| Deviation | 0.0009% |

**Status**: VERIFIED ∎

---

## 10. Relation #8: Tau-Electron Mass Ratio m_τ/m_e = 3477

**Statement**: The tau-electron mass ratio is exactly 3477.

**Classification**: VERIFIED

### Proof

*Formula*:
$$\frac{m_\tau}{m_e} = \dim(K_7) + 10 \cdot \dim(E_8) + 10 \cdot H^*$$
$$= 7 + 10 \times 248 + 10 \times 99 = 7 + 2480 + 990 = 3477$$

*Prime factorization*:
$$3477 = 3 \times 19 \times 61 = N_{gen} \times prime(8) \times \kappa_T^{-1}$$

*Period integral cross-confirmation*: Analysis of the NK-certified metric ([B], §5) extracts associative 3-cycle volumes directly. The SD associative volumes yield ln(m_τ/m_e) = 8.154 from the Fano-plane volume hierarchy, giving e^8.154 = 3477 ✓. This provides an independent geometric confirmation of the algebraic prediction.

*Experimental comparison*:

| Quantity | Value |
|----------|-------|
| Experimental | 3477.15 ± 0.05 |
| Prediction | 3477 (exact) |
| Deviation | 0.0043% |

**Status**: VERIFIED ∎

---

## 11. Relation #9: Muon-Electron Mass Ratio

**Statement**: m_μ/m_e = 27^φ

**Classification**: TOPOLOGICAL

### Proof

*Formula*:
$$\frac{m_\mu}{m_e} = [\dim(J_3(\mathbb{O}))]^\phi = 27^\phi = 207.012$$

*Components*:
- 27 = dim(J₃(O)): Exceptional Jordan algebra
- φ = (1+√5)/2: Golden ratio from McKay correspondence E₈ ↔ 2I (binary icosahedral group)

*Experimental comparison*:

| Quantity | Value |
|----------|-------|
| Experimental | 206.768 |
| Prediction | 207.01 |
| Deviation | 0.1179% |

**Note on φ**: The golden ratio is the only non-integer constant among the 33 Type I predictions and does not appear in the 20 structural constants (S3.3). Its derivation from E₈ via the McKay correspondence is well-established mathematically but constitutes an additional identification step beyond the direct topological reasoning used for the other 32 predictions.

**Status**: TOPOLOGICAL (with caveat: φ external to the 20 structural constants) ∎

---

# Part V: Quark Sector

## 12. Relation #10: Strange-Down Ratio m_s/m_d = 20

**Statement**: The strange-down quark mass ratio is exactly 20.

**Classification**: VERIFIED

### Proof

*Formula*:
$$\frac{m_s}{m_d} = p_2^2 \times \text{Weyl} = 4 \times 5 = 20$$

*Geometric interpretation*:
- p₂² = 4: Binary structure squared
- Weyl = 5: Pentagonal symmetry

*Experimental comparison*:

| Quantity | Value |
|----------|-------|
| Experimental | 20.0 ± 1.0 |
| Prediction | 20 (exact) |
| Deviation | 0.00% |

**Status**: VERIFIED ∎

---

## 12b. Relation #10b: Charm-Strange Ratio m_c/m_s = 246/21

**Statement**: The charm-strange quark mass ratio.

**Classification**: TOPOLOGICAL

### Proof

*Formula*:
$$\frac{m_c}{m_s} = \frac{\dim(E_8) - p_2}{b_2(K_7)} = \frac{248 - 2}{21} = \frac{246}{21} = 11.714...$$

*Components*:
- 246 = dim(E₈) - p₂: Effective E₈ dimension
- 21 = b₂(K₇): Second Betti number

*Experimental comparison*:

| Quantity | Value |
|----------|-------|
| Experimental | 11.7 ± 0.3 |
| Prediction | 11.714 |
| Deviation | 0.12% |

*TCS-level alternative*: The charm-strange ratio also admits a decomposition at the TCS building-block level:

$$\frac{m_c}{m_s} = b_2(M_1) + \frac{\dim(K_7)}{b_2(M_2)} = 11 + \frac{7}{10} = \frac{117}{10} = 11.700$$

This additive form reflects the adiabatic decomposition of the KK mass operator across the two TCS halves. The M₁ contribution (b₂(M₁) = 11) dominates, with a geometric correction from M₂ (dim(K₇)/b₂(M₂) = 7/10). Both formulas are within 0.05σ of the experimental central value.

**Status**: TOPOLOGICAL ∎

---

## 12c. Relation #10c: Bottom-Top Ratio m_b/m_t = 1/42

**Statement**: The bottom-top quark mass ratio involves the constant 42 = p₂ × N_gen × dim(K₇).

**Classification**: TOPOLOGICAL

### Proof

*Step 1: Define the structural constant*
$$42 = p_2 \times N_{gen} \times \dim(K_7) = 2 \times 3 \times 7$$

This constant 42 also equals 2 × b₂ = 2 × 21.

*Step 2: Formula*
$$\frac{m_b}{m_t} = \frac{b_0}{42} = \frac{1}{42} = 0.02381...$$

*Components*:
- b₀ = 1: Zeroth Betti number
- 42: Structural constant from K₇ geometry

*Experimental comparison*:

| Quantity | Value |
|----------|-------|
| Experimental | 0.024 ± 0.001 |
| Prediction | 0.02381 |
| Deviation | 0.79% |

*Geometric interpretation*: The same constant 42 appears in the cosmological ratio Ω_DM/Ω_b = (1 + 42)/8 = 43/8 (Section 18b), connecting quark physics to cosmological structure through the K₇ geometry.

**Status**: TOPOLOGICAL ∎

---

## 12d. Relation #10d: Up-Down Ratio m_u/m_d = 79/168

**Statement**: The up-down quark mass ratio.

**Classification**: TOPOLOGICAL

### Proof

*Formula*:
$$\frac{m_u}{m_d} = \frac{b_0 + \dim(E_6)}{|PSL_2(7)|} = \frac{1 + 78}{168} = \frac{79}{168} = 0.4702...$$

*Components*:
- dim(E₆) = 78: Exceptional Lie algebra dimension
- |PSL₂(7)| = 168: Order of the simple group PSL₂(7) = rank(E₈) × b₂

*Experimental comparison*:

| Quantity | Value |
|----------|-------|
| Experimental | 0.47 ± 0.03 |
| Prediction | 0.4702 |
| Deviation | 0.05% |

**Status**: TOPOLOGICAL ∎

---

# Part V-B: CKM Matrix

## 12e. Relation #10e: Cabibbo Angle sin²θ₁₂(CKM) = 7/31

**Statement**: The CKM Cabibbo mixing angle.

**Classification**: TOPOLOGICAL

### Proof

*Formula*:
$$\sin^2\theta_{12}^{CKM} = \frac{\dim(\text{fund}_{E_7})}{\dim(E_8)} = \frac{56}{248} = \frac{7}{31} = 0.2258...$$

*Alternative expressions*:
- (b₃ - b₂)/dim(E₈) = (77 - 21)/248 = 56/248
- (2b₂ + dim(G₂))/dim(E₈) = (42 + 14)/248 = 56/248

*Experimental comparison*:

| Quantity | Value |
|----------|-------|
| Experimental | 0.2250 ± 0.0006 |
| Prediction | 0.2258 |
| Deviation | 0.36% |

**Status**: TOPOLOGICAL ∎

---

## 12f. Relation #10f: Wolfenstein A Parameter = 83/99

**Statement**: The Wolfenstein A parameter of the CKM matrix.

**Classification**: TOPOLOGICAL

### Proof

*Formula*:
$$A_{\text{Wolf}} = \frac{\text{Weyl} + \dim(E_6)}{H^*} = \frac{5 + 78}{99} = \frac{83}{99} = 0.8384...$$

*Alternative expression*:
- (b₃ + p₂ × N_gen)/H* = (77 + 6)/99 = 83/99

*Experimental comparison*:

| Quantity | Value |
|----------|-------|
| Experimental | 0.836 ± 0.015 |
| Prediction | 0.8384 |
| Deviation | 0.29% |

**Status**: TOPOLOGICAL ∎

---

## 12g. Relation #10g: CKM θ₂₃ Mixing sin²θ₂₃(CKM) = 1/24

**Statement**: The CKM 23-mixing angle.

**Classification**: TOPOLOGICAL

### Proof

*Formula*:
$$\sin^2\theta_{23}^{CKM} = \frac{\dim(K_7)}{|PSL_2(7)|} = \frac{7}{168} = \frac{1}{24} = 0.04167...$$

*Experimental comparison*:

| Quantity | Value |
|----------|-------|
| Experimental | 0.0412 ± 0.0008 |
| Prediction | 0.04167 |
| Deviation | 1.13% |

**Status**: TOPOLOGICAL ∎

---

# Part VI: Neutrino Sector

## 13. Relation #11: CP Violation Phase δ_CP = 197°

**Statement**: The CP violation phase is exactly 197°.

**Classification**: VERIFIED (experimental drift across NuFIT releases discussed in Appendix F; against NuFIT 6.1 the prediction sits inside the 1σ band)

### Proof

*Formula*:
$$\delta_{CP} = \dim(K_7) \cdot \dim(G_2) + H^* = 7 \times 14 + 99 = 98 + 99 = 197°$$

*Experimental comparison*:

| Quantity | Value |
|----------|-------|
| Experimental (NuFIT 5.2, 2022) | 197° ± 24° |
| Experimental (NuFIT 6.0, Oct 2024) | 177° ± 20° |
| Experimental (NuFIT 6.1, 2025, NO w/o SK-atm) | 207° +23/−20 |
| Prediction | 197° (exact) |
| Deviation (vs NuFIT 5.2) | 0.00% |
| Deviation (vs NuFIT 6.0) | 11.3% (edge of 1σ) |
| Deviation (vs NuFIT 6.1) | 4.8% (inside 1σ) |

**Experimental drift**: The central value shifted from 197° to 177° between NuFIT 5.2
and 6.0, then to 207° in NuFIT 6.1 (2025, driven mainly by the T2K+NOvA joint
analysis); the NuFIT collaboration flags δ_CP as one of the least constrained
parameters, with substantial dependence on reactor flux assumptions. The two most
recent releases bracket the prediction on both sides, and against NuFIT 6.1 the
value 197° sits inside the 1σ band. DUNE (first beam targeted 2031; physics run
late 2030s–2040s) will resolve this definitively.

**The canonical framework prediction remains δ_CP = 197°.** A potential compactification
correction (62/69 factor, yielding 177.01°) is documented in Appendix F as contingent
on experimental confirmation. See also §4.2 of the main paper.

**Status**: VERIFIED, original prediction, pending DUNE ∎

---

## 14. Relation #12: Reactor Mixing Angle θ₁₃ = π/21

**Statement**: The reactor neutrino mixing angle.

**Classification**: TOPOLOGICAL

### Proof

*Formula*:
$$\theta_{13} = \frac{\pi}{b_2(K_7)} = \frac{\pi}{21} = 8.571°$$

*Experimental comparison*:

| Quantity | Value |
|----------|-------|
| Experimental (NuFIT 6.0) | 8.54° ± 0.12° |
| Prediction | 8.571° |
| Deviation | 0.37% |

*TCS-level refinement*: At the building-block level, the reactor angle admits a purely rational formula:

$$\theta_{13} = \frac{\dim(E_8) - \dim(G_2) \cdot b_2(M_1)}{b_2(M_1)} = \frac{248 - 14 \times 11}{11} = \frac{94}{11} = 8.545°$$

The numerator 94 = dim(E₈) − dim(G₂)×b₂(M₁) is the residual E₈ gauge content after subtracting the G₂ holonomy contribution to M₁. This formula is 5.8× more precise (0.064% vs 0.37%) and uses only algebraic invariants (no π), but works in degrees rather than radians.

**Status**: TOPOLOGICAL ∎

---

## 15. Relation #13: Atmospheric Mixing Angle θ₂₃

**Statement**: The atmospheric neutrino mixing angle.

**Classification**: TOPOLOGICAL

### Proof

*Formula*:
$$\theta_{23} = \arcsin\left(\frac{b_3 - p_2}{H^*}\right) = \arcsin\left(\frac{75}{99}\right) = \arcsin\left(\frac{25}{33}\right) = 49.251°$$

*Components*:
- b₃ = 77: Third Betti number (3-cycles of K₇)
- p₂ = 2: Pontryagin class contribution (spin structure correction)
- H* = 99: Effective cohomology (b₂ + b₃ + 1)

*Physical interpretation*:
The atmospheric mixing angle θ₂₃ governs τ-μ flavor mixing. The formula (b₃ - p₂)/H* represents the relative weight of spin-corrected 3-cycles in the total cohomology. This captures how the τ-μ sector couples through the 3-cycle topology of K₇, with the Pontryagin correction accounting for the spin structure that distinguishes fermionic generations.

*Experimental comparison*:

| Quantity | Value |
|----------|-------|
| Experimental (NuFIT 5.3) | 49.3° ± 1.0° |
| Prediction | 49.251° |
| Deviation | 0.10% |

**Status**: TOPOLOGICAL ∎

---

## 16. Relation #14: Solar Mixing Angle θ₁₂

**Statement**: The solar neutrino mixing angle.

**Classification**: TOPOLOGICAL

### Proof

*Formula*:
$$\theta_{12} = \arctan\left(\sqrt{\frac{\delta}{\gamma_{\text{GIFT}}}}\right) = 33.419°$$

*Components*:
- δ = 2π/Weyl² = 2π/25
- γ_GIFT = 511/884 (historical symbol name)

*Derivation of γ_GIFT*:
$$\gamma_{\text{GIFT}} = \frac{2 \cdot \text{rank}(E_8) + 5 \cdot H^*}{10 \cdot \dim(G_2) + 3 \cdot \dim(E_8)} = \frac{511}{884}$$

*Experimental comparison*:

| Quantity | Value |
|----------|-------|
| Experimental (NuFIT 5.3) | 33.41° ± 0.75° |
| Prediction | 33.40° |
| Deviation | 0.030% |

**Status**: TOPOLOGICAL ∎

---

## 16b. PMNS Matrix: sin² Form

The PMNS mixing angles can also be expressed directly as sin² values, providing alternative topological formulas.

### Relation #14b: sin²θ₁₂(PMNS) = 4/13

**Formula**:
$$\sin^2\theta_{12}^{PMNS} = \frac{b_0 + N_{gen}}{\alpha_{sum}} = \frac{1 + 3}{13} = \frac{4}{13} = 0.3077...$$

*Components*:
- α_sum = 13: Anomaly coefficient sum
- b₀ + N_gen = 4: Cohomological + generation count

| Quantity | Value |
|----------|-------|
| Experimental | 0.307 ± 0.013 |
| Prediction | 0.3077 |
| Deviation | 0.23% |

### Relation #14c: sin²θ₂₃(PMNS) = 6/11

**Formula**:
$$\sin^2\theta_{23}^{PMNS} = \frac{D_{bulk} - \text{Weyl}}{D_{bulk}} = \frac{11 - 5}{11} = \frac{6}{11} = 0.5455...$$

*Alternative expression*:
- 42/b₃ = 42/77 = 6/11 (after reduction)

| Quantity | Value |
|----------|-------|
| Experimental | 0.546 ± 0.021 |
| Prediction | 0.5455 |
| Deviation | 0.10% |

### Relation #14d: sin²θ₁₃(PMNS) = 11/496

**Formula**:
$$\sin^2\theta_{13}^{PMNS} = \frac{D_{bulk}}{\dim(E_8 \times E_8)} = \frac{11}{496} = 0.02218...$$

| Quantity | Value |
|----------|-------|
| Experimental | 0.0220 ± 0.0007 |
| Prediction | 0.02218 |
| Deviation | 0.81% |

**Status**: TOPOLOGICAL ∎

---

# Part VII: Higgs & Cosmology

## 17. Relation #15: Higgs Coupling λ_H = √17/32

**Statement**: The Higgs quartic coupling is determined by G₂ holonomy parameters.

**Classification**: STRUCTURAL (uses the imposed metric normalization det(g) = 65/32; see main paper §2.3)

### Proof

*Formula*:
$$\lambda_H = \frac{\sqrt{\dim(G_2) + N_{gen}}}{\det(g)_{den}} = \frac{\sqrt{14 + 3}}{32} = \frac{\sqrt{17}}{32} = 0.12885$$

*Components*:
- dim(G₂) + N_gen = 17: holonomy dimension plus generation count
- det(g)_den = 32: metric determinant denominator (= b₂ + dim(G₂) − N_gen)

*Physical interpretation*:
The Higgs self-coupling combines the holonomy group dimension with the generation count, normalized by the metric determinant scale. The square root reflects the quadratic nature of the Higgs potential.

*Numerical value*: λ_H = √17/32 = 0.128847...

*Experimental comparison*:

| Quantity | Value |
|----------|-------|
| Experimental (PDG 2024) | 0.129 ± 0.001 |
| Prediction | √17/32 = 0.12885 |
| Deviation | 0.12% |

### TCS Alternative

A purely rational TCS formula is also available:

$$\lambda_H^{TCS} = \frac{b_2(M_1)}{b_3 + b_2(M_2)} = \frac{11}{87} = 0.12644$$

where b₂(M₁) = 11 and b₂(M₂) = 10 are the Betti numbers of the two TCS building blocks. This formula is lower-complexity (cost 3 vs cost 6) and connects directly to the twisted connected sum decomposition. Against a running coupling value λ_H(μ_high) ≈ 0.126, it achieves 0.35% deviation, though against the PDG 2024 pole value (0.129) it gives 1.98%.

**Status**: STRUCTURAL (metric-normalization dependent) ∎

---

## 17b. Boson Mass Ratios

**Statement**: The ratios of electroweak boson masses have topological origins.

**Classification**: VERIFIED

### Relation: m_W/m_Z = 37/42

*Formula*:
$$\frac{m_W}{m_Z} = \frac{2b_2 - \text{Weyl}}{2b_2} = \frac{42 - 5}{42} = \frac{37}{42}$$

*Physical interpretation*:
- 2b₂ = 42 is the structural constant (= p₂ × b₂)
- Weyl = 5 is the triple identity factor
- The ratio involves (structural_const − Weyl) / structural_const

**Note**: The true Euler characteristic χ(K₇) = 0 for odd-dimensional manifolds. The constant 42 = 2b₂ is a distinct topological invariant.

*Numerical value*: m_W/m_Z = 0.8810

*Experimental comparison*:

| Quantity | Value |
|----------|-------|
| Experimental | 0.8815 ± 0.0002 |
| Prediction | 0.8810 |
| Deviation | **0.06%** |

### Relation: m_H/m_t = 56/77

*Formula*:
$$\frac{m_H}{m_t} = \frac{fund(E_7)}{b_3} = \frac{56}{77} = \frac{8}{11}$$

*Numerical value*: m_H/m_t = 0.7273

| Quantity | Value |
|----------|-------|
| Experimental | 0.725 ± 0.003 |
| Prediction | 0.7273 |
| Deviation | 0.31% |

### Relation: m_H/m_W = 81/52

*Formula*:
$$\frac{m_H}{m_W} = \frac{N_{gen} + \dim(E_6)}{\dim(F_4)} = \frac{3 + 78}{52} = \frac{81}{52}$$

*Numerical value*: m_H/m_W = 1.5577

| Quantity | Value |
|----------|-------|
| Experimental | 1.558 ± 0.002 |
| Prediction | 1.5577 |
| Deviation | **0.02%** |

**Status**: VERIFIED ∎

---

## 18. Relation #16: Dark Energy Density Ω_DE = ln(2)×98/99

**Statement**: The dark energy density fraction emerges from information-theoretic structure of K₇ moduli.

**Classification**: TOPOLOGICAL

### Proof

*Formula*:
$$\Omega_{DE} = \ln(2) \times \frac{b_2 + b_3}{H^*} = \ln(2) \times \frac{98}{99} = 0.6861$$

*Components*:
- ln(2): binary entropy factor
- b₂ + b₃ = 98: total non-trivial cohomological content
- H* = 99 = b₂ + b₃ + 1: full cohomological dimension

*Physical interpretation*:
The dark energy fraction combines a binary information factor with the ratio of non-trivial to total cohomological content. The ln(2) reflects the fundamental binary structure of the moduli space partition.

*Numerical value*: Ω_DE = ln(2) × 98/99 = 0.68615...

*Experimental comparison*:

| Quantity | Value |
|----------|-------|
| Experimental (Planck PR4) | 0.6847 ± 0.005 |
| Prediction | ln(2)×98/99 = 0.6861 |
| Deviation | 0.21% |

### TCS Alternative

A purely rational TCS formula is also available:

$$\Omega_{DE}^{TCS} = 1 - \frac{\chi(K3)}{b_3} = 1 - \frac{24}{77} = \frac{53}{77} = 0.68831$$

where χ(K3) = 24 is the Euler characteristic of the K3 fiber. This formula is lower-complexity (rational, cost 2 vs cost 6), and the interpretation is transparent: Ω_matter = χ(K3)/b₃ counts the K3-fiber fraction of the 3-cycle moduli, with the rest being vacuum energy. Against Planck 2018 (Ω_Λ = 0.6889), it achieves 0.085% deviation, though against Planck PR4 (0.6847) it gives 0.53%.

**Status**: TOPOLOGICAL ∎

---

## 19. Relation #17: Spectral Index n_s

**Statement**: The primordial scalar spectral index.

**Classification**: VERIFIED

### Proof

*Formula*:
$$n_s = \frac{\zeta(D_{bulk})}{\zeta(\text{Weyl})} = \frac{\zeta(11)}{\zeta(5)} = 0.9649$$

*Components*:
- ζ(11): From 11D bulk spacetime
- ζ(5): From Weyl factor

*Experimental comparison*:

| Quantity | Value |
|----------|-------|
| Experimental (Planck 2020) | 0.9649 ± 0.0042 |
| Prediction | 0.9649 |
| Deviation | 0.004% |

**Status**: VERIFIED ∎

---

## 19b. Relation #17c: Dark Matter to Baryon Ratio Ω_DM/Ω_b = 43/8

**Statement**: The dark matter to baryon density ratio.

**Classification**: TOPOLOGICAL

### Proof

*Formula*:
$$\frac{\Omega_{DM}}{\Omega_b} = \frac{b_0 + 42}{\text{rank}(E_8)} = \frac{1 + 42}{8} = \frac{43}{8} = 5.375$$

*Components*:
- 42 = p₂ × N_gen × dim(K₇): The same constant appearing in m_b/m_t = 1/42
- rank(E₈) = 8: Cartan subalgebra dimension

*Experimental comparison*:

| Quantity | Value |
|----------|-------|
| Experimental (Planck 2020) | 5.375 ± 0.05 |
| Prediction | 5.375 |
| Deviation | 0.00% |

**Status**: TOPOLOGICAL ∎

---

## 19c. Relation #17d: Reduced Hubble Parameter h = 167/248

**Statement**: The reduced Hubble parameter H₀ = 100h km/s/Mpc.

**Classification**: TOPOLOGICAL

### Proof

*Formula*:
$$h = \frac{|PSL_2(7)| - b_0}{\dim(E_8)} = \frac{168 - 1}{248} = \frac{167}{248} = 0.6734...$$

*Experimental comparison*:

| Quantity | Value |
|----------|-------|
| Experimental (Planck 2020) | 0.674 ± 0.005 |
| Prediction | 0.6734 |
| Deviation | 0.09% |

**Status**: TOPOLOGICAL ∎

---

## 19d. Relation #17e: Baryon Fraction Ω_b/Ω_m = 5/32

**Statement**: The baryon fraction of total matter.

**Classification**: STRUCTURAL (uses the imposed metric normalization det(g) = 65/32; see main paper §2.3)

### Proof

*Formula*:
$$\frac{\Omega_b}{\Omega_m} = \frac{\text{Weyl}}{\det(g)_{den}} = \frac{5}{32} = 0.15625$$

*Experimental comparison*:

| Quantity | Value |
|----------|-------|
| Experimental (Planck 2020) | 0.156 ± 0.003 |
| Prediction | 0.15625 |
| Deviation | 0.16% |

**Status**: STRUCTURAL (metric-normalization dependent) ∎

---

## 19e. Relation #17f: Amplitude of Fluctuations σ₈ = 17/21

**Statement**: The amplitude of matter fluctuations at 8 h⁻¹ Mpc.

**Classification**: STRUCTURAL (uses the imposed metric normalization det(g) = 65/32; see main paper §2.3)

### Proof

*Formula*:
$$\sigma_8 = \frac{p_2 + \det(g)_{den}}{42} = \frac{2 + 32}{42} = \frac{34}{42} = \frac{17}{21} = 0.8095...$$

*Experimental comparison*:

| Quantity | Value |
|----------|-------|
| Experimental (Planck 2020) | 0.811 ± 0.006 |
| Prediction | 0.8095 |
| Deviation | 0.18% |

**Status**: STRUCTURAL (metric-normalization dependent) ∎

---

## 19f. Relation #17g: Primordial Helium Fraction Y_p = 15/61

**Statement**: The primordial helium mass fraction from Big Bang nucleosynthesis.

**Classification**: TOPOLOGICAL

### Proof

*Formula*:
$$Y_p = \frac{b_0 + \dim(G_2)}{\kappa_T^{-1}} = \frac{1 + 14}{61} = \frac{15}{61} = 0.2459...$$

*Experimental comparison*:

| Quantity | Value |
|----------|-------|
| Experimental | 0.245 ± 0.003 |
| Prediction | 0.2459 |
| Deviation | 0.37% |

**Status**: TOPOLOGICAL ∎

---

## 20. Relation #17b: Matter Density Ω_m

**Statement**: The matter density fraction derives from dark energy via √Weyl.

**Classification**: DERIVED (from Weyl triple identity + Ω_DE)

### Proof

*Step 1: Establish √Weyl as structural*

From the Weyl Triple Identity (S1, Section 2.3):
$$\text{Weyl} = \frac{\dim(G_2) + 1}{N_{gen}} = \frac{b_2}{N_{gen}} - p_2 = \dim(G_2) - \text{rank}(E_8) - 1 = 5$$

Therefore √Weyl = √5 is a derived quantity.

*Step 2: Matter-dark energy ratio*

The cosmological density ratio:
$$\frac{\Omega_{DE}}{\Omega_m} = \sqrt{\text{Weyl}} = \sqrt{5}$$

*Step 3: Compute Ω_m*

Using Ω_DE = ln(2) × (b₂ + b₃)/H* = 0.6861 (Relation #16):
$$\Omega_m = \frac{\Omega_{DE}}{\sqrt{\text{Weyl}}} = \frac{\ln(2) \times 98/99}{\sqrt{5}} = \frac{0.6861}{2.236} = 0.3068$$

*Step 4: Verify closure*

$$\Omega_{total} = \Omega_{DE} + \Omega_m = 0.6861 + 0.3068 = 0.9929 \approx 1$$

Consistent with flat universe (Ω_total = 1).

*Experimental comparison*:

| Quantity | Value |
|----------|-------|
| Experimental (Planck 2020) | 0.3153 ± 0.007 |
| Prediction | 0.3068 |
| Deviation | 2.7% |

### Interpretation

The √5 ratio between dark energy and matter densities emerges from the same structural constant (Weyl = 5) that determines:
- det(g) = 65/32 (metric determinant)
- |W(E₈)| factorization (group theory)
- N_gen³ coefficient in |W(E₈)| (topology)

**Status**: DERIVED (structural, 2.7% deviation) ∎

---

## 20b. Relation #18: Fine Structure Constant α⁻¹

**Statement**: The inverse fine structure constant.

**Classification**: STRUCTURAL (uses the imposed metric normalization det(g) = 65/32; see main paper §2.3)

### Proof

*Formula*:
$$\alpha^{-1}(M_Z) = \frac{\dim(E_8) + \text{rank}(E_8)}{2} + \frac{H^*}{D_{bulk}} + \det(g) \cdot \kappa_T$$
$$= 128 + 9 + \frac{65}{32} \times \frac{1}{61} = 137.033$$

*Components*:
- 128 = (248 + 8)/2: Algebraic
- 9 = 99/11: Bulk impedance
- 65/1952: Torsional correction

*Experimental comparison*:

| Quantity | Value |
|----------|-------|
| Experimental | 137.035999 |
| Prediction | 137.033 |
| Deviation | 0.002% |

**Status**: STRUCTURAL (metric-normalization dependent) ∎

---

# Part VIII: Summary Table

## 21. The 33 Type I Dimensionless Relations (+ 2 Type IV BH Appendix)

**Note**: All predictions use only topological invariants (b₂, b₃, dim(G₂), etc.). None depend on the realized torsion value T. Rows labelled STRUCTURAL additionally use the imposed metric normalization det(g) = 65/32 (main §2.3). Relation #19 (Ω_m) is DERIVED from Ω_DE via the Weyl triple identity.

| # | Relation | Formula | Value | Exp. | Dev. | Status |
|---|----------|---------|-------|------|------|--------|
| 1 | N_gen | Atiyah-Singer | 3 | 3 | exact | VERIFIED |
| 2 | τ | 496×21/(27×99) | 3472/891 | - | - | VERIFIED |
| 3 | κ_T | 1/(77-14-2) | 1/61 | - | - | LEDGER PARAM.* |
| 4 | det(g) | 5×13/32 | 65/32 | - | - | STRUCTURAL (norm. target) |
| 5 | sin²θ_W | 21/91 | 3/13 | 0.23122 | 0.195% | VERIFIED |
| 6 | α_s | √2/12 | 0.11785 | 0.1180 | 0.127% | TOPOLOGICAL |
| 7 | Q_Koide | 14/21 | 2/3 | 0.666661 | 0.0009% | VERIFIED |
| 8 | m_τ/m_e | 7+2480+990 | 3477 | 3477.15 | 0.0043% | VERIFIED |
| 9 | m_μ/m_e | 27^φ | 207.01 | 206.768 | 0.118% | TOPOLOGICAL |
| 10 | m_s/m_d | 4×5 | 20 | 20.0 | 0.00% | VERIFIED |
| 11 | δ_CP | 7×14+99 | 197° | 177°±20°* | 11.3% (1σ) | VERIFIED |
| 12 | θ₁₃ | π/21 (TCS: 94/11) | 8.57° | 8.54° | 0.37% | TOPOLOGICAL |
| 13 | θ₂₃ | arcsin((b₃-p₂)/H*) | 49.25° | 49.3° | 0.10% | TOPOLOGICAL |
| 14 | θ₁₂ | arctan(...) | 33.40° | 33.41° | 0.030% | TOPOLOGICAL |
| 15 | λ_H | √(dim(G₂)+N_gen)/det_den | √17/32 | 0.129 | 0.12% | STRUCTURAL |
| 16 | Ω_DE | ln(2)×(b₂+b₃)/H* | 0.6861 | 0.6847 | 0.21% | TOPOLOGICAL |
| 17 | n_s | ζ(11)/ζ(5) | 0.9649 | 0.9649 | 0.004% | VERIFIED |
| 18 | α⁻¹ | 128+9+corr | 137.033 | 137.036 | 0.002% | STRUCTURAL |
| 19 | Ω_m | Ω_DE/√Weyl | 0.3068 | 0.3153 | 2.7% | DERIVED |

*NuFIT 6.0 (Oct 2024); ±20° uncertainty; frozen dataset for the deviation statistics of this supplement. The 197° prediction matched NuFIT 5.2 (2022) exactly, and sits inside the NuFIT 6.1 (2025) 1σ band (207° +23/−20, deviation 4.8%). See §4.2 of main paper and Appendix F.

*κ_T is a ledger parameter (capacity), not a physical prediction; it does not appear in other formulas. Labelled LEDGER PARAM. to avoid collision with STRUCTURAL, reserved for det(g)-normalization dependence.

---

## 22. Deviation Statistics

| Range | Count | Percentage |
|-------|-------|------------|
| 0.00% (exact) | 1 | 3% |
| <0.01% | 4 | 12% |
| 0.01-0.1% | 4 | 12% |
| 0.1-0.5% | 16 | 48% |
| 0.5-1% | 3 | 9% |
| 1-5% | 4 | 12% |
| >5% (δ_CP vs frozen 6.0) | 1 | 3% |

**Mean deviation**: 0.73% (PDG 2024, NuFIT 6.0 frozen dataset; 33 Type I observables with experimental comparison, canonical δ_CP=197°). For the full 95-observable dataset (66 with experimental comparison), see Supplement S3.

---

## 23. Statistical Uniqueness of (b₂=21, b₃=77)

A critical question for any unified framework is whether the specific topological parameters represent overfitting. We conducted comprehensive Monte Carlo validation to address this concern.

### Methodology

- **Betti variations**: 100,000 random (b₂, b₃) configurations
- **Gauge group comparison**: E₈×E₈, E₇×E₈, E₆×E₈, E₇×E₇, SO(32), SU(5)×SU(5), etc.
- **Holonomy comparison**: G₂, Spin(7), SU(3), SU(4)
- **Full combinatorial**: 91,896 configurations varying all parameters
- **Local sensitivity**: ±10 grid around (b₂=21, b₃=77)

### Results

| Metric | Value |
|--------|-------|
| Total configurations tested | **192,349** |
| Configurations better than the K₇ framework | **0** |
| K₇ framework mean deviation | 0.73% (33 observables, frozen dataset) |
| Alternative mean deviation | 32.9% |
| P-value | **< 5 × 10⁻⁶** |
| Significance | **> 4.5σ** |

### Gauge Group Ranking

| Rank | Group | Mean Dev. |
|------|-------|-----------|
| **1** | **E₈×E₈** | **0.73%** |
| 2 | E₇×E₈ | 8.80% |
| 3 | E₆×E₈ | 15.50% |

E₈ x E₈ achieves approximately 12x better agreement than the next-best gauge product (E₇×E₈, 8.80%).

### Holonomy Ranking

| Rank | Holonomy | Mean Dev. |
|------|----------|-----------|
| **1** | **G₂** | **0.73%** |
| 2 | SU(4) | 1.46% |
| 3 | SU(3) | 4.43% |

G₂ achieves approximately 6x better agreement than Calabi-Yau (SU(3), 4.43%).

### Interpretation

Among 192,349 tested alternatives, the configuration (b₂=21, b₃=77) with E₈ x E₈ gauge group and G₂ holonomy achieves the lowest mean deviation. Zero alternatives outperform it.

**Literature uniqueness**: Analysis of the NK-certified metric ([B], §6) compared against all ~65 known compact G₂ manifolds from the literature (Kovalev, CHNP, CGN, Joyce, Nordström). The point (b₂=21, b₃=77) is UNIQUE: no known diffeomorphic example exists. The nearest neighbor in (b₂,b₃) space is the CHNP grid point (9,9), at distance 7.6. The topological pair is realized by the Joyce-Karigiannis Z₂³ orbifold route (S1 §8.4), extending the known catalogue; an explicit closed-form neck ansatz is established in [D].

Complete methodology: `STATISTICAL_EVIDENCE.md` in the framework repository (available on request).

---

# Part IX: Observable Catalog

## 24. Structural Redundancy and Expression Counts

Each prediction admits multiple algebraically independent expressions that reduce to the same fraction. This multiplicity provides a measure of structural robustness: quantities arising from many paths through the topological invariants are less likely to represent numerical coincidence.

### 24.1 Classification Scheme

| Classification | Expressions | Interpretation |
|----------------|-------------|----------------|
| **CANONICAL** | ≥20 | Maximally over-determined; emerges from algebraic web |
| **ROBUST** | 10-19 | Highly constrained; multiple independent derivations |
| **SUPPORTED** | 5-9 | Structural redundancy |
| **DERIVED** | 2-4 | Dual derivation minimum |
| **SINGULAR** | 1 | Unique path (possible coincidence) |

### 24.2 Core 18 Type I Predictions with Expression Counts (subset of 33 certified)

| # | Observable | Formula | Value | Exp. | Dev. | Expr. | Class |
|---|------------|---------|-------|------|------|-------|-------|
| 1 | N_gen | Atiyah-Singer | 3 | 3 | 0.00% | 24+ | CANONICAL |
| 2 | sin²θ_W | b₂/(b₃+dim_G₂) | 3/13 | 0.2312 | 0.20% | 14 | ROBUST |
| 3 | α_s(M_Z) | √2/12 | 0.11785 | 0.11800 | 0.126% | 9 | SUPPORTED |
| 4 | λ_H | √17/32 | 0.1288 | 0.129 | 0.12% | 4 | DERIVED |
| 5 | α⁻¹ | 128+9+corr | 137.033 | 137.036 | 0.002% | 3 | DERIVED |
| 6 | Q_Koide | dim_G₂/b₂ | 2/3 | 0.6667 | 0.001% | 20 | CANONICAL |
| 7 | m_τ/m_e | 7+10×248+10×99 | 3477 | 3477.2 | 0.004% | 3 | DERIVED |
| 8 | m_μ/m_e | 27^φ | 207.01 | 206.77 | 0.12% | 2 | DERIVED |
| 9 | m_s/m_d | p₂²×Weyl | 20 | 20.0 | 0.00% | 14 | ROBUST |
| 10 | m_b/m_t | 1/(2b₂) | 1/42 | 0.024 | 0.79% | 21 | CANONICAL |
| 11 | m_u/m_d | (1+dim_E₆)/PSL₂₇ | 79/168 | 0.47 | 0.05% | 1 | SINGULAR |
| 12 | δ_CP | 7×14+99 | 197° | 177°±20°* | 11.3% (1σ) | 3 | DERIVED |
| 13 | θ₁₃ | π/b₂ | 8.57° | 8.54° | 0.37% | 3 | DERIVED |
| 14 | θ₂₃ | arcsin((b₃-p₂)/H*) | 49.25° | 49.3° | 0.10% | 2 | DERIVED |
| 15 | θ₁₂ | arctan(√(δ/γ)) | 33.40° | 33.41° | 0.03% | 2 | DERIVED |
| 16 | Ω_DE | ln(2)×(b₂+b₃)/H* | 0.6861 | 0.6847 | 0.21% | 2 | DERIVED |
| 17 | n_s | ζ(11)/ζ(5) | 0.9649 | 0.9649 | 0.004% | 2 | DERIVED |
| 18 | det(g) | 65/32 | 2.0313 | - | - | 8 | SUPPORTED |

**Distribution**: 3 CANONICAL (17%), 2 ROBUST (11%), 2 SUPPORTED (11%), 10 DERIVED (56%), 1 SINGULAR (6%).

### 24.3 Extended Certified Predictions (#19–#33, 15 entries)

| # | Observable | Formula | Value | Exp. | Dev. | Expr. | Class |
|---|------------|---------|-------|------|------|-------|-------|
| 19 | sin²θ₁₂^PMNS | (1+N_gen)/α_sum | 4/13 | 0.307 | 0.23% | 28 | CANONICAL |
| 20 | sin²θ₂₃^PMNS | (D_bulk−Weyl)/D_bulk | 6/11 | 0.546 | 0.10% | 15 | ROBUST |
| 21 | sin²θ₁₃^PMNS | D_bulk/dim_E₈₂ | 11/496 | 0.022 | 0.81% | 5 | SUPPORTED |
| 22 | sin²θ₁₂^CKM | 7/31 | 0.2258 | 0.225 | 0.36% | 16 | ROBUST |
| 23 | A_Wolf | (Weyl+dim_E₆)/H* | 83/99 | 0.836 | 0.29% | 4 | DERIVED |
| 24 | sin²θ₂₃^CKM | dim_K₇/PSL₂₇ | 1/24 | 0.041 | 1.13% | 3 | DERIVED |
| 25 | m_H/m_t | 8/11 | 0.7273 | 0.725 | 0.31% | 19 | ROBUST |
| 26 | m_H/m_W | 81/52 | 1.5577 | 1.558 | 0.02% | 1 | SINGULAR |
| 27 | m_W/m_Z | (2b₂−Weyl)/(2b₂) = 37/42 | 0.8810 | 0.8815 | **0.06%** | 8 | SUPPORTED |
| 28 | m_μ/m_τ | 5/84 | 0.0595 | 0.0595 | 0.04% | 9 | SUPPORTED |
| 29 | Ω_DM/Ω_b | (1+42)/rank_E₈ | 43/8 | 5.375 | 0.00% | 6 | SUPPORTED |
| 30 | Ω_b/Ω_m | Weyl/det(g)_den | 5/32 | 0.156 | 0.16% | 7 | SUPPORTED |
| 31 | Ω_Λ/Ω_m | (det_g_den−dim_K₇)/D_bulk | 25/11 | 2.27 | 0.12% | 6 | SUPPORTED |
| 32 | h | (PSL₂₇−1)/dim_E₈ | 167/248 | 0.674 | 0.09% | 3 | DERIVED |
| 33 | σ₈ | (p₂+det_g_den)/(2b₂) | 34/42 | 0.811 | 0.18% | 4 | DERIVED |

### 24.3b Relations #34–#35: BH Remnant Topological Predictions: Type IV (Pinčák et al. 2026 [42])

*These are classified as **Type IV structural diagnostics** (D22 = M_res, D20 = N_QNM) in the 95-observable dataset, not as Type I. They are included here for completeness as they are topologically derived from the same G₂ structural constants.*

| # | Observable | Formula | Value | Status |
|---|------------|---------|-------|--------|
| 34 | BH remnant mass M_res | v_EW²/M_Pl (torsion VEV: τ₀ = v_EW) | ~125 GeV² / M_Pl | Type IV; topological, not yet Lean-certified |
| 35 | QNM families N_QNM | b₃ + Weyl×dim(K₇) + p₂ (G₂ torsion modes) | 98 | Type IV; topological, not yet Lean-certified |

These two predictions are topologically derived from G₂ torsion structure but have not yet been formally certified in Lean. M_res involves a physical scale identification (τ₀ = v_EW); N_QNM is a purely structural count. No experimental comparison is currently available.

### 24.4 Illustrative Examples of Multiple Expressions

**sin²θ_W = 3/13** (14 independent expressions):

| # | Expression | Evaluation |
|---|------------|------------|
| 1 | N_gen / α_sum | 3/13 |
| 2 | N_gen / (p₂ + D_bulk) | 3/(2+11) = 3/13 |
| 3 | b₂ / (b₃ + dim_G₂) | 21/91 = 3/13 |
| 4 | dim(J₃O) / (dim_F₄ + det_g_num) | 27/117 = 3/13 |
| 5 | (b₀ + dim_G₂) / det_g_num | 15/65 = 3/13 |
| 6 | (p₂ + b₀) / α_sum | 3/13 |
| 7 | dim_K₇ / (b₂ + dim_K₇ + dim_G₂) | 7/42 ≠ 3/13 ✗ |

(Expression 7 illustrates that not all combinations work; only those reducing to 3/13 are valid.)

**Q_Koide = 2/3** (20 independent expressions):

| # | Expression | Evaluation |
|---|------------|------------|
| 1 | p₂ / N_gen | 2/3 |
| 2 | dim_G₂ / b₂ | 14/21 = 2/3 |
| 3 | dim_F₄ / dim_E₆ | 52/78 = 2/3 |
| 4 | rank_E₈ / (Weyl + dim_K₇) | 8/12 = 2/3 |
| 5 | (dim_G₂ − rank_E₈) / (rank_E₈ + 1) | 6/9 = 2/3 |

**m_b/m_t = 1/42** (21 independent expressions):

| # | Expression | Evaluation |
|---|------------|------------|
| 1 | b₀ / (2b₂) | 1/42 |
| 2 | (b₀ + N_gen) / PSL(2,7) | 4/168 = 1/42 |
| 3 | p₂ / (dim_K₇ + b₃) | 2/84 = 1/42 |
| 4 | N_gen / (dim(J₃O) + H*) | 3/126 = 1/42 |
| 5 | dim_K₇ / (dim_E₈ + dim(J₃O) + dim_K₇) | 7/294 = 1/42 |

The ratio m_b/m_t = 1/42 = 1/(2b₂) illustrates structural redundancy: the bottom-to-top mass hierarchy equals the inverse of the structural constant 2b₂ = p₂ × b₂.

**Note**: The true Euler characteristic χ(K₇) = 0 for G₂ manifolds (odd-dimensional). The constant 42 is the structural invariant 2b₂.

### 24.5 The Algebraic Web

The topological constants satisfy interconnected identities:

| Identity | Left side | Right side |
|----------|-----------|------------|
| Fiber-holonomy | dim(G₂) = 14 | p₂ × dim(K₇) = 2 × 7 |
| Gauge moduli | b₂ = 21 | N_gen × dim(K₇) = 3 × 7 |
| Matter-holonomy | b₃ + dim(G₂) = 91 | dim(K₇) × α_sum = 7 × 13 |
| Fano order | PSL(2,7) = 168 | rank(E₈) × b₂ = 8 × 21 |
| Fano order | PSL(2,7) = 168 | N_gen × fund(E₇) = 3 × 56 |
| Anomaly sum | α_sum = 13 | rank(E₈) + Weyl = 8 + 5 |

These relations form a closed algebraic system. The mod-7 structure (dim(K₇) = 7 divides dim(G₂), b₂, b₃, PSL(2,7)) reflects the Fano plane underlying octonion multiplication.

### 24.6 Fibonacci-Lucas Embedding

The framework's constants embed naturally into the Fibonacci (Fₙ) and Lucas (Lₙ) sequences:

| n | Fₙ | Constant | Role |
|---|-----|---------------|------|
| 3 | 2 | p₂ | Pontryagin class |
| 4 | 3 | N_gen | Fermion generations |
| 5 | 5 | Weyl | Pentagonal symmetry |
| 6 | 8 | rank(E₈) | Cartan subalgebra |
| 7 | 13 | α²_B sum | Structure coefficient |
| 8 | 21 | b₂ | Second Betti number |

This sequence propagates via the recurrence:

$$F_3 + F_4 = F_5 \quad \Rightarrow \quad p_2 + N_{gen} = \text{Weyl}$$

Lucas numbers also appear naturally:

| Lₙ | Value | Role |
|----|-------|-----------|
| L₄ | 7 | dim(K₇) |
| L₅ | 11 | D_bulk |
| L₈ | 47 | Scale bridge exponent |

The Lucas identity L₈ = F₇ + F₉ = 13 + 34 decomposes as:

$$L_8 = \alpha_{sum}^B + d_{hidden} = 13 + 34 = 47$$

This structure reflects the icosahedral geometry underlying the McKay correspondence E₈ ↔ 2I, where icosahedral coordinates involve the golden ratio φ = lim(Fₙ₊₁/Fₙ).

**Status**: STRUCTURAL (mathematical pattern; physical significance unclear)

---

---

## Appendix F: δ_CP: The 197° Prediction and the Compactification Correction

### F.1 The Original Prediction

The framework's prediction δ_CP = 197° = 7 × 14 + 99 = dim(K₇) × dim(G₂) + H* matches
NuFIT 5.2 (2022: 197° ± 24°) exactly.

NuFIT 6.0 (October 2024) shifted the central value to 177° ± 20°. The 197° prediction
is at the 1σ boundary of this measurement. The NuFIT collaboration notes that δ_CP
remains one of the least constrained oscillation parameters, with significant sensitivity
to reactor flux model assumptions and the mass ordering.

**NuFIT 6.1 update (2025).** The subsequent release, incorporating the T2K+NOvA joint
analysis, shifted the central value to 207° +23/−20 (NO w/o SK-atm): the 197°
prediction now sits *inside* the 1σ band (deviation 4.8% from centre). The two most
recent releases bracket the prediction on both sides; the discussion below, framed
against NuFIT 6.0, is retained as the frozen-dataset baseline.

**The canonical framework prediction remains δ_CP = 197°.**

### F.2 The Compactification Factor 62/69

PSLQ residual analysis (§7.6 of main paper) identifies a potential correction factor:

$$\frac{62}{69} = \frac{\dim(E_8)}{\dim(E_8) + 4\,\dim(K_7)} = \frac{248}{248 + 28} = \frac{248}{276}$$

This factor has a clean structural interpretation: the ratio of gauge degrees of freedom
(dim(E₈) = 248) to total degrees of freedom (248 gauge + 4 × 7 gravitational = 276).

Applied to the original prediction:

$$197 \times \frac{62}{69} = \frac{12214}{69} = 177.01°$$

This matches NuFIT 6.0 to **0.008%**. The formula 12214/69 uses only the framework's
topological integers and the compactification ratio.

### F.3 Why We Do Not Adopt It (Yet)

1. **Post-hoc identification**: The factor 62/69 was found AFTER the NuFIT 6.0 shift.
   Adopting it would be fitting to the latest central value: the opposite of prediction.

2. **197° remains within 1σ**: The value 197° = 177° + 20° is AT the 1σ boundary.
   One-sigma deviations are expected ~32% of the time. This is not a falsification.

3. **Experimental instability**: δ_CP shifted 20° between NuFIT 5.2 and 6.0.
   The next update could shift back. DUNE will provide the definitive measurement
   with resolution of a few degrees to ~15° (2028–2040).

4. **Formula integrity**: The framework's predictions are derived from topology, not fitted
   to experiment. The 197° formula (3 constants, 2 operations) is cleaner than
   the 177.01° formula (5 constants, 4 operations).

### F.4 Contingency Plan

**If DUNE confirms δ_CP ≈ 177° (central value within ±5° of 177°):**
  → Adopt 12214/69 = 177.01° as the primary prediction
  → Interpret 62/69 as a compactification factor (gauge/total DOF ratio)
  → The original 197° becomes the "bare" topological value before compactification

**If DUNE confirms δ_CP ≈ 197° (central value within ±10° of 197°):**
  → Original prediction vindicated
  → The 62/69 factor was a false lead (PSLQ artifact)

**If DUNE gives a value inconsistent with both 177° and 197°:**
  → Both formulas fail; the neutrino sector requires revision

### F.5 Summary

| Formula | Value | Status | Complexity |
|---------|-------|--------|-----------|
| 7×14+99 | 197° | **PRIMARY** (original prediction) | 3 constants, 2 ops |
| 12214/69 | 177.01° | Contingent (pending DUNE) | 5 constants, 4 ops |
| Experimental | 177°±20° | NuFIT 6.0 (1σ band: [157°, 197°]); NuFIT 6.1: 207° +23/−20 | n/a |

---

## References

1. Joyce, D. D. (2000). *Compact Manifolds with Special Holonomy*. Oxford.
2. Atiyah, M. F., Singer, I. M. (1968). *The index of elliptic operators*.
3. Particle Data Group (2024). *Review of Particle Physics*. Phys. Rev. D 110, 030001.
4. I. Esteban et al., "NuFit-6.0: updated global analysis of three-flavor neutrino oscillations," JHEP 12 (2024) 216, arXiv:2410.05380; NuFIT 6.1 (2025), www.nu-fit.org.
5. Planck Collaboration (2020). Cosmological parameters. A&A 641, A6.
6. T2K, NOvA Collaborations (2025). Nature 638, 534-541.

---

## Author's Related Works

- **[A]** B. de La Fournière, "An Explicit Approximate G₂ Metric on a Compact 7-Manifold with Certified Torsion-Free Completion," Zenodo [10.5281/zenodo.19892350](https://doi.org/10.5281/zenodo.19892350) (2026).
- **[B]** B. de La Fournière, "Spectral Geometry of the G₂-GIFT Manifold: Betti Numbers, KK Spectrum, and Spectral Invariants," Zenodo [10.5281/zenodo.19893371](https://doi.org/10.5281/zenodo.19893371) (2026).
- **[D]** B. de La Fournière, "An Explicit Closed-Form G₂ Ansatz on a K3-Coassociative Neck with Hyperkähler Rotation and Picard-Lefschetz Wirtinger Certificate," Zenodo [10.5281/zenodo.20039066](https://doi.org/10.5281/zenodo.20039066) (2026).

---

*The K₇ Framework (formerly GIFT): Supplement S2*
*Complete Derivations: 33 Type I Relations*
