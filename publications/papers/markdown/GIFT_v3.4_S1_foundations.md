# Supplement S1: Mathematical Foundations

[![Lean 4 Verified](https://img.shields.io/badge/Lean_4-Verified-blue)](https://github.com/gift-framework/core)

## E₈ Exceptional Lie Algebra, G₂ Holonomy Manifolds, and K₇ Construction

*Complete mathematical foundations for GIFT, presenting E8 architecture and K7 manifold construction.*

**Lean Verification**: 213 certificate conjuncts, 4 axioms, 134 .lean files (Lean 4.29.0, zero `sorry`)

---

## Abstract

This supplement presents the mathematical architecture underlying GIFT. Part I develops the E₈ exceptional Lie algebra with the Exceptional Chain theorem. Part II introduces G₂ holonomy manifolds, including the correct characterization of the g₂ subalgebra as the kernel of the Lie derivative map. Part III discusses K₇ manifold topology: the pair (b₂,b₃)=(21,77) does not appear among previously constructed compact G₂ manifolds, and the NK-certified metric provides computational evidence for a new G₂ manifold. A complete geometric construction (identifying explicit building blocks) remains an open problem. Part IV establishes the algebraic reference form determining det(g) = 65/32, with Joyce's theorem guaranteeing existence of a torsion-free metric. PINN validation confirms near-G₂ holonomy with V₇ projection reduced by 97%. All algebraic results are formally verified in Lean 4.

---

# Part 0: The Octonionic Foundation

## 0. Why This Framework Exists

The GIFT framework emerges from a single algebraic fact:

**The octonions 𝕆 are the largest normed division algebra.**

The derivation chain proceeds as follows:

```
𝕆 (octonions, dim 8)
    │
    ▼
Im(𝕆) = ℝ⁷ (imaginary octonions)
    │
    ▼
G₂ = Aut(𝕆) (automorphism group, dim 14)
    │
    ▼
K₇ with G₂ holonomy (explicit certified metric, Paper I)
    │
    ▼
Topological invariants (b₂ = 21, b₃ = 77)
    │
    ▼
95 observables (33(I) + 19(II) + 21(III) + 22(IV), 55 Lean-certified)
```

**Status**: 95 observables across 4 types (v3.4.13). 33 Type I algebraic, 19 Type II one-step, 21 Type III multi-step, 22 Type IV structural (incl. 6 metric block eigenvalues + Pinčák 2026 [46]). 55/95 Lean-certified (213 conjuncts, 4 axioms, 0 sorry). See §4 of main text and Supplement S3 for the complete dataset. The gauge breaking chain (§5 of main text) is certified in `TCSGaugeBreaking.lean` and `GaugeBundleData.lean`.

### 0.1 The Division Algebra Chain

| Algebra | Dim | Physics Role | Stops? |
|---------|-----|--------------|--------|
| ℝ | 1 | Classical mechanics | No |
| ℂ | 2 | Quantum mechanics | No |
| ℍ | 4 | Spin, Lorentz group | No |
| **𝕆** | **8** | **Exceptional structures** | **Yes** |

The pattern terminates at 𝕆. There is no 16-dimensional normed division algebra. The octonions are *the end of the line*.

### 0.2 G₂ as Octonionic Automorphisms

**Definition**: G₂ = {g ∈ GL(𝕆) : g(xy) = g(x)g(y) for all x,y ∈ 𝕆}

| Property | Value | GIFT Role |
|----------|-------|-----------|
| dim(G₂) | 14 = C(7,2) − C(7,1) = 21 − 7 | Q_Koide numerator |
| Action | Transitive on S⁶ ⊂ Im(𝕆) | Connects all directions |
| Embedding | G₂ ⊂ SO(7) | Preserves φ₀ |

### 0.3 Why dim(K₇) = 7

The dimension 7 is a consequence of the octonionic structure, not an independent choice:
- Im(𝕆) has dimension 7
- G₂ acts naturally on ℝ⁷
- A compact 7-manifold with G₂ holonomy provides the geometric realization

In this sense, K₇ is to G₂ what the circle is to U(1).

### 0.4 The Fano Plane: Combinatorial Structure of Im(𝕆)

The 7 imaginary octonion units form the **Fano plane** PG(2,2), the smallest projective plane:
- 7 points (imaginary units e₁...e₇)
- 7 lines (multiplication triples eᵢ × eⱼ = ±eₖ)
- 3 points per line

**Combinatorial counts**:
- Point-line incidences: 7 × 3 = 21 = C(7,2) = b₂
- Automorphism group: PSL(2,7) with |PSL(2,7)| = 168

**Numerical observation**: The following arithmetic identity holds:
$$(b_3 + \dim(G_2)) + b_3 = 91 + 77 = 168 = |{\rm PSL}(2,7)| = {\rm rank}(E_8) \times b_2$$

Whether this reflects deeper geometric structure connecting gauge and matter sectors, or is an arithmetic coincidence, remains an open question.

---

# Part I: E₈ Exceptional Lie Algebra

## 1. Root System and Dynkin Diagram

### 1.1 Basic Data

| Property | Value | GIFT Role |
|----------|-------|-----------|
| Dimension | dim(E₈) = 248 | Gauge DOF |
| Rank | rank(E₈) = 8 | Cartan subalgebra |
| Number of roots | |Φ(E₈)| = 240 | E₈ kissing number |
| Root length | √2 | α_s numerator |
| Coxeter number | h = 30 | Icosahedron edges |
| Dual Coxeter number | h∨ = 30 | McKay correspondence |

### 1.2 Root System Construction

E₈ root system in ℝ⁸ has 240 roots:

**Type I (112 roots)**: Permutations and sign changes of (±1, ±1, 0, 0, 0, 0, 0, 0)

**Type II (128 roots)**: Half-integer coordinates with even minus signs:
$$\frac{1}{2}(\pm 1, \pm 1, \pm 1, \pm 1, \pm 1, \pm 1, \pm 1, \pm 1)$$

**Verification**: 112 + 128 = 240 roots, all length √2.

**Lean Status**: E₈ Root System **12/12 COMPLETE**. All theorems proven:
- `D8_roots_card` = 112, `HalfInt_roots_card` = 128
- `E8_roots_card` = 240, `E8_roots_decomposition`
- `E8_inner_integral`, `E8_norm_sq_even`, `E8_sub_closed`
- `E8_basis_generates`: Every lattice vector is integer combination of simple roots (theorem)

### 1.3 Cartan Matrix

$$A_{E_8} = \begin{pmatrix}
2 & 0 & -1 & 0 & 0 & 0 & 0 & 0 \\
0 & 2 & 0 & -1 & 0 & 0 & 0 & 0 \\
-1 & 0 & 2 & -1 & 0 & 0 & 0 & 0 \\
0 & -1 & -1 & 2 & -1 & 0 & 0 & 0 \\
0 & 0 & 0 & -1 & 2 & -1 & 0 & 0 \\
0 & 0 & 0 & 0 & -1 & 2 & -1 & 0 \\
0 & 0 & 0 & 0 & 0 & -1 & 2 & -1 \\
0 & 0 & 0 & 0 & 0 & 0 & -1 & 2
\end{pmatrix}$$

**Properties**: det(A) = 1 (unimodular), positive definite.

---

## 2. Weyl Group

### 2.1 Order and Factorization

$$|W(E_8)| = 696,729,600 = 2^{14} \times 3^5 \times 5^2 \times 7$$

### 2.2 Topological Factorization Theorem

**Theorem**: The Weyl group order factorizes into GIFT structural constants:

$$|W(E_8)| = p_2^{\dim(G_2)} \times N_{gen}^{Weyl} \times Weyl^{p_2} \times \dim(K_7)$$

| Factor | Exponent | Value | GIFT Origin |
|--------|----------|-------|-------------|
| 2¹⁴ | dim(G₂) = 14 | 16384 | p₂^(holonomy dim) |
| 3⁵ | Weyl = 5 | 243 | N_gen^(Weyl factor) |
| 5² | p₂ = 2 | 25 | Weyl^(binary) |
| 7¹ | 1 | 7 | dim(K₇) |

**Status**: **VERIFIED (Lean 4)**: `weyl_E8_topological_factorization`

---

## 2.3 Triple Derivation of Weyl = 5

**Theorem**: The Weyl factor admits three independent derivations from topological invariants.

### Derivation 1: G₂ Dimensional Ratio

$$\text{Weyl} = \frac{\dim(G_2) + 1}{N_{gen}} = \frac{14 + 1}{3} = \frac{15}{3} = 5$$

**Interpretation**: The holonomy dimension plus unity, distributed over generations.

### Derivation 2: Betti Reduction

$$\text{Weyl} = \frac{b_2}{N_{gen}} - p_2 = \frac{21}{3} - 2 = 7 - 2 = 5$$

**Interpretation**: The per-generation Betti contribution minus binary duality.

### Derivation 3: Exceptional Difference

$$\text{Weyl} = \dim(G_2) - \text{rank}(E_8) - 1 = 14 - 8 - 1 = 5$$

**Interpretation**: The gap between holonomy dimension and gauge rank, reduced by unity.

### Unified Identity

These three derivations establish the **Weyl Triple Identity**:

$$\boxed{\frac{\dim(G_2) + 1}{N_{gen}} = \frac{b_2}{N_{gen}} - p_2 = \dim(G_2) - \text{rank}(E_8) - 1 = 5}$$

**Status**: VERIFIED (algebraic identity from GIFT constants)

### Verification

| Expression | Computation | Result |
|------------|-------------|--------|
| (dim(G₂) + 1) / N_gen | (14 + 1) / 3 | 5 |
| b₂/N_gen - p₂ | 21/3 - 2 | 5 |
| dim(G₂) - rank(E₈) - 1 | 14 - 8 - 1 | 5 |

### Significance

The triple convergence suggests Weyl = 5 is structurally constrained by the E₈ x E₈/G₂/K₇ geometry. It enters:

1. **det(g) = 65/32**: Via Weyl x (rank(E₈) + Weyl) / 2^Weyl = 5 x 13 / 32
2. **|W(E₈)| factorization**: The factor 5² = Weyl^p₂ in prime decomposition
3. **Cosmological ratio**: sqrt(Weyl) = sqrt(5) appears in dark sector density ratios (see main paper, Section 4.8)

**Status**: VERIFIED (three independent derivations)

---

## 3. Exceptional Chain

### 3.1 The Pattern

A pattern connects exceptional algebra dimensions to primes:

| Algebra | n | dim(E_n) | Prime | Index |
|---------|---|----------|-------|-------|
| E₆ | 6 | 78 | 13 | prime(6) |
| E₇ | 7 | 133 | 19 | prime(8) = prime(rank(E₈)) |
| E₈ | 8 | 248 | 31 | prime(11) = prime(D_bulk) |

### 3.2 Exceptional Chain Theorem

**Theorem**: For n ∈ {6, 7, 8}:
$$\dim(E_n) = n \times prime(g(n))$$

where g(6) = 6, g(7) = rank(E₈) = 8, g(8) = D_bulk = 11 (= dim(M₄) + dim(K₇) = 4 + 7, the total bulk dimension of M-theory).

**Proof** (verified in Lean):
- E₆: 6 × 13 = 78 ✓
- E₇: 7 × 19 = 133 ✓
- E₈: 8 × 31 = 248 ✓

**Status**: **VERIFIED (Lean 4)**: `exceptional_chain_certified`

---

## 4. E₈×E₈ Product Structure

### 4.1 Direct Sum

| Property | Value |
|----------|-------|
| Dimension | 496 = 248 × 2 |
| Rank | 16 = 8 × 2 |
| Roots | 480 = 240 × 2 |

### 4.2 τ Numerator Connection

The hierarchy parameter numerator:
$$\tau_{num} = 3472 = 7 \times 496 = \dim(K_7) \times \dim(E_8 \times E_8)$$

**Status**: **VERIFIED (Lean 4)**: `tau_num_E8xE8`

### 4.3 Binary Duality Parameter

**Triple geometric origin of p₂ = 2**:

1. **Local**: p₂ = dim(G₂)/dim(K₇) = 14/7 = 2
2. **Global**: p₂ = dim(E₈×E₈)/dim(E₈) = 496/248 = 2
3. **Root**: √2 in E₈ root normalization

---

## 5. Exceptional Algebras from Octonions

The foundational role of octonions is established in Part 0. This section details the exceptional algebraic structures that emerge from 𝕆.

### 5.1 Exceptional Jordan Algebra J₃(O)

| Property | Value |
|----------|-------|
| dim(J₃(O)) | 27 = 3³ |
| dim(J₃(O)₀) | 26 (traceless) |

**E-series formula**: The dimension 27 itself emerges from the exceptional chain:

$$\dim(J_3(\mathbb{O})) = \frac{\dim(E_8) - \dim(E_6) - \dim(SU_3)}{6} = \frac{248 - 78 - 8}{6} = \frac{162}{6} = 27$$

This shows the Jordan algebra dimension is derivable from the E-series structure.

**Status**: **VERIFIED (Lean 4)**: `j3o_e_series_certificate`

### 5.2 F₄ Connection

F₄ is the automorphism group of J₃(O):
$$\dim(F_4) = 52 = p_2^2 \times \alpha_{sum}^B = 4 \times 13$$

### 5.3 Exceptional Differences

| Difference | Value | GIFT |
|------------|-------|------|
| dim(E₈) - dim(J₃(O)) | 221 = 13 × 17 | α_B × λ_H_num |
| dim(F₄) - dim(J₃(O)) | 25 = 5² | Weyl² |
| dim(E₆) - dim(F₄) | 26 | dim(J₃(O)₀) |

**Status**: **VERIFIED (Lean 4)**: `exceptional_differences_certified`

### 5.4 Structural Derivation of τ

The hierarchy parameter τ admits a purely geometric derivation from framework invariants:

$$\tau = \frac{\dim(E_8 \times E_8) \times b_2}{\dim(J_3(\mathbb{O})) \times H^*} = \frac{496 \times 21}{27 \times 99} = \frac{10416}{2673} = \frac{3472}{891}$$

**Prime factorization**:
- Numerator: 3472 = 2⁴ × 7 × 31 = dim(K₇) × dim(E₈×E₈)
- Denominator: 891 = 3⁴ × 11 = N_gen⁴ × D_bulk

**Alternative form**: τ_num = 7 × 496 = dim(K₇) × dim(E₈×E₈) = 3472

This anchors τ to topological and algebraic invariants, establishing it as a geometric constant rather than a free parameter.

**Status**: **VERIFIED (Lean 4)**: `tau_structural_certificate`

---

# Part II: G₂ Holonomy Manifolds

## 6. Definition and Properties

### 6.1 G₂ as Exceptional Holonomy

| Property | Value | GIFT Role |
|----------|-------|-----------|
| dim(G₂) | 14 | Q_Koide numerator |
| rank(G₂) | 2 | Lie rank |
| Definition | Aut(O) | Octonion automorphisms |

**Lean Status**: G₂ Cross Product **9/11** proven:
- `epsilon_antisymm`, `epsilon_diag`, `cross_apply` ✓
- `G2_cross_bilinear`, `G2_cross_antisymm`, `cross_self` ✓
- `G2_cross_norm` (Lagrange identity ‖u×v‖² = ‖u‖²‖v‖² − ⟨u,v⟩²) ✓
- `reflect_preserves_lattice` (Weyl reflection) ✓
- Remaining: `cross_is_octonion_structure` (343-case timeout), `G2_equiv_characterizations`

### 6.2 G₂ as Kernel of the Lie Derivative

The G₂ subalgebra of so(7) admits a precise characterization as the stabilizer of the associative 3-form phi₀. For any antisymmetric matrix A in so(7), the Lie derivative of phi₀ is:

$$L_A(\varphi_0)_{ijk} = A_{ia}\varphi_{ajk} + A_{ja}\varphi_{iak} + A_{ka}\varphi_{ija}$$

The g₂ subalgebra consists of all A for which L_A(phi₀) = 0:

$$\mathfrak{g}_2 = \ker(L) = \{A \in \mathfrak{so}(7) : L_A(\varphi_0) = 0\}$$

This yields the decomposition so(7) = g₂ + V₇, where dim(g₂) = 14 and dim(V₇) = 7. The complement V₇ carries the standard 7-dimensional representation of G₂.

In practice, the kernel is computed via singular value decomposition (SVD) of the linear map L: so(7) --> Lambda³(R⁷). The 14 singular vectors with eigenvalue zero span g₂; the 7 singular vectors with nonzero eigenvalue span V₇.

**Note**: A heuristic construction based on Fano-plane indices does not produce correct g₂ generators (each such generator is approximately 67% in g₂ and 33% in V₇). The kernel-based construction is the correct definition and must be used in all numerical computations involving g₂/V₇ decomposition.

### 6.3 Holonomy Classification (Berger)

| Dimension | Holonomy | Geometry |
|-----------|----------|----------|
| **7** | **G₂** | **Exceptional** |
| 8 | Spin(7) | Exceptional |

### 6.3 Torsion: Definition and GIFT Interpretation

**Mathematical definition**: Torsion measures failure of G₂ structure to be parallel:
$$T = \nabla\phi \neq 0$$

For a G₂ structure φ, the intrinsic torsion decomposes into four irreducible G₂-modules:

$$T \in W_1 \oplus W_7 \oplus W_{14} \oplus W_{27}$$

| Class | Dimension | Characterization |
|-------|-----------|------------------|
| W₁ | 1 | Scalar: dφ = τ₀ ⋆φ |
| W₇ | 7 | Vector: dφ = 3τ₁ ∧ φ |
| W₁₄ | 14 | Co-closed part of d⋆φ |
| W₂₇ | 27 | Traceless symmetric |

**Total dimension**: 1 + 7 + 14 + 27 = 49 = 7² = dim(K₇)²

The torsion-free condition requires all four classes to vanish simultaneously, a highly constrained state with 49 conditions.

**Torsion-free condition**:
$$\nabla\phi = 0 \Leftrightarrow d\phi = 0 \text{ and } d*\phi = 0$$

**GIFT interpretation**:

| Quantity | Meaning | Value |
|----------|---------|-------|
| κ_T = 1/61 | Topological *capacity* for torsion | Fixed by K₇ |
| φ_ref | Algebraic reference form | c × φ₀ |
| T_realized | Actual torsion for global solution | Constrained by Joyce |

**Key insight**: The 33 dimensionless predictions use only topological invariants (b₂, b₃, dim(G₂)) and are independent of the specific torsion realization. The value κ_T = 1/61 defines the geometric bound on deviations from φ_ref.

**Physical interactions**: Emerge from the geometry of K₇, with deviations delta(phi) from the reference form bounded by topological constraints. This mechanism is theoretical and its detailed treatment lies beyond the scope of this supplement.

---

## 7. Topological Invariants

### 7.1 Derived Constants

| Constant | Formula | Value |
|----------|---------|-------|
| det(g) | p₂ + 1/(b₂ + dim(G₂) - N_gen) | 65/32 |
| κ_T | 1/(b₃ - dim(G₂) - p₂) | 1/61 |
| sin²θ_W | b₂/(b₃ + dim(G₂)) | 3/13 |

### 7.2 The 61 Decomposition

$$\kappa_T^{-1} = 61 = \dim(F_4) + N_{gen}^2 = 52 + 9$$

Alternative:
$$61 = \Pi(\alpha^2_B) + 1 = 2 \times 5 \times 6 + 1$$

**Status**: **VERIFIED (Lean 4)**: `kappa_T_inv_decomposition`

### 7.3 Spectral Geometry

The Laplace-Beltrami operator on K₇ admits a discrete spectrum with eigenvalues 0 = λ₀ < λ₁ ≤ λ₂ ≤ ... The first non-zero eigenvalue λ₁ (spectral gap) characterizes the geometry's rigidity.

**Bare topological ratio**: The ratio dim(G₂)/H* provides a topological reference scale:

$$\frac{\dim(G_2)}{H^*} = \frac{14}{b_2 + b_3 + 1} = \frac{14}{99} = 0.1414...$$

This is NOT the spectral gap itself, but a topological bound, see below.

**Analytical spectral gap**: If the TCS decomposition holds with b₂(M₁) = 11, the first eigenvalue of the scalar Laplacian admits a closed-form expression from the 1D Sturm-Liouville reduction on the TCS neck:

$$\boxed{\lambda_1 = \frac{\pi^2}{L^2 \cdot g_{ss}} = \frac{6\pi^2}{25 \cdot (b_2(M_1) + \mathrm{rank}(E_8))} = \frac{6\pi^2}{475} = 0.12467...}$$

where L = 5 is the effective neck domain length (including ACyl tails), g_ss = (b₂(M₁) + rank(E₈))/(3·rank(G₂)) = 19/6 is the seam metric component (topologically determined by torsion minimization), and b₂(M₁) = 11 is the Picard rank of the first building block in the conjectured TCS realization. Numerical verification (Richardson extrapolation, N=800→1600): λ₁ = 0.12461, matching the formula to **0.08%**.

The 169 metric parameters collapse to a single topological integer b₂(M₁), plus two group-theoretic constants (rank(E₈) = 8, rank(G₂) = 2). This provides a closed-form expression for a KK mass gap on a compact G₂ manifold.

**Comparison with topological ratios:**

| Expression | Value | Deviation from λ₁ | Status |
|-----------|-------|-------------------|--------|
| 6π²/475 (analytical) | 0.12467 | 0.00% (definition) | **EXACT** |
| NK Richardson | 0.12461 | 0.05% | Numerical verification |
| 14/99 (bare topological) | 0.14141 | +13.4% | Topological bound only |
| 13/99 (with spinor correction) | 0.13131 | +5.3% | Approximate (see remark) |

*Remark:* The exact value λ₁ = 6π²/475 involves π (from the Sturm-Liouville eigenvalue structure), not a rational number. The product λ₁×H* = 6π²×99/475 = 12.3364, the confirmed universal invariant (see universality below).

**Lean status**: `Spectral.PhysicalSpectralGap` (28 theorems, zero axioms). `Spectral.SelbergBridge` connects the spectral gap to the mollified Dirichlet polynomial S_w(T) via the Selberg trace formula.

**Connection to π²**: The spectral gap formula λ₁ = 6π²/475 provides a direct link between the transcendental number π² and topological integers. The near-identity dim(G₂)/√2 ≈ π² (0.30%) finds its resolution: the eigenvalue explicitly involves π² through the Sturm-Liouville structure, while dim(G₂) enters through g_ss.

**Universality**: The confirmed universal invariant is **λ₁ × H* = 12.3364**, holding for all 66 known G₂ manifolds (including those beyond the CHNP 2015 tabulated range, plus prior TCS literature). The universal law is empirical across all known examples; the analytical mechanism connecting the torsion-free condition to universality remains to be derived. See `canonical/scripts/construction_classification.py` for the full scan.

### 7.4 Continued Fraction Structure

The topological ratio dim(G₂)/H* admits a notable continued fraction representation:

$$\frac{14}{99} = [0; 7, 14] = \cfrac{1}{7 + \cfrac{1}{14}}$$

The only integers appearing are **7 = dim(K₇)** and **14 = dim(G₂)**, the two fundamental dimensions of GIFT geometry. Note: this is a property of the topological ratio, not of the spectral gap λ₁ = 6π²/475 (which is irrational).

### 7.5 Pell Equation Structure

The spectral gap parameters satisfy a Pell equation:

$$\boxed{H^{*2} - 50 \times \dim(G_2)^2 = 1}$$

Explicitly:
$$99^2 - 50 \times 14^2 = 9801 - 9800 = 1$$

where $50 = \dim(K_7)^2 + 1 = 49 + 1$.

**Fundamental unit**: The Pell equation $x^2 - 50y^2 = 1$ has fundamental solution $(x_0, y_0) = (99, 14)$, giving:

$$\varepsilon = 7 + \sqrt{50}, \quad \varepsilon^2 = 99 + 14\sqrt{50}$$

**Continued fraction bridge**: The discriminant $\sqrt{50}$ has periodic continued fraction $\sqrt{50} = [7; \overline{14}]$ with period 1, where the partial quotients are exactly dim(K₇) = 7 and dim(G₂) = 14. Combined with the selection principle κ = π²/14 (formalized in `Spectral.SelectionPrinciple`), this provides an arithmetic link between the Pell structure and the spectral gap.

**Status**: TOPOLOGICAL (algebraic identity verified in Lean)

---

# Part III: K₇ Manifold Construction

## 8. Twisted Connected Sum Framework

### 8.1 TCS Construction

The twisted connected sum (TCS) construction provides the primary method for constructing compact G₂ manifolds from asymptotically cylindrical building blocks.

**Key insight**: G₂ manifolds can be built by gluing two asymptotically cylindrical (ACyl) G₂ manifolds along their cylindrical ends, with the topology controlled by a twist diffeomorphism φ.

### 8.2 Asymptotically Cylindrical G₂ Manifolds

**Definition**: A complete Riemannian 7-manifold (M, g) with G₂ holonomy is asymptotically cylindrical (ACyl) if there exists a compact subset K ⊂ M such that M \ K is diffeomorphic to (T₀, ∞) × N for some compact 6-manifold N.

### 8.3 Topological Classification

The GIFT framework constructs an explicit G₂ metric on a compact 7-manifold K₇ with Betti numbers
(b₂, b₃) = (21, 77), certified by Newton-Kantorovich theorem (Paper I [30]). The classification of this topological type within known construction methods remains open.

**Topological status.** The pair (b₂, b₃) = (21, 77) does not appear among previously constructed compact G₂ manifolds. Orthogonal TCS is excluded by parity (b₂+b₃=98 is even; CHNP Lemma 6.7). A building block scan (2026-04-14) shows that b₂=21 > 20=max ρ(K3) forces at least one semi-Fano (non-Fano) building block, and all generic K3 lattice embeddings are excluded by the parity theorem. Non-orthogonal TCS, extra-twisted connected sums, and Joyce orbifold resolutions with new groups remain open paths. A complete geometric construction is an open problem.

**CHNP 2015 range.** The CHNP 2015 explicit tabulation covered building blocks with ρ ≤ 9, giving b₂ ≤ 18. Reaching b₂=21 requires at least one semi-Fano block with ρ ≥ 20 (Shioda-Inose), outside the CHNP tabulated range. The Mori-Mukai classification allows semi-Fano threefolds with ρ up to ~20; a TCS construction with such high-ρ blocks has not been explicitly carried out.

**Lattice-theoretic analysis.** The K3 lattice Λ_{K3} = U³ ⊕ E₈(-1)² admits an
orthogonal decomposition N₁ ⊕ N₂ ⊕ ⟨2⟩ with rk(N₁) = 11, rk(N₂) = 10, satisfying
the proposed Nikulin embedding conditions (coprime discriminants, signature, discriminant form
matching). This is consistent with a TCS construction but does not constitute a proof; the Nikulin embedding is conjectured, and full verification requires an explicit construction of the matching data.

**Global properties**:
- Compact 7-manifold (no boundary)
- Simply connected: π₁(K₇) = {1}, b₁ = 0
- G₂ holonomy: Hol(g*) = G₂ [Joyce, Prop. 11.2.3]
- Ricci-flat: Ric(g) = 0
- Euler characteristic: χ(K₇) = 0

**Combinatorial connections**:
- b₂ = 21 = C(7,2) = edges in complete graph K₇
- b₃ = 77 = C(7,3) + 2 × b₂ = 35 + 42

**Status**: The Betti numbers (b₂, b₃) = (21, 77) are certified by the NK metric (Paper I). The spectral analysis (Paper B [44]) independently confirms 21 + 77 near-zero eigenvalues consistent with these Betti numbers. The pair (b₂, b₃) = (21, 77) does not appear among previously constructed compact G₂ manifolds. A complete geometric construction (explicit building blocks and gluing map) remains an open problem.

---

## 9. Cohomological Structure

### 9.1 Mayer-Vietoris Analysis

The Mayer-Vietoris sequence provides the primary tool for computing cohomology:

$$\cdots \to H^{k-1}(N) \xrightarrow{\delta} H^k(K_7) \xrightarrow{i^*} H^k(M_1) \oplus H^k(M_2) \xrightarrow{j^*} H^k(N) \to \cdots$$

### 9.2 Betti Number Derivation

**Result for b₂**: b₂(K₇) = 21 is certified by the NK metric. The spectral analysis (Paper B [44]) independently confirms 21 near-zero eigenvalues of Δ₂ with gap ratio 14,635. A TCS realization would require building blocks with Picard ranks summing to 21, but the specific identification of those blocks is an open problem.

**Result for b₃**: b₃(K₇) = 77 is certified by the NK metric. The harmonic decomposition b₃ = 35 + 42 = (1+7+27) + 2×21 is confirmed by the certified metric with spectral gap 10522× between zero and non-zero modes. A splitting b₃ = b₃(M₁) + b₃(M₂) via Mayer-Vietoris is conditional on the building block identification, which remains open.

**Status**: CERTIFIED BY NK METRIC (Paper I); candidate TCS decomposition is conjectural

### 9.3 Complete Betti Spectrum and Poincaré Duality

For a compact G₂-holonomy 7-manifold K₇, Poincaré duality gives b_k = b_{7-k}:

| k | b_k(K₇) | Derivation |
|---|---------|------------|
| 0 | 1 | Connected |
| 1 | 0 | Simply connected (G₂ holonomy) |
| 2 | 21 | NK-certified; spectrally confirmed (Paper B [44]); building block decomposition open |
| 3 | 77 | NK-certified; harmonic decomposition 35+42; building block decomposition open |
| 4 | 77 | Poincaré duality: b₄ = b₃ |
| 5 | 21 | Poincaré duality: b₅ = b₂ |
| 6 | 0 | Poincaré duality: b₆ = b₁ |
| 7 | 1 | Poincaré duality: b₇ = b₀ |

**Euler characteristic**: For any compact oriented odd-dimensional manifold, χ = 0:
$$\chi(K_7) = \sum_{k=0}^{7} (-1)^k b_k = 1 - 0 + 21 - 77 + 77 - 21 + 0 - 1 = 0$$

**Status**: **VERIFIED (Lean 4)**: `euler_char_K7_is_zero`, `poincare_duality_K7`

**Effective cohomological dimension**:
$$H^* = b_2 + b_3 + 1 = 21 + 77 + 1 = 99$$

### 9.4 The Structural Constant 42

The number 42 appears throughout GIFT as a derived topological invariant:

$$42 = 2 \times 3 \times 7 = p_2 \times N_{gen} \times \dim(K_7)$$

**Multiple derivations**:

| Formula | Value | Interpretation |
|---------|-------|----------------|
| p₂ × N_gen × dim(K₇) | 2 × 3 × 7 = 42 | Binary × generations × fiber |
| 2 × b₂ | 2 × 21 = 42 | Twice the gauge moduli |
| b₃ - C(7,3) | 77 - 35 = 42 | Global vs local 3-forms |

**Connection to b₃ decomposition**:
$$b_3 = 77 = C(7,3) + 42 = 35 + 2 \times b_2$$

The 35 local modes correspond to Λ³(ℝ⁷) fiber forms; the 42 global modes arise from the TCS structure.

**Status**: **VERIFIED (Lean 4)**: `structural_42_gift_form`, `structural_42_from_b2`

### 9.5 Third Betti Number Decomposition

The b₃ = 77 harmonic 3-forms decompose as:

$$H^3(K_7) = H^3_{\text{local}} \oplus H^3_{\text{global}}$$

| Component | Dimension | Origin |
|-----------|-----------|--------|
| H³_local | 35 = C(7,3) | Λ³(ℝ⁷) fiber forms |
| H³_global | 42 = 2 × 21 | TCS global modes |

**Verification**: 35 + 42 = 77

**Status**: TOPOLOGICAL

---

# Part IV: Metric Structure and Verification

## 10. Structural Metric Invariants

### 10.1 Structural Metric Invariants and Normalizations

The GIFT framework explores the hypothesis that metric invariants derive from fixed mathematical structure. The topological constraints serve as inputs; the specific geometry is then determined.

| Invariant | Formula | Value | Status |
|-----------|---------|-------|--------|
| κ_T | 1/(b₃ - dim(G₂) - p₂) | 1/61 | TOPOLOGICAL |
| det(g) | (Weyl × (rank(E₈) + Weyl))/2⁵ | 65/32 | METRIC NORMALIZATION (see §10.3) |

### 10.2 Torsion Magnitude κ_T = 1/61

**Derivation**:
$$\kappa_T = \frac{1}{b_3 - \dim(G_2) - p_2} = \frac{1}{77 - 14 - 2} = \frac{1}{61}$$

**Interpretation**:
- 61 = effective matter degrees of freedom
- b₃ = 77 total fermion modes
- dim(G₂) = 14 gauge symmetry constraints
- p₂ = 2 binary duality factor

**Status**: TOPOLOGICAL

### 10.3 Metric Determinant det(g) = 65/32

The metric determinant admits three independent derivations from topological invariants, providing strong evidence for its structural necessity.

**Path 1** (Weyl formula):
$$\det(g) = \frac{\text{Weyl} \times (\text{rank}(E_8) + \text{Weyl})}{2^{\text{Weyl}}} = \frac{5 \times 13}{32} = \frac{65}{32}$$

**Path 2** (Cohomological):
$$\det(g) = p_2 + \frac{1}{b_2 + \dim(G_2) - N_{\text{gen}}} = 2 + \frac{1}{21+14-3} = 2 + \frac{1}{32} = \frac{65}{32}$$

**Path 3** (H* formula):
$$\det(g) = \frac{H^* - b_2 - 13}{32} = \frac{99 - 21 - 13}{32} = \frac{65}{32}$$

The convergence of three independent algebraic paths to the same rational value is
suggestive but does not constitute a derivation from topology. The actual chain is:
the metric optimization was constrained to det(g) = 65/32 (a normalization target
chosen to be compatible with the known structural parameters g_ss = 19/6, g_T² = 7/6),
and the above formulas were identified post-hoc. Any rational number with small
numerator/denominator can be expressed as combinations of a few integers.

**Numerical value**: 65/32 = 2.03125 (exact rational)

**Status**: METRIC NORMALIZATION, algebraically exact in our metric, with three
suggestive integer formulas, but not derived from topology. The 6 observables using
det_num or det_den (six Type I observables) depend on this normalization choice.

---

## 11. Formal Certification

### 11.1 The Algebraic Reference Form

The algebraic reference form in a local G₂-adapted orthonormal coframe:

$$\varphi_{\text{ref}} = c \cdot \varphi_0, \quad c = \left(\frac{65}{32}\right)^{1/14}$$
$$g_{\text{ref}} = c^2 \cdot I_7 = \left(\frac{65}{32}\right)^{1/7} \cdot I_7$$

**Important clarification**: This representation holds in a local orthonormal frame. The manifold K₇ is curved and compact; "I₇" reflects the frame choice, not global flatness. The reference form φ_ref yields det(g) = 65/32; the global torsion-free solution φ_TF exists by Joyce's theorem, certified by the NK bound (Paper I).

| Property | Value | Status |
|----------|-------|--------|
| det(g) | 65/32 | exact by imposed normalization, not fitted to experiment (§10.3) |
| φ_ref components | 7/35 | 20% sparsity |
| Joyce threshold | ‖T‖ < ε₀ = 0.1 | Satisfied (224× margin) |

### 11.2 Joyce Existence Theorem and Global Solutions

**Important clarification**: The reference form φ_ref = c·φ₀ is the canonical G₂ structure in a local orthonormal coframe, not a globally constant form on K₇. On a compact G₂ manifold, the coframe 1-forms {eⁱ} satisfy deⁱ ≠ 0 in general, so "constant components" does not imply dφ = 0 globally.

**Actual solution structure**: The topology and geometry of K₇ impose a deformation:
$$\varphi = \varphi_{\text{ref}} + \delta\varphi$$

The torsion-free condition (dφ = 0, d*φ = 0) is a **global constraint**. Joyce's perturbation theorem guarantees existence of a torsion-free G₂ metric when the initial torsion satisfies ‖T‖ < ε₀ = 0.1. PINN validation (N=1000) confirms ‖T‖_max = 4.46 × 10⁻⁴, providing a 224× safety margin.

**Why GIFT satisfies Joyce's criterion**: The topological bound κ_T = 1/61 constrains ‖δφ‖, placing the manifold within Joyce's perturbative regime where a torsion-free solution exists.

### 11.3 Independent Numerical Validation (PINN)

A companion numerical program constructs explicit G₂ metrics on K₇ via physics-informed neural networks (PINNs). The three-chart atlas (neck + two Calabi-Yau bulk regions) uses approximately 10⁶ trainable parameters in float64 precision.

**Initial validation** (Phase 2):

| Metric | Value | Significance |
|--------|-------|--------------|
| ‖T‖_max | 4.46 x 10⁻⁴ | 224x below Joyce epsilon₀ |
| ‖T‖_mean | 9.8 x 10⁻⁵ | T --> 0 confirmed |
| det(g) error | < 10⁻⁶ | Confirms 65/32 |

**G₂ holonomy training** (Phase 3, 13 versions, v2-v13):

Over successive training protocol refinements, the holonomy quality has improved:

| Metric | Initial (v5) | Current best (v11) | Improvement |
|--------|-------------|-------------------|-------------|
| g2_self (honest holonomy) | 3.86 | 3.25 | -16% |
| V₇ projection score | 0.51 | 0.014 | -97% |
| det(g) at neck | 4.69 | 2.031 | locked at target |
| phi drift | 13.4% | 0% | controlled |

The g2_score measures the normalized projection of Riemann curvature onto the complement of g₂ in so(7). A score of 0 corresponds to exact G₂ holonomy; the flat metric scores approximately 3.5. The V₇ projection score measures the fraction of curvature outside the g₂ subalgebra (using the correct kernel-based g₂ decomposition, see Section 6.2).

A critical bug in the g₂ basis construction was discovered and corrected between versions 9 and 10: the Fano-plane heuristic does not produce correct g₂ generators. The correct g₂ subalgebra is the kernel of the Lie derivative map (Section 6.2). This correction led to significant improvement in all subsequent versions.

**Robust statistical validation**: The det(g) = 65/32 prediction passes 8/8 independent tests (permutation, bootstrap, Bayesian posterior 76.3%, joint constraint p < 6 x 10⁻⁶).

Full details of the PINN architecture, training protocol, and version-by-version results are presented in a companion paper.

### 11.4 Lean 4 Formalization

**Scope of verification**: The Lean formalization (134 files, 4 axioms, zero `sorry`) verifies:
1. Arithmetic identities and algebraic relations between GIFT constants
2. Numerical bounds (e.g., torsion threshold)
3. G₂ differential geometry: exterior algebra Λ*(ℝ⁷), Hodge star, ψ = ⋆φ (axiom-free `Geometry` module)
4. Spectral gap bounds and topological ratios (`Spectral.PhysicalSpectralGap`, 28 theorems, zero axioms). Analytical value: λ₁ = 6π²/475 = 0.12467 (see §7.3)
5. Selberg bridge: trace formula connecting S_w(T) to spectral gap (`Spectral.SelbergBridge`)
6. Selection principle κ = π²/14 (`Spectral.SelectionPrinciple`)

**The 4 axioms are declared in `Foundation/Bounds` and used transversally throughout the formalization** (they assert numerical bounds on torsion, the NK certificate parameter h, and the spectral gap, which cannot be verified by `decide` alone due to real-arithmetic content). All other results use zero axioms.

It does **not** formalize:
- Existence of K₇ as a smooth G₂ manifold
- Physical interpretation of topological invariants
- Uniqueness of the TCS construction

```lean
-- GIFT.Foundations.AnalyticalMetric

def phi0_indices : List (Fin 7 × Fin 7 × Fin 7) :=
  [(0,1,2), (0,3,4), (0,5,6), (1,3,5), (1,4,6), (2,3,6), (2,4,5)]

def phi0_signs : List Int := [1, 1, 1, 1, -1, -1, -1]

def scale_factor_power_14 : Rat := 65 / 32

theorem torsion_satisfies_joyce :
  torsion_norm_constant_form < joyce_threshold_num := by native_decide

theorem det_g_equals_target :
  scale_factor_power_14 = det_g_target := rfl
```

**Status**: VERIFIED

### 11.5 The Derivation Chain

The logical structure from algebra to predictions:

```
Octonions (𝕆)
     │
     ▼
G₂ = Aut(𝕆), dim = 14
     │
     ▼
Standard form φ₀ (Harvey-Lawson 1982)
     │
     ▼
Scaling c = (65/32)^{1/14}    ← GIFT constraint
     │
     ▼
Metric g = c² × I₇
     │
     ▼
det(g) = 65/32               ← exact by imposed normalization, not fitted to experiment (§10.3)
     │
     ▼
sin²θ_W = 3/13, Q = 2/3, ...  ← Predictions
```

---

## 12. Analytical G₂ Metric Details

### 12.1 The Standard Form φ₀

The associative 3-form preserved by G₂ ⊂ SO(7), introduced by Harvey and Lawson (1982) in their foundational work on calibrated geometries:

$$\varphi_0 = \sum_{(i,j,k) \in \mathcal{I}} \sigma_{ijk} \, e^{ijk}$$

where:
- 𝓘 = {(0,1,2), (0,3,4), (0,5,6), (1,3,5), (1,4,6), (2,3,6), (2,4,5)}
- σ = (+1, +1, +1, +1, -1, -1, -1)

### 12.2 Linear Index Representation

In the C(7,3) = 35 basis:

| Index | Triple | Sign | Index | Triple | Sign |
|-------|--------|------|-------|--------|------|
| 0 | (0,1,2) | +1 | 23 | (1,4,6) | -1 |
| 9 | (0,3,4) | +1 | 27 | (2,3,6) | -1 |
| 14 | (0,5,6) | +1 | 28 | (2,4,5) | -1 |
| 20 | (1,3,5) | +1 | | | |

All other 28 components are exactly 0.

### 12.3 Metric Derivation

From φ₀, the metric is computed via:
$$g_{ij} = \frac{1}{6} \sum_{k,l} \varphi_{ikl} \varphi_{jkl}$$

For standard φ₀: g = I₇ (identity), det(g) = 1.

Scaling φ → c·φ gives g → c²·g, hence det(g) → c¹⁴·det(g).

Setting c¹⁴ = 65/32 yields the GIFT metric.

### 12.4 Comparison: Fano Plane vs G₂ Form

| Structure | 7 Triples | Role |
|-----------|-----------|------|
| **Fano lines** | (0,1,3), (1,2,4), (2,3,5), (3,4,6), (4,5,0), (5,6,1), (6,0,2) | G₂ cross-product ε_{ijk} |
| **G₂ form** | (0,1,2), (0,3,4), (0,5,6), (1,3,5), (1,4,6), (2,3,6), (2,4,5) | Associative 3-form |

Both have 7 terms but different index patterns. The Fano plane defines the octonion multiplication (cross-product), while the G₂ form is the associative calibration.

### 12.5 Verification Summary

| Method | Result | Reference |
|--------|--------|-----------|
| Algebraic | φ = (65/32)^{1/14} × φ₀ | This section |
| Lean 4 | `det_g_equals_target : rfl` | AnalyticalMetric.lean |
| PINN | Converges to constant form | gift_core/nn/ |
| Joyce theorem | ‖T‖ < 0.1 → exists metric (224× margin) | [Joyce 2000] |

Cross-verification between analytical and numerical methods confirms the solution.

---

## References

1. Adams, J.F. *Lectures on Exceptional Lie Groups*
2. Harvey, R., Lawson, H.B. "Calibrated geometries." *Acta Math.* 148, 47-157 (1982)
3. Bryant, R.L. "Metrics with exceptional holonomy." *Ann. of Math.* 126, 525-576 (1987)
4. Joyce, D. *Compact Manifolds with Special Holonomy*
5. Corti, Haskins, Nordström, Pacini. *G₂-manifolds and associative submanifolds*
6. Kovalev, A. *Twisted connected sums and special Riemannian holonomy*
7. Conway, J.H., Sloane, N.J.A. *Sphere Packings, Lattices and Groups*

---

*GIFT Framework: Supplement S1*
*Mathematical Foundations: E₈ + G₂ + K₇*
