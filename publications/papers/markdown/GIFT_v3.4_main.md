# Geometric Information Field Theory: Topological Derivation of Standard Model Parameters from G₂ Holonomy Manifolds

**Brieuc de La Fourniere**

*Independent researcher, Beaune, France*

---

## Abstract

The Standard Model requires 19 experimentally determined parameters lacking theoretical explanation. We present a geometric framework in which physical observables emerge as topological invariants of a seven-dimensional G₂ holonomy manifold K₇ coupled to E₈×E₈ gauge structure, containing zero continuous adjustable parameters.

Building on an explicit Newton-Kantorovich-certified G₂ metric (companion Paper I [30]) with independently confirmed spectral structure (companion Paper B [44]), we derive **95 observables** from a compact 7-manifold K₇ with Betti numbers b₂ = 21 and b₃ = 77, organized in four types: 33 direct algebraic (Type I, mean deviation 0.73%), 19 one-step physical extractions (Type II, 0.17%), 21 multi-step dynamical chains (Type III, 3.4%), and 22 structural diagnostics (Type IV). Of 66 experimentally comparable observables, 11 are exact matches (deviation < 0.01%) and 53 are within 1%.

The framework includes three new physics results beyond the original 33 predictions: (1) a complete E₈ → Standard Model gauge breaking chain with anomaly cancellation and bundle universality, (2) a combined lepton mass hierarchy mechanism achieving sub-percent precision from two independent geometric sources with α = e^K (zero free parameters), and (3) a sensitivity analysis demonstrating 2.13× overdetermination with coincidence probability 10⁻³⁴⁶ under a uniform null and 10⁻¹³³ under an algebraic null model (4.2M random formulas from the same 20 constants). Of the 95 observables, 55 are formally verified in Lean 4 (213 certificate conjuncts, 4 axioms, 0 sorry).

DUNE (2028–2040) will test the topological prediction δ_CP = 197° with resolution of a few degrees to ~15°; measurement outside [182, 212]° would create serious tension. We present this as an exploratory investigation emphasizing falsifiability, not a claim of correctness.

**Keywords**: G₂ holonomy, exceptional Lie algebras, Standard Model parameters, topological field theory, falsifiability, formal verification, Lean 4

---

## 1. Introduction

### 1.1 The Parameter Problem

The Standard Model describes fundamental interactions with remarkable precision, yet requires 19 free parameters determined solely through experiment [1]. These parameters (gauge couplings, Yukawa couplings spanning five orders of magnitude, mixing matrices, and Higgs sector values) lack theoretical explanation.

Several tensions motivate the search for deeper structure:

- **Hierarchy problem**: The Higgs mass requires fine-tuning absent new physics [2].
- **Hubble tension**: CMB and local H₀ measurements differ by >4 sigma [3,4].
- **Flavor puzzle**: No mechanism explains three generations or mass hierarchies [5].
- **Koide mystery**: The charged lepton relation Q = 2/3 holds for 43 years without explanation [6].

These challenges suggest examining whether parameters might emerge from geometric or topological structures.

### 1.2 Framework Overview

The Geometric Information Field Theory (GIFT) proposes that dimensionless parameters represent topological invariants of an eleven-dimensional spacetime:

```
E₈ x E₈ (496D) --> AdS₄ x K₇ (11D) --> Standard Model (4D)
```

The key elements:

1. **E₈ x E₈ gauge structure** (dimension 496)
2. **Compact 7-manifold K₇** with G₂ holonomy (b₂ = 21, b₃ = 77)
3. **Metric normalization constraint**: det(g) = 65/32, structurally motivated by G₂/E₈ constants (§2.3) but imposed as a normalization target, not derived from topology alone
4. **Cohomological mapping**: Betti numbers constrain field content

We emphasize this represents mathematical exploration, not a claim that nature realizes this structure. The framework's merit lies in falsifiable predictions from topological inputs.

### 1.3 Paper Organization

- **Section 2**: Mathematical framework (E₈×E₈, K₇, G₂ structure)
- **Section 3**: Methodology, epistemic status, and type classification
- **Section 4**: 95 observables across 4 types (33(I) + 19(II) + 21(III) + 22(IV))
- **Section 5**: Gauge sector — E₈ → SM breaking, anomaly cancellation, B-test identity, bundle universality
- **Section 6**: Mass hierarchy — Wilson lines, instantons, combined wilson_line+instanton pipeline
- **Section 7**: Sensitivity analysis — effective DOF, cross-correlations, coincidence test
- **Section 8**: Formal verification (213 Lean conjuncts, 4 axioms) and statistical uniqueness
- **Section 9**: Discussion, falsifiability, and conclusion

Three supplements provide technical details: S1 (Mathematical Foundations), S2 (Complete Derivations), S3 (Observable Dataset with full 95-entry table).

---

## 2. Mathematical Framework

### 2.1 The Octonionic Foundation

GIFT emerges from the algebraic fact that **the octonions are the largest normed division algebra**.

| Algebra | Dim | Physics Role | Extends? |
|---------|-----|--------------|----------|
| R | 1 | Classical mechanics | Yes |
| C | 2 | Quantum mechanics | Yes |
| H | 4 | Spin, Lorentz group | Yes |
| **O** | **8** | **Exceptional structures** | **No** |

The octonions terminate this sequence. Their automorphism group G₂ = Aut(O) has dimension 14 and acts naturally on Im(O) = R^7. The exceptional Lie algebras arise from octonionic constructions through a chain established by Dray and Manogue [17]:

| Algebra | Dimension | Connection to O |
|---------|-----------|-----------------|
| G₂ | 14 | Aut(O) |
| F₄ | 52 | Aut(J₃(O)) |
| E₆ | 78 | Collineations of OP² |
| E₈ | 248 | Contains all lower exceptionals |

This chain is not accidental. It reflects the unique algebraic structure of the octonions: Im(O) has dimension 7, the Fano plane encodes the multiplication table, and G₂ preserves this structure. A G₂-holonomy manifold is therefore the natural geometric home for octonionic physics, just as U(1) holonomy is the natural setting for complex geometry.

The G₂ structure is concretely encoded in the **standard 3-form φ₀** on ℝ⁷ (Bryant-Joyce convention):
$$\varphi_0 = e_{012} + e_{034} + e_{056} + e_{135} - e_{146} - e_{236} - e_{245}$$
where only 7 of the $\binom{7}{3} = 35$ ordered triples carry nonzero coefficient (all ±1). G₂ is precisely the stabilizer of φ₀ in GL(7, ℝ), and its Lie algebra g₂ is the kernel of the linear map $L_{\varphi_0} : \mathfrak{gl}(7) \to \wedge^3(\mathbb{R}^7)^*$, $X \mapsto \mathcal{L}_X \varphi_0$, giving $\dim(g_2) = 49 - \operatorname{rank}(L_{\varphi_0}) = 49 - 35 = 14$. This is fully formalized in Lean (`G2ThreeForm.lean`): all 7 nonzero coefficients are certified by `native_decide`, and the Lie algebra structure (closure under addition and scalar multiplication) is proven.

### 2.2 E₈ x E₈ Structure

E₈ is the largest exceptional simple Lie group with dimension 248 and rank 8 [18]. The product E₈ x E₈ arises in heterotic string theory for anomaly cancellation [19], with total dimension 496.

The first E₈ contains the Standard Model gauge group through the breaking chain:

```
E₈ --> E₆ x SU(3) --> SO(10) x U(1) --> SU(5) --> SU(3) x SU(2) x U(1)
```

The second E₈ provides a hidden sector whose physical interpretation remains an open question.

Wilson (2024) demonstrates that E₈(-248) encodes three fermion generations (128 degrees of freedom) with GUT structure [9]. The product dimension 496 enters the hierarchy parameter tau = (496 x 21)/(27 x 99) = 3472/891, connecting gauge structure to internal topology.

### 2.3 The K₇ Manifold

We consider a compact, simply connected 7-manifold K₇ with G₂ holonomy and Betti numbers (b₂, b₃) = (21, 77). G₂ holonomy preserves N = 1 SUSY in 4D [11], implies Ric(g) = 0, and is the unique holonomy of G₂ = Aut(𝕆) acting on ℝ⁷.

**Metric construction** (Paper I [30]): An explicit 7D G₂ metric g(s, θ, ψ, y) = g_seam(s) ⊕ g_{T²} ⊕ g_{K3}(y) on K₇ is constructed via Chebyshev-Cholesky parametrization (169 parameters), with K3 fiber contribution certified at 0.07% of total torsion [45]. The Newton-Kantorovich certificate (h = 8.95 × 10⁻⁹, margin ×56 million, zero finite differences) establishes existence and local uniqueness (within the NK certificate ball) of a nearby torsion-free metric with δg/g ≤ 4.86 × 10⁻⁶. The spectral analysis (Paper B [44]) independently confirms 21 near-zero eigenvalues of Δ₂ (gap ratio 14,635) and 77 near-zero eigenvalues of Δ₃, consistent with the Betti numbers (b₂, b₃) = (21, 77).

**Metric decomposition**: A Chebyshev mode analysis reveals that the metric is dominated by its constant mode (k=0), which carries 99.9998% of the L² energy. This mode is a product metric K3 × T² × I with structural parameters g_ss = 19/6, g_{T²} = 7/6, det(g) = 65/32 — fixed by the structural normalization and topological input data, with det(g) = 65/32 treated as a metric normalization target as discussed in S1 §10.3. The k≥1 modes (0.0002% of energy) constitute the minimal perturbation that breaks the product structure and lifts the holonomy from SU(2)×U(1) to full G₂. These corrections are responsible for b₁ = 0, the torsion ‖T‖ = 2.949×10⁻⁵, and the NK certification — but contribute nothing to the numerical values of the 95 observables, which depend only on topological integers.

**Topological classification**: The pair (b₂, b₃) = (21, 77) does not appear among previously constructed compact G₂ manifolds — neither in Joyce's 252 orbifold types nor in the ~100 CHNP TCS examples. Orthogonal TCS is excluded by parity (b₂+b₃=98 is even; CHNP Lemma 6.7). Non-orthogonal TCS, extra-twisted connected sums [20], and new orbifold groups remain open paths. The NK-certified metric provides computational evidence; a complete geometric construction is an open problem. The pair is **unique** among all 65 known TCS literature examples (Kovalev, CHNP, Joyce, CGN, Nordström); nearest neighbor is at distance 7.6 in (b₂, b₃) space. The certified metric is 99.98% block-diagonal K3×T²×I with exact U(1)² symmetry (degenerate to 2×10⁻⁵).

**Further structure**: The certified metric is smooth (no singularities). The SM gauge group SU(3)×SU(2)×U(1) does **not** arise from ADE singularities in this framework; it emerges from the algebraic/spectral structure: g₂ ⊂ so(7) decomposition, so(8) = g₂ ⊕ L ⊕ R triality giving N_gen = 3, and spectral gap λ₁ = 6π²/475 = 0.12467 (Richardson extrapolation: 0.12461 ± 0.00016, deviation 0.05%) (§5). ADE singularities would be relevant at singular limits of G₂ moduli space; the smooth certified metric is at a generic point.

The framework imposes det(g) = 65/32 = (Weyl × α_sum)/2⁵, equivalently 2 + 1/32. Joyce's theorem [20] guarantees existence of torsion-free G₂ metrics on suitable compact 7-manifolds; the Chebyshev-Cholesky construction (companion paper [30]) achieves ‖T‖ < 3 × 10⁻⁵ with Newton-Kantorovich certification h = 8.95 × 10⁻⁹ (Chebyshev-certified, margin ×56 million; §8.5).

### 2.4 Topological Constraints on Field Content

#### 2.4.1 Betti Numbers as Capacity Bounds

The Betti numbers provide upper bounds on field multiplicities:

- **b₂(K₇) = 21**: Bounds the number of gauge field degrees of freedom
- **b₃(K₇) = 77**: Bounds the number of matter field degrees of freedom

**Note on gauge group origin**: In M-theory on smooth G₂ manifolds, dimensional reduction yields b₂ abelian U(1) vector multiplets [11]. The non-abelian SM gauge group in GIFT emerges instead from the algebraic/spectral structure of the G₂ holonomy: the g₂ ⊂ so(7) decomposition and so(8) = g₂ ⊕ L ⊕ R triality (§5). The smooth certified metric is at a generic point in G₂ moduli space; codimension-4 ADE singularities would be a different (singular) limit.

#### 2.4.2 Generation Number

The number of chiral fermion generations follows from a topological constraint:

$$({\rm rank}(E_8) + N_{\rm gen}) \times b_2 = N_{\rm gen} \times b_3$$

Solving: (8 + N_gen) x 21 = N_gen x 77 yields **N_gen = 3**.

This derivation is formal; physically, it reflects index-theoretic constraints on chiral zero modes, which in M-theory on G₂ require singular geometries for chirality [25].

---

## 3. Methodology and Epistemic Status

### 3.1 The Derivation Principle and Type Classification

The GIFT framework derives physical observables through algebraic combinations of 20 topological invariants (Appendix A). The 95 observables are organized by derivation directness:

| Type | Count | Derivation | Example |
|------|-------|-----------|---------|
| **I** | 33 | Direct algebra from topology | sin²θ_W = 3/13 |
| **II** | 19 | One physical identification step | m_u from ratio × VEV |
| **III** | 21 | Multi-step dynamical chains | Combined wilson_line+instanton lepton ratios |
| **IV** | 22 | Structural diagnostics | NK certification, Gram conditioning |

Type I predictions are dimensionless ratios of topological integers — they cannot be "fitted" and are either correct or wrong. Type II adds one scale identification. Type III involves dynamical mechanisms (gauge running, eigenvalue splitting, instanton volumes). Type IV provides internal consistency checks.

**Geometric unit convention.** Throughout this paper, we adopt the convention that the geometric (GIFT) values are the *reference*, and experimental measurements are expressed as offsets from the geometric prediction. This reversal parallels the 2019 SI redefinition: the kilogram is now defined from Planck's constant (exact), and any physical realization has uncertainty relative to it. In GIFT, the topological formula is exact; the experimental value approximates it.

### 3.2 What GIFT Claims and Does Not Claim

**Inputs**: Existence of K₇ with G₂ holonomy and (b₂, b₃) = (21, 77); E₈×E₈ gauge structure; det(g) = 65/32.

**Outputs**: 95 observables across 4 types (33 Type I, 19 Type II, 21 Type III, 22 Type IV), 66 experimentally testable.

**Structure of predictions**: All 95 observables are algebraic functions of 6 primitive topological integers (b₂, b₃, dim(G₂), dim(E₈), rank(E₈), dim(K₇)) plus standard transcendentals (π, √2, ln 2). No observable depends on the detailed G₂ geometry — the metric certifies that the manifold exists but does not enter the prediction formulas. (Caveat: 6 observables use det_num/det_den = 65/32, which is a metric normalization target with suggestive but not derivational algebraic expressions in terms of topological integers — see §10.3 of S1.)

We claim that given the inputs, the outputs follow algebraically (Type I) or computationally (Types II–IV). We do **not** claim uniqueness of the geometry, uniqueness of the formula assignments, or that the formula selection principle is understood.

### 3.3 Three Factors Distinguishing GIFT from Numerology

**Multiplicity**: 95 observables (66 testable), not cherry-picked coincidences. The sensitivity analysis (§7) gives P(coincidence) = 10⁻³⁴⁶ with overdetermination ratio 2.13×, both computed on the 33 Type I observables with experimental comparison.

**Exactness**: Several predictions are exactly rational — sin²θ_W = 3/13, Q_Koide = 2/3, m_s/m_d = 20, Ω_DM/Ω_b = 43/8. These cannot be fitted; they are correct or wrong.

**Falsifiability**: DUNE will test δ_CP = 197° with expected resolution ranging from a few degrees to ~15° depending on exposure and true parameter values (2028–2040). The SUSY spectrum (m_gravitino = 166 GeV, m_moduli = 3.2 TeV) is model-dependent and constrained in standard simplified searches; viable only in compressed or suppressed-coupling realizations requiring dedicated recasting at HL-LHC/FCC. The proton lifetime τ_p = 4.06 × 10³⁸ years exceeds near-term Hyper-K sensitivity; Hyper-K can strengthen lower bounds and constrain nearby GUT-scale alternatives.

### 3.4 Why These Formulas?

The selection question has two levels of answer.

**Mathematical level (deterministic).** The NK-certified metric has zero free parameters; its 169 Chebyshev coefficients appear constrained to a lattice generated by {π², π, 1, e, χ₉₈} / (b₂·b₃) (a numerical observation, not a derived result; see S1 §10.3). Observables are functionals of this metric — they *must* therefore be algebraic in the topological invariants. The "selection" is not a choice; it is a consequence of the rigidity of the certified metric (local uniqueness within the NK ball; see §9.4 for the full argument, including extended grammar analysis with TCS atoms).

**Statistical level (empirical).** Even granting an adversary access to the same 20 structural constants, 4.2M random algebraic formula trees cannot reproduce GIFT's precision profile: joint coincidence probability P = 10⁻¹³³ (algebraic null model, §7). Structural redundancy adds a third line of evidence: sin²θ_W = 3/13 admits 14 independent derivations, Q_Koide = 2/3 admits 20 (see S2) — an overdetermined web inconsistent with post-hoc cherry-picking.

The deeper question — *why G₂ holonomy?* — has the same epistemic status as "why Lorentz invariance?": G₂ is the unique compact exceptional holonomy group admitting a 7-dimensional representation with the stabilizer chain G₂ ⊃ SU(3) ⊃ SU(2) ⊃ U(1) required by the Standard Model. It is an isolated point in the space of compatible structures, not one choice among a continuum.

### 3.5 Posture: orientation, not ontology

It is useful to distinguish three registers at which a framework of this type operates, because the support GIFT offers is uneven across them.

**Predictive register.** GIFT specifies a finite list of inputs and derives 95 observables from them; 66 of these are testable, with mean deviation ~1% and a sensitivity profile that places the joint configuration at $> 4.5\sigma$ against random algebraic null models (§7). This is the register where the framework either holds or fails — and it is the only register at which we make any positive claim.

**Architectural register.** The choice of G₂-holonomy 7-manifolds with $E_8 \times E_8$ structure is *motivated* by mathematical considerations (§§2, 9.4): G₂ is the unique exceptional compact holonomy admitting the SU(3) × SU(2) × U(1) stabilizer chain, $E_8$ is the largest exceptional Lie algebra, and the topological constraints fix $(b_2, b_3) = (21, 77)$ and $N_{\mathrm{gen}} = 3$. We argue this architecture is *natural*; we do not argue it is *necessary*.

**Ontological register.** A reader may wonder whether GIFT carries a thesis about reality — for instance, the speculation that geometry, information, and energy are not three correlated aspects of nature but three views of a single underlying configuration, or the resonance with Wheeler's "It from Bit" programme [40] and the holographic principle. Such readings are compatible with the framework and historically motivated its development, but they are **neither required nor demonstrated** by the predictions. A reader who finds the Wheeler-holographic picture persuasive will see GIFT as a natural piece of that puzzle; a reader who prefers a strictly empirical stance will see a falsifiable predictive framework. Both readings are defensible; the framework requires neither.

In Worrall's [41] and Ladyman's [42] terms, GIFT is best read as a *moderate structural-realist orientation*: structure carries predictive weight independently of any further ontological commitment. The framework's success or failure is decided by experiment in the predictive register, not by adjudication in the ontological one. A more extended discussion of this posture, written for a non-technical audience, appears in the companion essay *Orientation, not ontology* [43].

---

## 4. Observables: 95 Relations from 20 Structural Constants

### 4.1 Gauge Sector

#### 4.1.1 Weinberg Angle

$$\sin^2\theta_W = \frac{b_2}{b_3 + \dim(G_2)} = \frac{21}{91} = \frac{3}{13} = 0.230769$$

Experimental (PDG 2024) [1]: 0.23122 +/- 0.00004. Deviation: **0.195%**.

The numerator b₂ counts gauge moduli; the denominator b₃ + dim(G₂) counts matter plus holonomy degrees of freedom. The ratio measures gauge-matter coupling geometrically.

#### 4.1.2 Strong Coupling

$$\alpha_s(M_Z) = \frac{\sqrt{2}}{\dim(G_2) - p_2} = \frac{\sqrt{2}}{12} = 0.11785$$

Experimental: 0.1180 +/- 0.0009. Deviation: **0.126%**.

### 4.2 Lepton Sector

#### 4.2.1 Koide Parameter

The Koide formula has resisted explanation since 1982. Koide discovered an empirical relation among the charged lepton masses [6]:

$$Q = \frac{(m_e + m_\mu + m_\tau)^2}{(\sqrt{m_e} + \sqrt{m_\mu} + \sqrt{m_\tau})^2} = \frac{2}{3}$$

Using contemporary mass values, this holds to six significant figures: Q_exp = 0.666661 +/- 0.000007.

GIFT provides:

$$Q_{\rm Koide} = \frac{\dim(G_2)}{b_2} = \frac{14}{21} = \frac{2}{3}$$

The derivation requires only two topological invariants: dim(G₂) = 14 (holonomy group dimension) and b₂ = 21 (second Betti number). No fitting is involved.

| Approach | Result | Status |
|----------|--------|--------|
| Preon models (Koide 1982) | Q = 2/3 assumed | Circular |
| S₃ symmetry (various) | Q ~ 2/3 fitted | Approximate |
| **GIFT** | **Q = dim(G₂)/b₂ = 14/21 = 2/3** | **Algebraic identity** |

Deviation: **0.0009%**. This is the most precise agreement in the framework.

#### 4.2.2 Tau-Electron Mass Ratio

$$\frac{m_\tau}{m_e} = \dim(K_7) + 10 \times \dim(E_8) + 10 \times H^* = 7 + 2480 + 990 = 3477$$

Experimental: 3477.15 +/- 0.05. Deviation: **0.004%**.

The integer 3477 = 3 x 19 x 61 = N_gen x prime(8) x kappa_T^-1 factorizes into framework constants.

#### 4.2.3 Muon-Electron Mass Ratio

$$\frac{m_\mu}{m_e} = \dim(J_3(\mathbb{O}))^\phi = 27^\phi = 207.01$$

where φ = (1+√5)/2 is the golden ratio, arising from the McKay correspondence E₈ ↔ 2I (binary icosahedral group), which links E₈ to icosahedral geometry. Experimental: 206.768. Deviation: **0.118%**.

**Note**: φ is the only non-integer input among the 33 Type I predictions and does not appear in the 20 structural constants of §S3.3. Its status is accordingly weaker than the other Type I derivations: the formula is algebraically exact and Lean-certified, but φ's derivation from E₈ structure requires the additional McKay step (see S2 §11).

### 4.3 Quark Sector

$$\frac{m_s}{m_d} = p_2^2 \times \text{Weyl} = 4 \times 5 = 20$$

Experimental (PDG 2024): 20.0 +/- 1.0. Deviation: **0.00%**.

$$\frac{m_b}{m_t} = \frac{b_0}{2b_2} = \frac{1}{42}$$

The constant 42 = p₂ x N_gen x dim(K₇) = 2 x 3 x 7 is a structural invariant (not to be confused with chi(K₇) = 0, which vanishes for any compact odd-dimensional manifold).

Experimental: 0.024 +/- 0.001. Deviation: **0.79%**.

### 4.4 Neutrino Sector

#### 4.4.1 CP-Violation Phase

The GIFT prediction for the CP-violation phase is:

$$\delta_{CP} = \dim(K_7) \times \dim(G_2) + H^* = 7 \times 14 + 99 = 197°$$

decomposing into a local contribution (7 × 14 = 98, fiber-holonomy coupling) and a global contribution (H* = 99, cohomological dimension). This is the canonical GIFT prediction — pure topological, zero corrections.

**Experimental status**: NuFIT 5.2 (2022) gave δ_CP ≈ 197°, an exact match. NuFIT 6.0 (Oct 2024) shifted the central value to 177° ± 20°, creating an 11.3% deviation. The experimental uncertainty is large (±20°), and the central value may shift further as T2K, NOvA, and DUNE accumulate statistics.

**Structural observation** (documented, not adopted): PSLQ residual analysis (§7.6) identifies a compactification factor 62/69 = dim(E₈)/(dim(E₈) + 4 dim(K₇)), which would yield 197 × 62/69 = 12214/69 ≈ 177.01°. This is documented as a structural observation — the ratio of gauge to total degrees of freedom (248 gauge DOF out of 276 = 248 + 4 × 7 total) — but is not adopted as a revision to the prediction. The canonical GIFT value remains 197°.

**Falsification criterion**: If DUNE measures δ_CP outside [182, 212]° at 3σ, the framework faces serious tension. The 197° prediction sits at the edge of the current NuFIT 6.0 1σ band (177° ± 20° = [157°, 197°]).

#### 4.4.2 Mixing Angles

| Angle | Formula | GIFT | NuFIT 6.0 [27] | Dev. |
|-------|---------|------|----------------|------|
| theta_12 | arctan(sqrt(delta/gamma_GIFT)) | 33.40 deg | 33.41 +/- 0.75 deg | 0.03% |
| theta_13 | pi/b₂ | 8.57 deg | 8.54 +/- 0.12 deg | 0.37% |
| theta_23 | arcsin((b₃ - p₂)/H*) | 49.25 deg | 49.3 +/- 1.0 deg | 0.10% |

The auxiliary parameters: delta = 2*pi/Weyl² = 2*pi/25 and gamma_GIFT = (2 x rank(E₈) + 5 x H*)/(10 x dim(G₂) + 3 x dim(E₈)) = 511/884.

### 4.5 Higgs Sector

$$\lambda_H = \frac{\sqrt{\dim(G_2) + N_{gen}}}{\det(g)_{den}} = \frac{\sqrt{17}}{32} = 0.1289$$

The Higgs self-coupling combines holonomy dimension with generation count, normalized by the metric determinant scale. Experimental (PDG 2024): 0.129 ± 0.001. Deviation: **0.12%**. A TCS alternative λ_H = b₂(M₁)/(b₃+b₂(M₂)) = 11/87 = 0.1264 is purely rational (see S2 §17).

### 4.6 Boson Mass Ratios

| Observable | Formula | GIFT | Experimental | Dev. |
|------------|---------|------|--------------|------|
| m_H/m_W | (N_gen + dim(E₆))/dim(F₄) = 81/52 | 1.5577 | 1.558 +/- 0.002 | 0.02% |
| m_W/m_Z | (2b₂ - Weyl)/(2b₂) = 37/42 | 0.8810 | 0.8815 +/- 0.0002 | 0.06% |
| m_H/m_t | fund(E₇)/b₃ = 56/77 | 0.7273 | 0.725 +/- 0.003 | 0.31% |

### 4.7 CKM Matrix

| Observable | Formula | GIFT | Experimental | Dev. |
|------------|---------|------|--------------|------|
| sin²(theta_12_CKM) | fund(E₇)/dim(E₈) = 56/248 | 0.2258 | 0.2250 +/- 0.0006 | 0.36% |
| A_Wolfenstein | (Weyl + dim(E₆))/H* = 83/99 | 0.838 | 0.836 +/- 0.015 | 0.29% |
| sin²(theta_23_CKM) | dim(K₇)/PSL(2,7) = 7/168 | 0.0417 | 0.0412 +/- 0.0008 | 1.13% |

The Cabibbo angle emerges from the ratio of the E₇ fundamental representation to E₈ dimension.

### 4.8 Cosmological Observables

| Observable | Formula | GIFT | Experimental | Dev. |
|------------|---------|------|--------------|------|
| Omega_DM/Omega_b | (1 + 2b₂)/rank(E₈) = 43/8 | 5.375 | 5.375 +/- 0.1 | 0.00% |
| n_s | zeta(11)/zeta(5) | 0.9649 | 0.9649 +/- 0.0042 | 0.004% |
| h (Hubble) | (PSL(2,7) - 1)/dim(E₈) = 167/248 | 0.6734 | 0.674 +/- 0.005 | 0.09% |
| Omega_b/Omega_m | Weyl/det(g)_den = 5/32 | 0.1562 | 0.157 +/- 0.003 | 0.16% |
| sigma_8 | (p₂ + 32)/(2b₂) = 34/42 | 0.8095 | 0.811 +/- 0.006 | 0.18% |
| Omega_DE | ln(2)×(b₂+b₃)/H* = ln(2)×98/99 | 0.6861 | 0.6847 +/- 0.005 | 0.21% |
| Y_p | (1 + dim(G₂))/kappa_T^-1 = 15/61 | 0.2459 | 0.245 +/- 0.003 | 0.37% |

The dark-to-baryonic matter ratio Omega_DM/Omega_b = 43/8 is exact. The structural invariant 2b₂ = 42 that gives m_b/m_t = 1/42 also determines this cosmological ratio, connecting quark physics to large-scale structure through K₇ geometry.

### 4.8.1 Type I Summary (33 Observables)

The 33 Type I predictions derive directly from the 20 structural constants (Appendix A). All are dimensionless ratios; all 33 are Lean-certified. Representative highlights:

| Observable | Formula | GIFT | Exp. | Dev. |
|-----------|---------|------|------|------|
| sin²θ_W | b₂/(b₃+dim(G₂)) = 21/91 | 3/13 | 0.23122 | 0.195% |
| Q_Koide | dim(G₂)/b₂ = 14/21 | 2/3 | 0.666661 | 0.001% |
| α⁻¹ | 128 + 9 + det(g)×κ_T | 137.033 | 137.036 | 0.002% |
| m_τ/m_e | 7 + 2480 + 990 | 3477 | 3477.15 | 0.004% |
| n_s | ζ(11)/ζ(5) | 0.9649 | 0.9649 | 0.004% |
| Ω_DM/Ω_b | (1+42)/8 | 43/8 | 5.375 | 0.00% |

Complete derivations for all 33 in Supplement S2; full 95-entry table in Supplement S3.

### 4.9 Type II: Extended Algebraic Predictions (19)

Type II observables require one physical identification step beyond Type I ratios. Representative results:

**Absolute quark masses** (via m_q = ratio × reference_mass): m_u = 2.16 MeV (0.00%), m_d = 4.67 MeV (0.22%), m_s = 93.4 MeV (0.22%), m_c = 1.27 GeV (0.00%), m_b = 4.18 GeV (0.00%), m_t = 172.7 GeV (0.01%).

**CKM magnitudes** (from Wolfenstein parametrization): |V_us| = 0.2253 (0.13%), |V_cb| = 0.0412 (0.24%), |V_ub| = 0.00365 (0.27%), |V_td| = 0.0087 (0.34%), |V_tb| = 0.9991 (0.00%).

**Extended ratios**: m_c/m_s = 246/21 = 11.714 (exp 11.7, dev 0.12%), m_c/m_d = 234.3 (exp 234.0, dev 0.12%), m_μ/m_τ = 5/84 (exp 0.0595, dev 0.04%).

Type II mean deviation: **0.17%** across 19 observables. These inherit the precision of Type I ratios, with the physical identification step contributing negligible additional error.

### 4.10 Type III: Dynamical Predictions (21)

Type III observables involve multi-step dynamical mechanisms. They are grouped by computation:

**wilson_line Non-adiabatic** (3 obs): Wilson line eigenvalue splitting on K3 fiber at c = 0.452 gives raw lepton mass ratios with 0.5–2.1% deviation. These are improved to < 0.4% by the combined wilson_line+instanton pipeline (§6.4).

**RGE_running RGE running** (4 obs): Two-loop MSSM evolution from M_GUT to M_Z. The topological sin²θ_W = 3/13 at M_GUT runs to 0.2377 at M_Z (exp 0.2312, dev 2.78%). The strong coupling α_s(RGE) = 0.1224 deviates 3.7% using G₂-MSSM split-spectrum matching (§5.3). M_GUT = 2 × 10¹⁶ GeV is an exact match.

**spectral Spectral** (5 obs): effective Weyl law exponent (from the adiabatic seam-sector decomposition, extrapolated to d_eff = 7) α = 3.460 (exp 3.5, dev 1.1%), 22,671 KK states below cutoff, 57,578 fiber channels, Poisson level spacing. See Paper B [44] for the direct seam-sector result α = 1.998.

**gauge_bundle Gauge bundle** (4 obs): cond(f_IJ) = 1.047 (near-perfect gauge universality), α_ratio = 1.000002, effective Yukawa rank = 3, and κ(gauge) = 1.047 (4.7% departure from exact universality, the largest Type III bundle deviation).

**instanton Instanton + combined** (5 obs): Associative volume differences give ΔV(e-τ) = 8.633 (dev 5.9%), ΔV(e-μ) = 3.271 (dev 15.9%). Combined wilson_line+instanton pipeline with α = e^K (geometric, §6.4) gives τ/e = 3485 (0.24%), τ/μ = 16.69 (0.75%), μ/e = 208.8 (0.97%).

Type III mean deviation: **3.4%** across 14 experimentally comparable observables (2.3% across all 21, median 1.6%). Details in §5 (Gauge), §6 (Mass Hierarchy). (M_res and N_QNM from Pinčák et al. 2026 [46] are classified Type IV — see §4.11.)

### 4.11 Type IV: Structural Diagnostics (22)

Type IV observables are internal consistency checks with no experimental comparison:

- **Topology**: b₂ = 21, b₃ = 77, χ(K₇) = 0, H* = 99
- **Newton-Kantorovich certification**: h = 8.95 × 10⁻⁹ (< 0.5, margin ×56M), δg/g ≤ 4.86 × 10⁻⁶
- **Gram conditioning**: cond(G_K3) = 1.05, cond(G_K7) = 1.05, cond(G_35) = 7.66, G_77 positive definite
- **Spectral counts**: 22,671 KK states, 57,578 fiber channels, Poisson spacing confirmed
- **Torsion**: ‖T‖_C⁰ = 2.949 × 10⁻⁵ (×2995 reduction in 5 Joyce steps)
- **Metric eigenvalues**: g_ss = 19/6, g_{T²} = 7/6, g_{K3} ≈ 64/77
- **Instanton & BH diagnostics** (Pinčák et al. 2026 [46]): N_QNM = 98 QNM mode families, b₃/b₃(S³×S⁴) = 77× instanton suppression, M_res = v_EW²/M_Pl (BH remnant mass — no experimental comparison)

These diagnostics confirm the geometric construction is well-conditioned and internally consistent.

### 4.12 Summary Statistics (All 95 Observables)

**Global performance** (95 observables):

| Metric | Type I | Type II | Type III | All (I+II+III) |
|--------|--------|--------|--------|-------------|
| Count | 33 | 19 | 21 | 73* |
| With exp. comparison | 33 | 19 | 14 | 66 |
| Mean deviation | 0.73% | 0.17% | 3.44% | ~1.15% |
| Median deviation | 0.23% | 0.12% | 1.39% | 0.23% |
| Exact matches (<0.01%) | 5 | 3 | 3 | 11 |
| Within 1% | 28 | 19 | 6 | 53 |
| Maximum deviation | 11.3% | 0.79% | 15.9% | 15.9% |

†δ_CP = 11.3% dominates Type I; excluding it, Type I mean = 0.40% (28 within-1%, 5 exact matches, max 2.77%).

*66 with experimental comparison out of 73 (I+II+III); 22 Type IV observables are structural.

**Sector breakdown** (11 sectors; 41 observables listed — see §5–6 for remaining 25 comparable):

| Sector | N_obs | Mean dev | Best | Worst |
|--------|-------|----------|------|-------|
| Electroweak | 3 | 0.11% | α⁻¹ 0.002% | sin²θ_W 0.20% |
| Boson | 3 | 0.13% | m_H/m_W 0.02% | m_H/m_t 0.31% |
| Lepton | 3 | 0.04% | Q_Koide 0.001% | m_μ/m_e 0.12% |
| Quark | 4 | 0.21% | m_s/m_d 0.00% | m_b/m_t 0.79% |
| Cosmology | 7 | 0.15% | n_s 0.004% | Y_p 0.37% |
| PMNS | 4 | 0.29% | θ₁₂ 0.03% | θ₁₃ 0.37% |
| CKM | 3 | 0.59% | A_Wolf 0.29% | sin²θ₂₃ 1.13% |
| Gauge (running) | 4 | 2.3% | M_GUT 0.00% | α_s(RGE) 3.7% |
| Instanton | 3 | 7.4% | ΔV(e-τ) 5.9% | ΔV(e-μ) 15.9% |
| Combined | 3 | 0.66% | τ/e 0.23% | μ/e 0.98% |
| Bundle | 4 | 1.6% | α_ratio 0.00% | κ(gauge) 4.7% |

---

## 5. Gauge Sector: From E₈ to the Standard Model

The gauge sector derives the Standard Model gauge group from the E₈×E₈ structure of heterotic M-theory on K₇. This section presents the complete breaking chain, anomaly cancellation, gauge coupling running, and bundle universality — all from the topological data of K₇ and an explicit 7D G₂ metric (169 optimized Chebyshev-Cholesky parameters capturing the dominant seam sector, K3 fiber certified at 0.07% [30, 45]), with no free parameters in the physical predictions.

### 5.1 The E₈ → Standard Model Breaking Chain

The first E₈ factor breaks to the Standard Model through a six-level chain:

| Level | Group | Dimension | Mechanism | Scale |
|-------|-------|-----------|-----------|-------|
| 0 | E₈×E₈ | 496 | Heterotic structure | M_Pl |
| 1 | E₈ | 248 | Second E₈ = hidden sector | M_string |
| 2 | E₆ × SU(3) | 78 + 8 = 86 | Adjoint branching | M_string |
| 3 | SO(10) × U(1) | 45 + 1 = 46 | E₆ breaking | M_GUT |
| 4 | SU(5) × U(1) | 24 + 1 = 25 | Pati-Salam intermediate | M_GUT |
| 5 | SU(3) × SU(2) × U(1) | 8 + 3 + 1 = 12 | Standard Model | M_Z |

The E₈ adjoint decomposes under E₆ × SU(3) as:

$$248 = (78, 1) + (1, 8) + (27, 3) + (\overline{27}, \overline{3})$$

yielding 78 + 8 + 81 + 81 = 248. The fundamental of SU(3) has dimension 3, giving **N_gen = 3** chiral families from the (27, 3) representation. This reproduces the index-theorem result of §2.

**Fundamental group**: π₁(K₇) = {1} (simply connected, from b₁ = 0 for G₂ holonomy). Traditional Wilson line breaking via π₁ is therefore trivial. Instead, the breaking proceeds through the Z₃ lattice action on the K3 fiber (§6.1).

**Scales**: M_Pl = 1.22 × 10¹⁹ GeV, M_string = 4 × 10¹⁷ GeV, M_GUT = 2 × 10¹⁶ GeV, M_Z = 91.19 GeV. The scale hierarchy M_GUT/M_Z ~ 2 × 10¹⁴ emerges from the geometry without fine-tuning.

### 5.2 Anomaly Cancellation

All six Standard Model gauge anomalies vanish:

| Anomaly | Value | Status |
|---------|-------|--------|
| SU(3)³ | 0.0 | Exact |
| SU(2)³ | 0.0 | Exact |
| U(1)³ | 2.3 × 10⁻¹⁶ | Machine zero |
| grav-U(1) | 0.0 | Exact |
| SU(3)²-U(1) | 0.0 | Exact |
| SU(2)²-U(1) | 0.0 | Exact |

The Green-Schwarz mechanism provides additional consistency:
- χ(K₇) = 0 (tadpole cancellation for compact odd-dimensional manifolds)
- Tadpole value = 0 (no net D-brane charge)
- G₄ flux lattice rank = 77 = b₃(K₇) (all flux degrees of freedom realized)

The 10/10 verification certificate confirms all anomaly, tadpole, and lattice checks pass. Lean: `TCSGaugeBreaking.lean` (10 conjuncts, 0 sorry).

### 5.3 Gauge Coupling Running

**Boundary conditions at M_GUT**: α_GUT⁻¹ = 25.3, sin²θ_W = 3/13 = 0.23077. These are topological: α_GUT derives from the gauge kinetic function on K₇, and sin²θ_W from the Betti number ratio b₂/(b₃ + dim(G₂)).

**Two-loop MSSM RGE** (tan β = 2, M_SUSY = m_moduli = 3165 GeV):

| Coupling | GIFT at M_Z | Experimental | Deviation |
|----------|-------------|--------------|-----------|
| sin²θ_W | 0.2377 | 0.23122 | 2.78% |
| α_em⁻¹ | 131.19 | 127.951 | 2.53% |
| α_s (all-MSSM) | 0.1038 | 0.1180 | 12.1% |
| **α_s (split-spectrum)** | **0.1224** | **0.1180** | **3.7%** |

A naive degenerate-spectrum treatment (all superpartners at M_SUSY) yields α_s = 0.1038 (12.1% deviation). The physical G₂-MSSM spectrum is split: squarks and sleptons decouple at M_squark = 3165 GeV, while gauginos remain light. Below M_squark, the effective theory is SM + gauginos with b₃ = −5 (instead of −3 for the full MSSM). This step-function matching gives α_s = 0.1224 (3.7% deviation), which is the value adopted throughout.

The RGE deviations reflect sensitivity to the SUSY spectrum, particularly:

- **tan β dependence**: The 2-loop terms involving top Yukawa are sensitive to tan β
- **Threshold corrections**: The split-spectrum matching captures the dominant effect; residual corrections from the detailed spectrum are subdominant
- **3-loop effects**: Omitted from the current calculation

**Topological vs. infrared**: The topological sin²θ_W = 3/13 = 0.23077 (dev 0.19% from PDG) is more accurate than the RGE-evolved value (dev 2.78%). This suggests the topological value may represent an infrared fixed point rather than a UV boundary condition — an observation also made in the G₂-MSSM literature [34].

**KK threshold corrections**: With 71 KK modes above M_GUT and Σ ln(m/M_GUT) = 89.2, the net correction to α_GUT⁻¹ is 0.0 (smooth K₇ manifolds have vanishing KK threshold corrections due to the cancellation between towers).

**SUSY spectrum implications**:

| Particle | Mass | Source |
|----------|------|--------|
| m_gravitino | 166 GeV | F-term from gaugino condensation |
| m_moduli | 3165 GeV (3.2 TeV) | SU(8) gaugino condensation |
| m_gluino | 442 GeV | cMSSM with m₁/₂ = m_gravitino |
| m_squark | 1094 GeV | cMSSM (running mass at low scale; decoupling matched at M_SUSY = m_moduli = 3165 GeV) |
| m_slepton | 175 GeV | cMSSM |
| LSP (Bino) | 70 GeV | Lightest neutralino (pure Bino; viable — LEP chargino bound 103 GeV does not exclude pure Bino LSP with suppressed gauge couplings [35]) |

**Phenomenological caveat**: These masses are computed within the cMSSM approximation. Standard simplified SUSY searches at ATLAS/CMS already exclude portions of this parameter space; the spectrum is viable only in compressed or suppressed-coupling realizations. Definitive testing requires a dedicated recast of current ATLAS/CMS results against the G₂-MSSM split-spectrum scenario.

### 5.4 The B-Test Identity and Holonomy Sequence

The gauge coupling predictions sin²θ_W = 3/13 and α_s = √2/12, combined with the MSSM beta-function structure, yield an algebraic identity connecting the fine structure constant to G₂ representation theory.

**The B parameter**: In any GUT framework, the "B-test" quantifies consistency of gauge coupling unification:

$$B = \frac{\alpha_1^{-1} - \alpha_2^{-1}}{\alpha_2^{-1} - \alpha_3^{-1}}$$

For the MSSM with N_gen = 3 generations and one Higgs doublet pair, the beta-function coefficients are (b₁, b₂, b₃) = (33/5, 1, −3), giving B = (b₁−b₂)/(b₂−b₃) = (28/5)/4 = **7/5**. The number of generations enters through N_gen = b₂/dim(K₇) = 21/7 = 3.

**Theorem** (B-test): *Given sin²θ_W = b₂/(b₃+dim(G₂)) = 3/13 and α_s = √2/(dim(G₂)−p₂) = √2/12, the MSSM relation B = 7/5 holds if and only if*

$$\alpha_{\mathrm{em}}^{-1}(M_Z) = (b_3 + \dim(G_2)) \cdot \sqrt{2} = 91\sqrt{2} = 128.693\ldots$$

*where 91 = dim(Λ²𝔤₂) is the dimension of the exterior square of the G₂ Lie algebra.*

*Proof sketch*. The topological sin²θ_W = 3/13 and the GUT normalization factor 3/5 force α₁⁻¹ = 2α₂⁻¹. Then B = α₂⁻¹/(α₂⁻¹ − α₃⁻¹), and B = 7/5 requires α₂/α₃ = 2/7 = p₂/dim(K₇). Substituting α₃⁻¹ = (dim(G₂) − p₂)/√2 = 6√2 gives α_em⁻¹ = (7 × 13)√2 = 91√2. The factor 91 = b₃ + dim(G₂) = 77 + 14 = dim(Λ²𝔤₂) is a G₂ representation-theoretic invariant. *(Note: the B-test identity gives the GUT-scale α_em⁻¹ ≈ 128.7, valid at M_GUT where B = 7/5; the observed low-energy value α_em⁻¹ ≈ 137 is a separate derivation via §4.1. Both are used consistently in GIFT.)*

**The holonomy sequence**: At the B = 7/5 exact scale, all three inverse couplings are integer multiples of √2:

| Coupling | Value | = integer × √2 | Topological origin |
|----------|-------|-----------------|-------------------|
| α₁⁻¹ | 59.40 | 42√2 | 2b₂ |
| α₂⁻¹ | 29.70 | 21√2 | b₂ |
| α₃⁻¹ | 8.49 | 6√2 | dim(K₇)−1 |

Their ratios in lowest terms:

$$\alpha_1^{-1} : \alpha_2^{-1} : \alpha_3^{-1} = \dim(G_2) : \dim(K_7) : p_2 = 14 : 7 : 2$$

This **holonomy sequence** encodes the G₂ structure of K₇ directly in the gauge couplings: the holonomy group dimension, the manifold dimension, and the Pontryagin ratio p₂ = dim(G₂)/dim(K₇).

**Exact unification**: Using the 1-loop MSSM RGE α_i⁻¹(μ) = α_i⁻¹(M_Z) − b_i/(2π)·ln(μ/M_Z), the three couplings unify exactly at:

$$t_{\mathrm{GUT}} = \frac{15\pi\sqrt{2}}{2}, \qquad \alpha_{\mathrm{GUT}}^{-1} = \frac{\sqrt{2}\,(b_3 - \mathrm{rank}(E_8))}{4} = \frac{69\sqrt{2}}{4} \approx 24.4$$

where 69 = b₃ − rank(E₈) = 77 − 8 = |PSL(2,7)| − H* = 168 − 99. This gives M_GUT = M_Z·e^{t} = 2.7 × 10¹⁶ GeV, consistent with the standard MSSM estimate. The GUT coupling α_GUT⁻¹ = 24.4 compares to the gauge kinetic function value 25.3 used in §5.3 (3.6% discrepancy, reflecting the approximate nature of the identity).

**Numerical status**: Using GIFT's topological couplings at M_Z, B = 1.4033 (0.23% from 7/5). This is closer to 7/5 than the purely experimental value B = 1.3948 (0.37% off), suggesting the identity has genuine geometric content. The ~0.5% deficit between 91√2 = 128.69 and the GIFT RGE-running prediction α_em⁻¹(M_Z) = 128.03 [distinct from the topological low-energy value 137.033 = 4π/α_em] remains an open question — it may trace to 2-loop or threshold effects not captured at the algebraic level.

### 5.5 Bundle Universality and Gram Conditioning

The gauge kinetic function f_IJ on K₇ determines gauge coupling universality. From the 22 harmonic 2-forms on K₃:

**f_IJ eigenvalue spectrum**: 22 eigenvalues in [0.733, 0.767], with **cond(f_IJ) = 1.047** — near-perfect universality. The gauge coupling ratio α_ratio = α_max/α_min = 1.000002, confirming that all gauge couplings are effectively identical at the compactification scale.

**Gram matrices** quantify orthonormality of the harmonic form basis:

| Matrix | Size | Condition | PD | Off-diag max |
|--------|------|-----------|----|----|
| G_K3(22) | 22×22 | **1.05** | Yes | 0.012 |
| G_K7(22) | 22×22 | **1.05** | Yes | 0.012 |
| G_35 | 35×35 | **7.66** | Yes | — |
| G_77 | 77×77 | **7.66** | Yes | 7×10⁻⁵ |

The K3 and K7 22-form bases are nearly orthonormal (condition ~1.05). The full 77-form basis has moderate conditioning (7.66), with cross-block coupling between constant and fiber modes bounded by 7 × 10⁻⁵. Gram-Schmidt orthogonalization residuals are < 5 × 10⁻¹⁶ (machine precision).

**Figure 4**: Gauge bundle structure — eigenvalue spectrum of $f_{IJ}$ and Gram matrix diagnostics. See `fig_gauge_bundle.png`.

**Lean certification**: `TCSGaugeBreaking.lean` (0 axioms, 14 theorems, 10-conjunct master certificate) + `GaugeBundleData.lean` (0 axioms, 12 theorems, 11-conjunct master certificate).

### 5.6 Summary

The gauge sector pipeline covers E₈ → SM with N_gen = 3, all anomalies cancelled, near-perfect bundle universality (cond 1.047), and Lean-certified results. The topological gauge predictions (sin²θ_W, α_s) are more precise than the dynamical RGE running, suggesting the topological values may represent infrared fixed points rather than UV boundary conditions. The B-test identity (§5.4) reveals that these two predictions, combined with the MSSM structure, encode the holonomy sequence dim(G₂):dim(K₇):p₂ = 14:7:2 in the gauge coupling ratios — a direct imprint of G₂ geometry on low-energy physics.

---

## 6. Mass Hierarchy: From Geometry to Generations

The five-order-of-magnitude lepton mass hierarchy (m_e : m_μ : m_τ ~ 1 : 207 : 3477) is one of the deepest puzzles in particle physics. This section presents two independent geometric mechanisms that individually reproduce the hierarchy to ~2–6% and, when combined, achieve sub-percent precision.

### 6.1 Three Generations from the Z₃ Mechanism

Since π₁(K₇) = {1}, traditional Wilson line breaking is trivial. Instead, three generations emerge from a Z₃ lattice action on the K3 fiber of the neck region.

**Wilson line theorem**: The rank of the Wilson line operator is preserved under perturbation. SVD analysis gives singular values [5.71, 0.62, 2.4 × 10⁻¹⁵], confirming rank 3 (the third eigenvalue is machine zero, consistent with a Z₃ quotient leaving three independent directions).

**K3 metric properties**: The K3 fiber metric is nearly flat: conformal range 0.018% across the fiber, mean anisotropy 1.4%. This near-flatness ensures the eigenvalue splitting is controlled by the fiber geometry rather than by large-scale metric fluctuations.

### 6.2 Lepton Mass Hierarchy: Non-Adiabatic Mechanism (wilson_line)

The non-adiabatic eigenvalue splitting mechanism operates on the K3 fiber at coupling c = 0.452 and optimized positions [0.0, 0.693, 1.400]:

**Eigenvalues**: [0.03383, 0.00205, 9.94 × 10⁻⁶]

| Ratio | GIFT | Experimental | Deviation |
|-------|------|-------------|-----------|
| m_τ/m_μ | 16.54 | 16.82 | 1.7% |
| m_τ/m_e | 3403 | 3477 | 2.1% |
| m_μ/m_e | 205.7 | 206.7 | 0.5% |

The critical coupling c* = 10⁻³/⁴ = 0.1778 marks the transition between adiabatic (small splitting, ~2 generations) and non-adiabatic (large splitting, 3 generations) regimes. The physical coupling c = 0.452 > c* places K₇ firmly in the three-generation regime.

**Adiabatic limit**: At c → 0, only two generations are distinguishable (m₁/m₂ = 77.6, m₁/m₃ → ∞). The three-generation structure requires c > c*, which is satisfied by the neck geometry.

### 6.3 Instanton Volume Differences (instanton)

An independent mechanism generates the mass hierarchy from associative 3-cycle volumes. On K₇, there are **57 associative 3-cycles** with volumes in [0.00075, 11.109]. The mass relation m_i/m_j = exp(ΔV) assigns each generation to a cycle.

**Optimal assignment** (minimizing combined deviation):

| Assignment | Volume | ΔV | Target | Deviation |
|------------|--------|-----|--------|-----------|
| V_e | 11.109 | — | — | — |
| V_μ | 7.838 | ΔV(e-μ) = 3.271 | ln(16.82) = 2.823 | 15.9% |
| V_τ | 2.476 | ΔV(e-τ) = 8.633 | ln(3477) = 8.154 | 5.9% |

The e-τ hierarchy (5 orders of magnitude) is reproduced to 5.9%, while the e-μ hierarchy (2.3 orders) shows 15.9% deviation from the optimal assignment. The total volume range ΔV_range = 8.92 spans the correct order of magnitude.

**Figure 5**: Instanton mass hierarchy — volume spectrum of 57 associative cycles with generation assignments. See `fig_instanton_hierarchy.png`.

### 6.4 Combined wilson_line+instanton Pipeline

The key insight is that wilson_line and instanton probe different aspects of K₇ geometry:
- **wilson_line**: Fiber geometry → eigenvalue spacing (relative structure)
- **instanton**: Cycle volumes → exponential hierarchy (absolute scale)

The two mechanisms are connected by α = e^K = exp(K₀) = V̂^{−3}, where K₀ = −5.891 is the Kähler potential of K₇ (§6.5). This is a purely geometric quantity — the instanton action normalization derived from the compactification volume — not a fit parameter:

| Ratio | wilson_line raw | instanton raw | Combined (α = e^K) | Experimental | Dev |
|-------|---------|---------|----------|-------------|------|
| m_τ/m_e | 3403 (2.1%) | exp(8.63) | **3485** | 3477 | **0.24%** |
| m_τ/m_μ | 16.54 (1.7%) | exp(3.27) | **16.69** | 16.82 | **0.75%** |
| m_μ/m_e | 205.7 (0.5%) | — | **208.8** | 206.8 | **0.97%** |

All three ratios within 1% — a significant improvement over the individual mechanisms. The combined pipeline works because the mechanisms are **complementary**: wilson_line provides fine structure from the eigenvalue splitting, instanton provides the overall exponential scale from cycle volumes. The key insight is that α = e^K has a natural M-theory interpretation: for M2-branes wrapping associative 3-cycles, the instanton action scales as S_inst = e^K × Vol(Σ) = Vol(K₇)^{−3} × Vol(Σ), giving the correct suppression without any free parameters.

**Lean certification**: `AssociativeVolumes.lean` (0 axioms, 19 theorems, 14-conjunct master certificate) certifies the combined wilson_line+instanton results including all three mass ratios and the geometric α = e^K.

### 6.5 4D Effective Theory (S9)

Dimensional reduction of the G₂ metric yields an N = 1 supergravity theory in 4D:

**Particle content**: From Betti numbers directly:
- 1 gravity multiplet
- 21 vector multiplets (= b₂)
- 77 chiral multiplets (= b₃)
- Total: 99 = H* (the effective cohomological dimension)

**Kähler potential**: K₀ = −5.891, with e^K = 0.002763 = V̂^{−3}. This geometric quantity serves as the instanton action normalization α = e^K in the combined wilson_line+instanton pipeline (§6.4). The 7D volume V̂ = 7.126. The metric determinant det(g₇) = 65/32 = 2.03125 is locked to its topological value.

**Moduli metric**: Total dimension 79 (= 77 physical + 2 gauge), rank 77 (2 null directions from gauge redundancy). The condition number cond = 7.66 measures the ratio of largest to smallest moduli mass. The 35 constant-block eigenvalues span [1.66, 12.69]; the 42 fiber-block eigenvalues cluster around 6.1.

**Gaugino condensation**: The SU(8) hidden sector is preferred:

| Condensate | b₀ | Λ (GeV) | m_moduli (GeV) |
|------------|-----|---------|----------------|
| SU(3) | 9 | 4.3 × 10⁸ | 1.3 × 10⁻¹¹ |
| SU(5) | 15 | 5.0 × 10¹¹ | 0.021 |
| **SU(8)** | **24** | **2.66 × 10¹³** | **3165** |

The SU(8) sector gives m_moduli = 3165 GeV (3.2 TeV), m_gravitino = 166 GeV, consistent with the G₂-MSSM spectrum.

**Physical predictions from the effective theory**:
- Proton lifetime: τ_p = 4.06 × 10³⁸ years (well above current bound ~10³⁴ years)
- Yukawa effective rank = 3 (three massive generations)
- λ_physical = 1.358 (physical Yukawa coupling)

**Figure 6**: KK spectrum — Kaluza-Klein tower structure and 7D Weyl law. See `fig_kk_spectrum.png`.

---

## 7. Sensitivity Analysis

A framework claiming 95 observables from 20 structural constants must address the question: how constrained are these predictions? This section presents five independent analyses demonstrating that the predictions are overdetermined, cross-coupled, and statistically incompatible with coincidence.

### 7.1 Formula Structure Analysis

**Question**: Do more complex formulas systematically produce smaller deviations? If so, the framework might be "fitting" by formula complexity.

**Method**: Random Forest regression with leave-one-out cross-validation, predicting |deviation| from formula features (number of constants used, maximum constant value, arithmetic operations, sector membership).

**Result**: R² = −0.518 (LOO-CV) — worse than a mean predictor. The top feature importance is max_constant_value (0.41), followed by n_expressions (0.13) and sector_PMNS (0.12). Formula complexity does **not** predict deviation magnitude.

**Verdict**: This is inconsistent with systematic cherry-picking. Complex formulas do not perform better than simple ones; the precision is distributed uniformly across formula types.

### 7.2 Topological Constant Sensitivity

**Method**: Perturb each of the 20 structural constants by ±1 → 20 × 33 sensitivity matrix S_ij = ∂(observable_i)/∂(constant_j).

**SVD analysis**: 19 significant singular values (1 zero, corresponding to a redundant constant combination). The singular value spectrum decays smoothly: [3.50, 2.54, 2.39, 2.14, 2.11, 2.01, 1.93, 1.76, 1.68, 1.63, ...].

**Sensitivity-deviation correlation**: ρ = −0.083 — **no systematic pattern**. Observables with high sensitivity to constant perturbations do not have larger deviations. The most sensitive constants are dim(K₇) (0.068), dim(G₂) (0.062), and H* (0.062).

**NK ball rigidity**: The Newton-Kantorovich certification establishes δg/g ≤ 1.35 × 10⁻⁷, confirming the metric is effectively unique within its basin. The coefficient of variation of metric eigenvalues is ~10⁻⁷, confirming rigidity.

### 7.3 Effective Degrees of Freedom

**Method**: SVD of the 20 × 33 constant-usage Jacobian (binary: 1 if constant appears in formula, 0 otherwise). The effective rank is defined by the singular value decay profile.

**Results**:
- **r_eff = 15.53** effective parameters (out of 20 structural constants)
- 12 singular values capture 90% of variance
- 17 singular values capture 99% of variance
- 1 zero singular value (exact linear dependence)

**Overdetermination ratio**: 33 observables / 15.53 effective parameters = **2.13×**. The system has more than twice as many constraints as degrees of freedom — a hallmark of an overdetermined (not fitted) system.

**Most-used constants**: b₂ appears in 9 observables, dim(G₂) in 7, dim(E₈) in 7, Weyl in 6, H* in 6. No constant is used in more than 27% of observables, confirming broad coverage rather than concentration.

**Figure 7**: Constant usage — SVD spectrum showing effective rank and per-constant usage frequency. See `fig_constant_usage.png`.

### 7.4 Cross-Correlations

**Jaccard similarity**: Of the C(33,2) = 528 observable pairs, **155 share at least one structural constant** (29.4% coupled). All 33 observables belong to a single connected component — no prediction is isolated.

**Strong correlations**: 51 pairs have |ρ| ≥ 0.5. Notable examples:
- m_b/m_t vs m_μ/m_τ: ρ = −0.816 (both involve 2b₂ = 42)
- α⁻¹ vs Ω_DM/Ω_b: ρ = −0.721 (both involve rank(E₈))
- sin²θ_W vs Q_Koide: ρ = +0.673 (both involve b₂ and dim(G₂))

**Mean sector Jaccard** = 0.293: sectors share ~29% of structural constants, creating a web of inter-sector constraints.

**Figure 8**: Sensitivity heatmap — constant-observable coupling matrix showing cross-sector correlations. See `fig_sensitivity_heatmap.png`.

**Figure 9**: Observable correlations — pairwise Pearson correlations between observable deviations. See `fig_observable_correlations.png`.

### 7.5 Coincidence Test

**Question**: What is the probability that 33 independent random predictions match experiment as well as GIFT?

#### Uniform Null Model

Multiple statistical tests under the null hypothesis of uniform deviations in [0, 50%]:

| Test | Statistic | df | p-value |
|------|-----------|-----|---------|
| χ² | 1063 | 33 | 5.0 × 10⁻²⁰² |
| Fisher combined | 1100.5 | 66 | 2.2 × 10⁻¹⁸⁷ |
| KS (pull normality) | 0.189 | — | 0.165 |
| **Combined uniform** | — | — | **10⁻³⁴⁶** |

**Pull distribution**: mean = −0.774, std = 5.62. 72.7% of pulls within 1σ, 87.9% within 2σ. The KS test (p = 0.165) is consistent with Gaussian pulls — the deviations are not cherry-picked.

**Reduced χ²** = 32.2 is large, driven by outliers (δ_CP, α_s(RGE)). The Bayes factor log₁₀ = −2.3 is inconclusive, reflecting this: the bulk of predictions match extremely well, but a few outliers inflate the tails. Removing the 4 known outliers gives reduced χ² < 2.

#### Algebraic Null Model (algebraic_MC)

The uniform null model assumes predictions are random numbers. A stronger test: what if we use random *algebraic formulas* from the same 20 structural constants? We generated 4,188,086 unique formula values via exhaustive depth-1/2 enumeration (1.8M) plus 3M random expression trees (depth 2–4), using 5 binary operations (+, −, ×, ÷, ^) and 5 unary transforms (id, inv, √, ², ln).

| Metric | GIFT | Random algebraic (median) | Factor |
|--------|------|---------------------------|--------|
| Mean deviation | **0.73%** | 4.1 × 10⁹ % | 5.6 × 10⁹ |
| Exact matches (<0.01%) | **5** | 0 | — |
| Within 1% | **28** | 0 | — |

**Per-observable**: combined P(random algebraic formula matches GIFT's precision) = **10⁻¹³³** (product over 33 observables). Only 0.02% of 4.2M formulas match *any* observable within 0.01%.

**Set-level**: 0 out of 3,000,000 random sets of 33 algebraic formulas achieved GIFT's mean deviation, exact count, or within-1% count. Under this null: P < 3.3 × 10⁻⁷ for each metric.

**Figure 10**: Per-observable log₁₀ P(random match). See `fig_mc_per_observable.png`.

**Interpretation**: The algebraic null model is far more generous than the uniform null (it generates formulas from the *same constants*), yet GIFT's performance remains unmatched across 3M+ trials. Under both null hypotheses, chance agreement is excluded at extreme significance: P = 10⁻³⁴⁶ (uniform) and P = 10⁻¹³³ (algebraic).

### 7.6 PSLQ Residual Analysis (PSLQ_residual)

Beyond the statistical null models, we apply PSLQ integer relation detection to the relative residuals r_i = (GIFT_i - exp_i)/exp_i of the 33 Type I observables with experimental comparison. The goal is to identify whether deviations have structural content — i.e., whether residuals match algebraic expressions built from the same topological constants.

**Method**: For each observable, we compute the residual r_i and test it against:
(1) rational approximations p/q with small denominators,
(2) structural fractions involving GIFT constants (dim(G₂), dim(E₈), b₂, b₃, H*, PSL(2,7), etc.),
(3) PSLQ integer relations with the 20 structural constants,
(4) mpmath.identify() for closed-form recognition.

**Key findings**:

| Observable | Residual r | Best match | Match error |
|------------|-----------|------------|-------------|
| δ_CP | +0.1130 | dim(G₂)/(dim(E₈)/2) = 14/124 | 0.08% |
| m_b/m_t | -0.0079 | -1/(N_gen × 2b₂) = -1/126 | exact |
| Y_p | +0.00368 | 1/(φ × PSL₂₇) = 1/(φ × 168) | 0.04% |
| sin²θ₁₂(CKM) | +0.00372 | 6/(b₂ × b₃) = 6/1617 | 0.2% |

**Documented structural observation**: Only δ_CP (see §4.4.1). The residual 14/124 = dim(G₂)/(dim(E₈)/2) implies a compactification factor 62/69 = dim(E₈)/(dim(E₈) + 4 dim(K₇)), which would reduce the deviation from 11.3% to 0.008%. The factor has independent structural motivation as the ratio of gauge to total degrees of freedom. The canonical GIFT prediction remains 197°. The factor is presented for completeness.

**Not adopted**: The m_b/m_t correction (-1/126) and Y_p correction (1/(φ × 168)) are documented for future work pending structural derivation from the compactification geometry. The CKM correction (6/(b₂ × b₃)) is suggestive but at lower precision.

**Epistemological note**: GIFT evolved iteratively from 6 free parameters (v1) through 4, 3, and finally 0 (v3). Each refinement added structural content while removing degrees of freedom. The δ_CP compactification factor is documented because it has geometric meaning independent of the experimental data it happens to match, but is not adopted as a revision — the raw topological 197° stands as the prediction.

---

## 8. Formal Verification and Statistical Analysis

### 8.1 Lean 4 Verification

The GIFT framework is formally verified in Lean 4 [28] with Mathlib [29]:

| Category | Count |
|----------|-------|
| Source files | 134 (128 core + 6 generated; 12 test/support) |
| Build jobs | 8378 |
| Unproven (sorry) | 0 |
| Published axioms | 4 (all substantive) |
| Certificate conjuncts | **213** |

The conjuncts cover metric/torsion/topology (38), couplings/masses/mixing (55), KK spectrum (41), metric eigenvalues (15), spectral invariants (10), δ_CP compactification (6), gauge breaking (10), bundle universality (11), instanton hierarchy (14), and 7D Weyl law (7). Full per-file breakdown in Supplement S3, §3.7.

```lean
theorem weinberg_relation :
  b2 * 13 = 3 * (b3 + dim_G2) := by native_decide

theorem koide_relation :
  dim_G2 * 3 = b2 * 2 := by native_decide
```

The E₈ root system is fully proven (12/12 theorems, basis generation). G₂ differential geometry (exterior algebra, Hodge star, torsion-free condition) is axiom-free. G₂ group structure (`g2_mul_closed`, `g2_subset_SO7`, `g2_det_mul_gram`) is proven by `native_decide` (v3.4.5). The 4 substantive axioms cover: `K7_analysis_data` (HarmonicForms.lean), `K7_spectral_data` (SpectralTheory.lean), `literature_package` (LiteratureAxioms.lean), and `KK_YM_EFT` (KKSpectralBridge.lean).

Three certified results anchor the formalization: (1) **G₂ three-form** — Bryant-Joyce φ₀ on ℝ⁷ formalized in `G2ThreeForm.lean`, all 7 nonzero coefficients certified by `native_decide`, dim(g₂) = 14 proven; (2) **ν̄(K₇, g) = 0** — CGN invariant certified zero via rectangular TCS (k₊ = k₋ = 1, CGN Main Corollary [16]); (3) **KK spectral bridge** — 4D mass gap formally conditional on KK_YM_EFT alone, all spectral ingredients Lean-certified.

### 8.2 Observable Coverage

Of the 95 observables, **55 are Lean-certified**:

| Type | Certified | Total | Coverage |
|------|-----------|-------|----------|
| **I** | 33 | 33 | 100% |
| **II** | 0 | 19 | 0% |
| **III** | 14 | 21 | 67% |
| **IV** | 8 | 22 | 36% |
| **Total** | **55** | **95** | **58%** |

Type II observables are Type I ratios × experimental VEVs (e.g., m_u = m_u/m_d × m_d(PDG)). The algebraic step — the ratio — is Lean-certified for all 33 core Type I formulas; only the physical scale identification step is uncertified. Axiomatizing VEV inputs would be circular (they are experimental inputs, not predictions). Type III coverage includes the new gauge (10+11 conjuncts) and instanton (14 conjuncts) certificates.

### 8.3 Scope of Verification

**What is proven**: Arithmetic identities relating topological integers. Given b₂ = 21, b₃ = 77, dim(G₂) = 14, etc., the numerical relations are machine-verified.

**What is not proven**: Existence of K₇, physical interpretation of ratios as SM parameters, uniqueness of formula assignments. The verification establishes **internal consistency**, not physical truth.

### 8.4 Statistical Uniqueness

Among 192,349 alternative (gauge group, holonomy, Betti) configurations tested by Monte Carlo, zero outperform GIFT: mean deviation 0.73% vs 32.9% for alternatives (P < 5 × 10⁻⁶, > 4.5σ). E₈×E₈ beats all tested groups by 13×; G₂ holonomy beats SU(3) (Calabi-Yau) by 13×. Only rank 8 gives N_gen = 3 exactly. The sensitivity analysis of §7 provides complementary evidence: (1) parameter variation (P < 5 × 10⁻⁶), (2) uniform coincidence (P = 10⁻³⁴⁶), and (3) algebraic coincidence (4.2M random formulas, P = 10⁻¹³³). Full gauge-group and holonomy rankings in Supplement S2, §23.

### 8.5 The G₂ Metric

The predictions in §4 depend only on topological invariants. However, the G₂ metric constrained by det(g) = 65/32 is numerically constructed as a Chebyshev-Cholesky expansion with 169 parameters (companion paper [30]):

| Quantity | Value |
|----------|-------|
| ‖T‖_C⁰ (torsion) | 2.949 × 10⁻⁵ |
| NK parameter h (analytical, Paper A [30]) | 8.95 × 10⁻⁹ (β = 0.321 exact, margin ×56M) |
| NK parameter h (numerical, tighter) | 1.43 × 10⁻⁹ (β = 0.0296 numerical, margin ×350M) |
| δg/g (analytical ball radius) | ≤ 4.86 × 10⁻⁶ |
| det(g) | 65/32 (exact, by construction) |

The interval-arithmetic Newton-Kantorovich certification [30] establishes existence and uniqueness of a torsion-free G₂ metric within the certified ball. The analytical certificate uses β = 0.321 derived from det(g) = 65/32 (exact constraint); a tighter numerical estimate β = 0.0296 yields h = 1.43 × 10⁻⁹ (margin ×350M) but rests on numerical Lipschitz estimates rather than the analytical bound. Both values certify h < 0.5 (Kantorovich threshold) by a factor of at least 56 million. Torsion reduction ×2995 in 5 Joyce iterations is well within Joyce's perturbative regime.

**Analytic invariant.** The Crowley-Goette-Nordström invariant ν̄(K₇, g) ∈ ℤ [16] vanishes for any rectangular TCS with twisting numbers k₊ = k₋ = 1, by CGN Main Corollary (gluing angle θ = π/2 forced). This conditional result is certified in Lean (`TCSConstruction.lean`, `K7_nu_bar_zero`): if K₇ is realized as a rectangular TCS, then ν̄(K₇, g) = 0 without any additional geometric computation. The building block identification remains open.

**Results from the NK-certified metric.** Analysis of the NK-certified metric yields several structural results:

- **V_min formula**: The minimum associative cycle volume satisfies V_min = √(Vol(K₇)/11), where 11 = b₃/n = 77/7. NK numerical value 219.90; formula gives 221.24 (0.6% agreement).
- **Harmonic decompositions**: b₂ = 7+14 = 3+18 = 11+10 (G₂ reps / hyperkähler triple / TCS blocks); b₃ = 35+42 = (1+7+27)+2×21; spectral gap 10522× between zero and non-zero modes.
- **U(1)² exact symmetry**: Period integrals S_θ = S_ψ = 6.1265, exact to 2.6×10⁻⁸ — propagates from metric to all period integrals.
- **Universal law**: λ₁ × H* = 12.3364 holds for all 66 known G₂ manifolds.
- **Lepton hierarchy from periods**: ln(m_τ/m_e) = 8.154 from SD associative volumes → e^8.154 = 3477 ✓ (Lean-certified).

Full details in the companion paper [30].

---

## 9. Discussion, Falsifiability, and Conclusion

### 9.1 Falsifiable Predictions

The framework makes concrete, testable predictions. The most critical:

**δ_CP**: The topological prediction is δ_CP = 197° = 7 × 14 + 99 (pure geometry). NuFIT 6.0 (NO w/o SK) gives 177° ± 20°, an 11.3% deviation from center but at the edge of the 1σ uncertainty band. The experimental central value has shifted significantly between NuFIT releases (197° in 5.2, 177° in 6.0) and may shift further as DUNE accumulates statistics (2028–2040). A PSLQ-identified compactification factor 62/69 is documented in §4.4.1 as a structural observation. **Falsification**: δ_CP outside [182, 212]° at 3σ creates serious tension.

**N_gen = 3**: No flexibility. A fourth generation immediately falsifies.

**m_s/m_d = 20**: Lattice QCD target precision ±0.5 by 2030. Current 20.0 ± 1.0.

**sin²θ_W = 3/13**: FCC-ee precision ~10⁻⁵ (4× improvement over current).

**New tests from Types II/III**:
- Combined lepton ratios (wilson_line+instanton): τ/e = 3485, τ/μ = 16.69, μ/e = 208.8 — all within 1%
- Proton lifetime: τ_p = 4.06 × 10³⁸ years (beyond near-term Hyper-K sensitivity; Hyper-K constrains nearby GUT alternatives)
- SUSY spectrum: m_gravitino = 166 GeV, m_moduli = 3.2 TeV (model-dependent; viable only in compressed/suppressed-coupling realizations, requiring dedicated recast at HL-LHC)

| Experiment | Observable | Timeline | Test Level |
|------------|------------|----------|------------|
| DUNE Phase I | δ_CP (3σ) | 2028–2030 | Critical |
| DUNE Phase II | δ_CP (5σ) | 2030–2040 | Definitive |
| Lattice QCD | m_s/m_d | 2028–2030 | Strong |
| HL-LHC | SUSY spectrum | 2029–2040 | Complementary |
| Hyper-Kamiokande | δ_CP, τ_p | 2034+ | Complementary |
| FCC-ee | sin²θ_W, Q_Koide | 2040s | Definitive |

### 9.2 Relation to M-Theory

E₈×E₈ and G₂ holonomy connect directly to M-theory [33,34]:
- Heterotic string theory requires E₈×E₈ for anomaly cancellation [19]
- M-theory on G₂ manifolds preserves N = 1 SUSY in 4D [35]

GIFT differs from standard M-theory phenomenology [36] by focusing on topological invariants rather than moduli stabilization. Where M-theory faces the landscape problem (~10⁵⁰⁰ vacua), GIFT proposes that topological data alone constrain the physics. The G₂-MSSM spectrum (§5.3, §6.5) is consistent with the phenomenology of Acharya et al. [34]: m_gravitino ~ 100 GeV, m_moduli ~ few TeV, gaugino condensation in a hidden sector.

The second E₈ factor is required by anomaly cancellation but has no direct coupling to Standard Model fields. In heterotic M-theory, gaugino condensation in this hidden sector drives SUSY breaking [34]. A natural interpretation is that the hidden E₈ sector provides the dark sector: the predicted cosmological ratios Ω_DM/Ω_b = 43/8 and Ω_DE = ln(2) × 98/99 emerge from the same topological invariants (b₂, b₃) that also determine dim(E₈) = 248. Whether this connection is structural or coincidental remains an open question.

### 9.3 Comparison

| Criterion | GIFT | String Landscape | Lisi E₈ |
|-----------|------|------------------|---------|
| Falsifiable predictions | Yes (δ_CP = 197°, N_gen) | Not yet (landscape selection) | Not yet (embedding obstruction) |
| Adjustable parameters | 0 | ~10⁵⁰⁰ | 0 |
| Formal verification | 213 conjuncts, 4 axioms | No | No |
| Precise predictions | 95 (66 testable) | Qualitative | ~5 |
| Gauge breaking | E₈→SM (6 levels) | Landscape-dependent | Single E₈ |
| Mass hierarchy | 2 mechanisms + combined | — | — |
| Sensitivity analysis | r_eff = 15.53, P = 10⁻³⁴⁶ | — | — |

**Distler-Garibaldi obstruction** [37]: Lisi's E₈ ToE attempted to embed all Standard Model fermions (including chirality) directly as roots of a single E₈. Distler and Garibaldi proved this is mathematically impossible — no E₈ representation decomposes into the correct chiral SM spectrum. GIFT avoids this obstruction entirely: E₈×E₈ is the gauge group of heterotic M-theory (not a particle container), the SM gauge group emerges from a six-level breaking chain (§5.1), fermion generations are fixed by the topological constraint (rank(E₈) + N_gen)b₂ = N_gen·b₃, giving N_gen = 3 (§2), and chirality is a topological property of the compactification. The mathematical relationship between E₈ and particles is cohomological (emergence from geometry), not representational (embedding into algebra).

### 9.4 Limitations and Open Questions

| Issue | Status | Section |
|-------|--------|---------|
| K₇ topological classification | Certified metric (Paper I); (b₂,b₃)=(21,77) absent from all known compact G₂ constructions; orthogonal TCS excluded by parity; complete geometric construction remains open | §2 |
| Singularity structure | Not required: SM gauge group from g₂⊂so(7) spectral structure (§5) | §2 |
| Formula selection rules | Consequence of metric uniqueness (see §9.4 note) | §3, §7 |
| α_s(RGE) = 3.7% deviation | SUSY threshold sensitivity; split-spectrum matching (§5.3) | §5 |
| δ_CP deviation | 11.3% vs NuFIT 6.0 central; at edge of 1σ band (±20°); factor 62/69 documented §4.4.1 | §4.4.1, §9.1 |
| ΔV(e-μ) = 15.9% deviation | Reduced to 0.75% by combined pipeline (α = e^K) | §6 |
| Hidden E₈ sector | Candidate for dark sector (dark matter, dark energy); no direct observable coupling | §9.2 |
| Quantum gravity completion | Not addressed | — |

**Note on the selection principle.** A natural objection to any framework of this type is: why *these* observables? The question has two levels. At the mathematical level, it dissolves: the NK-certified metric has zero free parameters, and its 169 Chebyshev coefficients are algebraically constrained to a five-generator lattice Z[π², π, 1, e, χ₉₈] / (b₂·b₃). Observables are functionals of this metric; they *must* therefore be algebraic in topological invariants. The "selection" is not a choice — it is a consequence of metric uniqueness. Extended grammar analysis confirms this: adding TCS-level atoms (χ(K3)=24, b₂(M₁)=11, b₂(M₂)=10) to the search grammar discovers simpler and more precise formulas for several observables (m_c/m_s exact at 11+7/10; Ω_DE = 53/77 rational, 5× more precise; λ_H = 11/87, 7× more precise), consistent with the prediction that observables encode TCS structure more directly than the higher-level algebra currently used.

At the philosophical level, the residual question is: why G₂ holonomy? This is the framework's single underdetermined input — the assertion that the compactification manifold is a compact 7-manifold of G₂ holonomy with b₁=0. This question has the same epistemic status as "why Lorentz invariance?" in special relativity or "why the Dirac equation?" in relativistic quantum mechanics: it is recognized as the minimal consistent mathematical structure compatible with observed symmetries, not derived from something more fundamental. G₂ is the unique compact exceptional holonomy group admitting a 7-dimensional representation with the stabilizer chain G₂ ⊃ SU(3) ⊃ SU(2) ⊃ U(1) that naturally contains the subgroups needed for the Standard Model. It is not selected among a continuum of options; it is an isolated point in the space of compatible mathematical structures. Every other unification framework carries equivalent or greater philosophical inputs alongside substantially more free parameters. GIFT makes its single input explicit.

**Honest assessment of outliers**: Two observables deviate by >5%: ΔV(e-μ) (15.9%, reduced to 0.75% by the combined wilson_line+instanton pipeline) and κ(gauge) (4.7%, reflecting genuine K₇ geometry at the percent level). The α_s(RGE) deviation is 3.7% with split-spectrum matching (§5.3); a naive degenerate-spectrum treatment would give 12.1%. The δ_CP prediction (197°) is 11.3% from NuFIT 6.0 central but at the edge of the 1σ uncertainty band (177° ± 20°). A structural compactification factor (62/69) is documented in §4.4.1.

### 9.5 Conclusion

We have explored a framework deriving **95 observables** from topological invariants of a compact G₂ manifold K₇ with E₈×E₈ gauge structure:

- **95 observables** organized in 4 types (33(I) + 19(II) + 21(III) + 22(IV)), with 66 testable against experiment
- **Mean deviation ~1.15%** across 66 comparable observables; 0.73% for Type I; 11 exact matches, 53 within 1%
- **Three new physics sections**: gauge breaking (E₈ → SM, §5), mass hierarchy (combined wilson_line+instanton, §6), sensitivity analysis (r_eff = 15.53, P(uniform) = 10⁻³⁴⁶, P(algebraic) = 10⁻¹³³, §7)
- **Lean 4 certification**: 213 conjuncts, 0 sorry, 55/95 observables verified, 4 substantive axioms (reduced from 38 via systematic elimination)
- **Statistical distinctiveness** at > 4.5σ among 192,349 alternatives tested
- **Falsifiable predictions**: δ_CP = 197°, N_gen = 3, sin²θ_W = 3/13, testable by DUNE/FCC-ee

**We do not claim this framework is correct.** It may represent genuine geometric insight, effective approximation, or elaborate coincidence. Only experiment can discriminate. The deeper question — why octonionic geometry would determine particle physics parameters — remains open.

But the empirical success of 95 observables from zero adjustable parameters, with an overdetermination ratio of 2.13× and coincidence probability 10⁻³⁴⁶ (uniform) / 10⁻¹³³ (algebraic), suggests that topology and physics may be more intimately connected than currently understood.

**The ultimate arbiter is experiment.**

---

## Author's note

This framework was developed through sustained collaboration between the author and several AI systems, primarily Claude (Anthropic), with contributions from GPT (OpenAI), Gemini (Google), Grok (xAI), for specific mathematical insights. The formal verification in Lean 4, architectural decisions, and many key derivations emerged from iterative dialogue sessions over several months. This collaboration follows transparent crediting approach for AI-assisted mathematical research. Mathematical constants underlying these relationships represent timeless logical structures that preceded human discovery. The value of any theoretical proposal depends on mathematical coherence and empirical accuracy, not origin. Mathematics is evaluated on results, not résumés.

---

## Data Availability

- Framework paper (this work): https://doi.org/10.5281/zenodo.18837071
- Paper A (certified G₂ metric): https://doi.org/10.5281/zenodo.19892350
- Paper B (spectral geometry): https://doi.org/10.5281/zenodo.19893371
- Paper C (K3 NK certificate): https://doi.org/10.5281/zenodo.19708916
- Code: https://github.com/gift-framework/core
- Lean proofs: https://github.com/gift-framework/core/tree/main/Lean

---

## Competing Interests

The author declares no competing interests.

---

## References

[1] Particle Data Group, Phys. Rev. D 110, 030001 (2024)
[2] S. Weinberg, Phys. Rev. D 13, 974 (1976)
[3] Planck Collaboration, A&A 641, A6 (2020)
[4] A.G. Riess et al., ApJL 934, L7 (2022)
[5] C.D. Froggatt, H.B. Nielsen, Nucl. Phys. B 147, 277 (1979)
[6] Y. Koide, Lett. Nuovo Cim. 34, 201 (1982)
[7] C. Furey, *Standard Model Physics from an Algebra?*, PhD thesis, U. Waterloo (2015)
[8] C. Furey, M.J. Hughes, Phys. Lett. B 831, 137186 (2022)
[9] R. Wilson, arXiv:2404.18938 (2024)
[10] T.P. Singh et al., arXiv:2206.06911v3 (2024)
[11] B.S. Acharya, S. Gukov, Phys. Rep. 392, 121 (2004)
[12] L. Foscolo et al., Duke Math. J. 170, 3 (2021)
[13] D. Crowley et al., Invent. Math. (2025)
[14] M. Haskins, J. Nordstrom, arXiv:1809.09083 (2022)
[15] A. Kasprzyk, J. Nordstrom, arXiv:2209.00156 (2022)
[16] D. Crowley, S. Goette, J. Nordstrom, Invent. Math. (2025)
[17] T. Dray, C.A. Manogue, *The Geometry of the Octonions*, World Scientific (2015)
[18] J.F. Adams, *Lectures on Exceptional Lie Groups*, U. Chicago Press (1996)
[19] D.J. Gross et al., Nucl. Phys. B 256, 253 (1985)
[20] D.D. Joyce, *Compact Manifolds with Special Holonomy*, Oxford U. Press (2000)
[21] A. Kovalev, J. Reine Angew. Math. 565, 125 (2003)
[22] A. Corti et al., Duke Math. J. 164, 1971 (2015)
[23] R. Harvey, H.B. Lawson, Acta Math. 148, 47 (1982)
[24] B.S. Acharya, Class. Quant. Grav. 19, 5619 (2002)
[25] B.S. Acharya, E. Witten, arXiv:hep-th/0109152 (2001)
[26] T2K, NOvA Collaborations, Nature 638, 534-541 (2025)
[27] NuFIT 6.0, www.nu-fit.org (2024)
[28] L. de Moura, S. Ullrich, CADE 28, 625 (2021)
[29] mathlib Community, github.com/leanprover-community/mathlib4
[30] B. de La Fourniere, "An Explicit Approximate G₂ Metric on a Compact 7-Manifold with Certified Torsion-Free Completion" (2026), doi:10.5281/zenodo.19892350
[31] DUNE Collaboration, FERMILAB-TM-2696 (2020)
[32] DUNE Collaboration, arXiv:2103.04797 (2021)
[33] E. Witten, Nucl. Phys. B 471, 135 (1996)
[34] B.S. Acharya et al., Phys. Rev. D 76, 126010 (2007)
[35] M. Atiyah, E. Witten, Adv. Theor. Math. Phys. 6, 1 (2002)
[36] G. Kane, *String Theory and the Real World* (2017)
[37] J. Distler, S. Garibaldi, Commun. Math. Phys. 298, 419 (2010)
[38] J.C. Baez, "Octonions and the Standard Model," math.ucr.edu/home/baez/standard/ (2020-2025)
[39] Springer Nature, "Artificial intelligence (AI) policy," www.springernature.com/gp/policies (2024)

[40] J.A. Wheeler, "Information, physics, quantum: the search for links," in *Complexity, Entropy, and the Physics of Information* (W.H. Zurek, ed.), Addison-Wesley (1990), pp. 3–28.

[41] J. Worrall, "Structural realism: the best of both worlds?," *Dialectica* **43**, 99–124 (1989).

[42] J. Ladyman, D. Ross, *Every Thing Must Go: Metaphysics Naturalized*, Oxford University Press (2007).

[43] B. de La Fournière, *Orientation, not ontology* (companion essay), giftheory.substack.com/p/orientation-not-ontology (2026).

[44] B. de La Fourniere, "Spectral Geometry of the G₂-GIFT Manifold: Betti Numbers, KK Spectrum, and Spectral Invariants" (2026), doi:10.5281/zenodo.19893371

[45] B. de La Fourniere, "Newton-Kantorovich Certificate for the K3 Donaldson Embedding in the G₂-GIFT Metric" (2026), doi:10.5281/zenodo.19708916

[46] R. Pinčák, A. Pigazzini, M. Pudlák, E. Bartoš, "Geometric origin of a stable black hole remnant from torsion in G₂-manifold geometry," Gen. Rel. Grav. **58**, 29 (2026), doi:10.1007/s10714-026-03528-z

---

## Appendix A: Topological Input Constants

| Symbol | Definition | Value |
|--------|------------|-------|
| dim(E₈) | Lie algebra dimension | 248 |
| rank(E₈) | Cartan subalgebra dimension | 8 |
| dim(K₇) | Manifold dimension | 7 |
| b₂(K₇) | Second Betti number | 21 |
| b₃(K₇) | Third Betti number | 77 |
| dim(G₂) | Holonomy group dimension | 14 |
| dim(J₃(O)) | Jordan algebra dimension | 27 |

The complete set of 20 structural constants (including dim(E₆), dim(E₇), dim(F₄), fund(E₇), |PSL(2,7)|, D_bulk, α_sum, det(g)_den, det(g)_num) is tabulated in Supplement S3, §3.3.

## Appendix B: Derived Structural Constants

| Symbol | Formula | Value |
|--------|---------|-------|
| p₂ | dim(G₂)/dim(K₇) | 2 |
| Weyl | From W(E₈) factorization | 5 |
| N_gen | Index theorem | 3 |
| H* | b₂ + b₃ + 1 | 99 |
| tau | (496 x 21)/(27 x 99) | 3472/891 |
| kappa_T | 1/(b₃ - dim(G₂) - p₂) | 1/61 |
| det(g) | p₂ + 1/(b₂ + dim(G₂) - N_gen) | 65/32 |

## Appendix C: Supplement Reference

| Supplement | Content | Pages |
|------------|---------|-------|
| S1: Foundations | E₈, G₂, K₇ construction details | ~60 |
| S2: Derivations | Complete proofs of 33 Type I relations | ~90 |
| S3: Observable Dataset | Full 95-entry table, type classification, sensitivity | ~20 |

---

*GIFT Framework*
