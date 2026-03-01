# A PINN Framework for G₂ Metrics on Compact 7-Manifolds: From Torus Validation to Spectral Fingerprints and the Flat Attractor Problem

**Author**: Brieuc de La Fournière

Independent researcher

**v2** --- Substantially expanded from v1 (DOI: 10.5281/zenodo.18643069).

---

## Abstract

We develop a physics-informed neural network (PINN) framework to
approximate torsion-free G₂-structures in seven dimensions, using a
Cholesky parameterization of the metric with analytical warm-start.
Starting from the flat 7-torus T⁷ (where the methodology is validated
to machine precision across 5 independent seeds) and progressing to a
3-chart atlas inspired by the Twisted Connected Sum (TCS) construction
of a compact G₂ manifold K₇ with b₂ = 21, b₃ = 77, we report the
following results.

**Part I (Stages 1--4, from v1).** The atlas achieves machine-precision
interface matching (10⁻¹²) including through a non-trivial Kovalev
twist, and produces a qualitatively different Laplace spectrum compared
to the torus baseline, with Kovalev-twist-induced mode splitting.

**Part II (Stages 5--8, new in v2).** Post-atlas analysis reveals that
the PINN metric compresses to **28 numbers** --- a single 7×7 symmetric
matrix G acted on by the Kovalev twist (38,231× compression from
1,070,471 network parameters). The anisotropy ratio κ_T = 1/61 emerges
to 7 significant figures. The Laplace spectrum, after Kovalev
symmetrization, organizes into bands with degeneracies **[1, 10, 9, 30]**
that encode the Betti numbers of the TCS building blocks, at a Fisher
combined significance of 5.8σ (pre-registered, 4 null models). We then
disclose a fundamental difficulty: the trained atlas metric is
**essentially flat** (||R_autograd|| ~ 10⁻¹³), meaning that the small
torsion reported in Stages 1--4 is satisfied trivially. This "flat
attractor" is a general failure mode of PINNs for special-holonomy
problems. We present a remedy --- first-order barrier targets on metric
spatial gradients --- that escapes the flat attractor (curvature increased
by a factor of 10²²) and, combined with a TCS warm-start
parametrization, achieves a validated torsion floor ∇φ = 0.010 with
genuine curvature. A scaling law ∇φ(L) = 1.47 × 10⁻³/L² is
established and confirmed across 8 independent methods, closing the
1D metric program. Bulk metric optimization --- block-diagonal
rescaling of the background G₀ with optimal (a_t, a_f) =
(2.47, 1.60) --- then reduces the torsion by a further **42%**,
yielding a new scaling law ∇φ(L) = 8.46 × 10⁻⁴/L² validated across
20 seeds (CV < 1%). The torsion budget shifts from 71/29 to **65/35**
(t-derivative / fiber-connection) with the optimized metric. We characterize the residual torsion as intrinsic to the
interpolation path through G₂ structure space (g₂-valued, not
V₇-valued), and systematically compare interpolation strategies
(Cholesky vs log-Euclidean geodesic), identifying Cholesky
interpolation as 2× superior.

**Part III (Stages 9--12).** Landscape cartography (287 evaluations)
confirms the optimum is the unique global minimum (Hessian condition
number 92,392). The metric determinant is proven to be a pure gauge
parameter (∇φ_code ∝ det^{3/7}, exact to 8.4 × 10⁻¹⁵), and the
proper 3-form norm |φ|² = 42 = 7 × dim(G₂) is identified as an
exact topological invariant. Full 7D spectral analysis (117,648 modes)
confirms Weyl's law at 97.6%, with a critical crossing length
L_cross = 0.35. Sturm-Liouville eigenfunctions match flat-space cosines
to 4 × 10⁻⁶, yielding Yukawa selection rules (n₁ ± n₂ ± n₃ = 0,
9/56 allowed, universal coupling |Y| = 1/√(2V)) preserved under the
full metric (CV = 0.0001%). The G₂ representation-theoretic
decomposition of cup product Yukawas reveals an exact selection rule
Y(Ω²₇ × Ω²₇ × Ω³₇) = 0, and all J-invariant Yukawas vanish ---
physical couplings originate exclusively from the anti-invariant sector.

**Keywords**: G₂ holonomy, torsion-free structures, physics-informed
neural networks, exceptional holonomy, twisted connected sum, spectral
geometry, Cholesky parameterization, flat attractor, Yukawa couplings

**MSC 2020**: 53C29 (holonomy groups), 65N99 (numerical PDE methods),
58J50 (spectral problems on manifolds)

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Mathematical Background](#2-mathematical-background)
3. [The Target: K₇ and Its Metric](#3-the-target-k₇-and-its-metric)
4. [The PINN Methodology](#4-the-pinn-methodology)
5. [Stage 1: Torsion-Free G₂-Structure on the Flat 7-Torus](#5-stage-1-torsion-free-g₂-structure-on-the-flat-7-torus)
6. [Stage 2: Multi-Seed Identifiability](#6-stage-2-multi-seed-identifiability)
7. [Stage 3: The Topology Gap](#7-stage-3-the-topology-gap)
8. [Stage 4: The Atlas Construction](#8-stage-4-the-atlas-construction)
9. [Stage 5: The 28-Number Metric](#9-stage-5-the-28-number-metric)
10. [Stage 6: Spectral Fingerprints](#10-stage-6-spectral-fingerprints)
11. [Stage 7: The Flat Attractor](#11-stage-7-the-flat-attractor)
12. [Stage 8: Toward Genuine Curvature](#12-stage-8-toward-genuine-curvature)
13. [Discussion](#13-discussion)
14. [Conclusion](#14-conclusion)

---

## 1. Introduction

### 1.1 The problem

A compact Riemannian 7-manifold (M⁷, g) has holonomy contained in the
exceptional Lie group G₂ ⊂ SO(7) if and only if it admits a *torsion-free
G₂-structure*: a 3-form φ ∈ Ω³(M) that is simultaneously closed (dφ = 0)
and coclosed (d*φ = 0). The metric is then Ricci-flat [1, 2].

Joyce [3, 4] proved the existence of compact examples in 1996 by resolving
singularities of T⁷/Γ orbifolds. Kovalev [5] introduced the Twisted
Connected Sum (TCS) construction in 2003, gluing two asymptotically
cylindrical (ACyl) Calabi-Yau threefolds along a common K3 fiber.
Corti, Haskins, Nordström and Pacini (CHNP) [6] systematized the TCS
method and produced a large family of topological types.

These existence theorems guarantee that the metric exists, but they do
not compute it. To date, **no explicit numerical metric tensor g_ij(x)
has been reported for any compact G₂ manifold**, though substantial work
exists for non-compact examples (e.g. Brandhuber et al. [15],
cohomogeneity-one solitons [16]). This stands in contrast to Calabi-Yau
metrics, where machine-learning methods have achieved machine-precision
results (cymyc [17], Donaldson's algorithm [18]). Concurrent work by
Heyes, Hirst, Sa Earp and Silva [22] has independently applied neural
networks to approximate G₂-structures on contact Calabi-Yau
7-manifolds, with cross-citation of our preliminary results [21],
establishing methodological convergence in this emerging field.

### 1.2 Why G₂ is harder than Calabi-Yau

The difficulty gap between Calabi-Yau (CY) and G₂ metrics is structural,
not merely computational:

| Aspect | Calabi-Yau (complex dim 3) | G₂ (real dim 7) |
|--------|---------------------------|-----------------|
| **Key object** | Kähler potential K (1 scalar) | Associative 3-form φ (35 components) |
| **PDE** | Complex Monge-Ampère (scalar) | System dφ = 0, d*φ = 0 (35 coupled) |
| **Embedding** | Projective space (natural coordinates) | No projective structure |
| **Construction** | Yau's theorem + algebraic geometry | TCS gluing (two CY₃ + neck) |
| **Symmetry** | U(1) Kähler gauge | No continuous symmetry to exploit |

The Kähler potential reduces the CY problem to optimizing a single
function. For G₂, one must solve a system of 35 coupled PDEs on a
7-dimensional domain with no standard coordinate system.

### 1.3 Our approach

We apply physics-informed neural networks (PINNs) [7] to construct G₂
metrics. The key technical innovation is a **Cholesky parameterization
with analytical warm-start**: the network outputs a small
lower-triangular perturbation δL(x) around the Cholesky factor of an
analytically derived target metric, so that

$$g(x) = (L_0 + \delta L(x))(L_0 + \delta L(x))^T$$

guarantees positive definiteness and symmetry by construction, and
reduces the learning problem to small residual corrections.

We proceed through eight stages of increasing complexity:

1. **Flat-torus validation**: Validate the PINN methodology on T⁷,
   achieving machine-precision torsion-free G₂-structure constraints.
2. **Multi-seed identifiability**: Verify convergence robustness across
   5 independent random initializations.
3. **Topology gap analysis**: Demonstrate that local geometry on T⁷
   cannot reproduce the anisotropy of K₇.
4. **Atlas construction**: Build a 3-chart atlas on a TCS model of K₇
   with Schwarz interface matching and Kovalev twist.
5. **Metric compression**: Discover that the learned metric compresses
   to a single 7×7 matrix (28 parameters, 38,231× compression).
6. **Spectral fingerprints**: Establish that Laplacian degeneracies
   encode building-block Betti numbers at 5.8σ significance.
7. **Flat attractor diagnosis**: Reveal that the atlas metric is
   essentially flat --- a general PINN failure mode.
8. **Curvature recovery, 1D closure, and bulk optimization**: Develop
   techniques to force genuine curvature while controlling torsion,
   characterize the torsion scaling law ∇φ ~ L⁻², close the 1D metric
   optimization program, and optimize the bulk metric G₀ for a further
   42% torsion reduction.

Stages 1--4 appeared in v1 of this paper [21]; Stages 5--8 are new.

### 1.4 Relation to the GIFT framework

This work is motivated by the GIFT (Geometric Information Field Theory)
framework [12], which proposes connections between Standard Model
parameters and the topology of E₈ × E₈ compactifications on G₂
manifolds. The physical claims of GIFT are outside the scope of this
paper and have not been peer-reviewed.

We use several analytical targets derived in GIFT as concrete numerical
inputs for the PINN optimization:

| Input | Value | Role in this paper |
|-------|-------|-------------------|
| det(g) | 65/32 | Determinant constraint (from [12]) |
| κ_T | 1/61 | Anisotropy parameter (from [12]) |
| b₂, b₃ | 21, 77 | Betti numbers of K₇ (standard, [5, 6]) |

The Betti numbers are standard topological invariants of the Kovalev
TCS manifold [5, 6] and are independent of GIFT. The determinant
constraint det(g) = 65/32 and the anisotropy parameter κ_T = 1/61 are
derived within GIFT; alternative choices could be used without affecting
the methodology. The PINN framework presented here is **independent of
these specific target values**: it is a general-purpose tool for
computing torsion-free G₂ metrics under any prescribed constraints.

GIFT also makes a spectral prediction λ₁ × H* = 14 (where
H* = b₂ + b₃ + 1 = 99). We report spectral bridge computations but
treat this value as an open question, not an established target.

---

## 2. Mathematical Background

### 2.1 The octonions and G₂

The exceptional Lie group G₂ is the automorphism group of the octonion
algebra O. This is the unique simple Lie group of dimension 14 and rank 2.
It acts naturally on Im(O) ≅ ℝ⁷.

The **octonion multiplication table** is encoded by the Fano plane: the
unique projective plane of order 2, with 7 points and 7 lines, each line
containing exactly 3 points. The 7 lines are:

```
{1,2,4}, {2,3,5}, {3,4,6}, {4,5,7}, {5,6,1}, {6,7,2}, {7,1,3}
```

(in cyclic notation, with appropriate orientations). Each line defines
a triple of imaginary octonion units whose product equals +1:

$$e_i \cdot e_j = e_k \quad \text{for } (i,j,k) \text{ a positively oriented Fano line}$$

This structure is rigid: G₂ is the *only* subgroup of GL(7,ℝ) that
preserves the Fano multiplication.

### 2.2 The associative 3-form

G₂ preserves a canonical 3-form φ₀ ∈ Λ³(ℝ⁷), the **associative 3-form**
(also called the G₂ calibration):

$$\varphi_0 = e^{124} + e^{235} + e^{346} + e^{457} + e^{561} + e^{672} + e^{713}$$

where e^{ijk} = eⁱ ∧ eʲ ∧ eᵏ. The 7 terms correspond one-to-one with
the 7 lines of the Fano plane. This is the *unique* (up to scale)
G₂-invariant element of Λ³(ℝ⁷).

The **coassociative 4-form** is its Hodge dual:

$$\psi_0 = *\varphi_0 \in \Lambda^4(\mathbb{R}^7)$$

A G₂-structure on a 7-manifold M is a smooth 3-form φ ∈ Ω³(M) that is
pointwise equivalent to φ₀ (i.e., at each point, there exists a frame
in which φ = φ₀). The structure is **torsion-free** when:

$$d\varphi = 0 \quad \text{(closed)} \qquad \text{and} \qquad d{*}\varphi = 0 \quad \text{(coclosed)}$$

### 2.3 From 3-form to metric

The 3-form φ determines a unique Riemannian metric g and volume form
vol via the remarkable formula [2]:

$$g_{ij} \, \text{vol} = \frac{1}{6} \, \iota_{e_i}\varphi \wedge \iota_{e_j}\varphi \wedge \varphi$$

In coordinates, this simplifies to:

$$g_{ij} = \frac{1}{6} \sum_{k,l=1}^{7} \varphi_{ikl} \, \varphi_{jkl}$$

For the standard form φ₀, this gives g = I₇ (the identity). A rescaled
form φ = c · φ₀ yields g = c² · I₇ with det(g) = c¹⁴.

**The metric is determined by the 3-form.** This is the fundamental fact
that makes the G₂ approach to Ricci-flat metrics possible: instead of
optimizing a 28-parameter symmetric matrix at each point, one optimizes
a 35-component 3-form, and the metric follows.

### 2.4 Representation-theoretic decomposition

Under G₂, the space of k-forms on ℝ⁷ decomposes into irreducible
representations. The decomposition of 3-forms is particularly important:

$$\Lambda^3(\mathbb{R}^7) = \Lambda^3_1 \oplus \Lambda^3_7 \oplus \Lambda^3_{27}$$

with dimensions 1 + 7 + 27 = 35 = $\binom{7}{3}$. These components have
geometric meaning:

| Component | Dimension | Meaning |
|-----------|-----------|---------|
| Λ³₁ | 1 | Scalar multiple of φ₀ (overall scale) |
| Λ³₇ | 7 | Volume-changing deformations (Fano-aligned) |
| Λ³₂₇ | 27 | Traceless shape deformations |

### 2.5 The so(7) decomposition and holonomy

The Lie algebra so(7) decomposes under G₂ as:

$$\mathfrak{so}(7) = \mathfrak{g}_2 \oplus V_7$$

where dim(g₂) = 14 and dim(V₇) = 7. The **V₇ fraction** of the
Levi-Civita connection,

$$f_{V_7} = \frac{\|A^{V_7}\|^2}{\|A^{\mathfrak{g}_2}\|^2 + \|A^{V_7}\|^2}$$

is a pointwise diagnostic of holonomy: for a metric with Hol(g) ⊆ G₂,
the connection is entirely g₂-valued (f_{V₇} = 0). For a flat metric,
the decomposition is governed by noise equipartition:
f_{V₇} = dim(V₇)/dim(so(7)) = 7/21 = 1/3. This quantity plays a
central diagnostic role in Stages 7--8.

### 2.6 The Joyce existence theorem

**Theorem** (Joyce [3, 4]). *Let M be a compact 7-manifold admitting
a G₂-structure φ₀ with small torsion: ‖dφ₀‖_{C⁰} + ‖d*φ₀‖_{C⁰} < ε₀
for a constant ε₀ > 0 depending on the geometry. Then there exists a
nearby torsion-free G₂-structure φ̃ with*

$$d\tilde{\varphi} = 0, \qquad d{*}\tilde{\varphi} = 0$$

*In particular, the Riemannian metric g̃ associated to φ̃ has
Hol(g̃) ⊆ G₂ and is Ricci-flat.*

This is the fundamental tool: if we can construct a 3-form with
sufficiently small torsion on a compact 7-manifold, Joyce's theorem
guarantees the existence of a genuine Ricci-flat metric nearby.

---

## 3. The Target: K₇ and Its Metric

### 3.1 Twisted Connected Sum construction

The manifold K₇ is constructed as a Twisted Connected Sum (TCS) [5, 6]:

$$K_7 = M_1 \cup_\Phi M_2$$

where M₁ and M₂ are asymptotically cylindrical (ACyl) Calabi-Yau
threefolds, glued along their common asymptotic cross-section S¹ × K3
via a diffeomorphism Φ.

We adopt a specific TCS realization with building blocks chosen to
match the target Betti numbers b₂ = 21, b₃ = 77 (see CHNP [6] for a
systematic classification of such pairs):

| Building block | Role | b₂ | b₃ |
|---------------|------|-----|-----|
| M₁ | ACyl CY₃ | 11 | 40 |
| M₂ | ACyl CY₃ | 10 | 37 |
| K3 | Gluing fiber, b₂(K3) = 22 | ---|

**Model assumption.** The specific decomposition b₂ = 11 + 10 and
b₃ = 40 + 37 is one choice among potentially many TCS realizations
yielding the same total Betti numbers. Our PINN framework only depends
on the totals (b₂, b₃), not on the specific building blocks.

The topology of K₇ follows from the Mayer-Vietoris sequence:

$$b_2(K_7) = b_2(M_1) + b_2(M_2) = 11 + 10 = 21$$
$$b_3(K_7) = b_3(M_1) + b_3(M_2) = 40 + 37 = 77$$

Since K₇ is a compact oriented 7-manifold, Poincaré duality (bₖ = b_{7-k})
gives the full Betti sequence:

$$b_0 = b_7 = 1, \quad b_1 = b_6 = 0, \quad b_2 = b_5 = 21, \quad b_3 = b_4 = 77$$

The Euler characteristic vanishes:
χ(K₇) = 1 − 0 + 21 − 77 + 77 − 21 + 0 − 1 = 0.

### 3.2 The moduli space

The moduli space of torsion-free G₂ structures on K₇ is a smooth manifold
of dimension b₃(K₇) = 77 [3, 4]. Each modulus Πₖ corresponds to a period
integral of the associative 3-form over a 3-cycle Cₖ ∈ H₃(K₇, ℤ):

$$\Pi_k = \int_{C_k} \varphi, \qquad k = 1, \ldots, 77$$

### 3.3 The K3 lattice and E₈

The gluing data is constrained by the K3 lattice Λ_{K3} of signature
(3, 19) and rank 22. The K3 intersection form is:

$$\Lambda_{K3} = 3H \oplus 2(-E_8)$$

where H is the hyperbolic lattice and E₈ is the positive-definite E₈
root lattice.

### 3.4 The determinant formula

The determinant of the G₂ metric on K₇ is determined by a model
normalization. In the GIFT framework [12], this is derived as:

$$\det(g) = \frac{65}{32} = \frac{\dim(E_8) + \dim(G_2) + \text{rank}(E_8) + \dim(K_7)}{2^5} = \frac{248 + 14 + 8 + 7}{32}$$

This equals 2.03125 exactly. **This relation is derived within the GIFT
framework [12]; see §1.4 for its status and independence from the PINN
methodology.**

### 3.5 The torsion coupling constant

$$\kappa_T = \frac{1}{b_3 - \dim(G_2) - p_2} = \frac{1}{77 - 14 - 2} = \frac{1}{61} \approx 0.01639$$

where p₂ = dim(G₂)/dim(K₇) = 2 is the dimensional ratio. **This relation
is derived within the GIFT framework [12]; see §1.4 for its status.**

---

## 4. The PINN Methodology

### 4.1 Physics-informed neural networks

A physics-informed neural network [7] is a neural network f_θ(x) trained
to satisfy a PDE by incorporating the governing equations directly into
the loss function:

$$\mathcal{L}(\theta) = \mathcal{L}_{\text{PDE}}(\theta) + \lambda \, \mathcal{L}_{\text{BC}}(\theta) + \mu \, \mathcal{L}_{\text{data}}(\theta)$$

For the G₂ metric problem, the "PDE" is the torsion-free condition
dφ = 0, d*φ = 0; the "boundary conditions" include the determinant
constraint and positive definiteness; and the "data" term provides
supervised guidance from an analytical target.

### 4.2 Failed approaches

Before describing the successful architecture, we document two
instructive failures.

**Attempt 1: G₂ adjoint parameterization.** A network outputs 14
parameters in the G₂ Lie algebra, which generate a G₂ rotation of φ₀
via Lie derivatives. The problem: the map from 14 Lie algebra parameters
to 35 3-form components via Lie derivatives has **rank 6**, creating a
6-dimensional bottleneck in the 28-dimensional space of metric
perturbations. The network cannot access 22 of the 28 metric degrees
of freedom, regardless of training duration.

**Attempt 2: Additional anisotropy loss.** Same architecture as above
with an L₂ loss ‖⟨g⟩ − G_TARGET‖²_F. Now 97.6% of the gradient comes
from the anisotropy term, but the rank-6 bottleneck prevents the network
from responding. Loss plateaus after ~100 steps and remains constant.

**Lesson.** When the architecture fundamentally cannot represent the
target (rank deficiency), no amount of training or hyperparameter tuning
will help. The bottleneck must be removed at the architectural level.

### 4.3 The Cholesky parameterization

We parameterize the metric via a Cholesky decomposition:

$$g(x) = L(x) \cdot L(x)^T, \qquad L(x) = L_0 + \delta L(x)$$

where L₀ = chol(G_TARGET) is the Cholesky factor of an analytically
derived target metric, and δL(x) is a lower-triangular matrix-valued
function output by the neural network. This guarantees:

1. **Positive definiteness** by construction (LLᵀ ≥ 0 for any L)
2. **Symmetry** by construction (LLᵀ = (LLᵀ)ᵀ)
3. **Warm start** at the analytical target (δL = 0 ⟹ g = G_TARGET)
4. **Full rank**: 28 independent parameters per point (the full
   dimension of Sym⁺₇(ℝ))

### 4.4 Network architecture

```
Input: x ∈ ℝ⁷ (coordinates on the domain)
  │
  ├─ Fourier embedding: [sin(2πkxⱼ), cos(2πkxⱼ)] for k=1..32, j=1..7
  │   → 448-dimensional feature vector
  │
  ├─ MLP: 448 → 256 → 256 → 256 → 256 (SiLU activation)
  │
  ├─ Metric head: 256 → 28 (lower triangular δL)
  │     g(x) = (L₀ + 0.2·δL(x))(L₀ + 0.2·δL(x))ᵀ
  │
  └─ Output: g ∈ ℝ⁷ˣ⁷ (symmetric, positive definite)

Total parameters: 182,926
```

### 4.5 The perturbation scale

The perturbation scale multiplying the network's output is critical.
With a scale of 0.01 (as in an earlier version of the code), the
quadratic contribution to the metric is:

$$g = (L_0 + 0.01 \cdot \delta L)^T(L_0 + 0.01 \cdot \delta L) = L_0^T L_0 + 0.01 \cdot (\text{cross terms}) + 10^{-4} \cdot \delta L^T \delta L$$

The gradient is scaled by (0.01)² = 10⁻⁴. The PINN converges at
epoch 0: not because it learned the metric, but because it *could not
change it*. All losses are satisfied trivially by the background L₀.

**Fix**: scale = 0.2 (a single parameter change). This is the first
instance of a pattern that recurs throughout this paper: the optimizer
finding trivial solutions that satisfy the loss without learning the
physics. We encounter a more fundamental version of this failure in
Stage 7.

> **Diagnostic rule**: If a PINN converges in fewer than 10 epochs,
> check the scale of its output.

### 4.6 Loss functions

| Term | Formula | Weight | Purpose |
|------|---------|--------|---------|
| L_det | E_x[(det(g(x)) − 65/32)²] | 100 | Determinant constraint |
| L_torsion | E_x[‖∇×φ(x)‖²] | 1 → 50 | Torsion-free (dφ=0, d*φ=0) |
| L_supervised | E_x[‖g(x) − g_TCS(x)‖²_F] | 100 → 1 | Warm-start guidance |

The supervised weight is annealed from 100 to 1 during training.

### 4.7 From metric to torsion: the computational pipeline

**Step 1: Metric → orthonormal frame.** The Cholesky factor
L(x) = L₀ + 0.2·δL(x) defines a frame; an orthonormal frame
{eⁱ} is obtained via eⁱ = (L⁻ᵀ)ⁱⱼ dxʲ.

**Step 2: Frame → associative 3-form.** The associative 3-form is
constructed from the orthonormal frame:

$$\varphi = \sum_{(i,j,k) \in \text{Fano}} e^i \wedge e^j \wedge e^k$$

This produces a 3-form that is by construction pointwise equivalent
to the standard φ₀: it defines a G₂-structure compatible with g.

**Step 3: Exterior derivative via automatic differentiation.** All
partial derivatives ∂ᵢφ_{jkl} are evaluated exactly (to machine
precision) by automatic differentiation through the neural network.
No finite differences are used.

**Step 4: Hodge star and coclosure.** The coclosure condition d*φ = 0
is evaluated by applying Step 3 to the 4-form *φ.

**Step 5: Torsion norm.** The torsion is quantified as a Monte Carlo
L² norm over a batch of N = 4,096 independently sampled points:

$$\|T\|^2 = \frac{1}{N} \sum_{j=1}^{N} \left( \|d\varphi(x_j)\|^2_{g} + \|d{*}\varphi(x_j)\|^2_{g} \right)$$

Reported values are evaluated on a **separate test set** of 10,000
points drawn independently from the training distribution.

### 4.8 Training protocol

| Phase | Epochs | lr | Focus |
|-------|--------|-----|-------|
| 1. Warm-start | 0--500 | 10⁻³ | Pure supervised |
| 2. Transition | 500--1000 | 10⁻³ | Ramp down supervised, ramp up physics |
| 3. Physics | 1000--3000 | 10⁻³ | det + torsion + light supervised |
| 4. Torsion focus | 3000--4500 | 5×10⁻⁴ | Heavy torsion weight |
| 5. Refinement | 4500--5500 | 10⁻⁴ | Balanced final polish |

Training time: approximately 45 minutes per seed on an A100 GPU.

---

## 5. Stage 1: Torsion-Free G₂-Structure on the Flat 7-Torus

### 5.1 Experimental setup

We train the PINN on the periodic domain [0,1]⁷ ≅ T⁷ with the
analytical TCS metric as warm-start target.

### 5.2 Results

| Metric | Value | Assessment |
|--------|-------|------------|
| det(g) error | 1.75 × 10⁻¹⁵ | **Machine precision** (float64 limit) |
| Torsion ‖dφ‖ + ‖d*φ‖ | 3.1 × 10⁻⁸ | Small-torsion regime |
| Positive definite | Yes | All eigenvalues > 0 |
| Condition number | ~1.000 | Well-conditioned |

The trained metric, averaged over 10,000 random samples, is
g ≈ 1.1065 × I₇ --- a constant scalar multiple of the identity, with
c = (65/32)^{1/7}. The metric is perfectly isotropic: the correct
answer on T⁷.

### 5.3 The spectral bridge

We compute eigenvalues of the Laplace-Beltrami operator Δ_g via a
Monte Carlo Galerkin method (N = 98 Fourier basis functions,
M = 50,000 sample points). The result: 98 eigenvalues organized in
two degenerate bands (λ ≈ 35 with 14-fold degeneracy, λ ≈ 68). These
are flat-torus eigenvalues, not K₇ topology.

**λ₁ × H* = 3456 ≠ 14.**

---

## 6. Stage 2: Multi-Seed Identifiability

Five seeds (42, 137, 256, 314, 777), each trained for 2000 epochs:

| Seed | det(g) | det error | Torsion |
|------|--------|-----------|---------|
| 42 | 2.03125018 | 8.9 × 10⁻⁶ % | 1.68 × 10⁻⁷ |
| 137 | 2.03125021 | 1.0 × 10⁻⁵ % | 2.32 × 10⁻⁷ |
| 256 | 2.03125020 | 9.6 × 10⁻⁶ % | 1.80 × 10⁻⁷ |
| 314 | 2.03125028 | 1.4 × 10⁻⁵ % | 1.92 × 10⁻⁷ |
| 777 | 2.03125025 | 1.2 × 10⁻⁵ % | 2.64 × 10⁻⁷ |

Cross-seed: det(g) = 2.03125022 ± 3.7 × 10⁻⁸ (**9 significant digits
of agreement**). λ₁ = 34.58 ± 0.10 (0.3% variation). All seeds converge
to the same solution.

---

## 7. Stage 3: The Topology Gap

### 7.1 The κ_T experiment

Can we force κ_T = 1/61 on T⁷ via a loss term? After 500 epochs with
κ loss weight 2000:

| Method | κ achieved | % of target |
|--------|-----------|------------|
| PINN on T⁷ (optimized) | 0.000000 | 0.0% |
| Fine-tuned with κ loss | 0.000267 | 1.6% |
| Analytical TCS on T⁷ | 0.001879 | 11.5% |
| **K₇ topology (target)** | **0.016393** | **100%** |

### 7.2 Interpretation

> **Local geometry cannot encode global topology.**

On T⁷ (Hol = {1}), the optimal metric is the isotropic g = c × I₇. No
local loss function can force topological anisotropy. The gap between the
analytical TCS (11.5%) and K₇ (100%) is the **topology gap**: it can
only be crossed by changing the underlying manifold.

---

## 8. Stage 4: The Atlas Construction

### 8.1 Three-domain decomposition

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Bulk M₁    │◄───►│     Neck     │◄───►│   Bulk M₂    │
│  (CY₃ × S¹)  │     │  (S¹ × CY₃)  │     │  (CY₃ × S¹)  │
│   PINN_L     │     │  PINN_neck   │     │   PINN_R     │
└──────────────┘     └──────────────┘     └──────────────┘
```

Domains: Bulk_L [0.00, 0.40], Neck [0.25, 0.75], Bulk_R [0.60, 1.00].

### 8.2 The Kovalev twist

$$\Phi: (t, \theta, u_1, u_2, u_3, u_4, \psi) \mapsto (1-t, \psi, u_1, u_2, u_3, u_4, \theta)$$

Our atlas is a model inspired by the TCS construction; proving that it
faithfully represents K₇ topology would require global analysis beyond
the scope of this work.

### 8.3 Results

Total architecture: 564,678 parameters. After training on a Colab A100:

**Per-chart validation:**

| Chart | det error | Condition | Pos. def. |
|-------|-----------|-----------|-----------|
| Neck | 0.0002% | 1.0000003 | Yes |
| Bulk_L | 1.4 × 10⁻⁶ % | 1.0000010 | Yes |
| Bulk_R | 7.6 × 10⁻⁵ % | 1.0000013 | Yes |

**Interface matching:**

| Interface | Matching error |
|-----------|---------------|
| Left (direct) | 2.16 × 10⁻¹² |
| Right (through Kovalev twist) | 6.17 × 10⁻¹² |

**Spectral bridge (preliminary):**

```
Atlas spectrum (first modes):
 λ₁  =    9.07    ← isolated mode (absent on T⁷)
 λ₂--λ₈ ≈ 34--36  ← cluster of ~7 modes
 λ₉  =   80.71    ← isolated mode
```

The Kovalev twist lifts the T⁷ 14-fold degeneracy and introduces an
isolated low-lying mode. **Caveat**: as revealed in Stage 7 (§11),
the atlas metric is essentially flat, making these absolute eigenvalues
properties of the piecewise-constant background, not of a curved G₂
metric. The qualitative *mode splitting* induced by the Kovalev twist
remains meaningful; the absolute values do not.

---

## 9. Stage 5: The 28-Number Metric

### 9.1 The anisotropy parameter

The initial atlas (§8) had κ_T = 0, just like T⁷. We introduced a
single learnable parameter controlling the relative scaling between K3
and fiber directions. After training:

$$\kappa_T = 0.01639344 = \frac{1}{61} \text{ to 7 significant figures}$$

The anisotropy and torsion objectives are aligned, not competing:
κ_T = 1/61 is the torsion-minimizing value. (The torsion on the flat
atlas is trivially small regardless of κ_T; see §11. The alignment is
confirmed on the non-flat metric in Stage 8.)

### 9.2 Metric compression

A systematic analysis of the trained PINN weights revealed that the
learned metric is **piecewise constant**: averaged over 10,000 random
spatial points in each chart, the variance is negligible
(Var(g)/g² ~ 10⁻¹¹). The spatial variation is below the noise floor.

The entire metric is encoded by:

$$g = \begin{cases} G & \text{for } t < 3/4 \\ J^T G J & \text{for } t \geq 3/4 \end{cases}$$

where G is a single 7×7 symmetric positive-definite matrix (28 free
parameters) and J is the fixed Kovalev twist. The three PINNs, with
1,070,471 parameters total, have collectively found **28 numbers** --- a
compression ratio of **38,231×**.

The diagonal elements satisfy:
- det(G) = 65/32 (topological, from [12])
- κ_T(G) = (g_K3 − g_fib)/(g_K3 + g_fib) = 1/61 (topological, from [12])

κ_T arises from a topological invariant; det(g) is a model
normalization (see §3.4). Given these two constraints and the
near-diagonal structure, the metric has **zero continuous free
parameters**.

### 9.3 K₇ is not flat

**Theorem.** *The compact 7-manifold K₇ with b₃(K₇) = 77 does not
admit a flat metric.*

**Proof.** By Bieberbach's theorem, every compact flat n-manifold has
b₃ ≤ C(n,3). For n = 7: b₃ ≤ C(7,3) = 35. Since b₃(K₇) = 77 > 35,
K₇ cannot be flat. ∎

The matrix G is the leading-order adiabatic TCS metric:
g_exact = G + O(e^{−δL}), where L is the neck length and δ > 0 is
determined by the K3 geometry [5].

---

## 10. Stage 6: Spectral Fingerprints

> **Note**: All spectral measurements in this section were performed
> on the piecewise-constant atlas metric, before the flat attractor
> (§11) was discovered. The absolute eigenvalue values are properties
> of this flat background. The *degeneracy pattern* [1, 10, 9, 30],
> however, arises from the global topology (Kovalev twist structure)
> and remains valid (see §11.4).

### 10.1 Kovalev symmetrization

The Kovalev twist J is a discrete symmetry of K₇. Symmetrizing the
eigenspaces of the atlas Laplacian under J reorganizes the raw
degeneracies [1, 13, 12, 24] into:

| Band | Symmetrized degeneracy | Topological interpretation |
|------|----------------------|---------------------------|
| 1 | **1** | Temporal mode: cos(πt) |
| 2 | **10** | b₂(M₂): K3 harmonics of right building block |
| 3 | **9** | b₂(M₁) − 2: left building block modes |
| 4 | **30** | Mixed K3 × fiber modes |

### 10.2 Conformal invariance

Scanning 10 neck lengths L across a 33× range:

$$\lambda_1 \times L^2 = 8.812 \pm 0.004 \quad \text{(0.05\% variation)}$$

The K3 multiplicities [10, 9, 30] are **completely L-invariant** ---
topological, not geometric. A 128-point geometric scan shows that the
degeneracy [1, 10, 9, 30] is preserved if and only if all directions
are rescaled isotropically. Any direction-dependent deformation
shatters the pattern.

### 10.3 Statistical significance

We conducted a pre-registered statistical test (test statistics defined
before null-model simulations) with four null models, 100,000 trials
each on an A100:

| Null model | p-value |
|-----------|---------|
| Random metric perturbation (σ = 0.01) | < 10⁻⁵ |
| Shuffled Kovalev twist | < 10⁻⁵ |
| Random Betti numbers | 2.1 × 10⁻⁴ |
| Flat-spectrum (Poisson spacings) | 3.7 × 10⁻³ |

**Fisher combined p-value: 4.85 × 10⁻⁹ (5.8σ).**

A 1% metric perturbation destroys the degeneracy pattern. The spectral
fingerprint [1, 10, 9, 30] is a robust signature of the TCS topology,
not a numerical artifact.

### 10.4 Molecular orbital model

The four bands satisfy simple arithmetic relations:
Band₃ ≈ Band₂ + Band₁ (to 0.11%), Band₄ ≈ 2 × Band₂ (to 0.45%).
A three-parameter model (ν₀, ν_K3, t = ν₀/2) captures all four band
positions with sub-percent accuracy.

---

## 11. Stage 7: The Flat Attractor

### 11.1 Motivation

All results in Stages 1--6 were obtained with a metric that passed
every validation test applied: small torsion, correct determinant,
correct anisotropy, machine-precision interfaces, statistically
significant spectral fingerprints. A deeper diagnostic --- computing
the Riemann curvature tensor via full automatic differentiation --- revealed
a fundamental problem.

### 11.2 The curvature diagnostic

The full Riemann tensor R^i_{jkl} was computed via second-order automatic
differentiation through the PINN (not finite differences):

$$\|R_{\text{autograd}}\| \sim 2\text{--}3 \times 10^{-13}$$

The curvature is essentially zero. The metric is flat to machine
precision.

A comparison with finite-difference Riemann (used in all prior
curvature diagnostics) yielded:

$$\frac{\|R_{\text{FD}}\|}{\|R_{\text{autograd}}\|} \sim 10^8$$

All finite-difference curvature measurements were numerical noise,
amplified 10⁸× above the true signal. The V₇ fraction of the spin
connection was f_{V₇} = 0.330 ≈ 1/3 --- the noise equipartition value,
confirming flatness.

All three charts were tested independently, with the same result:
||δg||/||G₀|| ≈ 1%, Var(g)/g² ~ 10⁻¹¹, ||R_autograd|| ~ 10⁻¹³.

### 11.3 Interpretation

**The PINN converged to the trivial attractor**: a nearly-flat,
nearly-constant metric. For any constant metric g = G, the Christoffel
symbols vanish (Γ = 0), hence ∇φ = 0 automatically. The torsion loss
is zero regardless of the metric's holonomy. The optimizer found the
easiest path to a low total loss: make the metric constant.

This is a **general failure mode** for PINNs applied to special-holonomy
problems. The torsion-free condition (∇φ = 0) is satisfied by any
constant metric, and the optimizer will generically find this trivial
solution unless explicitly prevented. The failure is intrinsic to the
loss landscape, not specific to our architecture.

### 11.4 What remains valid

The critical distinction is between results that depend on the metric
being non-trivially curved and results that depend only on its global
structure:

| Result | Curvature-dependent? | Status |
|--------|---------------------|--------|
| det(g) = 65/32 | No | **Valid** |
| κ_T = 1/61 | No | **Valid** |
| Interface matching (10⁻¹²) | No | **Valid** |
| 28-number compression | No | **Valid** |
| K₇ not flat (b₃ = 77 > 35) | No | **Valid** |
| Spectral fingerprint [1,10,9,30] | No (see below) | **Valid** |
| Torsion small | Yes (trivially) | **Trivially true** |
| V₇ fraction measurements | Yes | **Invalidated** |

The spectral fingerprint arises from the global structure of the atlas
--- the piecewise-constant metric with Kovalev twist --- not from local
curvature. The Laplacian on a piecewise-constant metric is well-defined
and non-trivial. The 5.8σ significance remains valid.

### 11.5 Implications for the PINN community

1. **Check curvature independently.** Small torsion does not imply
   non-trivial geometry. Always compute the Riemann tensor via
   automatic differentiation, not finite differences.

2. **Finite differences on PINNs are unreliable for curvature.** The
   PINN output is smooth with second derivatives near machine epsilon.
   FD amplifies this to detectable but spurious signals.

3. **Trivial solutions are attractors.** Any loss function minimized
   by flat metrics will generically converge to one.

---

## 12. Stage 8: Toward Genuine Curvature

### 12.1 First-order barrier targets

**Why second-order targets fail.** Adding a loss targeting ||Rm||² > ε
directly does not work. At the flat attractor (Rm ≈ 0), the gradient
∂||Rm||²/∂θ ~ 2Rm · ∂Rm/∂θ ~ 10⁻¹³ --- a second-order quantity at a
degenerate zero. The optimizer cannot escape because the gradient
vanishes precisely where escape is needed.

**First-order alternative.** Instead, we target ||dg/dx||² --- the spatial
gradients of the metric. This is first-order with non-degenerate
gradient at zero. The connection is: dg/dx ≠ 0 ⟹ Γ ≠ 0 ⟹ R ≠ 0
generically. Implementation: a soft floor with annealing,
L_af = [softplus(γ₀ − ||dg/dx||²)]², γ₀ ramped from 10⁻⁸ to 10⁻⁴
over 400 epochs.

**Results (1000 epochs, RTX 2050):**

| Metric | Before (flat) | After | Factor |
|--------|--------------|-------|--------|
| E[||dg/dx||²] | 7.3 × 10⁻¹² | 6.82 | ×10¹¹ |
| E[||Rm||²] | 3.8 × 10⁻²⁶ | 4.7 × 10⁻⁴ | ×10²² |
| ∇φ | 4.1 × 10⁻¹² | 0.039 | bounded |

Curvature rose by 22 orders of magnitude. Torsion increased but
remained bounded. First-order barrier targets succeed where
second-order targets fail.

### 12.2 TCS warm-start parametrization

A more principled approach avoids the flat attractor entirely by starting
from a non-flat initial metric. The TCS construction provides such a
metric: in the neck region, the Cholesky factors of the two building
blocks must be interpolated, creating non-zero spatial gradients.

**Architecture.** The metric is parametrized as
g(t, x) = g_TCS(t) + δg(t, x), where g_TCS(t) smoothly interpolates
between Chol(G₀) and Chol(J^T G₀ J) in the neck [0.40, 0.75], and δg
is a learned perturbation.

### 12.3 Identifying the torsion bottleneck

A resolution diagnostic (torsion vs collocation density, N = 16 to 256)
confirmed that the torsion floor is a model property, not a
discretization artifact. Setting δg = 0 (pure baseline) gives
∇φ = 0.018 --- **85% of the total torsion comes from the TCS
baseline interpolation**, not from the learned correction. The torsion
is concentrated in the mid-neck [0.55, 0.65], where the two CY₃
geometries are stitched.

### 12.4 Interpolation profile optimization

Seven fixed profiles, four interpolation spaces, and seven blend widths
were systematically compared. The results:

- **Linear interpolation** in Cholesky space is optimal
  (∇φ = 0.010 baseline, vs quintic Hermite at 0.019)
- Torsion is proportional to max|dα/dt|: slower transitions produce
  less torsion
- Cholesky space outperforms log-Cholesky and direct metric
  interpolation by a factor of 2

### 12.5 PINN training on optimized baseline

Using the optimized baseline (nearly-linear profile, baseline
∇φ ≈ 0.010), 2000 epochs of PINN training yielded:

| Metric | Baseline | Trained | Factor |
|--------|---------|---------|--------|
| ∇φ | 0.010 | 0.015 (mean), **0.007** (best eval) | ---|
| ||Rm||² | 4.8 × 10⁻⁸ | 5.6 × 10⁻⁵ | ×1,200 |
| ||δg||/||g|| | ---| 0.88% | bounded |
| V₇ frac | 1/3 (noise) | **0.314** | < 1/3 |

The PINN adds ×1,200 curvature without increasing torsion above the
baseline. The best single evaluation reached ∇φ = 0.007 --- the first
time torsion dropped below 0.01. The V₇ fraction dipped to 0.314, the
first confirmed value below the noise-equipartition level of 1/3,
suggesting genuine G₂ holonomy structure.

Subsequent validation with best-checkpoint saving across 20+ seeds
confirmed that ∇φ = 0.007 was a stochastic transient. The validated
floor is **∇φ = 0.010**, consistent with the Cholesky baseline. The
0.007 value in the table above is retained as historical data.

### 12.6 Torsion decomposition: g₂-valued, not V₇-valued

A smooth SO(7) rotation R(t) = exp(Σ uₐ(t) Bₐ), with 7 cubic
B-spline profiles (56 total parameters), was applied to the Cholesky
blend to test whether the residual torsion has a V₇ component that
can be removed by gauge transformation.

The rotation reduced ∇φ from 0.011 to 0.010 (a 9% improvement), with
max|u| = 0.0006 --- effectively zero. **The baseline torsion is almost
entirely g₂-valued.** The V₇ component is negligible, ruling out the
entire class of gauge-based approaches.

### 12.7 Comparison of interpolation strategies

Given the negative results from gauge transformations, we
systematically compared the two natural interpolation strategies in
SPD(7): Cholesky interpolation (in the space of Cholesky factors)
and log-Euclidean geodesic interpolation (S(t) = (1−α)S_L + αS_R,
g(t) = exp(S(t))).

The log-Euclidean geodesic was tested with smooth B-spline corrections
δS(t) parametrized by 280 parameters (28 symmetric channels × 10 knots).
Care was taken to validate on held-out data: with fixed collocation
points, B-spline interpolators can achieve spuriously low training
errors while performing far worse on fresh evaluation points.

| Method | ∇φ (validated) | Parameters |
|--------|---------------|------------|
| Cholesky interpolation (linear) | **0.010** | 0 |
| Log-Euclidean geodesic | 0.020 | 0 |
| Log-Euclidean + B-spline correction | 0.019 | 280 |
| Log-Euclidean + data-driven compression | 0.019 | 40 |
| Cholesky + PINN δg | **0.010** (validated) | 1,070,471 |

The log-Euclidean geodesic produces 2× more torsion than Cholesky
interpolation, and no smooth spline correction improves upon it. Since
G_L and G_R quasi-commute (||[G_L, G_R]|| = 2×10⁻⁴), the Riemannian
geodesic in SPD(7) is effectively identical to the log-Euclidean path,
yielding the same torsion.

The Cholesky interpolation is genuinely superior --- it follows a
different path through SPD(7) that happens to produce lower torsion.
Combined with the PINN δg correction (which can explore the full
Sym(7) at each point), this yields the best validated result:
**∇φ = 0.010**.

### 12.8 The overdetermined system

The Cholesky interpolation defines a 1-dimensional path through the
space of G₂ structures. Torsion-freeness (∇φ = 0) imposes 245
independent conditions (the components of ∇φ in 7 dimensions). One
free function, 245 constraints: the system is massively overdetermined.

That ∇φ can be reduced to 0.010 by allowing the full 28 DOF of Sym(7)
to vary along t suggests that the torsion-free locus in metric space is
approachable --- but not from within any small subspace of corrections.

### 12.9 Summary of the curvature recovery trajectory

| Approach | ∇φ | ||Rm||² | Key finding |
|---------|-----|---------|-------------|
| Atlas (flat attractor) | ~10⁻⁶ * | ~10⁻¹³ | Trivially flat |
| First-order barrier | 0.039 | 4.7×10⁻⁴ | Escape confirmed |
| TCS warm-start (Hermite profile) | 0.026 | 2.1×10⁻⁴ | δg increases torsion |
| + gradient projection | 0.016 | 1.5×10⁻⁴ | Gradients quasi-orthogonal |
| Optimized linear baseline | **0.010** | 5.6×10⁻⁵ | Best validated (PINN δg) |
| SO(7) gauge rotation | 0.010 | ~10⁻⁸ | Torsion is g₂-valued |
| Log-Euclidean geodesic | 0.020 | ---| Cholesky is 2× better |
| Joyce iteration (φ₁ = φ₀ + dη) | 0.011 | ---| NULL (coclosure trivial) |
| True TCS + L-BFGS | 0.010 | ---| Same floor |
| Fiber-dependent g(t,θ) | 0.010 | ---| NULL (zero gradient) |
| KK gauge field g₀ᵢ(t) | 0.010 | ---| NULL (10 inits → baseline) |
| SO(7)/G₂ coset rotation | 0.010 | ---| NULL (marginal 0.25%) |
| **Bulk G₀ optimization** | **0.006** | 1.7×10⁻⁴ | **42% reduction** (a_t=2.47, a_f=1.60) |

\* Torsion in the flat atlas is small because the metric is flat, not
because the G₂ structure is non-trivially torsion-free. See §11.

† All ∇φ values are at the reference configuration (neck length
L ≈ 0.38). Values scale as L⁻²; the scaling law coefficients are
C = 1.4666 × 10⁻³ (isotropic G₀) and C = 8.462 × 10⁻⁴ (optimized
G₀*). See §12.10--12.11.

### 12.10 Exhaustive 1D metric optimization

The results of §12.5--12.9 left open whether the torsion floor
∇φ = 0.010 was an artifact of the Cholesky parametrization or a
geometric property of the TCS interpolation. We conducted an
exhaustive optimization campaign to settle this question.

**8-method convergence table.** Eight independent approaches,
spanning different parametrizations, optimization strategies, and
degrees of freedom, all converge to the same floor:

| Method | ∇φ | DOF | Result |
|--------|-----|-----|--------|
| Cholesky interpolation (linear) | 0.010 | 0 | Baseline |
| Optimized Cholesky (warm-start) | 0.010 | 28 | Same |
| PINN δg on Cholesky baseline | 0.010 | 1,070,471 | Same |
| Joyce iteration (φ₁ = φ₀ + dη) | 0.011 | ~10⁶ | NULL (+13%) |
| Scalar perturbation (4 modes) | 0.0095 | 4 | Marginal (-5%) |
| True TCS + L-BFGS | 0.010 | 28 | Same |
| Fiber-dependent g(t,θ) | 0.010 | +1 | NULL (zero gradient) |
| KK gauge field g₀ᵢ(t) | 0.010 | +6 | NULL (10 inits) |

**Scaling law.** Systematic calibration across multiple neck lengths
establishes the scaling exponents to 3 decimal places:

$$\boxed{\nabla\varphi(L) = 1.4666 \times 10^{-3} / L^2}$$

with |∂g/∂t| ~ L⁻¹·⁰⁰⁰, |Γ| ~ L⁻¹·⁰⁰⁰, ∇φ ~ L⁻²·⁰⁰⁰. The
profile shape is irrelevant: linear, tanh, bump, and exponential
cutoffs all converge to the same optimum.

**Joyce iteration.** A neural network learns a 2-form η such that
φ₁ = φ₀ + dη satisfies closure automatically (d²η = 0). The
coclosure d*φ₁ drops by a factor of 51.5 million (3 passes), but ∇φ
is unchanged: the network converges to η → 0. The coclosure was
already near-optimal; driving it to zero does not reduce the full
torsion norm.

**Fiber-dependent metric g(t,θ).** Adding a Fourier mode k = 1 on
the fiber direction yields purely quadratic coupling (ratio = 4.00)
with exactly zero gradient at δ = 0 (Fourier orthogonality). Six
optimization runs all converge to the 1D optimum. The 1D optimum IS
the 2D optimum.

**Kaluza-Klein gauge field.** Off-diagonal components g₀ᵢ(t) with
gradient ratio 0.76 (properly converged). Ten random initializations
all converge to the baseline. The KK gauge field is already optimal.

**Torsion budget.** The torsion decomposes into **71%
fiber-connection** (irreducible within any 1D metric family g(t)) and
**29% t-derivative**, constant across all neck lengths.

**Spectrum.** The Laplacian eigenvalues on the 1D metric follow
textbook Sturm-Liouville theory: λ₁·L² = π²⟨g^{tt}⟩, with ratios
1 : 4 : 9 : 16 : 25.

**The 1D metric program is CLOSED.** For any fixed bulk metric G₀,
no choice of parametrization, optimization strategy, perturbation
mode, fiber dependence, or off-diagonal component can reduce the
torsion below the geometric floor set by G₀.

### 12.11 Bulk metric optimization

The 1D closure (§12.10) exhausts all degrees of freedom in the
*interpolation* g(t) between two fixed endpoints G₀ and J^T G₀ J.
A distinct question is whether the *endpoints themselves* can be
improved. The bulk metric G₀ --- the 7×7 matrix defining the left
and right building blocks --- was isotropic (G₀ ≈ 1.10 × I₇ with
small K3 off-diagonals from the lattice structure). Changing G₀
changes the torsion floor that the 1D program converges to.

**Block-diagonal rescaling.** We parametrize a scaling matrix
S = diag(a_t, a_f, 1, 1, 1, 1, a_f) acting as G₀* = S · G₀ · S,
where a_t scales the seam direction and a_f scales the two fiber
directions (θ, ψ). The K3 directions are held fixed. This preserves
the G₂ structure while allowing anisotropic stretching of the bulk
metric.

A systematic optimization campaign (grid search + L-BFGS refinement
+ 4-parameter fine-tuning with K3 perturbations) finds optimal
parameters:

| Parameter | Value |
|-----------|-------|
| a_t | 2.47 |
| a_f | 1.60 |
| ε_K3 | 1.5 × 10⁻³ |
| ε_fiber | 1.0 × 10⁻⁵ |

**Results.** The optimized G₀* reduces the torsion scaling law
coefficient by **42.3%**:

$$\boxed{\nabla\varphi(L) = 8.462 \times 10^{-4} / L^2 \quad \text{(optimized } G_0^*\text{)}}$$

compared to 1.4666 × 10⁻³/L² for the isotropic G₀. The G₀*
eigenvalues split into three groups: {1.08, 1.10, 1.10, 1.10} (K3
block), {2.90, 2.90} (fiber pair), and {6.79} (seam), reflecting the
block-diagonal structure.

**Torsion budget shift.** With the optimized G₀*, the torsion
decomposes as **65% t-derivative** and **35% fiber-connection**
(compared to 71/29 for the isotropic G₀). The seam direction now
carries more relative torsion because the fiber directions have been
stretched to better accommodate the Kovalev twist.

**SO(7)/G₂ coset rotation.** Before optimizing G₀, we tested whether
a smooth SO(7)/G₂ rotation of the 3-form along the seam could reduce
fiber-connection torsion. The coset rotation yielded a marginal 0.25%
fiber torsion improvement with no effect on total ∇φ, confirming that
the residual torsion is not removable by frame rotations.

**Validation (20 seeds).** The new baseline was validated with:
- Torsion reproducibility: mean = 8.586 × 10⁻⁴, CV = 0.94%
- Scaling law: ∇φ · L² = 8.462 × 10⁻⁴ across L ∈ {0.5, 1, 2, 5}
  (spread 0.02%)
- Curvature: Kretschner scalar 1.7 × 10⁻⁸ to 2.4 × 10⁻³ (non-flat)
- Spectrum: λ₁·L² = π²⟨g^{tt}⟩, ratios 1:4:9:16:25 (Sturm-Liouville)

The G₀* matrix and optimized Chebyshev seam profile (40 nodes) define
the new official baseline for all subsequent experiments.

### 12.12 Landscape cartography

The 4-parameter optimization of §12.11 raises the question: is the
optimum unique? A systematic landscape exploration (287 evaluations)
addresses this through six phases: LHS screening (120 random starts,
94/120 non-SPD — the admissible domain is a small island in 4D),
1D sensitivity profiling, 2D grid scans, Powell refinement, and
Sobol sensitivity analysis.

**Key findings:**

| Quantity | Value |
|----------|-------|
| Total evaluations | 287 |
| SPD-admissible fraction | 22% |
| Basin count (6 starts) | 1 deep (unique) |
| Hessian condition number | 92,392 |
| Powell-refined ∇φ | 8.462 × 10⁻⁴ (identical to §12.11) |

The Hessian eigenvalue analysis reveals extreme anisotropy:
ε_f has curvature H = 155.6, ε_k has H = 41.2 (both with basin
width below resolution), while log(a_f) and log(a_t) have H ≈ 0.004
(basin widths ~2.5 and ~2.0 respectively). The condition number
κ = 92,392 means perturbations in the ε directions are catastrophic
while the log-scale parameters are gentle.

Sobol sensitivity indices: ε_k (0.26) > ε_f (0.24) > log(a_t) (0.15)
> log(a_f) (0.05). The most sensitive parameters have optimal values
near zero — anisotropy is lethal, not useful.

The optimum of §12.11 is the **unique global minimum** of the landscape.

### 12.13 Determinant gauge invariance and |φ|² = 42

An observation during landscape exploration (§12.12) suggested that
det(g) = 1.5 gives 11% lower ∇φ_code than the canonical 65/32.
We test whether this is a genuine improvement or a scale artifact.

**Theoretical prediction.** Under a global rescaling g → α·g (or
equivalently det(g) → β·det(g)), the code-reported torsion scales as
∇φ_code ∝ det^{3/7} while the proper (coordinate-invariant) torsion
scales as ∇φ_proper ∝ det^{-1/7}.

**Phase 1 (no re-optimization, 10 det values spanning 12× range):**
- ∇φ_code ∝ det^{0.428571} (predicted 3/7 = 0.428571) — **exact to
  8.4 × 10⁻¹⁵**
- ∇φ_proper ∝ det^{-0.142857} (predicted -1/7) — **exact**
- All proper-torsion ratios equal 1.000000 across the full range

**Phase 2 (with re-optimization, 8 det values, 200 steps each):**
- ∇φ_code / det^{3/7} = constant with CV = 0.0013%
- The optimizer finds identical geometry at every det value
- g^{tt} ∝ det^{-1/7} as predicted

**Phase 3 (spectral invariants):**
- Eigenvalue ratios 1:4:9:16:25 preserved at all det values
- λ₁·L² = π²·g^{tt} confirmed everywhere

**Conclusion:** det(g) is a **pure gauge parameter** with no physical
content. The apparent 11% improvement was exactly
(1.5/2.031)^{3/7} = 0.878, predicting -12.2% (observed: -12.2%).

**Bonus: |φ|²_proper = 42.000 = 7 × dim(G₂).** The proper norm of
the associative 3-form, computed from the metric-corrected volume
form, gives an exact topological invariant. This confirms the
calibration of the numerical framework.

### 12.14 Transverse spectrum and eigenfunctions

The 1D metric g(t) defines a longitudinal Sturm-Liouville problem.
The transverse directions (6D fiber perpendicular to the seam) define
independent eigenproblems whose spectrum governs the transition from
1D to fully 7D physics.

**Transverse metric profile.** The 6×6 transverse metric g_⊥ has two
groups of eigenvalues:
- Fiber (θ, ψ): g^{-1}_⊥ = 0.687 (2 near-degenerate, CV < 0.003%)
- K3 (4 directions): g^{-1}_⊥ = 1.32 (4 eigenvalues, CV < 0.06%)

These are nearly constant along the seam — the warped product
approximation is excellent.

**Full product spectrum.** From the flat-torus transverse Laplacian
with lattice vectors m ∈ Z⁶ (|m|_∞ ≤ 3): 117,648 total modes,
8,872 unique eigenvalue levels.

| Level | λ_⊥ | Degeneracy | Content |
|-------|------|------------|---------|
| 1 | 27.14 | 4 | Pure fiber (θ, ψ) |
| 2 | 52.36 | 2 | K3₄ |
| 3 | 52.39 | 4 | K3₃ |
| 4 | 52.41 | 2 | K3₁ |
| 5 | 54.28 | 4 | Mixed fiber |

Scale hierarchy: λ₁_⊥ / λ₁_long = 8.19× at L = 1.

**Regime transitions.** The critical crossing length is
L_cross = 0.35, where the first longitudinal and first transverse
eigenvalue coincide:
- L > 0.7: clean 1D regime (λ₂/λ₁ = 4.000 exact)
- L ~ 0.35–0.7: transition zone
- L < 0.35: transverse/degenerate regime

**Weyl's law.** The seam volume integral(√det · dt) = 1.4252.
At λ = 100: N_actual = 170 vs N_Weyl = 174, ratio = 0.976 (97.6%).

**Sturm-Liouville eigenfunctions.** Ten eigenvectors are extracted from
the 1D longitudinal problem:

| Property | Value |
|----------|-------|
| Eigenvalue ratios | 1 : 4 : 9 : 16 : 25 (exact) |
| λ₀ (zero mode) | 6 × 10⁻¹³ (numerical zero) |
| Orthonormality error | max off-diagonal = 1.86 × 10⁻¹⁰ |
| Deviation from cosines | ‖ψ_n − √2 cos(nπs)‖₂ < 4 × 10⁻⁶ |

The warping barely deforms the eigenfunctions — they are flat-space
cosines to 6 significant figures. This validates the warped product
structure of the metric.

### 12.15 Yukawa selection rules

The Sturm-Liouville eigenfunctions of §12.14 determine the
longitudinal Yukawa triple-overlap integrals
Y_{n₁,n₂,n₃} = ∫ ψ_{n₁} ψ_{n₂} ψ_{n₃} √det · dt.

**Selection rule.** Y_{n₁,n₂,n₃} ≠ 0 if and only if
n₁ ± n₂ ± n₃ = 0 for some sign combination.

| Metric | Value |
|----------|-------|
| Allowed triples (first 6 modes) | 9 / 56 |
| Universal coupling |Y| | 0.5923 = 1/√(2V) |
| Rule-violating residuals | ~10⁻⁷ (6 orders below allowed) |
| Full 7D triples (with transverse) | 200 valid (inheriting longitudinal structure) |

The coupling is **universal**: all 9 allowed triples have identical
|Y| = 0.5923, where V = 1.4252 is the seam volume. This is the
flat-space result, preserved by the warped metric to better than 10⁻⁶.

**Metric-corrected universality.** Integrating the seam metric g(t)
with Clenshaw-Curtis quadrature (40 Chebyshev-Gauss-Lobatto nodes):

| Quantity | Value |
|----------|-------|
| Metric-corrected |ỹ| | 1.1938 |
| CV | 0.0001% |
| max/min ratio | 1.000003 |
| Significant couplings | 210 (all consistent) |

Universality is preserved under the full metric with negligible
distortion.

### 12.16 G₂ decomposition and cup product Yukawas

Moving beyond scalar Yukawas, we compute the algebraic cup product
Y_{abI} = ∫ ω_a ∧ ω_b ∧ ψ_I over the torus T⁷, where
ω_a ∈ Ω² (2-forms) and ψ_I ∈ Ω³ (3-forms), and decompose by
G₂ irreducible representations.

**G₂ decomposition of forms (textbook verification).**

The Hodge-star composed with φ-wedge acts on Ω² with eigenvalues
**+2 (×7)** and **−1 (×14)** — exactly the G₂ representation decomposition:

$$\Omega^2 = \Omega^2_7 \oplus \Omega^2_{14} \qquad (7 + 14 = 21 = b_2)$$

$$\Omega^3 = \Omega^3_1 \oplus \Omega^3_7 \oplus \Omega^3_{27} \qquad (1 + 7 + 27 = 35)$$

Projector validation: P₇² = P₇ (error 0), P₁₄² = P₁₄ (error 4 × 10⁻¹⁶),
P₇ + P₁₄ = I (error 0). The norm ‖φ‖² = 7.

**Cup product on irreps.**

| Channel | Nonzero fraction | Max |Y| |
|---------|-----------------|---------|
| Ω²₇ × Ω²₇ × Ω³₇ | **0/343 (= 0)** | 0 |
| Ω²₇ × Ω²₇ × Ω³₁ | significant | 0.756 |
| Ω²₇ × Ω²₇ × Ω³₂₇ | 1299/5145 (25%) | 0.545 |
| Ω²₁₄ × Ω²₁₄ × Ω³ | **6860/6860 (100%)** | dense |

The key result is the **G₂ selection rule**:
$$\boxed{Y(\Omega^2_7 \times \Omega^2_7 \times \Omega^3_7) = 0}$$

This vanishing is exact (not numerical) and follows from
representation theory: the tensor product 7 ⊗ 7 decomposes as
1 ⊕ 7 ⊕ 14 ⊕ 27, which does not contain the dual of 7 in the
relevant coupling channel.

**Kovalev twist (J-action) and orbifold selection.**

The Kovalev twist J acts on forms with: order 8, det = −1
(orientation-reversing). J-invariant subspaces:
dim(Ω²)^J = 3, dim(Ω³)^J = 6. J mixes Ω²₇ and Ω²₁₄
(off-diagonal norm = 2.31) — J is not in G₂.

The cup product on the J-invariant subspace (3 × 3 × 6) is
**identically zero**. Forms surviving the orbifold identification have
vanishing torus Yukawa. Physical Yukawas originate exclusively from
the J-anti-invariant sector: Y(anti₂ × anti₂ × anti₃) has
14/112 nonzero entries, with max |Y| = 1.0.

**Normalized Yukawas.** After mass-matrix normalization
(cond(M₂) = 16.6, cond(M₃) = 44.1), all 210 nonzero entries
have |ỹ| = 0.3326 with ratio max/min = 1.00 and std = 0.000 —
**universal coupling**, consistent with the longitudinal result of §12.15.

---

## 13. Discussion

### 13.1 Summary of contributions

1. **First PINN computation of a G₂-structure on a compact model.**
   To our knowledge, no previous work has applied neural networks to
   compute torsion-free G₂-structures on any domain.

2. **Machine-precision constraints.** det(g) = 65/32 to 15 significant
   figures on T⁷, to 0.0002% on the atlas.

3. **Empirical robustness.** Five independent seeds converge to the same
   metric to 9 significant digits on T⁷.

4. **Topology gap.** Local geometry on T⁷ cannot reproduce K₇-like
   anisotropy.

5. **Multi-chart atlas with machine-precision interfaces.** The
   Kovalev twist is handled at 10⁻¹² precision.

6. **Metric compression.** 1,070,471 parameters compress to 28 numbers,
   with κ_T = 1/61 to 7 significant figures.

7. **Spectral fingerprints.** Degeneracies [1, 10, 9, 30] encode TCS
   building-block Betti numbers at 5.8σ (Fisher combined, 4 null
   models, pre-registered).

8. **Flat attractor disclosure.** The atlas metric is flat
   (||R|| ~ 10⁻¹³). This failure mode is general and should be
   expected in any PINN whose loss is minimized by constant metrics.

9. **Curvature recovery.** First-order barriers and TCS warm-start
   produce genuine curvature (||Rm||² up to 5.6 × 10⁻⁵) with
   controlled torsion (∇φ = 0.010).

10. **Interpolation comparison.** Cholesky interpolation outperforms
    log-Euclidean geodesic by 2× for torsion minimization. Residual
    torsion is g₂-valued, ruling out gauge-based corrections.

11. **Torsion scaling law and 1D closure.** ∇φ(L) = 1.47 × 10⁻³/L²,
    confirmed across 8 independent methods. The 1D metric optimization
    program is closed.

12. **Bulk metric optimization.** Block-diagonal rescaling of G₀
    reduces torsion by 42% to ∇φ(L) = 8.46 × 10⁻⁴/L², validated
    across 20 seeds (CV < 1%). The torsion budget shifts from 71/29
    to 65/35 (t/fiber).

13. **Landscape uniqueness.** Systematic exploration (287 evaluations)
    confirms the optimum is the unique global minimum with Hessian
    condition number 92,392.

14. **Determinant gauge invariance.** det(g) is a pure gauge parameter:
    ∇φ_code ∝ det^{3/7} (verified to 8.4 × 10⁻¹⁵ precision). The
    proper 3-form norm |φ|² = 42 = 7 × dim(G₂) is an exact
    topological invariant.

15. **Transverse spectrum.** Full 7D product spectrum (117,648 modes,
    8,872 unique levels), with Weyl's law at 97.6% accuracy. Critical
    crossing length L_cross = 0.35 separates 1D and transverse regimes.

16. **Yukawa selection rules.** n₁ ± n₂ ± n₃ = 0, with 9/56 allowed
    triples and universal coupling |Y| = 0.5923 = 1/√(2V). Universality
    preserved under full metric (CV = 0.0001%).

17. **G₂ cup product decomposition.** Y(Ω²₇ × Ω²₇ × Ω³₇) = 0 (exact
    G₂ selection rule). All J-invariant Yukawas vanish. Physical
    Yukawas originate exclusively from the J-anti-invariant sector.

### 13.2 Comparison with the state of the art

| Domain | Best result | Reference |
|--------|-----------|-----------|
| CY metrics (Kähler) | Machine precision on quintic | cymyc [17] |
| G₂ topology (not metric) | ML for Sasakian/G₂ invariants | Aggarwal et al. [19] |
| G₂ structure (contact CY₇) | NN-learned 3-form on CY link in S⁹ | Heyes et al. [22] |
| G₂ flow numerics | Cohomogeneity-one solitons | [16] |
| G₂ spectral estimates | Neck-stretching theory | Langlais [20] |
| **G₂ metric (this work)** | **28 numbers, 5.8σ spectral fingerprint, ∇φ ~ 8.5 × 10⁻⁴/L² (unique basin), Yukawa selection rule Y(7×7×7) = 0, |φ|² = 42** | ---|

### 13.3 Limitations

1. **TCS model, not full manifold.** The atlas is a computational model
   inspired by the TCS construction, not a proven faithful
   representation of K₇.

2. **Analytical warm-start.** The PINN starts from an analytical target,
   inheriting its structure.

3. **Residual torsion.** ∇φ(L) = 8.46 × 10⁻⁴/L² (after bulk
   optimization) is not yet within the small-torsion regime of Joyce's
   theorem, though the L⁻² scaling shows it is achievable with longer
   neck lengths.

4. **Spectral bridge limitations.** The Galerkin basis (Fourier on T⁷)
   is not adapted to K₇ geometry.

5. **The flat attractor was not detected until autograd curvature
   diagnostics were applied.** This underscores the importance of
   independent curvature checks for any PINN approach to geometric PDEs.

6. **Overfitting risk in collocation-based optimization.** B-spline
   corrections with fixed collocation points produced spuriously low
   training errors (§12.7). Fresh-point validation is essential.

### 13.4 Spectral status

The spectral fingerprints of §10 --- in particular the degeneracy
pattern [1, 10, 9, 30] at 5.8σ --- were computed on the
piecewise-constant (flat) atlas metric. The absolute eigenvalue
λ₁ × H* = 898 is a property of this flat background, not of a
curved G₂ metric, and should not be compared directly to the GIFT
prediction of 14 [12]. The spectrum of the non-trivially curved
metrics (Stage 8, with optimized G₀*) has not yet been investigated.
This remains an open question (§13.5).

### 13.5 Open questions

1. **Can torsion reach the Joyce threshold?** The scaling law
   ∇φ = 8.46 × 10⁻⁴/L² shows this is achievable with longer neck
   lengths. At L = 10, ∇φ = 8.5 × 10⁻⁶.

2. **What is the spectrum of the curved metric?** Does the fingerprint
   [1, 10, 9, 30] survive when genuine curvature is present?

3. **Is the flat attractor universal?** We conjecture it affects any
   PINN trained on ∇φ = 0 without anti-flatness constraints.

4. **Geodesic interpolation in SPD(7)/G₂.** ANSWERED: Tested.
   Cholesky is 2× better. No quotient geometry improvement found
   (§12.7).

5. **Fiber-dependent φ(t,θ) via Joyce η correction.** The remaining
   path to attack the 35% fiber-connection torsion (65% after bulk
   optimization).

6. **Further G₀ optimization.** The 4-parameter block-diagonal scaling
   already captures 42% of the available improvement. Can the remaining
   degrees of freedom in the full 28-parameter G₀* yield further gains?
   (This is orthogonal to the 1D closure: the 1D program is closed
   *for each fixed G₀*, but the choice of G₀ itself remains open.)

---

## 14. Conclusion

We have presented a systematic approach to computing torsion-free
G₂-structures on compact 7-manifolds, progressing from the flat torus
to the frontier of numerical G₂ geometry including spectral analysis,
Yukawa couplings, and G₂ representation-theoretic decomposition.

**Structural discoveries.** The PINN metric compresses to 28 numbers.
The anisotropy κ_T = 1/61 emerges to 7 significant figures. The
spectral degeneracies [1, 10, 9, 30] encode TCS building-block Betti
numbers at 5.8σ significance. The proper 3-form norm
|φ|² = 42 = 7 × dim(G₂) is an exact topological invariant. These
results depend on the global metric structure and the Kovalev twist,
not on local curvature.

**A general failure mode.** The trained atlas metric is essentially flat
(||R|| ~ 10⁻¹³). The small torsion reported in Stages 1--4 is satisfied
trivially. This "flat attractor" should be expected in any PINN whose
loss is minimized by constant metrics. We report this failure as a
contribution to the PINN literature.

**A viable path forward.** First-order barrier targets escape the flat
attractor (curvature × 10²²). TCS warm-start with optimized Cholesky
interpolation achieves ∇φ = 0.010 with genuine curvature, governed by
the scaling law ∇φ(L) = 1.47 × 10⁻³/L². Exhaustive 1D optimization
across 8 independent methods confirms this floor is geometric, closing
the 1D metric program. Bulk metric optimization --- block-diagonal
rescaling of G₀ with optimal (a_t, a_f) = (2.47, 1.60) --- then
reduces the torsion by a further 42%, yielding ∇φ(L) = 8.46 × 10⁻⁴/L²
validated across 20 seeds (CV < 1%). The landscape has a unique global
minimum (Hessian condition number 92,392), and det(g) is pure gauge
(verified to 8.4 × 10⁻¹⁵ precision). Cholesky interpolation outperforms
the log-Euclidean geodesic by 2×.

**Spectral and algebraic structure.** The full 7D product spectrum
(117,648 modes) satisfies Weyl's law at 97.6%, with a critical crossing
length L_cross = 0.35 separating 1D and transverse regimes. Longitudinal
Yukawa couplings obey the selection rule n₁ ± n₂ ± n₃ = 0 with
universal coupling |Y| = 1/√(2V), preserved under the full metric
(CV = 0.0001%). The G₂ decomposition of cup product Yukawas reveals an
exact selection rule Y(Ω²₇ × Ω²₇ × Ω³₇) = 0, and all J-invariant
Yukawas vanish — physical couplings originate exclusively from the
anti-invariant sector.

This is, to our knowledge, the first application of physics-informed
neural networks to exceptional holonomy geometry. The progression
provides both a toolkit for numerical G₂ geometry and a
characterization of the obstacles --- in particular the flat attractor
--- that must be overcome in PDE-constrained optimization on spaces of
Riemannian metrics.

---

## Acknowledgments

The mathematical foundations draw on work by Dominic Joyce, Alexei
Kovalev, Mark Haskins, Johannes Nordström, and collaborators on G₂
manifold construction. The standard associative 3-form φ₀ originates
from Harvey and Lawson's foundational work on calibrated geometries.
Spectral estimates for TCS manifolds follow the recent work of Langlais.
Computational resources were provided by Google Colab (A100 GPU) and a
local NVIDIA RTX 2050 (4 GB VRAM).

---

## Author's note

This work was developed through sustained collaboration between the
author and several AI systems, primarily Claude (Anthropic), with
contributions from GPT (OpenAI), Gemini (Google), Grok (xAI), and Kimi
for specific mathematical and editorial insights. The PINN architecture,
training protocols, spectral bridge computation, flat-attractor
diagnosis, and manuscript drafting emerged from iterative dialogue
sessions. This collaboration follows the transparent crediting approach
advocated for AI-assisted scientific research.

The methodology presented here stands on its mathematical merits
independently of its mode of development. Mathematics is evaluated on
results, not résumés.

---

## References

[1] Harvey, R. & Lawson, H.B. (1982). Calibrated geometries. *Acta Math.*
    148, 47--157.

[2] Bryant, R.L. (1987). Metrics with exceptional holonomy. *Ann. Math.*
    126(3), 525--576.

[3] Joyce, D.D. (1996). Compact Riemannian 7-manifolds with holonomy G₂.
    I, II. *J. Diff. Geom.* 43(2), 291--328 and 329--375.

[4] Joyce, D.D. (2000). *Compact Manifolds with Special Holonomy*. Oxford
    University Press.

[5] Kovalev, A.G. (2003). Twisted connected sums and special Riemannian
    holonomy. *J. Reine Angew. Math.* 565, 125--160.

[6] Corti, A., Haskins, M., Nordström, J. & Pacini, T. (2015). G₂-manifolds
    and associative submanifolds via semi-Fano 3-folds. *Duke Math. J.*
    164(10), 1971--2092.

[7] Raissi, M., Perdikaris, P. & Karniadakis, G.E. (2019). Physics-informed
    neural networks. *J. Comput. Phys.* 378, 686--707.

[8] Cai, S. et al. (2021). Physics-informed neural networks for
    fluid mechanics. *Acta Mechanica Sinica* 37, 1727--1738.

[9] Hermann, J. et al. (2020). Deep-neural-network solution of the
    electronic Schrödinger equation. *Nature Chemistry* 12, 891--897.

[10] Liao, S. & Petzold, L. (2023). Physics-informed neural networks for
     Einstein field equations. arXiv:2302.10696.

[11] Braun, A.P. et al. (2018). Infinitely many M2-instanton corrections
     to M-theory on G₂-manifolds. *JHEP* 2018, 101.

[12] de La Fournière, B. (2026). Geometric Information Field Theory:
     Standard Model parameters from E₈ × E₈ topology. Zenodo.
     DOI: 10.5281/zenodo.18643070.

[13] Lotay, J.D. & Wei, Y. (2019). Laplacian flow for closed G₂ structures.
     *Geom. Funct. Anal.* 29, 1048--1110.

[14] Donaldson, S.K. (2005). Some numerical results in complex differential
     geometry. *Pure Appl. Math. Q.* 1(2), 297--318.

[15] Brandhuber, A. et al. (2001). Gauge theory at large N and new G₂
     holonomy metrics. *Nuclear Phys. B* 611, 179--204.

[16] Foscolo, L., Haskins, M. & Nordström, J. (2021). Complete non-compact
     G₂-manifolds from asymptotically conical Calabi-Yau 3-folds.
     *Duke Math. J.* 170(15), 3323--3416.

[17] Berglund, P. et al. (2024). Machine learning Calabi-Yau metrics.
     *JHEP* 2024, 087.

[18] Douglas, M.R. et al. (2007). Numerical Calabi-Yau metrics.
     *J. Math. Phys.* 49, 032302.

[19] Aggarwal, D. et al. (2024). Machine learning Sasakian and G₂ topology.
     *Phys. Lett. B* 850, 138517. arXiv:2310.03064.

[20] Langlais, T. (2025). Analysis and spectral theory of neck-stretching
     problems. *Commun. Math. Phys.* 406, 7. arXiv:2301.03513.

[21] de La Fournière, B. (2026). A PINN Framework for Torsion-Free G₂
     Structures (v1). Zenodo. DOI: 10.5281/zenodo.18643069.

[22] Heyes, E., Hirst, E., Sa Earp, H.N. & Silva, T.S.R. (2026). Neural
     and numerical methods for G₂-structures on contact Calabi-Yau
     7-manifolds. arXiv:2602.12438.

---

## Appendix A. Constants

**Standard topological/algebraic constants** (independent of GIFT):

| Symbol | Value | Definition | Source |
|--------|-------|------------|--------|
| dim(K₇) | 7 | Manifold dimension | Standard |
| dim(G₂) | 14 | Holonomy group dimension | Standard |
| dim(E₈) | 248 | Exceptional Lie algebra dimension | Standard |
| rank(E₈) | 8 | Cartan subalgebra dimension | Standard |
| b₂(K₇) | 21 | Second Betti number | [5, 6] |
| b₃(K₇) | 77 | Third Betti number | [3, 5, 6] |

**GIFT-derived constants** (from [12]; see §1.4 for status):

| Symbol | Value | Definition |
|--------|-------|------------|
| H* | 99 | b₂ + b₃ + 1 |
| κ_T | 1/61 | Torsion coupling constant |
| det(g) | 65/32 | Metric determinant |

## Appendix B. Reproducibility

| Resource | Location |
|----------|----------|
| Atlas notebook | `notebooks/colab_atlas_g2_metric.ipynb` |
| Multi-seed notebook | `notebooks/colab_p2_multiseed.ipynb` |
| TCS warm-start training | `notebooks/run_a29_*` |
| SO(7) rotation solver | `notebooks/run_a30_seam_solver.py` |
| SPD(7) path comparison | `notebooks/run_a31_spd_path_solver.py` |
| Data-driven compression | `notebooks/run_a33_datadriven_compression.py` |
| Fresh PINN validation | `notebooks/run_a36_*` |
| Joyce iteration | `notebooks/run_a41_*` |
| Scaling law calibration | `notebooks/run_a46_*`, `notebooks/run_a47_*` |
| Fiber-dependent metric | `notebooks/run_a48_*` |
| KK gauge field | `notebooks/run_a49_*` |
| SO(7)/G₂ coset rotation | `notebooks/run_a50_*` |
| Bulk G₀ optimization | `notebooks/run_a51_*` |
| Global baseline lock | `notebooks/run_a52_*` |
| Repository | github.com/gift-framework |

**Hardware**: NVIDIA A100-SXM4 (Google Colab) for Stages 1--6;
NVIDIA RTX 2050 (4 GB VRAM, local) for Stages 7--8.

## Appendix C. Changes from v1

| Section | Change |
|---------|--------|
| Abstract | Rewritten for 8-stage arc |
| §1 | Updated overview |
| §2.5 | New: so(7) = g₂ ⊕ V₇ decomposition |
| §9 | **New**: Metric compression (28 numbers, κ_T = 1/61) |
| §10 | **New**: Spectral fingerprints ([1,10,9,30], 5.8σ) |
| §11 | **New**: Flat attractor diagnosis |
| §12 | **New**: Curvature recovery and interpolation comparison |
| §12.10 | **New**: Exhaustive 1D optimization, scaling law, 1D program closed |
| §12.11 | **New**: Bulk G₀ optimization (42% reduction, A51--A52) |
| §13--14 | Expanded discussion, rewritten conclusion |

---
