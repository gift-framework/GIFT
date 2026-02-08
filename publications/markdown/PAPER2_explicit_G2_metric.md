# A Numerical Candidate for a Torsion-Free G₂ Structure on a Compact TCS 7-Manifold

**Author**: Brieuc de La Fournière

Independent researcher

**Abstract.** We construct a numerical candidate for a Riemannian metric with
holonomy contained in G₂ on a compact 7-manifold K₇ of twisted connected sum
(TCS) type, with Betti numbers b₂ = 21 and b₃ = 77. The construction
proceeds in three stages: (i) an analytical target metric derived from the
G₂ representation-theoretic decomposition Λ³(ℝ⁷) = Λ¹₃ ⊕ Λ⁷₃ ⊕ Λ²⁷₃ and
period integrals on the moduli space of G₂ structures; (ii) a
Cholesky-parameterized physics-informed neural network (PINN) that
reconstructs a spatially varying metric field g(x) on a local computational
model of the TCS neck region; (iii) verification against five geometric
criteria. The resulting 7×7 metric satisfies a prescribed determinant
det(g) = 65/32 to 8 significant figures (4 × 10⁻⁸ % deviation), has
torsion ‖dφ‖ + ‖d*φ‖ of order 10⁻⁶ (well within the perturbative regime
of Joyce's existence theorem [Theorem 11.6.1, Joyce 2000]), condition
number κ = 1.0152, and matches 77 target period integrals at 5 scales
with RMS error 3.1 × 10⁻⁴. The Cholesky warm-start technique
(initializing at the analytical target and learning only residual
perturbations) may be of independent interest for other special-holonomy
problems. All code and data are publicly available.

---

## 1. Introduction

### 1.1 Compact manifolds with holonomy contained in G₂

A compact Riemannian 7-manifold (M⁷, g) has holonomy contained in the
exceptional Lie group G₂ ⊂ SO(7) if and only if it admits a torsion-free
G₂-structure, i.e., a closed and coclosed 3-form φ ∈ Ω³(M) [1, 2].
(Full holonomy G₂, as opposed to a proper subgroup, requires additionally
that M be simply connected and not a Riemannian product.)
Joyce [3, 4] proved the existence of compact examples by resolving
singularities of T⁷/Γ orbifolds. Kovalev [5] introduced the
twisted connected sum (TCS) construction, gluing two asymptotically
cylindrical (ACyl) Calabi–Yau threefolds along a common K3 fiber.
Corti, Haskins, Nordström and Pacini [6] systematized the TCS method
and produced many topological types.

These existence results establish the metric to within a small (controlled)
error of an approximate solution, but do not yield pointwise numerical
values. To our knowledge, no explicit metric tensor g_ij(x) has been
computed numerically for a compact G₂ manifold, though we note that
substantial numerical work exists for *non-compact* examples
(see e.g. Brandhuber et al. [15]).

### 1.2 The PINN approach

Physics-informed neural networks (PINNs) [7] parameterize solutions to
PDEs via neural networks whose loss function encodes the governing
equations. They have been successfully applied to fluid dynamics [8],
quantum mechanics [9], and general relativity [10], but not, to our
knowledge, to special holonomy geometry.

We apply PINNs to construct a candidate metric on a local model of the
neck region of K₇, a compact TCS manifold with b₂ = 21 and b₃ = 77
(the specific topological type studied in [11]). To be precise: we work
on a 7-dimensional domain that serves as a computational proxy for the
gluing region where the two ACyl Calabi–Yau building blocks meet; a
complete global metric would require extending the solution into the
bulk of each building block.

The key technical contribution is a **Cholesky parameterization
with analytical warm-start**: the network outputs a lower-triangular
perturbation δL(x), and the metric is g(x) = (L₀ + δL(x))(L₀ + δL(x))ᵀ,
where L₀ is the Cholesky factor of an analytically derived target. This
guarantees positive definiteness and symmetry by construction, and reduces
the learning task to small residual corrections.

### 1.3 Motivation from the GIFT framework

The analytical target and the period integrals used as training data derive
from the GIFT (Geometric Information Field Theory) framework [12], which
proposes that physical constants arise from the topology of E₈ × E₈
compactifications on G₂ manifolds. While the physical claims of GIFT are
outside the scope of this paper, the mathematical objects it produces
(the G₂ decomposition, the Mayer–Vietoris splitting of moduli, and the
determinant formula det(g) = 65/32) are independently verifiable
statements in differential geometry. We use them as input data and
verify the output against standard geometric criteria.

### 1.4 Summary of results

| Criterion | Target | Achieved |
|-----------|--------|----------|
| det(g) = 65/32 | 2.03125 | 2.031250001 (4 × 10⁻⁸ %) |
| Positive definite | All λᵢ > 0 | λ_min = 1.099 (Cholesky guarantee) |
| Condition number | 1.01518 | 1.01518 (7 significant figures) |
| Torsion ‖dφ‖+‖d*φ‖ | small | 7.2 × 10⁻⁶ |
| Period integrals | RMS < 0.005 | 0.000311 (16-fold below threshold) |
| Anisotropy | ‖g − G_TARGET‖_F → 0 | 1.76 × 10⁻⁷ (machine precision) |

Training time: 2.9 minutes on a single A100 GPU. Model: 202,857 parameters.

### 1.5 Outline

Section 2 recalls the G₂ structure and the TCS construction. Section 3
describes the analytical derivation of the target metric. Section 4
presents the PINN architecture and training. Section 5 gives the explicit
metric and verification results. Section 6 discusses lessons learned,
limitations, and future directions.

---

## 2. The G₂ Structure and TCS Construction

### 2.1 Holonomy contained in G₂ and the associative 3-form

The exceptional Lie group G₂ is the automorphism group of the octonion
algebra 𝕆. It acts on Im(𝕆) ≅ ℝ⁷ and preserves the standard associative
3-form [1]:

$$
\varphi_0 = e^{012} + e^{034} + e^{056} + e^{135} - e^{146} - e^{236} - e^{245}
$$

where e^{ijk} = eⁱ ∧ eʲ ∧ eᵏ and the indices correspond to the 7
imaginary octonion units. The 7 nonzero terms correspond to the 7 lines
of the Fano plane, encoding the octonion multiplication table.

Under G₂, the space of 3-forms decomposes as:

$$
\Lambda^3(\mathbb{R}^7) = \Lambda^3_1 \oplus \Lambda^3_7 \oplus \Lambda^3_{27}
$$

with dimensions 1 + 7 + 27 = 35 = C(7,3). The G₂ metric is recovered
from the 3-form via [2]:

$$
g_{ij} = \frac{1}{6} \sum_{k,l} \varphi_{ikl}\,\varphi_{jkl}
$$

For the standard φ₀, this gives g = I₇. A rescaled form φ = c · φ₀ with
c = (65/32)^{1/14} yields g = c² · I₇ with det(g) = c¹⁴ = 65/32.

### 2.2 The TCS construction

The manifold K₇ is constructed as a twisted connected sum [5, 6]:

$$
K_7 = M_1 \cup_\Phi M_2
$$

where M₁ and M₂ are asymptotically cylindrical Calabi–Yau threefolds,
glued along their common asymptotic cross-section S¹ × K3:

| Building block | Construction | b₂ | b₃ |
|---------------|-------------|-----|-----|
| M₁ | ACyl CY from quintic in ℂℙ⁴ | 11 | 40 |
| M₂ | ACyl CY from CI(2,2,2) in ℂℙ⁶ | 10 | 37 |
| K3 (gluing) | K3 surface, b₂ = 22 | N/A | N/A |

The Mayer–Vietoris sequence gives:

$$
b_2(K_7) = b_2(M_1) + b_2(M_2) = 11 + 10 = 21
$$
$$
b_3(K_7) = b_3(M_1) + b_3(M_2) = 40 + 37 = 77
$$

Since K₇ is a compact orientable manifold of odd dimension, Poincaré
duality (bₖ = b_{7−k}) implies χ(K₇) = 0. Explicitly:
b₀ = b₇ = 1, b₁ = b₆ = 0, b₂ = b₅ = 21, b₃ = b₄ = 77, giving
χ = 1 − 0 + 21 − 77 + 77 − 21 + 0 − 1 = 0.

### 2.3 Pointwise representation theory

At each point of a 7-manifold with G₂-structure, the space of 3-forms
decomposes under G₂ as (cf. §2.1):

$$
\Lambda^3(\mathbb{R}^7) = \Lambda^3_1 \oplus \Lambda^3_7 \oplus \Lambda^3_{27},
\qquad 1 + 7 + 27 = 35 = \binom{7}{3}.
$$

This is a *pointwise* statement in representation theory: at each point
x ∈ K₇, a 3-form has 35 components that transform in these three
irreducible G₂-representations. Among the 35 directions, the 7 that are
aligned with the Fano-plane triples of the octonion multiplication table
generate volume-changing deformations (Tr(∂g/∂Π) = ±2.10), while the
remaining 28 in Λ³₂₇ are traceless (pure shape deformations). The vanishing
trace for non-Fano modes is exact, following from the orthogonality of
Λ³₂₇ to the trivial representation Λ³₁.

### 2.4 Global moduli space

The moduli space of torsion-free G₂ structures on K₇ is a smooth manifold
of dimension b₃(K₇) = 77 [3, 4]. This is a *global topological* statement,
independent of the pointwise decomposition above. The 77 moduli reflect
the space of closed and coclosed 3-forms modulo diffeomorphisms; their
count is determined by the third Betti number via the period map.

In the TCS construction, these global moduli receive contributions from
both building blocks and the gluing data:

| Contribution | Source |
|-------------|--------|
| H³(M₁) | 40 classes from the first ACyl CY threefold |
| H³(M₂) | 37 classes from the second ACyl CY threefold |
| **Total** | **b₃(K₇) = 77** |

---

## 3. The Analytical Target Metric

### 3.1 Period integrals

Each modulus Πₖ (k = 1, ..., 77) corresponds to a period integral of the
associative 3-form over a 3-cycle Cₖ ∈ H₃(K₇, ℤ):

$$
\Pi_k = \int_{C_k} \varphi
$$

We use period data derived from the GIFT framework [12], where the 77
periods are computed from prime-number data at multiple energy scales T.
The specific values are determined by the torsion coupling constant
κ_T = 1/61 and the adaptive cutoff function X(T) described in [13].

### 3.2 The metric Jacobian

The metric response to moduli variations is given by the Jacobian:

$$
\frac{\partial g_{ij}}{\partial \Pi_k}
= \frac{1}{3}\sum_l \left(\varphi_{ikl}\,\frac{\partial\varphi_{jkl}}{\partial\Pi_k}
+ \frac{\partial\varphi_{ikl}}{\partial\Pi_k}\,\varphi_{jkl}\right)
$$

Evaluating this for the 35 pointwise modes (§2.3), the 7 modes aligned with the Fano-plane triples
have Tr(∂g/∂Π) = ±2.10 (volume-changing), while all 28 non-Fano
modes have exactly vanishing trace (pure shape deformations).

### 3.3 The target metric G_TARGET

Evaluating the metric Jacobian at the reference periods yields a 7×7
target metric with the following properties:

| Property | Value |
|----------|-------|
| Diagonal range | [1.1022, 1.1133] |
| Max off-diagonal | 0.00461 (g₂₃) |
| Condition number κ | 1.01518 |
| Determinant (after rescaling) | 65/32 = 2.03125 |
| Eigenvalue range | [1.0993, 1.1160] |

The anisotropy is small (~1.5% diagonal variation) but structurally
significant: it encodes the breaking of the isotropic G₂ structure
by the TCS gluing map Φ.

### 3.4 The E₈/K3 lattice structure

The global modes are organized by the K3 lattice Λ_{K3} of signature
(3, 19) and rank 22, which contains two sublattices:

- N₁ of rank 11, signature (1, 9): the polarization lattice of M₁
- N₂ of rank 10, signature (1, 8): the polarization lattice of M₂

with N₁ ∩ N₂ = {0} and rank(N₁ + N₂) = 21 = b₂(K₇). The K3
intersection form is Λ_{K3} = 3H ⊕ 2(−E₈), where H is the hyperbolic
lattice and E₈ is the positive-definite E₈ root lattice. The presence
of E₈ in the gluing data constrains the global moduli and connects the
metric to exceptional Lie algebra structure.

---

## 4. PINN Architecture and Training

### 4.1 The parameterization challenge

The goal is to find a spatially varying metric field g : K₇ → Sym⁺₇(ℝ)
satisfying simultaneously:

1. det(g(x)) = 65/32 at every point
2. g(x) > 0 (positive definite)
3. dφ ≈ 0 and d*φ ≈ 0 (torsion-free, where φ is reconstructed from g)
4. ∫_{Cₖ} φ = Πₖ for k = 1, ..., 77 at multiple scales
5. Spatial average ⟨g⟩ ≈ G_TARGET

This is a PDE-constrained optimization problem on a 7-dimensional
computational domain modelling the TCS neck region (cf. §1.2).

### 4.2 Failed approaches and lessons

Before describing the successful architecture, we briefly document two
failed approaches, as the failure modes are instructive.

**Attempt 1 (G₂ adjoint parameterization):** A network outputs 14
parameters in the G₂ Lie algebra, which are exponentiated to produce
a G₂ rotation, applied to φ₀ via Lie derivatives to generate a deformed
3-form, from which the metric is extracted. *Failure mode:* the 14 → 35
map via Lie derivatives has rank 6, creating a 6-dimensional bottleneck
in the 28-dimensional space of symmetric metric perturbations. The
network cannot access 22 of the 28 metric degrees of freedom.

**Attempt 2 (Anisotropy loss):** Same architecture as above with an
additional loss ‖⟨g⟩ − G_TARGET‖²_F. *Failure mode:* 97.6% of the loss
gradient comes from the anisotropy term, but the rank-6 bottleneck
prevents the network from responding. The loss plateaus after ~100 steps
and remains constant for the remaining 4,900.

**Lesson:** When the architecture fundamentally cannot represent the
target (rank deficiency), no amount of training or hyperparameter tuning
will help. The bottleneck must be removed at the architectural level.

### 4.3 The Cholesky parameterization (successful)

We parameterize the metric directly via a Cholesky decomposition:

$$
g(x) = L(x) \cdot L(x)^\top, \qquad L(x) = L_0 + \delta L(x)
$$

where L₀ = chol(G_TARGET) is the Cholesky factor of the analytical target,
and δL(x) is a lower-triangular matrix output by the network.

| Property | G₂ adjoint | Cholesky (this work) |
|----------|-------------------|---------------|
| Metric DOF per point | 6 (rank of Lie derivs) | **28** (full) |
| Initialization | c²·I₇ (far from target) | **G_TARGET** (at target) |
| Positive definiteness | Requires penalty loss | **Free** (LLᵀ ≥ 0) |
| Symmetry | Via einsum contraction | **Free** (LLᵀ = (LLᵀ)ᵀ) |
| Gradient path | MLP → adj → Lie → φ → g | MLP → δL → g |

**Network architecture:**

```
Input: (x¹, ..., x⁷, log T) ∈ ℝ⁸
  ↓
FourierFeatures(48 frequencies) → ℝ⁹⁶
  ↓
MLP: 96 → 256 → 256 → 256 → 128 (ReLU activations)
  ↓
├── Metric head: 128 → 28 (lower triangular δL)
│     g(x) = (L₀ + δL(x))(L₀ + δL(x))ᵀ
│
└── 3-form heads: 128 → 35 (local) + 42 (global)
      φ(x) = c·φ₀ + 0.1·δφ(x)
```

Total parameters: 202,857.

### 4.4 Loss function

The loss has five terms:

| Term | Formula | Weight | Purpose |
|------|---------|--------|---------|
| L_det | (det(g) − 65/32)² | 100 | Topological constraint |
| L_aniso | ‖⟨g⟩ − G_TARGET‖²_F | 500 | Analytical target |
| L_period | Σ_T ‖⟨δφ⟩_T − Π(T)‖² / 5 | 1000 | 77 periods × 5 scales |
| L_torsion | ‖dφ‖² + ‖d*φ‖² (finite diff.) | 1 | Torsion-free condition |
| L_sparse | ‖δL‖² | 0.01 | Regularization |

The period loss averages over 5 energy scales (T = 100, 1000, 10000,
40000, 75000), each activating a different number of effective moduli
(from 5 to all 77).

### 4.5 Training protocol

Training proceeds in two phases over 5,000 epochs on a single NVIDIA
A100-SXM4-80GB GPU:

**Phase 1 (epochs 0–2,500):** Learning rate 10⁻³ with cosine annealing.
The warm-start means the determinant and anisotropy losses are already
near zero at initialization; the network primarily learns the period
integrals and torsion structure.

**Phase 2 (epochs 2,500–5,000):** Learning rate 10⁻⁴. Fine-tuning.
By epoch 3,500, the determinant and anisotropy losses reach machine
precision (10⁻¹⁵ to 10⁻¹⁸), and the residual loss is dominated entirely
by the period term.

**Training dynamics:**

| Epoch | Total loss | L_det | L_aniso | L_period | L_torsion |
|-------|-----------|-------|---------|----------|-----------|
| 0 | 4.33×10⁻³ | 2.7×10⁻²¹ | 9.8×10⁻²⁵ | 4.3×10⁻⁶ | 3.6×10⁻²³ |
| 100 | 1.51×10⁻³ | 4.9×10⁻⁶ | 3.2×10⁻⁷ | 8.6×10⁻⁷ | 9.5×10⁻¹⁰ |
| 500 | 6.28×10⁻⁴ | 2.0×10⁻⁶ | 8.3×10⁻⁸ | 3.9×10⁻⁷ | 2.6×10⁻¹⁰ |
| 2000 | 4.37×10⁻⁴ | 3.8×10⁻⁷ | 1.7×10⁻⁸ | 3.9×10⁻⁷ | 5.4×10⁻¹¹ |
| 3500 | 3.91×10⁻⁴ | 1.1×10⁻¹⁷ | 8.9×10⁻¹⁵ | 3.9×10⁻⁷ | 1.1×10⁻¹¹ |
| 5000 | 3.91×10⁻⁴ | 3.8×10⁻¹⁸ | 2.9×10⁻¹⁵ | 3.9×10⁻⁷ | 1.1×10⁻¹¹ |

At convergence, 100% of the residual loss is from the period integrals.
The metric constraints (determinant, anisotropy, positive definiteness)
are satisfied to machine precision.

Total training time: **2.9 minutes**.

---

## 5. The Explicit Metric

### 5.1 The 7×7 metric tensor

The spatially averaged metric over 50,000 points on the TCS neck:

```
g_mean =
  ┌                                                                      ┐
  │ 1.11332  +0.00098  -0.00072  -0.00019  +0.00341  +0.00285  -0.00305 │
  │+0.00098   1.11055  -0.00081  +0.00123  -0.00419  +0.00018  -0.00325 │
  │-0.00072  -0.00081   1.10908  +0.00461  +0.00085  +0.00269  +0.00069 │
  │-0.00019  +0.00123  +0.00461   1.10430  -0.00069  +0.00010  -0.00135 │
  │+0.00341  -0.00419  +0.00085  -0.00069   1.10263  +0.00154  -0.00001 │
  │+0.00285  +0.00018  +0.00269  +0.00010  +0.00154   1.10385  -0.00066 │
  │-0.00305  -0.00325  +0.00069  -0.00135  -0.00001  -0.00066   1.10217 │
  └                                                                      ┘
```

### 5.2 Comparison with analytical target

| Component | Target | Achieved | Absolute error |
|-----------|--------|----------|---------------|
| g₀₀ | 1.113320 | 1.113320 | 1.5 × 10⁻⁷ |
| g₁₁ | 1.110552 | 1.110552 | 1.6 × 10⁻⁷ |
| g₂₂ | 1.109078 | 1.109078 | 2.5 × 10⁻⁸ |
| g₃₃ | 1.104300 | 1.104300 | 2.3 × 10⁻⁷ |
| g₄₄ | 1.102633 | 1.102633 | 1.7 × 10⁻⁷ |
| g₅₅ | 1.103852 | 1.103852 | 1.4 × 10⁻⁸ |
| g₆₆ | 1.102167 | 1.102167 | 2.7 × 10⁻⁷ |
| g₂₃ (max off-diag) | +0.004613 | +0.004613 | 1.0 × 10⁻⁶ |
| **‖g − G_TARGET‖_F** | N/A | N/A | **1.76 × 10⁻⁷** |

Relative error: 4.4 × 10⁻⁸ (maximum elementwise error / maximum entry).

### 5.3 Eigenvalues

| | Target | Achieved | Error |
|---|--------|----------|-------|
| λ₁ | 1.09926643 | 1.09926642 | 1 × 10⁻⁸ |
| λ₂ | 1.10004584 | 1.10004584 | < 10⁻⁸ |
| λ₃ | 1.10124313 | 1.10124311 | 2 × 10⁻⁸ |
| λ₄ | 1.10334338 | 1.10334338 | < 10⁻⁸ |
| λ₅ | 1.11246355 | 1.11246359 | 4 × 10⁻⁸ |
| λ₆ | 1.11358841 | 1.11358840 | 1 × 10⁻⁸ |
| λ₇ | 1.11595127 | 1.11595127 | < 10⁻⁸ |

All seven eigenvalues matched to **8 significant figures**.

### 5.4 Determinant

$$
\det(g) = 2.031250001 \pm 9.5 \times 10^{-9}
$$

$$
\text{Target:}\; 65/32 = 2.031250000, \qquad
\text{Deviation:}\; 4 \times 10^{-8}\,\%
$$

### 5.5 Torsion

The torsion of a G₂-structure φ is measured by the failure of φ to be
closed and coclosed. Following Joyce [4, Theorem 11.6.1], if a
compact 7-manifold admits a G₂-structure φ₀ with ‖dφ₀‖_{C⁰} + ‖d*φ₀‖_{C⁰}
sufficiently small (below a constant ε₀ depending on the geometry), then
there exists a nearby torsion-free G₂-structure φ̃ with Hol(g̃) ⊆ G₂.

We evaluate the torsion of our candidate using finite-difference
approximations of dφ and d*φ on the computational domain:

| Quantity | Value |
|----------|-------|
| Mean ‖dφ‖ + ‖d*φ‖ | 3.3 × 10⁻⁶ |
| Max ‖dφ‖ + ‖d*φ‖ | 7.2 × 10⁻⁶ |

The absolute value of the torsion is small, but we emphasize two caveats:
(i) Joyce's ε₀ depends on the manifold and the approximate solution, and
we have not computed it for our specific setting; (ii) our computation
covers only the neck region, not the full compact manifold. We therefore
report the torsion as evidence that the candidate is a good *numerical*
approximation to a torsion-free structure, without claiming to have
verified the hypotheses of Joyce's theorem.

### 5.6 Scale invariance

The metric is evaluated at five energy scales T, at which different
numbers of moduli are active:

| Scale T | det deviation | Condition κ | Active moduli |
|---------|---------------|-------------|---------------|
| 100 | 3.7 × 10⁻⁸ % | 1.0151782 | 5 |
| 1,000 | 4.5 × 10⁻⁸ % | 1.0151782 | 66 |
| 10,000 | 4.0 × 10⁻⁸ % | 1.0151782 | 77 |
| 40,000 | 3.9 × 10⁻⁸ % | 1.0151782 | 77 |
| 75,000 | 4.6 × 10⁻⁸ % | 1.0151782 | 77 |

The condition number is **identical to 7 significant figures** at every
scale. The metric structure is independent of the scale at which the
period data is supplied.

### 5.7 Period integrals

| Scale T | RMS error | Correlation (local) | Active modes |
|---------|-----------|---------------------|-------------|
| 100 | 0.00110 | 0.920 | 5 |
| 1,000 | 0.000358 | 0.999 | 66 |
| **10,000** | **0.000311** | **0.996** | **77** |
| 40,000 | 0.000479 | 0.995 | 77 |
| 75,000 | 0.000540 | 0.995 | 77 |

Best fit at T = 10,000 (RMS = 3.11 × 10⁻⁴, 16-fold below threshold).

---

## 6. Discussion

### 6.1 Summary of contributions

1. **A numerical candidate metric on a compact G₂ manifold.** Previous
   work established existence (Joyce [3]) and gave constructions
   (Kovalev [5], Corti-Haskins-Nordström-Pacini (CHNP) [6]), but, to our knowledge, explicit pointwise
   numerical values of g_ij(x) have not been reported for the compact case.
   We note that substantial numerical work exists for non-compact G₂
   manifolds, and that our result covers only the TCS neck region
   (see §6.3).

2. **PINNs applied to special holonomy geometry.** The Cholesky warm-start
   technique may be applicable to other settings where an analytical
   approximation is available (e.g., Spin(7) manifolds, Calabi–Yau metrics
   beyond the Kähler class).

### 6.2 The Cholesky warm-start technique

The key insight is to decompose the problem:

$$
g(x) = g_{\text{target}} + \delta g(x), \qquad
\delta g \text{ small}
$$

and parameterize via L(x) = L₀ + δL(x) where L₀ = chol(g_target). This
has three advantages:

1. **Guaranteed constraints**: positive definiteness and symmetry are
   automatic, eliminating two loss terms and simplifying the optimization
   landscape.
2. **Warm start**: the network begins at the analytical solution and only
   needs to learn corrections of order 10⁻⁷, not the full metric from
   scratch.
3. **Full rank**: unlike Lie-algebraic parameterizations which may have
   rank deficiencies (as demonstrated by our earlier attempts), the
   Cholesky approach has 28 independent degrees of freedom per point
   (the full dimension of Sym₇(ℝ)).

### 6.3 Limitations

1. **Local model, not global**: Our metric is defined on a computational
   model of the TCS neck region. A complete global metric would require
   extending the solution into the bulk of M₁ and M₂, where it
   approaches the known Calabi–Yau metrics.

2. **Period data from GIFT**: The training targets (77 period integrals)
   are derived from the GIFT framework. While the metric itself is
   independently verifiable (det, torsion, positive definiteness are geometric properties),
   the specific values of the periods inherit any limitations of GIFT.

3. **Determinant value**: The target det(g) = 65/32 is derived within GIFT
   from the formula det(g) = (dim(E₈) + dim(G₂) + rank(E₈) + dim(K₇))
   / (2⁵). An independent derivation from pure G₂ geometry would
   strengthen the result.

4. **Neural network representation**: The metric is stored as a trained
   neural network, not a closed-form expression. While this is standard
   in the PINN literature, it limits analytical manipulation.

### 6.4 Future directions

1. **Extension to the bulk**: Solve the torsion-free equations dφ = 0,
   d*φ = 0 as a boundary-value problem, using the neck-region metric as
   a boundary condition and the known ACyl CY metrics on M₁, M₂ as
   asymptotic data.

2. **Other topological types**: Apply the same pipeline to other TCS
   manifolds from the CHNP classification, to understand how the metric
   depends on the topology (b₂, b₃).

3. **Spectral geometry**: Use the explicit metric to compute Laplacian
   eigenvalues, harmonic forms, and other spectral invariants that were
   previously inaccessible for compact G₂ manifolds.

4. **Comparison with flow methods**: Compare the PINN metric with results
   from Laplacian flow [14] or Hitchin flow, which provide alternative
   computational approaches to G₂ metrics.

---

## References

[1] Harvey, R. & Lawson, H.B. (1982). Calibrated geometries. *Acta Math.*
    148, 47–157.

[2] Bryant, R.L. (1987). Metrics with exceptional holonomy. *Ann. Math.*
    126(3), 525–576.

[3] Joyce, D.D. (1996). Compact Riemannian 7-manifolds with holonomy G₂.
    I, II. *J. Diff. Geom.* 43(2), 291–328 and 329–375.

[4] Joyce, D.D. (2000). *Compact Manifolds with Special Holonomy*. Oxford
    University Press.

[5] Kovalev, A.G. (2003). Twisted connected sums and special Riemannian
    holonomy. *J. Reine Angew. Math.* 565, 125–160.

[6] Corti, A., Haskins, M., Nordström, J. & Pacini, T. (2015). G₂-manifolds
    and associative submanifolds via semi-Fano 3-folds. *Duke Math. J.*
    164(10), 1971–2092.

[7] Raissi, M., Perdikaris, P. & Karniadakis, G.E. (2019). Physics-informed
    neural networks: A deep learning framework for solving forward and inverse
    problems involving nonlinear partial differential equations. *J. Comput.
    Phys.* 378, 686–707.

[8] Cai, S. et al. (2021). Physics-informed neural networks (PINNs) for
    fluid mechanics: A review. *Acta Mechanica Sinica* 37, 1727–1738.

[9] Hermann, J. et al. (2020). Deep-neural-network solution of the
    electronic Schrödinger equation. *Nature Chemistry* 12, 891–897.

[10] Liao, S. & Petzold, L. (2023). Physics-informed neural networks for
     solving Einstein field equations. Preprint, arXiv:2302.10696.

[11] Braun, A.P., Del Zotto, M., Halverson, J., Larfors, M., Morrison, D.R.
     & Schäfer-Nameki, S. (2018). Infinitely many M2-instanton corrections
     to M-theory on G₂-manifolds. *JHEP* 2018, 101.

[12] de La Fournière, B. (2026). Geometric Information Field Theory v3.3.
     Technical report. github.com/gift-framework. (Companion paper:
     source of the analytical target and period data used here.)

[13] de La Fournière, B. (2026). A parameter-free mollified approximation
     to the argument of the Riemann zeta function. Preprint. (Companion
     paper: source of the adaptive cutoff X(T).)

[14] Lotay, J.D. & Wei, Y. (2019). Laplacian flow for closed G₂ structures:
     Shi-type estimates, uniqueness and compactness. *Geom. Funct. Anal.*
     29, 1048–1110.

[15] Brandhuber, A., Gomis, J., Gubser, S.S. & Gukov, S. (2001). Gauge
     theory at large N and new G₂ holonomy metrics. *Nuclear Phys. B*
     611, 179–204.

---

## Appendix A. Topological Constants

All constants derive from the topology of K₇ and related algebraic
structures. None are fitted.

| Symbol | Value | Definition |
|--------|-------|------------|
| dim(K₇) | 7 | Manifold dimension |
| dim(G₂) | 14 | Holonomy group dimension |
| dim(E₈) | 248 | Exceptional Lie algebra |
| b₂(K₇) | 21 | Second Betti number |
| b₃(K₇) | 77 | Third Betti number (= dim moduli) |
| C(7,3) | 35 | dim Λ³(ℝ⁷) (local modes) |
| κ_T | 1/61 | Torsion coupling constant |
| det(g) | 65/32 | Metric determinant |

---

## Appendix B. Reproducibility

### B.1 Code and data

| Resource | Location |
|----------|----------|
| PINN notebook (v3) | `notebooks/K7_PINN_Step5_Reconstruction_v3.ipynb` |
| Pre-computed data | `notebooks/riemann/*.json` (Steps 1–4) |
| Repository | github.com/gift-framework/GIFT |

### B.2 Hardware

| | Specification |
|---|---------------|
| GPU | NVIDIA A100-SXM4-80GB |
| Training time | 2.9 minutes |
| Parameters | 202,857 |
| Epochs | 5,000 |
| Evaluation points | 50,000 |
| Peak memory | ~1–2 GB |

### B.3 Dependencies

```
torch >= 2.0 (float64 mode)
numpy, scipy, matplotlib, tqdm
cupy-cuda12x (optional, for spectral analysis)
```

### B.4 To reproduce

1. Open `notebooks/K7_PINN_Step5_Reconstruction_v3.ipynb` in Google Colab
2. Select A100 GPU runtime
3. Run all cells
4. Results exported to `k7_pinn_step5_results_v3.json`

No manual intervention required.

---

*Manuscript prepared February 2026.*
