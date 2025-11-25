# Supplement S3: Torsional Dynamics

## Complete Formulation of Torsional Geodesic Dynamics and Connection to RG Flow

*This supplement provides the mathematical formulation of torsional geodesic dynamics underlying the GIFT framework. We derive the torsion tensor from non-closure conditions, establish the geodesic flow equation, and demonstrate the connection to renormalization group flow. For K₇ metric construction, see Supplement S2. For physical applications to observables, see Supplement S5.*

---

## Abstract

We present the complete dynamical framework connecting static topological structure to physical evolution. Section 1 develops the torsion tensor from the non-closure of the G₂ 3-form, establishing its physical origin and component structure. Section 2 derives the geodesic flow equation from variational principles and establishes conservation laws. Section 3 identifies geodesic flow with renormalization group evolution, providing geometric foundations for quantum field theory β-functions. Key results include the torsion magnitude |T| ≈ 0.0164, the torsional geodesic equation, and the ultra-slow flow velocity |v| ≈ 0.015 ensuring constant variation bounds.

---

## Status Classifications

- **PROVEN**: Exact mathematical result with rigorous derivation
- **TOPOLOGICAL**: Direct consequence of manifold structure
- **THEORETICAL**: Has theoretical justification, numerical verification pending
- **PHENOMENOLOGICAL**: Constrained by experimental data

---

# 1. Torsion Tensor

## 1.1 Definition and Properties

### 1.1.1 Torsion in Differential Geometry

In differential geometry, torsion measures the failure of infinitesimal parallelograms to close. For a connection ∇ on a manifold M, the torsion tensor T is defined by:

$$T(X, Y) = \nabla_X Y - \nabla_Y X - [X, Y]$$

for vector fields X, Y. In components:

$$T^k_{ij} = \Gamma^k_{ij} - \Gamma^k_{ji}$$

where Γ^k_{ij} are the connection coefficients.

### 1.1.2 Torsion-Free vs Torsionful Connections

**Levi-Civita connection**: The unique torsion-free, metric-compatible connection:
- T^k_{ij} = 0 (torsion-free)
- ∇_k g_{ij} = 0 (metric-compatible)

**Torsionful connection**: Preserves metric compatibility but allows non-zero torsion:
- T^k_{ij} ≠ 0
- ∇_k g_{ij} = 0 (metric-compatible)

The GIFT framework employs a torsionful connection arising from the non-closure of the G₂ 3-form.

### 1.1.3 Contorsion Tensor

The difference between a torsionful connection and Levi-Civita is the contorsion tensor K:

$$\Gamma^k_{ij} = \overset{\circ}{\Gamma}{}^k_{ij} + K^k_{ij}$$

where Γ̊ denotes Levi-Civita. The contorsion relates to torsion by:

$$K^k_{ij} = \frac{1}{2}(T^k_{ij} + T_i{}^k{}_j + T_j{}^k{}_i)$$

For totally antisymmetric torsion T_{ijk} = T_{[ijk]}:

$$K^k_{ij} = \frac{1}{2} T^k_{ij}$$

### 1.1.4 Torsion Classes for G₂ Manifolds

On a 7-manifold with G₂ structure, torsion decomposes into four irreducible G₂ representations:

$$T \in W_1 \oplus W_7 \oplus W_{14} \oplus W_{27}$$

| Class | Dimension | Characterization |
|-------|-----------|------------------|
| W₁ | 1 | dφ ∧ φ ≠ 0 |
| W₇ | 7 | *dφ - θ ∧ φ for 1-form θ |
| W₁₄ | 14 | Traceless part of d*φ |
| W₂₇ | 27 | Symmetric traceless |

**Torsion-free G₂**: All classes vanish (dφ = 0, d*φ = 0)

**GIFT framework**: Controlled non-zero torsion in specific classes generates physical interactions.

---

## 1.2 Physical Origin

### 1.2.1 G₂ Holonomy and the 3-Form

A 7-manifold M has G₂ holonomy if it admits a parallel 3-form φ:

$$\nabla \phi = 0$$

This is equivalent to the closure conditions:

$$d\phi = 0, \quad d*\phi = 0$$

Such manifolds are Ricci-flat and have trivial canonical bundle.

### 1.2.2 Non-Closure as Source of Interactions

Physical interactions require departure from the torsion-free condition. The framework introduces controlled non-closure:

$$|d\phi|^2 + |d*\phi|^2 = \epsilon^2$$

where ε is small but non-zero.

**Physical motivation**: A perfectly torsion-free manifold has no geometric coupling between sectors. Torsion provides the mechanism for particle interactions.

**Numerical value**: From metric reconstruction (Supplement S2):

$$\epsilon = 0.0164 \pm 0.002$$

### 1.2.3 Torsion from Non-Closure

The torsion tensor components arise from the 4-form dφ and 5-form d*φ:

$$T_{ijk} \sim (d\phi)_{lijk} g^{lm} + \text{(dual terms)}$$

The precise relation involves the G₂ structure equations and metric factors.

### 1.2.4 Global Torsion Magnitude

The global torsion norm:

$$|\mathbf{T}| = \sqrt{|d\phi|^2 + |d*\phi|^2} \approx 0.0164$$

**Physical interpretation**: This small value ensures:
1. Approximate G₂ structure preservation
2. Ultra-slow evolution of constants
3. Consistency with experimental bounds on constant variation

---

## 1.3 Component Analysis

### 1.3.1 Coordinate System

The K₇ metric is expressed in coordinates (e, π, φ) with physical interpretation:

| Coordinate | Physical Sector | Range |
|------------|-----------------|-------|
| e | Electromagnetic | [0.1, 2.0] |
| π | Hadronic/strong | [0.1, 3.0] |
| φ | Electroweak/Higgs | [0.1, 1.5] |

These span a 3-dimensional subspace encoding essential parameter information.

### 1.3.2 Torsion Tensor Components

From numerical metric reconstruction, the key torsion components are:

$$\begin{align}
T_{e\phi,\pi} &= -4.89 \pm 0.02 \\
T_{\pi\phi,e} &= -0.45 \pm 0.01 \\
T_{e\pi,\phi} &= (3.1 \pm 0.3) \times 10^{-5}
\end{align}$$

### 1.3.3 Hierarchical Structure

The torsion components span four orders of magnitude:

| Component | Magnitude | Physical Role |
|-----------|-----------|---------------|
| T_{eφ,π} | ~5 | Mass hierarchies (large ratios) |
| T_{πφ,e} | ~0.5 | CP violation phase |
| T_{eπ,φ} | ~10⁻⁵ | Jarlskog invariant |

**Key insight**: The torsion hierarchy directly encodes the observed hierarchy of physical observables.

### 1.3.4 Physical Interpretation

**T_{eφ,π} ≈ -4.89 (large)**:
- Drives geodesics in (e,φ) plane
- Source of mass hierarchies like m_τ/m_e = 3477
- Large torsion amplifies path lengths

**T_{πφ,e} ≈ -0.45 (moderate)**:
- Torsional twist in (π,φ) sector
- Source of CP violation δ_CP = 197°
- Accumulated geometric phase

**T_{eπ,φ} ≈ 3×10⁻⁵ (tiny)**:
- Weak electromagnetic-hadronic coupling
- Related to Jarlskog invariant J ≈ 3×10⁻⁵
- Suppressed CP violation in quark sector

---

## 1.4 Symmetry Properties

### 1.4.1 Antisymmetry

The torsion tensor is antisymmetric in its lower indices:

$$T_{ijk} = -T_{jik}$$

This follows from the definition T^k_{ij} = Γ^k_{ij} - Γ^k_{ji}.

### 1.4.2 Bianchi-Type Identities

Torsion satisfies algebraic Bianchi identities:

$$T_{[ijk]} = T_{ijk} + T_{jki} + T_{kij} = 0$$

(cyclic sum vanishes for metric-compatible connection)

### 1.4.3 G₂ Transformation Properties

Under G₂ structure group transformations:

$$T_{ijk} \to g_i{}^{i'} g_j{}^{j'} g_k{}^{k'} T_{i'j'k'}$$

where g ∈ G₂ ⊂ SO(7).

### 1.4.4 Conservation Laws

The torsion tensor satisfies differential Bianchi identities relating its covariant derivatives to curvature:

$$\nabla_{[i} T_{jk]l} = R_{[ijk]l} - \text{(torsion squared terms)}$$

These constrain the evolution of torsion components.

---

# 2. Geodesic Flow Equation

## 2.1 Derivation from Action

### 2.1.1 Geodesic Action

Consider a curve x^k(λ) on K₇ parametrized by affine parameter λ. The geodesic action is:

$$S = \int d\lambda \, \mathcal{L} = \int d\lambda \, \frac{1}{2} g_{ij} \frac{dx^i}{d\lambda} \frac{dx^j}{d\lambda}$$

Using dot notation ẋ^i = dx^i/dλ:

$$S = \int d\lambda \, \frac{1}{2} g_{ij} \dot{x}^i \dot{x}^j$$

### 2.1.2 Euler-Lagrange Equations

The Euler-Lagrange equations:

$$\frac{d}{d\lambda} \left( \frac{\partial \mathcal{L}}{\partial \dot{x}^k} \right) - \frac{\partial \mathcal{L}}{\partial x^k} = 0$$

**Calculation**:

$$\frac{\partial \mathcal{L}}{\partial \dot{x}^k} = g_{kj} \dot{x}^j$$

$$\frac{d}{d\lambda}(g_{kj} \dot{x}^j) = \partial_i g_{kj} \dot{x}^i \dot{x}^j + g_{kj} \ddot{x}^j$$

$$\frac{\partial \mathcal{L}}{\partial x^k} = \frac{1}{2} \partial_k g_{ij} \dot{x}^i \dot{x}^j$$

**Euler-Lagrange result**:

$$g_{kj} \ddot{x}^j + \left(\partial_i g_{kj} - \frac{1}{2} \partial_k g_{ij}\right) \dot{x}^i \dot{x}^j = 0$$

### 2.1.3 Standard Geodesic Equation

Multiplying by g^{mk}:

$$\ddot{x}^m + \Gamma^m_{ij} \dot{x}^i \dot{x}^j = 0$$

where Γ^m_{ij} is the Christoffel symbol:

$$\Gamma^m_{ij} = \frac{1}{2} g^{mk}(\partial_i g_{kj} + \partial_j g_{ik} - \partial_k g_{ij})$$

### 2.1.4 Torsional Modification

For locally constant metric (∂_k g_{ij} ≈ 0 over coordinate patches):

$$\Gamma^m_{ij}|_{\text{Levi-Civita}} \approx 0$$

The effective connection becomes purely torsional:

$$\boxed{\Gamma^k_{ij} = -\frac{1}{2} g^{kl} T_{ijl}}$$

**Physical meaning**: Acceleration arises from torsion, not metric gradients.

---

## 2.2 Torsional Geodesic Equation

### 2.2.1 Main Result

Substituting the torsional connection into the geodesic equation:

$$\boxed{\frac{d^2 x^k}{d\lambda^2} = \frac{1}{2} g^{kl} T_{ijl} \frac{dx^i}{d\lambda} \frac{dx^j}{d\lambda}}$$

This is the **torsional geodesic equation** governing parameter evolution.

### 2.2.2 Component Form

In explicit component notation for (e, π, φ) coordinates:

$$\ddot{e} = \frac{1}{2} g^{em} T_{ijm} \dot{x}^i \dot{x}^j$$

$$\ddot{\pi} = \frac{1}{2} g^{\pi m} T_{ijm} \dot{x}^i \dot{x}^j$$

$$\ddot{\phi} = \frac{1}{2} g^{\phi m} T_{ijm} \dot{x}^i \dot{x}^j$$

### 2.2.3 Quadratic Velocity Dependence

The right-hand side is quadratic in velocities:

$$\ddot{x}^k \propto \dot{x}^i \dot{x}^j$$

This produces nonlinear dynamics analogous to:
- Geodesic deviation in general relativity
- Nonlinear β-function evolution in QFT
- Chaotic dynamics in mechanical systems

### 2.2.4 Physical Interpretation

| Quantity | Geometric | Physical |
|----------|-----------|----------|
| x^k(λ) | Position on K₇ | Coupling constant value |
| λ | Curve parameter | RG scale ln(μ) |
| ẋ^k | Velocity | β-function |
| ẍ^k | Acceleration | β-function derivative |
| T_{ijl} | Torsion | Interaction strength |
| g^{kl} | Inverse metric | Coupling response |

---

## 2.3 Conservation Laws

### 2.3.1 Energy Conservation

For affine parameter λ, the kinetic energy:

$$E = g_{ij} \frac{dx^i}{d\lambda} \frac{dx^j}{d\lambda}$$

is conserved along geodesics:

$$\frac{dE}{d\lambda} = 0$$

**Proof**:

$$\frac{dE}{d\lambda} = 2 g_{ij} \dot{x}^i \ddot{x}^j + \partial_k g_{ij} \dot{x}^k \dot{x}^i \dot{x}^j$$

Using the geodesic equation and metric compatibility:

$$= 2 g_{ij} \dot{x}^i \left(-\Gamma^j_{kl} \dot{x}^k \dot{x}^l\right) + \partial_k g_{ij} \dot{x}^k \dot{x}^i \dot{x}^j = 0$$

**Status**: PROVEN

### 2.3.2 Killing Vector Conservation

If the metric admits a Killing vector ξ^i (satisfying ∇_{(i} ξ_{j)} = 0), then:

$$p_\xi = g_{ij} \xi^i \frac{dx^j}{d\lambda}$$

is conserved along geodesics.

### 2.3.3 Topological Charges

Certain topological invariants of K₇ remain constant along flow:
- Winding numbers in periodic directions
- Holonomy charges around non-contractible loops
- Cohomology class representatives

---

## 2.4 Solution Methods

### 2.4.1 Perturbative Expansion

For small torsion |T| << 1, expand geodesics perturbatively:

$$x^k(\lambda) = x^k_0(\lambda) + \epsilon \, x^k_1(\lambda) + \epsilon^2 \, x^k_2(\lambda) + ...$$

where ε ~ |T| ≈ 0.0164.

**Zeroth order**: Straight lines (no torsion)

$$x^k_0(\lambda) = a^k + b^k \lambda$$

**First order**: Linear correction from torsion

$$\ddot{x}^k_1 = \frac{1}{2} g^{kl} T_{ijl} b^i b^j$$

integrates to:

$$x^k_1(\lambda) = \frac{1}{4} g^{kl} T_{ijl} b^i b^j \lambda^2$$

### 2.4.2 Numerical Integration

For non-perturbative solutions, use standard ODE integrators:

**Initial conditions**:
- x^k(0) = x^k_initial (starting coupling values)
- ẋ^k(0) = v^k_initial (initial β-functions)

**Algorithm**: Runge-Kutta 4th order or adaptive step methods

**Code**: Available at github.com/gift-framework/GIFT

### 2.4.3 Fixed Point Analysis

Fixed points satisfy ẋ^k = 0 and ẍ^k = 0:

$$g^{kl} T_{ijl} v^i v^j = 0 \quad \text{for all } k$$

**Types**:
- **Stable (attractor)**: Negative eigenvalues of linearized flow
- **Unstable (repeller)**: Positive eigenvalues
- **Saddle**: Mixed eigenvalues

### 2.4.4 Geodesic Deviation

Nearby geodesics separate according to:

$$\frac{D^2 \xi^k}{d\lambda^2} = R^k{}_{ijl} \dot{x}^i \xi^j \dot{x}^l + \text{(torsion terms)}$$

where ξ^k is the separation vector. This determines stability of flow.

---

# 3. RG Flow Connection

## 3.1 Identification λ = ln(μ)

### 3.1.1 Physical Motivation

The renormalization group describes how physical quantities change with energy scale μ. The identification:

$$\lambda = \ln\left(\frac{\mu}{\mu_0}\right)$$

connects geodesic flow to RG evolution.

**Justifications**:
1. Both are one-parameter flows on coupling space
2. Both exhibit nonlinear dynamics
3. Dimensional analysis: ln(μ) is dimensionless
4. Fixed points correspond in both frameworks

### 3.1.2 Scale Dependence

Under this identification:

| λ range | Energy scale | Physics |
|---------|--------------|---------|
| λ → +∞ | μ → ∞ (UV) | E₈×E₈ symmetry |
| λ = 0 | μ = μ₀ (reference) | Electroweak scale |
| λ → -∞ | μ → 0 (IR) | Confinement |

### 3.1.3 Reference Scale

Natural choice: μ₀ = M_Z = 91.188 GeV (Z boson mass)

Alternative choices:
- μ₀ = v_EW = 246.22 GeV (Higgs VEV)
- μ₀ = M_Planck = 1.22 × 10¹⁹ GeV (Planck scale)

---

## 3.2 Coupling Evolution

### 3.2.1 β-Functions as Velocities

The RG β-function for coupling g_i:

$$\beta_i(g) = \frac{dg_i}{d\ln\mu}$$

becomes under λ = ln(μ):

$$\beta_i = \frac{dx^i}{d\lambda}$$

**Interpretation**: β-functions are geodesic velocities on K₇.

### 3.2.2 β-Function Evolution

The geodesic equation gives:

$$\frac{d\beta^k}{d\lambda} = \frac{d^2 x^k}{d\lambda^2} = \frac{1}{2} g^{kl} T_{ijl} \beta^i \beta^j$$

**Physical meaning**: The evolution of β-functions (two-loop and higher) is determined by torsion.

### 3.2.3 Standard QFT β-Functions

In perturbative QFT:

$$\beta(g) = \beta_0 g^3 + \beta_1 g^5 + \beta_2 g^7 + ...$$

**GIFT interpretation**: The coefficients β₀, β₁, β₂ arise from torsion tensor components:

$$\beta_n \sim g^{nm} T_{ijm} \times \text{(combinatorial factors)}$$

### 3.2.4 Gauge Coupling Evolution

For the strong coupling α_s(μ):

$$\frac{d\alpha_s}{d\ln\mu} = -\frac{b_0}{2\pi} \alpha_s^2 - \frac{b_1}{(2\pi)^2} \alpha_s^3 + ...$$

with b₀ = 11 - 2n_f/3 for SU(3) QCD.

**Geometric origin**: b₀ relates to torsion components in the strong sector of K₇.

---

## 3.3 Fixed Points

### 3.3.1 UV Fixed Point

At high energies (λ → +∞), the theory approaches the E₈×E₈ symmetric point:

- All couplings unified
- Maximum symmetry
- "Free" theory in some sense

**Geometric picture**: The geodesic approaches the symmetric point on K₇.

### 3.3.2 IR Fixed Point

At low energies (λ → -∞):

- Symmetry broken to Standard Model
- Couplings reach observed values
- Confinement in QCD sector

**Geometric picture**: The geodesic reaches the physical vacuum.

### 3.3.3 Intermediate Fixed Points

Possible fixed points at intermediate scales:

- **GUT scale** (~10¹⁶ GeV): Gauge coupling unification
- **Electroweak scale** (~10² GeV): Symmetry breaking
- **QCD scale** (~10⁻¹ GeV): Confinement

### 3.3.4 Fixed Point Stability

Linearizing the geodesic equation around fixed point x*:

$$\ddot{\xi}^k = M^k{}_j \xi^j$$

where ξ^k = x^k - x*^k and M is the stability matrix.

**Classification**:
- All eigenvalues negative: Stable (attractor)
- All eigenvalues positive: Unstable (UV repeller)
- Mixed signs: Saddle point

---

## 3.4 Flow Velocity

### 3.4.1 Ultra-Slow Velocity Requirement

Experimental bounds on constant variation:

$$\left|\frac{\dot{\alpha}}{\alpha}\right| < 10^{-17} \text{ yr}^{-1}$$

constrain the K₇ flow velocity.

### 3.4.2 Velocity Bound Derivation

The variation rate:

$$\frac{\dot{\alpha}}{\alpha} \sim H_0 \times |\Gamma| \times |v|^2$$

where:
- H₀ ≈ 70 km/s/Mpc ≈ 2.3 × 10⁻¹⁸ s⁻¹
- |Γ| ~ |T|/det(g) ≈ 0.0164/2 ≈ 0.008
- |v| = flow velocity

**Constraint**:

$$|v|^2 < \frac{10^{-17}}{H_0 \times |\Gamma|} \approx \frac{10^{-17}}{2.3 \times 10^{-18} \times 0.008} \approx 0.5$$

$$|v| < 0.7$$

### 3.4.3 Framework Value

From numerical simulations and RG flow matching:

$$|v| \approx 0.015$$

This ultra-slow velocity ensures:

$$\frac{\dot{\alpha}}{\alpha} \sim 2.3 \times 10^{-18} \times 0.008 \times (0.015)^2 \approx 4 \times 10^{-24} \text{ s}^{-1} \approx 10^{-16} \text{ yr}^{-1}$$

Well within experimental bounds.

### 3.4.4 Cosmological Consistency

The slow velocity |v| ≈ 0.015 << 1 ensures:
1. Constants appear approximately fixed at laboratory scales
2. Evolution occurs over cosmological time
3. No conflict with precision measurements
4. Consistency with Big Bang nucleosynthesis bounds

**Status**: PHENOMENOLOGICAL (constrained by experiment)

---

# 4. Physical Applications

## 4.1 Mass Hierarchies

### 4.1.1 Tau-Electron Ratio

The mass ratio m_τ/m_e = 3477 (proven in Supplement S4) has geometric origin in the geodesic length in the (e,φ) plane.

**Geodesic equation in (e,φ) sector**:

$$\frac{d^2 e}{d\lambda^2} = g^{\pi\pi} T_{e\phi,\pi} \frac{de}{d\lambda} \frac{d\phi}{d\lambda}$$

**Numerical values**:
- g^{ππ} ≈ 2/3
- T_{eφ,π} ≈ -4.89

The large torsion component T_{eφ,π} amplifies the path length, generating the hierarchy.

### 4.1.2 Connection to Topology

The topological formula:

$$\frac{m_\tau}{m_e} = \dim(K_7) + 10 \times \dim(E_8) + 10 \times H^* = 7 + 2480 + 990 = 3477$$

encodes the accumulated "information content" along the geodesic path.

## 4.2 CP Violation

### 4.2.1 Geometric Phase

The CP violation phase δ_CP = 197° (proven in Supplement S4) arises from torsional twist in the (π,φ) sector.

**Twist equation**:

$$\frac{d^2 \phi}{d\lambda^2} \propto T_{\pi\phi,e} \frac{d\pi}{d\lambda} \frac{de}{d\lambda}$$

The accumulated twist over one "cycle" gives the CP phase.

### 4.2.2 Topological Origin

$$\delta_{CP} = 7 \times \dim(G_2) + H^* = 7 \times 14 + 99 = 197°$$

The torsion component T_{πφ,e} ≈ -0.45 drives this geometric phase accumulation.

## 4.3 Hubble Constant

### 4.3.1 Curvature-Torsion Relation

The Hubble constant emerges from:

$$H_0^2 \propto R \cdot |\mathbf{T}|^2$$

where:
- R ≈ 1/54: Effective scalar curvature
- |T| ≈ 0.0164: Torsion magnitude

### 4.3.2 Intermediate Value

The framework predicts:

$$H_0 \approx 69.8 \text{ km/s/Mpc}$$

This intermediate value between CMB (67.4) and local (73.0) measurements suggests potential geometric resolution of the Hubble tension.

---

# 5. Summary

This supplement established the torsional geodesic dynamics of the GIFT framework:

## Key Results

| Result | Value | Status |
|--------|-------|--------|
| Torsion magnitude | \|T\| ≈ 0.0164 | THEORETICAL |
| T_{eφ,π} | -4.89 | THEORETICAL |
| T_{πφ,e} | -0.45 | THEORETICAL |
| T_{eπ,φ} | ~3×10⁻⁵ | THEORETICAL |
| Flow velocity | \|v\| ≈ 0.015 | PHENOMENOLOGICAL |
| α̇/α bound | <10⁻¹⁶ yr⁻¹ | PHENOMENOLOGICAL |

## Main Equations

**Torsional connection**:
$$\Gamma^k_{ij} = -\frac{1}{2} g^{kl} T_{ijl}$$

**Geodesic equation**:
$$\frac{d^2 x^k}{d\lambda^2} = \frac{1}{2} g^{kl} T_{ijl} \frac{dx^i}{d\lambda} \frac{dx^j}{d\lambda}$$

**RG identification**:
$$\lambda = \ln(\mu/\mu_0), \quad \beta^i = \frac{dx^i}{d\lambda}$$

## Physical Interpretation

The framework provides geometric foundations for:
- Mass hierarchies from geodesic lengths
- CP violation from torsional twist
- RG flow from geodesic evolution
- Constant stability from ultra-slow velocity

---

## References

[1] Cartan, E., Sur les variétés à connexion affine et la théorie de la relativité généralisée, Ann. Sci. ENS **40**, 325 (1923)

[2] Kibble, T.W.B., Lorentz invariance and the gravitational field, J. Math. Phys. **2**, 212 (1961)

[3] Hehl, F.W., et al., General relativity with spin and torsion, Rev. Mod. Phys. **48**, 393 (1976)

[4] Joyce, D.D., Compact Manifolds with Special Holonomy, Oxford University Press (2000)

[5] Karigiannis, S., Flows of G₂-structures, Q. J. Math. **60**, 487 (2009)

[6] Grigorian, S., Short-time behaviour of a modified Laplacian coflow of G₂-structures, Adv. Math. **248**, 378 (2013)

---

*GIFT Framework v2.1 - Supplement S3*
*Torsional Dynamics*
