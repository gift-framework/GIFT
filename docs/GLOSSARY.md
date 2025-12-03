# Glossary of Technical Terms

Comprehensive definitions of technical terms, mathematical notation, and acronyms used in the GIFT framework.

## Framework-Specific Terms

### GIFT
**Geometric Information Field Theory**. The framework proposing that fundamental physics parameters emerge as topological invariants from E₈×E₈ structure compactified on G₂ holonomy manifolds.

### K₇
A compact 7-dimensional Riemannian manifold with G₂ holonomy. The subscript 7 indicates dimension. Specific topological properties: b₂(K₇) = 21, b₃(K₇) = 77.

### Status Classifications
Framework uses hierarchical classification for results:
- **PROVEN (Lean)**: Formally verified by Lean 4 kernel with Mathlib (machine-checked proofs, zero domain axioms, zero sorry)
- **PROVEN**: Exact topological identity with rigorous mathematical proof
- **TOPOLOGICAL**: Direct consequence of topological structure
- **DERIVED**: Calculated from proven relations
- **THEORETICAL**: Has theoretical justification, proof incomplete
- **PHENOMENOLOGICAL**: Empirically accurate, derivation in progress
- **EXPLORATORY**: Preliminary investigation, mechanism uncertain

## Lie Algebras and Groups

### E₈
The largest exceptional simple Lie algebra. Properties:
- Dimension: 248
- Rank: 8
- Root system: 240 roots, all length √2
- Simply-laced (all roots equal length)
- Cartan matrix determinant: 1

### E₈×E₈
Product of two independent copies of E₈. Total dimension 496 = 2×248.

### G₂
The automorphism group of the octonions. A 14-dimensional exceptional Lie group. Smallest exceptional group, important for 7-dimensional geometry.

### SU(N)
**Special Unitary Group**. Group of N×N unitary matrices with determinant 1.
- SU(3): Strong force (quantum chromodynamics)
- SU(2): Weak force component
- U(1): Electromagnetic force component

### Exceptional Lie Algebras
Five Lie algebras that don't fit standard infinite families:
- G₂ (dimension 14, rank 2)
- F₄ (dimension 52, rank 4)
- E₆ (dimension 78, rank 6)
- E₇ (dimension 133, rank 7)
- E₈ (dimension 248, rank 8)

### Weyl Group
Symmetry group of root system. For E₈, the Weyl group has order 696,729,600.

### Root System
Set of vectors in Euclidean space satisfying reflection symmetry. For E₈: 240 roots arranged in highly symmetric configuration.

### Rank
Dimension of maximal torus (maximal abelian subgroup). For E₈: rank 8.

## Geometry and Topology

### Holonomy
Geometric property describing how vectors change when parallel-transported around closed loops. G₂ holonomy implies special geometric structure.

### Cohomology
Mathematical tool measuring topological features. For K₇:
- H²(K₇) = ℝ²¹: Related to gauge bosons
- H³(K₇) = ℝ⁷⁷: Related to chiral fermions

### Betti Numbers
Topological invariants counting independent homology classes.
- b₂: Second Betti number (2-dimensional holes)
- b₃: Third Betti number (3-dimensional holes)
For K₇: b₂ = 21, b₃ = 77

### AdS₄
**Anti-de Sitter space in 4 dimensions**. Maximally symmetric spacetime with negative cosmological constant. Used in holographic models and GIFT compactification.

### Compact Manifold
Topological space that is closed (contains all limit points) and bounded. K₇ is compact, allowing consistent dimensional reduction.

### Ricci-Flat
Manifold with Ricci curvature tensor equal to zero. G₂ holonomy manifolds are automatically Ricci-flat, suitable for compactification.

### Harmonic Forms
Differential forms satisfying Laplace equation. Zero modes in dimensional reduction corresponding to 4D fields.

### Kaluza-Klein Reduction
Process of compactifying extra dimensions to derive lower-dimensional effective theory. Fields decompose into tower of modes.

## Particle Physics

### Standard Model
Current theory of particle physics describing electromagnetic, weak, and strong forces. Contains 19 free parameters in conventional formulation.

### Generation
Family of fermions. Standard Model has three generations:
- First: (u, d, e, νₑ)
- Second: (c, s, μ, νμ)
- Third: (t, b, τ, ντ)

### Gauge Coupling
Strength of force interaction. Three in Standard Model:
- α: Electromagnetic (fine structure constant)
- g₂: Weak force
- g₃ (or α_s): Strong force

### Fine Structure Constant (α)
Dimensionless coupling for electromagnetic interaction. α ≈ 1/137.036.

### Weak Mixing Angle (θ_W)
Parameter relating electromagnetic and weak forces. sin²θ_W ≈ 0.231.

### Strong Coupling (α_s)
Coupling constant for quantum chromodynamics. α_s(M_Z) ≈ 0.118.

### CKM Matrix
**Cabibbo-Kobayashi-Maskawa matrix**. 3×3 unitary matrix describing quark mixing between generations. Contains 4 independent parameters (3 angles, 1 phase).

### PMNS Matrix
**Pontecorvo-Maki-Nakagawa-Sakata matrix**. Analogous to CKM for neutrino sector. Contains 3 mixing angles (θ₁₂, θ₁₃, θ₂₃) and CP violation phase (δ_CP).

### CP Violation
Breaking of combined charge conjugation (C) and parity (P) symmetry. Observed in quark and neutrino sectors.

### Yukawa Coupling
Interaction strength between fermions and Higgs field, determining fermion masses.

### Higgs Mechanism
Process by which gauge bosons acquire mass through spontaneous symmetry breaking.

### VEV
**Vacuum Expectation Value**. Non-zero value of Higgs field in vacuum, v ≈ 246 GeV.

## Specific Observables

### N_gen
Number of fermion generations. Experimentally: 3. GIFT predicts: rank(E₈) - rank(Weyl) = 3.

### δ_CP
CP-violating phase in neutrino mixing. GIFT predicts: 197° from formula 7·dim(G₂) + ζ(3) + √5.

### θ₁₂, θ₁₃, θ₂₃
Three neutrino mixing angles in PMNS matrix.
- θ₁₂ ≈ 33.44° (solar mixing)
- θ₁₃ ≈ 8.61° (reactor mixing)
- θ₂₃ ≈ 49.2° (atmospheric mixing)

### Q_Koide
Parameter in Koide formula relating charged lepton masses:
```
Q = (mₑ + mμ + mτ)² / (mₑ² + mμ² + mτ²)
```
Experimental: Q ≈ 2/3. GIFT: Q = 2/3 exactly.

### Ω_DE
Dark energy density as fraction of critical density. Experimental: Ω_DE ≈ 0.689. GIFT: Ω_DE = ln(2) ≈ 0.693.

### H₀
**Hubble constant**. Current expansion rate of universe. Local measurements: ~73 km/s/Mpc. CMB measurements: ~67 km/s/Mpc. "Hubble tension" refers to discrepancy.

## Mathematical Constants

### π
Pi, ratio of circle circumference to diameter. π ≈ 3.14159...

### e
Euler's number, base of natural logarithm. e ≈ 2.71828...

### φ
**Golden ratio**. φ = (1 + √5)/2 ≈ 1.61803...
Appears in E₈ via McKay correspondence.

### ζ(3)
**Riemann zeta function at 3**. ζ(3) = 1 + 1/8 + 1/27 + ... ≈ 1.202.
Appears in δ_CP formula.

### ln(2)
Natural logarithm of 2 ≈ 0.693. Related to dark energy density and information theory (bits vs nats).

## Information Theory

### Binary Architecture
Structure based on powers of 2. E₈×E₈ dimension 496 ≈ 2⁴⁸ + 48, suggestive of binary encoding.

### Quantum Error Correction
Method of protecting quantum information from errors. E₈ lattice forms [[248, 12, 56]] code.

### [[n, k, d]] Code
Notation for quantum error correction code:
- n: Number of physical qubits
- k: Number of logical qubits encoded
- d: Distance (number of errors correctable)

### Compression Ratio
Ratio of input to output dimensions. E₈×E₈: 496 → 99 gives ratio ≈ 5:1.

## Physics Experiments

### PDG
**Particle Data Group**. International collaboration compiling experimental particle physics data. Standard reference for measured parameters.

### LHC
**Large Hadron Collider**. Particle accelerator at CERN. Discovered Higgs boson in 2012.

### Belle II
Particle physics experiment in Japan studying B mesons and precision measurements.

### DUNE
**Deep Underground Neutrino Experiment**. Future neutrino experiment in USA for precision measurements.

### T2K
**Tokai to Kamioka**. Long-baseline neutrino experiment in Japan.

### NOvA
**NuMI Off-Axis νₑ Appearance**. Neutrino experiment in USA.

## Mathematical Notation

### dim(G)
Dimension of Lie group or algebra G. Example: dim(E₈) = 248.

### rank(G)
Rank of Lie algebra G. Example: rank(E₈) = 8.

### Hⁿ(M)
n-th cohomology group of manifold M. Example: H²(K₇) = ℝ²¹.

### bₙ
n-th Betti number, dimension of Hⁿ. Example: b₃(K₇) = 77.

### |·|
Absolute value or cardinality (size of set).

### ⊕
Direct sum of vector spaces or algebras.

### ⊗
Tensor product.

### ∈
"Element of" (set membership).

### ∀
"For all" (universal quantifier).

### ∃
"There exists" (existential quantifier).

### ≈
Approximately equal.

### ≡
Identically equal or defined as.

### ∼
Of the same order of magnitude, or equivalent to.

## Greek Letters in GIFT

### α (alpha)
Fine structure constant, α ≈ 1/137.

### β₀ (beta)
Base coupling parameter, β₀ = 1/(4π²).

### γ (gamma)
Heat kernel coefficient or Euler-Mascheroni constant.

### δ (delta)
CP violation phase (δ_CP) or small deviation.

### ε₀ (epsilon)
Symmetry breaking parameter, ε₀ = 1/8.

### ζ (zeta)
Riemann zeta function, ζ(3) appears in δ_CP.

### θ (theta)
Mixing angles (θ₁₂, θ₁₃, θ₂₃, θ_W).

### ξ (xi)
Correlation parameter, ξ = 5β₀/2.

### Ω (Omega)
Density parameters (Ω_DE for dark energy).

### φ (phi)
Golden ratio or angle.

## Acronyms

### GIFT
Geometric Information Field Theory

### SM
Standard Model (of particle physics)

### GUT
Grand Unified Theory

### QCD
Quantum Chromodynamics (strong force)

### QED
Quantum Electrodynamics (electromagnetic force)

### EW
Electroweak (unified electromagnetic and weak theory)

### SUSY
Supersymmetry

### CMB
Cosmic Microwave Background

### VEV
Vacuum Expectation Value

### RG
Renormalization Group

### UV
Ultraviolet (high energy)

### IR
Infrared (low energy)

### LO/NLO/NNLO
Leading Order / Next-to-Leading Order / Next-to-Next-to-Leading Order (perturbative expansion)

## Unit Conventions

### Natural Units
System where ℏ = c = 1. Energies, masses, and momenta have same dimensions.

### GeV
Giga-electron-volt, 10⁹ eV. Common energy/mass unit in particle physics.
- Proton mass: ~1 GeV
- Higgs boson mass: ~125 GeV
- Top quark mass: ~173 GeV

### MeV
Mega-electron-volt, 10⁶ eV.
- Electron mass: ~0.511 MeV
- Quark masses: few MeV to few GeV

## Status Indicators in Documentation

### [PROVEN]
Mathematical claim with rigorous proof provided.

### [TOPOLOGICAL]
Direct consequence of manifold topology.

### [EXACT]
Zero deviation by mathematical construction.

### [HIGH-PRECISION]
Experimental agreement <1% deviation.

### [PRELIMINARY]
Calculation or result pending refinement.

## Cross-References

For more detailed explanations:
- **Mathematical foundations**: See Supplement A
- **Particle physics background**: See main paper Section 1
- **Experimental values**: See Supplement D
- **Common questions**: See `docs/FAQ.md`

## Contributing

This glossary is continuously updated. To suggest additions or corrections:
1. Open issue at https://github.com/gift-framework/GIFT/issues
2. Tag as "documentation"
3. Specify term and proposed definition

### Torsional Dynamics (v2.1)
Framework introduced in v2.1 connecting non-zero torsion on K₇ to RG flow. Key parameters: |T_norm| = 0.0164, |T_costar| = 0.0141.

### Scale Bridge (v2.1)
Mathematical infrastructure linking dimensionless to dimensional observables: Λ_GIFT = 21×e⁸×248/(7×π⁴) ≈ 1.63×10⁶.

### Lean 4 (v2.3)
Theorem prover used for formal verification of GIFT exact relations. The `/Lean/` directory contains 17 modules proving all 13 exact relations with zero domain-specific axioms. Key theorem: `GIFT_framework_certified`.

---

Last updated: v2.3 (2025-12-03)

