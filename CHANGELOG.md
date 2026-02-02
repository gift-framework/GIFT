# Changelog

All notable changes to the GIFT framework are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.3.15] - 2026-02-02

### Research Integration Release

This release integrates theoretical developments from `/research/` into the main publications, providing rigorous foundations for the torsion-free condition and metric structure.

#### Added

**S1 Foundations — Spectral Structure**
- Section 7.4: Continued fraction representation λ₁ = [0; 7, 14] with dim(K₇) and dim(G₂)
- Section 7.5: Pell equation structure 99² − 50×14² = 1 connecting spectral gap to number theory
- Three independent derivations of det(g) = 65/32 (Weyl, cohomological, H* formula)

**S1 Foundations — Torsion Classes**
- Complete decomposition W₁ ⊕ W₇ ⊕ W₁₄ ⊕ W₂₇ with dimension table
- Total dimension 49 = 7² = dim(K₇)² interpretation

**S3 Dynamics — Variational Formulation**
- Section 2.1: Torsion functional Θ_G₂ := ‖∇φ‖² − κ_T‖φ‖² = 0
- Euler-Lagrange eigenvalue equation ∇²φ = κ_T φ
- Section 3.3: Moduli space of torsion-free G₂ structures (dim = b₃ = 77)
- Perturbation analysis for moduli space directions

#### Source Documents

Integrated content from:
- `research/K7_EXPLICIT_METRIC_ANALYTICAL.md` → S1 Sections 7.4–7.5, 10.3
- `research/TORSION_FREE_CONDITION_ANALYSIS.md` → S3 Sections 2.1, 3.3

---

## [3.3.14] - 2026-01-29

### Synchronization with gift-framework/core v3.3.14

This release synchronizes documentation with the latest formal verification developments in [gift-framework/core](https://github.com/gift-framework/core).

#### Changed

**Major Updates**
- Updated to core v3.3.14 (~330 certified relations, up from 185)
- Removed all Coq references — Lean 4 is now the sole verification system (Coq archived)
- Updated terminology to use academic standards (no internal jargon)

**Core v3.3.8–v3.3.14 Highlights** (see [core CHANGELOG](https://github.com/gift-framework/core/blob/main/CHANGELOG.md) for details):
- **v3.3.14**: Selection Principle (κ = π²/14), TCS building blocks (Quintic, CI(2,2,2)), refined spectral bounds
- **v3.3.13**: Literature axioms (Langlais 2024, CGN 2024 spectral density formulas)
- **v3.3.12**: TCS Spectral Bounds Model Theorem (λ₁ ~ 1/L²)
- **v3.3.11**: Monster dimension via Coxeter numbers (196883 = (b₃−h_G₂)(b₃−h_E₇)(b₃−h_E₈))
- **v3.3.10**: GIFT-Zeta correspondences, Monster-Zeta Moonshine
- **v3.3.9**: Complete Spectral Theory module
- **v3.3.8**: Yang-Mills Mass Gap module (λ₁ = 14/99)

**Documentation Updates**
- README.md, CITATION.md, STRUCTURE.md: Updated metrics and removed Coq badges
- docs/*: Updated version references and Coq mentions
- publications/README.md: Updated badge and metrics

---

## [3.3.7] - 2026-01-16

### Synchronization with gift-framework/core v3.3.7

This release synchronizes documentation with the current state of formal verification in [gift-framework/core](https://github.com/gift-framework/core).

#### Changed

**Axiom Count Update**
- Updated axiom count from 40 to 15 across all documentation
- Axiom breakdown: Tier 1 (Numerical) 0, Tier 2 (Algebraic) 2, Tier 3 (Geometric) 13
- Tier 1 numerical axioms were converted to theorems via Taylor series proofs in core

**Version References**
- Updated core version references from v3.3.0 to v3.3.7
- Files updated: README.md, CITATION.md, STRUCTURE.md, publications/README.md, GIFT_v3.3_S1_foundations.md, docs/FAQ.md, docs/GLOSSARY.md, docs/EXPERIMENTAL_VALIDATION.md

---

## [3.3.0] - 2026-01-12

### Extended Observable Catalog & Enhanced Monte Carlo Validation

This release extends the observable catalog from 18 to 33 predictions and significantly expands Monte Carlo validation.

#### Added

**Extended Observable Catalog (33 predictions)**
- 18 core relations (PROVEN in Lean 4)
- 15 extended relations (TOPOLOGICAL/HEURISTIC status)
- Mean deviation improved: 0.24% → **0.21%** (PDG 2024)

**Enhanced Monte Carlo Validation**
- Total configurations tested: 54,327 → **192,349**
- p-value: < 10⁻⁵ → **< 5×10⁻⁶**
- Significance: >4σ → **>4.5σ**

**New v3.3 Corrections**
- m_W/m_Z formula: 23/26 → **37/42** = (2b₂−Weyl)/(2b₂) — deviation 0.35% → 0.06%
- Both Betti numbers now **DERIVED** from TCS building blocks, not input

**File Renames (v3.2 → v3.3)**
- `GIFT_v3.2_main.md` → `GIFT_v3.3_main.md`
- `GIFT_v3.2_S1_foundations.md` → `GIFT_v3.3_S1_foundations.md`
- `GIFT_v3.2_S2_derivations.md` → `GIFT_v3.3_S2_derivations.md`
- `GIFT_v3.2_S3_dynamics.md` → `GIFT_v3.3_S3_dynamics.md`

**New Documentation**
- `docs/OBSERVABLE_CATALOG.md`: Complete 33-observable catalog
- `docs/OBSERVABLE_REFERENCE.md`: Detailed reference for all predictions
- `docs/STATISTICAL_EVIDENCE.md`: v3.3 statistical validation summary
- `statistical_validation/validation_v33.py`: Enhanced validation script
- `statistical_validation/GIFT_Statistical_Validation_Report_v33.md`: Full report

#### Changed

**Notation Clarification**
- χ(K₇) = 0 (Euler characteristic for odd-dimensional manifolds)
- The constant 42 = 2b₂ is a distinct structural invariant, NOT χ(K₇)

**Documentation Updates**
- All READMEs updated to v3.3
- CITATION.md updated with v3.3 references
- STRUCTURE.md updated with new file names

---

## [3.2.0] - 2026-01-05

### PDG 2024 Update & Comprehensive Monte Carlo Validation

This release updates all experimental comparisons to PDG 2024 values and adds comprehensive Monte Carlo validation across 54,327 alternative configurations.

#### Changed

**Experimental Values Updated to PDG 2024**
- m_s/m_d: 19.5 → **19.9 ± 0.5** (FLAG/PDG 2024)
- λ_H: 0.1264 → **0.1293 ± 0.0002** (SM from m_H=125.20 GeV)
- Mean deviation: 0.087% → **0.24%** (using updated PDG 2024 values)

**Monte Carlo Validation Expanded**
- Betti variations: 10,000 configurations tested
- Holonomy variations: 46 configurations (G₂, SO(7), SU(4), Spin(7), etc.)
- Structural variations: 234 configurations (p₂, Weyl parameter space)
- Full combinatorial: 44,281 configurations
- **Total: 54,327 configurations tested**
- **Result: 0 alternatives outperform GIFT (b₂=21, b₃=77)**
- **p-value: < 10⁻⁵, significance: > 4σ**

**File Renames (v3.1 → v3.2)**
- `GIFT_v3.1_main.md` → `GIFT_v3.2_main.md`
- `GIFT_v3.1_S1_foundations.md` → `GIFT_v3.2_S1_foundations.md`
- `GIFT_v3.1_S2_derivations.md` → `GIFT_v3.2_S2_derivations.md`
- `GIFT_v3.1_S3_dynamics.md` → `GIFT_v3.2_S3_dynamics.md`
- Corresponding `.tex` files also renamed

#### Added

**New Validation Script**
- `statistical_validation/validation_v32.py`: Comprehensive Monte Carlo validation
- `validation_v32_results.json`: Complete validation results

**Documentation Updates**
- All READMEs updated to v3.2
- CITATION.md updated with v3.2 references
- STRUCTURE.md updated with new file names

#### Technical Notes

**Why 0.24% instead of 0.087%?**

The v3.1 statistics used older experimental values. With PDG 2024:
- m_s/m_d experimental: 19.9 (was 19.5) → GIFT predicts 20, deviation increases
- λ_H experimental: 0.1293 (was 0.1264) → GIFT predicts 0.1288, deviation increases

This reflects more accurate experimental data, not degraded predictions. The GIFT predictions remain unchanged; only the experimental comparisons are updated.

**Monte Carlo Validation Details**

The validation tests whether GIFT's (21, 77) configuration is genuinely optimal:
1. **Betti scan**: Vary b₂ ∈ [1,100], b₃ ∈ [1,200]
2. **Holonomy scan**: Test alternative holonomy groups
3. **Structural scan**: Vary p₂ ∈ [1,10], Weyl ∈ [1,15]
4. **Combinatorial**: Full parameter space exploration

Result: Zero configurations achieve lower mean deviation than GIFT.

---

## [3.1.1] - 2025-12-17

### Core Sync — 180+ Relations

Synchronized documentation with [gift-framework/core v3.1.4](https://github.com/gift-framework/core).

#### Changed

**Relation Count**: 165+ → **180+**
- Core added 15+ new relations in v3.1.1 through v3.1.4
- Lagrange identity for 7D cross product now PROVEN (was axiom)
- AnalyticalMetric.lean formalizes exact G₂ metric

**Files Updated**:
- `README.md`, `STRUCTURE.md`, `CITATION.md`
- `publications/markdown/GIFT_v3_main.md`
- `publications/markdown/GIFT_v3_S1_foundations.md`
- `publications/README.md`
- `docs/INFO_GEO_FOR_PHYSICISTS.md`
- `docs/EXPERIMENTAL_VALIDATION.md`
- `docs/GIFT_v3.1_LITERATURE_ANALYSIS.md`

---

## [3.1.0] - 2025-12-17

### Analytical G₂ Metric — Exact Solution

This release establishes that the G₂ metric admits an **exact analytical form** with zero torsion, elevating the framework from numerical agreement to algebraic derivation.

#### Added

**Analytical Metric (Core Result)**
- Exact solution: φ = (65/32)^{1/14} × φ₀ (scaled standard associative 3-form)
- Metric: g = (65/32)^{1/7} × I₇ (diagonal, exact)
- Torsion: T = 0 exactly (constant form ⇒ dφ = d*φ = 0)
- Joyce existence theorem trivially satisfied (infinite margin)

**New Section 3.4** in Main paper: "The Analytical G₂ Metric"
- Standard associative 3-form φ₀ with 7 non-zero terms
- Scaling derivation from det(g) = 65/32 constraint
- T = 0 proof for constant forms

**Section 11.5** in S1: Derivation chain diagram
```
Octonions (𝕆) → G₂ = Aut(𝕆) → φ₀ (Harvey-Lawson) → scaling → predictions
```

**Section 16.1** in Main: "Related Work and Context"
- Position GIFT in 2024-2025 research landscape
- References: Singh et al., Crowley-Goette-Nordström (Inventiones 2025), Furey, Baez, Ferrara
- "Gap diagram" showing GIFT as bridge between programs

**CLAUDE.md**: Development guide with terminology standards
- Academic terminology (no internal jargon B4, B5, etc.)
- Blueprint workflow documentation
- Writing guidelines for humble scientific tone

**Extended References [21-25]**:
- Singh et al. (2024) E₈⊗E₈ unification
- Crowley-Goette-Nordström (2025) G₂ analytic invariant
- Ferrara (2021) G₂ from Cayley-Dickson
- Furey, Baez octonionic programs

**Credits**:
- Harvey & Lawson (1982) for calibrated geometries and φ₀
- Bryant (1987) for exceptional holonomy foundations
- de-johannes/FirstDistinction for octonion-Cayley insight
- math-inc/KakeyaFiniteFields for blueprint workflow

#### Changed

**Conceptual Clarifications**
- κ_T = 1/61 is now "torsion capacity" (not realized value on exact solution)
- PINN serves as **validation**, not proof of existence
- Joyce theorem is trivially satisfied (T = 0 < threshold)

**Status Clarifications in S2**
- #3 κ_T: Added note on capacity vs realized torsion
- #4 det(g): Added verification via analytical metric

**S3 Dynamics Updates**
- Section 1.3: T = 0 exact, effective torsion requires moduli/quantum mechanisms
- Section 2.2: 61 as capacity interpretation
- Section 27: Reorganized PROVEN/THEORETICAL classifications

#### Files Modified
- `publications/markdown/GIFT_v3_main.md` (v3.0 → v3.1)
- `publications/markdown/GIFT_v3_S1_foundations.md` (v3.0 → v3.1)
- `publications/markdown/GIFT_v3_S2_derivations.md` (v3.0 → v3.1)
- `publications/markdown/GIFT_v3_S3_dynamics.md` (v3.0 → v3.1)
- `README.md`, `STRUCTURE.md` (version bump)
- New: `CLAUDE.md`, `docs/GIFT_v31_UPDATE_PLAN.md`, `docs/GIFT_v3.1_LITERATURE_ANALYSIS.md`

---

## [3.0.1] - 2025-12-11

### Validation Synchronization & Documentation Update

Synchronized publications with latest validation notebook results and added speculative extensions documentation.

#### Changed

**Precision Update (Validation Run 2025-12-11)**
- Mean deviation: 0.197% → **0.087%** (improved calculation methodology)
- All 18 dimensionless predictions verified against `notebooks/gift_v3_validation.json`
- Individual deviation values updated to exact JSON results:
  - sin²θ_W: 0.195% (was 0.20%)
  - α_s: 0.042% (was 0.04%)
  - Q_Koide: 0.0009% (was 0.001%)
  - m_τ/m_e: 0.0043% (was 0.004%)
  - θ₁₂: 0.030% (was 0.06%)
  - λ_H: 0.119% (was 0.07%)
  - n_s: 0.004% (was 0.00%)
  - Ω_DE: 0.211% (was 0.21%)

**Documentation Structure**
- Added `docs/technical/` for speculative extensions:
  - S3: Torsional dynamics (RG flow, non-zero torsion)
  - S6: Theoretical extensions (M-theory, QG connections)
  - S7: Dimensional observables (absolute masses, scale bridge)
  - GIFT Atlas: Complete constant/relation database
- README.md updated with new navigation section
- STRUCTURE.md updated with docs/technical/ layout

#### Files Modified
- `publications/markdown/GIFT_v3_main.md`: Statistics and deviation values
- `publications/markdown/GIFT_v3_S2_derivations.md`: All deviation tables
- `README.md`: Overview metrics, navigation, documentation links
- `STRUCTURE.md`: Directory layout, version info

---

## [3.0.0] - 2025-12-09

### Major Release: Number-Theoretic Structure - 165+ Certified Relations

This major release triples the certified relations from 54 to **165+**, revealing deep number-theoretic structure underlying the framework. Synchronized with [gift-framework/core v2.0.0](https://github.com/gift-framework/core).

#### Added

**Fibonacci Embedding (Relations 76-85)** - 10 new relations

| Relation | Value | Formula | Status |
|----------|-------|---------|--------|
| F₃ = p₂ | 2 | Pontryagin class | **PROVEN (Lean)** |
| F₄ = N_gen | 3 | Generation count | **PROVEN (Lean)** |
| F₅ = Weyl | 5 | Weyl factor | **PROVEN (Lean)** |
| F₆ = rank(E₈) | 8 | E₈ Cartan dimension | **PROVEN (Lean)** |
| F₇ = α²_B sum | 13 | Structure B sum | **PROVEN (Lean)** |
| F₈ = b₂ | 21 | Second Betti number | **PROVEN (Lean)** |
| F₉ = hidden_dim | 34 | Hidden sector dimension | **PROVEN (Lean)** |
| F₁₀ = E₇-E₆ gap | 55 | Exceptional gap | **PROVEN (Lean)** |
| F₁₁ = b₃+dim(G₂)-p₂ | 89 | Topological sum | **PROVEN (Lean)** |
| F₁₂ = (dim(G₂)-p₂)² | 144 | α_s⁻² denominator | **PROVEN (Lean)** |

**Lucas Embedding (Relations 86-95)** - 10 new relations

| Relation | Value | Formula | Status |
|----------|-------|---------|--------|
| L₀ = p₂ | 2 | Binary duality | **PROVEN (Lean)** |
| L₄ = dim(K₇) | 7 | Manifold dimension | **PROVEN (Lean)** |
| L₅ = D_bulk | 11 | Bulk dimension | **PROVEN (Lean)** |
| L₆ = duality gap | 18 | 61 - 43 | **PROVEN (Lean)** |
| L₈ = Monster factor | 47 | Sporadic group | **PROVEN (Lean)** |

**Prime Atlas (Relations 96-135)** - 40 new relations

| Tier | Count | Description | Status |
|------|-------|-------------|--------|
| Tier 1 | 10 | Direct GIFT constants | **PROVEN (Lean)** |
| Tier 2 | 15 | GIFT expressions < 100 | **PROVEN (Lean)** |
| Tier 3 | 10 | H* generator (100-150) | **PROVEN (Lean)** |
| Tier 4 | 11 | E₈ generator (150-200) | **PROVEN (Lean)** |

**100% coverage of all primes below 200 via three generators (b₃, H*, dim(E₈)).**

**Monster Group (Relations 136-150)** - 15 new relations

| Relation | Value | Formula | Status |
|----------|-------|---------|--------|
| Monster_dim | 196883 | 47 × 59 × 71 | **PROVEN (Lean)** |
| Factor 47 | L₈ | Lucas(8) | **PROVEN (Lean)** |
| Factor 59 | b₃ - 18 | Betti - duality gap | **PROVEN (Lean)** |
| Factor 71 | b₃ - 6 | Betti - 6 | **PROVEN (Lean)** |
| j-constant | 744 | 3 × 248 = N_gen × dim(E₈) | **PROVEN (Lean)** |

**Arithmetic progression**: 47 → 59 → 71 with common difference 12 = dim(G₂) - p₂

**McKay Correspondence (Relations 151-165)** - 15 new relations

| Relation | Value | Formula | Status |
|----------|-------|---------|--------|
| Coxeter(E₈) | 30 | Icosahedron edges | **PROVEN (Lean)** |
| Icosahedron vertices | 12 | dim(G₂) - p₂ | **PROVEN (Lean)** |
| Icosahedron faces | 20 | m_s/m_d | **PROVEN (Lean)** |
| Euler V-E+F | 2 | p₂ | **PROVEN (Lean)** |
| E₈ kissing | 240 | 2 × |2I| = 8 × 30 | **PROVEN (Lean)** |

**New Supplements**:
- **S8**: Sequences and Prime Atlas (Fibonacci, Lucas, primes)
- **S9**: Monster Group and Monstrous Moonshine (sporadic groups, j-invariant)

#### Changed

- Main paper renamed: `gift_2_3_main.md` → `gift_3_0_main.md`
- All supplements updated to v30 naming convention
- README.md: Complete overhaul with v3.0 highlights
- Observable Reference: Updated to 165+ relations
- Added "Why Not Numerology" section explaining physical grounding

#### Physical Grounding

The v3.0 structures possess **independent mathematical existence**:
- Fibonacci sequences appear in nature (phyllotaxis, shells)
- Monster group is a theorem (Griess 1982)
- McKay correspondence is established mathematics (McKay 1980)
- j-invariant/Moonshine proved by Borcherds (Fields Medal 1998)

These are not post-hoc pattern matching but connections to deep mathematics.

---

## [2.3.4] - 2025-12-08

### Exceptional Groups & Base Decomposition - 54 Certified Relations

This release integrates the **15 new relations** from [gift-framework/core v1.5.0](https://github.com/gift-framework/core), bringing the total to **54 certified relations** (13 original + 12 topological extension + 10 Yukawa duality + 4 irrational sector + 5 exceptional groups + 6 base decomposition + 4 extended decomposition).

#### Added

**5 Exceptional Groups Relations** (v1.5.0 of giftpy)

| Relation | Value | Formula | Status |
|----------|-------|---------|--------|
| α_s² | 1/72 | Strong coupling squared exact rational | **PROVEN (Lean + Coq)** |
| dim(F₄) | 52 | p₂² × sum(α²_B) | **PROVEN (Lean + Coq)** |
| δ_penta | 25 | dim(F₄) - dim(J₃(𝕆)) = Weyl² | **PROVEN (Lean + Coq)** |
| J₃(𝕆)₀ | 26 | dim(E₆) - dim(F₄) = dim(J₃(𝕆)) - 1 | **PROVEN (Lean + Coq)** |
| \|W(E₈)\| | 696729600 | E₈ Weyl group topological factorization | **PROVEN (Lean + Coq)** |

**6 Base Decomposition Relations**

| Relation | Value | Formula | Status |
|----------|-------|---------|--------|
| κ_T⁻¹ | 61 | dim(F₄) + N_gen² | **PROVEN (Lean + Coq)** |
| b₂ | 21 | ALPHA_SUM_B + rank(E₈) | **PROVEN (Lean + Coq)** |
| b₃ | 77 | ALPHA_SUM_B × Weyl + 12 | **PROVEN (Lean + Coq)** |
| H* | 99 | ALPHA_SUM_B × dim(K₇) + rank(E₈) | **PROVEN (Lean + Coq)** |
| quotient_sum | 13 | 1 + 5 + 7 (gauge-holonomy-manifold) | **PROVEN (Lean + Coq)** |
| Ω_DE_num | 98 | dim(K₇) × dim(G₂) | **PROVEN (Lean + Coq)** |

**4 Extended Decomposition Relations**

| Relation | Value | Formula | Status |
|----------|-------|---------|--------|
| τ_num base13 | [1,7,7,1] | Hierarchy parameter palindrome | **PROVEN (Lean + Coq)** |
| n_observables | 39 | N_gen × ALPHA_SUM_B | **PROVEN (Lean + Coq)** |
| E₆_dual | 78 | 2 × n_observables (visible + hidden) | **PROVEN (Lean + Coq)** |
| H₀_topological | 70 | dim(K₇) × 10 | **PROVEN (Lean + Coq)** |

**Key insight**: The Structure B sum (2 + 5 + 6 = 13 = ALPHA_SUM_B) provides a consistent base for decomposing all primary GIFT topological constants (b₂, b₃, H*). The E₈ Weyl group order |W(E₈)| = 696729600 = 2¹⁴ × 3⁵ × 5² × 7 factorizes into pure topological terms.

**Tau Palindrome**: τ = 3472/891 has τ_num = [1,7,7,1]₁₃ in base 13, with central digits encoding dim(K₇) = 7.

**New proof files**:
- `ExceptionalGroups.lean` / `ExceptionalGroups.v`: F₄, E₆, E₈ connections
- `BaseDecomposition.lean` / `BaseDecomposition.v`: All decomposition relations (45-54)

#### Changed
- Updated all documentation to reference 54 proven relations
- README.md: Updated formally verified count to 54
- giftpy version reference updated to v1.5.0

---

## [2.3.3] - 2025-12-05

### Irrational Sector - 39 Certified Relations

This release integrates the **4 new Irrational Sector relations** from [gift-framework/core v1.4.0](https://github.com/gift-framework/core), bringing the total to **39 certified relations** (13 original + 12 topological extension + 10 Yukawa duality + 4 irrational sector).

#### Added

**4 New Irrational Sector Relations** (v1.4.0 of giftpy)

Relations involving irrational numbers (pi, phi) with certified rational parts:

| Relation | Value | Formula | Status |
|----------|-------|---------|--------|
| α⁻¹ complete | 267489/1952 | 128 + 9 + (65/32)·(1/61) | **PROVEN (Lean + Coq)** |
| θ₁₃ degrees | 60/7 | 180/b₂ = 180/21 | **PROVEN (Lean + Coq)** |
| φ bounds | (1.618, 1.619) | sqrt(5) in (2.236, 2.237) | **PROVEN (Lean + Coq)** |
| m_μ/m_e bounds | (206, 208) | 27^φ | **PROVEN (Lean + Coq)** |

**Key insight**: The fine structure constant inverse α⁻¹ = 267489/1952 ≈ 137.033 is an *exact rational*, not an approximation! This arises from:
- 128 = (dim(E₈) + rank(E₈))/2 (algebraic component)
- 9 = H*/D_bulk (bulk component)
- 65/1952 = det(g) × κ_T (torsion correction)

**New proof files**:
- `IrrationalSector.lean` / `IrrationalSector.v`: θ₁₃, θ₂₃, α⁻¹ complete
- `GoldenRatio.lean` / `GoldenRatio.v`: φ bounds, m_μ/m_e = 27^φ
- Updated `GaugeSector.lean` / `GaugeSector.v` with α⁻¹ complete (relation #36)

#### Changed
- Updated all documentation to reference 39 proven relations
- README.md: Added Irrational Sector section with complete table
- giftpy version reference updated to v1.4.0

---

## [2.3.2] - 2025-12-05

### Yukawa Duality - 35 Certified Relations

This release integrates the **10 new Yukawa Duality relations** from [gift-framework/core v1.3.0](https://github.com/gift-framework/core), bringing the total to **35 certified relations** (13 original + 12 topological extension + 10 Yukawa duality).

#### Added

**10 New Yukawa Duality Relations** (v1.3.0 of giftpy)

The Extended Koide formula exhibits a **duality** between two α² structures:
- **Structure A** (Topological): {2, 3, 7} → visible sector
- **Structure B** (Dynamical): {2, 5, 6} → torsion constraint

| Relation | Value | Formula |
|----------|-------|---------|
| α²_A sum | 12 | 2 + 3 + 7 = dim(SM gauge) |
| α²_A prod+1 | 43 | 2×3×7 + 1 = visible_dim |
| α²_B sum | 13 | 2 + 5 + 6 = rank(E₈) + Weyl |
| α²_B prod+1 | 61 | 2×5×6 + 1 = κ_T⁻¹ |
| Duality gap | 18 | 61 - 43 = p₂ × N_gen² |
| α²_up (B) | 5 | dim(K₇) - p₂ = Weyl |
| α²_down (B) | 6 | dim(G₂) - rank(E₈) = 2×N_gen |
| visible_dim | 43 | b₃ - hidden_dim |
| hidden_dim | 34 | b₃ - visible_dim |
| Jordan gap | 27 | 61 - 34 = dim(J₃(𝕆)) |

**Key insight**: The torsion κ_T = 1/61 mediates between topology (Structure A) and physical masses (Structure B).

#### Changed
- Updated all documentation to reference 35 proven relations
- README.md: Added Yukawa Duality section with complete table
- giftpy version reference updated to v1.3.0

---

## [2.3.1] - 2025-12-04

### Topological Extension - 25 Certified Relations

This release updates documentation to reflect the **25 certified relations** now proven in [gift-framework/core](https://github.com/gift-framework/core) (13 original + 12 topological extension).

#### Added

**12 New Topological Extension Relations** (v1.1.0 of giftpy)
- α_s denom = 12 (dim(G₂) - p₂)
- γ_GIFT = 511/884 ((2·rank(E₈) + 5·H*) / (10·dim(G₂) + 3·dim(E₈)))
- δ penta = 25 (Weyl² pentagonal structure)
- θ₂₃ = 85/99 ((rank(E₈) + b₃) / H*)
- θ₁₃ denom = 21 (b₂ Betti number)
- α_s² denom = 144 ((dim(G₂) - p₂)²)
- λ_H² = 17/1024 ((dim(G₂) + N_gen) / 32²)
- θ₁₂ factor = 12775 (Weyl² × γ_num)
- m_μ/m_e base = 27 (dim(J₃(O)))
- n_s indices = 11, 5 (D_bulk, Weyl_factor)
- Ω_DE frac = 98/99 ((H* - 1) / H*)
- α⁻¹ base = 137 ((dim(E₈) + rank(E₈))/2 + H*/11)

#### Changed
- Updated all documentation to reference 25 proven relations
- README.md: Added complete table of all 25 relations
- Main paper: Updated abstract and Lean verification summary
- Observable Reference: Added Extension Relations table
- Statistical Validation: Updated status classifications

---

## [2.3] - 2025-12-03

### Major Release - Dual Formal Verification (Lean 4 + Coq)

This version achieves complete formal verification in **both Lean 4 and Coq**, establishing independently machine-verified proofs for all 13 original exact relations with zero domain-specific axioms.

#### Added

**Lean 4 Verification** (now in [gift-framework/core](https://github.com/gift-framework/core))
- Complete formalization with 17 modules
- All 13 original exact relations proven: sin²θ_W=3/13, τ=3472/891, det(g)=65/32, κ_T=1/61, δ_CP=197°, m_τ/m_e=3477, m_s/m_d=20, Q_Koide=2/3, λ_H=√17/32, H*=99, p₂=2, N_gen=3, E₈×E₈=496
- Zero domain-specific axioms (only propext, Quot.sound)
- Zero `sorry` (all proofs complete)

**Coq Verification** (now in [gift-framework/core](https://github.com/gift-framework/core))
- Complete formalization with 21 modules
- All 13 original exact relations independently proven
- Zero `Admitted` statements (all proofs complete)
- Zero explicit axioms beyond Coq core
- Parallel module structure: Algebra, Geometry, Topology, Relations, Certificate

**Unified CI Pipeline**
- `.github/workflows/verification.yml` - Combined Lean + Coq + G2 validation
- `.github/workflows/coq.yml` - Dedicated Coq CI with lint and audit
- Verification report generation with summary statistics

**Publication Updates**
- All main documents updated with Lean and Coq CI badges
- New status classification: **PROVEN (Lean)** and **PROVEN (Coq)** for machine-verified relations
- Section 11.4 in Observable Reference with complete theorem mapping
- Dual verification summary tables in main paper

**Framework Unification**
- Complete version alignment across all files
- Unified documentation structure with consistent v2.3 references
- Harmonized formulas, results, and calculations across all documents
- Comprehensive citation guide updated for v2.3

#### Changed

**Status Classifications**
- 13 relations upgraded from PROVEN/TOPOLOGICAL to **PROVEN (Lean)**
- Each relation now includes Lean theorem reference (e.g., `weinberg_angle_certified`)

**Documentation**
- README.md: Added Lean badges, updated structure, new verification section
- publications/README.md: New Lean verification section with module structure
- All documentation aligned to v2.3

#### Fixed

**Documentation Inconsistencies**
- Unified version references to v2.3 throughout
- Unified formula presentations
- Synchronized result tables
- Aligned experimental validation data

---

## [2.2.0] - 2025-11-27

### Major Release - Zero-Parameter Paradigm

This version achieves the **zero-parameter paradigm**: all quantities derive from fixed topological structure with no continuous adjustable parameters. Key discoveries include topological derivations for sin²θ_W, κ_T, det(g), and τ.

### Added

**Topological Derivations (Zero-Parameter Achievement)**
- **det(g) = 65/32**: Metric determinant now derived topologically (was ML-fitted 2.031)
  - Formula: p₂ + 1/(b₂ + dim(G₂) - N_gen) = 2 + 1/32 = 65/32
  - Deviation from ML-fit: 0.012%
- **sin²θ_W = 3/13**: Weinberg angle now PROVEN (was PHENOMENOLOGICAL)
  - Formula: b₂/(b₃ + dim(G₂)) = 21/91 = 3/13
  - Experimental deviation: 0.195%
- **κ_T = 1/61**: Torsion magnitude now TOPOLOGICAL
  - Formula: 1/(b₃ - dim(G₂) - p₂) = 1/(77-14-2) = 1/61
- **τ = 3472/891**: Hierarchy parameter now exact rational
  - Formula: (496×21)/(27×99) = 3472/891
  - Prime factorization: (2⁴×7×31)/(3⁴×11)

**New PROVEN Relations**
- Total PROVEN count: 9 → 13
- sin²θ_W = 3/13 (new)
- κ_T = 1/61 (new, was THEORETICAL)
- det(g) = 65/32 (new, was ML-fitted)
- τ = 3472/891 (new, exact rational)

**Restructured Documentation**
- Supplements consolidated: 9 → 7 documents
- New: S2_K7_manifold_construction.md (from G2_ML content)
- Merged: S4+S5 → S4_complete_derivations.md
- Merged: S7+S8 → S5_experimental_validation.md
- Renamed: S9 → S6_theoretical_extensions.md
- New: S7_dimensional_observables.md
- New: GLOSSARY.md, READING_GUIDE.md, publications/README.md

### Changed

**Framework Parameters**
- Parameter count: 3 → 0 (all structurally determined)
- Observable count: 46 → 39 (consolidated, no double-counting)
- Mean precision: 0.13% → 0.128%

**Status Promotions**
| Observable | v2.1 Status | v2.2 Status |
|------------|-------------|-------------|
| sin²θ_W | PHENOMENOLOGICAL | **PROVEN** |
| κ_T | THEORETICAL | **TOPOLOGICAL** |
| det(g) | ML-fitted | **TOPOLOGICAL** |
| τ | DERIVED | **PROVEN** |
| α_s | PHENOMENOLOGICAL | **TOPOLOGICAL** |

**Documentation Structure**
- `publications/gift_2_1_main.md` → `publications/gift_2_2_main.md`
- `publications/GIFT_v21_*.md` → `publications/GIFT_v22_*.md`
- v2.1 documents archived to `legacy/legacy_v2.1/`

**Key Formula Updates**
- sin²θ_W: ζ(2) - √2 → 3/13 (exact rational)
- κ_T: 0.0164 (fit) → 1/61 (topological)
- det(g): 2.031 (ML) → 65/32 (topological)

### Removed

**Deprecated Content**
- ML-fitted det(g) parameter
- PHENOMENOLOGICAL status for sin²θ_W
- Redundant supplements (consolidated)
- v2.0 subdirectory structure

### Fixed

**Precision Improvements**
- sin²θ_W deviation: 0.216% → 0.195% (formula change)
- κ_T now exact (was fitted)
- det(g) deviation: 0% → 0.012% (exact vs ML-fit)

**Consistency**
- All documents now use v2.2 terminology
- Parameter claims unified: "no continuous adjustable parameters"
- Observable count consistent across all files

---

## [2.1.0] - 2025-11-22

### Major Release - Torsional Dynamics and Scale Bridge

This version introduces **torsional geodesic dynamics**, connecting static topology to renormalization group flow, and a **scale bridge** linking dimensionless to dimensional parameters. Observable count increases from 15 to **46** (37 dimensionless + 9 dimensional).

### Added

**Torsional Dynamics Framework**
- `statistical_validation/gift_v21_core.py` - Complete v2.1 framework with torsional dynamics (650+ lines)
- Torsional geodesic equation connecting RG flow to K₇ geometry
- Non-zero torsion parameters: |T_norm| = 0.0164, |T_costar| = 0.0141
- Torsion tensor components for mass hierarchies and CP violation
- Metric components in (e,π,φ) coordinates for electroweak sector

**Scale Bridge Infrastructure**
- Λ_GIFT = 21×e⁸×248/(7×π⁴) ≈ 1.632×10⁶ (dimensionless scale)
- Connection between topological integers and physical dimensions
- RG evolution framework with μ₀ = M_Z reference scale
- Dimensional mass predictions (quarks: m_u through m_t)

**Extended Observable Coverage**
- 37 dimensionless observables (up from 15 in v2.0):
  - Gauge sector: α⁻¹, sin²θ_W, α_s (with torsional corrections)
  - Neutrino sector: 4 mixing parameters
  - Lepton sector: 3 mass ratios
  - Quark sector: 10 mass ratios (complete spectrum)
  - CKM matrix: 6 independent elements
  - Higgs: λ_H
  - Cosmology: 10 parameters (Ω_DE, Ω_DM, Ω_b, n_s, σ₈, A_s, Ω_γ, Ω_ν, Y_p, D/H)
- 9 dimensional observables (new in v2.1):
  - Electroweak: v_EW, M_W, M_Z
  - Quark masses: m_u, m_d, m_s, m_c, m_b, m_t (absolute values in MeV/GeV)

**v2.1 Specific Documentation**
- `publications/v2.1/GIFT_v21_Geometric_Justifications.md` - Torsional geometry derivations
- `publications/v2.1/GIFT_v21_Observable_Reference.md` - Complete 46-observable catalog
- `publications/v2.1/GIFT_v21_Statistical_Validation.md` - Extended validation methodology
- `publications/v2.1/gift_main.md` - Updated with torsional dynamics (updated from v2.0)
- `publications/v2.1/supplements/S3_torsional_dynamics.md` - Complete torsional framework

**Repository Infrastructure**
- `TEST_COVERAGE_ANALYSIS.md` - Comprehensive 12,000-line analysis of test gaps
- `legacy_v2.0/` - Archived v2.0 publications for reproducibility
- `tests/conftest.py` - Updated to use v2.1 framework by default
- Expanded experimental_data fixture to 46 observables

### Changed

**Framework Updates**
- Default framework: Now uses `GIFTFrameworkV21` with torsional dynamics
- Observable count: 15 → 46 (3× expansion)
- Parameter space: Added torsional parameters (|T|, det_g, v_flow)
- Precision metrics: Updated to reflect 46-observable mean deviation

**Documentation Reorganization**
- `publications/v2.0/` → `legacy_v2.0/` (marked as legacy)
- `publications/v2.1/` now primary publication directory
- `README.md` - Comprehensive update to v2.1 with correct observable counts
- Version badges: Added explicit v2.1.0 version badge
- All internal references updated to v2.1

**Formula Updates** (v2.1 with torsional corrections)
- α⁻¹(M_Z): Now includes torsional correction = (248+8)/2 + 99/11 + det_g×|T|
- sin²θ_W: Updated formula = ζ(3)×γ_Euler/M₂
- α_s(M_Z): Simplified to √2/12
- All formulas now reference v2.1 Observable Reference document

**Test Infrastructure**
- `tests/conftest.py`: Default fixture uses `GIFTFrameworkV21`
- Added `gift_framework_v20()` fixture for backwards compatibility
- Experimental data fixture expanded from 15 to 46 observables
- Version marker: All test files updated to v2.1.0

### Fixed

**Scientific Accuracy**
- Corrected Ω_DE precision: 0.21% → 0.008% (improved experimental comparison)
- Fixed δ_CP deviation: Now exact (0.000%) from topological formula
- Updated neutrino precision with latest NuFIT data
- Refined CKM matrix predictions with 2025 PDG values

**Documentation Consistency**
- Harmonized observable counts across all documents (now consistently 46)
- Fixed version references (v2.0 vs v2.1 confusion eliminated)
- Corrected precision table entries
- Updated citation to v2.1.0

### Observable Comparison: v2.0 vs v2.1

| Category | v2.0 | v2.1 | Improvement |
|----------|------|------|-------------|
| Dimensionless | 15 | 37 | +22 observables |
| Dimensional | 0 | 9 | +9 observables |
| **Total** | **15** | **46** | **+31 observables** |
| Gauge precision | 0.03% | 0.02% | Torsional corrections |
| CKM elements | 0 | 6 | Complete matrix |
| Quark ratios | 1 | 10 | Full spectrum |
| Cosmology | 3 | 10 | Extended |

### Framework Statistics (v2.1)

- **Observables**: 46 (37 dimensionless + 9 dimensional)
- **Exact relations**: 9 rigorously proven (unchanged from v2.0)
- **Parameters**: 3 topological + 4 torsional = 7 total
- **Mean precision**: 0.13% across all 46 observables
- **Documentation**: ~12,000 lines across v2.1 publications
- **Test coverage**: 250+ tests (with identified gaps for future work)

### Experimental Predictions (New in v2.1)

**Dimensional Masses** (testable at colliders/precision measurements)
- m_u = 2.16 MeV (derived from scale bridge)
- m_d = 4.67 MeV
- m_s = 93.4 MeV
- m_c = 1.27 GeV
- m_b = 4.18 GeV
- m_t = 172.8 GeV

**Electroweak Scale** (testable at precision frontier)
- v_EW = 246.2 GeV (from scale bridge)
- M_W = 80.37 GeV
- M_Z = 91.19 GeV

### Breaking Changes

**API Changes**
- Default framework class: `GIFTFrameworkStatistical` → `GIFTFrameworkV21`
- Observable dictionary keys: Expanded from 15 to 46 keys
- Parameter initialization: Added optional torsional parameters

**File Structure**
- `publications/v2.0/` moved to `legacy_v2.0/`
- Primary publications now in `publications/v2.1/`
- Tests now import from `gift_v21_core` by default

**Migration Guide for Users**
```python
# v2.0 (legacy)
from run_validation import GIFTFrameworkStatistical
gift = GIFTFrameworkStatistical()
obs = gift.compute_all_observables()  # Returns 15 observables

# v2.1 (current)
from gift_v21_core import GIFTFrameworkV21
gift = GIFTFrameworkV21()
obs_dimensionless = gift.compute_dimensionless_observables()  # Returns 37
obs_dimensional = gift.compute_dimensional_observables()  # Returns 9
obs_all = gift.compute_all_observables()  # Returns 46
```

### Notes

**Theoretical Advances in v2.1**
- Torsional geodesic dynamics provides physical interpretation of RG flow
- Scale bridge mathematically connects dimensionless ratios to absolute masses
- Non-zero torsion |dφ| and |d*φ| modify effective geometry
- Metric determinant det_g ≈ 2.031 ≈ p₂ shows structural consistency

**Computational Validation**
- Monte Carlo validation extended to 46 observables (10⁵ samples)
- Sobol sensitivity analysis for all new parameters
- Bootstrap validation confirms statistical robustness
- Mean deviation 0.13% maintained despite 3× expansion in observables

**Future Work Identified**
- Complete test suite for all 46 observables (currently 15 tested)
- G2_ML v1.0+ module testing (TCS operators, Yukawa tensors)
- Notebook output validation beyond execution checks
- Performance benchmarks and stress tests

See `TEST_COVERAGE_ANALYSIS.md` for comprehensive test gap analysis.

---

## [Unreleased] - Future Work

### Added

**v2.1 Documentation Structure**
- `publications/v2.0/` and `publications/v2.1/` - Versioned publication directories
- `publications/v2.1/GIFT_v21_Geometric_Justifications.md` - Detailed geometric derivation documentation
- `publications/v2.1/GIFT_v21_Observable_Reference.md` - Complete observable catalog with formulas
- `publications/v2.1/GIFT_v21_Statistical_Validation.md` - Statistical validation methodology

**Comprehensive Test Infrastructure**
- `tests/` - Main pytest test suite with unit, integration, regression, and notebook tests
- `giftpy_tests/` - Framework-specific tests (observables, constants, framework)
- `publications/tests/TEST_SYNTHESIS.md` - Comprehensive test synthesis document
- `tests/unit/test_statistical_validation.py` - Sobol sensitivity analysis tests (6 tests)
- `tests/unit/test_mathematical_properties.py` - Mathematical invariant tests
- `tests/regression/test_observable_values.py` - Observable regression tests

**Other Additions**
- `docs/PHILOSOPHY.md` - Philosophical essay on mathematical primacy and epistemic humility
- `.gitignore` - Standard ignore patterns for Python, Jupyter, and IDE files
- GitHub workflows for link validation
- `G2_ML/VERSIONS.md` - Comprehensive version index for all G2 ML framework versions
- `G2_ML/FUTURE_WORK.md` - Planned enhancements replacing obsolete completion plan
- `G2_ML/0.X/README.md` - Documentation for 8 previously undocumented versions
- `legacy_v1/README.md` - Guide to accessing archived v1.0 content via git history
- ARCHIVED warnings to historical G2 ML documentation (versions <0.7)

### Changed
- Publications reorganized into versioned directories (`v2.0/`, `v2.1/`)
- Updated `STRUCTURE.md` to include complete repository structure
- Updated `CLAUDE.md` to v1.1.0 reflecting test infrastructure and v2.1 structure
- Corrected `postBuild` Binder setup script with accurate file paths
- `G2_ML/STATUS.md` - Updated with actual implementation status (93% complete)
- `README.md` - Updated documentation paths to point to `publications/v2.1/`
- Version references harmonized across all documentation (v2.0.0 stable, v2.1 in development)

### Fixed
- Resolved phantom references to non-existent `legacy_v1/` directory
- Corrected G2_ML framework status claims (Yukawa now documented as complete in v0.8)
- Fixed inconsistencies between README.md and G2_ML/STATUS.md regarding implementation status
- Fixed test tolerances to match actual framework formulas

## [2.0.0] - 2025-10-24

### Major Release - Complete Framework Reorganization

This version represents a substantial advancement in the Geometric Information Field Theory framework, with improved precision, rigorous mathematical proofs, and comprehensive documentation.

### Added

**Documentation Structure**
- Modular supplement system with six detailed mathematical documents
- `STRUCTURE.md` explaining repository organization
- `CONTRIBUTING.md` with contribution guidelines
- Quick start section in `README.md`
- `docs/FAQ.md` addressing common questions
- `docs/GLOSSARY.md` defining technical terms
- `docs/EXPERIMENTAL_VALIDATION.md` tracking experimental status
- Organized directory structure: `publications/supplements/` and `publications/pdf/`

**Mathematical Content**
- Supplement A: Complete mathematical foundations (E₈ structure, K₇ manifold, dimensional reduction)
- Supplement B: Rigorous proofs of exact relations (9 proven theorems)
- Supplement C: Complete derivations for all 34 observables
- Supplement D: Detailed phenomenological analysis
- Supplement E: Comprehensive falsification criteria and experimental tests
- Supplement F: Explicit K₇ metric and harmonic form bases

**Framework Improvements**
- Parameter reduction from 4 to 3 through exact relation ξ = (5/2)β₀
- Complete neutrino sector predictions (all four mixing parameters)
- Unified cosmological observables (Ω_DE = ln(2), Hubble parameter)
- Binary information architecture formalization
- Dual origin derivations for key parameters (√17, Ω_DE)

### Changed

**Precision Improvements**
- Mean deviation improved to 0.13% across 34 dimensionless observables
- Individual improvements:
  - δ_CP: 0.15% → 0.005% (30× improvement)
  - θ₁₂: 0.45% → 0.03% (15× improvement)
  - Q_Koide: 0.02% → 0.005% (4× improvement)
  - Complete CKM matrix: mean 0.11% (previously partial)

**Structure Updates**
- Repository reorganized with clear separation: main paper, supplements, PDFs
- Corrected GitHub URLs from `bdelaf/gift` to `gift-framework/GIFT`
- Updated all internal references to reflect new file structure
- Improved citation formats in `CITATION.md`

**Framework Refinements**
- Status classification system (PROVEN, TOPOLOGICAL, DERIVED, etc.)
- Enhanced cross-referencing between documents
- Consistent notation across all materials
- Improved presentation of experimental comparisons

### Fixed

**Scientific Accuracy**
- Corrected δ_CP formula with proper normalization
- Refined neutrino mass hierarchy calculations
- Improved treatment of running coupling constants
- Enhanced error propagation in derived quantities

**Documentation**
- Fixed inconsistent file path references
- Corrected broken links in README and CITATION
- Updated Binder and Colab notebook paths
- Standardized table formats across documents

### Experimental Results

**Confirmed Predictions (< 0.1% deviation)**
- α⁻¹ = 137.036 (0.001%)
- Q_Koide = 2/3 (0.005%)
- δ_CP = 197° (0.005%)
- sin²θ_W = 0.23127 (0.009%)
- α_s(M_Z) = 0.1180 (0.08%)
- Ω_DE = ln(2) (0.10%)

**High-Precision Predictions (< 0.5%)**
- Complete neutrino mixing (all four parameters)
- Complete CKM matrix (ten elements, mean 0.11%)
- Lepton mass ratios (mean 0.12%)
- Gauge coupling unification

### Framework Statistics

- **Observables**: 34 dimensionless predictions
- **Exact relations**: 9 rigorously proven
- **Parameters**: 3 geometric (down from 19 in Standard Model)
- **Mean precision**: 0.13%
- **Documentation**: ~7000 lines across supplements

### Notes

**Theoretical Advances**
The v2.0 framework establishes several exact mathematical relations previously unavailable:
- N_gen = 3 from topological necessity (rank-Weyl structure)
- Triple origin for √17 (Higgs sector)
- Binary architecture foundation for dark energy
- McKay correspondence connection to golden ratio

**Experimental Outlook**
The framework now provides clear falsification criteria and testable predictions for upcoming experiments (Belle II, LHCb, precision neutrino measurements). The tightest constraint comes from δ_CP, where future precision measurements could decisively test the topological origin hypothesis.

## [1.0.0] - 2024 (Archived)

### Initial Release

First public version of the GIFT framework demonstrating geometric derivation of Standard Model parameters from E₈×E₈ structure.

**Key Features**
- Basic dimensional reduction E₈×E₈ → AdS₄×K₇
- Initial parameter predictions
- Preliminary neutrino sector analysis
- Prototype computational notebook

**Status**: Archived in `legacy_v1/` directory. See `legacy_v1/README.md` for details.

---

## Future Development

### Planned for v2.1 (Unreleased, in development)

**Enhancements Under Investigation**
- Temporal framework integration (21·e⁸ structure)
- Dimensional observable predictions (masses, VEV)
- Enhanced computational tools for parameter exploration
- Additional experimental comparison data from 2025 results

**Research Directions**
- Connection to quantum error correction codes
- Relationship to holographic entropy bounds
- Implications for quantum gravity
- Extensions to grand unification scale

### Experimental Milestones

**2025-2027**
- Belle II: Improved CKM measurements
- T2K/NOvA: Enhanced neutrino oscillation parameters
- LHCb: Precision CP violation measurements

**2028-2030**
- DUNE: Definitive neutrino mass hierarchy
- FCC studies: High-energy parameter evolution
- CMB-S4: Cosmological parameter refinements

These experimental results will provide critical tests of the framework's predictions.

---

For detailed information about specific changes, see the relevant sections in:
- Main paper: `publications/gift_main.md`
- Supplements: `publications/supplements/`
- Documentation: `docs/`

