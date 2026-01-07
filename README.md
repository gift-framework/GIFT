# Geometric Information Field Theory v3.2

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-3.2.0-green.svg)](CHANGELOG.md)
[![Lean 4 + Coq](https://img.shields.io/badge/Formally_Verified-Lean_4_+_Coq-blue)](https://github.com/gift-framework/core)

**Standard Model parameters from pure geometry** — E₈×E₈ on G₂-holonomy manifold K₇, zero adjustable parameters.

---

## At a Glance

| | |
|---|---|
| **Precision** | 0.24% mean deviation across 18 dimensionless predictions (PDG 2024) |
| **Uniqueness** | #1 out of 54,327 configurations tested (>4σ significance) |
| **Parameters** | Zero adjustable (all structurally determined) |
| **Verified** | 185 relations proven in Lean 4 + Coq (40 axioms, core v3.2.0) |
| **Exact results** | sin²θ_W = 3/13 · τ = 3472/891 · det(g) = 65/32 · Monster = 47×59×71 |

**Dimensional reduction:** E₈×E₈ (496D) → AdS₄ × K₇ (11D) → Standard Model (4D)

---

## Quick Start

| Paper | Proofs | Video |
|:-----:|:------:|:-----:|
| [Main Paper](publications/markdown/GIFT_v3.2_main.md) | [Lean 4 + Coq](https://github.com/gift-framework/core) | [YouTube (8 min)](https://www.youtube.com/watch?v=6DVck30Q6XM) |

---

## Documentation

### Core

| Document | Description |
|----------|-------------|
| [Main Paper](publications/markdown/GIFT_v3.2_main.md) | Complete theoretical framework |
| [S1: Foundations](publications/markdown/GIFT_v3.2_S1_foundations.md) | E₈, G₂, K₇ mathematical construction |
| [S2: Derivations](publications/markdown/GIFT_v3.2_S2_derivations.md) | All 18 dimensionless derivations (0.24% mean, PDG 2024) |
| [S3: Dynamics](publications/markdown/GIFT_v3.2_S3_dynamics.md) | RG flow, torsional dynamics |

### For Specific Audiences

| Background | Start Here |
|------------|------------|
| Geometer | [GiftPy for Geometers](docs/GIFTPY_FOR_GEOMETERS.md) — G₂ metric construction pipeline |
| Physicist | [Info Geo for Physicists](docs/INFO_GEO_FOR_PHYSICISTS.md) — Topological approach to SM parameters |
| Formalization | [Lean for Physics](docs/LEAN_FOR_PHYSICS.md) — Machine-verified physical relations |
| Philosophy | [Philosophy](docs/PHILOSOPHY.md) — Foundational perspective |

### Exploratory Extensions

| Document | Description |
|----------|-------------|
| [Speculative Physics](publications/references/SPECULATIVE_PHYSICS.md) | Scale bridge, Yukawa, M-theory, quantum gravity |
| [Number-Theoretic Structures](publications/references/NUMBER_THEORETIC_STRUCTURES.md) | Fibonacci, Prime Atlas, Monster, Moonshine |

---

## Key Results

### Precision by Sector

| Sector | Predictions | Mean Deviation | Highlight |
|--------|:-----------:|:--------------:|-----------|
| Electroweak | 3 | 0.12% | sin²θ_W = 3/13 |
| Lepton | 3 | 0.04% | Q_Koide = 2/3 (0.0009%) |
| Quark | 1 | 0.00% | m_s/m_d = 20 (exact) |
| Neutrino | 4 | 0.15% | δ_CP = 197° |
| Cosmology | 3 | 0.07% | n_s = ζ(11)/ζ(5) (0.004%) |
| Structural | 4 | exact | N_gen = 3, τ = 3472/891 |

### Exact Relations (Lean 4 + Coq Verified)

| Relation | Value | Topological Origin |
|----------|:-----:|-------------------|
| sin²θ_W | 3/13 | b₂/(b₃ + dim(G₂)) |
| κ_T | 1/61 | 1/(b₃ − dim(G₂) − p₂) |
| τ | 3472/891 | (496 × 21)/(27 × 99) |
| det(g) | 65/32 | Metric determinant from G₂ structure |
| δ_CP | 197° | 7 × dim(G₂) + H* |
| m_s/m_d | 20 | p₂² × Weyl |
| Q_Koide | 2/3 | dim(G₂)/b₂ |
| **Monster** | **196883** | **L₈ × (b₃−18) × (b₃−6) = 47×59×71** |

### Analytical G₂ Metric

The G₂ structure admits an **exact closed form**:

| Property | Value | Status |
|----------|-------|:------:|
| Associative 3-form | φ = (65/32)^{1/14} × φ₀ | EXACT |
| Metric | g = (65/32)^{1/7} × I₇ | EXACT |
| Torsion | T = 0 (constant form) | EXACT |
| det(g) | 65/32 | EXACT |

Joyce's existence theorem is **trivially satisfied** — no numerical fitting required.

### Number-Theoretic Structures

| Structure | Result | Status |
|-----------|--------|:------:|
| Fibonacci F₃–F₁₂ | Maps to GIFT constants (p₂, N_gen, Weyl, rank, α_B, b₂, ...) | ✓ Proven |
| Prime Atlas | 100% coverage of primes < 200 | ✓ Proven |
| Monster dimension | 196883 = 47×59×71, arithmetic progression d=12 | ✓ Proven |
| j-invariant | 744 = 3 × 248 = N_gen × dim(E₈) | ✓ Proven |
| McKay correspondence | Coxeter(E₈) = 30 = icosahedron edges | ✓ Proven |

---

## Statistical Uniqueness

Comprehensive validation confirms that (b₂=21, b₃=77) is not merely a good choice but the **unique optimum** among G₂ manifold configurations.

### Exhaustive Search Results (v3.2)

| Metric | Value |
|--------|-------|
| Configurations tested | 54,327 |
| **GIFT rank** | **#1** |
| GIFT mean deviation | 0.24% (PDG 2024) |
| Better alternatives found | 0 |
| p-value | < 10⁻⁵ |

### Top 5 Configurations

| Rank | b₂ | b₃ | Deviation |
|:----:|:--:|:--:|:---------:|
| **1** | **21** | **77** | **0.23%** |
| 2 | 21 | 76 | 0.50% |
| 3 | 21 | 78 | 0.50% |
| 4 | 21 | 79 | 0.79% |
| 5 | 21 | 75 | 0.81% |

### Statistical Significance (v3.2 Monte Carlo)

- **LEE-corrected significance**: >4σ
- **p-value**: < 10⁻⁵
- **Total configurations tested**: 54,327 (Betti, holonomy, structural variations)
- **Better alternatives found**: 0

GIFT occupies a **sharp minimum**: no tested alternative matches its precision.

Details: [Uniqueness Test Report](statistical_validation/UNIQUENESS_TEST_REPORT.md)

---

## Falsification Tests

| Prediction | Experiment | Timeline | Falsification Criterion |
|------------|------------|----------|------------------------|
| δ_CP = 197° | DUNE | 2027-2030 | Outside [187°, 207°] |
| sin²θ_W = 3/13 | FCC-ee | 2040s | Outside [0.2295, 0.2320] |
| m_s/m_d = 20 | Lattice QCD | 2030 | Converges outside [19, 21] |
| N_gen = 3 | LHC | Ongoing | Fourth generation discovery |

Details: [S2 Section 10](publications/markdown/GIFT_v3.2_S2_derivations.md)

---

## Limitations

### What "Zero-Parameter" Means

The framework contains **no continuous adjustable parameters** fitted to data. However, it makes **discrete structural choices**:
- E₈×E₈ as gauge group
- K₇ manifold with (b₂=21, b₃=77)
- TCS building blocks

These are mathematically motivated but constitute model selection. The framework predicts observables *given* these choices — it does not explain *why* nature chose this geometry.

**However**: Statistical validation shows (b₂=21, b₃=77) is the unique optimum among 19,100 tested configurations. This doesn't explain the choice, but establishes it is not arbitrary.

### Why Not Numerology?

The v3.0 structures (Fibonacci, Monster, McKay) have **independent mathematical existence**:
- Fibonacci appears in phyllotaxis, shell spirals, golden ratio
- Monster group is a theorem (Griess 1982)
- McKay correspondence is established mathematics (McKay 1980)

Their appearance suggests structural rather than coincidental relationships.

---

## Interactive Visualizations

| Blueprint | Description |
|-----------|-------------|
| [GIFT Lean Blueprint](https://gift-framework.github.io/GIFT/docs/figures/gift_blueprint.html) | Dependency graph |
| [K7 Manifold Blueprint](https://gift-framework.github.io/GIFT/docs/figures/k7_blueprint.html) | K₇ construction |

*Created with [Lean Blueprint Copilot](https://github.com/augustepoiroux/LeanBlueprintCopilot)*

---

## Related Repositories

| Repository | Description |
|------------|-------------|
| [gift-framework/core](https://github.com/gift-framework/core) | Formal verification (Lean 4 + Coq), K₇ metric pipeline |

---

## Connect

| Platform | |
|----------|---|
| YouTube | [@giftheory](https://youtube.com/@giftheory) |
| Substack | [giftheory.substack.com](https://giftheory.substack.com/) |
| X | [@GIFTheory](https://x.com/GIFTheory) |

| Archive | |
|---------|---|
| Zenodo | [10.5281/zenodo.17979433](https://doi.org/10.5281/zenodo.17979433) |
| ResearchGate | [Author page](https://www.researchgate.net/profile/Brieuc-De-La-Fourniere) |

---

## Citation

```bibtex
@software{gift_framework_v32,
  title   = {GIFT Framework v3.2: Geometric Information Field Theory},
  author  = {de La Fournière, Brieuc},
  year    = {2025},
  url     = {https://github.com/gift-framework/GIFT},
  version = {3.2.0}
}
```

See [CITATION.md](CITATION.md) for additional formats.

---

## License

MIT License — see [LICENSE](LICENSE)

---

> *Gift from bit*

---
