# Geometric Information Field Theory v3.0

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-3.0.1-green.svg)](CHANGELOG.md)
[![Lean 4 Verified](https://img.shields.io/badge/Lean_4-Verified-blue)](https://github.com/gift-framework/core)
[![Relations](https://img.shields.io/badge/Relations-165+-orange)](https://github.com/gift-framework/core)
[![Precision](https://img.shields.io/badge/Mean_Deviation-0.087%25-brightgreen)](publications/markdown/GIFT_v3_S2_derivations.md)

A geometric framework deriving Standard Model parameters from topological invariants of E₈×E₈ gauge structure compactified on a G₂-holonomy manifold K₇.

---

## Overview

| Metric | Value |
|--------|-------|
| Precision | **0.087% mean deviation** across 18 dimensionless predictions |
| Adjustable parameters | Zero (all structurally determined) |
| Formally verified relations | **165+ proven in Lean 4 + Coq** (zero axioms) |
| Key exact results | sin²θ_W = 3/13, τ = 3472/891, det(g) = 65/32, Monster = 47×59×71 |

The dimensional reduction chain: **E₈×E₈ (496D) → AdS₄ × K₇ (11D) → Standard Model (4D)**

---

## Navigation

### "Show me the numbers"

→ [18 Dimensionless Predictions](publications/markdown/GIFT_v3_S2_derivations.md) Complete derivations with 0.087% mean deviation

### "Show me the proofs"

→ [gift-framework/core](https://github.com/gift-framework/core) Lean 4 + Coq formal verification (165+ relations, zero axioms)

→ [S2: Complete Derivations](publications/markdown/GIFT_v3_S2_derivations.md) All 18 dimensionless derivations

### "Show me the framework"

→ [Main Paper](publications/markdown/GIFT_v3_main.md) Complete theoretical framework (v3.0)

→ [S1: Foundations](publications/markdown/GIFT_v3_S1_foundations.md) E₈, G₂, K₇ mathematical foundations

### "Show me an introduction"

→ [YouTube (8 min)](https://www.youtube.com/watch?v=6DVck30Q6XM) Video overview

→ [Philosophy](docs/PHILOSOPHY.md) Foundational perspective

### Domain-specific guides

**"I'm a geometer"**
→ [GiftPy for Geometers](docs/GIFTPY_FOR_GEOMETERS.md) - G₂ metric construction pipeline

**"I'm a physicist"**
→ [Information Geometry for Physicists](docs/INFO_GEO_FOR_PHYSICISTS.md) - Topological approach to SM parameters

**"I'm interested in formalization"**
→ [Lean for Physics](docs/LEAN_FOR_PHYSICS.md) - Machine-verified physical relations

**"I want dimensional observables"**
→ [docs/technical/](docs/technical/) - Speculative extensions for absolute masses, scale bridge, and torsional dynamics

---

## Interactive Visualizations

[View GIFT Lean Blueprint](https://gift-framework.github.io/GIFT/docs/figures/gift_blueprint.html)
---
[View K7 Manifold Lean Blueprint](https://gift-framework.github.io/GIFT/docs/figures/k7_blueprint.html)
---
[created with Lean Blueprint Copilot](https://github.com/augustepoiroux/LeanBlueprintCopilot)

---

## Documentation

### Core Documents

| Document | Content |
|----------|---------|
| [Main Paper](publications/markdown/GIFT_v3_main.md) | Complete theoretical framework |
| [S1: Foundations](publications/markdown/GIFT_v3_S1_foundations.md) | E₈, G₂, K₇ mathematical construction |
| [S2: Derivations](publications/markdown/GIFT_v3_S2_derivations.md) | All 18 dimensionless derivations |

### Exploratory References

| Document | Content |
|----------|---------|
| [Yukawa & Mixing](publications/references/yukawa_mixing.md) | CKM/PMNS matrices, Yukawa couplings |
| [Sequences & Primes](publications/references/sequences_prime_atlas.md) | Fibonacci embedding, Prime Atlas |
| [Monster & Moonshine](publications/references/monster_moonshine.md) | Monster group, j-invariant |

### Speculative Extensions (docs/technical/)

| Document | Content |
|----------|---------|
| [S7: Dimensional Observables](docs/technical/S7_dimensional_observables_v30.md) | Absolute masses, scale bridge (speculative) |
| [S6: Theoretical Extensions](docs/technical/S6_theoretical_extensions_v30.md) | M-theory, quantum gravity connections |
| [S3: Torsional Dynamics](docs/technical/S3_torsional_dynamics_v30.md) | RG flow, non-zero torsion |
| [GIFT Atlas](docs/technical/atlas/) | Complete constant/relation database |

---

## Related Repositories

| Repository | Content |
|------------|---------|
| [gift-framework/core](https://github.com/gift-framework/core) | Formal verification (Lean 4 + Coq), K₇ metric pipeline |

---

## Key Results

### Precision by Sector (18 Dimensionless Predictions)

| Sector | Predictions | Mean Deviation | Highlight |
|--------|-------------|----------------|-----------|
| Electroweak | 3 | 0.12% | sin²θ_W = 3/13 |
| Lepton | 3 | 0.04% | Q_Koide = 2/3 (0.0009%) |
| Quark | 1 | 0.00% | m_s/m_d = 20 (exact) |
| Neutrino | 4 | 0.15% | δ_CP = 197° (exact) |
| Cosmology | 3 | 0.07% | n_s = ζ(11)/ζ(5) (0.004%) |
| Structural | 4 | exact | N_gen = 3, τ = 3472/891 |

### Selected Exact Relations (Lean 4 + Coq Verified)

| Relation | Value | Topological Formula |
|----------|-------|---------------------|
| sin²θ_W | 3/13 | b₂/(b₃ + dim(G₂)) |
| κ_T | 1/61 | 1/(b₃ − dim(G₂) − p₂) |
| τ | 3472/891 | (496 × 21)/(27 × 99) |
| det(g) | 65/32 | Metric determinant from G₂ structure |
| δ_CP | 197° | 7 × dim(G₂) + H* |
| m_s/m_d | 20 | p₂² × Weyl |
| Q_Koide | 2/3 | dim(G₂)/b₂ |
| **Monster** | **196883** | **L₈ × (b₃-18) × (b₃-6) = 47×59×71** |

### v3.0 Number-Theoretic Relations

| Structure | Result | Status |
|-----------|--------|--------|
| Fibonacci F₃–F₁₂ | = GIFT constants (p₂, N_gen, Weyl, rank, α_B, b₂, hidden, ...) | **PROVEN** |
| Prime Atlas | 100% coverage primes < 200 | **PROVEN** |
| Monster dim | 196883 = 47×59×71, arithmetic progression d=12 | **PROVEN** |
| j-invariant | 744 = 3 × 248 = N_gen × dim(E₈) | **PROVEN** |
| McKay | Coxeter(E₈) = 30 = icosahedron edges | **PROVEN** |

Full derivations: [S2: Derivations](publications/markdown/GIFT_v3_S2_derivations.md) | Proofs: [gift-framework/core](https://github.com/gift-framework/core)

---

## Falsification Tests

| Prediction | Experiment | Timeline | Falsification Criterion |
|------------|------------|----------|------------------------|
| δ_CP = 197° | DUNE | 2027-2030 | Measured value outside [187°, 207°] |
| sin²θ_W = 3/13 | FCC-ee | 2040s | Precision measurement outside [0.2295, 0.2320] |
| m_s/m_d = 20 | Lattice QCD | 2030 | Converges outside [19, 21] |
| N_gen = 3 | LHC | Ongoing | Fourth generation discovery |

Details: [S2: Derivations](publications/markdown/GIFT_v3_S2_derivations.md) (Section 10: Falsification)

---

## Limitations

### What "Zero-Parameter" Means (and Doesn't Mean)

The framework contains **no continuous adjustable parameters** fitted to data. However, it does make **discrete structural choices**:
- Selecting E₈×E₈ as gauge group
- Selecting the specific K₇ manifold (b₂=21, b₃=77)
- Selecting the TCS building blocks

These choices are mathematically motivated but constitute model selection. The framework predicts 39 observables *given* these choices (with 165+ certified relations); it does not explain *why* nature chose this particular geometry.

### Why Not Numerology?

The v3.0 structures-Fibonacci sequences, Monster group, McKay correspondence-possess **independent mathematical existence**:
- Fibonacci appears in phyllotaxis, shell spirals, golden rectangles
- The Monster group is a theorem of group theory (Griess 1982)
- McKay correspondence is established mathematics (McKay 1980)

Their appearance in GIFT suggests structural rather than coincidental relationships.

---

## Connect

| Platform | Link |
|----------|------|
| YouTube | [@giftheory](https://youtube.com/@giftheory) |
| Substack | [giftheory.substack.com](https://substack.com/@giftheory) |
| X/Twitter | [@GIFTheory](https://x.com/GIFTheory) |

| Archive | Link |
|---------|------|
| Zenodo | [10.5281/zenodo.17751250](https://doi.org/10.5281/zenodo.17751250) |
| ResearchGate | [Author page](https://www.researchgate.net/profile/Brieuc-De-La-Fourniere) |

---

## Citation

```bibtex
@software{gift_framework_v30,
  title   = {GIFT Framework v3.0: Geometric Information Field Theory},
  author  = {de La Fournière, Brieuc},
  year    = {2025},
  url     = {https://github.com/gift-framework/GIFT},
  version = {3.0.0}
}
```

See [CITATION.md](CITATION.md) for additional formats.

---

## License

MIT License. See [LICENSE](LICENSE)

---

> *Gift from bit*

---
