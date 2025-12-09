# Geometric Information Field Theory v3.0

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-3.0-green.svg)](CHANGELOG.md)
[![Lean 4 Verified](https://img.shields.io/badge/Lean_4-Verified-blue)](https://github.com/gift-framework/core)
[![Relations](https://img.shields.io/badge/Relations-165+-orange)](https://github.com/gift-framework/core)

A geometric framework deriving Standard Model parameters from topological invariants of E₈×E₈ gauge structure compactified on a G₂-holonomy manifold K₇.

---

## Overview

| Metric | Value |
|--------|-------|
| Precision | 0.197% mean deviation across 39 observables |
| Adjustable parameters | Zero (all structurally determined) |
| Formally verified relations | **165+ proven in Lean 4 + Coq** (zero axioms) |
| Key exact results | sin²θ_W = 3/13, τ = 3472/891, det(g) = 65/32, Monster = 47×59×71 |

### What's New in v3.0

| Discovery | Description |
|-----------|-------------|
| **Fibonacci Embedding** | F₃–F₁₂ map exactly to framework constants (p₂→N_gen→Weyl→rank→α_B→b₂) |
| **Prime Atlas** | 100% coverage of primes < 200 via 3 generators (b₃, H*, dim(E₈)) |
| **Monster Group** | 196883 = 47×59×71, all factors GIFT-expressible |
| **McKay Correspondence** | E₈ ↔ Binary Icosahedral → golden ratio emergence |
| **Exceptional Chain** | E₆=6×13, E₇=7×19, E₈=8×31 |

The dimensional reduction chain: **E₈×E₈ (496D) → AdS₄ × K₇ (11D) → Standard Model (4D)**

---

## Navigation

### "Show me the numbers"

→ [Observable Reference](publications/references/GIFT_v30_Observable_Reference.md) All 39 observables with formulas, values, deviations

→ [39 Observables CSV](publications/references/39_observables.csv) Machine-readable data

### "Show me the proofs"

→ [gift-framework/core](https://github.com/gift-framework/core) Lean 4 + Coq formal verification (165+ relations, zero axioms)

→ [S4: Complete Derivations](publications/markdown/S4_complete_derivations_v30.md) Mathematical proofs

### "Show me the framework"

→ [Main Paper](publications/markdown/gift_3_0_main.md) Complete theoretical framework

→ [Supplements S1-S9](publications/markdown/) Mathematical details by topic

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

---

## Interactive Visualizations

[View GIFT Lean Blueprint](https://gift-framework.github.io/GIFT/docs/figures/gift_blueprint.html)
---
[View K7 Manifold Lean Blueprint](https://gift-framework.github.io/GIFT/docs/figures/k7_blueprint.html)
---
[created with Lean Blueprint Copilot](https://github.com/augustepoiroux/LeanBlueprintCopilot)

---

## Supplements

| Document | Content |
|----------|---------|
| [S1](publications/markdown/S1_mathematical_architecture_v30.md) | E₈ exceptional Lie algebra, Exceptional Chain, McKay correspondence |
| [S2](publications/markdown/S2_K7_manifold_construction_v30.md) | K₇ twisted connected sum, Betti numbers b₂=21, b₃=77 |
| [S3](publications/markdown/S3_torsional_dynamics_v30.md) | Torsion tensor, geodesic flow, κ_T = 1/61 |
| [S4](publications/markdown/S4_complete_derivations_v30.md) | Complete mathematical proofs, 165+ certified relations |
| [S5](publications/markdown/S5_experimental_validation_v30.md) | Comparison with PDG 2024, falsification criteria |
| [S6](publications/markdown/S6_theoretical_extensions_v30.md) | Speculative extensions (M-theory, AdS/CFT) |
| [S7](publications/markdown/S7_dimensional_observables_v30.md) | Scale bridge from dimensionless ratios to GeV |
| **[S8](publications/markdown/S8_sequences_prime_atlas_v30.md)** | **NEW: Fibonacci/Lucas embedding, Prime Atlas** |
| **[S9](publications/markdown/S9_monster_moonshine_v30.md)** | **NEW: Monster group, Monstrous Moonshine** |

---

## Related Repositories

| Repository | Content |
|------------|---------|
| [gift-framework/core](https://github.com/gift-framework/core) | Formal verification (Lean 4 + Coq), K₇ metric pipeline |

---

## Key Results

### Precision by Sector

| Sector | Observables | Mean Deviation | Highlight |
|--------|-------------|----------------|-----------|
| Gauge | 3 | 0.06% | α_s = √2/12 |
| Lepton | 4 | 0.04% | Q_Koide = 2/3 (exact) |
| CKM | 6 | 0.08% | |
| Neutrino | 4 | 0.13% | δ_CP = 197° (exact) |
| Quark ratios | 10 | 0.18% | m_s/m_d = 20 (exact) |
| Cosmology | 2 | 0.11% | Ω_DE = ln(2) × 98/99 |

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

Full list: [Observable Reference](publications/references/GIFT_v30_Observable_Reference.md) | Proofs: [gift-framework/core](https://github.com/gift-framework/core)

---

## Falsification Tests

| Prediction | Experiment | Timeline | Falsification Criterion |
|------------|------------|----------|------------------------|
| δ_CP = 197° | DUNE | 2027-2030 | Measured value outside [187°, 207°] |
| sin²θ_W = 3/13 | FCC-ee | 2040s | Precision measurement outside [0.2295, 0.2320] |
| m_s/m_d = 20 | Lattice QCD | 2030 | Converges outside [19, 21] |
| N_gen = 3 | LHC | Ongoing | Fourth generation discovery |

Details: [S5: Experimental Validation](publications/markdown/S5_experimental_validation_v30.md)

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
