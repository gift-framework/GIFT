# Geometric Information Field Theory v2.3

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-2.3-green.svg)](CHANGELOG.md)
[![Lean 4 Verified](https://img.shields.io/badge/Lean_4-Verified-blue)](https://github.com/gift-framework/core)
[![Coq Verified](https://img.shields.io/badge/Coq_8.18-Verified-orange)](https://github.com/gift-framework/core)

A geometric framework deriving Standard Model parameters from topological invariants of E₈×E₈ gauge structure compactified on a G₂-holonomy manifold K₇.

---

## Overview

| Metric | Value |
|--------|-------|
| Precision | 0.128% mean deviation across 39 observables |
| Adjustable parameters | Zero (all structurally determined) |
| Formally verified relations | 39 proven in Lean 4 + Coq (dual verification, zero axioms) |
| Key exact results | sin²θ_W = 3/13, κ_T = 1/61, det(g) = 65/32, τ = 3472/891, δ_CP = 197° |

The dimensional reduction chain: **E₈×E₈ (496D) → AdS₄ × K₇ (11D) → Standard Model (4D)**

Whether this mathematical structure reflects fundamental reality or constitutes an effective description remains open to experimental determination.

---

## Navigation

### "Show me the numbers"

→ [Observable Reference](publications/references/GIFT_v23_Observable_Reference.md) All 39 observables with formulas, values, deviations

→ [39 Observables CSV](publications/references/39_observables.csv) Machine-readable data

### "Show me the proofs"

→ [gift-framework/core](https://github.com/gift-framework/core) Lean 4 + Coq formal verification (39 relations, zero axioms)

→ [S4: Complete Derivations](publications/markdown/S4_complete_derivations_v23.md) Mathematical proofs

### "Show me the framework"

→ [Main Paper](publications/markdown/gift_2_3_main.md) Complete theoretical framework

→ [Supplements S1-S7](publications/markdown/) Mathematical details by topic

### "Show me an introduction"

→ [YouTube (8 min)](https://www.youtube.com/watch?v=6DVck30Q6XM) Video overview

→ [Philosophy](docs/PHILOSOPHY.md) Foundational perspective

---

## Supplements

| Document | Content |
|----------|---------|
| [S1](publications/markdown/S1_mathematical_architecture_v23.md) | E₈ exceptional Lie algebra, root systems, branching rules |
| [S2](publications/markdown/S2_K7_manifold_construction_v23.md) | K₇ twisted connected sum, Betti numbers b₂=21, b₃=77 |
| [S3](publications/markdown/S3_torsional_dynamics_v23.md) | Torsion tensor, geodesic flow, κ_T = 1/61 |
| [S4](publications/markdown/S4_complete_derivations_v23.md) | Complete mathematical proofs for all 39 observables |
| [S5](publications/markdown/S5_experimental_validation_v23.md) | Comparison with PDG 2024, falsification criteria |
| [S6](publications/markdown/S6_theoretical_extensions_v23.md) | Speculative extensions (M-theory, AdS/CFT) |
| [S7](publications/markdown/S7_dimensional_observables_v23.md) | Scale bridge from dimensionless ratios to GeV |

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
| N_gen | 3 | rank(E₈) − Weyl |
| m_s/m_d | 20 | p₂² × Weyl |
| Q_Koide | 2/3 | dim(G₂)/b₂ |

Full list: [Observable Reference](publications/references/GIFT_v23_Observable_Reference.md) | Proofs: [gift-framework/core](https://github.com/gift-framework/core)

---

## Falsification Tests

| Prediction | Experiment | Timeline | Falsification Criterion |
|------------|------------|----------|------------------------|
| δ_CP = 197° | DUNE | 2027-2030 | Measured value outside [187°, 207°] |
| sin²θ_W = 3/13 | FCC-ee | 2040s | Precision measurement outside [0.2295, 0.2320] |
| m_s/m_d = 20 | Lattice QCD | 2030 | Converges outside [19, 21] |
| N_gen = 3 | LHC | Ongoing | Fourth generation discovery |

Details: [S5: Experimental Validation](publications/markdown/S5_experimental_validation_v23.md)

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
@software{gift_framework_v23,
  title   = {GIFT Framework v2.3: Geometric Information Field Theory},
  author  = {de La Fournière, Brieuc},
  year    = {2025},
  url     = {https://github.com/gift-framework/GIFT},
  version = {2.3.0}
}
```

See [CITATION.md](CITATION.md) for additional formats.

---

## License

MIT License. See [LICENSE](LICENSE)

---

## Related Repositories

| Repository | Content |
|------------|---------|
| [gift-framework/core](https://github.com/gift-framework/core) | Formal verification (Lean 4 + Coq), K₇ metric pipeline |

---

> *Gift from bit*

---
