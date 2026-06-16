---
title: "Citation Guide"
layout: default
---


Citation formats for the GIFT Framework v3.4.

## Software Citation (Recommended)

### BibTeX

```bibtex
@software{gift_framework_v34,
  title   = {GIFT Framework v3.4: Geometric Information Field Theory},
  author  = {de La Fournière, Brieuc},
  year    = {2026},
  url     = {https://github.com/gift-framework/GIFT},
  version = {3.4.26},
  license = {MIT},
  note    = {95 observables, 0.99\% mean deviation on 33 Type I relations (PDG 2024 / NuFIT 6.1), 140 conjuncts certified in Lean 4, 15 axioms (4 main-chain + 11 interval-arithmetic)}
}
```

### APA Style

```
de La Fournière, B. (2026). GIFT Framework v3.4: Geometric Information Field Theory (Version 3.4.26) [Software]. GitHub. https://github.com/gift-framework/GIFT
```

### Chicago Style

```
de La Fournière, Brieuc. "GIFT Framework v3.4: Geometric Information Field Theory." Version 3.4.26. GitHub, 2026. https://github.com/gift-framework/GIFT.
```

---

## Theoretical Framework Citation

### BibTeX

```bibtex
@article{gift_theory_v34,
  title  = {Geometric Information Field Theory v3.4: Topological Determination of Standard Model Parameters},
  author = {de La Fournière, Brieuc},
  year   = {2026},
  note   = {Mean deviation 0.99\% on 33 Type I relations (PDG 2024 / NuFIT 6.1), zero continuous adjustable parameters, 3 integer primitives (N=3, r₈=8, r₂=2), 140 conjuncts in Lean 4},
  url    = {https://github.com/gift-framework/GIFT}
}
```

---

## Framework v3.4 Papers (Zenodo [10.5281/zenodo.20070101](https://doi.org/10.5281/zenodo.20070101))

### Main Paper

```bibtex
@misc{gift_main_v34,
  title        = {Geometric Information Field Theory: Standard Model Parameters as Topological Invariants of a G\textsubscript{2} Holonomy Manifold},
  author       = {de La Fournière, Brieuc},
  year         = {2026},
  month        = {May},
  howpublished = {GIFT Framework v3.4, 44 pp.},
  doi          = {10.5281/zenodo.20070101},
  url          = {https://doi.org/10.5281/zenodo.20070101}
}
```

### S1: Foundations

```bibtex
@misc{gift_s1_foundations_v34,
  title        = {GIFT S1: Mathematical Foundations — E₈, G₂, K₇},
  author       = {de La Fournière, Brieuc},
  year         = {2026},
  howpublished = {GIFT Framework v3.4, Supplement S1},
  url          = {https://github.com/gift-framework/GIFT/blob/main/publications/papers/markdown/GIFT_v3.4_S1_foundations.md}
}
```

### S2: Derivations

```bibtex
@misc{gift_s2_derivations_v34,
  title        = {GIFT S2: Complete Derivations — 33 Type I Relations},
  author       = {de La Fournière, Brieuc},
  year         = {2026},
  howpublished = {GIFT Framework v3.4, Supplement S2},
  url          = {https://github.com/gift-framework/GIFT/blob/main/publications/papers/markdown/GIFT_v3.4_S2_derivations.md}
}
```

### S3: Observables

```bibtex
@misc{gift_s3_observables_v34,
  title        = {GIFT S3: Observable Catalog — 95 Predictions},
  author       = {de La Fournière, Brieuc},
  year         = {2026},
  howpublished = {GIFT Framework v3.4, Supplement S3},
  url          = {https://github.com/gift-framework/GIFT/blob/main/publications/papers/markdown/GIFT_v3.4_S3_observables.md}
}
```

---

## Triptyque (peer-reviewable companion papers, Zenodo)

### Paper A: Certified G₂ Structure

```bibtex
@misc{gift_paper_a_certified_neck,
  title        = {Certified Torsion-Free G₂ Structure on a TCS Neck Model via Computer-Assisted Proof},
  author       = {de La Fournière, Brieuc},
  year         = {2026},
  doi          = {10.5281/zenodo.19892350},
  url          = {https://doi.org/10.5281/zenodo.19892350},
  note         = {First computer-assisted existence proof for a metric with special holonomy; h ≤ 8.95×10⁻⁹, ×56 million margin below Joyce ε₀}
}
```

### Paper B: Spectral Geometry

```bibtex
@misc{gift_paper_b_spectral,
  title        = {Spectral Geometry of an Explicit G₂ Metric: Laplacian Spectrum and Harmonic Forms},
  author       = {de La Fournière, Brieuc},
  year         = {2026},
  doi          = {10.5281/zenodo.19893371},
  url          = {https://doi.org/10.5281/zenodo.19893371},
  note         = {Laplacian spectrum, harmonic forms, λ₁ ≈ 0.12461 (Richardson) ≈ 6π²/475}
}
```

### Paper C: Newton-Kantorovich on K3

```bibtex
@misc{gift_paper_c_k3_nk,
  title        = {Newton--Kantorovich Diagnostics on a Donaldson K3 Metric},
  author       = {de La Fournière, Brieuc},
  year         = {2026},
  doi          = {10.5281/zenodo.19708916},
  url          = {https://doi.org/10.5281/zenodo.19708916},
  note         = {NK certificate on CI(2,2,2) K3 with Donaldson-balanced metric, β_Lap k=4 margin ×6.4}
}
```

### Paper D: Donaldson Analytic Note

```bibtex
@misc{gift_paper_d_donaldson,
  title        = {An Explicit Closed-Form G\textsubscript{2} Ansatz on a K3-Coassociative Neck with Hyperk\"{a}hler Rotation and Picard--Lefschetz Wirtinger Certificate},
  author       = {de La Fournière, Brieuc},
  year         = {2026},
  month        = {May},
  doi          = {10.5281/zenodo.20039066},
  url          = {https://doi.org/10.5281/zenodo.20039066},
  note         = {Donaldson direct chain closed at 5 levels; 15 pp.}
}
```

---

## Formal Verification (Core)

```bibtex
@software{gift_core_v3426,
  title   = {GIFT Core: Formal Verification in Lean 4},
  author  = {de La Fournière, Brieuc},
  year    = {2026},
  url     = {https://github.com/gift-framework/core},
  version = {3.4.26},
  note    = {143 Lean 4 files, 140 conjuncts, 15 axioms (4 main-chain + 11 interval-arithmetic), 0 sorry, Donaldson coassociative fibration formalized}
}
```

---

## DOI Information

| Archive | Link |
|---------|------|
| Zenodo (framework v3.4) | [10.5281/zenodo.20070101](https://doi.org/10.5281/zenodo.20070101) |
| Zenodo (framework v3.3 archive) | [10.5281/zenodo.18837071](https://doi.org/10.5281/zenodo.18837071) |
| Zenodo (Paper A, certified G₂) | [10.5281/zenodo.19892350](https://doi.org/10.5281/zenodo.19892350) |
| Zenodo (Paper B, spectral) | [10.5281/zenodo.19893371](https://doi.org/10.5281/zenodo.19893371) |
| Zenodo (Paper C, K3 NK) | [10.5281/zenodo.19708916](https://doi.org/10.5281/zenodo.19708916) |
| Zenodo (Paper D, Donaldson analytic) | [10.5281/zenodo.20039066](https://doi.org/10.5281/zenodo.20039066) |
| ResearchGate | [Author page](https://www.researchgate.net/profile/Brieuc-De-La-Fourniere) |

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 3.4.26 | 2026-06-03 | Numerology purge; observables.json refreshed to NuFIT 6.1 (Type I 0.99%); 143 .lean files, 8391 jobs, 15 axioms (4+11), 140 conjuncts; giftpy 3.4.26 on PyPI |
| 3.4.24 | 2026-06-01 | Academic terminology cleanup across K3 modules and papers |
| 3.4.23 | 2026-05-19 | Closed-form K3 CY-residual witness, interval-certified (ε₃' < 10⁻³) |
| 3.4.20 | 2026-05-10 | Independent-validations refresh; documentation sync |
| 3.4.19 | 2026-05-07 | Framework v3.4 PDFs published (Zenodo 20070101); Donaldson direct chain closed (5-layer Wirtinger cert); Paper D published (Zenodo 20039066) |
| 3.4.13 | 2026-04-29 | Triptyque published (Papers A, B, C on Zenodo); axiom reduction 38→4 main-chain; K3NK v3.0 hardcore (Joyce ×17); γ² = 24π²/7 derived |
| 3.4.3 | 2026-04 | G₂ Mathlib steps 1-5 promoted to theorems (8→4 axioms); MollifiedSum archived |
| 3.4.0 | 2026-04 | Metric-first program complete; K3 CAP (Fermat ×990, CI(2,2,2) ×6.4) |
| 3.3.24 | 2026-03-02 | NuFIT 6.0 update, neutrino formulas, S1 softened |
| 3.3.18 | 2026-02-21 | Bullet-proof validation: Westfall-Young maxT, 3M+ exhaustive |
| 3.3.14 | 2026-01-28 | Selection Principle, TCS Spectral Bounds, 290+ relations |
| 3.3.0 | 2026-01-12 | 33 observables, PDG 2024 values, Monte Carlo validation |
| 3.1.0 | 2025-12-17 | Analytical G₂ metric, 185 certified relations |
| 3.0.0 | 2025-12-09 | Major release: 165+ relations, Fibonacci/Monster/McKay |
| 2.3.x | 2025-12 | Lean 4 verification |
| 2.2.0 | 2025-11-27 | Zero-parameter paradigm |
| 2.0.0 | 2025-10-24 | Framework reorganization |

---

## Usage in Publications

When using GIFT predictions:

1. **Cite framework**: Use software citation above
2. **Cite specific results**: Use the relevant document or triptyque entry
3. **Specify version**: Always include version number (v3.4)
4. **Link repository**: Include GitHub URL for reproducibility

### Example

> "We compare our measurements with predictions from the Geometric Information Field Theory (GIFT) framework [1], which derives Standard Model parameters from E₈×E₈ topology. The GIFT prediction for the CP violation phase is δ_CP = 197° [2]."
>
> [1] de La Fournière, B. "GIFT Framework v3.4," 2026, https://github.com/gift-framework/GIFT
>
> [2] de La Fournière, B. "GIFT S2: Complete Derivations," 2026.

---

## License

MIT License: See [LICENSE](LICENSE)

---

**Version**: 3.4.26 (2026-06-03)
