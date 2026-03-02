# Citation Guide

Citation formats for the GIFT Framework v3.3.

## Software Citation (Recommended)

### BibTeX

```bibtex
@software{gift_framework_v33,
  title   = {GIFT Framework v3.3: Geometric Information Field Theory},
  author  = {de La Fournière, Brieuc},
  year    = {2026},
  url     = {https://github.com/gift-framework/GIFT},
  version = {3.3.24},
  license = {MIT},
  note    = {33 dimensionless predictions, 0.24\% mean deviation (32 well-measured, PDG 2024 / NuFIT 6.0), 290+ certified relations}
}
```

### APA Style

```
de La Fournière, B. (2026). GIFT Framework v3.3: Geometric Information Field Theory (Version 3.3.24) [Software]. GitHub. https://github.com/gift-framework/GIFT
```

### Chicago Style

```
de La Fournière, Brieuc. "GIFT Framework v3.3: Geometric Information Field Theory." Version 3.3.24. GitHub, 2026. https://github.com/gift-framework/GIFT.
```

---

## Theoretical Framework Citation

### BibTeX

```bibtex
@article{gift_theory_v33,
  title  = {Geometric Information Field Theory v3.3: Topological Determination of Standard Model Parameters},
  author = {de La Fournière, Brieuc},
  year   = {2026},
  note   = {Mean deviation 0.24\% across 32 well-measured observables (PDG 2024 / NuFIT 6.0), zero continuous adjustable parameters, 290+ proven exact relations},
  url    = {https://github.com/gift-framework/GIFT}
}
```

---

## Citing Specific Documents

### Main Paper

```bibtex
@misc{gift_main_v33,
  title        = {Geometric Information Field Theory: Topological Determination of Standard Model Parameters},
  author       = {de La Fournière, Brieuc},
  year         = {2026},
  howpublished = {GIFT Framework v3.3},
  url          = {https://github.com/gift-framework/GIFT/blob/main/publications/papers/markdown/GIFT_v3.3_main.md}
}
```

### S1: Foundations

```bibtex
@misc{gift_s1_foundations,
  title        = {GIFT S1: Mathematical Foundations - E₈, G₂, K₇},
  author       = {de La Fournière, Brieuc},
  year         = {2026},
  howpublished = {GIFT Framework v3.3, Supplement S1},
  url          = {https://github.com/gift-framework/GIFT/blob/main/publications/papers/markdown/GIFT_v3.3_S1_foundations.md}
}
```

### S2: Derivations

```bibtex
@misc{gift_s2_derivations,
  title        = {GIFT S2: Complete Derivations - 33 Dimensionless Observables},
  author       = {de La Fournière, Brieuc},
  year         = {2026},
  howpublished = {GIFT Framework v3.3, Supplement S2},
  url          = {https://github.com/gift-framework/GIFT/blob/main/publications/papers/markdown/GIFT_v3.3_S2_derivations.md}
}
```

---

## Citing Specific Results

### CP Violation Phase

```bibtex
@misc{gift_cp_violation,
  title        = {Exact Topological Formula for CP Violation Phase: δ_CP = 197°},
  author       = {de La Fournière, Brieuc},
  year         = {2026},
  howpublished = {GIFT Framework v3.3},
  note         = {Formula: δ_CP = dim(K₇)×dim(G₂) + H* = 7×14 + 99 = 197°},
  url          = {https://github.com/gift-framework/GIFT/blob/main/publications/papers/markdown/GIFT_v3.3_S2_derivations.md}
}
```

### Weinberg Angle

```bibtex
@misc{gift_weinberg_angle,
  title        = {Topological Derivation of Weinberg Angle: sin²θ_W = 3/13},
  author       = {de La Fournière, Brieuc},
  year         = {2026},
  howpublished = {GIFT Framework v3.3},
  note         = {Formula: sin²θ_W = b₂/(b₃ + dim(G₂)) = 21/91 = 3/13},
  url          = {https://github.com/gift-framework/GIFT/blob/main/publications/papers/markdown/GIFT_v3.3_S2_derivations.md}
}
```

### Generation Number

```bibtex
@misc{gift_generation_number,
  title        = {Topological Proof of Three Fermion Generations},
  author       = {de La Fournière, Brieuc},
  year         = {2026},
  howpublished = {GIFT Framework v3.3},
  note         = {N_gen = 3 from Atiyah-Singer index theorem on K₇},
  url          = {https://github.com/gift-framework/GIFT/blob/main/publications/papers/markdown/GIFT_v3.3_S1_foundations.md}
}
```

---

## Formal Verification

```bibtex
@software{gift_core_v3319,
  title   = {GIFT Core: Formal Verification in Lean 4},
  author  = {de La Fournière, Brieuc},
  year    = {2026},
  url     = {https://github.com/gift-framework/core},
  version = {3.3.24},
  note    = {290+ relations verified, K₇ metric pipeline, Selection Principle, Spectral Theory}
}
```

---

## DOI Information

| Archive | Link |
|---------|------|
| Zenodo | [10.5281/zenodo.17979433](https://doi.org/10.5281/zenodo.17979433) |
| ResearchGate | [Author page](https://www.researchgate.net/profile/Brieuc-De-La-Fourniere) |

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 3.3.24 | 2026-03-02 | NuFIT 6.0 update, new neutrino formulas, Weyl→w rename, S1 softened, publications cleaned |
| 3.3.19 | 2026-02-21 | S3 removal, cross-ref cleanup, sync with core v3.3.19 |
| 3.3.18 | 2026-02-21 | Bullet-proof validation: Westfall-Young maxT, multi-stat PPC, noise curve, 3M+ exhaustive |
| 3.3.17 | 2026-02-04 | θ₂₃ formula correction, α_s = √2/12, 0.21% mean deviation |
| 3.3.14 | 2026-01-28 | Selection Principle, TCS Spectral Bounds, 290+ relations, Lean 4 only |
| 3.3.0 | 2026-01-12 | 33 observables, PDG 2024 values, 192,349-config Monte Carlo |
| 3.1.0 | 2025-12-17 | Analytical G₂ metric, 185 certified relations |
| 3.0.0 | 2025-12-09 | Major release: 165+ certified relations, Fibonacci/Monster/McKay |
| 2.3.x | 2025-12 | Lean 4 verification |
| 2.2.0 | 2025-11-27 | Zero-parameter paradigm |
| 2.1.0 | 2025-11-22 | Torsional dynamics |
| 2.0.0 | 2025-10-24 | Framework reorganization |

---

## Usage in Publications

When using GIFT predictions:

1. **Cite framework** - Use software citation above
2. **Cite specific results** - Use appropriate document citation
3. **Specify version** - Always include version number (v3.3)
4. **Link repository** - Include GitHub URL for reproducibility

### Example

> "We compare our measurements with predictions from the Geometric Information Field Theory (GIFT) framework [1], which derives Standard Model parameters from E₈×E₈ topology. The GIFT prediction for the CP violation phase is δ_CP = 197° [2]."
>
> [1] de La Fournière, B. "GIFT Framework v3.3," 2026, https://github.com/gift-framework/GIFT
>
> [2] de La Fournière, B. "GIFT S2: Complete Derivations," 2026.

---

## License

MIT License - See [LICENSE](LICENSE)

---

**Version**: 3.3.24 (2026-03-02)
