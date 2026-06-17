---
title: "Guide de citation"
layout: default
---


Formats de citation pour le cadre GIFT v3.4.

## Citation logicielle (recommandée)

### BibTeX

```bibtex
@software{gift_framework_v34,
  title   = {GIFT Framework v3.4: Geometric Information Field Theory},
  author  = {de La Fournière, Brieuc},
  year    = {2026},
  url     = {https://github.com/gift-framework/GIFT},
  version = {3.4.27},
  license = {MIT},
  note    = {95 observables, 0.99\% mean deviation on 33 Type I relations (PDG 2024 / NuFIT 6.1), 140 conjuncts certified in Lean 4, 15 axioms (4 main-chain + 11 interval-arithmetic)}
}
```

### Style APA

```
de La Fournière, B. (2026). GIFT Framework v3.4: Geometric Information Field Theory (Version 3.4.27) [Logiciel]. GitHub. https://github.com/gift-framework/GIFT
```

### Style Chicago

```
de La Fournière, Brieuc. "GIFT Framework v3.4: Geometric Information Field Theory." Version 3.4.27. GitHub, 2026. https://github.com/gift-framework/GIFT.
```

---

## Citation du cadre théorique

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

## Citer des documents spécifiques

### Article principal

```bibtex
@misc{gift_main_v34,
  title        = {Geometric Information Field Theory: Topological Determination of Standard Model Parameters},
  author       = {de La Fournière, Brieuc},
  year         = {2026},
  howpublished = {GIFT Framework v3.4},
  url          = {https://github.com/gift-framework/GIFT/blob/main/publications/papers/markdown/GIFT_v3.4_main.md}
}
```

### S1 : fondations

```bibtex
@misc{gift_s1_foundations_v34,
  title        = {GIFT S1: Mathematical Foundations, E₈, G₂, K₇},
  author       = {de La Fournière, Brieuc},
  year         = {2026},
  howpublished = {GIFT Framework v3.4, Supplement S1},
  url          = {https://github.com/gift-framework/GIFT/blob/main/publications/papers/markdown/GIFT_v3.4_S1_foundations.md}
}
```

### S2 : dérivations

```bibtex
@misc{gift_s2_derivations_v34,
  title        = {GIFT S2: Complete Derivations, 33 Type I Relations},
  author       = {de La Fournière, Brieuc},
  year         = {2026},
  howpublished = {GIFT Framework v3.4, Supplement S2},
  url          = {https://github.com/gift-framework/GIFT/blob/main/publications/papers/markdown/GIFT_v3.4_S2_derivations.md}
}
```

### S3 : observables

```bibtex
@misc{gift_s3_observables_v34,
  title        = {GIFT S3: Observable Catalog, 95 Predictions},
  author       = {de La Fournière, Brieuc},
  year         = {2026},
  howpublished = {GIFT Framework v3.4, Supplement S3},
  url          = {https://github.com/gift-framework/GIFT/blob/main/publications/papers/markdown/GIFT_v3.4_S3_observables.md}
}
```

---

## Triptyque (articles compagnons revus par les pairs, Zenodo)

### Article A : structure G₂ certifiée

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

### Article B : géométrie spectrale

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

### Article C : Newton-Kantorovich sur K3

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

---

## Vérification formelle (Core)

```bibtex
@software{gift_core_v3426,
  title   = {GIFT Core: Formal Verification in Lean 4},
  author  = {de La Fournière, Brieuc},
  year    = {2026},
  url     = {https://github.com/gift-framework/core},
  version = {3.4.27},
  note    = {143 Lean 4 files, 140 conjuncts, 15 axioms (4 main-chain + 11 interval-arithmetic), 0 sorry}
}
```

---

## Informations DOI

| Archive | Lien |
|---------|------|
| Zenodo (archive cadre v3.3) | [10.5281/zenodo.18837071](https://doi.org/10.5281/zenodo.18837071) |
| Zenodo (Article A, G₂ certifiée) | [10.5281/zenodo.19892350](https://doi.org/10.5281/zenodo.19892350) |
| Zenodo (Article B, spectral) | [10.5281/zenodo.19893371](https://doi.org/10.5281/zenodo.19893371) |
| Zenodo (Article C, K3 NK) | [10.5281/zenodo.19708916](https://doi.org/10.5281/zenodo.19708916) |
| ResearchGate | [Page auteur](https://www.researchgate.net/profile/Brieuc-De-La-Fourniere) |

---

## Historique des versions

| Version | Date | Faits marquants |
|---------|------|-----------------|
| 3.4.27 | 2026-06-03 | Purge numérologie ; observables.json rafraîchi sur NuFIT 6.1 (Type I 0,99 %) ; 143 fichiers .lean, 8391 jobs, 15 axiomes (4+11), 140 conjonctions ; giftpy 3.4.27 sur PyPI |
| 3.4.24 | 2026-06-01 | Nettoyage terminologie académique (modules K3 et articles) |
| 3.4.23 | 2026-05-19 | Témoin résidu CY K3 sous forme close, certifié par intervalles (ε₃' < 10⁻³) |
| 3.4.20 | 2026-05-10 | Rafraîchissement validations indépendantes ; synchronisation documentation |
| 3.4.13 | 2026-04-29 | Triptyque publié (Articles A, B, C sur Zenodo) ; réduction d'axiomes 38→4 sur la chaîne principale ; K3NK v3.0 hardcore (Joyce ×17) ; γ² = 24π²/7 dérivé |
| 3.4.3 | 2026-04 | Étapes 1-5 G₂ Mathlib promues en théorèmes (8→4 axiomes) ; MollifiedSum archivé |
| 3.4.0 | 2026-04 | Programme metric-first complet ; K3 CAP (Fermat ×990, CI(2,2,2) ×6.4) |
| 3.3.24 | 2026-03-02 | Mise à jour NuFIT 6.0, formules neutrinos, S1 assoupli |
| 3.3.18 | 2026-02-21 | Validation à toute épreuve : Westfall-Young maxT, 3M+ exhaustif |
| 3.3.14 | 2026-01-28 | Principe de sélection, bornes spectrales TCS, 290+ relations |
| 3.3.0 | 2026-01-12 | 33 observables, valeurs PDG 2024, validation Monte Carlo |
| 3.1.0 | 2025-12-17 | Métrique G₂ analytique, 185 relations certifiées |
| 3.0.0 | 2025-12-09 | Version majeure : 165+ relations, Fibonacci/Monster/McKay |
| 2.3.x | 2025-12 | Vérification Lean 4 |
| 2.2.0 | 2025-11-27 | Paradigme à zéro paramètre |
| 2.0.0 | 2025-10-24 | Réorganisation du cadre |

---

## Utilisation en publication

Lorsque vous utilisez des prédictions GIFT :

1. **Citer le cadre** : utilisez la citation logicielle ci-dessus
2. **Citer des résultats spécifiques** : utilisez l'entrée du document ou du triptyque correspondante
3. **Préciser la version** : incluez toujours le numéro de version (v3.4)
4. **Lier le dépôt** : incluez l'URL GitHub pour la reproductibilité

### Exemple

> « Nous comparons nos mesures avec les prédictions du cadre Geometric Information Field Theory (GIFT) [1], qui dérive les paramètres du Modèle Standard de la topologie E₈×E₈. La prédiction GIFT pour la phase de violation CP est δ_CP = 197° [2]. »
>
> [1] de La Fournière, B. "GIFT Framework v3.4," 2026, https://github.com/gift-framework/GIFT
>
> [2] de La Fournière, B. "GIFT S2: Complete Derivations," 2026.

---

## Licence

Licence MIT : voir [LICENSE](LICENSE)

---

**Version** : 3.4.27 (2026-06-03)
