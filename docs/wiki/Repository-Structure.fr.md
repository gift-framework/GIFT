---
title: "Structure du dépôt"
layout: default
---

# Structure du dépôt

Ce dépôt contient la documentation théorique de GIFT v3.4.

## Arborescence

```
GIFT/
├── publications/                      # Contenu publié et validation
│   ├── papers/                        # Articles scientifiques
│   │   ├── markdown/                  # Documents principaux (v3.4, sources canoniques)
│   │   │   ├── GIFT_v3.4_main.md              # Article principal
│   │   │   ├── GIFT_v3.4_S1_foundations.md    # Fondations E₈, G₂, K₇
│   │   │   ├── GIFT_v3.4_S2_derivations.md    # 33 dérivations Type I
│   │   │   ├── GIFT_v3.4_S3_observables.md    # Catalogue de 95 observables
│   │   │   ├── g2_certified_neck.md           # Article A, structure G₂ certifiée
│   │   │   └── g2_spectral.md                 # Article B, géométrie spectrale
│   │   ├── tex/                       # Sources LaTeX (recompilation v3.4 en attente)
│   │   ├── pdf/                       # PDF compilés (Articles A, B publiés)
│   │   │   ├── g2_certified_neck.pdf          # Article A (Zenodo 19892350)
│   │   │   └── g2_spectral.pdf                # Article B (Zenodo 19893371)
│   │   ├── legacy/v3.3/               # PDFs + markdown du cadre v3.3 (archivés)
│   │   ├── figures/                   # Figures de publication (PDF + PNG)
│   │   └── notebooks/                 # Notebooks Jupyter compagnons
│   │       ├── g2_certified_neck_companion.ipynb
│   │       └── g2_spectral_companion.ipynb
│   ├── outreach/                      # Vulgarisation et articles de blog
│   ├── references/                    # Données et catalogues de référence
│   │   ├── GIFT_ATLAS.json            # Atlas structuré canonique
│   │   ├── observables.csv            # Observables lisibles par machine
│   │   ├── OBSERVABLE_REFERENCE.md    # Catalogue complet d'observables
│   │   ├── STATISTICAL_EVIDENCE.md    # Analyse statistique rigoureuse
│   │   ├── INDEPENDENT_VALIDATIONS.md # Recherches externes convergeant avec GIFT
│   │   └── Bibliography.md            # Références
│   └── validation/                    # Validation Monte Carlo
│       └── legacy/v3.3/               # Scripts de validation v3.3 (archivés ;
│                                      # rafraîchissement stats v3.4 dans core/private)
│
├── docs/                              # Documentation utilisateur et site Jekyll
│   ├── index.html                     # Page d'accueil (arithmon.github.io/K7)
│   ├── _config.yml                    # Configuration Jekyll
│   ├── GIFT_FOR_EVERYONE.md           # Guide complet avec analogies du quotidien
│   ├── FAQ.md                         # Questions fréquentes
│   ├── GLOSSARY.md                    # Termes techniques
│   ├── GIFTPY_FOR_GEOMETERS.md        # Guide pour géomètres
│   ├── INFO_GEO_FOR_PHYSICISTS.md     # Guide pour physiciens
│   ├── LEAN_FOR_PHYSICS.md            # Guide pour formalisation
│   ├── wiki/                          # Miroir du Wiki GitHub (EN + FR)
│   └── figures/                       # Blueprints Lean, schémas
│
├── README.md                          # Vue d'ensemble du dépôt principal
├── CHANGELOG.md                       # Historique des versions
├── CITATION.md                        # Comment citer
├── STRUCTURE.md                       # Ce fichier
└── LICENSE                            # Licence MIT
```

## Navigation rapide

| Vous cherchez... | Allez à |
|------------------|---------|
| Vue d'ensemble du cadre | `README.md` |
| Guide accessible aux débutants | `docs/GIFT_FOR_EVERYONE.md` |
| Théorie complète | `publications/papers/markdown/k7_framework_3_5_main.md` |
| Toutes les dérivations | `publications/papers/markdown/k7_framework_3_5_S2_derivations.md` |
| Catalogue d'observables | `publications/papers/markdown/k7_framework_3_5_S3_observables.md` |
| Article A (G₂ certifiée) | `publications/papers/pdf/g2_certified_neck.pdf` |
| Article B (géométrie spectrale) | `publications/papers/pdf/g2_spectral.pdf` |
| Notebooks compagnons | `publications/papers/notebooks/` |
| Données d'observables | `publications/references/observables.csv` |
| Articles de blog et vulgarisation | `publications/outreach/` |
| Vérification formelle | [Arithmon/K7-Lean](https://github.com/Arithmon/K7-Lean) |
| Archive v3.3 | `publications/papers/legacy/v3.3/` |
| Définitions techniques | `docs/GLOSSARY.md` |

## Documents principaux (v3.4)

| Document | Contenu |
|----------|---------|
| GIFT_v3.4_main.md | Cadre théorique complet |
| GIFT_v3.4_S1_foundations.md | Construction mathématique E₈, G₂, K₇ |
| GIFT_v3.4_S2_derivations.md | 33 dérivations Type I avec preuves |
| GIFT_v3.4_S3_observables.md | Catalogue de 95 observables (33 I + 19 II + 21 III + 22 IV) |
| g2_certified_neck.md (Article A) | Preuve d'existence G₂ assistée par ordinateur |
| g2_spectral.md (Article B) | Spectre laplacien, formes harmoniques |

## Wiki

Le **[Wiki GitHub](https://github.com/Arithmon/K7/wiki)** fournit un hub multi-public navigable consolidant toute la documentation, les résumés d'articles, les données de référence, les articles de blog et le méta-projet.

## Dépôts liés

| Dépôt | Contenu |
|-------|---------|
| [Arithmon/K7-Lean](https://github.com/Arithmon/K7-Lean) | Preuves formelles (Lean 4), pipeline de métrique K₇, giftpy |

## Version

**Actuelle** : v3.4.27 (2026-05-10)
**Relations** : 140 conjonctions certifiées (core v3.4.27, 15 axiomes (4 principaux + 11 d'arithmétique d'intervalle))
**Prédictions** : 95 observables (33 Type I, cibles exactes, 0,99 % d'écart moyen ; NuFIT 6.1 / PDG 2024 / Planck 2018 / CODATA 2022)
**Validation** : 3 000 000 de jeux de formules aléatoires, aucun ne reproduit le profil joint (borne au niveau ensemble ~10⁻⁶, sans hypothèse d'indépendance)
**Triptyque** : Articles A (Zenodo 19892350) + B (19893371) + C (19708916) publiés le 2026-04-29
