---
title: "Structure du dépôt"
layout: default
---

# Structure du dépôt

Ce dépôt contient la documentation théorique de GIFT v3.3.

## Arborescence

```
GIFT/
├── publications/                      # Contenu publié et validation
│   ├── papers/                        # Articles scientifiques
│   │   ├── markdown/                  # Documents principaux (v3.3)
│   │   │   ├── GIFT_v3.3_main.md         # Article principal
│   │   │   ├── GIFT_v3.3_S1_foundations.md   # Fondations E₈, G₂, K₇
│   │   │   ├── GIFT_v3.3_S2_derivations.md   # 33 dérivations sans dimension
│   │   │   ├── Explicit_G2_Metric.md          # Construction de la métrique G₂ analytique
│   │   │   └── Spectral_Geometry.md           # Spectre KK, Yukawa, unification de jauge
│   │   ├── tex/                       # Sources LaTeX
│   │   ├── pdf/                       # PDFs compilés
│   │   ├── figures/                   # Figures de publication (PDF + PNG)
│   │   └── notebooks/                 # Notebooks Jupyter compagnons
│   │       ├── g2_certified_neck_companion.ipynb
│   │       └── g2_spectral_companion.ipynb
│   ├── outreach/                      # Vulgarisation et articles de blog
│   │   └── (7 posts Substack)
│   ├── references/                    # Données et catalogues de référence
│   │   ├── GIFT_ATLAS.json            # Atlas structuré canonique (v3.3)
│   │   ├── observables.csv            # Observables lisibles par machine
│   │   ├── OBSERVABLE_REFERENCE.md    # Catalogue complet des observables
│   │   ├── STATISTICAL_EVIDENCE.md    # Analyse statistique rigoureuse
│   │   ├── INDEPENDENT_VALIDATIONS.md # Recherches externes convergentes avec GIFT
│   │   └── Bibliography.md            # Références
│   └── validation/                    # Validation Monte Carlo (v3.3 uniquement)
│       ├── validation_v33.py          # Formules principales et données expérimentales
│       ├── bulletproof_validation_v33.py    # Validation bullet-proof à 7 composantes
│       ├── exhaustive_validation_v33.py     # Recherche exhaustive (3M+ configs)
│       ├── comprehensive_statistics_v33.py  # Tests statistiques avancés
│       └── selection/                 # Sélection de formules et analyse de Pareto
│
├── docs/                              # Documentation utilisateur
│   ├── GIFT_FOR_EVERYONE.md           # Guide complet avec analogies du quotidien
│   ├── FAQ.md                         # Questions fréquentes
│   ├── GLOSSARY.md                    # Termes techniques
│   ├── GIFTPY_FOR_GEOMETERS.md        # Guide pour les géomètres
│   ├── INFO_GEO_FOR_PHYSICISTS.md     # Guide pour les physiciens
│   ├── LEAN_FOR_PHYSICS.md            # Guide pour la formalisation
│   └── figures/                       # Blueprints Lean, diagrammes
│
├── README.md                          # Aperçu principal du dépôt
├── CHANGELOG.md                       # Historique des versions
├── CITATION.md                        # Comment citer
├── STRUCTURE.md                       # Ce fichier
└── LICENSE                            # Licence MIT
```

## Navigation rapide

| Vous cherchez... | Allez à |
|------------------|---------|
| Vue d'ensemble du cadre | [Accueil](Home.fr.html) |
| Guide pour débutants | [GIFT pour tout le monde](GIFT-for-Everyone.fr.html) |
| Théorie complète | [Article principal](Paper-Main-Framework.html) |
| Toutes les dérivations | [Article S2 dérivations](Paper-S2-Derivations.html) |
| Géométrie spectrale | [Article géométrie spectrale](Paper-Spectral-Geometry.html) |
| Notebooks compagnons | [publications/papers/notebooks/](https://github.com/gift-framework/GIFT/tree/main/publications/papers/notebooks) |
| Données d'observables | [observables.csv](https://github.com/gift-framework/GIFT/blob/main/publications/references/observables.csv) |
| Validation Monte Carlo | [publications/validation/](https://github.com/gift-framework/GIFT/tree/main/publications/validation) |
| Articles de blog et vulgarisation | Voir Articles de blog dans la barre latérale |
| Vérification formelle | [gift-framework/core](https://github.com/gift-framework/core) |
| Définitions techniques | [Glossaire](Glossary.fr.html) |

## Documents principaux (v3.3)

| Document | Contenu |
|----------|---------|
| GIFT_v3.3_main.md | Cadre théorique complet |
| GIFT_v3.3_S1_foundations.md | Construction mathématique E₈, G₂, K₇ |
| GIFT_v3.3_S2_derivations.md | 33 dérivations sans dimension avec preuves |
| Explicit_G2_Metric.md | Construction de la métrique G₂ analytique |
| Spectral_Geometry.md | Spectre KK, Yukawa, unification de jauge à partir de la métrique G₂ |

## Dépôts associés

| Dépôt | Contenu |
|-------|---------|
| [gift-framework/core](https://github.com/gift-framework/core) | Preuves formelles (Lean 4), pipeline métrique K₇, giftpy |

## Version

**Actuelle** : v3.4.8 (2026-04-11)
**Relations** : 460+ certifiées (core v3.4.8, 7 axiomes)
**Prédictions** : 33 prédictions (**0,24 % d'écart moyen** sur 32 bien mesurées, 0,57 % incl. δ_CP ; PDG 2024 / NuFIT 6.0)
**Validation** : 3 070 396 configurations exhaustives + bullet-proof à 7 composantes (Westfall-Young, Bayésien, PPC)
**Résultat clé** : métrique G₂ analytique avec T = 0 exactement
