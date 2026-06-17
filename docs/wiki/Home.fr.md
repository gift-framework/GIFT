---
title: "Accueil"
layout: default
---

# GIFT : Geometric Information Field Theory

**Un cadre théorique à zéro paramètre, qui dérive les constantes du Modèle Standard à partir de la géométrie d'holonomie G₂.**

---

## En bref

| Propriété | Valeur |
|---|---|
| **Prédictions** | 95 observables (33 Type I + 19 Type II + 21 Type III + 22 Type IV) |
| **Écart moyen** | 0,99 % sur 33 relations Type I (cible exacte, NuFIT 6.1 / PDG 2024 / Planck 2018 / CODATA 2022) |
| **Paramètres libres** | 0 (3 primitives entières : N=3, r₈=8, r₂=2) |
| **Vérification Lean 4** | 143 fichiers, 8391 jobs de build, 0 sorry, 15 axiomes (4 principaux + 11 d'arithmétique d'intervalle) (core v3.4.27) |
| **Significativité statistique** | p < 2×10⁻⁵ (σ > 4,2), unique parmi 3M+ configurations |
| **Modèle nul Monte Carlo** | P(algébrique) = 10⁻¹³⁸ sur 3M+ formules |

---

## Liens rapides

| | |
|---|---|
| **Vous découvrez ?** | [Pour commencer](Getting-Started.fr.html) : choisissez votre voie |
| **Lire les articles** | [Article principal](Paper-Main-Framework.html) ([Métrique G₂ explicite](Paper-Explicit-G2-Metric.html)) [Géométrie spectrale](Paper-Spectral-Geometry.html) |
| **Parcourir les prédictions** | [Référence des observables](Observable-Reference.html), 95 observables avec formules |
| **Vérifier les preuves** | [Formalisation Lean](Lean-Formalization.html), certificat à 140 conjonctions |

---

## Résultats clés

| Observable | Formule GIFT | Valeur | Exp. | Écart |
|---|---|---|---|---|
| sin²θ_W | b₂/(b₃+dim(G₂)) = 21/91 | 3/13 | 0,2312 | 0,19 % |
| N_gen | rang(E₈) − Weyl | 3 | 3 | exact |
| Q_Koide | dim(G₂)/b₂ = 14/21 | 2/3 | 0,6667 | 0,001 % |
| α_s(M_Z) | √2/(dim(G₂)−p₂) | √2/12 | 0,1179 | 0,04 % |
| δ_CP | dim(K₇)×dim(G₂)+H* | 197° | ~207-212° (NuFIT 6.1) | ~1σ |
| m_τ/m_e | 7+10×248+10×99 | 3477 | 3477,2 | 0,004 % |
| Ω_DE | ln(2)×98/99 | 0,686 | 0,685 | 0,21 % |

## Relations algébriques exactes

Toutes les prédictions découlent des constantes topologiques d'une variété compacte G₂ K₇ de nombres de Betti b₂ = 21, b₃ = 77, couplée à la structure de jauge E₈×E₈ (dim = 496) :

```
sin²θ_W = b₂ / (b₃ + dim(G₂))  = 21/91  = 3/13
Q_Koide  = dim(G₂) / b₂          = 14/21  = 2/3
N_gen    = |PSL(2,7)| / fond(E₇) = 168/56 = 3
κ_T      = 1/(b₃ − dim(G₂) − p₂) = 1/61
α        = e^K (couplage géométrique, zéro paramètre libre)
```

## Tests de falsification

| Prédiction | Expérience | Calendrier | État |
|---|---|---|---|
| δ_CP = 197° (fenêtre [182°, 212°]) | DUNE | 2028–2040 | en attente |
| sin²θ_W = 3/13 | FCC-ee | 2040s | en attente |
| N_gen = 3 (pas de 4e gén.) | LHC/FCC | en cours | cohérent |
| m_s/m_d = 20 | QCD sur réseau | ~2030 | cohérent |

---

## Vulgarisation

Une écriture accessible sur les idées derrière GIFT vit sur le [blog Substack](https://arithmon.substack.com/) (en anglais) ; les traductions FR sont sur ce wiki, voir l'[index des articles de blog](Site-Map.fr.html#articles-de-blog) pour la liste complète. Dernier paru : *[Le verre qui chante](Blog-The-Singing-Glass.fr.html)*, quand un verre de Bourgogne explique les dimensions cachées de l'univers.

---

## Structure

Ce wiki regroupe la documentation de GIFT à travers trois dépôts :

- **[gift-framework/GIFT](https://github.com/gift-framework/GIFT)** : documentation théorique, articles, vulgarisation
- **[gift-framework/core](https://github.com/gift-framework/core)** : preuves Lean 4, paquet Python, blueprint
- **Dépôts Zenodo** : données archivées avec DOI

Parcourez la barre latérale pour la navigation complète, ou consultez le [plan du site](Site-Map.html) pour un index complet.

---

*GIFT Framework v3.4.27 | [GitHub](https://github.com/gift-framework/GIFT) | [Core](https://github.com/gift-framework/core) | [Blueprint](https://gift-framework.github.io/core/) | [Zenodo (Article A)](https://doi.org/10.5281/zenodo.19892350) · [Article B](https://doi.org/10.5281/zenodo.19893371) · [Article C](https://doi.org/10.5281/zenodo.19708916) | Licence MIT*
