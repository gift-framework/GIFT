---
title: GIFT: Geometric Information Field Theory
layout: default
---

# Geometric Information Field Theory

**Et si la physique n'était pas finement ajustée, mais seulement bien formée ?**

GIFT dérive les paramètres du Modèle Standard à partir de la géométrie d'une seule variété de dimension 7. Aucun paramètre libre. Aucun ajustement. Chaque prédiction est une conséquence de la forme : la théorie de jauge E₈ × E₈ compactifiée sur une variété K₇ à holonomie G₂, de nombres de Betti (b₂, b₃) = (21, 77).

---

## Par où commencer

| | |
|:---:|:---:|
| [**Regarder** (8 min)](https://www.youtube.com/watch?v=6DVck30Q6XM) | [**Lire le blog**](https://giftheory.substack.com/) |
| [**GIFT pour Tout le Monde**](GIFT_FOR_EVERYONE.fr.html) | [**FAQ**](wiki/FAQ.fr.html) |

---

## En bref

| | |
|---|---|
| **Précision** | 0,24 % d'écart moyen sur 32 observables bien mesurés (PDG 2024 / NuFIT 6.0) |
| **Observables** | 92 au total (66 avec comparaison expérimentale) |
| **Paramètres** | Zéro ajustable (tous structurellement déterminés) |
| **Vérifié** | 130 fichiers Lean 4, 7 axiomes, zéro preuve incomplète ([core v3.4.7](https://github.com/gift-framework/core)) |
| **Unicité** | #1 sur 3 070 396 configurations testées |

**Réduction dimensionnelle** : E₈ × E₈ (496D) → AdS₄ × K₇ (11D) → Modèle Standard (4D)

---

## Quelques résultats clés

### Relations exactes (vérifiées en Lean 4)

| Relation | Valeur | Origine topologique |
|---|:---:|---|
| sin²θ_W | 3/13 | b₂/(b₃ + dim(G₂)) |
| κ_T | 1/61 | 1/(b₃ − dim(G₂) − p₂) |
| τ | 3472/891 | (496 × 21)/(27 × 99) |
| det(g) | 65/32 | déterminant de la métrique issu de la structure G₂ |
| δ_CP | 197° | 7 × dim(G₂) + H* |
| m_s/m_d | 20 | p₂² × w |
| Q_Koide | 2/3 | dim(G₂)/b₂ |

---

## Pour les différents publics

| Profil | Commencer par |
|---|---|
| Tout le monde | [GIFT pour Tout le Monde](GIFT_FOR_EVERYONE.fr.html), guide complet avec analogies du quotidien |
| Découverte rapide | [Pour commencer](wiki/Getting-Started.fr.html), choisissez votre voie |
| Vue d'ensemble | [Accueil du wiki](wiki/Home.fr.html), résultats, structure, références |
| Questions | [FAQ](wiki/FAQ.fr.html), questions courantes |
| Définitions | [Glossaire](wiki/Glossary.fr.html), termes techniques |

## Articles de blog (vulgarisation)

- [Ce qui vient avant](wiki/Blog-On-What-Comes-First.fr.html), humilité devant les nombres
- [Gift from Bit](wiki/Blog-Gift-from-Bit.fr.html), quand la géométrie donne
- [13 théorèmes, zéro confiance requise](wiki/Blog-13-Theorems-Zero-Trust.fr.html): GIFT rencontre Lean 4
- [Le théorème de Joyce, désormais en Lean](wiki/Blog-Joyce-Theorem-in-Lean.fr.html), existence des variétés G₂
- [L'algèbre qui attendait](wiki/Blog-The-Algebra-That-Waited.fr.html), sur les octonions et un casse-tête de 43 ans
- [Les poux de l'univers](wiki/Blog-Lice-of-the-Universe.fr.html), perception et dimensions cachées
- [La géométrie de l'impossible](wiki/Blog-Roberto-Carlos-Geometry.fr.html), ce que Roberto Carlos savait

---

> 🇬🇧 La majorité de la documentation technique reste en anglais. Cette page d'accueil et les pages essentielles (FAQ, glossaire, GIFT pour Tout le Monde, et les billets de blog) sont disponibles en français. La traduction des pages techniques pour spécialistes (Pour les physiciens, Pour les géomètres, articles de recherche) viendra plus tard.

---

*Licence MIT · GIFT Framework v3.4*
