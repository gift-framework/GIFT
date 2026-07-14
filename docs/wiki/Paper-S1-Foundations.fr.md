---
title: "Article S1 fondations"
layout: default
---

# Article : S1, fondations mathématiques

**Supplément S1 : fondations mathématiques, algèbre de Lie exceptionnelle E₈, variétés à holonomie G₂ et construction de K₇**

*Brieuc de La Fournière (2026)*
[Texte intégral (markdown, v3.4)](https://github.com/Arithmon/K7/blob/main/publications/papers/markdown/GIFT_v3.4_S1_foundations.md) | [archive v3.3 sur Zenodo : 10.5281/zenodo.18837071](https://doi.org/10.5281/zenodo.18837071)

> **Mise à jour v3.4.** Le contenu canonique est dans [`GIFT_v3.4_S1_foundations.md`](https://github.com/Arithmon/K7/blob/main/publications/papers/markdown/GIFT_v3.4_S1_foundations.md). Le LaTeX/PDF v3.4 est en cours de recompilation avec le template GIFT.

---

## Résumé

Développe l'architecture E₈, les variétés à holonomie G₂ via le noyau de la dérivée de Lie, et la construction de K₇ via twisted connected sum. Établit la forme algébrique de référence det(g) = 65/32 et le théorème d'existence de Joyce garantissant une métrique sans torsion.

---

## Résultats clés

| Résultat | Valeur | Statut |
|----------|--------|--------|
| Chaîne d'algèbres à division | ℝ(1) → ℂ(2) → ℍ(4) → 𝕆(8) | terminale à 8 |
| Système de racines E₈ | 240 racines = 112 D₈ + 128 demi-entières | vérifié |
| \|W(E₈)\| | 2¹⁴ × 3⁵ × 5² × 7 = 696 729 600 | vérifié en Lean |
| Blocs de construction TCS | M₁(quintique)[b₂=11,b₃=40] + M₂(CI(2,2,2))[b₂=10,b₃=37] | → K₇[21,77] |
| det(g) | 65/32 (3 chemins indépendants) | exact |
| Spectral gap | λ₁ = 13/99 | algébrique |

---

## Structure des sections

- **Partie 0** : fondation octonionique, pourquoi 𝕆 est terminale, G₂ = Aut(𝕆), plan de Fano
- **Partie I** : algèbre de Lie exceptionnelle E₈, système de racines, groupe de Weyl, chaîne exceptionnelle
- **Partie II** : variétés à holonomie G₂, définition, classification de Berger, classes de torsion W₁ à W₂₇
- **Partie III** : construction de la variété K₇, cadre TCS, blocs ACyl, Mayer-Vietoris
- **Partie IV** : structure métrique et vérification, κ_T = 1/61, det(g) = 65/32, existence de Joyce

---

## L'identité triple de Weyl

```
Weyl = (dim(G₂)+1)/N_gen = b₂/N_gen − p₂ = dim(G₂) − rang(E₈) − 1 = 5
```

---

## Liens connexes

- [Article principal](Paper-Main-Framework.html) : article principal
- [Article S2 dérivations](Paper-S2-Derivations.html) : les 33 dérivations
- [Article métrique G₂ explicite](Paper-Explicit-G2-Metric.html) : métrique numérique
- [Glossaire](Glossary.fr.html) : définitions des termes
