---
title: "Article principal"
layout: default
---

# Article : cadre principal

**Geometric Information Field Theory : dérivation topologique des paramètres du Modèle Standard à partir de variétés à holonomie G₂**

*Brieuc de La Fournière (2026)*
[Texte intégral (markdown, v3.4)](https://github.com/gift-framework/GIFT/blob/main/publications/papers/markdown/GIFT_v3.4_main.md) | [archive v3.3 sur Zenodo : 10.5281/zenodo.18837071](https://doi.org/10.5281/zenodo.18837071)

> **Mise à jour v3.4.** Le contenu canonique est dans [`GIFT_v3.4_main.md`](https://github.com/gift-framework/GIFT/blob/main/publications/papers/markdown/GIFT_v3.4_main.md). Le LaTeX/PDF v3.4 est en cours de recompilation avec le template GIFT. Le triptyque relecture par les pairs (articles A, B, C) est publié sur Zenodo : [A — G₂ certifié](https://doi.org/10.5281/zenodo.19892350) · [B — spectral](https://doi.org/10.5281/zenodo.19893371) · [C — K3 NK](https://doi.org/10.5281/zenodo.19708916).

---

## Résumé

Cadre proposant que les paramètres sans dimension du Modèle Standard émergent comme invariants topologiques d'une variété K₇ à holonomie G₂ de dimension 7 avec nombres de Betti (b₂=21, b₃=77) couplée à une structure de jauge E₈×E₈. La v3.4 catalogue 95 observables (35 relations Type I à cible exacte) avec un écart moyen de 0,39 % sur Type I (PDG 2024 / NuFIT 6.0) ; 213 conjonctions certifiées en Lean 4 avec 4 axiomes principaux. DUNE testera le critère de falsification δ_CP = 197°.

---

## Résultats clés

| Observable | Formule GIFT | Valeur | Exp. | Écart |
|------------|--------------|--------|------|-------|
| sin² θ_W | b₂/(b₃+dim(G₂)) | 3/13 | 0,2312 | 0,19 % |
| N_gen | rang(E₈)−Weyl | 3 | 3 | exact |
| Q_Koide | dim(G₂)/b₂ | 2/3 | 0,6667 | 0,001 % |
| α_s(M_Z) | √2/12 | 0,1179 | 0,1179 | 0,04 % |
| δ_CP | 7×14+99 | 197° | 177°±20° | 1σ |
| m_τ/m_e | 7+10×248+10×99 | 3477 | 3477,2 | 0,004 % |
| n_s | ζ(11)/ζ(5) | 0,9649 | 0,9649 | 0,004 % |

**Global** : 33 prédictions, 0,24 % d'écart moyen (32 bien mesurées), 4 exactes, 28/33 sous 1 %

---

## Structure des sections

1. **Introduction** : problème des paramètres, contexte contemporain, vue d'ensemble du cadre
2. **Cadre mathématique** : octonions, E₈×E₈, hypothèse K₇, structure G₂
3. **Méthodologie et statut épistémique** : principe de dérivation, ce qui est revendiqué vs non revendiqué
4. **Dérivation des 33 prédictions** : secteurs jauge, leptonique, des quarks, neutrinos, Higgs, cosmologique
5. **Vérification formelle et statistiques** : Lean 4 (290+ théorèmes), unicité parmi 192 349 alternatives
6. **Programme de la métrique G₂** : construction d'atlas PINN, résultats de qualité métrique
7. **Prédictions falsifiables**, δ_CP via DUNE, bornes de quatrième génération
8. **Discussion** : connexions M-théorie, comparaison avec d'autres approches, limitations
9. **Conclusion**

---

## Validation statistique

- (21,77) unique optimal parmi 192 349 configurations (p < 5×10⁻⁶)
- E₈×E₈ atteint 12,8× meilleur accord que le groupe de jauge suivant
- L'holonomie G₂ atteint 13× meilleur que Calabi-Yau (SU(3))
- Vérification Lean 4 : 290+ théorèmes, 0 sorry, 0 axiome spécifique au domaine

---

## Liens connexes

- [Article S1 fondations](Paper-S1-Foundations.html) : fondations mathématiques
- [Article S2 dérivations](Paper-S2-Derivations.html) : dérivations complètes
- [Article métrique G₂ explicite](Paper-Explicit-G2-Metric.html) : métrique G₂ numérique
- [Article géométrie spectrale](Paper-Spectral-Geometry.html) : spectre KK
- [Référence des observables](Observable-Reference.fr.html) : catalogue complet des prédictions
- [Preuves statistiques](Statistical-Evidence.fr.html), validation à 7 composantes
