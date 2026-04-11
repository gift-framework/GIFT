---
title: "Article géométrie spectrale"
layout: default
---

# Article : géométrie spectrale

**Géométrie spectrale d'une métrique G₂ explicite sur une 7-variété compacte**

*Brieuc de La Fournière (2026)*
[Texte intégral (markdown)](https://github.com/gift-framework/GIFT/blob/main/publications/papers/markdown/Spectral_Geometry.md) | [DOI Zenodo : 10.5281/zenodo.18920368](https://doi.org/10.5281/zenodo.18920368)

---

## Résumé

Premier calcul numérique explicite du spectre de Kaluza-Klein sur une variété G₂ compacte. La décomposition adiabatique K₇ ≈ K3 × T² × I réduit les EDP 7D à des EDO de Sturm-Liouville 1D. Tous les nombres de Betti confirmés spectralement : b₀=1, b₁=0, b₂=21, b₃=77. Écart SD/ASD dans la matrice d'intersection K3 : 2210×.

---

## Résultats clés

### Spectre scalaire

| Quantité | Valeur |
|----------|--------|
| Mode zéro λ₀ | 3,47×10⁻¹³ (zéro machine) |
| Spectral gap λ₁ | 0,1244 ± 0,0001 |
| Loi de Weyl | λₙ = 0,125n², α = 1,998 (exact : 2,0) |

### Confirmation des nombres de Betti

| Betti | Spectral | Rapport de gap |
|-------|----------|----------------|
| b₀ = 1 | 1 mode zéro | |
| b₁ = 0 | aucune 1-forme zéro | |
| b₂ = 21 | 21 valeurs propres proches de zéro | 14 635× |
| b₃ = 77 | 77 valeurs propres proches de zéro | |

### Hiérarchie des masses (depuis le gap SD/ASD)

| Rapport | Spectral | Exp. | Écart |
|---------|----------|------|-------|
| m₁/m₂ (τ/μ) | 16,5 | 16,82 | 1,9 % |
| m₁/m₃ (τ/e) | 3400 | 3477 | 2,2 % |
| Gap SD/ASD | 2210× | (\|) | |

### Validation adiabatique (5 tests)

| Test | Résultat |
|------|----------|
| Platitude des fibres | < 0,002 % de variation max en s |
| Erreur d'additivité | 0,003 à 0,023 % |
| Exposant de la loi de Weyl | α = 1,998 (exact : 2,0) |
| Isotropie T² | \|g^θθ − g^ψψ\| = 3×10⁻⁷ |
| Rondeur K3 | étalement < 0,1 % |

### Tour KK

- 1744 valeurs propres distinctes (λ < 20)
- 4460 états avec multiplicités
- Hiérarchie à trois échelles : col, T², K3

---

## Structure des sections

1. **Introduction** : contexte, validation de l'ansatz adiabatique
2. **La métrique** : résumé Chebyshev-Cholesky, certification
3. **Laplacien scalaire** : spectral gap, loi de Weyl, tour KK
4. **Laplacien de Hodge sur les 2-formes**, confirmation b₂=21, structure SD/ASD
5. **Formes harmoniques et nombres de Betti** : formes K3, assemblage K₇, b₃=77
6. **Laplacien de Hodge sur les 1-formes** : démocratie spectrale à 10⁻⁴, b₁=0
7. **Limites singulières** : modèle de singularité ADE, stabilité spectrale
8. **Discussion** : G₂-MSSM, F-théorie, paysage des cordes
9. **Conclusion**

---

## Figures

1. Profils de la métrique : transition du col et décroissance ACyl
2. Escalier des valeurs propres scalaires (loi de Weyl)
3. Cinq premières fonctions propres scalaires
4. Spectres des canaux T² (additivité adiabatique)
5. Spectre des 2-formes avec gap de 14 635×

---

## Liens connexes

- [Article métrique G₂ explicite](Paper-Explicit-G2-Metric.html) : la métrique que cet article analyse
- [Article principal](Paper-Main-Framework.html) : prédictions physiques de cette géométrie
- [Article S1 fondations](Paper-S1-Foundations.html) : théorie de la construction TCS
- [Référence des observables](Observable-Reference.fr.html) : catalogue des prédictions
