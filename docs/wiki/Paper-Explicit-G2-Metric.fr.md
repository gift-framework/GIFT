---
title: "Article métrique G₂ explicite"
layout: default
---

# Article : métrique G₂ explicite

**Une métrique G₂ approximative explicite sur une 7-variété TCS compacte avec complétion sans torsion certifiée**

*Brieuc de La Fournière (2026)*
[Texte intégral (markdown)](https://github.com/gift-framework/GIFT/blob/main/publications/papers/markdown/g2_certified_neck.md) | [DOI Zenodo : 10.5281/zenodo.19892350](https://doi.org/10.5281/zenodo.19892350)

---

## Résumé

Construit une métrique Chebyshev-Cholesky explicite à 169 paramètres sur la TCS compacte K₇. Le certificat de Newton-Kantorovich prouve qu'une métrique G₂ unique sans torsion g* existe à une distance ≤ 4,86×10⁻⁶. Torsion initiale ‖T‖ = 8,94×10⁻² réduite à 2,98×10⁻⁵ en 5 itérations de Joyce (réduction de 3000×).

---

## Résultats clés

### Chaîne de certification

| Quantité | Valeur |
|----------|--------|
| Torsion initiale ‖T‖₀ | 8,936×10⁻² |
| Torsion finale ‖T‖₅ | 2,984×10⁻⁵ |
| Facteur de réduction | 2995× |
| Contraction NK h | 6,65×10⁻⁸ |
| Seuil NK | 0,5 |
| Marge de sécurité | ×7,5M |
| Distance à la métrique exacte | ≤ 4,86×10⁻⁶ |

### Propriétés métriques

| Propriété | Valeur |
|-----------|--------|
| Paramètres | 169 (168 Chebyshev + 1 décroissance ACyl) |
| det(g) | 65/32 (exact) |
| \|φ\|² | 42 (erreur < 10⁻¹⁴) |
| Holonomie | Hol(g*) = G₂ |
| Classe de torsion | 99,6 % dans W₃, \|dφ\|²/\|d*φ\|² = 1/5 |

### Hiérarchie des valeurs propres

Structure à trois échelles :
- **Col** (couture) : λ₀ ≈ 6,8
- **T²** (fibre) : λ₁,₆ ≈ 2,9
- **K3** (fibre) : λ₂₋₅ ≈ 1,1

---

## Structure des sections

1. **Introduction** : contexte, objectif, portée et revendications
2. **La variété** : construction TCS, topologie (b₂=21, b₃=77)
3. **La métrique** : hiérarchie de modèles, coordonnées, paramétrisation Chebyshev-Cholesky
4. **Définitions de normes et domaine** : distance métrique, normes de torsion, norme NK
5. **Analyse de torsion** : approximation initiale, vérification K3, réduction Gauss-Newton
6. **Certification** : convergence NK, arithmétique d'intervalle, preuve d'holonomie
7. **Invariants géométriques**, det(g)=65/32, |φ|²=42, Hol(g*)=G₂
8. **Discussion** : limitations, comparaison avec les travaux antérieurs
9. **Reproductibilité** : fichiers de données, notebook compagnon (< 1 min de runtime)

---

## Figures

- Visualisation TCS avec coloration d'intensité de torsion
- Schéma de carte d'atlas
- Profil des valeurs propres (hiérarchie à trois échelles)
- Convergence de la torsion (échelle log, 5 itérations)

---

## Liens connexes

- [Article principal](Paper-Main-Framework.html) : application physique
- [Article S1 fondations](Paper-S1-Foundations.html) : théorie de la construction TCS
- [Article géométrie spectrale](Paper-Spectral-Geometry.html) : analyse spectrale de cette métrique
- [Pour les géomètres](For-Geometers.fr.html) : vue d'ensemble du pipeline computationnel
