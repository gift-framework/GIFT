# Mazzeo-Melrose Analytic Surgery

**Relevance pour GIFT**: Cette théorie explique comment les valeurs propres se comportent quand on "colle" des variétés — exactement ce qu'on fait dans les constructions TCS de Kovalev.

---

## Papers Fondamentaux

### 1. "Analytic Surgery and the Accumulation of Eigenvalues"
**Auteurs**: Andrew Hassell, Rafe Mazzeo, Richard B. Melrose
**Publication**: Communications in Analysis and Geometry, Vol. 3, pp. 115-222 (1995)
**PDF**: [International Press](https://intlpress.com/site/pub/files/_fulltext/journals/cag/1995/0003/0001/CAG-1995-0003-0001-a004.pdf)

### 2. "Analytic Surgery and the Eta Invariant"
**Auteurs**: Rafe Mazzeo, Richard B. Melrose
**Publication**: Geometric and Functional Analysis, Vol. 5, pp. 14-75 (1995)
**Link**: [Springer](https://link.springer.com/article/10.1007/BF01928215)

---

## Concepts Clés

### Analytic Surgery
C'est une **déformation paramétrique** d'une variété M qui l'étire le long d'une hypersurface séparatrice H :

```
Avant:    M = M₊ ∪_H M₋       (variété compacte)

Après:    M_ε avec cou de longueur 2/ε

Limite:   M₊ ⊔ M₋              (deux variétés avec bouts cylindriques)
```

### Accumulation des Valeurs Propres
Quand ε → 0 (cou qui s'allonge) :

1. **Les petites valeurs propres** (λ → 0) s'accumulent vers 0
2. Leur comportement est gouverné par l'**opérateur modèle** sur le cylindre H × ℝ
3. Le nombre de petites valeurs propres est lié aux **harmoniques** sur H

### Résolvante et Noyau de la Chaleur
L'article développe une analyse uniforme de :
- La résolvante (Δ - λ)⁻¹
- Le noyau de la chaleur e^{-tΔ}

quand ε → 0, avec contrôle explicite des constantes.

---

## Application aux Variétés G₂ (TCS)

### Construction de Kovalev
```
M₇ = (X₁ × S¹) ∪_{cou} (X₂ × S¹)
```
où X₁, X₂ sont des variétés de Calabi-Yau asymptotiquement cylindriques.

### Conséquence pour les Valeurs Propres
Soit T = longueur du cou (→ ∞ dans la limite). Alors :

```
λₙ(M_T) ~ C_n / T²    pour les "petites" valeurs propres
```

Les coefficients C_n sont déterminés par :
- Les harmoniques sur la section transverse (K3 surface)
- Les conditions de matching au recollement

---

## Formule Clé pour η-invariant

Pour la chirurgie analytique :

```
η(M_ε) = η(M₊) + η(M₋) + terme de correction
```

Le terme de correction dépend des **valeurs propres croisées** (scattering) à travers le cou.

---

## Pertinence pour GIFT

### Ce qui pourrait aider

1. **Comprendre λ₁ vs longueur du cou**
   - Si λ₁ = 14/H*, comment ça se comporte quand T → ∞ ?
   - Est-ce que λ₁ reste borné inférieurement ?

2. **Prédire la constante universelle**
   - Les harmoniques sur K3 ont des contraintes topologiques
   - Le nombre de Betti de K3 est b₂ = 22
   - Le matching pourrait forcer une relation avec H*

3. **Valider numériquement**
   - Construire des TCS avec différentes longueurs de cou
   - Vérifier le comportement asymptotique

### Questions Ouvertes

1. **Peut-on calculer C₁ (coefficient de la première valeur propre) ?**
2. **Comment les nombres de Betti (b₂, b₃) de M₇ contraignent-ils C₁ ?**
3. **Est-ce que C₁ = 14 indépendamment des choix dans la construction ?**

---

## Références Additionnelles

- [Edge Operators in Geometry](https://link.springer.com/chapter/10.1007/978-3-663-11577-9_13) (Mazzeo)
- [Pseudodifferential operators on manifolds with fibred boundary](https://link.springer.com/article/10.1007/s002090050068) (Mazzeo-Melrose)
- [Gluing and moduli for noncompact geometric problems](https://www.cambridge.org/core/books/abs/geometric-theory-of-singular-phenomena-in-partial-differential-equations/gluing-and-moduli-for-noncompact-geometric-problems/0EE073A2D7B1C8F8F6A1F5A5A7B6C1D0) (Mazzeo-Pollack)

---

## Prochaine Étape Recommandée

Lire en détail la Section 4 de Hassell-Mazzeo-Melrose (1995) sur l'**accumulation des valeurs propres** et voir si leur formule asymptotique peut être spécialisée au cas G₂.
