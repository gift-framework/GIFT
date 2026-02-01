# Théorie de Hodge sur les Variétés G₂

**Relevance pour GIFT**: Les nombres de Betti b₂, b₃ qui apparaissent dans H* = b₂ + b₃ + 1 sont liés aux formes harmoniques. Comprendre la décomposition de Hodge pour G₂ est essentiel.

---

## Décomposition des Formes sous G₂

### Action de G₂ sur les k-formes
Le groupe G₂ ⊂ SO(7) agit sur l'espace Λᵏ(ℝ⁷)* des k-formes.
Cette action décompose Λᵏ en **représentations irréductibles** de G₂.

### Décomposition Explicite

| k | dim Λᵏ | Décomposition G₂ |
|---|--------|------------------|
| 0 | 1 | **1** |
| 1 | 7 | **7** |
| 2 | 21 | **7** ⊕ **14** |
| 3 | 35 | **1** ⊕ **7** ⊕ **27** |
| 4 | 35 | **1** ⊕ **7** ⊕ **27** |
| 5 | 21 | **7** ⊕ **14** |
| 6 | 7 | **7** |
| 7 | 1 | **1** |

### Notation Standard
- **1** = représentation triviale
- **7** = représentation standard (vecteurs)
- **14** = représentation adjointe (algèbre de Lie g₂)
- **27** = représentation "tensorielle"

---

## La 3-forme φ et la 4-forme ψ

### 3-forme Associative φ
```
φ ∈ Ω³_1(M)    (composante triviale des 3-formes)
```

La 3-forme φ définit la structure G₂ :
- Fixée par l'holonomie G₂
- Parallèle : ∇φ = 0 (si torsion = 0)
- Encode la métrique : g_ij = (1/6) Σ_{k,l} φ_ikl φ_jkl

### 4-forme Coassociative ψ
```
ψ = *φ ∈ Ω⁴_1(M)
```

où * est l'étoile de Hodge.

---

## Nombres de Betti pour G₂

### Formules de Hodge
Pour une variété G₂ compacte M⁷ avec holonomie exactement G₂ :

```
b₀ = 1              (fonction constante)
b₁ = 0              (pas de 1-formes harmoniques)
b₂ = dim H²(M)      (variable selon la construction)
b₃ = b₄             (dualité de Poincaré)
b₅ = b₂             (dualité)
b₆ = 0
b₇ = 1
```

### Décomposition des Formes Harmoniques
Les formes harmoniques se décomposent sous G₂ :

```
H²(M) ⊂ Ω²_7 ⊕ Ω²_14
H³(M) ⊂ Ω³_1 ⊕ Ω³_7 ⊕ Ω³_27
```

Pour holonomie **exactement** G₂ :
- dim H³_1 = 1 (juste [φ])
- dim H³_7 = 0
- dim H³_27 = b₃ - 1

---

## Laplacien de Hodge sur les k-formes

### Définition
Le Laplacien de Hodge-de Rham :
```
Δₖ = dd* + d*d : Ωᵏ(M) → Ωᵏ(M)
```

### Formule de Weitzenböck
```
Δₖ = ∇*∇ + R(ω)
```

où R est l'opérateur de courbure de Weitzenböck.

### Cas G₂ (Ricci-plat)
Pour les variétés G₂ (Ric = 0), le terme R simplifie :
```
Δ₀ = ∇*∇                   (Laplacien scalaire)
Δ₁ = ∇*∇                   (sur 1-formes, car Ric = 0)
Δ₂ = ∇*∇ + termes en Riem  (non trivial)
```

---

## Spectre du Laplacien et Topologie

### Théorème de Hodge
```
ker(Δₖ) ≅ Hᵏ(M, ℝ)
```

Le noyau du Laplacien sur les k-formes est isomorphe à la cohomologie.

### Implication pour GIFT
Les nombres de Betti b₂, b₃ sont :
```
b₂ = dim ker(Δ₂)
b₃ = dim ker(Δ₃)
```

La question est : **les valeurs propres non-nulles dépendent-elles aussi des bₖ ?**

---

## Hypothèse GIFT

### Formule du Gap Spectral
GIFT propose :
```
λ₁(Δ₀) = 14 / (b₂ + b₃ + 1) = 14 / H*
```

### Interprétation Hodge-théorique

Le dénominateur H* = b₂ + b₃ + 1 peut s'écrire :
```
H* = dim H²(M) + dim H³(M) + 1
   = (nombre total de "cycles" de degré moyen) + 1
```

### Question Clé
**Existe-t-il une formule de type Cheeger/Lichnerowicz qui lie λ₁ aux dimensions des espaces de cohomologie ?**

Pour les variétés de courbure positive, Lichnerowicz donne :
```
λ₁ ≥ n/(n-1) · Ric_min
```

Mais pour G₂, Ric = 0 donc cette borne est triviale (λ₁ ≥ 0).

---

## Pistes pour une Borne Spectrale

### Borne de Cheeger
```
λ₁ ≥ h²/4
```

où h = constante de Cheeger (isopérimétrique).

Pour G₂, il faudrait relier h aux Betti numbers.

### Borne via Diamètre
```
λ₁ ≥ C/diam(M)²
```

Pour G₂, comment diam(M) dépend-il de H* ?

### Borne via Volume
```
λ₁ × Vol^{2/7} = ?
```

Est-ce que cette quantité est universelle pour G₂ ?

---

## Cas Spécial : Variétés de Joyce

### Nombres de Betti Typiques

| Variété | b₂ | b₃ | H* |
|---------|----|----|-----|
| J1 | 12 | 43 | 56 |
| J2 | 8 | 47 | 56 |
| J3 | 0 | 71 | 72 |
| K₇ (GIFT) | 21 | 77 | 99 |

### Observation
Les variétés avec même H* mais différents (b₂, b₃) ont (selon GIFT) le même λ₁.
C'est la **split-independence** confirmée numériquement.

---

## Références

1. [G₂ manifold - Wikipedia](https://en.wikipedia.org/wiki/G2_manifold)

2. [The holonomy group G₂ - Joyce Handout](https://people.maths.ox.ac.uk/joyce/G2Handout.pdf)

3. [Compact Riemannian 7-manifolds with holonomy G₂](https://projecteuclid.org/journals/journal-of-differential-geometry/volume-43/issue-2/Compact-Riemannian-7-manifolds-with-holonomy-G_2-I/10.4310/jdg/1214458109.full) (Joyce 1996)

4. [LAPLACIAN FLOW FOR CLOSED G₂ STRUCTURES](http://www.homepages.ucl.ac.uk/~ucahjdl/YWei_LaplacianFlow.pdf) (Wei)

5. [On the formality of nearly Kähler manifolds and Joyce's examples](https://arxiv.org/html/2012.10915)

---

## Prochaine Étape Recommandée

**Chercher dans la littérature** s'il existe des bornes spectrales qui relient λ₁ aux nombres de Betti pour des variétés à holonomie spéciale. Mots-clés : "spectral bounds Betti numbers special holonomy".
