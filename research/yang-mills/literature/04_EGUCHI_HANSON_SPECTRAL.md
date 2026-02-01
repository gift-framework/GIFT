# Spectral Theory sur Eguchi-Hanson

**Relevance pour GIFT**: Les singularités ℂ²/ℤ₂ dans les orbifolds de Joyce sont résolues par Eguchi-Hanson. Comprendre le spectre local pourrait donner la brique de base pour λ₁ = 14/H*.

---

## L'Espace Eguchi-Hanson

### Définition
L'espace Eguchi-Hanson (EH) est une variété riemannienne 4D :
- **Topologie** : T*S² (fibré cotangent de la sphère)
- **Métrique** : Ricci-plate, auto-duale
- **Holonomie** : SU(2)
- **Asymptotique** : ALE (Asymptotically Locally Euclidean)

### Métrique Explicite
En coordonnées (r, θ, φ, ψ) :

```
ds² = (1 - a⁴/r⁴)⁻¹ dr² + r²/4 [(1 - a⁴/r⁴)(dψ + cos θ dφ)² + dθ² + sin²θ dφ²]
```

où a > 0 est le paramètre de résolution.

### Résolution de Singularité
EH résout la singularité orbifold ℂ²/ℤ₂ :
- À r = a : le "bolt" S² (sphère de rayon a/2)
- À r → ∞ : approche ℝ⁴/ℤ₂

---

## Équation aux Valeurs Propres

### Paper de Référence
"The eigenvalue equation on the Eguchi-Hanson space" (disponible sur [ResearchGate](https://www.researchgate.net/publication/2103935_The_eigenvalue_equation_on_the_Eguchi-Hanson_space))

### Résultat Principal
L'équation aux valeurs propres du Laplacien scalaire :

```
Δ f = λ f
```

se réduit à une **équation de Heun confluente** :

```
d²u/dz² + p(z) du/dz + q(z) u = 0
```

avec symbole d'Ince [0, 2, 1₂].

### Caractéristiques

1. **Singularité irrégulière à l'infini** → pas de type Fuchsien
2. **Solutions WKB** (Liouville-Green) disponibles en approximation
3. **Valeurs propres discrètes** obtenues par fraction continue (T-fraction)
4. **Phase de diffusion** calculable par monodromie

---

## Connexion avec Pöschl-Teller

### Le Potentiel Pöschl-Teller
Pour certaines réductions de l'équation de Heun, on obtient un potentiel effectif de type Pöschl-Teller :

```
V(x) = -λ(λ-1) / cosh²(x)
```

Ce potentiel est **exactement soluble** avec valeurs propres :

```
E_n = -(λ - n - 1)²    pour n = 0, 1, ..., [λ-1]
```

### Implication pour EH
Si l'équation radiale sur EH se réduit à un problème Pöschl-Teller, alors :
- Les valeurs propres sont **explicites**
- Le gap spectral est **calculable exactement**

---

## Hypothèse de Claude (AI Council)

### Lemme Local Proposé
```
λ₁(ℂ³/ℤ₂, Eguchi-Hanson) = 1/4
```

indépendamment du paramètre de résolution ε.

### Justification Heuristique
1. L'équation se réduit à Sturm-Liouville avec potentiel Pöschl-Teller
2. Pour le bon choix de λ, la première valeur propre est 1/4
3. C'est insensible à ε car la métrique EH est self-similar

### Ce qu'il faudrait vérifier
1. **Réduire explicitement** l'équation de Heun à Pöschl-Teller
2. **Identifier le paramètre λ** correspondant à ℂ³/ℤ₂
3. **Calculer** E₁ = ?

---

## Passage à ℂ³/ℤ₂ (Cas 6D)

### Différence avec ℂ²/ℤ₂
- ℂ²/ℤ₂ → EH en 4D
- ℂ³/ℤ₂ → résolution plus complexe en 6D

### Structure de la Résolution
La résolution de ℂ³/ℤ₂ est un **fibré ALE** :
```
EH → ℂ³/ℤ₂ → ℂ
```

Le spectre sur cette variété 6D combine :
- Spectre de EH (4D)
- Spectre sur la base ℂ (2D)

### Formule Attendue
```
λ_6D = λ_EH + λ_ℂ
```

Si λ_EH = 1/4 et λ_ℂ = 0 (mode constant sur la base), alors :
```
λ₁(ℂ³/ℤ₂) = 1/4
```

---

## Application aux Orbifolds de Joyce

### Construction T⁷/Γ
Joyce construit des variétés G₂ en :
1. Prenant le tore T⁷
2. Quotientant par un groupe Γ (typiquement ℤ₂³)
3. Résolvant les 16 singularités de type ℂ³/ℤ₂

### Contribution de Chaque Singularité
Si chaque singularité résolue contribue λ_local = 1/4 au spectre :

```
λ₁(M₇) = ?
```

La question est : comment ces contributions se **combinent** ?

### Hypothèse de Synchronisation
Les 16 singularités sont liées par la symétrie ℤ₂³.
Leurs modes propres pourraient se **synchroniser** pour donner :

```
λ₁(M₇) = λ_local × (facteur de couplage)
```

Si le facteur = 14/99 / (1/4) = 56/99, cela donnerait λ₁ = 14/99.

---

## Références

1. [Eguchi-Hanson space - Wikipedia](https://en.wikipedia.org/wiki/Eguchi–Hanson_space)

2. [The eigenvalue equation on the Eguchi-Hanson space](https://www.researchgate.net/publication/2103935_The_eigenvalue_equation_on_the_Eguchi-Hanson_space)

3. [A detailed look at the Calabi-Eguchi-Hanson spaces](https://arxiv.org/abs/2201.07295)

4. [Pöschl-Teller potential - Wikipedia](https://en.wikipedia.org/wiki/Pöschl–Teller_potential)

5. [EGUCHI-HANSON SINGULARITIES IN U(2)-INVARIANT RICCI FLOW](https://math.berkeley.edu/~appleton/version2.pdf)

---

## Prochaine Étape Recommandée

**Calculer explicitement** les valeurs propres du Laplacien scalaire sur l'espace EH en utilisant les techniques de :
1. Équation de Heun confluente
2. Approximation WKB
3. Réduction à Pöschl-Teller (si possible)

Puis généraliser à ℂ³/ℤ₂.
