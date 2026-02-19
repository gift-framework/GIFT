# Atiyah-Patodi-Singer Index Theorem

**Relevance pour GIFT**: Le +1 dans H* = b₂ + b₃ + 1 pourrait venir de l'η-invariant dans la formule d'indice APS.

---

## La Formule Fondamentale

Pour un opérateur de Dirac D sur une variété M à bord ∂M = N :

```
ind(D) = ∫_M Â(M) - (h + η(D_N))/2
```

où :
- **Â(M)** = genre Â (intégrande d'Atiyah-Singer)
- **h** = dim ker(D_N) = dimension du noyau sur le bord
- **η(D_N)** = η-invariant de l'opérateur de Dirac sur le bord

---

## L'η-invariant

### Définition
Pour un opérateur elliptique autoadjoint A sur une variété compacte N :

```
η_A(s) = Σ_{λ≠0} sgn(λ) |λ|^{-s}
```

Cette série converge pour Re(s) > dim(N) et admet un prolongement méromorphe à ℂ.

**L'η-invariant** est :
```
η(A) = η_A(0)
```

### Interprétation
- **Mesure l'asymétrie spectrale** : combien de valeurs propres positives vs négatives
- Si le spectre est symétrique par rapport à 0 : η = 0
- Pour un opérateur de Dirac, η encode l'information topologique/géométrique

---

## Connexion avec Chern-Simons

### Formule de base
```
η(D_A) ≈ CS(A) + termes locaux
```

où CS(A) est l'invariant de Chern-Simons de la connexion A.

### Implication
L'η-invariant peut être vu comme une **version non-perturbative** de l'invariant de Chern-Simons.

---

## Application aux Variétés G₂

### Contexte
Une variété G₂ compacte M⁷ est sans bord, donc APS ne s'applique pas directement.

**MAIS** : Dans les constructions par résolution d'orbifolds (Joyce) ou TCS (Kovalev), on peut voir M⁷ comme limite de variétés à bord.

### Le ν-invariant de Crowley-Nordström

Crowley, Goette et Nordström ont défini un invariant analytique ν̄ pour les variétés G₂ :

```
ν̄(M, g) ∈ ℤ
```

**Propriétés** :
- Localement constant sur l'espace des modules des métriques G₂
- Défini via les η-invariants et les courants de Mathai-Quillen
- Permet de distinguer des composantes connexes de l'espace des modules

**Référence** : [An analytic invariant of G₂ manifolds](https://link.springer.com/article/10.1007/s00222-024-01310-z) (Inventiones 2025)

---

## L'Hypothèse pour GIFT

### Observation
```
H* = dim(G₂) × dim(K₇) + 1 = 14 × 7 + 1 = 99
```

Le **+1** ressemble à la contribution du noyau **h** dans la formule APS.

### Conjecture
Dans la limite où K₇ est construit comme résolution d'un orbifold T⁷/Γ avec 16 singularités ℂ³/ℤ₂ :

1. Chaque singularité contribue un terme à l'η-invariant
2. Ces contributions se synchronisent
3. Le +1 final vient de dim ker(D) = 1 (le spineur parallèle)

### Ce qu'il faudrait vérifier

1. **Calculer η pour ℂ³/ℤ₂** résolu par Eguchi-Hanson
2. **Sommer les 16 contributions** selon la symétrie ℤ₂³
3. **Vérifier que** η_total = 1 (mod 2) ou que h = 1

---

## Formule d'Indice pour Dirac sur G₂

Sur une variété G₂ compacte M⁷ :

```
ind(D) = 0    (car dim M = 7 est impair)
```

Mais le **spectre** de D n'est pas trivial ! Les valeurs propres satisfont :

```
D² ψ = λ ψ    avec λ ≥ 0
```

Et par Lichnerowicz (Ric = 0 pour G₂) :
```
D² = ∇*∇    (pas de terme de courbure)
```

### Implication pour le Gap
Le **gap spectral** de D² (première valeur propre non-nulle) est relié à la géométrie de M⁷, pas juste à sa topologie.

---

## Références

1. [Spectral asymmetry and Riemannian geometry I, II, III](https://www.cambridge.org/core/journals/mathematical-proceedings-of-the-cambridge-philosophical-society/article/abs/spectral-asymmetry-and-riemannian-geometry-iii/7BED1D09DBD103244C1CFF6BC3E6C607) (Atiyah-Patodi-Singer, 1975-1976)

2. [Eta invariants and manifolds with boundary](https://projecteuclid.org/journals/journal-of-differential-geometry/volume-40/issue-2/Eta-invariants-and-manifolds-with-boundary/10.4310/jdg/1214455539.pdf) (Werner Müller, 1994)

3. [η-invariant and Chern-Simons current](https://arxiv.org/pdf/math/0307120) (Zhang, 2003)

4. [An analytic invariant of G₂ manifolds](https://arxiv.org/abs/1505.02734) (Crowley-Goette-Nordström)

---

## Prochaine Étape Recommandée

**Calculer l'η-invariant de l'opérateur de Dirac sur la résolution Eguchi-Hanson de ℂ²/ℤ₂** (cas 4D, plus simple) puis généraliser à ℂ³/ℤ₂ (cas 6D pertinent pour G₂).
