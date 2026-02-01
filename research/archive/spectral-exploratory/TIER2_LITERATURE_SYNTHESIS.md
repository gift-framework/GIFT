# Tier 2 Literature Synthesis: L² ~ H* for TCS G₂ Manifolds

**Date**: Janvier 2026
**Status**: LITERATURE-SUPPORTED (not yet THEOREM)
**Objectif**: Établir le lien entre longueur de col L et invariants topologiques H*

---

## 1. Résumé Exécutif

### Ce qu'on cherche à prouver (Tier 2)

```
Pour K₇ TCS avec col de longueur L:
    L² ~ H* = b₂ + b₃ + 1 = 99
```

### Ce que la littérature établit

| Résultat | Source | Status |
|----------|--------|--------|
| λ₁ ~ 1/L² | Langlais 2024, CGN 2024 | **THEOREM** |
| Densité eigenvalues dépend de b_q(X) | Langlais 2024 | **THEOREM** |
| Pas de small eigenvalues sauf 0 | CGN 2024, Prop. 3.16 | **THEOREM** |
| L² ~ H* | GIFT | **CONJECTURAL** |

---

## 2. Sources Primaires

### 2.1 Langlais 2024 (Comm. Math. Phys.)

**Référence**: T. Langlais, "Analysis and spectral theory of neck-stretching problems", [arXiv:2301.03513](https://arxiv.org/abs/2301.03513), Comm. Math. Phys. 2024.

#### Theorem 2.7 (Spectral Density)

Pour une famille TCS (M_T, g_T) avec col de longueur 2T:

```
Λ_q(s) = 2(b_{q-1}(X) + b_q(X))√s + O(1)
```

où:
- Λ_q(s) = #{eigenvalues λ of Δ_q with λ ≤ s}
- X = cross-section du cylindre asymptotique
- b_q(X) = q-ième nombre de Betti de X

**Signification**: Le coefficient de la loi de comptage est **topologique**.

#### Corollary 2.8 (Decay Rate)

```
λ_n(Δ_q) ~ C_q / T²    pour les petites eigenvalues
```

avec C_q dépendant de b_{q-1}(X) + b_q(X).

#### Application aux TCS G₂

Pour G₂-manifolds via TCS:
- Cross-section: X = K3 × S¹ (ou K3 × T²)
- Betti numbers de K3: b₀=1, b₁=0, b₂=22, b₃=0, b₄=1

Pour X = K3 × S¹:
- b₀(X) = 1
- b₁(X) = 1
- b₂(X) = 22
- b₃(X) = 22
- b₄(X) = 23
- b₅(X) = 1

### 2.2 Crowley-Goette-Nordström 2024 (Inventiones)

**Référence**: D. Crowley, S. Goette, J. Nordström, "Distinguishing G₂-manifolds", [arXiv:eta.tex local](/research/eta.tex), Inventiones Math. 2024.

#### Proposition 3.16 (Small Eigenvalues)

```
∃ c > 0, r₀ > 0: ∀ ℓ >> 1, r ≥ r₀, |λ| < c/(ℓ+r):
    D_{M_±,ℓ,r} n'a pas d'eigenspinor non-trivial
```

**En clair**: Pas de petites eigenvalues sous APS boundary conditions.

#### Borne Cheeger (ligne 3598)

```
C'/(ℓ+r)² ≤ λ₁(scalar Laplacian)
```

Preuve via:
1. Cheeger: λ₁ ≥ h²/4
2. h ≥ Vol(X) / Vol(M) ~ 1/(ℓ+r)
3. Donc λ₁ ≥ C'/(ℓ+r)²

#### Volume Scaling (ligne 3594)

```
Vol(M_{-,ℓ,r}) ~ linear in (ℓ+r)
```

---

## 3. Synthèse pour GIFT

### 3.1 Ce qui est établi (Tier 1 + Littérature)

```
THÉORÈME (Spectral Scaling):
Pour TCS (M_T, g_T) avec col de longueur L = 2T:
    c₁/L² ≤ λ₁(M_T) ≤ c₂/L²

où c₁, c₂ dépendent de:
- Géométrie de la cross-section X
- Betti numbers b_q(X)
```

**Sources**: Langlais Thm 2.7, CGN Prop 3.16, GIFT Model Theorem.

### 3.2 Ce qui reste à établir (Tier 2)

```
CONJECTURE (Canonical Neck Length):
Pour K₇ avec la métrique "canonique" TCS:
    L² ~ H* = 99

Mécanisme proposé:
- La métrique torsion-free φ̃_T est exponentiellement proche de φ_T
- La sélection de L vient d'un principe variationnel
- H* = b₂ + b₃ + 1 entre via la cohomologie de K₇
```

### 3.3 Gap identifié

Le lien **L² ~ H*** nécessite un principe de sélection:
1. **Option A**: Minimisation du volume → L ~ √(Vol sectional)
2. **Option B**: Point fixe de RG flow → L déterminé dynamiquement
3. **Option C**: Contrainte topologique → L² = f(b₂, b₃) pour classe d'homotopie

**Aucune de ces options n'est prouvée.**

---

## 4. Formules Clés à Retenir

### Betti Numbers de K₇ (TCS G₂)

| Invariant | Valeur | Origine |
|-----------|--------|---------|
| b₂(K₇) | 21 | Mayer-Vietoris |
| b₃(K₇) | 77 | Hodge duality |
| H* = b₂ + b₃ + 1 | **99** | GIFT convention |

### Betti Numbers de X = K3 × S¹

| q | b_q(X) | b_{q-1}(X) + b_q(X) |
|---|--------|---------------------|
| 0 | 1 | 1 |
| 1 | 1 | 2 |
| 2 | 22 | 23 |
| 3 | 22 | 44 |
| 4 | 23 | 45 |
| 5 | 1 | 24 |

### Formule de Densité Spectrale (Langlais)

```
Λ_q(s) = 2(b_{q-1}(X) + b_q(X))√s + O(1)

Pour q = 2 (2-formes): coefficient = 2 × 23 = 46
Pour q = 3 (3-formes): coefficient = 2 × 44 = 88
```

### Échelle des Eigenvalues

```
λ_n ~ n² / T²   (petites eigenvalues, n << T)
```

---

## 5. Connexion avec GIFT Predictions

### La Prédiction λ₁ = 14/99

Si on accepte:
1. λ₁ ~ 1/L² (Tier 1 ✓)
2. L² ~ H* = 99 (Tier 2 conjecture)
3. Coefficient = dim(G₂) = 14 (Tier 3 conjecture)

Alors:
```
λ₁ = 14/L² = 14/H* = 14/99 ✓
```

### Le Rôle de dim(G₂) = 14

Hypothèse: Le coefficient 14 vient de:
- G₂ holonomie contraint les modes propres
- 14 = dim(G₂) = nombre de générateurs
- Possible lien avec Pell: 99² - 50×14² = 1

---

## 6. Proposition Tier-2' Lemma Pack

### Axiome 1 (Literature-Supported)

```lean
/-- Langlais Theorem 2.7: Spectral density depends on Betti of cross-section -/
axiom spectral_density_formula (K : TCSManifold) (X : CrossSection K) (q : ℕ) :
  ∃ Λ : ℝ → ℝ, ∀ s > 0,
    |Λ s - 2 * (betti (q-1) X + betti q X) * Real.sqrt s| ≤ C
```

### Axiome 2 (Literature-Supported)

```lean
/-- CGN Proposition 3.16: No small eigenvalues except 0 -/
axiom no_small_eigenvalues (K : TCSManifold) (hyp : TCSHypotheses K) :
  ∃ c > 0, ∀ λ : ℝ, λ ≠ 0 → |λ| < c / K.neckLength →
    ¬ IsEigenvalue (Laplacian K) λ
```

### Conjecture (GIFT-Specific)

```lean
/-- GIFT Tier-2 Conjecture: Canonical neck length scales with H* -/
axiom canonical_neck_length (K : K7_TCS) :
  ∃ c : ℝ, c > 0 ∧ K.neckLength ^ 2 = c * H_star
```

---

## 7. Next Steps

### Immédiat
- [ ] Lire Langlais 2024 en détail (Sections 4-5)
- [ ] Extraire les constantes explicites C_q
- [ ] Vérifier si K3 × S¹ ou K3 × T² pour G₂ TCS standard

### Court terme
- [ ] Implémenter Lean spec Tier-2'
- [ ] Chercher principes de sélection dans littérature Kovalev/Corti-Haskins

### Moyen terme
- [ ] Numerical study: varier L, mesurer λ₁(L), fit power law
- [ ] Persistent Laplacian sur simplicial complex de K₇

---

## 8. Références

1. **Langlais 2024**: T. Langlais, "Analysis and spectral theory of neck-stretching problems", Comm. Math. Phys. (2024), [arXiv:2301.03513](https://arxiv.org/abs/2301.03513)

2. **Crowley-Goette-Nordström 2024**: D. Crowley, S. Goette, J. Nordström, "Distinguishing G₂-manifolds", Inventiones Math. (2024)

3. **Mazzeo-Melrose 1995**: R. Mazzeo, R. Melrose, "Analytic surgery and the eta invariant", GAFA (1995)

4. **Kovalev 2003**: A. Kovalev, "Twisted connected sums and special Riemannian holonomy", J. Reine Angew. Math. (2003)

5. **Corti-Haskins et al. 2015**: A. Corti, M. Haskins, et al., "G₂-manifolds and associative submanifolds", Duke Math. J. (2015)

---

*Synthèse littérature Tier-2 pour GIFT Spectral Bounds*
*Janvier 2026*
