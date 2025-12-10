# GIFT Publications Restructuring Plan v3.0

## Objectif

Refactoriser les publications markdown pour distinguer clairement :
1. **Zenodo-ready** : Prédictions dimensionless rigoureusement prouvées
2. **Repo-only** : Contenu exploratoire/spéculatif (transparent mais non publié)

---

## Nouvelle Structure Proposée

```
publications/
├── zenodo/                          # Publications officielles Zenodo
│   ├── GIFT_v3_main.md              # Paper principal condensé (~25-30 pages)
│   ├── GIFT_v3_S1_foundations.md    # Fusion S1+S2 : Architecture + K₇
│   └── GIFT_v3_S2_derivations.md    # S4 refait : Dérivations béton uniquement
│
├── markdown/                        # Archive versions (garder pour historique)
│   └── [fichiers existants v23/v30]
│
├── exploratory/                     # Contenu spéculatif (repo uniquement)
│   ├── dimensional_observables.md   # Ex-S7 : Masses absolues (heuristique)
│   ├── sequences_prime_atlas.md     # Ex-S8 : Fibonacci, primes
│   ├── monster_moonshine.md         # Ex-S9 : Monster, McKay
│   └── theoretical_extensions.md    # Ex-S6 : Quantum gravity, etc.
│
├── references/                      # Garde structure actuelle
└── Lean/                            # Garde structure actuelle
```

---

## PARTIE 1 : Contenu Zenodo (3 fichiers)

### 1.1 GIFT_v3_main.md (~25-30 pages)

**Structure proposée :**

```markdown
# GIFT v3.0 : Topological Determination of Standard Model Parameters

## Abstract (court, focus dimensionless)

## 1. Introduction
- Le problème des 19 paramètres
- Approche géométrique : E₈×E₈ → K₇ → SM

## 2. Framework Structure (condensé)
- E₈×E₈ gauge structure (dim=496)
- K₇ manifold : b₂=21, b₃=77, H*=99
- G₂ holonomy

## 3. Topological Parameters (PROVEN)
- τ = 3472/891
- det(g) = 65/32
- κ_T = 1/61
- N_gen = 3

## 4. Dimensionless Predictions (PROVEN)
### 4.1 Gauge Sector
- sin²θ_W = 3/13
- α_s = √2/12

### 4.2 Lepton Sector
- Q_Koide = 2/3
- m_τ/m_e = 3477
- m_μ/m_e = 27^φ

### 4.3 Quark Sector
- m_s/m_d = 20

### 4.4 Neutrino Sector
- δ_CP = 197°
- θ₁₃ = π/21
- θ₂₃ = 85/99 rad

### 4.5 Higgs Sector
- λ_H = √17/32

### 4.6 Cosmology (dimensionless ratios)
- Ω_DE = ln(2) × 98/99
- n_s = ζ(11)/ζ(5)

## 5. Lean/Coq Verification Summary
- 75 relations originales
- Modules et certificats

## 6. Experimental Tests
- DUNE (δ_CP, 2027-2030)
- Lattice QCD (m_s/m_d, 2030)
- FCC-ee (sin²θ_W, 2040s)

## 7. Conclusion

## Appendix A : Notation
## Appendix B : Liste des 15-20 relations PROVEN
```

**Ce qu'on RETIRE du main :**
- Fibonacci/Lucas/Prime Atlas (→ exploratory)
- Monster/Moonshine (→ exploratory)
- Masses absolues en GeV/MeV (→ exploratory)
- Hubble tension speculation (→ exploratory)

---

### 1.2 GIFT_v3_S1_foundations.md (Fusion S1 + S2)

**Fusionne :**
- S1_mathematical_architecture_v30.md
- S2_K7_manifold_construction_v30.md

**Structure proposée :**

```markdown
# Supplement S1: Mathematical Foundations

## Part I: E₈ Exceptional Lie Algebra
- Root system, Weyl group
- E₈×E₈ product structure
- Exceptional chain (E₆, E₇, E₈)

## Part II: G₂ Holonomy
- Definition et propriétés
- Torsion conditions

## Part III: K₇ Manifold Construction
- TCS framework (Joyce-Kovalev)
- Building blocks : Quintic + CI(2,2,2)
- Betti numbers : b₂=21, b₃=77

## Part IV: Cohomological Structure
- H* = 99 derivation
- Metric structure
- det(g) = 65/32 proof

## Part V: Torsion Tensor
- κ_T = 1/61 derivation
- Geometric interpretation

## References
```

---

### 1.3 GIFT_v3_S2_derivations.md (S4 refait, DIMENSIONLESS ONLY)

**Base :** S4_complete_derivations_v30.md

**MAIS on retire :**
- Toutes les masses absolues (GeV/MeV)
- Les formules heuristiques type `m_u = √(14/3) × MeV`

**Structure proposée :**

```markdown
# Supplement S2: Complete Derivations (Dimensionless)

## Status Classification
- PROVEN (Lean) : formellement vérifié
- TOPOLOGICAL : conséquence directe de la topologie

## Part I: Foundational Theorems
1. N_gen = 3 (trois preuves)
2. τ = 3472/891 (exact rational)
3. κ_T = 1/61 (torsion)
4. det(g) = 65/32 (metric)

## Part II: Gauge Sector
5. sin²θ_W = 3/13 = b₂/(b₃+dim(G₂))
6. α_s = √2/12
7. α⁻¹ derivation (topologique)

## Part III: Lepton Ratios
8. Q_Koide = 2/3 = dim(G₂)/b₂
9. m_τ/m_e = 3477 (exact integer!)
10. m_μ/m_e = 27^φ

## Part IV: Quark Ratios
11. m_s/m_d = 20 = p₂² × Weyl
12. Autres ratios (derived)

## Part V: Neutrino Sector
13. δ_CP = 197° = 7×14 + 99
14. θ₁₃ = π/21 = π/b₂
15. θ₂₃ = 85/99

## Part VI: Higgs & Cosmology
16. λ_H = √17/32
17. Ω_DE = ln(2) × 98/99
18. n_s = ζ(11)/ζ(5)

## Summary Table: 18 PROVEN Dimensionless Relations
```

---

## PARTIE 2 : Contenu Exploratory (Repo Only)

### 2.1 exploratory/dimensional_observables.md

**Source :** S7_dimensional_observables_v30.md

**Header à ajouter :**
```markdown
# Dimensional Observables (EXPLORATORY)

> **STATUS: EXPLORATORY**
> Ce document contient des formules heuristiques pour les masses absolues.
> Ces formules fonctionnent numériquement mais n'ont pas de justification
> topologique complète. À utiliser avec précaution.

## Limitations
- m_e est un INPUT (non prédit)
- Plusieurs formules mélangent constants topologiques + fitting
- La transition dimensionless → dimensional n'est pas rigoureuse
```

---

### 2.2 exploratory/sequences_prime_atlas.md

**Source :** S8_sequences_prime_atlas_v30.md

**Header à ajouter :**
```markdown
# Sequences and Prime Atlas (EXPLORATORY)

> **STATUS: EXPLORATORY - Pattern Recognition**
> Les embeddings Fibonacci/Lucas et le Prime Atlas sont des patterns
> observés dans les constantes GIFT. Leur signification physique reste
> spéculative. Mathématiquement vérifiés en Lean, mais la connexion
> physique n'est pas établie.
```

---

### 2.3 exploratory/monster_moonshine.md

**Source :** S9_monster_moonshine_v30.md

**Header à ajouter :**
```markdown
# Monster Group and Monstrous Moonshine (EXPLORATORY)

> **STATUS: HIGHLY SPECULATIVE**
> Les connexions Monster-GIFT sont fascinantes mais très spéculatives.
> La factorisation 196883 = 47×59×71 est mathématiquement vraie,
> mais son interprétation physique reste ouverte.
```

---

### 2.4 exploratory/theoretical_extensions.md

**Source :** S6_theoretical_extensions_v30.md

**Header :**
```markdown
# Theoretical Extensions (EXPLORATORY)

> **STATUS: SPECULATIVE**
> Extensions vers quantum gravity, dark matter identity, etc.
> Directions de recherche, non des prédictions établies.
```

---

## PARTIE 3 : Les 18 Relations "BÉTON" (Dimensionless)

### Liste définitive pour Zenodo

| # | Relation | Formule | Valeur | Status |
|---|----------|---------|--------|--------|
| 1 | **N_gen** | Atiyah-Singer | 3 | PROVEN |
| 2 | **τ** | 496×21/(27×99) | 3472/891 | PROVEN |
| 3 | **det(g)** | p₂ + 1/32 | 65/32 | PROVEN |
| 4 | **κ_T** | 1/(b₃-dim(G₂)-p₂) | 1/61 | PROVEN |
| 5 | **sin²θ_W** | b₂/(b₃+dim(G₂)) | 3/13 | PROVEN |
| 6 | **α_s** | √2/(dim(G₂)-p₂) | √2/12 | PROVEN |
| 7 | **Q_Koide** | dim(G₂)/b₂ | 2/3 | PROVEN |
| 8 | **m_τ/m_e** | 7+10×248+10×99 | 3477 | PROVEN |
| 9 | **m_s/m_d** | p₂²×Weyl | 20 | PROVEN |
| 10 | **δ_CP** | dim(K₇)×dim(G₂)+H* | 197° | PROVEN |
| 11 | **θ₁₃** | π/b₂ | π/21 | PROVEN |
| 12 | **θ₂₃** | (rank+b₃)/H* | 85/99 rad | PROVEN |
| 13 | **λ_H** | √(dim(G₂)+N_gen)/2^Weyl | √17/32 | PROVEN |
| 14 | **Ω_DE** | ln(p₂)×(b₂+b₃)/H* | ln(2)×98/99 | PROVEN |
| 15 | **n_s** | ζ(D_bulk)/ζ(Weyl) | ζ(11)/ζ(5) | PROVEN |
| 16 | **m_μ/m_e** | dim(J₃(O))^φ | 27^φ | TOPOLOGICAL |
| 17 | **θ₁₂** | arctan(√(δ/γ)) | 33.42° | TOPOLOGICAL |
| 18 | **α⁻¹** | 128+9+det(g)×κ_T | 137.033 | TOPOLOGICAL |

### Ce qu'on EXCLUT de la liste "béton"

| Relation | Raison d'exclusion |
|----------|-------------------|
| Masses absolues (MeV/GeV) | Requiert m_e comme input + formules heuristiques |
| CKM elements individuels | Dérivés, pas directement topologiques |
| Fibonacci embedding | Pattern recognition, pas physique |
| Prime Atlas | Pattern recognition |
| Monster factorization | Spéculatif |
| Ω_DM = b₂/b₃ | 2.9% déviation, moins solide |
| Hubble ratio | Exploratoire |

---

## PARTIE 4 : Actions pour Claude Code

### Étape 1 : Créer la nouvelle structure

```bash
mkdir -p publications/zenodo
mkdir -p publications/exploratory
```

### Étape 2 : Créer les fichiers Zenodo

1. **GIFT_v3_main.md** : Réécrire depuis gift_3_0_main.md
   - Retirer sections Fibonacci/Lucas/Monster
   - Focus sur les 18 relations béton
   - Condensé ~25-30 pages

2. **GIFT_v3_S1_foundations.md** : Fusionner S1+S2
   - Architecture math + Construction K₇
   - ~15-20 pages

3. **GIFT_v3_S2_derivations.md** : Refaire S4
   - UNIQUEMENT dérivations dimensionless
   - Retirer toutes masses absolues
   - ~20 pages

### Étape 3 : Déplacer vers exploratory

```bash
cp publications/markdown/S6_theoretical_extensions_v30.md \
   publications/exploratory/theoretical_extensions.md

cp publications/markdown/S7_dimensional_observables_v30.md \
   publications/exploratory/dimensional_observables.md

cp publications/markdown/S8_sequences_prime_atlas_v30.md \
   publications/exploratory/sequences_prime_atlas.md

cp publications/markdown/S9_monster_moonshine_v30.md \
   publications/exploratory/monster_moonshine.md
```

### Étape 4 : Ajouter headers "EXPLORATORY"

Chaque fichier dans `exploratory/` doit avoir un header clair indiquant le status spéculatif.

### Étape 5 : Mettre à jour README

Documenter la nouvelle structure et la distinction Zenodo vs Exploratory.

---

## PARTIE 5 : Checklist Finale

- [ ] Créer `publications/zenodo/`
- [ ] Créer `publications/exploratory/`
- [ ] Écrire GIFT_v3_main.md (condensé, 18 relations béton)
- [ ] Fusionner S1+S2 → S1_foundations.md
- [ ] Refaire S4 → S2_derivations.md (dimensionless only)
- [ ] Déplacer S6, S7, S8, S9 vers exploratory/
- [ ] Ajouter headers EXPLORATORY
- [ ] Mettre à jour README principal
- [ ] Mettre à jour publications/README.md

---

## Notes pour l'implémentation

1. **Garder les v23/v30 existants** dans `markdown/` pour l'historique
2. **Ne pas supprimer** - déplacer et réorganiser
3. **Lean/Coq reste inchangé** - le code formel ne bouge pas
4. **Les références restent** dans `references/`

---

*Document créé pour guider la restructuration des publications GIFT v3.0*
*Focus : Séparation claire entre prédictions dimensionless PROVEN et contenu exploratoire*
