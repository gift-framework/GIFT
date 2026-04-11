---
title: "Article S2 dérivations"
layout: default
---

# Article : S2, dérivations complètes

**Supplément S2 : dérivations complètes (sans dimension), les 33 prédictions sans dimension**

*Brieuc de La Fournière (2026)*
[Texte intégral (markdown)](https://github.com/gift-framework/GIFT/blob/main/publications/papers/markdown/GIFT_v3.3_S2_derivations.md) | [DOI Zenodo : 10.5281/zenodo.18837071](https://doi.org/10.5281/zenodo.18837071)

---

## Résumé

Fournit les dérivations algébriques complètes pour les 33 prédictions sans dimension à partir des invariants topologiques (b₂, b₃, dim(G₂), etc.). 18 relations principales VÉRIFIÉES en Lean 4 ; 15 prédictions étendues utilisent des formules topologiques. Inclut les comptages d'expressions montrant la redondance structurelle.

---

## Résultats clés

### Distribution des écarts

| Plage | Nombre | % |
|-------|--------|---|
| Exact (0 %) | 4 | 22 % |
| < 0,01 % | 3 | 17 % |
| < 0,1 % | 4 | 22 % |
| < 0,5 % | 7 | 39 % |

### Comptage d'expressions (top des observables)

| Observable | # Expressions | Statut |
|------------|---------------|--------|
| Q_Koide = 2/3 | 27 | CANONIQUE |
| N_gen = 3 | 24+ | CANONIQUE |
| sin² θ₁₂ᴾᴹᴺˢ = 4/13 | 21 | CANONIQUE |
| sin² θ_W = 3/13 | 19 | ROBUSTE |
| m_H/m_t = 56/77 | 16 | ROBUSTE |

### Comparaison jauge / holonomie

| Configuration | Écart moyen | Facteur |
|---------------|-------------|---------|
| E₈×E₈ | 0,26 % | 1× (optimal) |
| E₇×E₈ | 8,80 % | 34× pire |
| Holonomie SU(3)/CY | 4,43 % | 17× pire |

---

## Structure des sections

- **Partie 0** : philosophie de la dérivation, entrées vs sorties, ce qui est revendiqué vs non
- **Partie I** : fondations, classification des statuts, notation
- **Partie II** : théorèmes fondationnels: N_gen=3, τ=3472/891, κ_T=1/61, det(g)=65/32
- **Partie III** : secteur de jauge, sin² θ_W=3/13, α_s=√2/12
- **Partie IV** : secteur leptonique: Q_Koide=2/3, m_τ/m_e=3477, m_μ/m_e=27^φ
- **Partie V** : secteur des quarks, m_s/m_d=20, m_b/m_t=1/42, angles CKM
- **Partie VI** : secteur des neutrinos, δ_CP=197°, angles de mélange
- **Partie VII** : Higgs et cosmologie, λ_H=√17/32, Ω_DE, n_s, h, σ₈
- **Partie VIII** : synthèse (18 VÉRIFIÉES + 15 étendues)
- **Partie IX** : catalogue des observables

---

## Liens connexes

- [Article principal](Paper-Main-Framework.html) : article principal
- [Article S1 fondations](Paper-S1-Foundations.html) : fondations mathématiques
- [Référence des observables](Observable-Reference.fr.html) : catalogue complet des observables
- [Formalisation Lean](Lean-Formalization.fr.html) : preuves vérifiées par machine
