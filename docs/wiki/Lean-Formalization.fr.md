---
title: "Formalisation Lean"
layout: default
---

# Formalisation Lean

## Vue d'ensemble

Le cadre GIFT est formellement vérifié en **Lean 4** avec Mathlib. La formalisation établit que toutes les relations algébriques revendiquées entre les entrées topologiques et les prédictions physiques suivent par pur calcul.

| Métrique | Valeur |
|----------|--------|
| **Fichiers Lean** | 143 |
| **Jobs de build** | 8391 |
| **Axiomes** | 15 (4 chaîne principale + 11 certificats d'intervalle K3) |
| **Énoncés `sorry`** | 0 |
| **Avertissements** | 0 |
| **Conjonctions du certificat** | 140 (à travers les piliers Foundations / Predictions / Spectral) |

## Dépôt

**Code** : [github.com/Arithmon/K7-Lean](https://github.com/Arithmon/K7-Lean)
**Blueprint** : [arithmon.github.io/K7-Lean](https://arithmon.github.io/K7-Lean/)
**Version Lean** : 4.29.0
**Version Mathlib** : 4.29.0

## Architecture

```
K7-Lean/Lean/GIFT/
├── Core.lean              # constantes (dim_E8, b2, b3, H*, ...)
├── Certificate.lean       # théorème maître (127 conjonctions)
├── Foundations/            # racines E8, produit vectoriel G2
├── Geometry/               # géométrie différentielle
├── Spectral/               # théorie spectrale, mass gap, Yukawa
├── Relations/              # prédictions physiques
│   ├── GaugeSector.lean
│   ├── NeutrinoSector.lean
│   ├── LeptonSector.lean
│   ├── YukawaDuality.lean
│   └── ...
├── ExplicitG2Metric.lean   # 400 lignes
├── NewtonKantorovich.lean   # 401 lignes, 0 axiome
├── K3Harmonic.lean          # 447 lignes
├── K7Orthonormality.lean    # 0 axiome, 13 théorèmes
├── TCSGaugeBreaking.lean    # 0 axiome, 14 théorèmes
├── GaugeBundleData.lean     # 0 axiome, 12 théorèmes
├── AssociativeVolumes.lean  # 0 axiome, 19 théorèmes
├── CompactificationCorrection.lean  # δ_CP = 12214/69
└── ComputedWeylLaw.lean     # 0 axiome, 8 théorèmes
```

## Certificats clés

### Certificat maître (127 conjonctions)

Le théorème `GIFT_framework_certified` vérifie toutes les relations en une seule compilation :

- **Foundations** (34) : dimensions de E₈, structure G₂, topologie de K₇, groupe de Weyl
- **Predictions** (56) : jauge, leptons, quarks, neutrinos, CKM, bosons, cosmologie
- **Spectral** (37) : mass gap, rapports de Yukawa, loi de Weyl, orthonormalité

### Modules zéro-axiome

Plusieurs modules prouvent leurs résultats avec **zéro axiome spécifique au domaine** :

| Module | Axiomes | Théorèmes | Certificat |
|--------|---------|-----------|------------|
| NewtonKantorovich | 0 | | convergence NK |
| TCSGaugeBreaking | 0 | 14 | 10 conjonctions |
| GaugeBundleData | 0 | 12 | 11 conjonctions |
| AssociativeVolumes | 0 | 19 | 14 conjonctions |
| CompactificationCorrection | 0 | 12 | 6 conjonctions |
| ComputedWeylLaw | 0 | 8 | 7 conjonctions |
| K7Orthonormality | 0 | 13 | matrices de Gram |

## Audit des axiomes

Tous les 7 axiomes sont substantiels (théorèmes mathématiques standards ou conjectures GIFT). Aucun n'est un placeholder. Catégories :

- **Entrées topologiques** : b₂ = 21, b₃ = 77, dim(G₂) = 14, etc.
- **Propriétés de E₈** : système de racines, factorisation du groupe de Weyl
- **Entrée Joyce/anisotrope** : l'existence compacte d'une métrique `G_2` sans
  torsion sur `K_7` est une hypothèse/cible analytique, suivie dans
  `docs/analytic_status.md`
- **Bornes spectrales** : résultats numériques de calcul certifié

## Vérification

Pour vérifier toutes les preuves localement :

```bash
git clone https://github.com/Arithmon/K7-Lean
cd K7-Lean/Lean
lake build
```

Se termine en environ 30 secondes (cache chaud). La CI reconstruit à chaque commit.

## Ce qui est prouvé vs ce qui ne l'est pas

**Prouvé** : étant donné les entrées topologiques (b₂, b₃, dim(G₂), etc.), toutes les 127 relations algébriques suivent par pur calcul. Zéro axiome spécifique au domaine pour l'arithmétique.

**Non prouvé** : (1) l'existence de K₇ avec ces nombres de Betti spécifiques est axiomatisée, pas construite à partir de zéro. (2) L'interprétation physique des relations. (3) L'unicité de la construction.

---

## Liens connexes

- [Pour les experts en formalisation](For-Formalization-Experts.fr.html) : méthodologie et contexte
- [Article principal](Paper-Main-Framework.html) : prédictions physiques que ces preuves certifient
- [Article S2 dérivations](Paper-S2-Derivations.html) : les dérivations qui sont formalisées
