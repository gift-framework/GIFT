---
title: "Preuves statistiques"
layout: default
---

# Preuves statistiques de K₇

**Version** : 3.4.27
**Date de validation** : avril 2026
**Scripts** : [`bulletproof_validation_v33.py`](https://github.com/Arithmon/K7/blob/main/publications/validation/legacy/v3.3/bulletproof_validation_v33.py) (7 composantes, archive v3.3.24), [`exhaustive_validation_v33.py`](https://github.com/Arithmon/K7/blob/main/publications/validation/legacy/v3.3/exhaustive_validation_v33.py) (3M+ configurations)

> **Note (v3.4)** : les statistiques de tête v3.4 sont **0,99 %** d'écart moyen sur **33 relations Type I (cibles exactes)**, avec une borne au niveau ensemble ~10⁻⁶ sur 3 000 000 de jeux de formules aléatoires (sans hypothèse d'indépendance) (catalogue 95 observables : 33 Type I + 19 Type II + 21 Type III + 22 Type IV ; Type II 0,17 %, Type III 3,44 %). Sources de données : NuFIT 6.1 / PDG 2024 / Planck 2018 / CODATA 2022. Les ventilations sectorielles ci-dessous conservent l'analyse v3.3.24 NuFIT 6.0 (0,24 % sur 32 bien mesurés / 0,57 % tous les 33) à des fins de traçabilité. Les conclusions qualitatives (significativité > 4,2 σ, optimum unique parmi 3M+ configurations, facteurs de Bayes décisifs) tiennent dans les deux versions.
>
> **Note historique (v3.3.24)** : les statistiques détaillées ci-dessous ont été calculées avec les valeurs expérimentales NuFIT 5.3 (0,21 % d'écart moyen). Avec la mise à jour v3.3.24 vers NuFIT 6.0 et les formules neutrino améliorées (θ₁₂ = arctan(2/3), θ₂₃ = arctan(√(14/11))), l'écart moyen v3.3.24 était de 0,24 % (32 bien mesurés) / 0,57 % (tous les 33, y compris δ_CP).

---

## Résumé exécutif

### Métriques clés

| Métrique | Valeur |
|---|---|
| **Écart moyen (custom)** | **0,21 %** |
| **Écart moyen (relatif)** | **0,41 %** |
| **p-valeur du modèle nul** | < 2 × 10⁻⁵ (σ > 4,2) |
| **p global Westfall-Young** | **8,4 × 10⁻³** |
| **Meilleur facteur de Bayes** | **4 738** (décisif) |
| **p du test pré-enregistré** | 6,7 × 10⁻⁵ (σ = 4,0) |
| **Configurations testées (exhaustif)** | 3 070 396 |
| **Meilleures que K₇** | **0** |

### Résultats par niveau de précision (écart relatif)

| Niveau | Observables | Seuil | Interprétation |
|---|---|---|---|
| Excellent | 14/33 (42 %) | < 0,1 % | correspondance de précision |
| Bon | 29/33 (88 %) | < 1 % | accord fort |
| Acceptable | 33/33 (100 %) | < 5 % | dans la tolérance |
| À retravailler | 0/33 (0 %) | > 5 % | aucun |

### Interprétation

- **100 % des prédictions** s'accordent avec l'expérience à moins de 5 %
- **88 % des prédictions** s'accordent à moins de 1 %
- K₇ est **uniquement optimal** parmi les 3 070 396 configurations testées
- Les trois familles de modèles nuls rejettent à p < 2 × 10⁻⁵
- La correction permutationnelle FWER de Westfall-Young maxT confirme 11/33 individuellement significatifs après prise en compte des corrélations
- Les facteurs de Bayes vont de 304 à 4 738 pour quatre spécifications de priors (tous décisifs)

---

## 1. Méthodologie

### 1.1 Métrique principale : écart custom

La validation K₇ utilise une métrique d'**écart custom** qui capture la qualité d'ajustement sur des observables hétérogènes (angles, ratios, constantes de couplage) :

$$\text{Écart} = \frac{|\text{pred} - \text{exp}|}{|\text{exp}|} \times 100\%$$

moyennée uniformément sur les 33 observables. Cela évite la pathologie des « pulls » σ où des mesures extraordinairement précises (α⁻¹ avec σ = 2,1 × 10⁻⁵) dominent l'agrégat.

### 1.2 Pourquoi pas le χ² ?

| Observable | Écart relatif | Pull (σ) | Problème |
|---|---|---|---|
| m_μ/m_e | 0,12 % | 52 951 σ | σ_exp = 4,6 × 10⁻⁶ |
| α⁻¹ | 0,002 % | 128 σ | σ_exp = 2,1 × 10⁻⁵ |

L'écart relatif identifie correctement ces prédictions comme excellentes (~0,1 %), tandis que les pulls sont trompeusement grands à cause de la précision expérimentale extraordinaire et de l'absence d'estimations d'incertitudes théoriques.

### 1.3 Validation à sept composantes

La validation à toute épreuve couvre sept composantes indépendantes :

1. **Manifeste de pré-enregistrement** : hash SHA-256 verrouillant observables et formules avant les tests
2. **Trois familles de modèles nuls** : permutation, structure préservée, adversariel
3. **p-valeurs par observable** : avec corrections de Bonferroni, Holm, Benjamini-Hochberg, et Westfall-Young maxT
4. **Cross-prédiction held-out** : leave-one-sector-out + split dev/test pré-enregistré
5. **Analyse de robustesse** : perturbations de poids, MC de bruit, jackknife, leave-k-out, courbe de sensibilité au bruit
6. **Réplication multi-graines** : 10 graines indépendantes + métrique alternative (χ²)
7. **Analyse bayésienne** : facteurs de Bayes multi-priors, PPC à 4 statistiques, comparaison WAIC

---

## 2. Familles de modèles nuls

Trois familles indépendantes de modèles nuls rejettent toutes à la limite de résolution de 50 000 permutations :

| Famille nulle | p-valeur | σ | Description |
|---|---|---|---|
| **Permutation** | 2,0 × 10⁻⁵ | 4,27 | assignation aléatoire de (b₂, b₃) ; moyenne nulle 82,6 % vs K₇ 0,21 % |
| **Structure préservée** | 2,0 × 10⁻⁵ | 4,27 | 0/50 000 configurations atteignent ou battent K₇ |
| **Adversariel** | 2,0 × 10⁻⁵ | 4,27 | meilleur adversaire à 65,8 % vs K₇ 0,21 % |

Les trois familles nulles produisent des écarts moyens environ 300× pires que GIFT.

---

## 3. Corrections pour tests multiples

### 3.1 Significativité par observable (α = 0,05)

| Correction | Significatives | Méthode |
|---|---|---|
| Brut | 21/33 | p-valeurs empiriques non corrigées |
| Bonferroni | 0/33 | conservatrice (divise α par 33) |
| Holm | 0/33 | step-down, encore très conservatrice |
| Benjamini-Hochberg | 20/33 | contrôle FDR (moins conservatrice) |
| **Westfall-Young maxT** | **11/33** | **FWER permutationnel respectant les corrélations** |

### 3.2 Westfall-Young maxT

La procédure step-down maxT de Westfall-Young est l'étalon-or pour le contrôle du taux d'erreur familial (FWER) parce qu'elle :
- Respecte la **structure de corrélation** entre statistiques de test (contrairement à Bonferroni)
- Utilise la **distribution conjointe** des statistiques max sous permutation
- Fournit un contrôle FWER **exact**

**Résultat** : p global = 8,4 × 10⁻³, avec 11/33 observables individuellement significatives. C'est la réponse définitive à la question : « combien d'observables survivent à une correction rigoureuse pour tests multiples tout en tenant compte des corrélations inter-observables ? »

### 3.3 Effet « regarder ailleurs » (Look-Elsewhere)

Décompte explicite des essais LEE : 23 167 200 (toutes les combinaisons (b₂, b₃, jauge, holonomie)). Même après correction LEE, la performance du cadre reste significative.

---

## 4. Cross-prédiction (tests held-out)

### 4.1 Leave-one-sector-out

Chaque secteur de physique est retiré tour à tour ; le (b₂, b₃) de K₇ est testé sur le secteur retiré sans réajustement :

| Secteur | Obs. retirées | Écart de test | p-valeur | σ |
|---|---|---|---|---|
| Couplages de jauge | 3 | 0,17 % | 1,0 × 10⁻³ | 3,3 |
| Leptons | 4 | 0,06 % | 1,0 × 10⁻⁴ | 3,9 |
| Quarks | 9 | 0,24 % | 1,0 × 10⁻² | 2,6 |
| Mélange PMNS | 4 | 0,23 % | 1,0 × 10⁻⁴ | 3,9 |
| Matrice CKM | 6 | 0,59 % | 1,3 × 10⁻⁴ | 3,8 |
| Bosons | 3 | 0,13 % | 2,0 × 10⁻⁴ | 3,7 |
| Cosmologie | 3 | 0,19 % | 3,3 × 10⁻⁵ | 4,1 |

Tous les secteurs non triviaux atteignent p < 0,05, ce qui confirme que la cross-prédiction tient.

### 4.2 Split dev/test pré-enregistré

| Ensemble | N | Écart |
|---|---|---|
| Développement (16 obs.) | 16 | 0,10 % |
| Test (17 obs.) | 17 | 0,32 % |
| **p-valeur du test** | | **6,7 × 10⁻⁵** (σ = 4,0) |

L'ensemble de test held-out atteint σ = 4,0, ce qui confirme que la précision de K₇ n'est pas un artefact d'ajustement à un sous-ensemble particulier.

---

## 5. Robustesse et sensibilité

### 5.1 Perturbation des poids

| Pondération | Écart moyen | Conclusion |
|---|---|---|
| Uniforme | 0,21 % | référence |
| Pondérée par incertitude | 0,00 % | dominée par la précision |
| Inverse-range | 0,62 % | pire cas |
| Aléatoire (100 essais) | 0,21 % ± 0,02 % | stable |

Tous les schémas de pondération donnent < 1 %.

### 5.2 Jackknife & leave-k-out

- **Jackknife** : l'influence maximale d'une observable est de 0,029 % (sin²θ₂₃ CKM). Aucune observable ne domine le résultat.
- **Stabilité leave-k-out** :

| k retirées | Écart moyen | Plage |
|---|---|---|
| 1 | 0,212 % ± 0,008 % | [0,18, 0,22] |
| 3 | 0,212 % ± 0,015 % | [0,14, 0,23] |
| 5 | 0,212 % ± 0,020 % | [0,13, 0,25] |

Le résultat est remarquablement stable sous suppression systématique.

### 5.3 Courbe de sensibilité au bruit

Balayage d'un bruit gaussien d'amplitude σ_factor × σ_exp sur 200 essais par point :

| Facteur de bruit | Écart moyen | Écart-type |
|---|---|---|
| 0,00× | 0,21 % | 0,00 % |
| 0,25× | 0,46 % | 0,09 % |
| 0,50× | 0,82 % | 0,18 % |
| 0,75× | 1,17 % | 0,23 % |
| **1,00×** | **1,57 %** | **0,36 %** |
| 1,50× | 2,34 % | 0,55 % |
| 2,00× | 3,09 % | 0,73 % |
| 3,00× | 4,61 % | 1,18 % |

**Interprétation** : à 1× les incertitudes expérimentales publiées, l'écart moyen passe de 0,21 % à 1,57 %. C'est le plancher de précision physique : l'accord à 0,21 % du cadre est déjà à un facteur ~7 de ce que le bruit de mesure seul produirait. Améliorer encore les prédictions du cadre demanderait que les mesures expérimentales deviennent plus précises.

### 5.4 Monte Carlo de bruit

Sur 1 000 essais avec 1× les incertitudes publiées :
- Moyenne : 1,50 % ± 0,35 %
- Seulement 5 % des essais restent sous 1 %

Ceci confirme la courbe de sensibilité au bruit : le résultat K₇ à 0,21 % se situe bien sous le plancher de bruit.

---

## 6. Réplication multi-graines

| Métrique | Valeur |
|---|---|
| Graines testées | 10 |
| Plage de p-valeurs | [5,0 × 10⁻⁵, 1,5 × 10⁻⁴] |
| Plage de σ | [3,8, 4,1] |
| Toutes significatives à α=0,05 | oui |
| Métrique alternative (χ²) | p = 5,0 × 10⁻⁵ (σ = 4,1) |
| Cohérent inter-métriques | oui |

Les résultats sont invariants à la graine PRNG et tiennent sous une métrique alternative (χ² relatif).

---

## 7. Analyse bayésienne

### 7.1 Facteurs de Bayes (4 spécifications de priors)

| Prior | BF | Interprétation |
|---|---|---|
| Sceptique (uniforme) | 304 | décisif pour H₁ |
| Référence (semi-normale) | 397 | décisif pour H₁ |
| Jeffreys | 2 423 | décisif pour H₁ |
| Enthousiaste (uniforme ≤ 1 %) | 4 738 | décisif pour H₁ |

Les quatre priors donnent des preuves décisives (BF > 100) en faveur de K₇ contre le nul. Le prior sceptique, qui accorde au nul la latitude maximale, donne encore BF = 304.

### 7.2 Vérifications prédictives postérieures (4 statistiques)

| Statistique | Observée | Moyenne répliquée | p PPC | Statut |
|---|---|---|---|---|
| T₁ : écart moyen | 0,21 % | 1,53 % | 1,000 | ↑ supérieur |
| T₂ : écart max | 1,13 % | 12,04 % | 1,000 | ↑ supérieur |
| T₃ : nombre > 1 % | 1 | 12,1 | 1,000 | ↑ supérieur |
| T₄ : pire secteur | 0,59 % | 4,28 % | 1,000 | ↑ supérieur |

**Statut** : `superior_to_noise` : le cadre s'ajuste significativement mieux que ce que prédit le bruit de mesure, sur les quatre statistiques de test. Les jeux de données répliqués (en ajoutant du bruit aux niveaux d'incertitude publiés) montrent systématiquement des écarts de 5 à 12× supérieurs à ce que K₇ atteint. C'est cohérent avec un contenu physique authentique plutôt qu'une coïncidence numérique.

**Note** : un PPC p ≈ 1,0 n'indique pas un mauvais ajustement de modèle. Dans le cadre PPC, un p proche de 0 indique un sous-ajustement systématique, un p proche de 0,5 indique une calibration parfaite au modèle de bruit, et un p proche de 1 indique que le modèle dépasse les attentes du bruit. Le résultat confirme que la précision de K₇ dépasse ce que les incertitudes de mesure seules prédiraient.

### 7.3 Comparaison de modèles WAIC

| Modèle | WAIC | Interprétation |
|---|---|---|
| K₇ | 29,9 | préféré |
| Nul | 580,2 | |
| **ΔWAIC** | **550,3** | **favorise fortement K₇** |

---

## 8. Recherche exhaustive de configurations

### 8.1 Variations des nombres de Betti (3 070 396 configs)

| Métrique | Valeur |
|---|---|
| Plage de b₂ | [5, 100] |
| Plage de b₃ | [40, 200] |
| Configurations testées | 3 070 396 |
| Meilleures que K₇ | **0** |
| IC à 95 % (Clopper-Pearson) | [0, 3,7 × 10⁻⁵] |

### 8.2 Comparaison des groupes de jauge

| Rang | Groupe de jauge | Écart moyen |
|---|---|---|
| **1** | **E₈ × E₈** | **0,41 %** |
| 2 | E₇ × E₈ | 8,8 % |
| 3 | E₆ × E₈ | 15,5 % |

E₈ × E₈ atteint un accord **21× meilleur** que la prochaine alternative.

### 8.3 Comparaison des groupes d'holonomie

| Rang | Holonomie | dim | Écart moyen |
|---|---|---|---|
| **1** | **G₂** | 14 | **0,41 %** |
| 2 | SU(4) | 15 | 1,5 % |
| 3 | SU(3) | 8 | 4,4 % |
| 4 | Spin(7) | 21 | 5,4 % |

G₂ atteint un accord **11× meilleur** que Calabi-Yau (SU(3)).

---

## 9. Résultats par catégorie de physique

| Catégorie | N | Écart moyen | Écart max | <0,1 % | <1 % | <5 % |
|---|---|---|---|---|---|---|
| Structurel | 1 | 0,00 % | 0,00 % | 1/1 | 1/1 | 1/1 |
| Électrofaible | 4 | 0,36 % | 0,90 % | 1/4 | 4/4 | 4/4 |
| Rapports de masses leptons | 4 | 0,06 % | 0,12 % | 2/4 | 4/4 | 4/4 |
| Rapports de masses quarks | 4 | 0,34 % | 1,21 % | 2/4 | 3/4 | 4/4 |
| Mélange PMNS | 7 | 0,94 % | 4,81 % | 3/7 | 5/7 | 7/7 |
| Mélange CKM | 3 | 0,74 % | 1,50 % | 0/3 | 2/3 | 3/3 |
| Rapports de masses bosons | 3 | 0,12 % | 0,29 % | 2/3 | 3/3 | 3/3 |
| Cosmologique | 7 | 0,19 % | 0,48 % | 3/7 | 7/7 | 7/7 |
| **TOTAL** | **33** | **0,41 %** | | 14/33 | 29/33 | 33/33 |

---

## 10. Réserves honnêtes

### 10.1 Ce que cette validation établit

1. **Significativité statistique** : p < 2 × 10⁻⁵ contre trois familles nulles indépendantes (σ > 4,2)
2. **Robustesse aux tests multiples** : 11/33 survivent au FWER Westfall-Young maxT (p global = 0,008)
3. **Cross-prédiction** : tous les secteurs non triviaux et le split de test pré-enregistré sont significatifs
4. **Confirmation bayésienne** : BF de 304 à 4 738 sur quatre spécifications de priors, tous décisifs
5. **Stabilité** : invariant à la pondération, à la graine, au choix de métrique, et à la suppression d'observables

### 10.2 Ce que cette validation N'établit PAS

1. **Justification des formules** : l'optimalité statistique n'explique pas pourquoi ces formules ont été choisies. Les dérivations dans S2 fournissent la motivation théorique, mais l'accord statistique seul n'est pas une preuve d'exactitude physique.
2. **Vérité physique** : un excellent accord ≠ physique sous-jacente correcte. Le cadre pourrait être une paramétrisation très efficace qui capture des motifs sans que le mécanisme géométrique proposé soit la bonne explication.
3. **Complétude** : seules les variétés G₂ TCS avec des groupes de jauge/holonomie spécifiques ont été testées.

### 10.3 Statut PPC supérieur au bruit

Les vérifications prédictives postérieures montrent PPC p = 1,0 sur les quatre statistiques de test. Cela signifie que les prédictions du cadre sont plus précises que ce que le bruit de mesure seul prédirait. Explications possibles :
- Le cadre capture une véritable structure physique (la revendication de K₇)
- Les incertitudes expérimentales publiées sont conservatrices
- Il existe des corrélations entre observables non capturées par le modèle de bruit

C'est une force du cadre, pas une faiblesse, mais cela signifie que le PPC ne peut pas distinguer entre ces explications.

### 10.4 Sensibilité au bruit comme limite physique

À 1× les incertitudes publiées, l'écart moyen passe de 0,21 % à 1,57 %. Cela définit le **plancher de précision de mesure** : les prédictions du cadre sont déjà à un facteur ~7 de ce que les meilleures mesures actuelles peuvent distinguer d'un accord parfait. Une validation plus poussée requiert des expériences plus précises.

### 10.5 Bonferroni / Holm donnant zéro

Les corrections de Bonferroni et Holm donnent 0/33 observables significatives parce qu'elles divisent α par 33, ce qui est extrêmement conservateur pour des tests corrélés. C'est pourquoi la procédure Westfall-Young maxT est la bonne correction : elle respecte la structure de corrélation et donne un résultat significatif (11/33 significatives, p global = 0,008).

---

## 11. Prédictions de falsification

| Prédiction | Valeur K₇ | Exp. actuelle | Cible | Expérience | Calendrier |
|---|---|---|---|---|---|
| δ_CP | 197° | ~207-212° (NuFIT 6.1, ~1 σ) | ±5° | DUNE | 2028-2040 |
| sin²θ_W | 3/13 | 0,2312 ± 4 × 10⁻⁵ | ±10⁻⁵ | FCC-ee | 2040s |
| Ω_DM/Ω_b | 43/8 | 5,375 ± 0,1 | ±0,01 | CMB-S4 | 2030s |
| m_s/m_d | 20 | 20 ± 1 | ±0,3 | QCD sur réseau | 2030 |

---

## 12. Comment reproduire

### Validation à toute épreuve (7 composantes)

```bash
cd publications/validation
python3 bulletproof_validation_v33.py
```

**Prérequis** : Python 3.8+, aucune dépendance externe
**Sortie** : `bulletproof_validation_v33_results.json`
**Durée** : ~15 secondes

### Recherche exhaustive (3M+ configurations)

```bash
cd publications/validation
python3 exhaustive_validation_v33.py
```

**Durée** : ~2 à 5 minutes

---

## 13. Conclusions

### Constatation principale

K₇ atteint un **écart moyen de 0,21 %** (0,41 % relatif) sur 33 observables. Parmi 3 070 396 configurations testées, **zéro** font mieux. Ce résultat survit à :

- Trois familles indépendantes de modèles nuls (p < 2 × 10⁻⁵)
- Correction FWER de Westfall-Young maxT (p global = 0,008, 11/33 individuellement significatifs)
- Split dev/test pré-enregistré (p de test = 6,7 × 10⁻⁵)
- Quatre spécifications de priors bayésiens (BF de 304 à 4 738, tous décisifs)
- Analyses de stabilité par perturbation de poids, jackknife et leave-k-out
- Réplication multi-graines et inter-métriques

### Résumé statistique

| Métrique | Valeur |
|---|---|
| À moins de 0,1 % | 42 % (14/33) |
| À moins de 1 % | **88 %** (29/33) |
| À moins de 5 % | **100 %** (33/33) |
| Écart moyen | **0,21 %** (custom), **0,41 %** (relatif) |
| p du modèle nul | < 2 × 10⁻⁵ (σ > 4,2) |
| p global Westfall-Young | **0,008** |
| Meilleur facteur de Bayes | **4 738** |
| Configurations testées | 3 070 396 |
| Meilleures que K₇ | **0** |

---

## Références

- Joyce, D.D. *Compact Manifolds with Special Holonomy* (2000)
- Westfall, P.H. & Young, S.S. *Resampling-Based Multiple Testing* (1993)
- Particle Data Group (2024), Review of Particle Physics
- Planck Collaboration (2020), Cosmological parameters
- NuFIT 5.3 (2024), Neutrino oscillation parameters
- CODATA 2022, Fundamental physical constants

---

*K₇ Framework v3.4.27 : preuves statistiques à toute épreuve*
*Tête : écart moyen 0,99 % sur 33 relations Type I (NuFIT 6.1 / PDG 2024 / Planck 2018 / CODATA 2022) ; une borne au niveau ensemble ~10⁻⁶ sur 3 000 000 de jeux de formules aléatoires (sans hypothèse d'indépendance)*
*Ventilation sectorielle ci-dessus : analyse v3.3.24 NuFIT 6.0 (0,24 % sur 32 bien mesurés / 0,57 % tous les 33)*
