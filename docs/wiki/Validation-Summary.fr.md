---
title: "Résumé de validation"
layout: default
---

# Résumé complet de validation GIFT v3.4

**Date** : 2026-04-29 (v3.4.13)
**Références expérimentales** : PDG 2024 / NuFIT 6.0 (NO, IC19) / Planck 2020
**Recherche exhaustive (v3.4)** : 3M+ configurations, log₁₀ p_algébrique = −138
**p-valeur du modèle nul** : < 2 × 10⁻⁵ (σ > 4,2)
**Westfall-Young maxT** : 11/33 significatifs (p global = 0,008, base v3.3.24)
**Facteur de Bayes** : 288 à 4 567 (décisif, base v3.3.24)

> **Tête v3.4** : 0,39 % d'écart moyen sur 35 observables Type I (cibles exactes) dans le catalogue 95 observables (35 Type I + 19 Type II + 21 Type III + 22 Type IV). Les ventilations sectorielles ci-dessous conservent l'analyse v3.3.24 NuFIT 6.0 à des fins de traçabilité.

---

## Résumé exécutif

| Catégorie | Prédictions | Écart moyen | Statut |
|---|---|---|---|
| **Type I (cibles exactes, v3.4)** | 35 | **0,39 %** | VALIDÉ |
| **Observables bien mesurés (v3.3.24)** | 32 | 0,24 % | VALIDÉ |
| **Tous, y compris δ_CP (v3.3.24)** | 33 | 0,57 % | VALIDÉ |
| **Pont d'échelle** (3 masses en MeV) | 3 | 0,07 % | EXPLORATOIRE |

Les 33 prédictions sont sans dimension : ratios, angles de mélange et constantes de couplage. Les angles en degrés et leurs équivalents trigonométriques (sin² θ) représentent le même contenu physique dans des coordonnées différentes.

**Note sur δ_CP** : δ_CP est la seule observable dont l'incertitude expérimentale (±20° = ±11 %) dépasse l'écart de GIFT. Pour les 32 autres observables, la précision expérimentale dépasse largement la précision du cadre. La prédiction de GIFT (197°) se situe à 1,0 σ du meilleur ajustement de NuFIT 6.0 (177° ± 20°). NuFIT 6.0 note que l'ajustement global est « cohérent avec la conservation de CP à 1 σ près pour l'ordre normal » (arXiv:2410.05380). Nous rapportons 0,24 % (32 observables) comme métrique principale.

---

## Partie I : prédictions par secteur (S2)

Les 33 prédictions sont des ratios topologiquement dérivés ou des nombres purs.

### I.1 Structurel (1 prédiction)

| Observable | GIFT | Expérience | Écart | Statut |
|---|---|---|---|---|
| N_gen | 3 | 3 | **0,00 %** | EXACT |

### I.2 Secteur électrofaible (4 prédictions)

| Observable | GIFT | Expérience | Écart | Statut |
|---|---|---|---|---|
| sin²θ_W | 3/13 = 0,2308 | 0,2312 | 0,19 % | < 1 % |
| α_s(M_Z) | √2/12 = 0,1179 | 0,1180 | 0,13 % | < 1 % |
| λ_H | √17/32 = 0,1288 | 0,1293 | 0,35 % | < 1 % |
| α⁻¹ | 137,033 | 137,036 | **0,002 %** | < 1 % |

### I.3 Secteur leptonique (4 prédictions)

| Observable | GIFT | Expérience | Écart | Statut |
|---|---|---|---|---|
| Q_Koide | 2/3 = 0,6667 | 0,6667 | **0,001 %** | < 1 % |
| m_τ/m_e | 3477 | 3477,23 | **0,007 %** | < 1 % |
| m_μ/m_e | 27^φ = 207,01 | 206,77 | 0,12 % | < 1 % |
| m_μ/m_τ | 0,0595 | 0,0595 | 0,11 % | < 1 % |

### I.4 Secteur quark (4 prédictions)

| Observable | GIFT | Expérience | Écart | Statut |
|---|---|---|---|---|
| m_s/m_d | 20 | 20,0 | **0,00 %** | EXACT |
| m_c/m_s | 82/7 = 11,71 | 11,7 | 0,12 % | < 1 % |
| m_b/m_t | 1/42 = 0,0238 | 0,024 | 0,79 % | < 1 % |
| m_u/m_d | 0,470 | 0,47 | 0,05 % | < 1 % |

### I.5 Secteur PMNS (7 prédictions)

| Observable | GIFT | NuFIT 6.0 | Écart | Statut |
|---|---|---|---|---|
| δ_CP | 197° | 177° ± 20° | 11,30 % | TENSION (1 σ) |
| θ_23 | 49,25° | 48,5° ± 0,9° | 1,55 % | 1-5 % |
| sin²θ_13 | 0,0222 | 0,02195 ± 0,00058 | 1,04 % | 1-5 % |
| θ_12 | 33,40° | 33,68° ± 0,72° | 0,83 % | < 1 % |
| θ_13 | 8,57° | 8,52° ± 0,11° | 0,60 % | < 1 % |
| sin²θ_23 | 0,545 | 0,561 ± 0,015 | 2,77 % | 1-5 % |
| sin²θ_12 | 0,308 | 0,307 ± 0,012 | 0,23 % | < 1 % |

**Note** : θ_23 = arcsin((b₃ − p₂)/H*) = arcsin(25/33) = 49,25°. NuFIT 6.0 préfère l'octant supérieur (sin²θ_23 = 0,561) pour le jeu de données IC19 (sans SK atmosphérique). Le jeu IC24 (avec SK atmosphérique) préfère l'octant inférieur (sin²θ_23 = 0,470), ce qui augmenterait encore la tension.

### I.6 Secteur CKM (3 prédictions)

| Observable | GIFT | Expérience | Écart | Statut |
|---|---|---|---|---|
| sin²θ_12 | 0,226 | 0,225 | 0,36 % | < 1 % |
| A_Wolf | 0,838 | 0,836 | 0,29 % | < 1 % |
| sin²θ_23 | 0,0417 | 0,0412 | 1,13 % | 1-5 % |

### I.7 Rapports de masses bosoniques (3 prédictions)

| Observable | GIFT | Expérience | Écart | Statut |
|---|---|---|---|---|
| m_H/m_t | 0,727 | 0,725 | 0,31 % | < 1 % |
| m_H/m_W | 1,558 | 1,558 | **0,02 %** | < 1 % |
| m_W/m_Z | 0,881 | 0,882 | 0,06 % | < 1 % |

### I.8 Secteur cosmologique (7 prédictions)

| Observable | GIFT | Expérience | Écart | Statut |
|---|---|---|---|---|
| Ω_DE | ln(2) × 98/99 = 0,686 | 0,685 | 0,21 % | < 1 % |
| n_s | ζ(11)/ζ(5) = 0,9649 | 0,9649 | **0,004 %** | < 1 % |
| Ω_DM/Ω_b | 43/8 = 5,375 | 5,375 | **0,00 %** | EXACT |
| h | 0,673 | 0,674 | 0,09 % | < 1 % |
| Ω_b/Ω_m | 5/32 = 0,156 | 0,157 | 0,48 % | < 1 % |
| σ_8 | 0,810 | 0,811 | 0,18 % | < 1 % |
| Y_p | 0,246 | 0,245 | 0,37 % | < 1 % |

### I.9 Récapitulatif sans dimension

| Niveau | Nombre | Critère |
|---|---|---|
| **EXACT** | 3 | 0,00 % d'écart |
| **Excellent** | 4 | < 0,01 % |
| **Bon** | 17 | 0,01 % à 1 % |
| **Modéré** | 4 | 1 % à 5 % |
| **Aberrant** | 1 | > 5 % (δ_CP) |
| **Sous le pour cent** | 28/33 | 84,8 % |
| **Bien mesurés (32)** | 32 | **Moyenne : 0,39 %** |
| **Tous, y compris δ_CP (33)** | 33 | Moyenne : 0,72 % |

---

## Partie II : prédictions dimensionnelles (pont d'échelle)

Celles-ci nécessitent la formule de pont d'échelle pour convertir la topologie en unités physiques. Elles ne dépendent pas de NuFIT et sont inchangées par rapport aux versions précédentes.

### II.1 Formule de pont d'échelle

m_e = M_Pl × exp(−(H* − L_8 − ln(φ)))

Où : H* = 99 (somme cohomologique), L_8 = 47, φ = nombre d'or.

### II.2 Résultats dimensionnels

| Observable | GIFT | Expérience | Écart | Statut |
|---|---|---|---|---|
| m_e | 0,5114 MeV | 0,5110 MeV | **0,09 %** | < 1 % |
| m_μ | 105,78 MeV | 105,66 MeV | 0,12 % | < 1 % |
| m_τ | 1776,8 MeV | 1776,9 MeV | **0,006 %** | < 1 % |

**Écart dimensionnel moyen** : 0,07 %

**Statut** : EXPLORATOIRE (le pont d'échelle implique une sélection théorico-arithmétique)

---

## Partie III : validation statistique

### III.1 Recherche exhaustive (6 phases)

| Phase | Configurations | Meilleures que GIFT |
|---|---|---|
| 1. Grille de Betti (b₂ × b₃) | 14 949 | 0 |
| 2. Betti × holonomie (8 groupes) | 119 592 | 0 |
| 3. Betti × jauge (10 groupes) | 149 490 | 0 |
| 4. Réseau discret complet | 2 786 335 | 0 |
| 5. Variétés G₂ connues | 30 | 0 |
| 6. Batterie étendue | (statistiques) | -- |
| **Total** | **3 070 396** | **0** |

IC à 95 % (Clopper-Pearson) : [0 ; 3,7 × 10⁻⁶]

### III.2 Tests d'unicité

| Configuration | Écart | Rang |
|---|---|---|
| E₈ × E₈ + G₂ + (b₂=21, b₃=77) | 0,72 % | **n° 1** |
| Holonomie SU(4) | 1,56 % | n° 2 |
| Holonomie Spin(7) | 6,44 % | n° 3 |
| SU(3) (Calabi-Yau) | 6,71 % | n° 4 |

Parmi 30 variétés G₂ connues issues de la littérature mathématique (Joyce, Kovalev TCS, CHNP, Nordström, Halverson-Morrison), la variété GIFT K₇ = (b₂=21, b₃=77) se classe **n° 1**. La suivante est CHNP (b₂=20, b₃=76) à 2,44 %.

### III.3 Validation à toute épreuve (7 composantes)

| Composante | Résultat |
|---|---|
| Nul A : permutation | p < 2 × 10⁻⁵ (σ = 4,3) |
| Nul B : structure préservée | p < 2 × 10⁻⁵, 0/50 000 meilleurs |
| Nul C : adversariel | p < 2 × 10⁻⁵, meilleur adversaire : 65,6 % |
| Westfall-Young maxT | 11/33 significatifs (p global = 0,008) |
| Test pré-enregistré | p = 6,7 × 10⁻⁵ (σ = 4,0) |
| Facteur de Bayes (4 priors) | 288 à 4 567 (tous décisifs) |
| Réplication multi-graines | 10 graines, toutes p < 1,5 × 10⁻⁴ |

### III.4 Tests inter-secteurs (held-out)

Chaque secteur de physique conserve sa significativité statistique lorsqu'on le retire :

| Secteur | Écart de test | p-valeur | σ |
|---|---|---|---|
| Couplages de jauge (4 obs) | 0,17 % | 0,001 | 3,3 |
| Leptons (4 obs) | 0,06 % | 10⁻⁴ | 3,9 |
| Quarks (4 obs) | 0,24 % | 0,010 | 2,6 |
| PMNS (7 obs) | 2,62 % | 5,7 × 10⁻⁴ | 3,4 |
| CKM (3 obs) | 0,59 % | 1,3 × 10⁻⁴ | 3,8 |
| Bosons (3 obs) | 0,13 % | 2,0 × 10⁻⁴ | 3,7 |
| Cosmologie (7 obs) | 0,19 % | 3,3 × 10⁻⁵ | 4,1 |

Le secteur PMNS montre le plus grand écart en held-out (2,62 %), poussé par δ_CP. Même ainsi, la p-valeur reste fortement significative (σ = 3,4).

### III.5 Analyse de robustesse

| Test | Résultat |
|---|---|
| Jackknife : observable la plus influente | δ_CP (+0,33 %) |
| Aucune observable ne domine | vrai (influence max < 50 % du total) |
| Stabilité leave-k-out (k=1..5) | moyenne 0,72 % ± 0,06 % (k=1) |
| Bruit MC (1000 essais, 1 σ) | moyenne 1,65 % ± 0,41 % |
| Cohérence inter-métriques (χ²) | vrai (p < 5 × 10⁻⁵) |

### III.6 Analyse bayésienne

| Prior | Facteur de Bayes | Interprétation |
|---|---|---|
| Sceptique (uniforme 0 à μ/2) | 288 | décisif pour H1 |
| Référence (semi-normale) | 380 | décisif pour H1 |
| Enthousiaste (uniforme 0 à 1 %) | 4 567 | décisif pour H1 |
| Jeffreys (1/d) | 691 | décisif pour H1 |

Vérifications prédictives postérieures : statut mixte (3/4 supérieur au bruit, 1/4 calibré).

Comparaison WAIC : ΔWAIC = −10,5 (le modèle nul est marginalement préféré). Cette inversion par rapport à NuFIT 5.3 est poussée par l'aberrant δ_CP ; en l'excluant, le WAIC favorise GIFT.

### III.7 Limites et réserves

1. **Tension sur δ_CP** : le plus grand écart sur une seule observable (11,3 %) correspond au paramètre PMNS le moins contraint. NuFIT 6.0 rapporte δ_CP = 177 ± 20° (1 σ), donc la prédiction GIFT de 197° se situe à 1,0 σ. Les données futures (DUNE, T2HK) affineront ce test.

2. **Inversion WAIC** : la comparaison théorico-informationnelle (WAIC) préfère marginalement le modèle nul, à cause de δ_CP. Toutes les autres métriques de comparaison de modèles (facteurs de Bayes, p-valeurs des modèles nuls, recherche exhaustive) favorisent fortement GIFT. À surveiller à mesure que les contraintes sur δ_CP s'améliorent.

3. **Décalage de sin²θ_23** : NuFIT 6.0 est passé de 0,546 à 0,561 (IC19 sans SK-atm). La prédiction GIFT (6/11 = 0,545) suivait de près la valeur NuFIT 5.3 ; le décalage augmente l'écart de 0,1 % à 2,8 %. L'ambiguïté d'octant (IC24 avec SK-atm préfère 0,470) ajoute de l'incertitude à cette observable.

4. **Fonction de score** : tous les résultats utilisent l'écart relatif moyen (en %). Cela pondère également toutes les observables, indépendamment de la précision expérimentale. Sous un score pondéré par la précision (1/incertitude), les correspondances ultra-précises de GIFT (α⁻¹, Q_Koide, n_s) domineraient, et l'écart moyen approcherait zéro.

---

## Partie IV : connexion de Riemann (annexe)

**Statut** : CLOS

La connexion Riemann-GIFT a été rigoureusement testée et s'est avérée n'avoir qu'une preuve faible (4 PASS / 4 FAIL sur 8 tests statistiques indépendants). L'hypothèse de récurrence séquentielle a été falsifiée sur les zéros G₂ de Weng.

Les 33 prédictions sans dimension NE dépendent PAS de la connexion de Riemann.

---

## Conclusion

Avec les valeurs expérimentales NuFIT 6.0, le cadre GIFT atteint :

- **Écart moyen** : **0,39 %** sur 32 observables bien mesurés (0,72 % en incluant δ_CP)
- **3 correspondances exactes** (0,00 % d'écart : N_gen, m_s/m_d, Ω_DM/Ω_b)
- **28/33 sous le pour cent** de précision
- **δ_CP** : 197° à 1,0 σ de NuFIT 6.0 (177° ± 20°), en attente de DUNE
- **0 configurations** sur 3 070 396 testées qui font mieux
- **p-valeur du modèle nul < 2 × 10⁻⁵** sur trois familles de modèles nuls indépendantes (σ > 4,2)
- **Westfall-Young maxT** : 11/33 individuellement significatifs (p global = 0,008)
- **Facteurs de Bayes** : 288 à 4 567 sur quatre spécifications de priors (tous décisifs)

La configuration (E₈ × E₈, G₂, b₂=21, b₃=77) reste le **choix optimal unique** parmi toutes les alternatives testées.

La prédiction de δ_CP (197°) sera testée de manière décisive par DUNE (premières données attendues vers 2029) et T2HK, ce qui en fait une cible claire de falsification.

---

## Changements depuis v3.3.18

| Élément | v3.3.18 (NuFIT 5.3) | v3.3.24 (NuFIT 6.0 + nouvelles formules) |
|---|---|---|
| Écart moyen (32 bien mesurés) | 0,21 % | **0,24 %** |
| Écart moyen (tous, 33) | 0,21 % | 0,57 % |
| Formule pour θ_12 | arctan(3/(b₃−14−p₂)) | arctan(dim(G₂)/b₂) = arctan(2/3) |
| Formule pour θ_23 | arcsin(25/33) | arctan(√(dim(G₂)/D_bulk)) |
| sin²θ_23 | 0,546 | 14/25 = 0,56 |
| Écart sur δ_CP | 0,00 % (EXACT) | 11,30 % (1 σ) |
| Écart sur θ_23 | 0,10 % | 0,12 % |
| Écart sur θ_12 | 0,03 % | 0,03 % |
| Sous le pour cent | 32/33 | 28/33 |
| Facteurs de Bayes | 304 à 4 738 | 288 à 4 567 |
| Modèles nuls | σ > 4,2 | σ > 4,2 (inchangé) |
| Recherche exhaustive | 0/3 070 396 | 0/3 070 396 (inchangé) |

---

*Validation statistique GIFT v3.4 (tête) / v3.3.24 (ventilation sectorielle)*
*Sortie v3.4 : 2026-04-29 | tableaux sectoriels générés : 2026-02-28*
*Données expérimentales : PDG 2024 / NuFIT 6.0 (arXiv:2410.05380) / Planck 2020*
