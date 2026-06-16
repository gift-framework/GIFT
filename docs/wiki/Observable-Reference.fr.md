---
title: "Référence des observables"
layout: default
---

# Référence des observables GIFT

**Version** : 3.4 (instantané de haut niveau ; voir Supplément S3 pour le jeu complet de 95 observables). Le découpage par secteur v3.3.24 ci-dessous est conservé pour la traçabilité de l'analyse NuFIT 6.0.
**Statut** : documentation de référence
**Date** : mai 2026

---

## Résumé exécutif (v3.4)

| Métrique | Valeur |
|----------|--------|
| **Total observables** | **95** (33 Type I + 19 Type II + 21 Type III + 22 Type IV) |
| **Certifiées Lean** | 55 / 95 (Type I : 33/33, Type III : 14/21, Type IV : 8/22) |
| **Avec comparaison expérimentale** | 66 / 95 |
| Écart moyen (Type I, 33 obs) | 0,99 % |
| Écart moyen (Type II, 19 obs) | 0,17 % |
| Écart moyen (Type III, 21 obs) | 3,44 % |
| Correspondances exactes (< 0,01 %) | 11 |
| Sous 1 % | 53 |
| Paramètres libres (continûment ajustables) | 0 |
| Certificat Lean | 140 conjonctions, 15 axiomes (4 principaux + 11 d'arithmétique d'intervalle), 0 sorry, 143 fichiers .lean |
| Tests statistiques nuls | uniforme P=10⁻³⁴⁶, null algébrique (4,2 M formules aléatoires) P=10⁻¹³³, surdétermination 2,13× |

---

### Découpage v3.3.24 par secteur (conservé pour traçabilité)

| Métrique | Valeur (v3.3.24) |
|----------|--------|
| Prédictions sans dimension principales | 18 |
| Sans dimension étendues | 15 |
| Paramètres cosmologiques | 11 |
| Constantes structurelles | 18 |
| Total des quantités cataloguées | 51 |
| Écart moyen (32 bien mesurées) | 0,99 % (NuFIT 6.1 / PDG 2024 / Planck 2018 / CODATA 2022) |
| Correspondances exactes (< 0,1 %) | 14 (42 %) |
| Multiplement déterminées (≥3 expressions) | 92 % |
| Total des expressions équivalentes | 280+ |
| Paramètres libres | 0 |

---

## 1. Constantes topologiques GIFT

### 1.1 Constantes primaires

| Symbole | Valeur | Définition | mod 7 | Facteur |
|---------|--------|------------|-------|---------|
| b_0 | 1 | nombre de Betti zéroième | 1 | - |
| p_2 | 2 | paramètre de dualité | 2 | - |
| N_gen | 3 | nombre de générations | 3 | - |
| Weyl | 5 | facteur de Weyl | 5 | - |
| dim(K_7) | 7 | dimension de la variété compacte | **0** | 7 |
| rang(E_8) | 8 | rang Cartan de E_8 | 1 | - |
| D_bulk | 11 | dimension du bulk | 4 | - |
| alpha_sum | 13 | somme des anomalies | 6 | - |
| dim(G_2) | 14 | dimension de l'holonomie G_2 | **0** | 2x7 |
| b_2 | 21 | second nombre de Betti | **0** | 3x7 |
| dim(J_3(O)) | 27 | algèbre de Jordan exceptionnelle | 6 | - |
| det(g)_den | 32 | dénominateur du déterminant métrique | 4 | 2^5 |
| 2b_2 | 42 | constante structurelle (= p₂ × b₂) | **0** | 6x7 |
| dim(F_4) | 52 | dimension de F_4 | 3 | - |
| fund(E_7) | 56 | représentation fondamentale de E_7 | **0** | 8x7 |
| kappa_T^-1 | 61 | capacité de torsion inverse | 5 | premier |
| det(g)_num | 65 | numérateur du déterminant métrique | 2 | 5x13 |
| b_3 | 77 | troisième nombre de Betti | **0** | 11x7 |
| dim(E_6) | 78 | dimension de E_6 | 1 | - |
| H* | 99 | cohomologie totale (b_2+b_3+1) | 1 | 9x11 |
| PSL(2,7) | 168 | ordre de la symétrie de Fano | **0** | 24x7 |
| dim(E_8) | 248 | dimension de E_8 | 3 | - |
| dim(E_8xE_8) | 496 | dimension du groupe de jauge | 6 | - |

### 1.2 Identités algébriques maîtresses

```
dim(G_2)       = p_2 x dim(K_7)           = 2 x 7   = 14
b_2            = N_gen x dim(K_7)         = 3 x 7   = 21
b_3 + dim(G_2) = dim(K_7) x alpha_sum     = 7 x 13  = 91
alpha_sum      = rang(E_8) + Weyl         = 8 + 5   = 13
D_bulk         = rang(E_8) + N_gen        = 8 + 3   = 11
2b_2           = p_2 x b_2                = 2 x 21  = 42  (constante structurelle)
H*             = b_2 + b_3 + 1            = 21+77+1 = 99

PSL(2,7) = 168 = rang(E_8) x b_2          = 8 x 21
               = N_gen x fund(E_7)        = 3 x 56
               = (b_3 + dim(G_2)) + b_3   = 91 + 77
```

---

## 2. Classification de l'inévitabilité structurelle

Chaque observable reçoit une classification basée sur le nombre d'expressions algébriques indépendantes :

| Classification | Critères | Interprétation |
|----------------|----------|----------------|
| **CANONIQUE** | ≥20 expressions | maximalement sur-déterminée ; la valeur émerge du réseau algébrique |
| **ROBUSTE** | 10-19 expressions | hautement contraint ; multiples dérivations indépendantes |
| **SOUTENUE** | 5-9 expressions | multiplement dérivée ; redondance structurelle |
| **DÉRIVÉE** | 2-4 expressions | au moins double dérivation |
| **SINGULIÈRE** | 1 expression | dérivation unique (coïncidence numérique possible) |

**Référence croisée avec les étiquettes de statut de GIFT_ATLAS.json** :

| Ce document | Équivalent atlas | Justification du mappage |
|-------------|------------------|--------------------------|
| CANONIQUE | VERIFIED | prouvé en Lean, maximalement sur-déterminée |
| ROBUSTE | VERIFIED | multiples dérivations indépendantes confirment |
| SOUTENUE | TOPOLOGICAL | conséquence topologique directe |
| DÉRIVÉE | TOPOLOGICAL | conséquence algébrique des invariants topologiques |
| SINGULIÈRE | TOPOLOGICAL | dérivation unique, mais ancrée topologiquement |

---

## 3. Les 18 prédictions sans dimension principales

### 3.1 Structurelle

| # | Observable | Formule GIFT | Valeur | Exp. | Écart | # Expr. | Statut |
|---|------------|--------------|--------|------|-------|---------|--------|
| 1 | **N_gen** | indice d'Atiyah-Singer | **3** | 3 | 0,00 % | 24+ | CANONIQUE |

### 3.2 Secteur électrofaible

| # | Observable | Formule GIFT | Valeur | Exp. | Écart | # Expr. | Statut |
|---|------------|--------------|--------|------|-------|---------|--------|
| 2 | **sin² θ_W** | b_2/(b_3+dim_G_2) | 3/13 = 0,2308 | 0,23122 | 0,20 % | 19 | ROBUSTE |
| 3 | **alpha_s(M_Z)** | sqrt(2)/(dim_G2 - p_2) | sqrt(2)/12 = 0,1179 | 0,1179 | 0,042 % | 9 | TOPOLOGICAL |
| 4 | **lambda_H** | sqrt(17)/32 | 0,1288 | 0,129 | 0,12 % | 4 | DÉRIVÉE |
| 5 | **alpha^-1(M_Z)** | 128+9+corr | 137,033 | 137,036 | 0,002 % | 3 | DÉRIVÉE |

### 3.3 Secteur leptonique

| # | Observable | Formule GIFT | Valeur | Exp. | Écart | # Expr. | Statut |
|---|------------|--------------|--------|------|-------|---------|--------|
| 6 | **Q_Koide** | dim_G_2/b_2 | 2/3 | 0,666661 | 0,001 % | 27 | CANONIQUE |
| 7 | **m_tau/m_e** | 7+10x248+10x99 | 3477 | 3477,15 | 0,004 % | 3 | DÉRIVÉE |
| 8 | **m_mu/m_e** | 27^phi | 207,01 | 206,768 | 0,12 % | 2 | DÉRIVÉE |

### 3.4 Secteur des quarks

| # | Observable | Formule GIFT | Valeur | Exp. | Écart | # Expr. | Statut |
|---|------------|--------------|--------|------|-------|---------|--------|
| 9 | **m_s/m_d** | p_2² x Weyl | 4 x 5 = 20 | 20,0 | 0,00 % | 14 | VERIFIED |
| 10 | **m_c/m_s** | (dim_E8-p_2)/b_2 | 246/21 = 11,71 | 11,7 | 0,12 % | 5 | SOUTENUE |
| 11 | **m_b/m_t** | 1/(2b₂) | 1/42 = 0,0238 | 0,024 | 0,79 % | 12 | ROBUSTE |
| 12 | **m_u/m_d** | (1+dim_E6)/PSL_27 | 79/168 = 0,470 | 0,47 | 0,05 % | 4 | DÉRIVÉE |

### 3.5 Secteur neutrinos / PMNS

| # | Observable | Formule GIFT | Valeur | Exp. | Écart | # Expr. | Statut |
|---|------------|--------------|--------|------|-------|---------|--------|
| 13 | **delta_CP** | dim_K7 x dim_G2 + H* | 197° | 197° ± 24° | 0,00 % | 3 | DÉRIVÉE |
| 14 | **theta_13^PMNS** | π/b_2 | 8,57° | 8,54° | 0,37 % | 3 | DÉRIVÉE |
| 15 | **theta_23^PMNS** | arcsin((b_3-p_2)/H*) = arcsin(25/33) | 49,25° | 49,3° | 0,10 % | 2 | TOPOLOGICAL |
| 16 | **theta_12^PMNS** | arctan(sqrt(delta/gamma)) | 33,40° | 33,41° | 0,03 % | 2 | DÉRIVÉE |

### 3.6 Secteur cosmologique

| # | Observable | Formule GIFT | Valeur | Exp. | Écart | # Expr. | Statut |
|---|------------|--------------|--------|------|-------|---------|--------|
| 17 | **Omega_DE** | ln(2)x(b_2+b_3)/H* | 0,6861 | 0,6847 | 0,21 % | 2 | DÉRIVÉE |
| 18 | **n_s** | zeta(11)/zeta(5) | 0,9649 | 0,9649 | 0,004 % | 2 | DÉRIVÉE |

---

## 4. Prédictions sans dimension étendues (15)

### 4.1 Forme PMNS sin²

| # | Observable | Formule GIFT | Valeur | Exp. | Écart | # Expr. | Statut |
|---|------------|--------------|--------|------|-------|---------|--------|
| 19 | **sin² θ_12^PMNS** | (1+N_gen)/alpha_sum | 4/13 = 0,308 | 0,307 | 0,23 % | 21 | CANONIQUE |
| 20 | **sin² θ_23^PMNS** | (D_bulk-Weyl)/D_bulk | 6/11 = 0,545 | 0,546 | 0,10 % | 13 | ROBUSTE |
| 21 | **sin² θ_13^PMNS** | D_bulk/dim_E8² | 11/496 = 0,022 | 0,0220 | 0,81 % | 5 | SOUTENUE |

### 4.2 Matrice CKM

| # | Observable | Formule GIFT | Valeur | Exp. | Écart | # Expr. | Statut |
|---|------------|--------------|--------|------|-------|---------|--------|
| 22 | **sin² θ_12^CKM** | fund_E7/dim_E8 | 56/248 = 0,2258 | 0,2250 | 0,36 % | 16 | ROBUSTE |
| 23 | **A_Wolfenstein** | (Weyl+dim_E6)/H* | 83/99 = 0,838 | 0,836 | 0,29 % | 7 | SOUTENUE |
| 24 | **sin² θ_23^CKM** | dim_K7/PSL_27 | 7/168 = 0,042 | 0,0412 | 1,13 % | 4 | DÉRIVÉE |

### 4.3 Rapports de masses bosoniques

| # | Observable | Formule GIFT | Valeur | Exp. | Écart | # Expr. | Statut |
|---|------------|--------------|--------|------|-------|---------|--------|
| 25 | **m_H/m_t** | fund_E7/b_3 | 56/77 = 0,7273 | 0,725 | 0,31 % | 16 | ROBUSTE |
| 26 | **m_H/m_W** | (N_gen+dim_E6)/dim_F4 | 81/52 = 1,5577 | 1,558 | 0,02 % | 3 | DÉRIVÉE |
| 27 | **m_W/m_Z** | (chi-Weyl)/chi | 37/42 = 0,8810 | 0,8815 | 0,06 % | 8 | SOUTENUE |

**Note** : m_W/m_Z = 37/42 est une **correction v3.3**. La formule précédente (23/26) avait un écart de 0,35 % ; la nouvelle formule atteint 0,06 %.

### 4.4 Rapports leptoniques étendus

| # | Observable | Formule GIFT | Valeur | Exp. | Écart | # Expr. | Statut |
|---|------------|--------------|--------|------|-------|---------|--------|
| 28 | **m_mu/m_tau** | (b_2-D_bulk)/PSL_27 | 10/168 = 0,0595 | 0,0595 | 0,04 % | 9 | SOUTENUE |

---

## 5. Paramètres cosmologiques (complets)

### 5.1 Composition de l'univers

| # | Observable | Planck 2018 | GIFT | Valeur | Écart | # Expr |
|---|------------|-------------|------|--------|-------|--------|
| 29 | **Omega_DM/Omega_b** | 5,375 ± 0,1 | (b_0+chi)/rang | **43/8 = 5,375** | **0,00 %** | 3 |
| 30 | **Omega_c/Omega_Lambda** | 0,387 ± 0,01 | det_g_num/PSL_27 | 65/168 = 0,3869 | 0,01 % | 5 |
| 31 | **Omega_Lambda/Omega_m** | 2,175 ± 0,05 | (dim_G2+H*)/dim_F4 | 113/52 = 2,173 | 0,07 % | 6 |
| 32 | **h (Hubble)** | 0,674 ± 0,005 | (PSL_27-b_0)/dim_E8 | 167/248 = 0,6734 | 0,09 % | 4 |
| 33 | **Omega_b/Omega_m** | 0,156 ± 0,003 | Weyl/det_g_den | 5/32 = 0,1562 | 0,16 % | 7 |
| 34 | **Omega_c/Omega_m** | 0,841 ± 0,01 | (dim_E8²-dim_E6)/dim_E8² | 0,8427 | 0,17 % | 4 |
| 35 | **sigma_8** | 0,811 ± 0,006 | (p_2+det_g_den)/chi | 34/42 = 0,8095 | 0,18 % | 3 |
| 36 | **Omega_m/Omega_Lambda** | 0,460 ± 0,01 | (b_0+dim_J3O)/kappa_T | 28/61 = 0,459 | 0,18 % | 5 |
| 37 | **Y_p** (He primordial) | 0,245 ± 0,003 | (b_0+dim_G2)/kappa_T | 15/61 = 0,2459 | 0,37 % | 4 |
| 38 | **Omega_Lambda/Omega_b** | 13,9 ± 0,3 | (dim_E8²-dim_F4)/det_g_den | 13,875 | 0,14 % | 3 |
| 39 | **Omega_b/Omega_Lambda** | 0,072 ± 0,002 | b_0/dim_G2 | 1/14 = 0,0714 | 0,75 % | 2 |

### 5.2 Le 42 en cosmologie

**Résultat notable** :

$$\frac{\Omega_{DM}}{\Omega_b} = \frac{b_0 + 2b_2}{\text{rang}(E_8)} = \frac{1 + 42}{8} = \frac{43}{8} = 5{,}375$$

Le rapport matière noire / matière baryonique **contient explicitement la constante structurelle 2b₂ = 42**.

**Note** : la caractéristique d'Euler χ(K₇) = 0 pour toute variété compacte de dimension impaire comme K₇. La valeur 42 = p₂ × b₂ est une constante structurelle distincte dérivée des nombres de Betti.

---

## 6. Constantes structurelles (18)

### 6.1 Structure E_8

| # | Constante | Valeur | Définition | # Expr. | Statut |
|---|-----------|--------|------------|---------|--------|
| 40 | **dim(E_8)** | 248 | dimension de l'algèbre de Lie E_8 | 5+ | SOUTENUE |
| 41 | **rang(E_8)** | 8 | sous-algèbre de Cartan | 3+ | DÉRIVÉE |
| 42 | **dim(E_8 x E_8)** | 496 | groupe produit | 2 | DÉRIVÉE |
| 43 | **|W(E_8)|** | 696 729 600 | ordre du groupe de Weyl | 1 | SINGULIÈRE |

### 6.2 Topologie G_2 et K_7

| # | Constante | Valeur | Définition | # Expr. | Statut |
|---|-----------|--------|------------|---------|--------|
| 44 | **dim(G_2)** | 14 | groupe d'holonomie | 4+ | DÉRIVÉE |
| 45 | **dim(K_7)** | 7 | variété compacte | 5+ | SOUTENUE |
| 46 | **b_2(K_7)** | 21 | second Betti (modules de jauge) | 3+ | DÉRIVÉE |
| 47 | **b_3(K_7)** | 77 | troisième Betti (modes matière) | 3+ | DÉRIVÉE |
| 48 | **H*** | 99 | b_2+b_3+1 (cohomologie totale) | 5+ | SOUTENUE |
| 49 | **2b₂** | 42 | constante structurelle (p₂ × b₂) | 3+ | DÉRIVÉE |

### 6.3 Algèbres exceptionnelles

| # | Constante | Valeur | Définition | # Expr. | Statut |
|---|-----------|--------|------------|---------|--------|
| 50 | **dim(J_3(O))** | 27 | Jordan exceptionnelle | 2+ | DÉRIVÉE |
| 51 | **dim(F_4)** | 52 | dimension de F_4 | 3+ | DÉRIVÉE |

---

## 7. Analyse de sur-détermination

### 7.1 Top des expressions équivalentes par fraction

| Fraction | Observable | # Expressions |
|----------|------------|---------------|
| 2/3 | Q_Koide | **27** |
| 21/7 = 3 | N_gen | **24** |
| 4/13 | sin² θ_12^PMNS | **21** |
| 3/13 | sin² θ_W | **19** |
| 8/11 = 56/77 | m_H/m_t | **16** |
| 56/248 | sin² θ_12^CKM | **16** |
| 1/42 | m_b/m_t | **12** |
| 6/11 | sin² θ_23^PMNS | **13** |
| 37/42 | m_W/m_Z | **8** |

**Total : 280+ expressions pour les observables principales**

### 7.2 Exemple : Q_Koide = 2/3 (27 expressions)

| # | Expression | Calcul |
|---|------------|--------|
| 1 | p_2 / N_gen | 2/3 |
| 2 | dim_G_2 / b_2 | 14/21 = 2/3 |
| 3 | dim_F_4 / dim_E_6 | 52/78 = 2/3 |
| 4 | rang_E_8 / (Weyl + dim_K_7) | 8/12 = 2/3 |
| 5 | chi / (b_2 + chi) | 42/63 = 2/3 |
| ... | ... | ... |

### 7.3 Significativité statistique

Pour une numérologie aléatoire avec ~20 constantes :
- Expressions attendues par fraction : ~1-2
- Observées : ~16 en moyenne

**Probabilité par hasard** : p < 10⁻¹²

La structure est **réelle**, pas coïncidente.

---

## 8. Distribution statistique

### 8.1 Par écart (33 observables)

| Plage | Nombre | % | Exemples |
|-------|--------|---|----------|
| Exact (0 %) | 2 | 6 % | N_gen, Omega_DM/Omega_b |
| < 0,1 % | 12 | 36 % | Q_Koide, m_H/m_W, m_W/m_Z, h |
| 0,1-0,5 % | 12 | 36 % | sin² θ_W, m_mu/m_e, m_H/m_t |
| 0,5-1 % | 5 | 15 % | m_b/m_t, sin² θ_13^PMNS |
| > 1 % | 2 | 6 % | sin² θ_23^CKM |

### 8.2 Par catégorie

| Catégorie | Observables | Écart moyen | Meilleure correspondance |
|-----------|-------------|-------------|--------------------------|
| Électrofaible | 4 | 0,27 % | m_W/m_Z (0,06 %) |
| PMNS | 4 | 0,29 % | sin² θ_23 (0,10 %) |
| Masses des quarks | 5 | 0,35 % | m_s/m_d (0,00 %) |
| Masses des leptons | 2 | 0,04 % | m_mu/m_tau (0,04 %) |
| Masses bosoniques | 3 | 0,13 % | m_H/m_W (0,02 %) |
| CKM | 4 | 0,59 % | A_Wolf (0,29 %) |
| Cosmologie | 11 | 0,16 % | Omega_DM/Omega_b (0,00 %) |
| **Total** | **32+1** | **0,99 %** | - |

### 8.3 Par classification structurelle

| Classification | Nombre | % |
|----------------|--------|---|
| CANONIQUE | 4 | 12 % |
| ROBUSTE | 8 | 24 % |
| SOUTENUE | 12 | 36 % |
| DÉRIVÉE | 8 | 24 % |
| SINGULIÈRE | 1 | 3 % |

---

## 9. Expressions uniques (prudence)

Observables avec une seule expression GIFT (coïncidence numérique possible) :

| Observable | Expression | Valeur | Statut |
|------------|------------|--------|--------|
| |W(E_8)| | 696 729 600 | - | définition |

---

## 10. Analyse d'unicité

### 10.1 Unicité du groupe de jauge

E₈×E₈ est **optimal** parmi tous les groupes de jauge physiquement motivés testés.

| Rang | Groupe de jauge | Écart moyen | N_gen | Statut |
|------|-----------------|-------------|-------|--------|
| **1** | **E₈×E₈** | **0,24 %** | **3,000** | ✓ OPTIMAL |
| 2 | E₇×E₈ | 3,06 % | 2,625 | ✗ |
| 3 | E₆×E₈ | 5,72 % | 2,250 | ✗ |
| 4 | E₇×E₇ | 6,05 % | 2,625 | ✗ |
| 5 | SO(32) | 6,82 % | 6,000 | ✗ |
| 6 | E₆×E₆ | 14,52 % | 2,250 | ✗ |

**Facteur d'amélioration** : E₈×E₈ est **12,8× meilleur** que le suivant (E₇×E₈).

**Pourquoi rang=8 est spécial** :
```
N_gen = (rang × b₂) / (b₃ - b₂) = (rang × 21) / 56

Pour N_gen = 3 exactement : rang = 168/21 = 8 ✓
Note : 168 = |PSL(2,7)| = ordre de la symétrie du plan de Fano
```

Seul E₈ (rang 8) donne exactement 3 générations.

### 10.2 Unicité de l'holonomie

L'holonomie G₂ atteint un accord significativement meilleur. Les variétés Calabi-Yau donnent de mauvais résultats.

| Rang | Holonomie | dim_K | SUSY | Écart moyen | Statut |
|------|-----------|-------|------|-------------|--------|
| **1** | **G₂** | 7 | N=1 | **0,24 %** | ✓ |
| 2 | SU(4) | 8 | N=1 | 0,71 % | ✗ |
| 3 | SU(3) | 6 | N=2 | 3,12 % | ✗✗ |
| 4 | Spin(7) | 8 | N=0 | 3,56 % | ✗✗ |

**Pénalité Calabi-Yau** : l'holonomie SU(3) échoue d'un facteur **13×**.

### 10.3 La connexion PSL(2,7)

```
N_gen = |PSL(2,7)| / fund(E₇) = 168 / 56 = 3
      = |symétrie_Fano| / E₇_fondamentale
```

Le nombre de générations est égal à l'ordre de la symétrie du plan de Fano divisé par la dimension de la représentation de E₇.

Ce n'est **pas de la numérologie**, c'est la structure octonionique de Fano se manifestant dans les générations de particules.

### 10.4 Script de validation

Analyse complète disponible : [`publications/validation/validation_v33.py`](https://github.com/gift-framework/GIFT/blob/main/publications/validation/validation_v33.py)

```bash
python publications/validation/validation_v33.py
```

Résultats : [`publications/references/observables.csv`](https://github.com/gift-framework/GIFT/blob/main/publications/references/observables.csv)

---

## 11. Calendrier de falsification

| Prédiction | Actuel | Cible | Expérience | Année |
|------------|--------|-------|------------|-------|
| **delta_CP = 197°** | ± 24° | ± 10° | DUNE (premiers résultats) | 2028-2030 |
| **delta_CP = 197°** | ± 10° | ± 5° | DUNE (précision) | 2034-2039 |
| **sin² θ_W = 3/13** | ± 0,00004 | ± 0,00001 | FCC-ee | années 2040 |
| **N_gen = 3** | 3 | 4ᵉ génération ? | LHC/FCC | en cours |
| **m_s/m_d = 20** | ± 1,0 | ± 0,3 | QCD sur réseau | 2030 |
| **Q_Koide** | ± 0,000007 | ± 0,000001 | usines à tau | années 2030 |

**Note** : le calendrier DUNE suit les projections Snowmass 2021. Premier faisceau ~2028 ; précision de ± 5° requiert une exploitation prolongée jusqu'à la fin des années 2030.

---

## 12. L'analogie de Balmer

| Aspect | Balmer (1885) | GIFT |
|--------|---------------|------|
| Formule empirique | λ = B x n²/(n²-4) | sin² θ_W = 3/13 |
| Correspond à l'expérience | oui | oui |
| Formule unique | oui | oui (à équivalence près) |
| Dérivation venue plus tard | Bohr (1913), QM (1926) | ? |

---

## 13. Références

- Harvey, R., Lawson, H.B. « Calibrated geometries. » Acta Math. 148 (1982)
- Joyce, D.D. *Compact Manifolds with Special Holonomy*. Oxford (2000)
- Koide, Y. « Fermion-boson two-body model. » Lett. Nuovo Cim. 34 (1982)
- Particle Data Group (2024), Review of Particle Physics
- Planck Collaboration (2020), paramètres cosmologiques
- Publications GIFT : [Article principal](Paper-Main-Framework.html), [Article S2 dérivations](Paper-S2-Derivations.html)

---

*Cadre GIFT v3.4, référence des observables*
