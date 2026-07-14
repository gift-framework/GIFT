---
title: Info Geo pour les physiciens
---

# Géométrie de l'information en physique des particules : la perspective K₇

## Résumé

Le cadre K₇ propose que les paramètres du Modèle Standard ne sont pas des constantes libres nécessitant une détermination expérimentale, mais plutôt des invariants topologiques d'une structure géométrique sous-jacente. Plus précisément, le cadre dérive 33 relations Type I sans dimension à partir de la cohomologie d'une variété K₇ à holonomie G₂ de dimension 7, couplée à une architecture de jauge E₈×E₈. Ce document présente la structure conceptuelle pour les physiciens théoriciens, en se concentrant sur la manière dont la topologie pourrait éliminer le besoin de paramètres ajustables.

## 1. Le problème des paramètres

Le Modèle Standard décrit avec une grande précision les interactions électromagnétique, faible et forte, tout en contenant 19 paramètres libres : 3 couplages de jauge, 9 masses de fermions, 4 paramètres CKM et 3 constantes additionnelles (masse du Higgs, VEV du Higgs, angle de vide QCD). Ils s'étalent sur six ordres de grandeur sans explication théorique de leurs valeurs.

Les approches d'unification traditionnelles n'ont pas résolu cette situation. Les théories de grande unification introduisent des paramètres supplémentaires en tentant d'expliquer les 19 d'origine. L'espace des modules de la théorie des cordes englobe environ 10⁵⁰⁰ vides, transformant le problème des paramètres en un problème de sélection de vide. La question persiste : ces paramètres sont-ils vraiment libres, ou encodent-ils une structure plus profonde ?

Le cadre K₇ explore une alternative : et si les 19 paramètres étaient des invariants topologiques d'une variété interne compacte ? Dans cette image, il n'y a rien à ajuster car les données topologiques discrètes n'admettent aucune variation continue.

## 2. La construction géométrique

### 2.1 Pourquoi E₈×E₈ ?

Le groupe produit E₈×E₈ apparaît pour plusieurs raisons. E₈ est la plus grande algèbre de Lie simple exceptionnelle (dimension 248, rang 8). Son groupe de Weyl W(E₈) a pour ordre 696 729 600 = 2¹⁴ × 3⁵ × 5² × 7, contenant tous les nombres premiers et puissances qui réapparaîtront dans les formules des observables. Le produit E₈×E₈ apparaît naturellement dans la théorie des cordes hétérotique et satisfait les conditions d'annulation des anomalies.

Important : le cadre n'embarque pas directement les particules du Modèle Standard dans des représentations de E₈. L'embarquement direct se heurte à l'obstruction de Distler-Garibaldi : E₈ ne peut accueillir trois générations chirales avec les bons nombres quantiques. À la place, E₈×E₈ fournit une architecture théorico-informationnelle, les particules physiques émergeant de la réduction dimensionnelle sur la variété interne.

### 2.2 La variété interne K₇

Le cadre postule une variété riemannienne compacte K₇ de dimension 7 à holonomie G₂. G₂ est le groupe d'automorphismes des octonions (dimension 14, rang 2). C'est l'holonomie exceptionnelle minimale en dimension 7 et elle préserve exactement un spineur, donnant une supersymétrie N=1 lors de la compactification.

K₇ est construite via la méthode du twisted connected sum (TCS) : on recolle deux variétés de Calabi-Yau de dimension 3 asymptotiquement cylindriques le long de leur frontière commune S¹ × K3. Les blocs de construction spécifiques déterminent la topologie :

| Invariant | Valeur | Interprétation physique |
|-----------|--------|-------------------------|
| b₂(K₇) | 21 | Multiplicité des champs de jauge (2-formes harmoniques) |
| b₃(K₇) | 77 | Multiplicité des champs de matière (3-formes harmoniques) |
| H* = b₂ + b₃ + 1 | 99 | Dimension cohomologique effective |

### 2.3 Le mécanisme de torsion

Les variétés à holonomie G₂ standard ont une 3-forme sans torsion : dφ = 0, d*φ = 0. Les interactions physiques requièrent un écart contrôlé par rapport à cette idéalisation. Le cadre introduit une torsion d'amplitude :

κ_T = 1/(b₃ - dim(G₂) - p₂) = 1/(77 - 14 - 2) = 1/61

Cette torsion engendre une dynamique via le flot géodésique sur K₇, fournissant une interprétation géométrique des équations du groupe de renormalisation.

## 3. De la topologie aux observables

### 3.1 La réduction dimensionnelle

La chaîne procède ainsi :

E₈×E₈ (496D) → AdS₄ × K₇ (11D) → Modèle Standard (4D)

Lors de la compactification, les champs de jauge proviennent des 2-formes harmoniques sur K₇ (d'où 21 générateurs, contenant SU(3)×SU(2)×U(1) plus un secteur caché), tandis que les fermions chiraux proviennent des 3-formes harmoniques (d'où 77 modes, contenant 3 générations de 16 fermions de Weyl plus des états additionnels).

### 3.2 Quelques mappages représentatifs

Le cadre dérive les observables à partir de données cohomologiques et algébriques. Exemples :

**Angle de Weinberg** (mélange électrofaible) :
sin²θ_W = b₂/(b₃ + dim(G₂)) = 21/(77 + 14) = 21/91 = 3/13 = 0,23077

C'est un nombre rationnel exact déterminé par la topologie.

**Nombre de générations** :
N_gen = rang(E₈) - Weyl = 8 - 5 = 3

Le facteur 5 provient de 5² dans |W(E₈)|.

**Phase de violation CP** (secteur des neutrinos) :
δ_CP = 7 × dim(G₂) + H* = 7 × 14 + 99 = 197°

Une formule additive combinant la dimension de la variété, la dimension de l'holonomie et la dimension cohomologique.

**Relation de Koide** (masses des leptons chargés) :
Q = dim(G₂)/b₂ = 14/21 = 2/3

La formule empirique de Koide (m_e + m_μ + m_τ)/(√m_e + √m_μ + √m_τ)² = 2/3 émerge comme un rapport topologique.

### 3.3 Ce qui est revendiqué

Le cadre produit 33 relations Type I sans dimension couvrant les couplages de jauge, le mélange des neutrinos, les rapports de masses des leptons, les rapports de masses des quarks et les observables cosmologiques. L'écart moyen par rapport aux valeurs expérimentales est de 0,99 % (NuFIT 6.1 / PDG 2024 / Planck 2018 / CODATA 2022).

Toutes les 460+ relations certifiées ont été formellement vérifiées dans l'assistant de preuve Lean 4, avec 15 axiomes (4 principaux + 11 d'arithmétique d'intervalle) et 0 sorry.

### 3.4 Ce qui n'est pas revendiqué

Le cadre ne constitue pas une théorie complète de la gravité quantique. Il ne dérive pas le principe d'action à partir de premiers principes. Le mécanisme dynamique reliant la géométrie à la physique reste incomplet ; le mappage entre données topologiques et observables physiques est établi mais non dérivé d'un principe plus profond.

La description « zéro-paramètre » fait référence à l'absence de quantités ajustables continues. Des choix structurels discrets demeurent : E₈×E₈ plutôt que SO(32), ce K₇ spécifique plutôt que d'autres variétés G₂. Ces choix sont motivés par des exigences de cohérence (annulation des anomalies, contenu en champs correct) mais ne sont pas uniquement déterminés.

La question de savoir si cette structure mathématique reflète la réalité fondamentale ou constitue une description effective reste ouverte.

## 4. Tests expérimentaux

Le cadre fait des prédictions spécifiques testables par des expériences à court terme :

| Prédiction | Valeur actuelle | Expérience | Calendrier |
|------------|-----------------|------------|------------|
| δ_CP = 197° | meilleur ajustement ~207-212° (NuFIT 6.1), 197° à ~1σ | DUNE | 2028-2040 |
| sin²θ_W = 3/13 | 0,23122 ± 0,00003 | FCC-ee | années 2040 |
| m_s/m_d = 20 | 20,0 ± 1,0 | QCD sur réseau | 2030 |
| N_gen = 3 | 3 | LHC | en cours |

**Critères de falsification** : le cadre serait réfuté par une mesure de δ_CP en dehors de [182°, 212°], la découverte d'un fermion de quatrième génération, ou une détermination de précision de m_s/m_d significativement différente de 20. Ce sont de véritables tests expérimentaux, pas des accommodements a posteriori.

## 5. Relation aux autres approches

**Diffère de l'embarquement E₈ direct** : les premières tentatives d'embarquer les champs du Modèle Standard directement dans des représentations de E₈ ont rencontré l'obstruction de Distler-Garibaldi. K₇ utilise E₈×E₈ comme architecture, pas comme espace de représentation direct pour les fermions.

**Diffère du paysage des cordes** : les 10⁵⁰⁰ vides de la théorie des cordes créent un problème de sélection. K₇ propose que des contraintes topologiques déterminent uniquement les observables, éliminant la dégénérescence des vides (bien que cela requière la construction K₇ spécifique).

**Connexion avec la M-théorie** : le bulk à 11 dimensions et la compactification G₂ sont standards en M-théorie. K₇ peut être vu comme l'extraction de prédictions spécifiques d'un coin particulier du paysage de la M-théorie.

**Interprétation théorico-informationnelle** : le facteur de dualité binaire p₂ = 2, l'apparition de ln(2) dans les formules cosmologiques, et la compression dimensionnelle 496 → 99 → 4 suggèrent des connexions avec la théorie de l'information et la correction d'erreurs quantique.

## 6. Synthèse

Le cadre K₇ explore si les paramètres du Modèle Standard pourraient être des invariants topologiques plutôt que des constantes libres. La proposition spécifique implique une structure de jauge E₈×E₈, une holonomie G₂ sur une 7-variété K₇ avec nombres de Betti b₂ = 21 et b₃ = 77, et une torsion contrôlée fournissant la dynamique.

Les 33 relations Type I sans dimension qui en résultent correspondent à l'expérience avec une précision moyenne de 0,99 % (NuFIT 6.1 / PDG 2024 / Planck 2018 / CODATA 2022). Que cela reflète une physique fondamentale ou une coïncidence élaborée sera déterminé par les expériences, en particulier la mesure par DUNE de la phase de violation CP δ_CP dans les années à venir.

La valeur du cadre, indépendamment de sa correction physique, réside dans la démonstration que des principes géométriques peuvent contraindre substantiellement les paramètres de la physique des particules. Il fournit un exemple concret de la manière dont la topologie pourrait remplacer l'ajustement.

## Références

- Article principal : [Article principal](Paper-Main-Framework.html)
- Fondations mathématiques : [Article S1 fondations](Paper-S1-Foundations.html)
- Dérivations complètes : [Article S2 dérivations](Paper-S2-Derivations.html)
- Géométrie spectrale : [Article géométrie spectrale](Paper-Spectral-Geometry.html)
- Vérification formelle : [Arithmon/K7-Lean](https://github.com/Arithmon/K7-Lean)

---

*Cadre K₇ v3.4.27*
