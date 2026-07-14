---
title: Glossaire
---

# Glossaire des termes techniques

Définitions complètes des termes techniques, de la notation mathématique et des acronymes utilisés dans le cadre théorique GIFT.

## Termes spécifiques au cadre

### GIFT
**Geometric Information Field Theory**. Le cadre théorique qui propose que les paramètres fondamentaux de la physique émergent comme des invariants topologiques d'une structure E₈×E₈ compactifiée sur des variétés à holonomie G₂.

### K₇
Une variété riemannienne compacte de dimension 7 à holonomie G₂. L'indice 7 indique la dimension. Propriétés topologiques spécifiques : b₂(K₇) = 21, b₃(K₇) = 77.

### Classifications de statut
Le cadre utilise une classification hiérarchique des résultats :
- **VERIFIED (Lean 4)** : formellement vérifié par le noyau Lean 4 avec Mathlib (preuves vérifiées par machine, zéro axiome de domaine, zéro sorry)
- **VERIFIED** : identité topologique exacte avec preuve mathématique rigoureuse
- **TOPOLOGICAL** : conséquence directe de la structure topologique
- **DERIVED** : calculé à partir de relations prouvées
- **THEORETICAL** : justification théorique, preuve incomplète
- **PHENOMENOLOGICAL** : empiriquement précis, dérivation en cours
- **EXPLORATORY** : enquête préliminaire, mécanisme incertain

## Algèbres et groupes de Lie

### E₈
La plus grande algèbre de Lie simple exceptionnelle. Propriétés :
- Dimension : 248
- Rang : 8
- Système de racines : 240 racines, toutes de longueur √2
- Simplement lacée (toutes les racines de même longueur)
- Déterminant de la matrice de Cartan : 1

### E₈×E₈
Produit de deux copies indépendantes de E₈. Dimension totale : 496 = 2×248.

### G₂
Le groupe d'automorphismes des octonions. Un groupe de Lie exceptionnel de dimension 14. Le plus petit groupe exceptionnel, important pour la géométrie en 7 dimensions.

### SU(N)
**Groupe spécial unitaire**. Groupe des matrices unitaires N×N de déterminant 1.
- SU(3) : force forte (chromodynamique quantique)
- SU(2) : composante de la force faible
- U(1) : composante de la force électromagnétique

### Algèbres de Lie exceptionnelles
Cinq algèbres de Lie qui ne rentrent pas dans les familles infinies standards :
- G₂ (dimension 14, rang 2)
- F₄ (dimension 52, rang 4)
- E₆ (dimension 78, rang 6)
- E₇ (dimension 133, rang 7)
- E₈ (dimension 248, rang 8)

### Groupe de Weyl
Groupe de symétrie d'un système de racines. Pour E₈, le groupe de Weyl a un ordre de 696 729 600.

### Système de racines
Ensemble de vecteurs dans un espace euclidien satisfaisant à une symétrie de réflexion. Pour E₈ : 240 racines disposées dans une configuration hautement symétrique.

### Rang
Dimension du tore maximal (sous-groupe abélien maximal). Pour E₈ : rang 8.

## Géométrie et topologie

### Holonomie
Propriété géométrique décrivant comment les vecteurs changent lorsqu'on les transporte parallèlement le long de boucles fermées. L'holonomie G₂ implique une structure géométrique particulière.

### Cohomologie
Outil mathématique mesurant les caractéristiques topologiques. Pour K₇ :
- H²(K₇) = ℝ²¹ : lié aux bosons de jauge
- H³(K₇) = ℝ⁷⁷ : lié aux fermions chiraux

### Nombres de Betti
Invariants topologiques comptant les classes d'homologie indépendantes.
- b₂ : deuxième nombre de Betti (trous 2D)
- b₃ : troisième nombre de Betti (trous 3D)
Pour K₇ : b₂ = 21, b₃ = 77

### AdS₄
**Espace anti-de Sitter en 4 dimensions**. Espace-temps maximalement symétrique avec constante cosmologique négative. Utilisé dans les modèles holographiques et la compactification GIFT.

### Variété compacte
Espace topologique fermé (contient tous ses points limites) et borné. K₇ est compact, ce qui permet une réduction dimensionnelle cohérente.

### Ricci-plat
Variété dont le tenseur de courbure de Ricci est nul. Les variétés à holonomie G₂ sont automatiquement Ricci-plates, adaptées à la compactification.

### Formes harmoniques
Formes différentielles satisfaisant l'équation de Laplace. Modes zéro dans la réduction dimensionnelle correspondant aux champs 4D.

### Réduction de Kaluza-Klein
Processus de compactification des dimensions supplémentaires pour dériver une théorie effective de dimension inférieure. Les champs se décomposent en tour de modes.

## Physique des particules

### Modèle Standard
Théorie actuelle de la physique des particules décrivant les forces électromagnétique, faible et forte. Contient 19 paramètres libres dans sa formulation conventionnelle.

### Génération
Famille de fermions. Le Modèle Standard a trois générations :
- Première : (u, d, e, νₑ)
- Deuxième : (c, s, μ, νμ)
- Troisième : (t, b, τ, ντ)

### Couplage de jauge
Intensité d'une interaction de force. Trois dans le Modèle Standard :
- α : électromagnétique (constante de structure fine)
- g₂ : force faible
- g₃ (ou α_s) : force forte

### Constante de structure fine (α)
Couplage sans dimension de l'interaction électromagnétique. α ≈ 1/137,036.

### Angle de mélange faible (θ_W)
Paramètre reliant les forces électromagnétique et faible. sin²θ_W ≈ 0,231.

### Couplage fort (α_s)
Constante de couplage de la chromodynamique quantique. α_s(M_Z) ≈ 0,118.

### Matrice CKM
**Matrice de Cabibbo-Kobayashi-Maskawa**. Matrice unitaire 3×3 décrivant le mélange des quarks entre générations. Contient 4 paramètres indépendants (3 angles, 1 phase).

### Matrice PMNS
**Matrice de Pontecorvo-Maki-Nakagawa-Sakata**. Analogue de la CKM pour le secteur neutrino. Contient 3 angles de mélange (θ₁₂, θ₁₃, θ₂₃) et une phase de violation CP (δ_CP).

### Violation CP
Brisure de la symétrie combinée de conjugaison de charge (C) et de parité (P). Observée dans les secteurs des quarks et des neutrinos.

### Couplage de Yukawa
Intensité d'interaction entre les fermions et le champ de Higgs, déterminant les masses des fermions.

### Mécanisme de Higgs
Processus par lequel les bosons de jauge acquièrent une masse via la brisure spontanée de symétrie.

### VEV
**Valeur moyenne dans le vide** (*Vacuum Expectation Value*). Valeur non nulle du champ de Higgs dans le vide, v ≈ 246 GeV.

## Observables spécifiques

### N_gen
Nombre de générations de fermions. Expérimentalement : 3. GIFT prédit : rang(E₈) − rang(Weyl) = 3.

### δ_CP
Phase violant CP dans le mélange des neutrinos. GIFT prédit : 197° à partir de la formule dim(K₇)×dim(G₂) + H* = 7×14 + 99 = 197°.

### θ₁₂, θ₁₃, θ₂₃
Trois angles de mélange des neutrinos dans la matrice PMNS.
- θ₁₂ ≈ 33,44° (mélange solaire)
- θ₁₃ ≈ 8,61° (mélange réacteur)
- θ₂₃ ≈ 49,2° (mélange atmosphérique)

### Q_Koide
Paramètre de la formule de Koide reliant les masses des leptons chargés :
```
Q = (mₑ + mμ + mτ)² / (mₑ² + mμ² + mτ²)
```
Expérimental : Q ≈ 2/3. GIFT : Q = 2/3 exactement.

### Ω_DE
Densité d'énergie noire en fraction de la densité critique. Expérimental : Ω_DE ≈ 0,689. GIFT : Ω_DE = ln(2) ≈ 0,693.

### H₀
**Constante de Hubble**. Taux d'expansion actuel de l'univers. Mesures locales : ~73 km/s/Mpc. Mesures CMB : ~67 km/s/Mpc. La « tension de Hubble » désigne cet écart.

## Constantes mathématiques

### π
Pi, rapport entre la circonférence et le diamètre d'un cercle. π ≈ 3,14159...

### e
Nombre d'Euler, base du logarithme naturel. e ≈ 2,71828...

### φ
**Nombre d'or**. φ = (1 + √5)/2 ≈ 1,61803...
Apparaît dans E₈ via la correspondance de McKay.

### ζ(3)
**Fonction zêta de Riemann en 3**. ζ(3) = 1 + 1/8 + 1/27 + ... ≈ 1,202.
Apparaît dans la formule de δ_CP.

### ln(2)
Logarithme naturel de 2 ≈ 0,693. Lié à la densité d'énergie noire et à la théorie de l'information (bits vs nats).

## Théorie de l'information

### Architecture binaire
Structure basée sur les puissances de 2. La dimension 496 de E₈×E₈ ≈ 2⁴⁸ + 48, suggérant un encodage binaire.

### Correction d'erreur quantique
Méthode de protection de l'information quantique contre les erreurs. Le réseau E₈ forme le code [248, 12, 56](248,-12,-56.html).

### Code [n, k, d](n,-k,-d.html)
Notation pour un code de correction d'erreur quantique :
- n : nombre de qubits physiques
- k : nombre de qubits logiques encodés
- d : distance (nombre d'erreurs corrigibles)

### Rapport de compression
Rapport entre les dimensions d'entrée et de sortie. E₈×E₈ : 496 → 99 donne un rapport ≈ 5:1.

## Expériences de physique

### PDG
**Particle Data Group**. Collaboration internationale qui compile les données expérimentales de la physique des particules. Référence standard pour les paramètres mesurés.

### LHC
**Large Hadron Collider**. Accélérateur de particules au CERN. A découvert le boson de Higgs en 2012.

### Belle II
Expérience de physique des particules au Japon, qui étudie les mésons B et fait des mesures de précision.

### DUNE
**Deep Underground Neutrino Experiment**. Future expérience neutrino aux États-Unis pour des mesures de précision.

### T2K
**Tokai to Kamioka**. Expérience neutrino à long parcours au Japon.

### NOvA
**NuMI Off-Axis νₑ Appearance**. Expérience neutrino aux États-Unis.

## Notation mathématique

### dim(G)
Dimension d'un groupe ou d'une algèbre de Lie G. Exemple : dim(E₈) = 248.

### rang(G)
Rang d'une algèbre de Lie G. Exemple : rang(E₈) = 8.

### Hⁿ(M)
n-ième groupe de cohomologie d'une variété M. Exemple : H²(K₇) = ℝ²¹.

### bₙ
n-ième nombre de Betti, dimension de Hⁿ. Exemple : b₃(K₇) = 77.

### |·|
Valeur absolue ou cardinalité (taille d'un ensemble).

### ⊕
Somme directe d'espaces vectoriels ou d'algèbres.

### ⊗
Produit tensoriel.

### ∈
« Élément de » (appartenance à un ensemble).

### ∀
« Pour tout » (quantificateur universel).

### ∃
« Il existe » (quantificateur existentiel).

### ≈
Approximativement égal.

### ≡
Identiquement égal ou défini comme.

### ∼
Du même ordre de grandeur, ou équivalent à.

## Lettres grecques dans GIFT

### α (alpha)
Constante de structure fine, α ≈ 1/137.

### β₀ (bêta)
Paramètre de couplage de base, β₀ = 1/(4π²).

### γ (gamma)
Coefficient du noyau de chaleur ou constante d'Euler-Mascheroni.

### δ (delta)
Phase de violation CP (δ_CP) ou petit écart.

### ε₀ (epsilon)
Paramètre de brisure de symétrie, ε₀ = 1/8.

### ζ (zêta)
Fonction zêta de Riemann, ζ(3) apparaît dans δ_CP.

### θ (thêta)
Angles de mélange (θ₁₂, θ₁₃, θ₂₃, θ_W).

### ξ (xi)
Paramètre de corrélation, ξ = 5β₀/2.

### Ω (Oméga)
Paramètres de densité (Ω_DE pour l'énergie noire).

### φ (phi)
Nombre d'or ou angle.

## Acronymes

### GIFT
Geometric Information Field Theory

### SM
Standard Model (Modèle Standard de la physique des particules)

### GUT
Grand Unified Theory (théorie de grande unification)

### QCD
Quantum Chromodynamics (chromodynamique quantique, force forte)

### QED
Quantum Electrodynamics (électrodynamique quantique, force électromagnétique)

### EW
Electroweak (théorie unifiée électromagnétique et faible)

### SUSY
Supersymétrie

### CMB
Cosmic Microwave Background (fond diffus cosmologique)

### VEV
Vacuum Expectation Value (valeur moyenne dans le vide)

### RG
Renormalization Group (groupe de renormalisation)

### UV
Ultraviolet (haute énergie)

### IR
Infrared (basse énergie)

### LO/NLO/NNLO
Leading Order / Next-to-Leading Order / Next-to-Next-to-Leading Order (ordres de l'expansion perturbative)

## Conventions d'unités

### Unités naturelles
Système où ℏ = c = 1. Énergies, masses et impulsions ont les mêmes dimensions.

### GeV
Giga-électron-volt, 10⁹ eV. Unité d'énergie/masse usuelle en physique des particules.
- Masse du proton : ~1 GeV
- Masse du boson de Higgs : ~125 GeV
- Masse du quark top : ~173 GeV

### MeV
Méga-électron-volt, 10⁶ eV.
- Masse de l'électron : ~0,511 MeV
- Masses des quarks : de quelques MeV à quelques GeV

## Indicateurs de statut dans la documentation

### [VERIFIED]
Affirmation mathématique avec preuve rigoureuse fournie.

### [TOPOLOGICAL]
Conséquence directe de la topologie de la variété.

### [EXACT]
Écart nul par construction mathématique.

### [HIGH-PRECISION]
Accord expérimental à <1 % d'écart.

### [PRELIMINARY]
Calcul ou résultat en cours de raffinement.

## Renvois

Pour des explications plus détaillées :
- **Fondations mathématiques** : voir [S1 : Foundations](Paper-S1-Foundations.html)
- **Contexte physique des particules** : voir [Article principal](Paper-Main-Framework.html) Section 1
- **Valeurs expérimentales** : voir [S2 : Derivations](Paper-S2-Derivations.html)
- **Questions courantes** : voir [FAQ](FAQ.fr.html)

## Contribuer

Ce glossaire est continuellement mis à jour. Pour suggérer des ajouts ou des corrections :
1. Ouvrez un ticket sur https://github.com/Arithmon/K7/issues
2. Marquez-le « documentation »
3. Précisez le terme et la définition proposée

### Dynamique torsionnelle (v2.1)
Cadre introduit en v2.1 reliant la torsion non nulle sur K₇ au flot du groupe de renormalisation. Paramètres clés : |T_norm| = 0,0164, |T_costar| = 0,0141.

### Pont d'échelle (v2.1)
Infrastructure mathématique reliant les observables sans dimension aux observables dimensionnels : Λ_GIFT = 21×e⁸×248/(7×π⁴) ≈ 1,63×10⁶.

### Lean 4 (v3.3)
Démonstrateur de théorèmes utilisé pour la vérification formelle des relations exactes de GIFT. Le dépôt [gift-framework/core](https://github.com/gift-framework/core) contient 455+ relations certifiées, incluant le système de racines de E₈, les propriétés du produit vectoriel G₂, la théorie spectrale, les bornes TCS, et les rapports de masses Yukawa. Théorème clé : `GIFT_framework_certified`.

---

Dernière mise à jour : v3.3.31 (2026-03-09)
