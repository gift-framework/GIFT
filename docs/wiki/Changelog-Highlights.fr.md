---
title: "Points clés du changelog"
layout: default
---

# Points clés du changelog

Historique des versions abrégé. Pour le changelog complet, voir [`CHANGELOG.md`](https://github.com/gift-framework/GIFT/blob/main/CHANGELOG.md).

---

## v3.4.13, 2026-04-29

**Publication du triptyque & jalon de réduction d'axiomes**

- **Articles A, B, C publiés sur Zenodo** (DOIs 19892350, 19893371, 19708916), le triptyque relecture par les pairs (structure G₂ certifiée, géométrie spectrale, diagnostics NK sur K3)
- **Réduction Lean** : 38 → 4 axiomes principaux (15 au total avec les certificats d'arithmétique d'intervalles), 0 sorry, 213 conjonctions certifiées
- **K3NewtonKantorovich v3.0 hardcore** : η ×2,4 plus serré, marge Joyce ×17 sous ε₀
- **γ² = 24π²/7 dérivé** (laplacien de Hodge sur T² + H¹(K3)=0 ; 135/4 était un artefact PSLQ)
- **95 observables** : 35 Type I + 19 Type II + 21 Type III + 22 Type IV ; écart moyen 0,39 % sur Type I (PDG 2024 / NuFIT 6.0)
- **3 primitives entières** : N=3, r₈=8, r₂=2 (zéro paramètre continu ajustable)

## v3.4.3, 2026-04

**Étapes 1-5 G₂ Mathlib promues en théorèmes**

- φ₀ 3-forme ordonnée sur ℝ⁷, identité de Bryant ∑φ₀² = 6δ, rang=35 → dim(g₂)=14, B = 144δ, théorème det·gram (Aristotle)
- 8 → 4 axiomes par promotion des identités algébriques en preuves native_decide
- MollifiedSum archivé ; G₂ThreeForm axiomatisé proprement

## v3.4.0, 2026-04

**Programme métrique-first complet · K3 CAP**

- Preuve assistée par ordinateur d'existence d'une métrique G₂ sans torsion sur modèle de cou TCS : h ≤ 8,95×10⁻⁹, marge ×56 millions sous ε₀ de Joyce
- Certificats NK pour K3 : quartique de Fermat ×990, CI(2,2,2) ×6,4 (formalisés en Lean)
- Verdict PSLQ hors-diagonale : les formules L[4,2], L[5,3] étaient des artefacts PSLQ (abandonnées)

## v3.3.24, 2026-03-02

**Mise à jour NuFIT 6.0 et nettoyage des publications**

- Mise à jour aux valeurs expérimentales NuFIT 6.0 (δ_CP : 177°±20°)
- Nouvelles formules neutrino : θ₁₂ = arctan(2/3), θ₂₃ = arctan(√(14/11))
- Insight clé : tan(θ₁₂) = Q_Koide = 2/3
- Écart moyen : **0,24 %** (32 bien mesurés) / 0,57 % (les 33 incl. δ_CP)
- Affirmations de la construction S1 assouplies (conditionnelles au théorème d'existence de Joyce)
- Renommage Weyl → w pour éviter la collision avec la courbure de Weyl

## v3.3.17, 2026-02-04

**Correction de la formule θ₂₃**

- Correction de l'angle de mélange atmosphérique : arcsin(25/33) = 49,25° (était 59,16°)
- Écart sur θ₂₃ : 20 % → **0,10 %**
- Écart moyen : 0,84 % → **0,21 %**

## v3.3.14, 2026-01-28

**Principe de sélection et 290+ relations**

- Principe de sélection formalisé en Lean 4
- Bornes spectrales TCS ajoutées
- 290+ relations certifiées au total
- Lean 4 comme unique système de vérification (Coq archivé)

## v3.3.0, 2026-01-12

**33 observables et PDG 2024**

- Étendu à 33 prédictions sans dimension
- Mise à jour aux valeurs expérimentales PDG 2024
- Validation Monte Carlo sur 192 349 configurations

## v3.1.0, 2025-12-17

**Métrique G₂ analytique**

- Métrique Chebyshev-Cholesky explicite (169 paramètres)
- Certification Newton-Kantorovich (h = 6,65×10⁻⁸)
- 185 relations Lean certifiées

## v3.0.0, 2025-12-09

**Release majeure**

- 165+ relations certifiées en Lean 4
- Explorations de structure de groupes exceptionnels

## v2.3.x, 2025-12

- Vérification Lean 4 introduite

## v2.2.0, 2025-11-27

- Paradigme zéro-paramètre établi

## v2.1.0, 2025-11-22

- Cadre dynamique torsionnel
- Pont d'échelle (Λ_GIFT = 21×e⁸×248/(7×π⁴))

## v2.0.0, 2025-10-24

- Réorganisation du cadre

---

*Pour les détails complets, voir le [changelog complet](https://github.com/gift-framework/GIFT/blob/main/CHANGELOG.md).*
