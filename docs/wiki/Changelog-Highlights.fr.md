---
title: "Points clés du changelog"
layout: default
---

# Points clés du changelog

Historique des versions abrégé. Pour le changelog complet, voir [`CHANGELOG.md`](https://github.com/gift-framework/GIFT/blob/main/CHANGELOG.md).

---

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
