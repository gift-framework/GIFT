---
title: "Pour les géomètres"
layout: default
---

# gift_core : outils computationnels pour la géométrie G₂

## Résumé

`gift_core` fournit un pipeline computationnel validé pour construire et analyser des métriques d'holonomie G₂ sur des variétés twisted connected sum (TCS). Le paquet implémente des méthodes numériques pour approximer les structures G₂, calculer les invariants topologiques, et certifier l'existence via le théorème de perturbation de Joyce. Initialement développé pour une application en physique (le cadre GIFT), les outils géométriques sont généraux et pourraient présenter un intérêt indépendant pour les chercheurs étudiant l'holonomie exceptionnelle.

## 1. Le défi computationnel

Les métriques d'holonomie G₂ sont notoirement difficiles à calculer explicitement. Les travaux fondateurs de Joyce ont établi des théorèmes d'existence pour les variétés G₂ compactes via la résolution d'orbifolds et les méthodes de perturbation, mais ces preuves sont non constructives. La construction twisted connected sum (TCS) de Kovalev et Corti-Haskins-Nordström-Pacini fournit un chemin plus explicite : recoller deux variétés de Calabi-Yau de dimension 3 asymptotiquement cylindriques le long d'une frontière commune S¹ × K3. Pourtant, même les méthodes TCS donnent l'existence plutôt que des coefficients métriques explicites.

Le défi est triple. Premièrement, les équations de structure G₂ forment un système couplé d'EDP non linéaires. Deuxièmement, la vérification de l'absence de torsion (dφ = 0, d*φ = 0) requiert le calcul des dérivées extérieures d'une 3-forme définie sur un espace à 7 dimensions. Troisièmement, l'extraction de quantités physiques telles que les nombres de Betti, les formes harmoniques et les tenseurs de courbure exige des méthodes numériques robustes avec des bornes d'erreur quantifiées.

`gift_core` répond à ces défis par une combinaison de réseaux de neurones informés par la physique (PINNs), de méthodes spectrales pour l'extraction des formes harmoniques, et de ponts de vérification formelle vers des assistants de preuve.

## 2. Le pipeline

### 2.1 Construction TCS

Le paquet implémente le cadre twisted connected sum pour les variétés G₂. À partir de deux variétés de Calabi-Yau de dimension 3 asymptotiquement cylindriques (ACyl) Y₁ et Y₂, chacune avec une extrémité cylindrique difféomorphe à (0, ∞) × S¹ × K3, la construction procède ainsi :

1. Tronquer chaque variété ACyl à une longueur de col T, obtenant M₁ᵀ et M₂ᵀ
2. Identifier les frontières S¹ × K3 via une rotation hyper-Kähler
3. Lisser la région de recollement pour obtenir une 7-variété compacte K₇ = M₁ᵀ ∪_φ M₂ᵀ

Pour la construction spécifique dans GIFT, les blocs de construction donnent les nombres de Betti calculés via Mayer-Vietoris :

| Bloc | Origine | b₂ | b₃ |
|------|---------|----|----|
| M₁ | Quintique dans P⁴ | 11 | 40 |
| M₂ | CI(2,2,2) dans P⁶ | 10 | 37 |
| K₇ | Recollement TCS | 21 | 77 |

### 2.2 Approximation de la métrique par PINN

Les métriques G₂ explicites sont approximées à l'aide de réseaux de neurones informés par la physique. Le réseau paramétrise une 3-forme G₂ φ sur un patch de coordonnées :

```
Entrée : x ∈ R⁷
    ↓
Caractéristiques de Fourier : 64 fréquences → 128 dimensions
    ↓
Couches cachées : 4 × 256 neurones (activation SiLU)
    ↓
Sortie : 35 composantes indépendantes de φ ∈ Λ³(R⁷)
```

L'entraînement minimise une perte composite :

L = w_T ||dφ||² + w_T ||d*φ||² + w_det |det(g) - cible|² + w_pos ReLU(-λ_min(g))

Les termes de torsion poussent vers la condition sans torsion. La contrainte de déterminant fixe la forme de volume à une cible spécifiée (65/32 dans l'application GIFT). Le terme de positivité garantit que la métrique induite g(φ) reste définie positive.

L'entraînement s'exécute en 5 à 10 minutes sur du matériel grand public et atteint :
- det(g) = 2,0312490 ± 0,0001 (cible : 65/32 = 2,03125, écart : 0,00005 %)
- ||T|| = 0,00286 (bien en dessous du seuil de Joyce)
- λ_min(g) = 1,078 (définie positive)

### 2.3 Extraction topologique

À partir de la métrique entraînée, le pipeline extrait les invariants topologiques :

**Nombres de Betti par analyse spectrale** : le laplacien de Hodge Δ_k = dd* + d*d est discrétisé sur un maillage. Le clustering des valeurs propres identifie les formes harmoniques comme des modes nuls (à la tolérance numérique près). Pour k=2, l'analyse spectrale retrouve b₂ = 21 exactement. Pour k=3, l'écart spectral apparaît à la position 76-77, indiquant b₃ = 77 (avec un mode à la frontière numérique).

**Base de formes harmoniques** : les vecteurs propres correspondant aux valeurs propres proches de zéro fournissent une base numérique pour H^k(K₇). Ces formes sont utilisées en aval pour calculer les intégrales, les couplages de Yukawa et d'autres quantités géométriques.

**Tenseurs de courbure** : les symboles de Christoffel, la courbure de Riemann, le tenseur de Ricci et la courbure scalaire sont calculés via la différentiation automatique du réseau de neurones.

## 3. Pont de vérification formelle

Une caractéristique distinctive de `gift_core` est sa connexion avec les
assistants de preuve formelle. Les anciens certificats numériques/TCS alimentent
des contrôles Lean 4, mais ils ne doivent pas être lus comme le théorème Level E
compact courant sur `K_7`.

**Portée courante** : la branche rank-one `D0` suit l'existence compacte sans
torsion dans `docs/analytic_status.md`. Le théorème anisotrope de Joyce `(J)`
reste une entrée analytique ouverte.

Les anciens contrôles numériques peuvent certifier des diagnostics de petite
torsion ou des modèles locaux/TCS-neck, mais ils ne prouvent pas à eux seuls une
métrique compacte sans torsion sur le datum `K_7` courant.

**Ce qui reste numérique ou conditionnel** : les coefficients métriques explicites
de l'ancien pipeline PINN sont numériques, pas en forme close. Le théorème
compact courant requiert aussi le théorème de perturbation anisotrope et les
wrappers compacts datum/topologie.

## 4. Utilisation

### Installation

```bash
pip install gift-core
```

Prérequis : Python 3.10+, PyTorch 2.0+, NumPy, SciPy.

### Modules clés

```python
import gift_core as gc

# Exécuter le pipeline complet
config = gc.PipelineConfig(neck_length=15.0, use_pinn=True)
result = gc.run_pipeline(config)

# Accéder aux résultats
print(f"det(g) = {result.det_g}")           # 2.03125
print(f"Torsion = {result.torsion_norm}")   # 0.00286
print(f"b₂ = {result.b2}, b₃ = {result.b3}") # 21, 77

# Exporter le certificat Lean
lean_proof = result.certificate.to_lean()
```

### Structure des modules

| Module | Contenu |
|--------|---------|
| `gift_core.geometry` | K3, ACyl CY3, construction TCS |
| `gift_core.g2` | 3-forme G₂, holonomie, calcul de torsion |
| `gift_core.harmonic` | Laplacien de Hodge, analyse spectrale |
| `gift_core.nn` | Architecture et entraînement du PINN |
| `gift_core.verification` | Génération du certificat Lean 4 |

## 5. Limitations et problèmes ouverts

**Spécificité** : l'implémentation actuelle est ajustée pour la construction K₇ avec b₂ = 21, b₃ = 77. Généraliser à d'autres blocs de construction TCS (différentes 3-variétés de Fano, différents difféomorphismes de recollement) requiert d'adapter les contraintes topologiques.

**Bornes TCS standards** : à noter que les constructions TCS typiques donnent b₂ ≤ 9. Le K₇ de GIFT avec b₂ = 21 emploie soit des blocs de construction non standards, soit doit être compris via la caractérisation variationnelle plutôt que via le recollement TCS explicite.

**Métrique explicite** : le PINN fournit une approximation numérique, pas une métrique en forme close. Pour les applications nécessitant des expressions analytiques, des travaux supplémentaires sont nécessaires.

**Espace des modules** : l'unicité de la métrique G₂ au sein de sa classe de modules n'est pas abordée. Plusieurs métriques avec les mêmes invariants topologiques peuvent exister.

**Invitation ouverte** : étendre `gift_core` à d'autres variétés G₂ (résolutions d'orbifolds de Joyce, autres exemples TCS, ou les constructions plus récentes de Foscolo-Haskins-Nordström) serait une contribution précieuse au domaine.

## Références

- Joyce, D.D. (2000). *Compact Manifolds with Special Holonomy*. Oxford University Press.
- Kovalev, A. (2003). « Twisted connected sums and special Riemannian holonomy. » *J. Reine Angew. Math.* 565, 125-160.
- Corti, A., Haskins, M., Nordström, J., Pacini, T. (2015). « G₂-manifolds and associative submanifolds via semi-Fano 3-folds. » *Duke Math. J.* 164(10), 1971-2092.
- Raissi, M., Perdikaris, P., Karniadakis, G.E. (2019). « Physics-informed neural networks. » *J. Comp. Phys.* 378, 686-707.

**Dépôt de code** : [github.com/gift-framework/core](https://github.com/gift-framework/core)

**Documentation associée** : [S1 : Fondations](Paper-S1-Foundations.html)

---

*`gift_core` fait partie du cadre GIFT v3.4. Pour l'application physique, voir l'[article principal](Paper-Main-Framework.html).*
